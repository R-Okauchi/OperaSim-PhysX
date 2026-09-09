using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Unity.Collections;
using UnityEditor;
using UnityEngine;

namespace CapKit.Editor
{
    /// <summary>Read-only prefab survey. All geometry is in metres, ROS FLU, relative to kit_hub.
    /// Does not use the live scene, physics colliders, world Renderer.bounds, or save prefabs.</summary>
    public static class CapKitGeometryDump
    {
        private static readonly string[] Machines = { "zx120", "zx200", "mst110cr", "ic120", "c30r" };
        private static readonly string[] Pods = { "fl", "fr", "rl", "rr" };
        private const string Output = "Assets/CAP/machine_geometry.json";

        [Serializable] public sealed class Candidate
        {
            public string sensor_id;
            public float[] xyz_m;
        }
        [Serializable] public sealed class CandidateMachine
        {
            public string machine_id;
            public Candidate[] candidates;
        }
        [Serializable] public sealed class CandidateFile { public CandidateMachine[] machines; }
        [Serializable] private sealed class Box
        {
            public bool valid;
            public float[] min_m, max_m;
            public void Add(Vector3 p)
            {
                if (!valid) { min_m = Array(p); max_m = Array(p); valid = true; return; }
                for (int i = 0; i < 3; i++) { min_m[i] = Mathf.Min(min_m[i], p[i]); max_m[i] = Mathf.Max(max_m[i], p[i]); }
            }
            public void Add(Box b) { if (b.valid) { Add(Vec(b.min_m)); Add(Vec(b.max_m)); } }
            public bool Contains(Vector3 p) => valid && Enumerable.Range(0, 3).All(i => p[i] >= min_m[i] && p[i] <= max_m[i]);
        }
        [Serializable] private sealed class Surface
        {
            public string path, mesh_asset, group, subtree;
            public bool included_in_static_body, renderer_enabled, active_self, active_in_hierarchy, geometry_valid;
            public Box bounds = new Box();
            // Packed xyz and triangle indices; only static meshes retained to keep the output manageable.
            public float[] vertices_xyz_m;
            public int[] triangle_indices;
        }
        [Serializable] private sealed class Group
        {
            public string group, subtree;
            public Box bounds = new Box();
            public List<string> renderer_paths = new List<string>();
        }
        [Serializable] private sealed class Top
        {
            public bool valid;
            public float z_hub_m;
        }
        [Serializable] private sealed class Polar
        {
            public int azimuth_bins = 72;
            public float start_deg = -180f, bin_width_deg = 5f;
            public string semantics = "Vertex maxima for z < threshold; -1 means no vertices, NOT empty space. Not a solid cross-section. Use triangles for height/occlusion queries.";
            public float[] max_radius_below_1_m = Enumerable.Repeat(-1f, 72).ToArray();
            public float[] max_radius_below_2_5_m = Enumerable.Repeat(-1f, 72).ToArray();
            public int[] vertex_count_below_1_m = new int[72], vertex_count_below_2_5_m = new int[72];
        }
        [Serializable] private sealed class Probe
        {
            public string sensor_id, source, nearest_mesh;
            public float[] xyz_m;
            public bool inside_static_body_bounds, clearance_valid;
            public float clearance_m;
        }
        [Serializable] private sealed class Machine
        {
            public string machine_id, prefab, prefab_dependency_hash, hub_path;
            public bool complete = true;
            public float[] hub_local_to_root_unity_row_major;
            public Box static_body_bounds = new Box();
            public Top upper_structure_top = new Top(), tracks_top = new Top();
            public Polar silhouette = new Polar();
            public List<Surface> renderers = new List<Surface>();
            public List<Group> groups = new List<Group>();
            public List<Probe> candidates = new List<Probe>();
            public List<string> warnings = new List<string>();
        }
        [Serializable] private sealed class Document
        {
            public string schema_version = "1.0", generated_at_utc;
            public string frame = "kit_hub_ros_flu";
            public string coordinates = "x_ros=z_unity, y_ros=-x_unity, z_ros=y_unity, AFTER mesh-local -> root -> hub; metres";
            public string static_body_policy = "All non-kit MeshRenderers except boom/arm/bucket descendants. Includes inactive/disabled meshes conservatively, and vessel/tailgate in saved prefab pose. Unknown subtrees included and warned. Visual mesh, not collider geometry.";
            public string clearance_semantics = "Unsigned point-to-triangle surface distance, not signed penetration or AABB distance; inside_static_body_bounds is only a union-AABB test.";
            public List<Machine> machines = new List<Machine>();
            public List<string> warnings = new List<string>(), errors = new List<string>();
        }

        [MenuItem("CAP/Perception Kit/Dump Machine Geometry")]
        public static void Dump()
        {
            var report = new CapKitRig.Report();
            var doc = new Document { generated_at_utc = DateTime.UtcNow.ToString("o") };
            // CLI overrides env; JSON schema is CandidateFile above. Defaults come from sensor_rig.json.
            string candidatesPath = Environment.GetEnvironmentVariable("CAP_KIT_GEOMETRY_CANDIDATES");
            var args = Environment.GetCommandLineArgs();
            int arg = System.Array.IndexOf(args, "-capKitGeometryCandidates");
            if (arg >= 0)
            {
                if (arg + 1 >= args.Length) throw new ArgumentException("-capKitGeometryCandidates needs a JSON path");
                candidatesPath = args[arg + 1];
            }
            CandidateFile overrides = string.IsNullOrEmpty(candidatesPath) ? null
                : JsonUtility.FromJson<CandidateFile>(File.ReadAllText(candidatesPath));
            CapKitContract contract = null;
            try { contract = CapKitContract.Load(); }
            catch (Exception e) { report.Warn("Default candidate contract unavailable: " + e.Message); }
            foreach (string id in Machines)
            {
                string path = "Assets/Prefab/" + id + ".prefab";
                if (!File.Exists(path)) { report.Warn(id + ": prefab missing, skipped."); continue; }
                GameObject root = null;
                try
                {
                    root = PrefabUtility.LoadPrefabContents(path);
                    var hubs = root.GetComponentsInChildren<CapKitFrameMarker>(true).Where(m => m.name == "kit_hub").ToArray();
                    if (hubs.Length != 1) { report.Warn(id + ": expected one kit_hub CapKitFrameMarker, found " + hubs.Length + "; skipped."); continue; }
                    var m = Survey(id, path, root, hubs[0].transform, report);
                    var custom = overrides?.machines?.SingleOrDefault(c => c.machine_id == id);
                    if (custom != null)
                        foreach (var c in custom.candidates ?? new Candidate[0]) ProbePoint(m, c, "candidate override (hub ROS)");
                    else
                    {
                        var spec = contract?.machines?.SingleOrDefault(c => c.machine_id == id);
                        foreach (string pod in Pods)
                        {
                            var sensor = spec?.sensors?.SingleOrDefault(s => s.sensor_id == pod);
                            if (sensor == null) { Warn(m, report, "Missing exported contract candidate " + pod + "; skipped."); continue; }
                            try
                            {
                                // Contract mounts are parent-link relative, not necessarily hub relative.
                                var parent = CapKitRigObjects.FindLink(root, sensor.parent_link);
                                var point = Ros(hubs[0].transform.worldToLocalMatrix.MultiplyPoint3x4(parent.localToWorldMatrix.MultiplyPoint3x4(sensor.unity.Position)));
                                ProbePoint(m, new Candidate { sensor_id = pod, xyz_m = Array(point) }, "sensor_rig.json " + sensor.parent_link + " converted to hub");
                            }
                            catch (Exception e) { Warn(m, report, pod + ": " + e.Message + "; candidate skipped."); }
                        }
                    }
                    doc.machines.Add(m);
                    report.Info(id + ": " + m.renderers.Count + " non-kit renderers, " + m.groups.Count + " groups, " + m.candidates.Count + " candidates; complete=" + m.complete);
                }
                catch (Exception e) { report.Error(id + ": " + e.GetType().Name + ": " + e.Message); }
                finally { if (root != null) PrefabUtility.UnloadPrefabContents(root); }
            }
            doc.warnings.AddRange(report.Warnings);
            doc.errors.AddRange(report.Errors);
            File.WriteAllText(Output, JsonUtility.ToJson(doc, true) + "\n");
            AssetDatabase.ImportAsset(Output);
            report.Info("Wrote " + Output + " (" + doc.machines.Count + "/5 machines, " + report.Warnings.Count + " warnings, " + report.Errors.Count + " errors).");
            if (Application.isBatchMode && (report.Errors.Count != 0 || doc.machines.Count != Machines.Length))
                throw new InvalidOperationException("Geometry dump incomplete; inspect JSON and [CAP Kit] log.");
        }

        private static Machine Survey(string id, string path, GameObject root, Transform hub, CapKitRig.Report report)
        {
            Matrix4x4 hubToRoot = root.transform.worldToLocalMatrix * hub.localToWorldMatrix;
            var m = new Machine { machine_id = id, prefab = path, hub_path = PathOf(hub, root.transform),
                prefab_dependency_hash = AssetDatabase.GetAssetDependencyHash(path).ToString(),
                hub_local_to_root_unity_row_major = Enumerable.Range(0, 16).Select(i => hubToRoot[i / 4, i % 4]).ToArray() };
            foreach (var renderer in root.GetComponentsInChildren<MeshRenderer>(true))
            {
                var chain = Ancestors(renderer.transform, root.transform).ToArray();
                if (chain.Any(t => t.name.StartsWith("kit_", StringComparison.OrdinalIgnoreCase))) continue;
                string category = Category(chain);
                var link = chain.FirstOrDefault(t => t.name.EndsWith("_link", StringComparison.OrdinalIgnoreCase)) ?? renderer.transform;
                var s = new Surface { path = PathOf(renderer.transform, root.transform), group = category,
                    subtree = PathOf(link, root.transform), included_in_static_body = category != "attachment",
                    renderer_enabled = renderer.enabled, active_self = renderer.gameObject.activeSelf,
                    active_in_hierarchy = renderer.gameObject.activeInHierarchy };
                m.renderers.Add(s);
                try
                {
                    var filter = renderer.GetComponent<MeshFilter>();
                    if (filter == null || filter.sharedMesh == null) throw new InvalidDataException("Missing MeshFilter/sharedMesh");
                    var mesh = filter.sharedMesh;
                    s.mesh_asset = AssetDatabase.GetAssetPath(mesh);
                    Vector3[] vertices;
                    int[] triangles;
                    ReadMesh(mesh, out vertices, out triangles);
                    if (vertices.Length == 0 || triangles.Length == 0) throw new InvalidDataException("Empty triangle mesh");
                    Matrix4x4 meshToHub = hubToRoot.inverse * root.transform.worldToLocalMatrix * renderer.transform.localToWorldMatrix;
                    for (int i = 0; i < vertices.Length; i++) { vertices[i] = Ros(meshToHub.MultiplyPoint3x4(vertices[i])); s.bounds.Add(vertices[i]); }
                    s.geometry_valid = true;
                    if (s.included_in_static_body)
                    {
                        s.vertices_xyz_m = vertices.SelectMany(Array).ToArray();
                        s.triangle_indices = triangles;
                        m.static_body_bounds.Add(s.bounds);
                        foreach (var v in vertices)
                        {
                            int bin = Mathf.Clamp(Mathf.FloorToInt((Mathf.Atan2(v.y, v.x) * Mathf.Rad2Deg + 180f) / 5f), 0, 71);
                            float radius = new Vector2(v.x, v.y).magnitude;
                            if (v.z < 1f) { m.silhouette.max_radius_below_1_m[bin] = Mathf.Max(m.silhouette.max_radius_below_1_m[bin], radius); m.silhouette.vertex_count_below_1_m[bin]++; }
                            if (v.z < 2.5f) { m.silhouette.max_radius_below_2_5_m[bin] = Mathf.Max(m.silhouette.max_radius_below_2_5_m[bin], radius); m.silhouette.vertex_count_below_2_5_m[bin]++; }
                        }
                        if (category == "upper_structure" || category == "cab") AddTop(m.upper_structure_top, s.bounds.max_m[2]);
                        if (category == "undercarriage") AddTop(m.tracks_top, s.bounds.max_m[2]);
                    }
                    var group = m.groups.FirstOrDefault(g => g.group == category && g.subtree == s.subtree);
                    if (group == null) { group = new Group { group = category, subtree = s.subtree }; m.groups.Add(group); }
                    group.bounds.Add(s.bounds); group.renderer_paths.Add(s.path);
                }
                catch (Exception e)
                {
                    if (s.included_in_static_body) m.complete = false;
                    Warn(m, report, s.path + ": " + e.Message + "; mesh skipped.");
                }
            }
            foreach (var g in m.groups.Where(g => g.group == "unknown")) Warn(m, report, "Unclassified subtree included conservatively: " + g.subtree);
            if (!m.static_body_bounds.valid) { m.complete = false; Warn(m, report, "No static body geometry."); }
            if (!m.upper_structure_top.valid) Warn(m, report, "Upper structure/cab not identifiable; top unavailable.");
            if (!m.tracks_top.valid) Warn(m, report, "Tracks/undercarriage not identifiable; top unavailable.");
            if (!m.groups.Any(g => g.group == "cab")) Warn(m, report, "Cab not separately identifiable; inspect body/base mesh grouping before interpreting top heights.");
            if (m.groups.Any(g => g.subtree.ToLowerInvariant().Contains("vessel") || g.subtree.ToLowerInvariant().Contains("tailgate")))
                Warn(m, report, "Vessel/tailgate included in saved pose only; tipping/articulation is not validated.");
            if (root.GetComponentsInChildren<SkinnedMeshRenderer>(true).Any(r => !Ancestors(r.transform, root.transform).Any(t => t.name.StartsWith("kit_"))))
            { m.complete = false; Warn(m, report, "Non-kit SkinnedMeshRenderer omitted; only MeshRenderer is supported."); }
            return m;
        }

        private static void ReadMesh(Mesh mesh, out Vector3[] vertices, out int[] triangles)
        {
            // Editor API permits non-readable imported meshes without changing importer settings.
            using (var data = MeshUtility.AcquireReadOnlyMeshData(mesh))
            {
                var md = data[0];
                using (var buffer = new NativeArray<Vector3>(md.vertexCount, Allocator.Temp))
                { md.GetVertices(buffer); vertices = buffer.ToArray(); }
                var indices = new List<int>();
                for (int sub = 0; sub < md.subMeshCount; sub++)
                {
                    var sm = md.GetSubMesh(sub);
                    if (sm.topology != MeshTopology.Triangles) throw new InvalidDataException("Non-triangle submesh");
                    using (var buffer = new NativeArray<int>(sm.indexCount, Allocator.Temp))
                    { md.GetIndices(buffer, sub, true); indices.AddRange(buffer.ToArray()); }
                }
                triangles = indices.ToArray();
            }
        }

        private static string Category(Transform[] chain)
        {
            var names = chain.Select(t => t.name.ToLowerInvariant()).ToArray();
            if (names.Any(n => n.StartsWith("boom") || n.StartsWith("arm_") || n == "arm" || n.StartsWith("bucket"))) return "attachment";
            // Closest named subtree wins (a track can be nested below body_link).
            foreach (string n in names)
            {
                if (n == "cab" || n.StartsWith("cab_") || n.Contains("cabin")) return "cab";
                if (n.Contains("track") || n.Contains("crawler") || n.Contains("wheel") || n.Contains("undercarriage")) return "undercarriage";
                if (n == "body_link" || n.StartsWith("vessel") || n.StartsWith("tailgate")) return "upper_structure";
                if (n == "base_link") return "undercarriage";
            }
            return "unknown";
        }

        private static void ProbePoint(Machine m, Candidate candidate, string source)
        {
            if (candidate.xyz_m == null || candidate.xyz_m.Length != 3 || candidate.xyz_m.Any(v => float.IsNaN(v) || float.IsInfinity(v)))
                throw new InvalidDataException("Invalid candidate xyz_m: " + candidate.sensor_id);
            Vector3 p = Vec(candidate.xyz_m);
            var probe = new Probe { sensor_id = candidate.sensor_id, xyz_m = candidate.xyz_m, source = source,
                inside_static_body_bounds = m.static_body_bounds.Contains(p) };
            float best = float.PositiveInfinity;
            foreach (var s in m.renderers.Where(s => s.included_in_static_body && s.geometry_valid))
            {
                for (int i = 0; i < s.triangle_indices.Length; i += 3)
                {
                    Vector3 a = Vertex(s, s.triangle_indices[i]), b = Vertex(s, s.triangle_indices[i + 1]), c = Vertex(s, s.triangle_indices[i + 2]);
                    float distance = TriangleDistanceSquared(p, a, b, c);
                    if (distance < best) { best = distance; probe.nearest_mesh = s.path; }
                }
            }
            probe.clearance_valid = !float.IsInfinity(best);
            probe.clearance_m = probe.clearance_valid ? Mathf.Sqrt(best) : -1f;
            m.candidates.Add(probe);
        }

        private static float TriangleDistanceSquared(Vector3 p, Vector3 a, Vector3 b, Vector3 c)
        {
            Vector3 ab = b - a, ac = c - a, n = Vector3.Cross(ab, ac);
            float nn = n.sqrMagnitude;
            if (nn > 1e-20f)
            {
                Vector3 q = p - n * (Vector3.Dot(p - a, n) / nn);
                if (Vector3.Dot(Vector3.Cross(b - a, q - a), n) >= 0f &&
                    Vector3.Dot(Vector3.Cross(c - b, q - b), n) >= 0f &&
                    Vector3.Dot(Vector3.Cross(a - c, q - c), n) >= 0f) return (p - q).sqrMagnitude;
            }
            return Mathf.Min(SegmentDistanceSquared(p, a, b), Mathf.Min(SegmentDistanceSquared(p, b, c), SegmentDistanceSquared(p, c, a)));
        }
        private static float SegmentDistanceSquared(Vector3 p, Vector3 a, Vector3 b)
        { Vector3 edge = b - a; return (p - (a + edge * (edge.sqrMagnitude == 0f ? 0f : Mathf.Clamp01(Vector3.Dot(p - a, edge) / edge.sqrMagnitude)))).sqrMagnitude; }
        private static Vector3 Vertex(Surface s, int i) => new Vector3(s.vertices_xyz_m[3 * i], s.vertices_xyz_m[3 * i + 1], s.vertices_xyz_m[3 * i + 2]);
        private static void AddTop(Top top, float z) { if (!top.valid || z > top.z_hub_m) top.z_hub_m = z; top.valid = true; }
        private static void Warn(Machine m, CapKitRig.Report report, string message) { m.warnings.Add(message); report.Warn(m.machine_id + ": " + message); }
        private static Vector3 Ros(Vector3 unity) => new Vector3(unity.z, -unity.x, unity.y);
        private static Vector3 Vec(float[] a) => new Vector3(a[0], a[1], a[2]);
        private static float[] Array(Vector3 v) => new[] { v.x, v.y, v.z };
        private static IEnumerable<Transform> Ancestors(Transform t, Transform root)
        { for (; t != null; t = t.parent) { yield return t; if (t == root) yield break; } }
        private static string PathOf(Transform t, Transform root) => string.Join("/", Ancestors(t, root).Reverse().Select(x => x.name));
    }
}
