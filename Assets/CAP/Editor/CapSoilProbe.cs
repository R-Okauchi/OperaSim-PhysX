using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace CapKit.Editor
{
    /// <summary>
    /// Read-only survey of what the particle-less soil (SoilParticleSettings.particles = false) rests on, from the
    /// scene as it loads (nothing saved):
    ///   Unity -batchmode -quit -projectPath ... -executeMethod CapKit.Editor.CapSoilProbe.Run
    /// env CAP_SCENE (default Assets/Scenes/SimpleScene.unity), CAP_SOIL_PROBE_OUT (default Temp/cap_soil_probe.json).
    /// - the terrain: size, heightmap resolution, the height range, and one dig's volume in m3 (the Dig kernel's
    ///   9 texels of diff / 22.5 in the heightmap texture's scale)
    /// - each bucket (BucketRocks): its link's colliders in the link's frame, the cutting edge (bucket_end_link),
    ///   and the mouth's direction two ways - normal to the plane of the pin axis and the cutting edge, away from
    ///   the shell's centroid; and the mean of the directions a ray from inside the cavity leaves the shell by
    /// - each vessel (VesselController): its colliders in its frame, the hinge (its ArticulationBody's anchor) and
    ///   the drive's limits
    /// </summary>
    public static class CapSoilProbe
    {
        private static readonly CultureInfo Inv = CultureInfo.InvariantCulture;

        public static void Run()
        {
            string scene = Environment.GetEnvironmentVariable("CAP_SCENE");
            if (string.IsNullOrWhiteSpace(scene)) scene = "Assets/Scenes/SimpleScene.unity";
            string output = Environment.GetEnvironmentVariable("CAP_SOIL_PROBE_OUT");
            if (string.IsNullOrWhiteSpace(output)) output = "Temp/cap_soil_probe.json";
            EditorSceneManager.OpenScene(scene, OpenSceneMode.Single);
            Physics.SyncTransforms();
            var sb = new StringBuilder();
            sb.Append("{\"scene\": \"").Append(scene).Append("\",\n");
            Terrains(sb);
            Buckets(sb);
            Vessels(sb);
            sb.Append("\"robots\": [").Append(string.Join(", ", GameObject.FindGameObjectsWithTag("robot")
                .Select(g => "\"" + g.name + "\""))).Append("]}\n");
            File.WriteAllText(output, sb.ToString());
            Debug.Log("[CapSoilProbe] -> " + output);
        }

        private static string V(Vector3 v) => string.Format(Inv, "[{0:F4}, {1:F4}, {2:F4}]", v.x, v.y, v.z);
        private static string F(float f) => f.ToString("G6", Inv);

        private static void Terrains(StringBuilder sb)
        {
            sb.Append("\"terrains\": [");
            bool first = true;
            foreach (var soil in UnityEngine.Object.FindObjectsOfType<SoilParticleSettings>())
            {
                var t = soil.GetComponent<Terrain>();
                var d = t.terrainData;
                int res = d.heightmapResolution;
                float[,] h = d.GetHeights(0, 0, res, res);
                float lo = float.MaxValue, hi = float.MinValue;
                foreach (var x in h) { lo = Mathf.Min(lo, x); hi = Mathf.Max(hi, x); }
                float particle = (float)(4.0 / 3.0 * Math.PI * Math.Pow(soil.particleVisualRadius, 3));
                // Dig.compute: 9 texels each += diff / 9 / 2.5 in the heightmap texture's scale (normalized height
                // times kNormalizedHeightScale = 32766 / 65535)
                float texel = particle / 9f / 2.5f;
                float dh = texel / (32766f / 65535f) * d.size.y;
                float dx = d.size.x / (res - 1), dz = d.size.z / (res - 1);
                float unit = 9f * dh * dx * dz;
                if (!first) sb.Append(", ");
                first = false;
                sb.Append("{\"name\": \"").Append(t.name).Append("\", \"enable\": ").Append(soil.enable ? "true" : "false")
                  .Append(", \"position\": ").Append(V(t.transform.position)).Append(", \"size\": ").Append(V(d.size))
                  .Append(", \"resolution\": ").Append(res).Append(", \"heights01\": [").Append(F(lo)).Append(", ")
                  .Append(F(hi)).Append("], \"particle_radius_m\": ").Append(F(soil.particleVisualRadius))
                  .Append(", \"particle_volume_m3\": ").Append(F(particle)).Append(", \"dig_texel_dh_m\": ")
                  .Append(F(dh)).Append(", \"texel_m\": [").Append(F(dx)).Append(", ").Append(F(dz))
                  .Append("], \"dig_unit_m3\": ").Append(F(unit)).Append(", \"tiled\": ")
                  .Append(soil.GetComponent<TerrainTiler>() != null ? "true" : "false").Append("}");
            }
            sb.Append("],\n");
        }

        private static string Colliders(Transform link, IEnumerable<Collider> colliders, out List<Vector3> points)
        {
            points = new List<Vector3>();
            var parts = new List<string>();
            foreach (var c in colliders)
            {
                var p = CapSoilGeometry.ColliderPoints(c, link).ToList();
                points.AddRange(p);
                Vector3 lo = p.Aggregate(Vector3.positiveInfinity, Vector3.Min), hi = p.Aggregate(Vector3.negativeInfinity, Vector3.Max);
                parts.Add("{\"path\": \"" + Path(c.transform, link) + "\", \"type\": \"" + c.GetType().Name + "\", \"trigger\": " +
                          (c.isTrigger ? "true" : "false") + ", \"enabled\": " + (c.enabled ? "true" : "false") +
                          ", \"layer\": " + c.gameObject.layer + ", \"min\": " + V(lo) + ", \"max\": " + V(hi) + "}");
            }
            return "[" + string.Join(",\n  ", parts) + "]";
        }

        private static string Path(Transform t, Transform root)
        {
            var names = new List<string>();
            for (var x = t; x != null && x != root; x = x.parent) names.Add(x.name);
            names.Reverse();
            return string.Join("/", names);
        }

        private static void Buckets(StringBuilder sb)
        {
            sb.Append("\"buckets\": [");
            bool first = true;
            foreach (var rocks in UnityEngine.Object.FindObjectsOfType<BucketRocks>())
            {
                Transform link = rocks.transform.parent;                 // bucket_inner sits under bucket_link
                var machine = rocks.transform.root.name;
                var tipT = link.Find("bucket_end_link");
                var own = CapSoilGeometry.BucketShell(link);
                string shell = Colliders(link, own, out var points);
                string inner = Colliders(link, rocks.GetComponentsInChildren<Collider>(true), out _);
                Vector3 tip = tipT != null ? link.InverseTransformPoint(tipT.position) : Vector3.zero;
                var mouth = CapSoilGeometry.BucketMouth(link, own);
                Vector3 centroid = mouth.centroid;
                var ab = link.GetComponent<ArticulationBody>();
                // the pin's axis in the link's frame: an ArticulationBody revolves about its anchor's x
                Vector3 axis = ab != null ? (ab.anchorRotation * Vector3.right).normalized : Vector3.right;
                Vector3 edge = tip - Vector3.Project(tip, axis);
                Vector3 off = centroid - Vector3.Project(centroid, axis);
                Vector3 inPlane = Vector3.Cross(axis, edge).normalized;   // the plane's normal (the pin axis, the edge)
                Vector3 planeMouth = Vector3.Dot(off, inPlane) > 0 ? -inPlane : inPlane;
                Vector3 rayMouth = mouth.direction;
                int escaped = mouth.escaped, cast = mouth.cast;
                float spanDeg = mouth.spanDeg;
                if (!first) sb.Append(",\n");
                first = false;
                sb.Append("{\"machine\": \"").Append(machine).Append("\", \"link\": \"").Append(link.name)
                  .Append("\", \"link_world\": ").Append(V(link.position)).Append(", \"link_rotation\": ")
                  .Append(V(link.rotation.eulerAngles)).Append(", \"pin_axis\": ").Append(V(axis))
                  .Append(", \"tip\": ").Append(V(tip)).Append(", \"centroid\": ").Append(V(centroid))
                  .Append(", \"mouth_plane\": ").Append(V(planeMouth)).Append(", \"mouth_rays\": ").Append(V(rayMouth))
                  .Append(", \"rays_escaped\": ").Append(escaped).Append(", \"rays_cast\": ").Append(cast)
                  .Append(", \"escape_span_deg\": ").Append(F(spanDeg))
                  .Append(", \"mouth_from_rays\": ").Append(mouth.fromRays ? "true" : "false")
                  .Append(", \"capacity_m3\": ").Append(F(CapSoilGeometry.BucketCapacityM3(CapSoilGeometry.Machine(rocks.transform).name)))
                  .Append(", \"mouth_world_rest\": ").Append(V(link.TransformDirection(rayMouth)))
                  .Append(",\n \"inner\": ").Append(inner).Append(",\n \"shell\": ").Append(shell).Append("}");
            }
            sb.Append("],\n");
        }

        private static void Vessels(StringBuilder sb)
        {
            sb.Append("\"vessels\": [");
            bool first = true;
            foreach (var vessel in UnityEngine.Object.FindObjectsOfType<VesselController>())
            {
                var link = vessel.transform;
                string cols = Colliders(link, link.GetComponentsInChildren<Collider>(true)
                                                  .Where(c => c.GetComponentInParent<VesselController>() == vessel), out var pts);
                Vector3 lo = pts.Count > 0 ? pts.Aggregate(Vector3.positiveInfinity, Vector3.Min) : Vector3.zero;
                Vector3 hi = pts.Count > 0 ? pts.Aggregate(Vector3.negativeInfinity, Vector3.Max) : Vector3.zero;
                var ab = vessel.GetComponent<ArticulationBody>();
                if (!first) sb.Append(",\n");
                first = false;
                sb.Append("{\"machine\": \"").Append(vessel.transform.root.name).Append("\", \"link\": \"").Append(link.name)
                  .Append("\", \"link_world\": ").Append(V(link.position)).Append(", \"link_rotation\": ")
                  .Append(V(link.rotation.eulerAngles)).Append(", \"min\": ").Append(V(lo)).Append(", \"max\": ").Append(V(hi));
                if (ab != null)
                {
                    var drive = ab.xDrive;
                    sb.Append(", \"anchor\": ").Append(V(ab.anchorPosition)).Append(", \"anchor_axis\": ")
                      .Append(V((ab.anchorRotation * Vector3.right).normalized)).Append(", \"joint\": \"").Append(ab.jointType.ToString())
                      .Append("\", \"limits_deg\": [").Append(F(drive.lowerLimit)).Append(", ").Append(F(drive.upperLimit))
                      .Append("], \"mass\": ").Append(F(ab.mass));
                }
                // CapSoilGeometry.BedOf against the colliders: a 3x3 grid of rays down inside the bed - where each meets
                // the vessel (its height in the vessel's frame, the table's floor min.y), and whether it is the floor
                var bed = CapSoilGeometry.BedOf(CapSoilGeometry.Machine(link).name, link);
                var own = link.GetComponentsInChildren<Collider>(true).Where(c => c.GetComponentInParent<VesselController>() == vessel).ToList();
                var floor = new List<string>();
                for (int i = 0; i < 3; i++)
                {
                    for (int j = 0; j < 3; j++)
                    {
                        var local = new Vector3(Mathf.Lerp(bed.min.x, bed.max.x, 0.15f + 0.35f * i), bed.max.y + 2f,
                                                Mathf.Lerp(bed.min.z, bed.max.z, 0.15f + 0.35f * j));
                        var ray = new Ray(link.TransformPoint(local), -link.up);
                        float best = float.MaxValue;
                        foreach (var c in own)
                        {
                            if (c.Raycast(ray, out var hit, 10f) && hit.distance < best) best = hit.distance;
                        }
                        floor.Add(best < float.MaxValue ? F(local.y - best) : "null");
                    }
                }
                sb.Append(", \"bed\": {\"min\": ").Append(V(bed.min)).Append(", \"max\": ").Append(V(bed.max))
                  .Append(", \"capacity_m3\": ").Append(F(bed.capacityM3)).Append(", \"known\": ").Append(bed.known ? "true" : "false")
                  .Append(", \"floor_hits_y\": [").Append(string.Join(", ", floor)).Append("]}");
                sb.Append(",\n \"colliders\": ").Append(cols).Append("}");
            }
            sb.Append("],\n");
        }
    }
}
