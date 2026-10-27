using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using UnityEngine;

/// <summary>
/// A run played back for its video (CAP's scripts/replay_video.py writes the pack, CapCycleRecorder's views capture
/// it): the region's terrain frames, a machine's pose and joints, and the people's walks, each at its time on the run's
/// Unity clock, at `speed` times its pace - with physics off and the machine's own drives, blade and sensors still. The
/// run itself is not touched: its numbers are the cluster's; this is only what it looked like.
///
/// The recorder's config: "replay": {"pack": ".../replay.json", "speed": 20, "t0": 0, "t1": 5400, "machine": "d37pxi",
///   "quit": true}. The pack:
///   {"frames": [{"t": 10.0, "file": ".../frame_10.f32", "x0": .., "y0": .., "cell_m": .., "nx": .., "ny": ..}, ...],
///    "poses": ".../poses_u.jsonl",          lines {"t": unity s, "pose": [x, y, z, qx, qy, qz, qw], "joints": {name: rad}}
///    "people": [{"name": "person-1", "path": [x0, y0, x1, y1, ...], "speed_mps": 1.2, "start_t": 1234.5}]}
/// (ROS/map frame, as the run's records: Unity = (-y, z, x)).
/// </summary>
public class CapReplay : MonoBehaviour
{
    [Serializable]
    public class Config
    {
        public string pack;
        public float speed = 20f;
        public float t0;
        public float t1 = -1f;                 // <0: the pack's last pose
        public string machine = "d37pxi";
        public bool quit = true;
        public string control;                 // the recorder's control.json: recording stopped at the end
    }

    [Serializable]
    private class Frame
    {
        public float t;
        public string file;
        public float x0, y0, cell_m;
        public int nx, ny;
    }

    [Serializable]
    private class PersonWalk
    {
        public string name;
        public float[] path;
        public float speed_mps = 1.2f;
        public float start_t;
    }

    [Serializable]
    private class Pack
    {
        public Frame[] frames;
        public string poses;
        public PersonWalk[] people;
    }

    private struct PoseRow
    {
        public float t;
        public Vector3 pos;
        public Quaternion rot;
        public Dictionary<string, float> joints;
    }

    private class JointPlay
    {
        public string name;
        public Transform link;
        public Quaternion parentAnchorRot, anchorRot;
        public Vector3 parentAnchorPos, anchorPos;
    }

    public Config config;
    private Pack pack;
    private readonly List<PoseRow> poses = new List<PoseRow>();
    private readonly List<JointPlay> joints = new List<JointPlay>();
    private readonly List<(PersonWalk walk, Transform body)> people = new List<(PersonWalk, Transform)>();
    private Transform hub;
    private int nextFrame;
    private int poseAt;
    private float startReal;
    private float t;
    private bool ended;
    private float endedAt;

    void Start()
    {
        pack = JsonUtility.FromJson<Pack>(File.ReadAllText(config.pack));
        ReadPoses(pack.poses);
        if (config.t1 < 0f && poses.Count > 0) config.t1 = poses[poses.Count - 1].t;
        Physics.simulationMode = SimulationMode.Script;            // nothing moves but what the replay sets
        var machine = GameObject.Find(config.machine);
        if (machine != null) Still(machine);
        foreach (var w in pack.people ?? new PersonWalk[0])
        {
            var body = GameObject.CreatePrimitive(PrimitiveType.Capsule);
            body.name = w.name;
            body.transform.localScale = new Vector3(0.6f, 0.9f, 0.6f);   // 1.8 m tall
            Destroy(body.GetComponent<Collider>());
            body.SetActive(false);
            people.Add((w, body.transform));
        }
        t = config.t0;
        startReal = Time.realtimeSinceStartup;
        Apply(t);
        Debug.Log($"[{nameof(CapReplay)}] {poses.Count} poses, {pack.frames?.Length ?? 0} frames, {people.Count} people: " +
                  $"t {config.t0:F0}..{config.t1:F0} at x{config.speed:F0}");
    }

    void Update()
    {
        if (ended)
        {
            if (config.quit && Time.realtimeSinceStartup - endedAt > 3f) Application.Quit();
            return;
        }
        t = config.t0 + (Time.realtimeSinceStartup - startReal) * config.speed;
        Apply(t);
        if (t >= config.t1)
        {
            ended = true;
            endedAt = Time.realtimeSinceStartup;
            if (!string.IsNullOrEmpty(config.control))
                File.WriteAllText(config.control, "{\"recording\": false}");
            Debug.Log($"[{nameof(CapReplay)}] ended at t {t:F0}");
        }
    }

    /// <summary>The replay's time on the run's Unity clock (for a caption).</summary>
    public float Now => t;

    private void Apply(float at)
    {
        // the latest frame due (the ones it passes over are not drawn); none while the soil is not up yet
        int due = nextFrame;
        while (pack.frames != null && due < pack.frames.Length && pack.frames[due].t <= at) due++;
        if (due > nextFrame && ApplyFrame(pack.frames[due - 1])) nextFrame = due;
        ApplyPose(at);
        foreach (var (walk, body) in people)
        {
            var p = Along(walk, at);
            body.gameObject.SetActive(p.HasValue);
            if (p.HasValue) body.position = p.Value;
        }
    }

    // ---- the machine

    /// <summary>The machine as a model only: its scripts off (drives, blade, MC, sensors, publishers), its joints read
    /// for their anchors and then its physics taken off (articulations, bodies, joints), deepest first.</summary>
    private void Still(GameObject machine)
    {
        var drive = machine.GetComponentInChildren<CapBladeDrive>(true);
        if (drive != null)
        {
            foreach (var j in drive.joints)
            {
                if (j.body == null) continue;
                joints.Add(new JointPlay
                {
                    name = j.name, link = j.body.transform,
                    parentAnchorRot = j.body.parentAnchorRotation, anchorRot = j.body.anchorRotation,
                    parentAnchorPos = j.body.parentAnchorPosition, anchorPos = j.body.anchorPosition,
                });
            }
        }
        foreach (var t in machine.GetComponentsInChildren<Transform>(true))
            if (t.name == "base_link") hub = t;
        foreach (var b in machine.GetComponentsInChildren<MonoBehaviour>(true))
            if (b != null && b != this) b.enabled = false;
        var bodies = new List<ArticulationBody>(machine.GetComponentsInChildren<ArticulationBody>(true));
        bodies.Sort((a, b) => Depth(b.transform).CompareTo(Depth(a.transform)));
        foreach (var j in machine.GetComponentsInChildren<Joint>(true)) DestroyImmediate(j);
        foreach (var a in bodies) DestroyImmediate(a);
        foreach (var r in machine.GetComponentsInChildren<Rigidbody>(true)) DestroyImmediate(r);
        Debug.Log($"[{nameof(CapReplay)}] {machine.name}: {joints.Count} joints, {bodies.Count} articulations taken off");
    }

    private static int Depth(Transform t)
    {
        int d = 0;
        for (var p = t; p != null; p = p.parent) d++;
        return d;
    }

    private void ReadPoses(string path)
    {
        if (string.IsNullOrEmpty(path) || !File.Exists(path)) return;
        foreach (var line in File.ReadLines(path))
        {
            if (string.IsNullOrWhiteSpace(line)) continue;
            var row = PoseLine.Parse(line);
            if (row.pose == null || row.pose.Length < 7) continue;
            // ROS/map -> Unity: position (-y, z, x); orientation as ROS-TCP's FLU -> RUF
            poses.Add(new PoseRow
            {
                t = row.t,
                pos = new Vector3(-row.pose[1], row.pose[2], row.pose[0]),
                rot = new Quaternion(-row.pose[4], row.pose[5], row.pose[3], -row.pose[6]),
                joints = row.joints,
            });
        }
    }

    private void ApplyPose(float at)
    {
        if (poses.Count == 0 || hub == null) return;
        while (poseAt + 1 < poses.Count && poses[poseAt + 1].t <= at) poseAt++;
        var a = poses[poseAt];
        var b = poseAt + 1 < poses.Count ? poses[poseAt + 1] : a;
        float u = b.t > a.t ? Mathf.Clamp01((at - a.t) / (b.t - a.t)) : 0f;
        hub.SetPositionAndRotation(Vector3.Lerp(a.pos, b.pos, u), Quaternion.Slerp(a.rot, b.rot, u));
        foreach (var j in joints)
        {
            if (!a.joints.TryGetValue(j.name, out float qa)) continue;
            float qb = b.joints.TryGetValue(j.name, out float v) ? v : qa;
            float q = Mathf.Lerp(qa, qb, u) * Mathf.Rad2Deg;
            // an articulation's revolute joint turns its link about the X of its anchor frame:
            // child-in-parent = parentAnchor * AngleAxis(q, x) * inverse(anchor)
            var rot = j.parentAnchorRot * Quaternion.AngleAxis(q, Vector3.right) * Quaternion.Inverse(j.anchorRot);
            j.link.localRotation = rot;
            j.link.localPosition = j.parentAnchorPos - rot * j.anchorPos;
        }
    }

    // ---- the terrain

    /// <summary>A region frame (heights [m, world], rows of ROS x rising within ROS y rising - CapScenarioOps.WriteRegion)
    /// into the soil's working heightmap, committed to the terrain.</summary>
    private bool ApplyFrame(Frame f)
    {
        var soil = SoilParticleSettings.instance;
        if (soil == null || soil.WorkingHeightmap == null) return false;
        if (!File.Exists(f.file)) return true;                       // gone: the next one
        var bytes = File.ReadAllBytes(f.file);
        int n = f.nx * f.ny;
        if (bytes.Length < n * 4) return true;
        var terrain = soil.GetComponent<Terrain>();
        Vector3 p = terrain.transform.position, size = terrain.terrainData.size;
        var res = soil.HeightmapSize;
        float cell = size.x / (res.x - 1);
        float scale = size.y / (32766f / 65535f);
        // the frame's first row's ROS x -> texel y; its first column's ROS y (the lowest) -> the highest texel x
        int ty0 = Mathf.RoundToInt((f.x0 - p.z) / cell);
        int tx1 = Mathf.RoundToInt((-f.y0 - p.x) / cell);
        int w = f.ny, h = f.nx, tx0 = tx1 - (w - 1);
        if (tx0 < 0 || ty0 < 0 || tx1 >= res.x || ty0 + h > res.y) return true;
        var px = new float[w * h];                                // the working heightmap is RFloat
        for (int j = 0; j < w; j++)
        for (int i = 0; i < h; i++)
        {
            float height = BitConverter.ToSingle(bytes, (j * f.nx + i) * 4);
            int u = (tx1 - j) - tx0, v = i;
            px[v * w + u] = (height - p.y) / scale;
        }
        var tex = new Texture2D(w, h, TextureFormat.RFloat, false, true);
        tex.SetPixelData(px, 0);
        tex.Apply(false, false);
        Graphics.CopyTexture(tex, 0, 0, 0, 0, w, h, soil.WorkingHeightmap, 0, 0, tx0, ty0);
        Destroy(tex);
        soil.Commit(new RectInt(tx0, ty0, w, h));
        return true;
    }

    // ---- people

    private static Vector3? Along(PersonWalk w, float at)
    {
        if (w.path == null || w.path.Length < 4 || at < w.start_t) return null;
        float s = (at - w.start_t) * Mathf.Max(0.01f, w.speed_mps);
        for (int i = 0; i + 3 < w.path.Length; i += 2)
        {
            var a = new Vector2(w.path[i], w.path[i + 1]);
            var b = new Vector2(w.path[i + 2], w.path[i + 3]);
            float d = Vector2.Distance(a, b);
            if (s <= d)
            {
                var q = Vector2.Lerp(a, b, d > 0f ? s / d : 0f);
                var world = new Vector3(-q.y, 0f, q.x);
                var terrain = Terrain.activeTerrain;
                world.y = (terrain != null ? terrain.SampleHeight(world) + terrain.transform.position.y : 0f) + 0.9f;
                return world;
            }
            s -= d;
        }
        return null;                                               // walked its path: gone
    }

    // ---- a pose line ({"t": .., "pose": [..], "joints": {..}}): JsonUtility reads no dictionaries

    private struct PoseLine
    {
        public float t;
        public float[] pose;
        public Dictionary<string, float> joints;

        public static PoseLine Parse(string line)
        {
            var row = new PoseLine { joints = new Dictionary<string, float>() };
            row.t = Number(line, "\"t\":");
            int ps = line.IndexOf("\"pose\":", StringComparison.Ordinal);
            if (ps >= 0)
            {
                int a = line.IndexOf('[', ps), b = line.IndexOf(']', a + 1);
                if (a >= 0 && b > a)
                {
                    var parts = line.Substring(a + 1, b - a - 1).Split(',');
                    row.pose = new float[parts.Length];
                    for (int i = 0; i < parts.Length; i++)
                        row.pose[i] = float.Parse(parts[i].Trim(), CultureInfo.InvariantCulture);
                }
            }
            int js = line.IndexOf("\"joints\":", StringComparison.Ordinal);
            if (js >= 0)
            {
                int a = line.IndexOf('{', js), b = line.IndexOf('}', a + 1);
                if (a >= 0 && b > a)
                {
                    foreach (var kv in line.Substring(a + 1, b - a - 1).Split(','))
                    {
                        var pair = kv.Split(':');
                        if (pair.Length != 2) continue;
                        var key = pair[0].Trim().Trim('"');
                        if (float.TryParse(pair[1].Trim(), NumberStyles.Float, CultureInfo.InvariantCulture, out float v))
                            row.joints[key] = v;
                    }
                }
            }
            return row;
        }

        private static float Number(string line, string key)
        {
            int i = line.IndexOf(key, StringComparison.Ordinal);
            if (i < 0) return 0f;
            i += key.Length;
            int j = i;
            while (j < line.Length && (char.IsDigit(line[j]) || line[j] == '.' || line[j] == '-' || line[j] == 'e' ||
                                       line[j] == 'E' || line[j] == '+' || line[j] == ' ')) j++;
            return float.Parse(line.Substring(i, j - i).Trim(), CultureInfo.InvariantCulture);
        }
    }
}
