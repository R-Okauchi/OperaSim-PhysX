using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Reflection;
using System.Text;
using RosMessageTypes.Std;
using Unity.Robotics.ROSTCPConnector;
using UnityEngine;

/// <summary>
/// A scenario's set-up and its live operations, from CapCycleRecorder's config ("scenario"), so one scene serves many
/// scenarios (CAP's scenario runner writes the config; coordinates are ROS/map [m], as the machines' poses):
/// - before the soil starts (AfterSceneLoad): the ground shaped first (flatten a rectangle to a height with a little
///   roughness: ShapeTerrain, before the scene's machines are placed), then machines spawned from Resources/Prefab (the
///   D37PXi) or cloned from the scene's, or switched off (Prepare);
/// - once the soil runs: piles laid as cones (exact volumes, CapSoilStamp), a dump truck's stand-in that lays the piles
///   a Site asks for on /scenario/dump_request after a delay (and says so on /scenario/dump_done; as a driver would,
///   not onto a machine: while one stands within the pile's foot and clearM of the spot the truck waits, up to
///   dump_wait_s, then says it did not dump), the region's heights written every period (a float file per dump: the
///   scorer's error-map frames) and on request (control.json "terrain_dump": "&lt;name&gt;", for the final score),
///   people walking given paths (capsules, for the safety tests: each says how close it came to every machine), and
///   faults switched on at given times or when asked (/scenario/fault_request: {kind, machine, target, duration_s}, a
///   person {kind: "person", target: its name, path, speed_mps} - a harness's fault on its own run's events).
/// </summary>
public class CapScenarioOps : MonoBehaviour
{
    [Serializable]
    public class Spawn
    {
        public string prefab;               // Resources/Prefab/<prefab>, else the scene's machine of that name (cloned)
        public string name;                 // the root's name (its topics' namespace)
        public float x;
        public float y;
        public float yaw_deg;               // ROS: from +x, counter-clockwise
    }

    [Serializable]
    public class TerrainOp
    {
        public string op;                   // flatten, ramp (the height going from z to z_to along the rectangle), or
                                            // mesa (a flat top with sloped sides run_m wide all round it)
        public float x0, y0, x1, y1;        // ROS rectangle
        public float z;                     // the height [m] (world), or above the rectangle's mean height (z_relative)
        public bool z_relative;
        public float z_to;                  // ramp: the height [m] (world) at the rectangle's far end (x1 or y1) ...
        public string along = "x";          // ... going along ROS x (from x0 to x1) or y (from y0 to y1)
        public float run_m;                 // mesa: its sides' horizontal run [m] (their slope: atan(height / run))
        public float noise_m;               // roughness: +-noise_m, smooth over ~1 m
        public int seed;
    }

    [Serializable]
    public class Pile
    {
        public float x, y;
        public float volume_m3;
        public float repose_deg = 34f;
    }

    [Serializable]
    public class Region
    {
        public float x0, y0, x1, y1;        // ROS rectangle
        public float period_s;              // >0: its heights every period
        public string prefix;               // <prefix>_<seconds>.f32 + .json
    }

    [Serializable]
    public class Person
    {
        public string name = "person";
        public float[] path;                // x0, y0, x1, y1, ... (ROS) walked once at speed
        public float speed_mps = 1.2f;
        public float start_s;
    }

    [Serializable]
    public class Fault
    {
        public string kind;                 // blade_stall | lidar_off | person
        public string machine;
        public string target;               // lidar_off: the sensor's object name; person: the person's name
        public float at_s;
        public float duration_s;
        public float[] path;                // person: x0, y0, x1, y1, ... (ROS), walked once
        public float speed_mps;             // person: its speed (1.2 m/s when not given)
    }

    [Serializable]
    public class Config
    {
        public Spawn[] spawn;
        public string[] disable;
        public TerrainOp[] terrain;
        public Pile[] piles;
        public bool dump_service;
        public float dump_delay_s = 120f;
        public float dump_repose_deg = 34f;
        public float dump_wait_s = 600f;    // a truck waits this long at a spot a machine stands on, then gives up
        public Region region;
        public Person[] people;
        public Fault[] faults;
    }

    public Config config;
    public string outDir;                   // the recorder's: the ground before the piles to <outDir>/terrain/ground
    private float t0;
    private ROSConnection ros;
    private readonly Queue<(float at, float x, float y, float v, string id)> dumps = new Queue<(float, float, float, float, string)>();
    private float dumpWaitSince = -1f;
    public float clearM = 4f;               // a machine's hub this far beyond a pile's foot leaves the spot clear

    // ---- before the soil starts (CapCycleRecorder.Boot, AfterSceneLoad)

    public static void Prepare(Config cfg)
    {
        if (cfg == null) return;
        if (cfg.disable != null)
        {
            foreach (var n in cfg.disable)
            {
                var go = GameObject.Find(n);
                if (go != null)
                {
                    go.SetActive(false);
                    Debug.Log($"[{nameof(CapScenarioOps)}] {n} switched off");
                }
            }
        }
        if (cfg.spawn != null)
        {
            foreach (var s in cfg.spawn)
            {
                var prefab = Resources.Load<GameObject>("Prefab/" + s.prefab);
                // No such prefab: a machine of the scene by that name, cloned with what the scene gives it (its kit, its
                // controllers) - a second mst110cr. Cloned asleep and woken under its own name: its topics take the
                // root's name when they start ([robot_name]), never "<name>(Clone)".
                var scene = prefab == null ? GameObject.Find(s.prefab) : null;
                if (prefab == null && scene == null)
                {
                    Debug.LogWarning($"[{nameof(CapScenarioOps)}] no Resources/Prefab/{s.prefab} and no {s.prefab} in the scene");
                    continue;
                }
                var at = new Vector3(-s.y, 0f, s.x);
                float g = Ground(at);
                at.y = (float.IsNaN(g) ? 0f : g) + 0.6f;
                // ROS yaw (counter-clockwise from +x) is Unity's clockwise turn about up from +z (ROS +x)
                GameObject go;
                if (prefab != null)
                {
                    go = Instantiate(prefab, at, Quaternion.Euler(0f, -s.yaw_deg, 0f));
                }
                else
                {
                    bool awake = scene.activeSelf;
                    scene.SetActive(false);
                    go = Instantiate(scene, at, Quaternion.Euler(0f, -s.yaw_deg, 0f));
                    scene.SetActive(awake);
                }
                go.name = string.IsNullOrEmpty(s.name) ? s.prefab : s.name;
                if (prefab == null && go.name != s.prefab)
                {
                    // The source's components resolved [robot_name] to its name when they started and keep it in their
                    // fields, and a clone copies what they hold: renamed here, before it wakes (11-01, the chain's
                    // rehearsal 20: mst110cr_2 published on mst110cr's topics, none of its own ever seen).
                    int renamed = RenameMachine(go, s.prefab, go.name);
                    Debug.Log($"[{nameof(CapScenarioOps)}] {go.name}: {renamed} names of {s.prefab} in its components made its own");
                }
                go.SetActive(true);
                Debug.Log($"[{nameof(CapScenarioOps)}] spawned {go.name} at ({s.x:F1}, {s.y:F1}) yaw {s.yaw_deg:F0} deg");
            }
        }
    }

    /// <summary>The scenario's ground shaped (its terrain ops, in order): before the scene's machines are placed and
    /// the spawned ones made, so that each stands on the ground it is given - a bench a loader stands on raised under
    /// it, not into it (11-01: the five machines' chain puts zx120 and its mound on a bench 1.5 m up; shaped after the
    /// placements, the ground would have risen through its tracks).</summary>
    public static void ShapeTerrain(Config cfg)
    {
        if (cfg?.terrain == null) return;
        foreach (var op in cfg.terrain) Shape(op);
    }

    /// <summary>Each string a component under <paramref name="go"/> holds - its fields of any access, its own and its
    /// bases', strings in arrays and lists too - with every '/'-separated segment equal to <paramref name="from"/> made
    /// <paramref name="to"/>: the topics, frames and robot names its source resolved when it started. The count of
    /// strings changed. Arrays and lists are replaced, never changed in place (a clone may share one with its source).</summary>
    public static int RenameMachine(GameObject go, string from, string to)
    {
        int n = 0;
        const BindingFlags flags = BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.DeclaredOnly;
        foreach (var mb in go.GetComponentsInChildren<MonoBehaviour>(true))
        {
            if (mb == null) continue;
            for (var t = mb.GetType(); t != null && t != typeof(MonoBehaviour) && t != typeof(Behaviour) && t != typeof(Component); t = t.BaseType)
            {
                foreach (var f in t.GetFields(flags))
                {
                    if (f.IsLiteral || f.IsInitOnly) continue;
                    if (f.FieldType == typeof(string))
                    {
                        var v = (string)f.GetValue(mb);
                        var r = RenameSegment(v, from, to);
                        if (!ReferenceEquals(r, v)) { f.SetValue(mb, r); n++; }
                    }
                    else if (f.FieldType == typeof(string[]) && f.GetValue(mb) is string[] a)
                    {
                        var copy = (string[])a.Clone();
                        int k = 0;
                        for (int i = 0; i < copy.Length; i++)
                        {
                            var r = RenameSegment(copy[i], from, to);
                            if (!ReferenceEquals(r, copy[i])) { copy[i] = r; k++; }
                        }
                        if (k > 0) { f.SetValue(mb, copy); n += k; }
                    }
                    else if (f.FieldType == typeof(List<string>) && f.GetValue(mb) is List<string> l)
                    {
                        var copy = new List<string>(l);
                        int k = 0;
                        for (int i = 0; i < copy.Count; i++)
                        {
                            var r = RenameSegment(copy[i], from, to);
                            if (!ReferenceEquals(r, copy[i])) { copy[i] = r; k++; }
                        }
                        if (k > 0) { f.SetValue(mb, copy); n += k; }
                    }
                }
            }
        }
        return n;
    }

    /// <summary><paramref name="v"/> with each '/'-separated segment equal to <paramref name="from"/> made
    /// <paramref name="to"/> ("mst110cr/global_pose", "/mst110cr/kit/imu", "mst110cr"); the same string when it has
    /// none - a name that only begins with it ("mst110cr_2", "mst110cr-body") is not it.</summary>
    public static string RenameSegment(string v, string from, string to)
    {
        if (string.IsNullOrEmpty(v) || string.IsNullOrEmpty(from) || v.IndexOf(from, StringComparison.Ordinal) < 0) return v;
        var parts = v.Split('/');
        bool changed = false;
        for (int i = 0; i < parts.Length; i++)
        {
            if (parts[i] == from) { parts[i] = to; changed = true; }
        }
        return changed ? string.Join("/", parts) : v;
    }

    private static float Ground(Vector3 at)
    {
        foreach (var t in Terrain.activeTerrains)
        {
            var b = new Bounds(t.terrainData.bounds.center + t.transform.position, t.terrainData.bounds.size);
            if (at.x >= b.min.x && at.x <= b.max.x && at.z >= b.min.z && at.z <= b.max.z)
                return t.SampleHeight(at) + t.transform.position.y;
        }
        return float.NaN;
    }

    /// <summary>Flatten a ROS rectangle of the terrain to a height, with smooth roughness (the CPU heights: the soil's
    /// working heightmap is built from them when it starts) - or, a ramp, from z at its near end to z_to at its far one
    /// along ROS x or y (a machine's way up onto a bench: 11-02, CAP's chain from the machines' yard).</summary>
    private static void Shape(TerrainOp op)
    {
        if (op.op == "mesa") { Mesa(op); return; }
        bool ramp = op.op == "ramp";
        if (op.op != "flatten" && !ramp) return;
        var terrain = Terrain.activeTerrain;
        if (terrain == null) return;
        var d = terrain.terrainData;
        int res = d.heightmapResolution;
        Vector3 p = terrain.transform.position, size = d.size;
        // ROS x <- Unity z (rows), ROS y <- -Unity x (columns)
        float ux0 = -op.y1, ux1 = -op.y0, uz0 = op.x0, uz1 = op.x1;
        int c0 = Mathf.Clamp(Mathf.FloorToInt((ux0 - p.x) / size.x * (res - 1)), 0, res - 1);
        int c1 = Mathf.Clamp(Mathf.CeilToInt((ux1 - p.x) / size.x * (res - 1)), 0, res - 1);
        int r0 = Mathf.Clamp(Mathf.FloorToInt((uz0 - p.z) / size.z * (res - 1)), 0, res - 1);
        int r1 = Mathf.Clamp(Mathf.CeilToInt((uz1 - p.z) / size.z * (res - 1)), 0, res - 1);
        int w = c1 - c0 + 1, h = r1 - r0 + 1;
        float z = op.z;
        if (op.z_relative)
        {
            var now = d.GetHeights(c0, r0, w, h);
            double sum = 0.0;
            foreach (var v in now) sum += v;
            z += (float)(sum / now.Length) * size.y + p.y;
        }
        var heights = new float[h, w];
        var rnd = new System.Random(op.seed);
        float ox = (float)rnd.NextDouble() * 100f, oy = (float)rnd.NextDouble() * 100f;
        float cell = size.x / (res - 1);
        for (int r = 0; r < h; r++)
        for (int c = 0; c < w; c++)
        {
            float wx = (c0 + c) * cell, wz = (r0 + r) * cell;
            float n = op.noise_m > 0f ? (Mathf.PerlinNoise(ox + wx * 0.8f, oy + wz * 0.8f) * 2f - 1f) * op.noise_m : 0f;
            float at = z;
            if (ramp)
            {
                // ROS x <- Unity z, ROS y <- -Unity x: this vertex's way along the ramp, 0 at its near end, 1 at its far
                float t = op.along == "y" ? (-(p.x + wx) - op.y0) / Mathf.Max(op.y1 - op.y0, 1e-3f)
                                          : ((p.z + wz) - op.x0) / Mathf.Max(op.x1 - op.x0, 1e-3f);
                at = Mathf.Lerp(op.z, op.z_to, Mathf.Clamp01(t));
            }
            heights[r, c] = Mathf.Clamp01((at + n - p.y) / size.y);
        }
        d.SetHeights(c0, r0, heights);
        if (ramp)
            Debug.Log($"[{nameof(CapScenarioOps)}] ramp ({op.x0}, {op.y0})..({op.x1}, {op.y1}) from {op.z:F3} to {op.z_to:F3} m along {op.along}");
        else
            Debug.Log($"[{nameof(CapScenarioOps)}] flattened ({op.x0}, {op.y0})..({op.x1}, {op.y1}) to {z:F3} m +-{op.noise_m} m");
    }

    /// <summary>A mesa: the ROS rectangle raised to z (world, or over its mean height: z_relative) and its sides sloping
    /// down to the ground about it over run_m (rounded at its corners: by the distance to the rectangle) - never below
    /// the ground there. A bench whose edge no machine's planner reads as a cliff, that a loader drives up onto from any
    /// side (11-02, CAP's chain from the machines' yard: a 1.5 m cliff's steep band blocked the carrier's slot below
    /// it and the loader's stand on it at once).</summary>
    private static void Mesa(TerrainOp op)
    {
        var terrain = Terrain.activeTerrain;
        if (terrain == null || op.run_m <= 0f) return;
        var d = terrain.terrainData;
        int res = d.heightmapResolution;
        Vector3 p = terrain.transform.position, size = d.size;
        float run = op.run_m;
        // ROS x <- Unity z (rows), ROS y <- -Unity x (columns): the top grown by its sides
        float ux0 = -op.y1 - run, ux1 = -op.y0 + run, uz0 = op.x0 - run, uz1 = op.x1 + run;
        int c0 = Mathf.Clamp(Mathf.FloorToInt((ux0 - p.x) / size.x * (res - 1)), 0, res - 1);
        int c1 = Mathf.Clamp(Mathf.CeilToInt((ux1 - p.x) / size.x * (res - 1)), 0, res - 1);
        int r0 = Mathf.Clamp(Mathf.FloorToInt((uz0 - p.z) / size.z * (res - 1)), 0, res - 1);
        int r1 = Mathf.Clamp(Mathf.CeilToInt((uz1 - p.z) / size.z * (res - 1)), 0, res - 1);
        int w = c1 - c0 + 1, h = r1 - r0 + 1;
        var now = d.GetHeights(c0, r0, w, h);
        float z = op.z;
        if (op.z_relative)
        {
            // over the top's own mean height (its ground's, the site flattened first)
            double sum = 0.0;
            int n = 0;
            float cell0 = size.x / (res - 1);
            for (int r = 0; r < h; r++)
            for (int c = 0; c < w; c++)
            {
                float rx = p.z + (r0 + r) * cell0, ry = -(p.x + (c0 + c) * cell0);
                if (rx < op.x0 || rx > op.x1 || ry < op.y0 || ry > op.y1) continue;
                sum += now[r, c];
                n++;
            }
            z += (n > 0 ? (float)(sum / n) : 0f) * size.y + p.y;
        }
        float cell = size.x / (res - 1);
        for (int r = 0; r < h; r++)
        for (int c = 0; c < w; c++)
        {
            float rx = p.z + (r0 + r) * cell, ry = -(p.x + (c0 + c) * cell);
            float dx = Mathf.Max(0f, Mathf.Max(op.x0 - rx, rx - op.x1));
            float dy = Mathf.Max(0f, Mathf.Max(op.y0 - ry, ry - op.y1));
            float off = Mathf.Sqrt(dx * dx + dy * dy);
            if (off >= run) continue;
            float ground = now[r, c] * size.y + p.y;
            float at = Mathf.Lerp(z, ground, off / run);
            now[r, c] = Mathf.Clamp01((Mathf.Max(at, ground) - p.y) / size.y);
        }
        d.SetHeights(c0, r0, now);
        Debug.Log($"[{nameof(CapScenarioOps)}] mesa ({op.x0}, {op.y0})..({op.x1}, {op.y1}) at {z:F3} m, its sides {run:F1} m");
    }

    // ---- once the soil runs

    IEnumerator Start()
    {
        t0 = Time.time;
        while (SoilParticleSettings.instance == null || SoilParticleSettings.instance.WorkingHeightmap == null)
            yield return null;
        var soil = SoilParticleSettings.instance;
        if (config.region != null && !string.IsNullOrEmpty(outDir))
        {
            string dir = Path.Combine(outDir, "terrain");
            Directory.CreateDirectory(dir);
            WriteRegion(config.region, Path.Combine(dir, "ground"));       // before any pile: the score's ground
            Debug.Log($"[{nameof(CapScenarioOps)}] the ground before the piles -> {dir}/ground");
        }
        if (config.piles != null)
        {
            double laid = 0.0;
            foreach (var pile in config.piles)
                laid += CapSoilStamp.Cone(soil, new Vector3(-pile.y, 0f, pile.x), pile.volume_m3, pile.repose_deg);
            Debug.Log($"[{nameof(CapScenarioOps)}] {config.piles.Length} piles, {laid:F3} m3");
        }
        ros = ROSConnection.GetOrCreateInstance();
        if (config.dump_service)
        {
            ros.Subscribe<StringMsg>("/scenario/dump_request", OnDumpRequest);
            ros.RegisterPublisher<StringMsg>("/scenario/dump_done");
        }
        ros.Subscribe<StringMsg>("/scenario/fault_request", OnFaultRequest);
        if (config.people != null)
        {
            foreach (var person in config.people) StartCoroutine(Walk(person));
        }
        if (config.faults != null)
        {
            foreach (var f in config.faults) StartCoroutine(FaultAt(f));
        }
        if (config.region != null && config.region.period_s > 0f && !string.IsNullOrEmpty(config.region.prefix))
            StartCoroutine(RegionDumps());
    }

    [Serializable]
    private class DumpRequest
    {
        public string id;
        public float x, y;
        public float volume_m3;
    }

    private void OnDumpRequest(StringMsg msg)
    {
        DumpRequest r;
        try { r = JsonUtility.FromJson<DumpRequest>(msg.data); }
        catch (Exception e)
        {
            Debug.LogWarning($"[{nameof(CapScenarioOps)}] dump request unreadable: {e.Message}");
            return;
        }
        float at = Time.time + config.dump_delay_s * (dumps.Count + 1);       // one truck after another
        dumps.Enqueue((at, r.x, r.y, r.volume_m3, r.id));
        Debug.Log($"[{nameof(CapScenarioOps)}] dump {r.id} of {r.volume_m3} m3 at ({r.x}, {r.y}) in {at - Time.time:F0} s");
    }

    /// <summary>A machine's root (tag robot, active) within the pile's foot and clearM of the spot (ROS x, y).</summary>
    private bool Blocked(float x, float y, float volume)
    {
        float t = Mathf.Tan(config.dump_repose_deg * Mathf.Deg2Rad);
        float foot = Mathf.Pow(3f * volume / (Mathf.PI * t), 1f / 3f);
        var spot = new Vector2(-y, x);                                  // Unity's x, z
        foreach (var m in GameObject.FindGameObjectsWithTag("robot"))
        {
            if (!m.activeInHierarchy || m.transform.parent != null) continue;
            var p = m.transform.position;
            if (Vector2.Distance(new Vector2(p.x, p.z), spot) <= foot + clearM) return true;
        }
        return false;
    }

    void Update()
    {
        while (dumps.Count > 0 && dumps.Peek().at <= Time.time)
        {
            var d = dumps.Peek();
            bool landed = true;
            if (Blocked(d.x, d.y, d.v))
            {
                if (dumpWaitSince < 0f)
                {
                    dumpWaitSince = Time.time;
                    Debug.Log($"[{nameof(CapScenarioOps)}] dump {d.id}: a machine at ({d.x}, {d.y}): the truck waits");
                }
                if (Time.time - dumpWaitSince < config.dump_wait_s) break;  // still waiting (the next frame looks again)
                landed = false;
            }
            dumps.Dequeue();
            float waited = dumpWaitSince < 0f ? 0f : Time.time - dumpWaitSince;
            dumpWaitSince = -1f;
            double laid = landed ? CapSoilStamp.Cone(SoilParticleSettings.instance, new Vector3(-d.y, 0f, d.x), d.v,
                                                     config.dump_repose_deg) : 0.0;
            ros?.Publish("/scenario/dump_done", new StringMsg(DumpDoneJson(d.id, d.x, d.y, laid, landed, waited)));
            Debug.Log($"[{nameof(CapScenarioOps)}] {(landed ? "dumped" : "not dumped (the spot never cleared)")} {d.id}: {laid:F3} m3 at ({d.x}, {d.y})");
        }
    }

    /// <summary>A load's /scenario/dump_done: what landed where, whether, after how long a wait. Each number written on
    /// its own: composite formatting reads "{5:F1}}}" as the format "F1}" (the "}}" inside the item is an escaped
    /// brace) and wrote "waited_s": F1} - JSON no reader takes (10-26: the Site dropped every load's done).</summary>
    public static string DumpDoneJson(string id, float x, float y, double laidM3, bool landed, float waitedS)
    {
        var inv = CultureInfo.InvariantCulture;
        return "{\"id\": \"" + id + "\", \"x\": " + x.ToString("R", inv) + ", \"y\": " + y.ToString("R", inv) +
               ", \"volume_m3\": " + laidM3.ToString("F4", inv) + ", \"landed\": " + (landed ? "true" : "false") +
               ", \"waited_s\": " + waitedS.ToString("F1", inv) + "}";
    }

    private void OnFaultRequest(StringMsg msg)
    {
        Fault f;
        try { f = JsonUtility.FromJson<Fault>(msg.data); }
        catch (Exception e)
        {
            Debug.LogWarning($"[{nameof(CapScenarioOps)}] fault request unreadable: {e.Message}");
            return;
        }
        f.at_s = Time.time - t0;                                       // now
        StartCoroutine(FaultAt(f));
    }

    private IEnumerator Walk(Person p)
    {
        if (p.path == null || p.path.Length < 4) yield break;
        while (Time.time - t0 < p.start_s) yield return null;
        var body = GameObject.CreatePrimitive(PrimitiveType.Capsule);
        body.name = p.name;
        body.transform.localScale = new Vector3(0.5f, 0.85f, 0.5f);          // 1.7 m tall, 0.5 m across
        Destroy(body.GetComponent<Collider>());
        var kinematic = body.AddComponent<CapsuleCollider>();
        kinematic.isTrigger = true;                                            // seen by the LiDAR's rays, pushes nothing
        // how close it comes to each machine: its centre to the machine's colliders, across the ground [m]
        var closest = new Dictionary<string, (float d, float t)>();
        Debug.Log($"[{nameof(CapScenarioOps)}] {p.name} walks from ({p.path[0]:F1}, {p.path[1]:F1}) at {p.speed_mps:F1} m/s");
        for (int i = 0; i + 3 < p.path.Length; i += 2)
        {
            var a = new Vector3(-p.path[i + 1], 0f, p.path[i]);
            var b = new Vector3(-p.path[i + 3], 0f, p.path[i + 2]);
            float len = Vector3.Distance(a, b), walked = 0f;
            while (walked < len)
            {
                var at = Vector3.Lerp(a, b, walked / len);
                float g = Ground(at);
                at.y = (float.IsNaN(g) ? 0f : g) + 0.85f;
                body.transform.position = at;
                foreach (var m in GameObject.FindGameObjectsWithTag("robot"))
                {
                    float d = ClosestAcross(m, at);
                    if (!closest.TryGetValue(m.name, out var was) || d < was.d) closest[m.name] = (d, Time.time - t0);
                }
                walked += p.speed_mps * Time.deltaTime;
                yield return null;
            }
        }
        var sb = new StringBuilder();
        foreach (var kv in closest)
            sb.Append($" {kv.Key} {kv.Value.d.ToString("F2", CultureInfo.InvariantCulture)} m at t {kv.Value.t:F1}");
        Debug.Log($"[{nameof(CapScenarioOps)}] {p.name} walked its path; closest to{sb}");
        Destroy(body);
    }

    /// <summary>How far a point is from a machine across the ground [m]: to the nearest point of its colliders (convex
    /// shapes; its wheels' and triggers' left out), in the horizontal plane. Infinity without one.</summary>
    public static float ClosestAcross(GameObject machine, Vector3 at)
    {
        float d = float.PositiveInfinity;
        foreach (var c in machine.GetComponentsInChildren<Collider>())
        {
            if (c.isTrigger || !c.enabled || c is WheelCollider) continue;
            if (c is MeshCollider mc && !mc.convex) continue;              // ClosestPoint takes convex shapes only
            var q = c.ClosestPoint(new Vector3(at.x, Mathf.Clamp(at.y, c.bounds.min.y, c.bounds.max.y), at.z));
            d = Mathf.Min(d, Vector2.Distance(new Vector2(q.x, q.z), new Vector2(at.x, at.z)));
        }
        return d;
    }

    private IEnumerator FaultAt(Fault f)
    {
        while (Time.time - t0 < f.at_s) yield return null;
        var machine = f.kind == "person" ? null : GameObject.Find(f.machine);
        if (machine == null && f.kind != "person")
        {
            Debug.LogWarning($"[{nameof(CapScenarioOps)}] fault {f.kind}: no {f.machine}");
            yield break;
        }
        if (f.kind == "person")                                            // a person walks in: no machine of its own
        {
            Debug.Log($"[{nameof(CapScenarioOps)}] fault person: {f.target} walks {f.path?.Length / 2} points at {f.speed_mps} m/s");
            yield return Walk(new Person { name = string.IsNullOrEmpty(f.target) ? "person" : f.target, path = f.path,
                                           speed_mps = f.speed_mps > 0f ? f.speed_mps : 1.2f, start_s = 0f });
            yield break;
        }
        Debug.Log($"[{nameof(CapScenarioOps)}] fault {f.kind} on {f.machine} for {f.duration_s} s");
        if (f.kind == "blade_stall")
        {
            var drive = machine.GetComponent<CapBladeDrive>();
            if (drive != null) drive.enabled = false;                          // the blade holds where it is
            yield return new WaitForSeconds(f.duration_s);
            if (drive != null) drive.enabled = true;
        }
        else if (f.kind == "lidar_off")
        {
            Transform sensor = null;
            foreach (var t in machine.GetComponentsInChildren<Transform>(true))
                if (t.name == f.target) sensor = t;
            if (sensor == null) yield break;
            sensor.gameObject.SetActive(false);
            yield return new WaitForSeconds(f.duration_s);
            sensor.gameObject.SetActive(true);
        }
        Debug.Log($"[{nameof(CapScenarioOps)}] fault {f.kind} on {f.machine} over");
    }

    private IEnumerator RegionDumps()
    {
        var wait = new WaitForSeconds(config.region.period_s);
        while (true)
        {
            yield return wait;
            WriteRegion(config.region, $"{config.region.prefix}_{Time.time - t0:F0}");
        }
    }

    /// <summary>The region's heights [m, world] from the soil's working heightmap as float32 rows (ROS x rising, then ROS
    /// y rising) with a .json of its grid: what the scorer and the error-map frames read.</summary>
    public static void WriteRegion(Region r, string path)
    {
        var grid = CapTerrainDEMPublisher.Grid(r.x0, r.y0, r.x1, r.y1);
        if (grid == null) return;
        using (var w = new BinaryWriter(File.Open(path + ".f32", FileMode.Create)))
        {
            foreach (var h in grid.heights) w.Write(h);
        }
        File.WriteAllText(path + ".json", string.Format(CultureInfo.InvariantCulture,
            "{{\"x0\": {0}, \"y0\": {1}, \"cell_m\": {2}, \"nx\": {3}, \"ny\": {4}, \"file\": \"{5}.f32\", \"t\": {6}}}",
            grid.x0, grid.y0, grid.cell, grid.nx, grid.ny, Path.GetFileName(path), Time.time));
    }
}
