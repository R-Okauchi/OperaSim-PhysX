using System;
using System.Collections;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Threading;
using UnityEngine;
using UnityEngine.Experimental.Rendering;
using UnityEngine.Rendering;
using Unity.Robotics.ROSTCPConnector;

/// <summary>
/// 動画用の録画 (CAP の scripts/cycle_video.py): 固定カメラを何台か、一定の実時間間隔で JPEG の連番に書き出す。
///
/// 環境変数 CAP_UNITY_RECORD_CONFIG に設定の JSON のパスがあるときだけ、シーンの読み込み後に自分を作る
/// (シーンのファイルには触らない)。設定 (位置は ROS/world フレーム: cap の座標、Unity = (-ros.y, ros.z, ros.x)):
///   {"out_dir": "...", "fps": 15, "width": 1280, "height": 720, "quality": 90,
///    "views": [{"name": "overview", "pos": [x, y, z], "target": [x, y, z], "fov_deg": 50}, ...]}
/// 操作は out_dir/control.json を 0.5 s ごとに読む:
///   {"recording": true, "session": "cycle1"}  録画 (false で止める)
///   {"snapshot": "test1"}                     各カメラ 1 枚 (out_dir/snapshots/test1_<view>.jpg、構図の確認)
///   {"config_rev": 2}                          設定を読み直す (カメラの置き直し)
/// 機体の置き直し (任意): "placements": [{"name": "ic120", "x": -40, "y": 35, "set_yaw": false, "yaw_deg": 0}, ...]
///   シーンの読み込み直後 (物理の最初の 1 歩の前) に、その名前の機体を ROS/world の (x, y) へ (地面からの高さはそのまま、
///   set_yaw なら ROS の yaw_deg の向きへ)。使わない機体をフィールドの端へ寄せる・録画の配置を決めるため (2026-10-15)。
/// 地形の書き出し (任意): "terrain_dump": "/path/prefix" → prefix.json と prefix_&lt;i&gt;.f32 (各地形の高さ、行 = Unity z)。
/// ROS の接続先 (任意): "ros_ip": "100.106.45.66", "ros_port": 10000 → ROSConnectionPrefab の代わりにそこへ繋ぐ。
/// 土 (任意): "soil_particles": "off" → 粒子を出さない (SoilParticleSettings.particles = false: 掘った量はバケットと荷台が
///   持ち、地形は掘った所が下がり、こぼれた所・ダンプの後ろが上がる)。"on" で粒子、空ならシーンのまま。
///   "soil_selftest": true → 起動の 3 s 後に遠くで 20 回掘り 0.3 m3 置いて地形の体積で確かめる (CapSoilSelfTest、ログに
///   "[CapSoilSelfTest] PASS|FAIL")、"soil_selftest_quit": true でその後に終わる (Linux 版を GPU の node で試す)。
/// 表示 (任意): "visualizers": "off" → UnitySensors の点群の表示を切る (Linux 版はその shader を持たない)。
/// 速さ (任意): "pace_log_s": 10 → 10 s ごとに "[CapPace] sim +X s in Y s wall (factor F) ..." (CapPaceLog)。
/// 画面 (任意): "screen_cameras": "off" → 画面に描くだけのカメラを切る (センサと録画のカメラはそのまま)、
///   "frame_rate": 60 → フレームの上限 (描かない頭無しの実行は 590 fps まで回って CPU を食う。物理は 50 Hz のまま)。
/// こまで進める (任意): "frame_locked": true → Time.captureFramerate = fps。毎フレームを撮り、模擬の時計は 1 フレームで
///   1/fps 進む (描くのが遅くてもこまを落とさない: クラスタの CPU の描画で再生 (CapReplay) を撮る。読み出しが詰まれば待つ)。
/// 書き出し: out_dir/&lt;session&gt;/&lt;view&gt;/%06d.jpg と out_dir/&lt;session&gt;/frames.csv
/// (frame, wall_s, unity_time_s, realtime_s: 同じ番号は全カメラで同じ瞬間)、止めたら out_dir/&lt;session&gt;/done.json。
/// 描画は camera.Render → RenderTexture (Game ビューのフォーカスに依らない)、読み出しは AsyncGPUReadback、
/// JPEG 化と書き込みは別スレッド (シミュレーションのフレームを止めない)。書き込みが詰まればそのフレームは捨てて数える。
/// </summary>
public class CapCycleRecorder : MonoBehaviour
{
    [Serializable]
    private class ViewConfig
    {
        public string name;
        public float[] pos;
        public float[] target;
        public float fov_deg = 50f;
        public string follow;             // a machine's root: the camera rides with its base_link (pos and target unused)
        public float[] offset;            // ... at (behind, left, up) [m] in the machine's heading (yaw only)
        public float look_up_m = 1.0f;    // ... looking at its base_link this far up
    }

    [Serializable]
    private class Placement
    {
        public string name;               // シーンの機体のルート (zx200, mst110cr, ic120, c30r, zx120)
        public float x;                   // ROS/world [m]
        public float y;
        public bool set_yaw;              // true: yaw_deg (ROS、+x から反時計回り) へ向ける。false: 向きはそのまま
        public float yaw_deg;
    }

    [Serializable]
    private class RecorderConfig
    {
        public string out_dir;
        public float fps = 15f;
        public int width = 1280;
        public int height = 720;
        public int quality = 90;
        public string flip = "no";        // yes | no: 読み出した画像の上下を返すか (Metal の AsyncGPUReadback は
                                          // Texture2D と同じ下から上の行の順で、そのままで正しい向き: 2026-10-14)
        public ViewConfig[] views;
        public Placement[] placements;
        public string terrain_dump;
        public string ros_ip;             // 空でなければ ROS-TCP の接続先をこれに (ROSConnectionPrefab は変えない)
        public int ros_port;              // 0 なら prefab のまま
        public string soil_particles;     // off | on | 空 (シーンのまま)
        public bool soil_selftest;        // 起動の 3 s 後に土の自己試験 (CapSoilSelfTest: Linux 版の GPU で地形が掘れるか)
        public bool soil_selftest_quit;   // その後に終わる (合格 0、不合格 3)
        public string visualizers;        // off: UnitySensors の点群の表示を切る (Linux 版はその shader を持たない)
        public float pace_log_s;          // >0: その秒ごとに模擬の速さを記録 (CapPaceLog)
        public string screen_cameras;     // off: 画面に描くだけのカメラ (targetTexture の無いもの) を切る (頭無しの実行)
        public int frame_rate;            // >0: Application.targetFrameRate (頭無しの描かない実行は 600 fps 近くまで回る)
        public bool frame_locked;         // true: Time.captureFramerate = fps, every frame captured, none dropped (offline)
        public CapScenarioOps.Config scenario;   // a scenario's machines, ground, piles, trucks, people, faults (CapScenarioOps)
        public CapReplay.Config replay;   // a run played back for its video (CapReplay): its pack, speed, span, machine
        public DemGrid dem_grid;          // the survey on /terrain/dem_grid (CapTerrainDEMPublisher): its region and period
    }

    [Serializable]
    private class DemGrid
    {
        public float x0, y0, x1, y1;      // ROS rectangle
        public float period_s = 3f;
    }

    [Serializable]
    private class Control
    {
        public bool recording;
        public string session;
        public string snapshot;
        public int config_rev;
        public string terrain_dump;       // a name: the scenario region's heights now, to <out_dir>/terrain/<name> (the score)
    }

    private class View
    {
        public string name;
        public Camera cam;
        public RenderTexture rt;
        public ViewConfig cfg;
        public Transform follow;          // the followed machine's base_link (found on the first capture)
    }

    private struct Job
    {
        public byte[] pixels;
        public string path;
        public int width, height, quality;
        public bool flip;
    }

    public string configPath;

    private RecorderConfig _config;
    private readonly List<View> _views = new List<View>();
    private string _session;
    private string _sessionDir;
    private StreamWriter _csv;
    private int _frame;
    private int _dropped;
    private int _written;
    private string _lastSnapshot = "";
    private string _lastTerrainDump = "";
    private int _configRev;
    private bool _flip;
    private BlockingCollection<Job> _queue;
    private ConcurrentBag<byte[]> _pool;
    private Thread _writer;
    private int _pending;
    private const int MaxPending = 24;

    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
    private static void Boot()
    {
        string path = Environment.GetEnvironmentVariable("CAP_UNITY_RECORD_CONFIG");
        if (string.IsNullOrEmpty(path) || !File.Exists(path))
        {
            return;
        }
        if (Array.IndexOf(Environment.GetCommandLineArgs(), "-runTests") >= 0)
        {
            // The PlayMode tests' scenes are theirs (CAP review a846 #9: a shell with the run's config set placed the
            // machines and dug the self-test's pit into the tests' ground)
            Debug.Log($"[{nameof(CapCycleRecorder)}] -runTests: {path} not applied");
            return;
        }
        RecorderConfig cfg;
        try
        {
            cfg = JsonUtility.FromJson<RecorderConfig>(File.ReadAllText(path));
        }
        catch (Exception e)
        {
            Debug.LogWarning($"[{nameof(CapCycleRecorder)}] {path} unreadable: {e.Message}");
            return;
        }
        // シーンの読み込み直後 = 物理の最初の 1 歩の前: 置き直しも地形もここで (センサが最初の値を出す前)。段ごとに試し、
        // 1 つが投げても他は効く (a846 #9: 置き直しが投げると粒子のまま走り、地形の書き出しが投げると残りが黙って飛んだ)。
        // 土の粒子が先頭: どの道を走るかを決める
        Stage("soil particles", () => SoilParticles(cfg.soil_particles));
        // the scenario's ground before its machines are placed on it (a bench under a loader: 11-01), and before the
        // soil starts (its working heightmap is built from these heights); the ground's dump below is then the ground
        // before any pile
        Stage("terrain", () => CapScenarioOps.ShapeTerrain(cfg.scenario));
        Stage("placements", () => PlaceMachines(cfg.placements));
        // a scenario's machines (spawned, cloned, switched off) on that ground
        Stage("scenario", () => CapScenarioOps.Prepare(cfg.scenario));
        Stage("ROS endpoint", () => PointRos(cfg.ros_ip, cfg.ros_port));
        Stage("terrain dump", () => DumpTerrain(cfg.terrain_dump));
        if (cfg.visualizers == "off")
        {
            Stage("visualizers", SilenceVisualizers);
        }
        if (cfg.screen_cameras == "off")
        {
            Stage("screen cameras", ScreenCamerasOff);
        }
        if (cfg.frame_rate > 0)
        {
            QualitySettings.vSyncCount = 0;
            Application.targetFrameRate = cfg.frame_rate;
            Debug.Log($"[{nameof(CapCycleRecorder)}] frame rate {cfg.frame_rate}");
        }
        if (cfg.frame_locked)
        {
            Time.captureFramerate = Mathf.Max(1, Mathf.RoundToInt(cfg.fps));
            Debug.Log($"[{nameof(CapCycleRecorder)}] frame-locked: {Time.captureFramerate} frames a second of the game's time");
        }
        if (cfg.pace_log_s > 0f)
        {
            Stage("pace log", () =>
            {
                var pace = new GameObject(nameof(CapPaceLog));
                DontDestroyOnLoad(pace);
                pace.AddComponent<CapPaceLog>().period = cfg.pace_log_s;
            });
        }
        if (cfg.soil_selftest)
        {
            Stage("soil self-test", () =>
            {
                var check = new GameObject(nameof(CapSoilSelfTest));
                DontDestroyOnLoad(check);
                check.AddComponent<CapSoilSelfTest>().quitAfter = cfg.soil_selftest_quit;
            });
        }
        if (cfg.scenario != null)
        {
            Stage("scenario ops", () =>
            {
                var ops = new GameObject(nameof(CapScenarioOps));
                DontDestroyOnLoad(ops);
                var scenario = ops.AddComponent<CapScenarioOps>();
                scenario.config = cfg.scenario;
                scenario.outDir = cfg.out_dir;
            });
        }
        if (cfg.dem_grid != null && cfg.dem_grid.x1 > cfg.dem_grid.x0 && cfg.dem_grid.y1 > cfg.dem_grid.y0)
        {
            Stage("survey", () =>
            {
                var dem = new GameObject(nameof(CapTerrainDEMPublisher)).AddComponent<CapTerrainDEMPublisher>();
                DontDestroyOnLoad(dem.gameObject);
                dem.x0 = cfg.dem_grid.x0;
                dem.y0 = cfg.dem_grid.y0;
                dem.x1 = cfg.dem_grid.x1;
                dem.y1 = cfg.dem_grid.y1;
                dem.periodS = cfg.dem_grid.period_s;
            });
        }
        if (cfg.replay != null && !string.IsNullOrEmpty(cfg.replay.pack))
        {
            Stage("replay", () =>
            {
                var play = new GameObject(nameof(CapReplay));
                DontDestroyOnLoad(play);
                var replay = play.AddComponent<CapReplay>();
                replay.config = cfg.replay;
                if (string.IsNullOrEmpty(replay.config.control))
                    replay.config.control = Path.Combine(cfg.out_dir, "control.json");
            });
        }
        var go = new GameObject(nameof(CapCycleRecorder));
        DontDestroyOnLoad(go);
        go.AddComponent<CapCycleRecorder>().configPath = path;
    }

    // ROS-TCP の接続先: ROSConnection は Start で繋ぐので、シーンの読み込み直後 (Start の前) に変えればそこへ繋ぐ。
    // 既に繋いでいれば繋ぎ直す。録画とは別の用途 (機体の PC 役のホストの中継へ、2026-10-15 の本番どおりの通し)。
    private static void PointRos(string ip, int port)
    {
        if (string.IsNullOrEmpty(ip))
        {
            return;
        }
        var ros = ROSConnection.GetOrCreateInstance();
        bool connected = ros.HasConnectionThread;
        if (connected)
        {
            ros.Disconnect();
        }
        ros.RosIPAddress = ip;
        if (port > 0)
        {
            ros.RosPort = port;
        }
        if (connected)
        {
            ros.Connect();
        }
        Debug.Log($"[{nameof(CapCycleRecorder)}] ROS-TCP -> {ros.RosIPAddress}:{ros.RosPort}" + (connected ? " (reconnected)" : ""));
    }

    // UnitySensors の点群の表示 (UnitySensors.Visualization の振る舞い、その shader の描画): 表示だけのもので、Linux 版は
    // shader を持たない (毎フレーム NullReferenceException)、描くのも時間を食う。Start の前に切れば購読もしない
    // (エディタの通しは CapKit.Editor.CapKitPlay が同じことをする)。
    private static void SilenceVisualizers()
    {
        int behaviours = 0, renderers = 0;
        foreach (var behaviour in Resources.FindObjectsOfTypeAll<Behaviour>())
        {
            if (behaviour == null || behaviour.gameObject.scene.rootCount == 0) continue;
            string ns = behaviour.GetType().Namespace;
            if (ns != null && ns.StartsWith("UnitySensors.Visualization", StringComparison.Ordinal) && behaviour.enabled)
            {
                behaviour.enabled = false;
                behaviours++;
            }
        }
        foreach (var renderer in Resources.FindObjectsOfTypeAll<Renderer>())
        {
            if (renderer == null || renderer.gameObject.scene.rootCount == 0 || !renderer.enabled) continue;
            var material = renderer.sharedMaterial;
            if (material != null && material.shader != null && material.shader.name.StartsWith("UnitySensors/", StringComparison.Ordinal))
            {
                renderer.enabled = false;
                renderers++;
            }
        }
        Debug.Log($"[{nameof(CapCycleRecorder)}] visualizers off: {behaviours} behaviours, {renderers} renderers");
    }

    // 画面に描くだけのカメラ (targetTexture の無いもの: Main Camera など): 頭無しの実行 (Linux 版をソフトの OpenGL で) では
    // 誰も見ない画面に毎フレーム描いて模擬を遅らせる。センサのカメラと録画のカメラ (RenderTexture に描く) はそのまま。
    private static void Stage(string what, Action act)
    {
        try
        {
            act();
        }
        catch (Exception e)
        {
            Debug.LogWarning($"[{nameof(CapCycleRecorder)}] {what}: {e.Message}");
        }
    }

    // The scene's Main Camera, kept before the screen's cameras are switched off: the recording's cameras copy its sky
    // and rendering (Camera.main is none once it is off)
    private static Camera s_mainCamera;

    private static void ScreenCamerasOff()
    {
        int n = 0;
        s_mainCamera = Camera.main;
        foreach (var cam in Resources.FindObjectsOfTypeAll<Camera>())
        {
            // A camera of an object not active (yet) is left as it is: a sensor's gets its target texture in its
            // Awake, which has not run (a846 #10)
            if (cam == null || cam.gameObject.scene.rootCount == 0 || !cam.gameObject.activeInHierarchy || !cam.enabled
                || cam.targetTexture != null) continue;
            cam.enabled = false;
            n++;
        }
        Debug.Log($"[{nameof(CapCycleRecorder)}] screen cameras off: {n}");
    }

    // 土の粒子: SoilParticleSettings は Awake で自分を instance にし、Start で地形と荷台を用意するので、その間 (ここ) で決める
    private static void SoilParticles(string mode)
    {
        if (string.IsNullOrEmpty(mode))
        {
            return;
        }
        bool on = mode == "on" || mode == "true" || mode == "yes";
        bool off = mode == "off" || mode == "false" || mode == "no";
        var soil = SoilParticleSettings.instance;
        if (soil == null || !(on || off))
        {
            Debug.LogWarning($"[{nameof(CapCycleRecorder)}] soil_particles \"{mode}\": " +
                             (soil == null ? "no SoilParticleSettings in the scene" : "neither on nor off"));
            return;
        }
        soil.particles = on;
        Debug.Log($"[{nameof(CapCycleRecorder)}] soil particles {(on ? "on" : "off")}");
    }

    private static float GroundAt(Vector3 p)
    {
        foreach (var t in Terrain.activeTerrains)
        {
            Vector3 o = t.transform.position, size = t.terrainData.size;
            if (p.x >= o.x && p.x <= o.x + size.x && p.z >= o.z && p.z <= o.z + size.z)
            {
                return t.SampleHeight(p) + o.y;
            }
        }
        return float.NaN;
    }

    private static void PlaceMachines(Placement[] placements)
    {
        if (placements == null)
        {
            return;
        }
        foreach (var p in placements)
        {
            GameObject go = GameObject.Find(p.name);
            if (go == null)
            {
                Debug.LogWarning($"[{nameof(CapCycleRecorder)}] placement: no object named {p.name}");
                continue;
            }
            Vector3 from = go.transform.position;
            var to = new Vector3(-p.y, from.y, p.x);
            float g0 = GroundAt(from), g1 = GroundAt(to);
            if (!float.IsNaN(g0) && !float.IsNaN(g1))
            {
                to.y = from.y + (g1 - g0);    // 地面からの高さはそのまま
            }
            Quaternion turn = p.set_yaw ? Quaternion.Euler(0f, -p.yaw_deg - go.transform.eulerAngles.y, 0f)
                                        : Quaternion.identity;
            go.transform.SetPositionAndRotation(to, turn * go.transform.rotation);
            foreach (var ab in go.GetComponentsInChildren<ArticulationBody>())
            {
                if (ab.isRoot)
                {
                    ab.TeleportRoot(ab.transform.position, ab.transform.rotation);   // 物理の体もそこへ
                    ab.velocity = Vector3.zero;
                    ab.angularVelocity = Vector3.zero;
                }
            }
            foreach (var rb in go.GetComponentsInChildren<Rigidbody>())
            {
                rb.position = rb.transform.position;
                rb.rotation = rb.transform.rotation;
                if (!rb.isKinematic)
                {
                    rb.velocity = Vector3.zero;
                    rb.angularVelocity = Vector3.zero;
                }
            }
            Debug.Log($"[{nameof(CapCycleRecorder)}] placed {p.name} at ({p.x:F1}, {p.y:F1})" +
                      (p.set_yaw ? $" yaw {p.yaw_deg:F0} deg" : "") + $" (Unity {to})");
        }
    }

    private static void DumpTerrain(string prefix)
    {
        if (string.IsNullOrEmpty(prefix))
        {
            return;
        }
        var parts = new List<string>();
        var terrains = Terrain.activeTerrains;
        for (int i = 0; i < terrains.Length; i++)
        {
            TerrainData d = terrains[i].terrainData;
            int res = d.heightmapResolution;
            float[,] h = d.GetHeights(0, 0, res, res);
            string file = $"{prefix}_{i}.f32";
            using (var w = new BinaryWriter(File.Open(file, FileMode.Create)))
            {
                for (int r = 0; r < res; r++)
                {
                    for (int c = 0; c < res; c++)
                    {
                        w.Write(h[r, c]);
                    }
                }
            }
            Vector3 o = terrains[i].transform.position, size = d.size;
            parts.Add(string.Format(CultureInfo.InvariantCulture,
                "{{\"file\": \"{0}\", \"resolution\": {1}, \"position\": [{2}, {3}, {4}], \"size\": [{5}, {6}, {7}]}}",
                file, res, o.x, o.y, o.z, size.x, size.y, size.z));
        }
        File.WriteAllText(prefix + ".json", "{\"terrains\": [" + string.Join(", ", parts) + "]}");
        Debug.Log($"[{nameof(CapCycleRecorder)}] terrain: {terrains.Length} -> {prefix}.json");
    }

    private void Start()
    {
        _queue = new BlockingCollection<Job>(new ConcurrentQueue<Job>());
        _pool = new ConcurrentBag<byte[]>();
        _writer = new Thread(WriteLoop) { IsBackground = true, Name = "CapCycleRecorder" };
        _writer.Start();
        LoadConfig();
        StartCoroutine(Loop());
    }

    private void LoadConfig()
    {
        _config = JsonUtility.FromJson<RecorderConfig>(File.ReadAllText(configPath));
        _flip = _config.flip == "yes";
        foreach (var v in _views)
        {
            if (v.rt != null) { v.rt.Release(); Destroy(v.rt); }
            if (v.cam != null) { Destroy(v.cam.gameObject); }
        }
        _views.Clear();
        Camera main = Camera.main != null ? Camera.main : s_mainCamera;
        foreach (var vc in _config.views ?? new ViewConfig[0])
        {
            var go = new GameObject("CapRec_" + vc.name);
            go.transform.SetParent(transform, false);
            var cam = go.AddComponent<Camera>();
            if (main != null)
            {
                cam.CopyFrom(main);           // 空・描画の設定は Main Camera のもの
            }
            var rt = new RenderTexture(_config.width, _config.height, 24, RenderTextureFormat.ARGB32);
            rt.Create();
            cam.enabled = false;              // 描くのはこちらから (Render)
            cam.targetTexture = rt;
            cam.aspect = _config.width / (float)_config.height;
            cam.fieldOfView = vc.fov_deg;     // 垂直 FOV (fusion_viz も)
            // a following view's place is set each capture (Follow): here only a start
            float[] ap = vc.pos != null && vc.pos.Length == 3 ? vc.pos : new[] { 0f, 0f, 10f };
            float[] at = vc.target != null && vc.target.Length == 3 ? vc.target : new[] { 1f, 0f, 0f };
            Vector3 pos = new Vector3(-ap[1], ap[2], ap[0]);
            Vector3 target = new Vector3(-at[1], at[2], at[0]);
            Vector3 forward = target - pos;
            go.transform.position = pos;
            go.transform.rotation = Quaternion.LookRotation(forward, Vector3.up);
            float dist = forward.magnitude;
            cam.nearClipPlane = Mathf.Max(0.1f, dist * 0.01f);
            cam.farClipPlane = Mathf.Max(1000.0f, dist * 4.0f);
            if (!string.IsNullOrEmpty(vc.follow))
            {
                cam.nearClipPlane = 0.1f;
                cam.farClipPlane = 1000f;
            }
            _views.Add(new View { name = vc.name, cam = cam, rt = rt, cfg = vc });
        }
        Directory.CreateDirectory(_config.out_dir);
        Debug.Log($"[{nameof(CapCycleRecorder)}] {_views.Count} views at {_config.width}x{_config.height} {_config.fps} fps, " +
                  $"flip={_flip} -> {_config.out_dir}");
    }

    private IEnumerator Loop()
    {
        double period = 1.0 / Mathf.Max(1f, _config.fps);
        double next = Time.realtimeSinceStartupAsDouble;
        double nextPoll = 0.0;
        while (true)
        {
            double now = Time.realtimeSinceStartupAsDouble;
            if (now >= nextPoll)
            {
                nextPoll = now + 0.5;
                PollControl();
                period = 1.0 / Mathf.Max(1f, _config.fps);
            }
            if (_session != null && _config.frame_locked)
            {
                CaptureTick();                // こまで進める: 毎フレーム (模擬の時計はフレームごとに 1/fps)
            }
            else if (_session != null && now >= next)
            {
                CaptureTick();
                next += period;
                if (next < now)
                {
                    next = now + period;      // 遅れたら追いつこうとしない (詰めて撮らない)
                }
            }
            yield return null;
        }
    }

    private void PollControl()
    {
        string path = Path.Combine(_config.out_dir, "control.json");
        if (!File.Exists(path))
        {
            return;
        }
        Control c;
        try
        {
            c = JsonUtility.FromJson<Control>(File.ReadAllText(path));
        }
        catch (Exception)
        {
            return;                           // 書きかけ: 次に読む
        }
        if (c == null)
        {
            return;
        }
        if (c.config_rev != _configRev)
        {
            _configRev = c.config_rev;
            LoadConfig();
        }
        if (!string.IsNullOrEmpty(c.terrain_dump) && c.terrain_dump != _lastTerrainDump)
        {
            _lastTerrainDump = c.terrain_dump;
            var region = _config.scenario?.region;
            if (region != null)
            {
                string dir = Path.Combine(_config.out_dir, "terrain");
                Directory.CreateDirectory(dir);
                CapScenarioOps.WriteRegion(region, Path.Combine(dir, c.terrain_dump));
                Debug.Log($"[{nameof(CapCycleRecorder)}] terrain {c.terrain_dump} -> {dir}");
            }
        }
        if (!string.IsNullOrEmpty(c.snapshot) && c.snapshot != _lastSnapshot)
        {
            _lastSnapshot = c.snapshot;
            string dir = Path.Combine(_config.out_dir, "snapshots");
            Directory.CreateDirectory(dir);
            foreach (var v in _views)
            {
                Capture(v, Path.Combine(dir, $"{c.snapshot}_{v.name}.jpg"));
            }
        }
        if (c.recording && !string.IsNullOrEmpty(c.session) && c.session != _session)
        {
            StopSession();
            StartSession(c.session);
        }
        else if (!c.recording && _session != null)
        {
            StopSession();
        }
    }

    private void StartSession(string session)
    {
        _session = session;
        _sessionDir = Path.Combine(_config.out_dir, session);
        foreach (var v in _views)
        {
            Directory.CreateDirectory(Path.Combine(_sessionDir, v.name));
        }
        _csv = new StreamWriter(Path.Combine(_sessionDir, "frames.csv"), false);
        _csv.WriteLine("frame,wall_s,unity_time_s,realtime_s");
        _frame = 0;
        _dropped = 0;
        _written = 0;
        Debug.Log($"[{nameof(CapCycleRecorder)}] recording {session} -> {_sessionDir}");
    }

    private void StopSession()
    {
        if (_session == null)
        {
            return;
        }
        _csv.Flush();
        _csv.Dispose();
        _csv = null;
        string done = Path.Combine(_sessionDir, "done.json");
        int frames = _frame;
        int dropped = _dropped;
        string session = _session;
        // 書き込みの済んだ後に印を置く (動画にするのはそれから)
        new Thread(() =>
        {
            var until = DateTime.UtcNow.AddSeconds(60);
            while (Interlocked.CompareExchange(ref _pending, 0, 0) > 0 && DateTime.UtcNow < until)
            {
                Thread.Sleep(100);
            }
            File.WriteAllText(done, $"{{\"session\": \"{session}\", \"frames\": {frames}, \"dropped\": {dropped}, " +
                                    $"\"written\": {Interlocked.CompareExchange(ref _written, 0, 0)}}}");
        }) { IsBackground = true }.Start();
        Debug.Log($"[{nameof(CapCycleRecorder)}] stopped {session}: {frames} frames, {dropped} dropped");
        _session = null;
    }

    private void CaptureTick()
    {
        double wall = (DateTime.UtcNow - new DateTime(1970, 1, 1, 0, 0, 0, DateTimeKind.Utc)).TotalSeconds;
        int frame = _frame++;
        _csv.WriteLine(string.Format(CultureInfo.InvariantCulture, "{0},{1:F4},{2:F4},{3:F4}", frame, wall, Time.time,
                                     Time.realtimeSinceStartupAsDouble));
        if (frame % 30 == 0)
        {
            _csv.Flush();
        }
        foreach (var v in _views)
        {
            if (!string.IsNullOrEmpty(v.cfg?.follow)) Follow(v);
            Capture(v, Path.Combine(_sessionDir, v.name, frame.ToString("D6") + ".jpg"));
        }
    }

    /// <summary>A following view: behind, left, up of the machine's base_link in its heading (its yaw only: the camera
    /// does not roll and pitch with the tracks), looking at it.</summary>
    private static void Follow(View v)
    {
        if (v.follow == null)
        {
            var root = GameObject.Find(v.cfg.follow);
            if (root == null) return;
            foreach (var t in root.GetComponentsInChildren<Transform>(true))
                if (t.name == "base_link") v.follow = t;
            if (v.follow == null) v.follow = root.transform;
        }
        var o = v.cfg.offset != null && v.cfg.offset.Length == 3 ? v.cfg.offset : new[] { 12f, 6f, 7f };
        Vector3 fwd = v.follow.forward;
        fwd.y = 0f;
        fwd = fwd.sqrMagnitude > 1e-6f ? fwd.normalized : Vector3.forward;
        // ROS left (+y) is Unity's -x of the heading's right
        Vector3 right = Vector3.Cross(Vector3.up, fwd);
        Vector3 at = v.follow.position + Vector3.up * v.cfg.look_up_m;
        Vector3 pos = v.follow.position - fwd * o[0] - right * o[1] + Vector3.up * o[2];
        v.cam.transform.SetPositionAndRotation(pos, Quaternion.LookRotation(at - pos, Vector3.up));
    }

    private void Capture(View v, string path)
    {
        if (_config.frame_locked)
        {
            while (Interlocked.CompareExchange(ref _pending, 0, 0) >= MaxPending)
            {
                AsyncGPUReadback.WaitAllRequests();   // こまで進める: 落とさず、読み出しと書き込みが空くのを待つ
                Thread.Sleep(2);
            }
        }
        else if (Interlocked.CompareExchange(ref _pending, 0, 0) >= MaxPending)
        {
            _dropped++;
            return;
        }
        v.cam.Render();
        Interlocked.Increment(ref _pending);
        int w = _config.width, h = _config.height, quality = _config.quality;
        bool flip = _flip;
        AsyncGPUReadback.Request(v.rt, 0, TextureFormat.RGBA32, req =>
        {
            if (req.hasError)
            {
                Interlocked.Decrement(ref _pending);
                _dropped++;
                return;
            }
            var data = req.GetData<byte>();
            if (!_pool.TryTake(out byte[] buffer) || buffer.Length != data.Length)
            {
                buffer = new byte[data.Length];
            }
            data.CopyTo(buffer);
            _queue.Add(new Job { pixels = buffer, path = path, width = w, height = h, quality = quality, flip = flip });
        });
    }

    private void WriteLoop()
    {
        foreach (var job in _queue.GetConsumingEnumerable())
        {
            try
            {
                int w = job.width, h = job.height;
                if (job.flip)
                {
                    FlipRows(job.pixels, w, h);
                }
                byte[] jpg = ImageConversion.EncodeArrayToJPG(job.pixels, GraphicsFormat.R8G8B8A8_SRGB, (uint)w,
                                                              (uint)h, 0, job.quality);
                File.WriteAllBytes(job.path, jpg);
                Interlocked.Increment(ref _written);
            }
            catch (Exception e)
            {
                Debug.LogWarning($"[{nameof(CapCycleRecorder)}] {job.path}: {e.Message}");
            }
            finally
            {
                _pool.Add(job.pixels);
                Interlocked.Decrement(ref _pending);
            }
        }
    }

    private static void FlipRows(byte[] pixels, int w, int h)
    {
        int stride = w * 4;
        var row = new byte[stride];
        for (int y = 0; y < h / 2; y++)
        {
            int a = y * stride, b = (h - 1 - y) * stride;
            Buffer.BlockCopy(pixels, a, row, 0, stride);
            Buffer.BlockCopy(pixels, b, pixels, a, stride);
            Buffer.BlockCopy(row, 0, pixels, b, stride);
        }
    }

    private void OnDestroy()
    {
        StopSession();
        _queue?.CompleteAdding();
        foreach (var v in _views)
        {
            if (v.rt != null) { v.rt.Release(); }
        }
    }
}
