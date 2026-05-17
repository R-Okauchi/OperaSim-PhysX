using UnityEngine;
using UnityEditor;
using System.Collections.Generic;
using System.Linq;

/// <summary>
/// センサ配置最適化 Editor Window。
/// メニュー: CAP > Optimize Sensor Placement
///
/// シーン内のロボットに対して候補位置グリッドを生成し、
/// 各候補のカバレッジスコアを算出。Scene Viewにヒートマップを描画し、
/// 最適位置を提示する。
/// </summary>
public class SensorPlacementOptEditor : EditorWindow
{
    // ── State ─────────────────────────────────────────────────

    GameObject selectedRobot;
    bool isExcavator;
    string machineType = "";

    List<CandidateScore> results = new List<CandidateScore>();
    bool hasResults;
    Vector2 scrollPos;

    // Current position for comparison
    static readonly Vector3 CurrentLidarOffset = new Vector3(0f, 2.5f, 0f);

    struct CandidateScore
    {
        public Vector3 localPos;
        public float totalScore;
        public Dictionary<string, float> zoneCoverages;
    }

    // ── Menu ──────────────────────────────────────────────────

    [MenuItem("CAP/Optimize Sensor Placement")]
    static void OpenWindow()
    {
        var window = GetWindow<SensorPlacementOptEditor>("Sensor Placement Optimizer");
        window.minSize = new Vector2(420, 500);
        window.Show();
    }

    // ── GUI ───────────────────────────────────────────────────

    void OnGUI()
    {
        EditorGUILayout.Space(8);
        EditorGUILayout.LabelField("Sensor Placement Optimizer", EditorStyles.boldLabel);
        EditorGUILayout.HelpBox(
            "ロボットの自己遮蔽を考慮して、LiDARの最適マウント位置を計算します。\n" +
            "シーン内の 'robot' タグ付きGameObjectを選択してください。",
            MessageType.Info);

        EditorGUILayout.Space(4);

        // Robot selection
        selectedRobot = (GameObject)EditorGUILayout.ObjectField(
            "Target Robot", selectedRobot, typeof(GameObject), true);

        if (selectedRobot != null)
        {
            machineType = DetectMachineType(selectedRobot);
            isExcavator = machineType.Contains("zx");
            EditorGUILayout.LabelField("Detected Type",
                isExcavator ? $"Excavator ({machineType})" : $"Dump Truck ({machineType})");
        }

        EditorGUILayout.Space(8);

        EditorGUI.BeginDisabledGroup(selectedRobot == null);
        if (GUILayout.Button("Run Optimization", GUILayout.Height(32)))
        {
            RunOptimization();
        }
        EditorGUI.EndDisabledGroup();

        // Results
        if (hasResults && results.Count > 0)
        {
            EditorGUILayout.Space(12);
            DrawResults();
        }
    }

    // ── Optimization ─────────────────────────────────────────

    void RunOptimization()
    {
        results.Clear();
        hasResults = false;

        Transform robotRoot = selectedRobot.transform;

        // Determine mount link
        Transform mountLink;
        if (isExcavator)
        {
            mountLink = SensorCoverageEvaluator.FindChildRecursive(robotRoot, "body_link");
            if (mountLink == null)
                mountLink = SensorCoverageEvaluator.FindChildRecursive(robotRoot, "base_link");
        }
        else
        {
            mountLink = SensorCoverageEvaluator.FindChildRecursive(robotRoot, "base_link");
        }

        if (mountLink == null)
        {
            mountLink = robotRoot;
            Debug.LogWarning("[SensorOpt] mount link not found, using robot root.");
        }

        // Generate candidates and zones
        Vector3[] candidates = isExcavator
            ? SensorCoverageEvaluator.GenerateExcavatorCandidates()
            : SensorCoverageEvaluator.GenerateDumpTruckCandidates();

        SensorCoverageEvaluator.ZoneDef[] zones = isExcavator
            ? SensorCoverageEvaluator.GenerateExcavatorZones()
            : SensorCoverageEvaluator.GenerateDumpTruckZones();

        // Save original pose
        var savedPose = SensorCoverageEvaluator.SavePose(robotRoot);

        // Ensure physics colliders are synced
        Physics.SyncTransforms();

        int totalSteps = candidates.Length * (isExcavator
            ? SensorCoverageEvaluator.ExcavatorPhases.Length
            : SensorCoverageEvaluator.DumpTruckPhases.Length);
        int step = 0;

        foreach (var candidate in candidates)
        {
            var configCoverages = new List<Dictionary<string, float>>();

            if (isExcavator)
            {
                foreach (var phase in SensorCoverageEvaluator.ExcavatorPhases)
                {
                    step++;
                    EditorUtility.DisplayProgressBar(
                        "Optimizing Sensor Placement",
                        $"Candidate {results.Count + 1}/{candidates.Length} - Phase: {phase.name}",
                        (float)step / totalSteps);

                    // Pose the arm
                    SensorCoverageEvaluator.PoseExcavatorArm(robotRoot, phase);
                    Physics.SyncTransforms();

                    // Evaluate
                    Vector3 sensorWorld = mountLink.TransformPoint(candidate);
                    var cov = SensorCoverageEvaluator.EvaluateSingleConfig(
                        sensorWorld, mountLink, robotRoot, zones);
                    configCoverages.Add(cov);
                }
            }
            else
            {
                foreach (var phase in SensorCoverageEvaluator.DumpTruckPhases)
                {
                    step++;
                    EditorUtility.DisplayProgressBar(
                        "Optimizing Sensor Placement",
                        $"Candidate {results.Count + 1}/{candidates.Length} - Phase: {phase.name}",
                        (float)step / totalSteps);

                    SensorCoverageEvaluator.PoseDumpTruckVessel(robotRoot, phase);
                    Physics.SyncTransforms();

                    Vector3 sensorWorld = mountLink.TransformPoint(candidate);
                    var cov = SensorCoverageEvaluator.EvaluateSingleConfig(
                        sensorWorld, mountLink, robotRoot, zones);
                    configCoverages.Add(cov);
                }
            }

            // Worst-case across configs
            var minCov = SensorCoverageEvaluator.MinAcrossConfigs(configCoverages);
            float score = SensorCoverageEvaluator.ComputeWeightedScore(minCov, zones);

            results.Add(new CandidateScore
            {
                localPos = candidate,
                totalScore = score,
                zoneCoverages = minCov,
            });
        }

        EditorUtility.ClearProgressBar();

        // Restore original pose
        SensorCoverageEvaluator.RestorePose(robotRoot, savedPose);
        Physics.SyncTransforms();

        // Sort by score descending
        results.Sort((a, b) => b.totalScore.CompareTo(a.totalScore));
        hasResults = true;

        // Log summary
        if (results.Count > 0)
        {
            var best = results[0];
            Debug.Log($"[SensorOpt] Best position: ({best.localPos.x:F2}, {best.localPos.y:F2}, {best.localPos.z:F2}) " +
                      $"Score: {best.totalScore:F3}");

            // Evaluate current position for comparison
            float currentScore = EvaluateCurrentPosition(mountLink, robotRoot, zones);
            Debug.Log($"[SensorOpt] Current position (0, 2.5, 0) Score: {currentScore:F3}");
            Debug.Log($"[SensorOpt] Improvement: {((best.totalScore - currentScore) / Mathf.Max(currentScore, 0.001f) * 100):F1}%");
        }

        SceneView.RepaintAll();
        Repaint();
    }

    float EvaluateCurrentPosition(Transform mountLink, Transform robotRoot, SensorCoverageEvaluator.ZoneDef[] zones)
    {
        var savedPose = SensorCoverageEvaluator.SavePose(robotRoot);
        var configCoverages = new List<Dictionary<string, float>>();

        if (isExcavator)
        {
            foreach (var phase in SensorCoverageEvaluator.ExcavatorPhases)
            {
                SensorCoverageEvaluator.PoseExcavatorArm(robotRoot, phase);
                Physics.SyncTransforms();
                Vector3 sensorWorld = mountLink.TransformPoint(CurrentLidarOffset);
                configCoverages.Add(SensorCoverageEvaluator.EvaluateSingleConfig(
                    sensorWorld, mountLink, robotRoot, zones));
            }
        }
        else
        {
            foreach (var phase in SensorCoverageEvaluator.DumpTruckPhases)
            {
                SensorCoverageEvaluator.PoseDumpTruckVessel(robotRoot, phase);
                Physics.SyncTransforms();
                Vector3 sensorWorld = mountLink.TransformPoint(CurrentLidarOffset);
                configCoverages.Add(SensorCoverageEvaluator.EvaluateSingleConfig(
                    sensorWorld, mountLink, robotRoot, zones));
            }
        }

        SensorCoverageEvaluator.RestorePose(robotRoot, savedPose);
        Physics.SyncTransforms();

        var minCov = SensorCoverageEvaluator.MinAcrossConfigs(configCoverages);
        return SensorCoverageEvaluator.ComputeWeightedScore(minCov, zones);
    }

    // ── Results display ──────────────────────────────────────

    void DrawResults()
    {
        EditorGUILayout.LabelField("Results", EditorStyles.boldLabel);

        // Best result highlight
        var best = results[0];
        EditorGUILayout.BeginVertical(EditorStyles.helpBox);
        EditorGUILayout.LabelField("Best Position", EditorStyles.boldLabel);
        EditorGUILayout.LabelField($"  Local: ({best.localPos.x:F2}, {best.localPos.y:F2}, {best.localPos.z:F2})");
        EditorGUILayout.LabelField($"  Score: {best.totalScore:F3}");

        EditorGUILayout.Space(4);
        EditorGUILayout.LabelField("Zone Breakdown:", EditorStyles.miniLabel);
        foreach (var kvp in best.zoneCoverages.OrderByDescending(k => k.Value))
        {
            float pct = kvp.Value * 100f;
            Color barColor = pct > 80 ? Color.green : pct > 50 ? Color.yellow : Color.red;
            EditorGUILayout.BeginHorizontal();
            EditorGUILayout.LabelField($"  {kvp.Key}", GUILayout.Width(150));
            Rect r = EditorGUILayout.GetControlRect(GUILayout.Width(200));
            EditorGUI.DrawRect(new Rect(r.x, r.y, r.width * kvp.Value, r.height), barColor);
            EditorGUI.DrawRect(new Rect(r.x, r.y, r.width, r.height),
                new Color(0.3f, 0.3f, 0.3f, 0.3f));
            EditorGUILayout.LabelField($"{pct:F1}%", GUILayout.Width(50));
            EditorGUILayout.EndHorizontal();
        }
        EditorGUILayout.EndVertical();

        EditorGUILayout.Space(4);

        if (GUILayout.Button("Apply Best Position to SensorSetup", GUILayout.Height(28)))
        {
            ApplyBestPosition(best.localPos);
        }

        // Top 10 list
        EditorGUILayout.Space(8);
        EditorGUILayout.LabelField($"Top Candidates (showing 10 of {results.Count}):", EditorStyles.miniLabel);

        scrollPos = EditorGUILayout.BeginScrollView(scrollPos, GUILayout.Height(200));
        int showCount = Mathf.Min(10, results.Count);
        for (int i = 0; i < showCount; i++)
        {
            var r = results[i];
            string label = $"#{i + 1}  ({r.localPos.x:F2}, {r.localPos.y:F2}, {r.localPos.z:F2})  Score: {r.totalScore:F3}";
            EditorGUILayout.LabelField(label);
        }
        EditorGUILayout.EndScrollView();
    }

    void ApplyBestPosition(Vector3 pos)
    {
        if (selectedRobot == null) return;

        Transform baseLink = SensorCoverageEvaluator.FindChildRecursive(
            selectedRobot.transform, "base_link");
        if (baseLink == null) baseLink = selectedRobot.transform;

        Transform lidarMount = baseLink.Find("lidar_mount");
        if (lidarMount != null)
        {
            Undo.RecordObject(lidarMount, "Apply Optimal Sensor Position");
            lidarMount.localPosition = pos;
            EditorUtility.SetDirty(lidarMount.gameObject);
            Debug.Log($"[SensorOpt] Applied position ({pos.x:F2}, {pos.y:F2}, {pos.z:F2}) to {lidarMount.name}");
        }
        else
        {
            Debug.LogWarning("[SensorOpt] lidar_mount not found. Run 'CAP > Setup Sensors on All Robots' first.");
        }
    }

    // ── Scene View gizmo drawing ─────────────────────────────

    void OnEnable()
    {
        SceneView.duringSceneGui += OnSceneGUI;
    }

    void OnDisable()
    {
        SceneView.duringSceneGui -= OnSceneGUI;
    }

    void OnSceneGUI(SceneView sceneView)
    {
        if (!hasResults || results.Count == 0 || selectedRobot == null) return;

        Transform mountLink;
        if (isExcavator)
        {
            mountLink = SensorCoverageEvaluator.FindChildRecursive(
                selectedRobot.transform, "body_link");
            if (mountLink == null)
                mountLink = SensorCoverageEvaluator.FindChildRecursive(
                    selectedRobot.transform, "base_link");
        }
        else
        {
            mountLink = SensorCoverageEvaluator.FindChildRecursive(
                selectedRobot.transform, "base_link");
        }

        if (mountLink == null) return;

        float bestScore = results[0].totalScore;
        float worstScore = results[results.Count - 1].totalScore;
        float range = Mathf.Max(bestScore - worstScore, 0.001f);

        // Draw all candidates as colored spheres
        foreach (var r in results)
        {
            float t = (r.totalScore - worstScore) / range; // 0=worst, 1=best
            Color c = Color.Lerp(Color.red, Color.green, t);
            c.a = 0.4f + 0.6f * t; // more opaque for better candidates

            Vector3 worldPos = mountLink.TransformPoint(r.localPos);
            Handles.color = c;
            Handles.SphereHandleCap(0, worldPos, Quaternion.identity, 0.15f, EventType.Repaint);
        }

        // Highlight best with larger sphere + label
        var best = results[0];
        Vector3 bestWorld = mountLink.TransformPoint(best.localPos);
        Handles.color = new Color(0f, 1f, 0f, 0.8f);
        Handles.SphereHandleCap(0, bestWorld, Quaternion.identity, 0.3f, EventType.Repaint);

        GUIStyle labelStyle = new GUIStyle(EditorStyles.boldLabel);
        labelStyle.normal.textColor = Color.green;
        labelStyle.fontSize = 14;
        Handles.Label(bestWorld + Vector3.up * 0.4f,
            $"BEST: {best.totalScore:F3}\n({best.localPos.x:F2}, {best.localPos.y:F2}, {best.localPos.z:F2})",
            labelStyle);

        // Draw current position for comparison
        Vector3 currentWorld = mountLink.TransformPoint(CurrentLidarOffset);
        Handles.color = new Color(1f, 1f, 0f, 0.8f);
        Handles.SphereHandleCap(0, currentWorld, Quaternion.identity, 0.25f, EventType.Repaint);
        GUIStyle currentStyle = new GUIStyle(EditorStyles.boldLabel);
        currentStyle.normal.textColor = Color.yellow;
        currentStyle.fontSize = 12;
        Handles.Label(currentWorld + Vector3.up * 0.4f, "CURRENT", currentStyle);
    }

    // ── Helpers ───────────────────────────────────────────────

    static string DetectMachineType(GameObject robot)
    {
        string name = robot.name.ToLower();
        if (name.Contains("zx135")) return "zx135u";
        if (name.Contains("zx120")) return "zx120";
        if (name.Contains("zx200")) return "zx200";
        if (name.Contains("ic120")) return "ic120";
        if (name.Contains("c30r")) return "c30r";
        if (name.Contains("mst110")) return "mst110cr";

        // Check children for URDF-derived names
        if (SensorCoverageEvaluator.FindChildRecursive(robot.transform, "boom_link") != null)
            return "zx_unknown";

        return "unknown";
    }
}
