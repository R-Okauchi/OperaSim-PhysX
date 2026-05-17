using UnityEngine;
using System.Collections.Generic;

/// <summary>
/// LiDARセンサの配置候補に対してカバレッジ（視認率）を評価する。
/// ロボット自身のColliderによる自己遮蔽を検出し、クリティカルゾーンごとの
/// カバレッジ率を返す。Editor / Runtime 両方から利用可能。
/// </summary>
public static class SensorCoverageEvaluator
{
    // ── Zone definitions ──────────────────────────────────────

    /// <summary>クリティカルゾーン1件の定義</summary>
    public struct ZoneDef
    {
        public string name;
        public float weight;
        public Vector3[] samplePoints; // mount link ローカル座標
    }

    /// <summary>評価結果</summary>
    public struct CoverageResult
    {
        public Vector3 candidateLocalPos;
        public Dictionary<string, float> zoneCoverages; // zone name → 0..1
        public float totalScore; // weighted sum
    }

    // ── Excavator arm phases (mirrors specs.py _ZX120_ARM_PHASES) ────

    public struct ArmPhase
    {
        public string name;
        public float turntableDeg;
        public float boomDeg;
        public float armDeg;
        public float bucketDeg;
    }

    public static readonly ArmPhase[] ExcavatorPhases = new ArmPhase[]
    {
        new ArmPhase { name = "idle",              turntableDeg = 0,   boomDeg = -46, armDeg = 114, bucketDeg = 46 },
        new ArmPhase { name = "moving_to_dig",     turntableDeg = 0,   boomDeg = -46, armDeg = 114, bucketDeg = 46 },
        new ArmPhase { name = "digging",           turntableDeg = 0,   boomDeg = -10, armDeg = 60,  bucketDeg = 30 },
        new ArmPhase { name = "swinging_to_load",  turntableDeg = 90,  boomDeg = 10,  armDeg = 45,  bucketDeg = 20 },
        new ArmPhase { name = "loading_truck",     turntableDeg = 90,  boomDeg = 15,  armDeg = 35,  bucketDeg = -15 },
        new ArmPhase { name = "waiting_for_truck",  turntableDeg = 90,  boomDeg = 10,  armDeg = 40,  bucketDeg = 15 },
    };

    // Dump truck vessel phases
    public struct VesselPhase
    {
        public string name;
        public float vesselDeg;
    }

    public static readonly VesselPhase[] DumpTruckPhases = new VesselPhase[]
    {
        new VesselPhase { name = "normal",  vesselDeg = 0 },
        new VesselPhase { name = "dumping", vesselDeg = -58 },
    };

    // ── Zone sample generation ───────────────────────────────

    /// <summary>ショベル用クリティカルゾーンを生成</summary>
    public static ZoneDef[] GenerateExcavatorZones()
    {
        return new ZoneDef[]
        {
            GenerateSwingZone(),
            GenerateDigZone(),
            GenerateBucketWorkspace(),
            GenerateTruckDetectionZone(),
            GenerateTerrainAheadZone(),
            GenerateSurroundZone(5f, 30f),
        };
    }

    /// <summary>ダンプ用クリティカルゾーンを生成</summary>
    public static ZoneDef[] GenerateDumpTruckZones()
    {
        return new ZoneDef[]
        {
            GenerateForwardPathZone(),
            GenerateSideClearanceZone(),
            GenerateDumpAreaZone(),
            GenerateSurroundZone(5f, 20f),
        };
    }

    static ZoneDef GenerateSwingZone()
    {
        // 360° annular, r=2..10m, h=-1..3m
        var pts = new List<Vector3>();
        int nAngles = 36, nRadii = 5, nHeights = 3;
        float[] radii = { 2f, 4f, 6f, 8f, 10f };
        float[] heights = { -1f, 1f, 3f };

        for (int a = 0; a < nAngles; a++)
        {
            float angle = (360f / nAngles) * a * Mathf.Deg2Rad;
            for (int r = 0; r < nRadii; r++)
            {
                for (int hi = 0; hi < nHeights; hi++)
                {
                    float x = radii[r] * Mathf.Sin(angle);
                    float y = heights[hi];
                    float z = radii[r] * Mathf.Cos(angle);
                    pts.Add(new Vector3(x, y, z));
                }
            }
        }

        return new ZoneDef { name = "swing_zone", weight = 0.35f, samplePoints = pts.ToArray() };
    }

    static ZoneDef GenerateDigZone()
    {
        // Forward 2-5m, ±1m lateral, ground level (y ~ -2.5m from sensor height)
        var pts = new List<Vector3>();
        for (float fwd = 2f; fwd <= 5f; fwd += 0.5f)
            for (float lat = -1f; lat <= 1f; lat += 0.5f)
                for (float h = -2.5f; h <= -1.5f; h += 0.5f)
                    pts.Add(new Vector3(lat, h, fwd));

        return new ZoneDef { name = "dig_point", weight = 0.20f, samplePoints = pts.ToArray() };
    }

    static ZoneDef GenerateBucketWorkspace()
    {
        // Forward hemisphere, r=2..10m, elevation -30..+45 deg
        var pts = new List<Vector3>();
        int nAz = 12, nEl = 4, nR = 4;
        float[] radii = { 3f, 5f, 7f, 10f };

        for (int az = 0; az < nAz; az++)
        {
            float azAngle = (-90f + (180f / nAz) * az) * Mathf.Deg2Rad;
            for (int el = 0; el < nEl; el++)
            {
                float elAngle = (-30f + (75f / nEl) * el) * Mathf.Deg2Rad;
                for (int ri = 0; ri < nR; ri++)
                {
                    float r = radii[ri];
                    float x = r * Mathf.Cos(elAngle) * Mathf.Sin(azAngle);
                    float y = r * Mathf.Sin(elAngle);
                    float z = r * Mathf.Cos(elAngle) * Mathf.Cos(azAngle);
                    pts.Add(new Vector3(x, y, z));
                }
            }
        }

        return new ZoneDef { name = "bucket_workspace", weight = 0.15f, samplePoints = pts.ToArray() };
    }

    static ZoneDef GenerateTruckDetectionZone()
    {
        // 90° direction (right side), 3-8m
        var pts = new List<Vector3>();
        for (float r = 3f; r <= 8f; r += 1f)
            for (float angOff = -30f; angOff <= 30f; angOff += 15f)
                for (float h = -1f; h <= 2f; h += 1f)
                {
                    float angle = (90f + angOff) * Mathf.Deg2Rad;
                    pts.Add(new Vector3(r * Mathf.Sin(angle), h, r * Mathf.Cos(angle)));
                }

        return new ZoneDef { name = "truck_detection", weight = 0.15f, samplePoints = pts.ToArray() };
    }

    static ZoneDef GenerateTerrainAheadZone()
    {
        // Forward 1-5m, ±2m, ground level
        var pts = new List<Vector3>();
        for (float fwd = 1f; fwd <= 5f; fwd += 1f)
            for (float lat = -2f; lat <= 2f; lat += 1f)
                pts.Add(new Vector3(lat, -2.5f, fwd));

        return new ZoneDef { name = "terrain_ahead", weight = 0.10f, samplePoints = pts.ToArray() };
    }

    static ZoneDef GenerateSurroundZone(float minR, float maxR)
    {
        // 360° general awareness
        var pts = new List<Vector3>();
        int nAngles = 24;
        float[] radii = { minR, (minR + maxR) / 2, maxR };

        for (int a = 0; a < nAngles; a++)
        {
            float angle = (360f / nAngles) * a * Mathf.Deg2Rad;
            for (int ri = 0; ri < radii.Length; ri++)
            {
                pts.Add(new Vector3(radii[ri] * Mathf.Sin(angle), 0f, radii[ri] * Mathf.Cos(angle)));
            }
        }

        return new ZoneDef
        {
            name = "surround",
            weight = 0.05f,
            samplePoints = pts.ToArray()
        };
    }

    static ZoneDef GenerateForwardPathZone()
    {
        // Forward 0-30m, ±3m
        var pts = new List<Vector3>();
        for (float fwd = 1f; fwd <= 30f; fwd += 3f)
            for (float lat = -3f; lat <= 3f; lat += 1.5f)
                for (float h = -1f; h <= 1f; h += 1f)
                    pts.Add(new Vector3(lat, h, fwd));

        return new ZoneDef { name = "forward_path", weight = 0.40f, samplePoints = pts.ToArray() };
    }

    static ZoneDef GenerateSideClearanceZone()
    {
        // ±5m lateral, 0-10m forward
        var pts = new List<Vector3>();
        for (float fwd = 0f; fwd <= 10f; fwd += 2f)
            for (float lat = -5f; lat <= 5f; lat += 2f)
                pts.Add(new Vector3(lat, 0f, fwd));

        return new ZoneDef { name = "side_clearance", weight = 0.25f, samplePoints = pts.ToArray() };
    }

    static ZoneDef GenerateDumpAreaZone()
    {
        // Rear 3-10m
        var pts = new List<Vector3>();
        for (float rear = -10f; rear <= -3f; rear += 1f)
            for (float lat = -2f; lat <= 2f; lat += 1f)
                for (float h = -1f; h <= 1f; h += 1f)
                    pts.Add(new Vector3(lat, h, rear));

        return new ZoneDef { name = "dump_area", weight = 0.20f, samplePoints = pts.ToArray() };
    }

    // ── Core evaluation ──────────────────────────────────────

    /// <summary>
    /// 1つの候補位置に対して、1つのアーム姿勢でのカバレッジを評価する。
    /// </summary>
    /// <param name="sensorWorldPos">センサのワールド座標</param>
    /// <param name="mountTransform">マウントリンクのTransform（サンプル点のローカル→ワールド変換用）</param>
    /// <param name="robotRoot">ロボットのルートTransform（自己遮蔽判定用）</param>
    /// <param name="zones">評価するゾーン定義</param>
    /// <param name="maxRange">Raycast最大距離</param>
    /// <returns>ゾーン別カバレッジ率</returns>
    public static Dictionary<string, float> EvaluateSingleConfig(
        Vector3 sensorWorldPos,
        Transform mountTransform,
        Transform robotRoot,
        ZoneDef[] zones,
        float maxRange = 50f)
    {
        var coverages = new Dictionary<string, float>();

        foreach (var zone in zones)
        {
            int visible = 0;
            int total = zone.samplePoints.Length;

            for (int i = 0; i < total; i++)
            {
                // Sample point: mount link local → world
                Vector3 sampleWorld = mountTransform.TransformPoint(zone.samplePoints[i]);
                Vector3 direction = sampleWorld - sensorWorldPos;
                float distance = direction.magnitude;

                if (distance < 0.01f)
                {
                    visible++;
                    continue;
                }

                RaycastHit hit;
                if (Physics.Raycast(sensorWorldPos, direction.normalized, out hit, Mathf.Min(distance + 0.1f, maxRange)))
                {
                    // Hit something — check if it's the robot itself
                    if (hit.transform.IsChildOf(robotRoot) || hit.transform == robotRoot)
                    {
                        // Self-occlusion: this point is blocked by the machine body
                        continue;
                    }
                    // Environment hit before reaching sample point
                    if (hit.distance < distance - 0.1f)
                    {
                        // Blocked by environment, but for placement purposes we count
                        // environment occlusion as "visible" (environment changes, machine doesn't)
                        visible++;
                    }
                    else
                    {
                        visible++;
                    }
                }
                else
                {
                    // No hit — clear line of sight
                    visible++;
                }
            }

            coverages[zone.name] = total > 0 ? (float)visible / total : 1f;
        }

        return coverages;
    }

    /// <summary>
    /// 重み付き合計スコアを算出する。
    /// </summary>
    public static float ComputeWeightedScore(Dictionary<string, float> coverages, ZoneDef[] zones)
    {
        float score = 0f;
        foreach (var zone in zones)
        {
            if (coverages.ContainsKey(zone.name))
                score += zone.weight * coverages[zone.name];
        }
        return score;
    }

    /// <summary>
    /// 複数アーム姿勢の最悪ケースカバレッジを算出する。
    /// zoneCoveragesの各ゾーンについてmin(全config)を返す。
    /// </summary>
    public static Dictionary<string, float> MinAcrossConfigs(List<Dictionary<string, float>> configCoverages)
    {
        if (configCoverages.Count == 0)
            return new Dictionary<string, float>();

        var result = new Dictionary<string, float>(configCoverages[0]);

        for (int c = 1; c < configCoverages.Count; c++)
        {
            foreach (var kvp in configCoverages[c])
            {
                if (result.ContainsKey(kvp.Key))
                    result[kvp.Key] = Mathf.Min(result[kvp.Key], kvp.Value);
                else
                    result[kvp.Key] = kvp.Value;
            }
        }

        return result;
    }

    // ── Arm posing helpers ───────────────────────────────────

    /// <summary>
    /// ショベルのアーム姿勢をTransform.localRotationで設定する。
    /// Editor modeでArticulationBodyを使わずにポーズを変更するため。
    /// </summary>
    public static void PoseExcavatorArm(Transform robotRoot, ArmPhase phase)
    {
        // Find links by name (recursive)
        Transform bodyLink = FindChildRecursive(robotRoot, "body_link");
        Transform boomLink = FindChildRecursive(robotRoot, "boom_link");
        Transform armLink = FindChildRecursive(robotRoot, "arm_link");
        Transform bucketLink = FindChildRecursive(robotRoot, "bucket_link");

        // ZX135U uses different naming
        if (bodyLink == null)
            bodyLink = FindChildRecursive(robotRoot, "rotator_link");
        if (bucketLink == null)
            bucketLink = FindChildRecursive(robotRoot, "backet_link");

        // Apply rotations (URDF joints are typically around Y-axis for boom/arm/bucket, Z for swing)
        if (bodyLink != null)
        {
            // Swing joint: rotation around local Z (or Y depending on URDF)
            bodyLink.localRotation = Quaternion.Euler(0, 0, phase.turntableDeg);
        }
        if (boomLink != null)
        {
            boomLink.localRotation = Quaternion.Euler(0, phase.boomDeg, 0);
        }
        if (armLink != null)
        {
            armLink.localRotation = Quaternion.Euler(0, phase.armDeg, 0);
        }
        if (bucketLink != null)
        {
            bucketLink.localRotation = Quaternion.Euler(0, phase.bucketDeg, 0);
        }
    }

    /// <summary>
    /// ダンプのベッセル姿勢を設定する。
    /// </summary>
    public static void PoseDumpTruckVessel(Transform robotRoot, VesselPhase phase)
    {
        Transform vesselLink = FindChildRecursive(robotRoot, "vessel_link");
        if (vesselLink != null)
        {
            vesselLink.localRotation = Quaternion.Euler(0, phase.vesselDeg, 0);
        }
    }

    /// <summary>
    /// 現在のlocalRotationを保存して後で復元するためのヘルパー。
    /// </summary>
    public static Dictionary<string, Quaternion> SavePose(Transform robotRoot)
    {
        var saved = new Dictionary<string, Quaternion>();
        SavePoseRecursive(robotRoot, saved);
        return saved;
    }

    public static void RestorePose(Transform robotRoot, Dictionary<string, Quaternion> saved)
    {
        RestorePoseRecursive(robotRoot, saved);
    }

    static void SavePoseRecursive(Transform t, Dictionary<string, Quaternion> dict)
    {
        dict[GetFullPath(t)] = t.localRotation;
        foreach (Transform child in t)
            SavePoseRecursive(child, dict);
    }

    static void RestorePoseRecursive(Transform t, Dictionary<string, Quaternion> dict)
    {
        string path = GetFullPath(t);
        if (dict.ContainsKey(path))
            t.localRotation = dict[path];
        foreach (Transform child in t)
            RestorePoseRecursive(child, dict);
    }

    static string GetFullPath(Transform t)
    {
        string path = t.name;
        while (t.parent != null)
        {
            t = t.parent;
            path = t.name + "/" + path;
        }
        return path;
    }

    public static Transform FindChildRecursive(Transform parent, string name)
    {
        foreach (Transform child in parent)
        {
            if (child.name == name) return child;
            var found = FindChildRecursive(child, name);
            if (found != null) return found;
        }
        return null;
    }

    // ── Candidate generation ─────────────────────────────────

    /// <summary>ショベル用の候補位置グリッドを生成（body_link ローカル座標）</summary>
    public static Vector3[] GenerateExcavatorCandidates()
    {
        var candidates = new List<Vector3>();
        for (float x = -0.5f; x <= 1.0f; x += 0.25f)
            for (float y = 2.0f; y <= 3.0f; y += 0.25f)
                for (float z = -0.5f; z <= 0.5f; z += 0.25f)
                    candidates.Add(new Vector3(x, y, z));
        return candidates.ToArray();
    }

    /// <summary>ダンプ用の候補位置グリッドを生成（base_link ローカル座標）</summary>
    public static Vector3[] GenerateDumpTruckCandidates()
    {
        var candidates = new List<Vector3>();
        for (float x = -1.0f; x <= 2.0f; x += 0.25f)
            for (float y = 2.0f; y <= 3.0f; y += 0.25f)
                for (float z = -0.5f; z <= 0.5f; z += 0.25f)
                    candidates.Add(new Vector3(x, y, z));
        return candidates.ToArray();
    }
}
