using UnityEngine;
using UnityEditor;
using System.Collections.Generic;

/// <summary>
/// Unity Editorメニューから全ロボットにセンサコンポーネントを自動アタッチするユーティリティ。
/// メニュー: CAP > Setup Sensors on All Robots
///
/// LiDARマウント位置は機種別の最適化結果を使用。
/// 最適化は CAP > Optimize Sensor Placement で実行可能。
///
/// Phase 5β-3-cal: マルチカメラ配置 (front/bucket/bed/rear/left/right) を
/// 機種別に script で attach する。配置データは
/// cap-platform/shared/camera_extrinsics.json と同期 (CI で sync test が
/// drift を検出する)。
///
/// 座標系: JSON は ROS convention (forward=+x, left=+y, up=+z, yaw CCW)。
/// C# 側で Unity local frame (x=right, y=up, z=forward, yaw_y CW) に変換:
///   Unity.x = -lateral_m
///   Unity.y =  height_m
///   Unity.z =  forward_m
///   Unity yaw_y = -yaw_deg  (CCW → CW)
///   Unity pitch_x = pitch_deg  (どちらも nose-down 正)
/// </summary>
public class SensorSetup
{
    /// <summary>
    /// 機種別の最適LiDARマウント位置。
    /// SensorPlacementOptEditor の結果で更新する。
    /// キーが見つからない場合はデフォルト (0, 2.5, 0) を使用。
    /// </summary>
    static readonly Dictionary<string, Vector3> OptimalLidarPosition = new Dictionary<string, Vector3>
    {
        // デフォルト値 — CAP > Optimize Sensor Placement で実測後に更新
        { "zx135u",   new Vector3(0f, 2.5f, 0f) },
        { "zx120",    new Vector3(0f, 2.5f, 0f) },
        { "zx200",    new Vector3(0f, 2.5f, 0f) },
        { "ic120",    new Vector3(0f, 2.5f, 0f) },
        { "c30r",     new Vector3(0f, 2.5f, 0f) },
        { "mst110cr", new Vector3(0f, 2.5f, 0f) },
    };

    static readonly Vector3 DefaultLidarPosition = new Vector3(0f, 2.5f, 0f);

    /// <summary>
    /// Phase 5β-3-cal — 機種別カメラ extrinsics。
    /// MUST stay in sync with cap-platform/shared/camera_extrinsics.json.
    /// CI test (test_camera_extrinsics_sync.py) verifies the two match.
    ///
    /// Values are in ROS convention (forward, lateral, height in meters;
    /// yaw_deg/pitch_deg in degrees). Conversion to Unity local frame
    /// happens in PlaceCamera() below.
    /// </summary>
    public class CameraSpec
    {
        public float ForwardM;
        public float LateralM;
        public float HeightM;
        public float YawDeg;
        public float PitchDeg;
        public float FovDeg;
        public string ParentLink;  // empty = base_link

        public CameraSpec(float forward, float lateral, float height, float yaw, float pitch, float fov, string parentLink = "")
        {
            ForwardM = forward;
            LateralM = lateral;
            HeightM = height;
            YawDeg = yaw;
            PitchDeg = pitch;
            FovDeg = fov;
            ParentLink = parentLink;
        }
    }

    static readonly Dictionary<string, Dictionary<string, CameraSpec>> OptimalCameraPlacements =
        new Dictionary<string, Dictionary<string, CameraSpec>>
    {
        { "zx120", new Dictionary<string, CameraSpec> {
            { "front",  new CameraSpec(0.8f,  0.0f, 2.7f,   0f,  8f, 85f) },
            { "bucket", new CameraSpec(4.5f,  0.0f, 1.5f,   0f, 45f, 60f, "arm_link") },
            { "rear",   new CameraSpec(-1.5f, 0.0f, 2.7f, 180f, 12f, 95f) },
        }},
        { "zx200", new Dictionary<string, CameraSpec> {
            { "front",  new CameraSpec(1.0f,  0.0f, 2.8f,   0f,  8f, 85f) },
            { "bucket", new CameraSpec(5.5f,  0.0f, 1.7f,   0f, 45f, 60f, "arm_link") },
            { "rear",   new CameraSpec(-1.8f, 0.0f, 2.8f, 180f, 12f, 95f) },
        }},
        { "ic120", new Dictionary<string, CameraSpec> {
            { "front",  new CameraSpec(1.2f,  0.0f, 2.7f,   0f,  5f, 90f) },
            { "bed",    new CameraSpec(0.5f,  0.0f, 3.0f, 180f, 35f, 100f) },
            { "rear",   new CameraSpec(-3.5f, 0.0f, 2.0f, 180f, 10f, 110f) },
            { "left",   new CameraSpec(0.0f,  1.3f, 2.4f,  90f, 10f, 90f) },
            { "right",  new CameraSpec(0.0f, -1.3f, 2.4f, -90f, 10f, 90f) },
        }},
        { "c30r", new Dictionary<string, CameraSpec> {
            { "front",  new CameraSpec(1.0f,  0.0f, 2.5f,   0f,  5f, 90f) },
            { "bed",    new CameraSpec(0.3f,  0.0f, 2.8f, 180f, 35f, 100f) },
            { "rear",   new CameraSpec(-2.7f, 0.0f, 1.8f, 180f, 10f, 110f) },
            { "left",   new CameraSpec(0.0f,  1.1f, 2.2f,  90f, 10f, 90f) },
            { "right",  new CameraSpec(0.0f, -1.1f, 2.2f, -90f, 10f, 90f) },
        }},
    };

    static readonly Dictionary<string, CameraSpec> DefaultCameraPlacements = new Dictionary<string, CameraSpec>
    {
        { "front", new CameraSpec(0.5f, 0.0f, 2.5f, 0f, 10f, 90f) },
    };

    static Vector3 GetLidarPosition(string robotName)
    {
        string key = robotName.ToLower();
        foreach (var kvp in OptimalLidarPosition)
        {
            if (key.Contains(kvp.Key))
                return kvp.Value;
        }
        return DefaultLidarPosition;
    }

    static Dictionary<string, CameraSpec> GetCameraPlacements(string robotName)
    {
        string key = robotName.ToLower();
        foreach (var kvp in OptimalCameraPlacements)
        {
            if (key.Contains(kvp.Key))
                return kvp.Value;
        }
        return DefaultCameraPlacements;
    }

    [MenuItem("CAP/Setup Sensors on All Robots")]
    static void SetupSensors()
    {
        var robots = GameObject.FindGameObjectsWithTag("robot");

        if (robots.Length == 0)
        {
            Debug.LogWarning("[SensorSetup] No GameObjects with tag 'robot' found. Open a scene with robots first.");
            return;
        }

        int sensorsAdded = 0;

        foreach (var robot in robots)
        {
            string robotName = robot.name;
            Debug.Log($"[SensorSetup] Setting up sensors for: {robotName}");

            // Find base_link child
            Transform baseLink = FindChildRecursive(robot.transform, "base_link");
            if (baseLink == null)
            {
                Debug.LogWarning($"[SensorSetup] No 'base_link' found for {robotName}, skipping.");
                continue;
            }

            // --- IMU on base_link ---
            if (baseLink.GetComponent<IMUPublisher>() == null)
            {
                var imu = baseLink.gameObject.AddComponent<IMUPublisher>();
                imu.topicName = "[robot_name]/imu/body";
                imu.frameName = "[robot_name]/imu_body_link";
                imu.publishMessageInterval = 0.02f; // 50Hz
                sensorsAdded++;
                Debug.Log($"  + IMUPublisher on {robotName}/base_link");
            }

            // --- GNSS on base_link ---
            if (baseLink.GetComponent<GNSSPublisher>() == null)
            {
                var gnss = baseLink.gameObject.AddComponent<GNSSPublisher>();
                gnss.topicName = "[robot_name]/gnss/fix";
                gnss.frameName = "[robot_name]/gnss_link";
                gnss.publishMessageInterval = 0.1f; // 10Hz
                gnss.originLatitude = 36.0;
                gnss.originLongitude = 140.0;
                gnss.originAltitude = 0.0;
                sensorsAdded++;
                Debug.Log($"  + GNSSPublisher on {robotName}/base_link");
            }

            // --- LiDAR on lidar_mount (create child if needed) ---
            Transform lidarMount = baseLink.Find("lidar_mount");
            if (lidarMount == null)
            {
                var lidarGO = new GameObject("lidar_mount");
                lidarGO.transform.SetParent(baseLink, false);
                Vector3 lidarPos = GetLidarPosition(robotName);
                lidarGO.transform.localPosition = lidarPos;
                lidarMount = lidarGO.transform;
                Debug.Log($"  + Created lidar_mount at ({lidarPos.x:F2}, {lidarPos.y:F2}, {lidarPos.z:F2}) for {robotName}");
            }

            if (lidarMount.GetComponent<LiDARPublisher>() == null)
            {
                var lidar = lidarMount.gameObject.AddComponent<LiDARPublisher>();
                lidar.topicName = "[robot_name]/lidar/front/points";
                lidar.frameName = "[robot_name]/lidar_link";
                lidar.publishMessageInterval = 0.1f; // 10Hz
                lidar.horizontalRays = 360;
                lidar.verticalChannels = 16;
                lidar.maxRange = 50f;
                lidar.minRange = 0.5f;
                sensorsAdded++;
                Debug.Log($"  + LiDARPublisher on {robotName}/base_link/lidar_mount");
            }

            // --- LiDAR Visualizer (点群可視化) ---
            if (lidarMount.GetComponent<LiDARVisualizer>() == null)
            {
                lidarMount.gameObject.AddComponent<LiDARVisualizer>();
                sensorsAdded++;
                Debug.Log($"  + LiDARVisualizer on {robotName}/base_link/lidar_mount");
            }

            // --- Phase 5β-3-cal: Multi-direction cameras ---
            // Iterate the per-machine extrinsics dict and create a
            // camera_<direction>_mount GameObject + CameraImagePublisher
            // for each direction. The resulting topic shape mirrors the
            // Phase 5α-2 / cap_bridge_ros2 expectation:
            //   /<robot>/camera/<direction>/image_raw
            // and frame:
            //   <robot>/camera_<direction>_link
            var cameraPlacements = GetCameraPlacements(robotName);
            foreach (var kvp in cameraPlacements)
            {
                string direction = kvp.Key;
                CameraSpec spec = kvp.Value;
                if (PlaceCamera(baseLink, robotName, direction, spec))
                    sensorsAdded++;
            }

            // --- Excavator-specific: IMU on boom/arm/bucket ---
            bool isExcavator = robotName.Contains("zx");
            if (isExcavator)
            {
                AddIMUToLink(baseLink, "boom_link", "[robot_name]/imu/boom", ref sensorsAdded, robotName);
                AddIMUToLink(baseLink, "arm_link", "[robot_name]/imu/arm", ref sensorsAdded, robotName);
                AddIMUToLink(baseLink, "bucket_link", "[robot_name]/imu/bucket", ref sensorsAdded, robotName);
            }

            // Mark the prefab instance as dirty so changes are saved
            EditorUtility.SetDirty(robot);
        }

        Debug.Log($"[SensorSetup] Done! Added {sensorsAdded} sensor components to {robots.Length} robots.");
        Debug.Log("[SensorSetup] Remember to save the scene (Ctrl+S) to persist changes.");
    }

    /// <summary>
    /// Phase 5β-3-cal — create or update a single direction's camera_mount.
    /// Returns true if any new component was added (so the caller can
    /// increment its counter). Idempotent: existing camera_mount is
    /// reused; existing CameraImagePublisher is not duplicated.
    /// </summary>
    static bool PlaceCamera(Transform baseLink, string robotName, string direction, CameraSpec spec)
    {
        // Resolve parent: spec.ParentLink overrides base_link (e.g. excavator
        // bucket camera attaches to arm_link so it follows boom motion).
        Transform parent = baseLink;
        if (!string.IsNullOrEmpty(spec.ParentLink))
        {
            Transform overrideParent = FindChildRecursive(baseLink, spec.ParentLink);
            if (overrideParent != null)
            {
                parent = overrideParent;
            }
            else
            {
                Debug.LogWarning(
                    $"  ! {robotName}: parent_link '{spec.ParentLink}' for {direction} camera not found, "
                    + $"falling back to base_link");
            }
        }

        string mountName = $"camera_{direction}_mount";
        Transform mount = parent.Find(mountName);

        // Phase 5β-3-cal v1: also clean up the legacy "camera_mount"
        // (Phase 5α single-camera default name) when its direction is "front"
        // — this preserves backward compatibility for the front camera while
        // moving to the per-direction naming convention.

        bool addedSomething = false;
        if (mount == null)
        {
            var go = new GameObject(mountName);
            go.transform.SetParent(parent, false);
            // ROS forward (+x) → Unity forward (+z). ROS left (+y) → Unity right (-x).
            // ROS up (+z) → Unity up (+y).
            go.transform.localPosition = new Vector3(
                -spec.LateralM, spec.HeightM, spec.ForwardM);
            // ROS yaw is CCW positive about +z (looking down).
            // Unity yaw_y is CW positive viewed down +y. So Unity yaw = -ROS yaw.
            // ROS pitch (positive = nose down) ≡ Unity pitch_x positive.
            go.transform.localRotation = Quaternion.Euler(spec.PitchDeg, -spec.YawDeg, 0f);
            mount = go.transform;
            addedSomething = true;
            Debug.Log(
                $"  + Created {mountName} at "
                + $"(local {go.transform.localPosition.x:F2},"
                + $" {go.transform.localPosition.y:F2},"
                + $" {go.transform.localPosition.z:F2})"
                + $" yaw={-spec.YawDeg:F0}° pitch={spec.PitchDeg:F0}° "
                + $"under '{parent.name}' for {robotName}");
        }

        if (mount.GetComponent<CameraImagePublisher>() == null)
        {
            var cam = mount.gameObject.AddComponent<CameraImagePublisher>();
            cam.topicName = $"[robot_name]/camera/{direction}/image_raw";
            cam.frameName = $"[robot_name]/camera_{direction}_link";
            cam.publishMessageInterval = 0.1f; // 10Hz
            cam.imageWidth = 640;
            cam.imageHeight = 480;
            // Phase 5β-3-cal-2: push fov from spec into the publisher's
            // public field. CameraImagePublisher.Start() applies it to
            // the Unity Camera component (created in Start).
            cam.fovDeg = spec.FovDeg;
            addedSomething = true;
            Debug.Log($"  + CameraImagePublisher on {robotName}/{mountName} (fov={spec.FovDeg:F0}°)");
        }

        return addedSomething;
    }

    static void AddIMUToLink(Transform baseLink, string linkName, string topic, ref int count, string robotName)
    {
        Transform link = FindChildRecursive(baseLink, linkName);
        if (link == null)
        {
            Debug.LogWarning($"  ! {linkName} not found for {robotName}");
            return;
        }
        if (link.GetComponent<IMUPublisher>() == null)
        {
            var imu = link.gameObject.AddComponent<IMUPublisher>();
            imu.topicName = topic;
            imu.frameName = $"[robot_name]/{linkName.Replace("_link", "")}_imu_link";
            imu.publishMessageInterval = 0.02f; // 50Hz
            count++;
            Debug.Log($"  + IMUPublisher on {robotName}/{linkName}");
        }
    }

    static Transform FindChildRecursive(Transform parent, string name)
    {
        foreach (Transform child in parent)
        {
            if (child.name == name) return child;
            var found = FindChildRecursive(child, name);
            if (found != null) return found;
        }
        return null;
    }
}
