using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEngine;
using UnitySensors.Sensor.GNSS;
using UnitySensors.Sensor.IMU;
using UnitySensors.Sensor.LiDAR;
using UnitySensors.ROS.Publisher.GNSS;
using UnitySensors.ROS.Publisher.IMU;
using CapKit;
using Object = UnityEngine.Object;

namespace CapKit.Editor
{
    /// <summary>
    /// CAP perception kit editor tool. Reads Assets/CAP/sensor_rig.json (generated from the
    /// cap-pangaea contract by scripts/export_sensor_rig.py) and PERMANENTLY edits the machine
    /// prefabs: kit_hub (ground-truth odometry stand-in, GNSS pair, heading), one kit_&lt;sensor_id&gt;
    /// mount per declared sensor (roof + fl/fr/rl/rr), and the LiDAR/IMU devices for populated sensors.
    /// Idempotent: re-running updates poses/params and prunes mounts no longer in the JSON.
    /// </summary>
    public static class CapKitRig
    {
        private const string PrefabFolder = "Assets/Prefab";
        private const string DeviceName = "device";

        public sealed class Report
        {
            public readonly List<string> Infos = new List<string>();
            public readonly List<string> Warnings = new List<string>();
            public readonly List<string> Errors = new List<string>();
            public void Info(string m) { Infos.Add(m); Debug.Log("[CAP Kit] " + m); }
            public void Warn(string m) { Warnings.Add(m); Debug.LogWarning("[CAP Kit] " + m); }
            public void Error(string m) { Errors.Add(m); Debug.LogError("[CAP Kit] " + m); }
        }

        [MenuItem("CAP/Perception Kit/Apply From Contract")]
        public static void ApplyFromContract() => RunMenu("Apply Perception Kit", Apply);

        [MenuItem("CAP/Perception Kit/Remove From Prefabs")]
        public static void RemoveFromPrefabs() => RunMenu("Remove Perception Kit", Remove);

        /// <summary>Never throws out of a menu handler; collects findings into one dialog.</summary>
        public static void RunMenu(string title, Action<Report> body)
        {
            var report = new Report();
            try { body(report); }
            catch (Exception e) { report.Error(e.GetType().Name + ": " + e.Message); }
            finally { AssetDatabase.SaveAssets(); }
            string summary = report.Infos.Count + " ok, " + report.Warnings.Count + " warnings, " + report.Errors.Count + " errors";
            var lines = report.Errors.Concat(report.Warnings).Take(12).ToList();
            EditorUtility.DisplayDialog(title, summary + (lines.Count > 0 ? "\n\n" + string.Join("\n", lines) : "") +
                "\n\nDetails: Console", "OK");
        }

        // ── apply ─────────────────────────────────────────────────────────────────────
        private static void Apply(Report report)
        {
            var contract = CapKitContract.Load();
            foreach (var machine in contract.machines)
            {
                string path = PrefabFolder + "/" + machine.machine_id + ".prefab";
                if (!File.Exists(path)) { report.Warn(machine.machine_id + ": prefab not found (" + path + "), skipped."); continue; }
                try { machine.Validate(); }
                catch (Exception e) { report.Error(machine.machine_id + ": contract invalid: " + e.Message); continue; }
                GameObject root = PrefabUtility.LoadPrefabContents(path);
                try
                {
                    int devices = ApplyMachine(root, machine, report);
                    PrefabUtility.SaveAsPrefabAsset(root, path);
                    report.Info(machine.machine_id + ": kit applied (" + machine.sensors.Length + " frames, " + devices + " devices).");
                }
                catch (Exception e) { report.Error(machine.machine_id + ": " + e.Message); }
                finally { PrefabUtility.UnloadPrefabContents(root); }
            }
        }

        private static int ApplyMachine(GameObject root, CapKitContract.Machine m, Report report)
        {
            var keep = new HashSet<GameObject>();
            Transform hubLink = CapKitRigObjects.FindLink(root, m.hub_link);
            var hub = CapKitRigObjects.Mount(root, hubLink, "kit_hub", m.hub.unity.Position, m.hub.unity.Rotation, keep);
            Describe(hub, "hub", "kit_hub", true, m.kit_odom_topic, m.kit_child_frame);

            // Kit odometry stand-in: OperaSim GroundTruthPublisher (nav_msgs/Odometry, frame map, ROS conversion).
            var odom = CapKitRigObjects.Component<GroundTruthPublisher>(hub.gameObject);
            odom.topicName = m.kit_odom_topic;
            odom.childFrameName = m.kit_child_frame;
            odom.publishMessageInterval = 0.02f; // 50 Hz
            EditorUtility.SetDirty(odom);

            int devices = 0;
            if (m.gnss != null)
            {
                float half = Mathf.Max(0f, m.gnss.baseline_m) * 0.5f;
                // Antenna pair along the hub forward axis (Unity local +z == ROS +x).
                ApplyGnss(root, hub, "kit_gnss_a", new Vector3(0f, 0f, +half), m.gnss.fix_topic, m.machine_id + "/kit_gnss_a", keep);
                ApplyGnss(root, hub, "kit_gnss_b", new Vector3(0f, 0f, -half), m.gnss.fix_topic + "_b", m.machine_id + "/kit_gnss_b", keep);
                var heading = CapKitRigObjects.Component<QuaternionStampedPublisher>(hub.gameObject);
                heading.topicName = m.gnss.heading_topic;
                heading.frameID = "map";
                heading.publishMessageInterval = 0.2f; // 5 Hz moving-base rate
                EditorUtility.SetDirty(heading);
                report.Warn(m.machine_id + ": GNSSSensor needs a scene GeoCoordinateSystem assigned on the kit_gnss_* sensors " +
                            "(scene object; cannot be stored in the prefab). Heading is the hub world yaw, not a two-antenna solution.");
            }

            foreach (var s in m.sensors)
            {
                Transform parent = CapKitRigObjects.FindLink(root, s.parent_link);
                var marker = CapKitRigObjects.Mount(root, parent, "kit_" + s.sensor_id, s.unity.Position, s.unity.Rotation, keep);
                Describe(marker, s.sensor_id, s.model, s.populated, s.topic, s.frame_id);
                if (!s.populated)
                {
                    CapKitRigObjects.RemoveDevice(marker);
                    CapKitRigObjects.Remove<IMUMsgPublisher>(marker.gameObject);
                    CapKitRigObjects.Remove<IMUSensor>(marker.gameObject);
                    continue;
                }
                ApplyLidar(marker, s, report);
                devices++;
                if (s.imu != null)
                {
                    // Compensating IMU rides on the same rigid mount as its LiDAR (never on a compliant sub-mount).
                    var imu = CapKitRigObjects.Component<IMUSensor>(marker.gameObject);
                    CapKitRigObjects.Frequency(imu, Mathf.Min(s.imu.rate_hz, 100f));
                    var pub = CapKitRigObjects.Component<IMUMsgPublisher>(marker.gameObject);
                    CapKitRigObjects.Publisher(pub, s.imu.topic, s.imu.frame_id, Mathf.Min(s.imu.rate_hz, 100f));
                    devices++;
                }
                else
                {
                    CapKitRigObjects.Remove<IMUMsgPublisher>(marker.gameObject);
                    CapKitRigObjects.Remove<IMUSensor>(marker.gameObject);
                }
            }
            int pruned = CapKitRigObjects.Prune(root, keep);
            if (pruned > 0) report.Info(m.machine_id + ": pruned " + pruned + " stale kit object(s).");
            return devices;
        }

        private static void ApplyGnss(GameObject root, CapKitFrameMarker hub, string name, Vector3 localPos,
            string topic, string frame, HashSet<GameObject> keep)
        {
            var marker = CapKitRigObjects.Mount(root, hub.transform, name, localPos, Quaternion.identity, keep);
            Describe(marker, name.Replace("kit_", ""), "zed_f9p", true, topic, frame);
            var sensor = CapKitRigObjects.Component<GNSSSensor>(marker.gameObject);
            CapKitRigObjects.Frequency(sensor, 5f);
            var pub = CapKitRigObjects.Component<NavSatFixMsgPublisher>(marker.gameObject);
            CapKitRigObjects.Publisher(pub, topic, frame, 5f);
        }

        private static void ApplyLidar(CapKitFrameMarker marker, CapKitContract.Sensor s, Report report)
        {
            GameObject device = marker.sensorInstance;
            bool fromPrefab = !string.IsNullOrWhiteSpace(s.sim.unity_prefab);
            if (device != null && device.transform.parent != marker.transform) { CapKitRigObjects.RemoveDevice(marker); device = null; }
            if (device != null && fromPrefab)
            {
                // Re-instantiate when the requested prefab changed.
                var src = PrefabUtility.GetCorrespondingObjectFromSource(device);
                if (src == null || src.name != s.sim.unity_prefab) { CapKitRigObjects.RemoveDevice(marker); device = null; }
            }
            if (device == null)
            {
                if (fromPrefab)
                {
                    GameObject prefab = FindSensorPrefab(s.sim.unity_prefab);
                    device = (GameObject)PrefabUtility.InstantiatePrefab(prefab, marker.transform);
                }
                else
                {
                    device = new GameObject(DeviceName);
                    device.transform.SetParent(marker.transform, false);
                    device.AddComponent<RaycastLiDARSensor>();
                }
                device.transform.localPosition = Vector3.zero;
                device.transform.localRotation = Quaternion.identity;
                device.transform.localScale = Vector3.one;
                marker.sensorInstance = device;
                EditorUtility.SetDirty(marker);
            }
            var lidar = device.GetComponentInChildren<RaycastLiDARSensor>(true);
            if (lidar == null) throw new InvalidOperationException(s.sensor_id + ": device has no RaycastLiDARSensor.");
            CapKitRigObjects.Edit(lidar, so =>
            {
                if (!fromPrefab)
                    CapKitRigObjects.Property(so, "_scanPattern").objectReferenceValue = CapScanPatternGenerator.GetOrCreate(s.sim.pattern_asset);
                CapKitRigObjects.Property(so, "_pointsNumPerScan").intValue = s.sim.points_per_update;
                CapKitRigObjects.Property(so, "_minRange").floatValue = s.range_m.min;
                CapKitRigObjects.Property(so, "_maxRange").floatValue = s.range_m.max;
                CapKitRigObjects.Property(so, "_gaussianNoiseSigma").floatValue = s.sim.gaussian_sigma_m;
                CapKitRigObjects.Property(so, "_frequency").floatValue = s.sim.rate_hz;
            });
            // Kit publisher (XYZ+intensity, single timestamp) replaces any stock PointCloud2 publisher on the device.
            foreach (var stock in device.GetComponentsInChildren<UnitySensors.ROS.Publisher.LiDAR.RaycastLiDARPointCloud2MsgPublisher>(true))
                Object.DestroyImmediate(stock);
            var pub = CapKitRigObjects.Component<KitRaycastLiDARPointCloud2MsgPublisher>(lidar.gameObject);
            CapKitRigObjects.Publisher(pub, s.topic, s.frame_id, s.rate_hz);
            if (fromPrefab && string.Equals(s.sim.unity_prefab, "Mid-360", StringComparison.OrdinalIgnoreCase))
                report.Info(s.sensor_id + ": Mid-360 prefab (real scan pattern) mounted inverted per contract rpy [180,0,0].");
        }

        private static GameObject FindSensorPrefab(string name)
        {
            var hits = AssetDatabase.FindAssets(name + " t:Prefab")
                .Select(AssetDatabase.GUIDToAssetPath)
                .Where(p => Path.GetFileNameWithoutExtension(p) == name && p.Contains("/Prefabs/"))
                .ToList();
            if (hits.Count == 0) throw new InvalidOperationException("UnitySensors prefab '" + name + "' not found.");
            var packageHit = hits.FirstOrDefault(p => p.StartsWith("Packages/")) ?? hits[0];
            return AssetDatabase.LoadAssetAtPath<GameObject>(packageHit);
        }

        private static void Describe(CapKitFrameMarker marker, string id, string model, bool populated, string topic, string frame)
        {
            marker.sensorId = id;
            marker.model = model;
            marker.populated = populated;
            marker.topic = topic;
            marker.frameId = frame;
            EditorUtility.SetDirty(marker);
        }

        // ── remove ────────────────────────────────────────────────────────────────────
        private static void Remove(Report report)
        {
            foreach (string path in Directory.GetFiles(PrefabFolder, "*.prefab"))
            {
                string assetPath = path.Replace('\\', '/');
                GameObject root = PrefabUtility.LoadPrefabContents(assetPath);
                try
                {
                    int removed = CapKitRigObjects.Prune(root, new HashSet<GameObject>());
                    if (removed > 0)
                    {
                        PrefabUtility.SaveAsPrefabAsset(root, assetPath);
                        report.Info(Path.GetFileNameWithoutExtension(assetPath) + ": removed " + removed + " kit object(s).");
                    }
                }
                catch (Exception e) { report.Error(assetPath + ": " + e.Message); }
                finally { PrefabUtility.UnloadPrefabContents(root); }
            }
        }
    }
}
