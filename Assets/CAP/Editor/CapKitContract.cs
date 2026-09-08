using System;
using System.Collections.Generic;
using System.IO;
using System.Text.RegularExpressions;
using UnityEngine;

namespace CapKit.Editor
{
    // JsonUtility reads the exported Unity poses directly; unknown descriptive fields are ignored.
    [Serializable] internal sealed class CapKitContract
    {
        public const string Path = "Assets/CAP/sensor_rig.json";
        public string schema_version;
        public Machine[] machines;

        public static CapKitContract Load()
        {
            var contract = JsonUtility.FromJson<CapKitContract>(File.ReadAllText(Path));
            Require(contract != null && contract.schema_version == "1.0", "Expected sensor_rig schema_version 1.0.");
            Require(contract.machines != null, "Contract has no machines array.");
            var ids = new HashSet<string>(StringComparer.Ordinal);
            foreach (var machine in contract.machines)
            {
                Require(machine != null, "Null machine entry.");
                Identifier(machine.machine_id);
                Require(ids.Add(machine.machine_id), "Duplicate machine_id: " + machine.machine_id);
            }
            return contract;
        }

        [Serializable] internal sealed class Machine
        {
            public string machine_id, hub_link, kit_odom_topic, kit_odom_frame, kit_child_frame;
            public Hub hub;
            public Gnss gnss;
            public Sensor[] sensors;

            public void Validate()
            {
                Identifier(hub_link);
                Require(hub != null, "Missing hub pose.");
                Pose.Validate(hub.unity);
                Text(kit_odom_topic, "kit_odom_topic");
                Text(kit_child_frame, "kit_child_frame");
                Require(kit_odom_frame == "map",
                    "GroundTruthPublisher supports only kit_odom_frame=map; prefab left unchanged.");
                if (gnss != null)
                {
                    Require(Finite(gnss.baseline_m) && gnss.baseline_m >= 0f, "Invalid GNSS baseline.");
                    Require(Finite(gnss.heading_offset_deg), "Invalid heading offset.");
                    Text(gnss.fix_topic, "gnss.fix_topic");
                    Text(gnss.heading_topic, "gnss.heading_topic");
                }
                Require(sensors != null, "Missing sensors array (use [] to remove sensors).");
                var names = new HashSet<string>(StringComparer.Ordinal) { "hub", "gnss_a", "gnss_b" };
                foreach (var sensor in sensors)
                {
                    Require(sensor != null, "Null sensor entry.");
                    Identifier(sensor.sensor_id);
                    Require(names.Add(sensor.sensor_id), "Duplicate/reserved sensor_id: " + sensor.sensor_id);
                    Identifier(sensor.parent_link);
                    Pose.Validate(sensor.unity);
                    Text(sensor.frame_id, "sensor.frame_id");
                    if (!sensor.populated) continue;
                    Require(sensor.kind == "lidar", "Unsupported populated sensor kind: " + sensor.kind);
                    Text(sensor.topic, "sensor.topic");
                    Positive(sensor.rate_hz, "sensor.rate_hz");
                    Require(sensor.sim != null && sensor.range_m != null, "Populated LiDAR requires sim and range_m.");
                    Positive(sensor.sim.rate_hz, "sim.rate_hz");
                    Require(sensor.sim.points_per_update > 0, "points_per_update must be positive.");
                    Require(Finite(sensor.range_m.min) && sensor.range_m.min >= 0f
                        && Finite(sensor.range_m.max) && sensor.range_m.max > sensor.range_m.min, "Invalid LiDAR range.");
                    Require(Finite(sensor.sim.gaussian_sigma_m) && sensor.sim.gaussian_sigma_m >= 0f, "Invalid range noise.");
                    Require(!string.IsNullOrWhiteSpace(sensor.sim.unity_prefab)
                        || !string.IsNullOrWhiteSpace(sensor.sim.pattern_asset), "LiDAR requires a prefab or pattern_asset.");
                    if (sensor.imu != null)
                    {
                        Text(sensor.imu.topic, "imu.topic");
                        Text(sensor.imu.frame_id, "imu.frame_id");
                        Positive(sensor.imu.rate_hz, "imu.rate_hz");
                    }
                }
            }
        }

        [Serializable] internal sealed class Hub { public Pose unity; }
        [Serializable] internal sealed class Pose
        {
            public float[] position, euler;
            public Vector3 Position => new Vector3(position[0], position[1], position[2]);
            public Quaternion Rotation => Quaternion.Euler(euler[0], euler[1], euler[2]);
            public static void Validate(Pose pose)
            {
                Require(pose != null, "Missing Unity pose.");
                foreach (var values in new[] { pose.position, pose.euler })
                {
                    Require(values != null && values.Length == 3, "Unity pose requires three position/euler values.");
                    foreach (float value in values) Require(Finite(value), "Non-finite Unity pose.");
                }
            }
        }
        [Serializable] internal sealed class Gnss
        {
            public float baseline_m, heading_offset_deg;
            public string fix_topic, heading_topic;
        }
        [Serializable] internal sealed class Sensor
        {
            public string sensor_id, kind, model, parent_link, frame_id, topic;
            public bool populated;
            public Pose unity;
            public Range range_m;
            public float rate_hz;
            public Simulation sim;
            public Imu imu;
        }
        [Serializable] internal sealed class Range { public float min, max; }
        [Serializable] internal sealed class Simulation
        {
            public string unity_prefab, pattern_asset;
            public int points_per_update;
            public float rate_hz, gaussian_sigma_m;
        }
        [Serializable] internal sealed class Imu
        {
            public string sensor_id, topic, frame_id;
            public float rate_hz, range_g;
        }

        private static bool Finite(float value) => !float.IsNaN(value) && !float.IsInfinity(value);
        private static void Positive(float value, string field) => Require(Finite(value) && value > 0f, field + " must be positive.");
        private static void Text(string value, string field) => Require(!string.IsNullOrWhiteSpace(value), "Missing " + field);
        private static void Identifier(string value) => Require(value != null && Regex.IsMatch(value, @"\A[A-Za-z0-9_-]+\z"), "Invalid identifier: " + value);
        private static void Require(bool valid, string message)
        {
            if (!valid) throw new InvalidDataException(message);
        }
    }
}
