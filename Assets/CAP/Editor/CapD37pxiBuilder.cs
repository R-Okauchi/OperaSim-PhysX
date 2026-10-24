using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Unity.Robotics.UrdfImporter;
using UnityEditor;
using UnityEngine;
using Object = UnityEngine.Object;

namespace CapKit.Editor
{
    /// <summary>
    /// The D37PXi dozer as a machine, made the way the other machines are (mst110cr's pattern): imported from PWRI's
    /// URDF (Assets/Machines/d37pxi, from upstream feature/add_d37pxi), then a root Rigidbody carrying six
    /// WheelColliders for the tracks (DiffDriveController), the articulation (body and blade) fixed to it, the blade's
    /// joints driven at a cylinder's speed (CapBladeDrive), the joints and the pose published under d37pxi/, and every
    /// collider but the wheels on the "Ignore Collision" layer - the blade meets the ground through CAP's soil, not
    /// through PhysX (a blade that collided would ride on its own berm, and CapTrackRollerFix would treat it by where it
    /// happened to stand). Written to Assets/Resources/Prefab/d37pxi.prefab, where a scene can spawn it by name.
    ///   Unity -batchmode -quit -projectPath ... -executeMethod CapKit.Editor.CapD37pxiBuilder.Build
    /// Built again from the URDF each time, nothing set by hand.
    /// </summary>
    public static class CapD37pxiBuilder
    {
        const string Urdf = "Assets/Machines/d37pxi/d37pxi24.urdf";
        public const string Output = "Assets/Resources/Prefab/d37pxi.prefab";
        const string Ns = "d37pxi";

        // The asset's URDF is older than PWRI's d37pxi_ros2 (fd0a90a): the joints by PWRI's names now, so joint_states
        // say what the real machine's say.
        static readonly Dictionary<string, string> Rename = new Dictionary<string, string>
        {
            { "blade_arm_joint", "lift_joint" }, { "blade_angle_joint", "angle_joint" }, { "blade_tilt_joint", "tilt_joint" },
        };

        // Links that only follow another joint in the URDF (mimic): the importer leaves them free, so their joints go and
        // their pieces stay where they are drawn.
        static readonly string[] Mimic =
        {
            "blade_arm_cylinder_link", "blade_arm_piston_link", "blade_angle_piston_left_link", "blade_angle_piston_right_link",
        };

        // [kg] about 9.25 t in all (D37PXi-24: about 9.4 t): the root that carries the wheels 1500 (as every machine's),
        // the body 5800, the blade's arm 700, its angle frame 50, the blade 900, six wheels 50 each.
        const float RootMass = 1500f, BodyMass = 5800f, ArmMass = 700f, AngleMass = 50f, BladeMass = 900f, WheelMass = 50f;

        // The tracks as wheels (URDF frame: x forward, y left, z up) from the body's mesh: shoes y 0.57-1.16, ground contact
        // x -1.20..+1.38, the track bottom at z -0.05.
        static readonly float[] WheelX = { 1.13f, 0.09f, -0.95f };
        const float WheelY = 0.865f, WheelRadius = 0.30f, TrackBottom = -0.05f, Suspension = 0.4f, SuspensionTarget = 0.5f;

        // The blade's cutting edge, its ends in blade_link (URDF frame, the mesh's lowest vertices).
        static readonly Vector3 EdgeLeftUrdf = new Vector3(0.39f, 1.607f, -0.301f);
        static readonly Vector3 EdgeRightUrdf = new Vector3(0.39f, -1.607f, -0.301f);

        public static void Build()
        {
            var settings = new ImportSettings
            {
                chosenAxis = ImportSettings.axisType.yAxis,
                convexMethod = ImportSettings.convexDecomposer.unity,
            };
            GameObject robot = null;
            var steps = UrdfRobotExtensions.Create(Path.GetFullPath(Urdf), settings, false);
            while (steps.MoveNext())
            {
                if (steps.Current != null) robot = steps.Current;
            }
            if (robot == null) throw new Exception("[CapD37pxi] the URDF import made nothing");
            try
            {
                Configure(robot);
                Directory.CreateDirectory(Path.GetDirectoryName(Output));
                PrefabUtility.SaveAsPrefabAsset(robot, Output, out bool saved);
                if (!saved) throw new Exception($"[CapD37pxi] {Output} not saved");
                Debug.Log($"[CapD37pxi] built {Output}");
                Report(robot);
            }
            finally
            {
                Object.DestroyImmediate(robot);
            }
            AssetDatabase.SaveAssets();
        }

        static void Configure(GameObject robot)
        {
            robot.name = Ns;
            robot.tag = "robot";
            var root = robot.transform;
            // The importer's own parts go: its keyboard controller, and UrdfRobot, whose collision exceptions need a
            // collider on every link (base_link has none) and threw at Start. Nothing of the dozer collides anyway.
            foreach (var c in robot.GetComponents<Unity.Robotics.UrdfImporter.Control.Controller>()) Object.DestroyImmediate(c);
            foreach (var c in robot.GetComponents<UrdfRobot>()) Object.DestroyImmediate(c);

            foreach (var uj in robot.GetComponentsInChildren<UrdfJoint>(true))
            {
                if (Rename.TryGetValue(uj.jointName, out var renamed)) uj.jointName = renamed;
            }
            foreach (var name in Mimic) RemoveJoint(Find(root, name).gameObject);

            var baseLink = Find(root, "base_link");
            var body = Body(Find(root, "body_link"));
            var arm = Body(Find(root, "blade_arm_link"));
            var angle = Body(Find(root, "blade_angle_link"));
            var blade = Body(Find(root, "blade_link"));

            body.mass = BodyMass;
            body.automaticCenterOfMass = false;
            body.centerOfMass = Unity(new Vector3(0.1f, 0f, 0.75f));      // low and a little ahead of the hub
            arm.mass = ArmMass;
            angle.mass = AngleMass;
            blade.mass = BladeMass;
            foreach (var ab in new[] { arm, angle, blade })
            {
                // per radian: the blade's weight (about 35 kN m on the lift) sagged 0.06 rad at 5e5; at 1e7 it is 3.5 mrad,
                // which CAP's machine control takes out. Damped near critical (the blade and its arm about 8700 kg m2).
                var d = ab.xDrive;
                d.stiffness = 1e7f;
                d.damping = 6e5f;
                d.forceLimit = 1e7f;
                ab.xDrive = d;
                ab.jointFriction = 0f;
                ab.angularDamping = 0.05f;
            }

            SetLayers(robot, 3);                    // nothing of the dozer collides; the wheels (below) are put back

            var rb = robot.AddComponent<Rigidbody>();
            rb.mass = RootMass;
            rb.drag = 0f;
            rb.angularDrag = 0.05f;
            var box = robot.AddComponent<BoxCollider>();
            box.center = new Vector3(0f, 0.5f, 0f);
            box.size = new Vector3(3.24f, 1f, 4.35f);
            var fixedJoint = robot.AddComponent<FixedJoint>();
            fixedJoint.connectedArticulationBody = body;
            fixedJoint.connectedMassScale = 50f;

            var wheels = new GameObject("wheels").transform;
            wheels.SetParent(baseLink, false);
            wheels.gameObject.layer = 3;
            var left = new List<GameObject>();
            var right = new List<GameObject>();
            string[] where = { "front", "middle", "rear" };
            for (int i = 0; i < WheelX.Length; i++)
            {
                left.Add(Wheel(wheels, $"left_{where[i]}_wheel_link", new Vector3(WheelX[i], WheelY, 0f), i == 1));
                right.Add(Wheel(wheels, $"right_{where[i]}_wheel_link", new Vector3(WheelX[i], -WheelY, 0f), i == 1));
            }

            var drive = robot.AddComponent<DiffDriveController>();
            drive.EnableVWBehaviorMode = true;
            drive.VWDecelFactor = 1.0;
            drive.VWRatioFactor = 0.9;
            drive.leftWheels = left;
            drive.rightWheels = right;
            drive.robotName = Ns;
            drive.TwistTopicName = $"{Ns}/cmd_vel";
            drive.OdomTopicName = $"{Ns}/odom_pose";
            drive.Com3TopicName = $"{Ns}/track_cmd";
            drive.childFrameName = $"{Ns}/base_link";
            drive.treadCollectionFactor = 3.0;
            drive.pGain = 600.0;
            drive.iGain = 100.0;
            drive.dGain = 10.0;
            drive.torqueLimit = 6000.0;
            drive.brakeTorque = 200000f;
            drive.maxLinearVelocity = 2.3;          // 8.5 km/h
            drive.maxAngularVelocity = 0.6;
            drive.publishMessageInterval = 0.02f;

            robot.AddComponent<EmergencyStop>();    // [robot_name]/emg_stop
            var joints = robot.AddComponent<JointStatePublisher>();
            joints.topicName = $"{Ns}/joint_states";
            joints.publishMessageInterval = 0.02f;
            joints.enableJointEffortSensor = false;

            var bladeDrive = robot.AddComponent<CapBladeDrive>();
            bladeDrive.joints = new List<CapBladeDrive.BladeJoint>
            {
                new CapBladeDrive.BladeJoint { name = "lift_joint", body = arm, topic = $"{Ns}/lift_joint/cmd", rateRadPerS = 0.30f, initialRad = -0.2f },
                new CapBladeDrive.BladeJoint { name = "angle_joint", body = angle, topic = $"{Ns}/angle_joint/cmd", rateRadPerS = 0.30f },
                new CapBladeDrive.BladeJoint { name = "tilt_joint", body = blade, topic = $"{Ns}/tilt_joint/cmd", rateRadPerS = 0.20f },
            };

            var truth = baseLink.gameObject.AddComponent<GroundTruthPublisher>();
            truth.topicName = $"{Ns}/global_pose";
            truth.childFrameName = $"{Ns}/base_link";
            truth.publishMessageInterval = 0.02f;
            var pose = baseLink.gameObject.AddComponent<PoseStampedPublisher>();
            pose.topicName = $"{Ns}/base_link/pose";
            pose.publishMessageInterval = 0.05f;

            var edgeLeft = Marker(blade.transform, "cap_edge_left", EdgeLeftUrdf);
            var edgeRight = Marker(blade.transform, "cap_edge_right", EdgeRightUrdf);

            // the blade in the soil (CapBlade.compute): cuts, carries, spills and lays down, before the soil's slides
            var soil = robot.AddComponent<CapDozerBlade>();
            soil.edgeLeft = edgeLeft;
            soil.edgeRight = edgeRight;
            soil.hub = baseLink;
            soil.bladeDrive = bladeDrive;
            soil.capacityM3 = 2.0f;                 // W H^2 / (2 tan 30) for 3.21 m x 0.88 m; SAE J1265 gives 1.9-2.1
            soil.widthM = 3.21f;
            soil.soilTopic = $"{Ns}/blade/soil";
        }

        static GameObject Wheel(Transform parent, string name, Vector3 urdf, bool middle)
        {
            var go = new GameObject(name);
            go.layer = 0;                           // the wheels alone meet the ground
            go.transform.SetParent(parent, false);
            // the wheel's centre at rest is its target point down the suspension: its bottom on the track's bottom
            var centre = new Vector3(urdf.x, urdf.y, TrackBottom + WheelRadius + Suspension * SuspensionTarget);
            go.transform.localPosition = Unity(centre);
            var w = go.AddComponent<WheelCollider>();
            w.radius = WheelRadius;
            w.mass = WheelMass;
            w.suspensionDistance = Suspension;
            w.suspensionSpring = new JointSpring { spring = 3e5f, damper = 3e4f, targetPosition = SuspensionTarget };
            var fwd = w.forwardFriction;
            fwd.stiffness = 1.5f;
            w.forwardFriction = fwd;
            var side = w.sidewaysFriction;
            side.stiffness = middle ? 1.0f : 0.25f;   // the middle pair holds the line, the ends let the tracks turn
            w.sidewaysFriction = side;
            return go;
        }

        static Transform Marker(Transform parent, string name, Vector3 urdf)
        {
            var go = new GameObject(name);
            go.layer = 3;
            go.transform.SetParent(parent, false);
            go.transform.localPosition = Unity(urdf);
            return go.transform;
        }

        static void RemoveJoint(GameObject go)
        {
            foreach (var c in go.GetComponents<UrdfJoint>()) Object.DestroyImmediate(c);
            foreach (var c in go.GetComponents<UrdfInertial>()) Object.DestroyImmediate(c);
            foreach (var c in go.GetComponents<ArticulationBody>()) Object.DestroyImmediate(c);
        }

        static ArticulationBody Body(Transform t)
        {
            var ab = t.GetComponent<ArticulationBody>();
            if (ab == null) throw new Exception($"[CapD37pxi] {t.name} has no ArticulationBody");
            return ab;
        }

        static Transform Find(Transform root, string name)
        {
            var t = root.GetComponentsInChildren<Transform>(true).FirstOrDefault(x => x.name == name);
            if (t == null) throw new Exception($"[CapD37pxi] no {name} in the import");
            return t;
        }

        static void SetLayers(GameObject go, int layer)
        {
            foreach (var t in go.GetComponentsInChildren<Transform>(true)) t.gameObject.layer = layer;
        }

        /// <summary>ROS (x forward, y left, z up) to Unity (x right, y up, z forward), as the URDF importer converts.</summary>
        static Vector3 Unity(Vector3 ros) => new Vector3(-ros.y, ros.z, ros.x);

        static Vector3 Ros(Vector3 unity) => new Vector3(unity.z, -unity.x, unity.y);

        /// <summary>The built dozer's frames in base_link (URDF axes), to check against the URDF by hand.</summary>
        static void Report(GameObject robot)
        {
            var root = robot.transform;
            var baseLink = Find(root, "base_link");
            foreach (var name in new[] { "body_link", "blade_arm_link", "blade_angle_link", "blade_link", "cap_edge_left",
                                         "cap_edge_right", "left_middle_wheel_link", "right_front_wheel_link" })
            {
                var t = Find(root, name);
                var p = Ros(baseLink.InverseTransformPoint(t.position));
                Debug.Log($"[CapD37pxi] {name} at ({p.x:F3}, {p.y:F3}, {p.z:F3}) in base_link");
            }
            var joints = robot.GetComponentsInChildren<UrdfJoint>(true)
                .Where(j => j.GetComponent<ArticulationBody>() != null && !(j is UrdfJointFixed))
                .Select(j => $"{j.jointName} [{j.GetComponent<ArticulationBody>().xDrive.lowerLimit:F1}, " +
                             $"{j.GetComponent<ArticulationBody>().xDrive.upperLimit:F1}] deg");
            Debug.Log($"[CapD37pxi] joints published: {string.Join(", ", joints)}");
        }
    }
}
