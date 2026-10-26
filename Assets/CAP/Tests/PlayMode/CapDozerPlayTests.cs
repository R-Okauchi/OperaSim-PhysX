#if UNITY_EDITOR
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using NUnit.Framework;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.TestTools;

namespace CapKit.Tests
{
    /// <summary>
    /// The D37PXi dozer (Assets/Resources/Prefab/d37pxi.prefab, CapD37pxiBuilder) on SimpleScene's open ground, without
    /// ROS: it stands without creeping, drives straight, stops, turns on the spot, backs up, and moves its blade at the
    /// cylinders' speed, the body steady while it does.
    ///   Unity -batchmode -projectPath ... -runTests -testPlatform PlayMode -testFilter CapDozer -testResults out.xml
    /// </summary>
    public class CapDozerPlayTests
    {
        private const string Scene = "Assets/Scenes/SimpleScene.unity";
        // Open, flat ground away from the scene's machines (ROS x 10, y -35: Unity (35, 0, 10)), facing ROS +x (Unity +z)
        private static readonly Vector3 Start = new Vector3(35f, 0.6f, 10f);
        // The soil test's lanes, clear of the drive test's (the terrain's changes outlive a scene's reload)
        private static readonly Vector3 LaneA = new Vector3(27f, 0.6f, 10f);
        private static readonly Vector3 LaneB = new Vector3(23f, 0.6f, 10f);
        // The edge above the flat ground: it cuts the piles, not the ground - 7 cm up with the kinematic tracks
        // (base_link 5 cm over the ground; with the WheelColliders it was 8 mm); lower, the front tracks run down into
        // what the blade cut and it digs deeper as it goes (a blade held to the body, not to the ground: CAP's machine
        // control holds it to the design). Tests that fill hold the edge at the ground themselves (DriveHolding)
        private const double EdgeAtGround = -0.02;

        private static void Log(string s) => Debug.Log("[CapDozerTest] " + s);

        private static void NoParticles(Scene scene, LoadSceneMode mode)
        {
            if (SoilParticleSettings.instance != null) SoilParticleSettings.instance.particles = false;
        }

        private static GameObject Spawn(Vector3 position, float yawDeg)
        {
            var prefab = Resources.Load<GameObject>("Prefab/d37pxi");
            Assert.IsNotNull(prefab, "no Resources/Prefab/d37pxi (CapD37pxiBuilder.Build)");
            var go = Object.Instantiate(prefab, position, Quaternion.Euler(0f, yawDeg, 0f));
            go.name = "d37pxi";                       // topics are named after the root ([robot_name])
            return go;
        }

        private static IEnumerator Seconds(float s)
        {
            float until = Time.time + s;
            while (Time.time < until) yield return new WaitForFixedUpdate();
        }

        private static float Yaw(Transform t) => t.eulerAngles.y;

        /// <summary>Unity (x right, y up, z forward) to ROS (x forward, y left, z up).</summary>
        private static Vector3 Ros(Vector3 u) => new Vector3(u.z, -u.x, u.y);

        private static float YawChange(float from, float to) => Mathf.DeltaAngle(from, to);

        // The rig's self masks (Assets/CAP/sensor_rig.json, scripts/export_sensor_rig.py: the contract's kit.self_mask in
        // the hub frame, ROS): the volumes CAP's kit takes for the machine's own returns
        [System.Serializable] private sealed class RigFile { public RigMachine[] machines; }
        [System.Serializable] private sealed class RigMachine { public string machine_id; public RigMask self_mask; public RigSensor[] sensors; }
        [System.Serializable] private sealed class RigMask { public RigHull hull; public RigBox undercarriage; public RigBox[] extra_boxes; }
        [System.Serializable] private sealed class RigHull { public float[] center_xy_m; public float radius_m, z_min_m, z_max_m; }
        [System.Serializable] private sealed class RigBox { public float[] center_xy_m; public float length_m, width_m, z_min_m, z_max_m; }
        [System.Serializable] private sealed class RigSensor { public string sensor_id, kind; public bool populated; public RigFov fov; }
        [System.Serializable] private sealed class RigFov { public float h_deg, v_min_deg, v_max_deg; }

        /// <summary>How far a hub-frame ROS point lies outside the mask's volumes [m] (0: inside one of them).</summary>
        private static float OutsideMask(Vector3 p, RigMask m)
        {
            float Band(float z, float lo, float hi) => Mathf.Max(0f, lo - z, z - hi);
            var h = m.hull;
            float best = Mathf.Max(Mathf.Max(0f, Mathf.Sqrt(Sq(p.x - h.center_xy_m[0]) + Sq(p.y - h.center_xy_m[1])) - h.radius_m),
                                   Band(p.z, h.z_min_m, h.z_max_m));
            var boxes = new List<RigBox>(m.extra_boxes ?? new RigBox[0]);
            if (m.undercarriage != null && m.undercarriage.length_m > 0f) boxes.Add(m.undercarriage);
            foreach (var b in boxes)
            {
                float out_ = Mathf.Max(Mathf.Max(0f, Mathf.Abs(p.x - b.center_xy_m[0]) - b.length_m / 2f),
                                       Mathf.Max(0f, Mathf.Abs(p.y - b.center_xy_m[1]) - b.width_m / 2f),
                                       Band(p.z, b.z_min_m, b.z_max_m));
                best = Mathf.Min(best, out_);
            }
            return best;
        }

        private static float Sq(float v) => v * v;

        [UnityTest, Timeout(900000)]
        public IEnumerator DozerOwnReturnsStayInItsSelfMask()
        {
            // What the kit's LiDARs meet of the dozer itself - its colliders: every raycast meets layer 3 - stands inside
            // the self mask the contract gives CAP's kit, or the guard reads it as an obstacle beside the machine (10-26:
            // the root's box collider, 0.575 m past the body's rear, stood a 1 m "object" at its rear corner and stopped
            // every pass). Each kit LiDAR's field of view cast at 1 deg from its frame, the blade raised, held, lowered and
            // tilted.
            LogAssert.ignoreFailingMessages = true;  // ROS-TCP has no endpoint here
            SceneManager.sceneLoaded += NoParticles;
            yield return CapTestScene.LoadFresh(Scene);
            SceneManager.sceneLoaded -= NoParticles;
            yield return new WaitForSeconds(1.0f);

            var rig = JsonUtility.FromJson<RigFile>(File.ReadAllText("Assets/CAP/sensor_rig.json"));
            RigMachine spec = null;
            foreach (var m in rig.machines) if (m.machine_id == "d37pxi") spec = m;
            Assert.IsNotNull(spec, "no d37pxi in sensor_rig.json (scripts/export_sensor_rig.py)");
            Assert.IsNotNull(spec.self_mask, "the rig has no self mask for d37pxi");
            Assert.Greater(spec.self_mask.hull.radius_m, 0f);

            var dozer = Spawn(Start, 0f);
            var blade = dozer.GetComponent<CapBladeDrive>();
            var hub = dozer.transform.Find("base_link");
            Assert.IsNotNull(hub);
            yield return Seconds(4f);                 // drops onto its tracks and settles
            var frames = new Dictionary<string, Transform>();
            foreach (var t in dozer.GetComponentsInChildren<Transform>(true)) frames[t.name] = t;

            double[,] poses = { { -0.2, 0.0 }, { -0.41, 0.0 }, { 0.18, 0.0 }, { -0.41, 0.13 }, { -0.41, -0.13 } };  // lift, tilt
            int selfHits = 0, rays = 0;
            float worst = 0f;
            string worstAt = "";
            for (int k = 0; k < poses.GetLength(0); k++)
            {
                blade.SetTarget("lift_joint", poses[k, 0]);
                blade.SetTarget("tilt_joint", poses[k, 1]);
                yield return Seconds(4f);                 // 0.3 rad/s: any change within 2.1 s
                int posed = 0;
                foreach (var s in spec.sensors)
                {
                    if (s.kind != "lidar" || !s.populated) continue;
                    Assert.IsTrue(frames.ContainsKey("kit_" + s.sensor_id), "no kit_" + s.sensor_id + " on the dozer");
                    var frame = frames["kit_" + s.sensor_id];
                    for (float el = s.fov.v_min_deg; el <= s.fov.v_max_deg + 1e-3f; el += 1f)
                    {
                        for (float az = 0f; az < 360f; az += 1f)
                        {
                            float ce = Mathf.Cos(el * Mathf.Deg2Rad), se = Mathf.Sin(el * Mathf.Deg2Rad);
                            float ca = Mathf.Cos(az * Mathf.Deg2Rad), sa = Mathf.Sin(az * Mathf.Deg2Rad);
                            // the sensor's ROS direction (x forward, y left, z up) in Unity's axes of its frame
                            var dir = frame.TransformDirection(new Vector3(-ce * sa, se, ce * ca));
                            rays++;
                            // as the LiDAR's RaycastCommand: the default layers, triggers as the project says
                            if (!Physics.Raycast(frame.position, dir, out RaycastHit hit, 40f)) continue;
                            if (!hit.collider.transform.IsChildOf(dozer.transform)) continue;
                            selfHits++;
                            posed++;
                            Vector3 u = hub.InverseTransformPoint(hit.point);
                            var ros = new Vector3(u.z, -u.x, u.y);
                            float outside = OutsideMask(ros, spec.self_mask);
                            if (outside > worst)
                            {
                                worst = outside;
                                worstAt = $"{s.sensor_id} lift {poses[k, 0]:F2} tilt {poses[k, 1]:F2}: {hit.collider.name} at " +
                                          $"({ros.x:F3}, {ros.y:F3}, {ros.z:F3}) in the hub";
                            }
                        }
                    }
                }
                Log($"M: lift {poses[k, 0]:F2} tilt {poses[k, 1]:F2}: {posed} of the dozer's own returns");
            }
            Log($"M: {rays} rays, {selfHits} met the dozer itself; the farthest outside its self mask {worst:F3} m ({worstAt})");
            Assert.Greater(selfHits, 1000, "the kit's LiDARs never met the dozer: the check checks nothing");
            Assert.LessOrEqual(worst, 0.0f, "a return of the dozer's own outside its self mask: " + worstAt);
            Object.Destroy(dozer);
        }

        [UnityTest, Timeout(900000)]
        public IEnumerator DozerDrivesAndMovesItsBlade()
        {
            LogAssert.ignoreFailingMessages = true;  // ROS-TCP has no endpoint here
            SceneManager.sceneLoaded += NoParticles;
            yield return CapTestScene.LoadFresh(Scene);
            SceneManager.sceneLoaded -= NoParticles;
            yield return new WaitForSeconds(1.0f);

            var dozer = Spawn(Start, 0f);
            var drive = dozer.GetComponent<CapTrackKinematic>();
            var blade = dozer.GetComponent<CapBladeDrive>();
            var hub = dozer.transform.Find("base_link");
            Assert.IsNotNull(drive);
            Assert.IsNotNull(blade);
            Assert.IsNotNull(hub);
            yield return Seconds(4f);                 // drops onto its tracks and settles

            // ---- A: it stands without creeping
            Vector3 p0 = hub.position;
            float yaw0 = Yaw(hub);
            yield return Seconds(20f);
            float creep = Vector3.Distance(new Vector3(p0.x, 0f, p0.z), new Vector3(hub.position.x, 0f, hub.position.z));
            Log($"A: stood 20 s: crept {creep * 1000f:F1} mm, yaw {YawChange(yaw0, Yaw(hub)):F2} deg, height {hub.position.y:F3}");
            Assert.Less(creep, 0.001f, "creeps while standing");

            // ---- B: 15 m straight at 0.8 m/s
            p0 = hub.position;
            yaw0 = Yaw(hub);
            Vector3 ahead = hub.forward;
            ahead.y = 0f;
            ahead.Normalize();
            drive.SetCommand(0.8, 0.0);
            float t0 = Time.time;
            float along = 0f;
            float worstSide = 0f;
            while (along < 15f && Time.time - t0 < 40f)
            {
                yield return new WaitForFixedUpdate();
                Vector3 d = hub.position - p0;
                d.y = 0f;
                along = Vector3.Dot(d, ahead);
                worstSide = Mathf.Max(worstSide, Vector3.Cross(ahead, d).magnitude);
            }
            float driveTime = Time.time - t0;
            float yawDrift = YawChange(yaw0, Yaw(hub));
            Log($"B: {along:F2} m in {driveTime:F1} s ({along / driveTime:F2} m/s), off the line {worstSide:F3} m, yaw {yawDrift:F2} deg");
            Assert.GreaterOrEqual(along, 15f, "never covered 15 m");
            Assert.Less(worstSide, 0.10f, "drifts off its line");
            Assert.Less(Mathf.Abs(yawDrift), 1.0f, "turns while going straight");

            // ---- C: it stops
            drive.SetCommand(0.0, 0.0);
            Vector3 atStop = hub.position;
            float tStop = Time.time;
            Vector3 last = hub.position;
            float stillSince = -1f;
            while (Time.time - tStop < 10f)
            {
                yield return new WaitForFixedUpdate();
                float speed = (hub.position - last).magnitude / Time.fixedDeltaTime;
                last = hub.position;
                if (speed < 0.02f) { if (stillSince < 0f) stillSince = Time.time; }
                else stillSince = -1f;
                if (stillSince > 0f && Time.time - stillSince > 0.5f) break;
            }
            float stopDistance = Vector3.Distance(atStop, hub.position);
            Log($"C: stopped in {stillSince - tStop:F2} s and {stopDistance:F2} m");
            Assert.Greater(stillSince, 0f, "never stops");
            Assert.Less(stopDistance, 1.0f, "too far to stop");

            // ---- D: 90 degrees on the spot
            yaw0 = Yaw(hub);
            p0 = hub.position;
            drive.SetCommand(0.0, 0.6);
            t0 = Time.time;
            while (Mathf.Abs(YawChange(yaw0, Yaw(hub))) < 90f && Time.time - t0 < 20f) yield return new WaitForFixedUpdate();
            float pivotTime = Time.time - t0;
            drive.SetCommand(0.0, 0.0);
            yield return Seconds(2f);
            float pivotMove = Vector3.Distance(new Vector3(p0.x, 0f, p0.z), new Vector3(hub.position.x, 0f, hub.position.z));
            Log($"D: 90 deg in {pivotTime:F1} s, the hub moved {pivotMove:F2} m, ended at {YawChange(yaw0, Yaw(hub)):F1} deg");
            Assert.Less(pivotTime, 6f, "turns too slowly");
            Assert.Less(pivotMove, 0.5f, "the pivot wanders");

            // ---- E: it backs up
            p0 = hub.position;
            ahead = hub.forward;
            ahead.y = 0f;
            ahead.Normalize();
            drive.SetCommand(-0.8, 0.0);
            yield return Seconds(4f);
            drive.SetCommand(0.0, 0.0);
            yield return Seconds(2f);
            float back = Vector3.Dot(hub.position - p0, ahead);
            Log($"E: backed {back:F2} m in 4 s");
            Assert.Less(back, -1.5f, "does not back up");

            // ---- F: the blade at the cylinders' speed, the body steady
            float y0 = hub.position.y;
            float worstBob = 0f;
            blade.SetTarget("lift_joint", -0.41);     // up
            t0 = Time.time;
            while (blade.Position("lift_joint") > -0.40 && Time.time - t0 < 5f)
            {
                yield return new WaitForFixedUpdate();
                worstBob = Mathf.Max(worstBob, Mathf.Abs(hub.position.y - y0));
            }
            float upTime = Time.time - t0;
            blade.SetTarget("lift_joint", 0.18);      // down (through the ground: the blade collides with nothing)
            t0 = Time.time;
            while (blade.Position("lift_joint") < 0.17 && Time.time - t0 < 5f)
            {
                yield return new WaitForFixedUpdate();
                worstBob = Mathf.Max(worstBob, Mathf.Abs(hub.position.y - y0));
            }
            float downTime = Time.time - t0;
            blade.SetTarget("angle_joint", 0.4);
            blade.SetTarget("tilt_joint", 0.13);
            yield return Seconds(2.5f);
            Log($"F: lift up {upTime:F2} s, down {downTime:F2} s, angle {blade.Position("angle_joint"):F3} rad, " +
                $"tilt {blade.Position("tilt_joint"):F3} rad, the hub moved {worstBob * 1000f:F1} mm up or down");
            Assert.Less(upTime, 2.0f);
            Assert.Less(downTime, 3.0f);
            Assert.AreEqual(0.4, blade.Position("angle_joint"), 0.02);
            Assert.AreEqual(0.13, blade.Position("tilt_joint"), 0.02);
            Assert.Less(worstBob, 0.05f, "the body bobs with the blade");

            // ---- G: the cutting edge at five poses, in base_link (ROS axes): what CAP's blade kinematics must give
            var left = dozer.transform.Find("base_link").GetComponentsInChildren<Transform>(true);
            Transform edgeL = System.Array.Find(left, t => t.name == "cap_edge_left");
            Transform edgeR = System.Array.Find(left, t => t.name == "cap_edge_right");
            Assert.IsNotNull(edgeL);
            Assert.IsNotNull(edgeR);
            double[,] poses = { { -0.40, 0.0, 0.0 }, { -0.20, 0.0, 0.0 }, { 0.0, 0.0, 0.0 }, { 0.15, 0.0, 0.0 },
                                { -0.10, 0.30, 0.10 } };
            for (int i = 0; i < poses.GetLength(0); i++)
            {
                blade.SetTarget("lift_joint", poses[i, 0]);
                blade.SetTarget("angle_joint", poses[i, 1]);
                blade.SetTarget("tilt_joint", poses[i, 2]);
                yield return Seconds(3.5f);
                Vector3 l = Ros(hub.InverseTransformPoint(edgeL.position));
                Vector3 r = Ros(hub.InverseTransformPoint(edgeR.position));
                Log($"G: lift {blade.Position("lift_joint"):F4} angle {blade.Position("angle_joint"):F4} tilt " +
                    $"{blade.Position("tilt_joint"):F4} left ({l.x:F4}, {l.y:F4}, {l.z:F4}) right ({r.x:F4}, {r.y:F4}, {r.z:F4})");
            }
            Object.Destroy(dozer);
        }

        [UnityTest, Timeout(600000)]
        public IEnumerator DozerClimbsWhatItsBladeLeaves()
        {
            // A fill's edge (0.20 m, a layer's height) across its lane and a 30-degree heap's foot: the tracks go over
            // both at its pace (live 10-25: WheelColliders stopped at a 0.15 m fill's edge, slipping)
            LogAssert.ignoreFailingMessages = true;
            SceneManager.sceneLoaded += NoParticles;
            yield return CapTestScene.LoadFresh(Scene);
            SceneManager.sceneLoaded -= NoParticles;
            yield return new WaitForSeconds(1.0f);
            var soil = SoilParticleSettings.instance;
            var dozer = Spawn(LaneB + new Vector3(-8f, 0f, 0f), 0f);
            var drive = dozer.GetComponent<CapTrackKinematic>();
            var hub = dozer.transform.Find("base_link");
            Assert.IsNotNull(drive, "no CapTrackKinematic on the dozer");
            yield return Seconds(2f);
            Vector3 ahead = hub.forward;
            ahead.y = 0f;
            ahead.Normalize();
            float ground0 = hub.position.y;
            // the fill: 4 m along, 4 m across, 0.20 m high, 6 m ahead - its edge a vertical step of the heightmap
            Vector3 start = hub.position + ahead * 6f;
            var at = new System.Collections.Generic.List<Vector2Int>();
            var w = new System.Collections.Generic.List<float>();
            Vector3 t0 = soil.ToTexel(start - ahead * 0f + Vector3.Cross(Vector3.up, ahead) * -2f);
            Vector3 t1 = soil.ToTexel(start + ahead * 4f + Vector3.Cross(Vector3.up, ahead) * 2f);
            for (int y = Mathf.FloorToInt(Mathf.Min(t0.y, t1.y)); y <= Mathf.CeilToInt(Mathf.Max(t0.y, t1.y)); y++)
            for (int x = Mathf.FloorToInt(Mathf.Min(t0.x, t1.x)); x <= Mathf.CeilToInt(Mathf.Max(t0.x, t1.x)); x++)
            {
                at.Add(new Vector2Int(x, y));
                w.Add(1f);
            }
            float cell = Mathf.Sqrt(soil.TexelArea);
            CapSoilStamp.Stamp(soil, at, w, at.Count * soil.TexelArea * 0.20);
            CapSoilStamp.Cone(soil, start + ahead * 9f, 1.5, 30f);       // a heap's foot beyond it
            yield return Seconds(0.5f);
            Vector3 p0 = hub.position;
            drive.SetCommand(0.7, 0.0);
            float tStart = Time.time, topSpeed = 0f, slowest = float.PositiveInfinity, highest = 0f;
            Vector3 last = hub.position;
            while (Vector3.Dot(hub.position - p0, ahead) < 14f && Time.time - tStart < 40f)
            {
                yield return new WaitForFixedUpdate();
                float v = Vector3.Dot(hub.position - last, ahead) / Time.fixedDeltaTime;
                last = hub.position;
                if (Time.time - tStart > 3f) slowest = Mathf.Min(slowest, v);
                topSpeed = Mathf.Max(topSpeed, v);
                highest = Mathf.Max(highest, hub.position.y - ground0);
            }
            drive.SetCommand(0.0, 0.0);
            float covered = Vector3.Dot(hub.position - p0, ahead);
            Log($"K: over a 0.20 m fill's edge and a heap's foot: {covered:F1} m in {Time.time - tStart:F1} s, slowest " +
                $"{slowest:F2} m/s, top {topSpeed:F2} m/s, the hub up to {highest:F3} m (cell {cell:F3} m)");
            Assert.GreaterOrEqual(covered, 14f, "stopped at the fill or the heap");
            Assert.Greater(slowest, 0.5f, "held up on the way");
            Assert.Greater(highest, 0.15f, "never climbed onto the fill");
            Object.Destroy(dozer);
        }

        [UnityTest, Timeout(600000)]
        public IEnumerator DozerMachineControlHoldsTheEdgeToTheDesign()
        {
            // The machine's own MC (CapBladeMC) engaged at a design 0.08 m above flat ground: over a 1.5 m3 heap and a
            // 0.15 m fill's edge the cut is at the design (the edge held on the machine, every physics step)
            LogAssert.ignoreFailingMessages = true;
            SceneManager.sceneLoaded += NoParticles;
            yield return CapTestScene.LoadFresh(Scene);
            SceneManager.sceneLoaded -= NoParticles;
            yield return new WaitForSeconds(1.0f);
            var soil = SoilParticleSettings.instance;
            var terrain = soil.GetComponent<Terrain>();
            var dozer = Spawn(LaneB + new Vector3(0f, 0f, 20f), 0f);
            var drive = dozer.GetComponent<CapTrackKinematic>();
            var mc = dozer.GetComponent<CapBladeMC>();
            var hub = dozer.transform.Find("base_link");
            Assert.IsNotNull(mc, "no CapBladeMC on the dozer");
            yield return Seconds(2f);
            Vector3 ahead = hub.forward;
            ahead.y = 0f;
            ahead.Normalize();
            float ground = terrain.SampleHeight(hub.position) + terrain.transform.position.y;
            Vector3 heap = hub.position + ahead * 8f;
            heap.y = 0f;
            CapSoilStamp.Cone(soil, heap, 1.5, 30f);
            // a fill's edge across the lane further on: 0.15 m up for 3 m
            var at = new System.Collections.Generic.List<Vector2Int>();
            var w = new System.Collections.Generic.List<float>();
            Vector3 side = Vector3.Cross(Vector3.up, ahead);
            Vector3 f0 = soil.ToTexel(hub.position + ahead * 12f - side * 2.5f), f1 = soil.ToTexel(hub.position + ahead * 15f + side * 2.5f);
            for (int y = Mathf.FloorToInt(Mathf.Min(f0.y, f1.y)); y <= Mathf.CeilToInt(Mathf.Max(f0.y, f1.y)); y++)
            for (int x = Mathf.FloorToInt(Mathf.Min(f0.x, f1.x)); x <= Mathf.CeilToInt(Mathf.Max(f0.x, f1.x)); x++)
            {
                at.Add(new Vector2Int(x, y));
                w.Add(1f);
            }
            CapSoilStamp.Stamp(soil, at, w, at.Count * soil.TexelArea * 0.15);
            yield return Seconds(0.5f);
            double design = ground + 0.08;
            Vector3 u0 = hub.position;
            mc.Engage(design, 0.0, 0.0, 0.0, 0.0, 0.0);         // level (the ROS frame's terms only matter for a slope)
            yield return Seconds(1.5f);
            Vector3 p0 = hub.position;
            drive.SetCommand(0.6, 0.0);
            float t0 = Time.time;
            while (Vector3.Dot(hub.position - p0, ahead) < 16f && Time.time - t0 < 50f) yield return new WaitForFixedUpdate();
            drive.SetCommand(0.0, 0.0);
            yield return Seconds(1.5f);
            mc.Disengage();
            // the swath's middle where the heap stood and where the fill's top was: at the design
            float worst = 0f;
            int n = 0;
            foreach (float s in new[] { 7.0f, 7.5f, 8.0f, 8.5f, 9.0f, 13.0f, 13.5f, 14.0f })
            {
                Vector3 p = u0 + ahead * s;
                float h = terrain.SampleHeight(p) + terrain.transform.position.y;
                worst = Mathf.Max(worst, Mathf.Abs((float)(h - design)));
                n++;
            }
            Log($"M: the machine's MC at {design:F3} m: the swath within {worst * 100f:F1} cm of the design at {n} " +
                $"stations; edge error {mc.EdgeErrorM * 100.0:F1} cm, {mc.Ticks} ticks, {mc.OutOfReach} out of reach");
            Assert.Greater(mc.Ticks, 100, "the MC never ran");
            Assert.Less(worst, 0.03f, "the cut is off the design");
            Object.Destroy(dozer);
        }

        [UnityTest, Timeout(600000)]
        public IEnumerator DozerBladeNeverDigsWithANegativeLoad()
        {
            // A load a rounding below zero, the edge 0.10 m above flat ground (where the blade fills): nothing is
            // filled and nothing dug (10-25: a negative fill fraction lowered the ground, its count wrapped round and
            // the blade dug to the terrain's bottom)
            LogAssert.ignoreFailingMessages = true;
            SceneManager.sceneLoaded += NoParticles;
            yield return CapTestScene.LoadFresh(Scene);
            SceneManager.sceneLoaded -= NoParticles;
            yield return new WaitForSeconds(1.0f);
            var soil = SoilParticleSettings.instance;
            var dozer = Spawn(LaneA + new Vector3(0f, 0f, 20f), 0f);
            var drive = dozer.GetComponent<CapTrackKinematic>();
            var bladeDrive = dozer.GetComponent<CapBladeDrive>();
            var blade = dozer.GetComponent<CapDozerBlade>();
            var hub = dozer.transform.Find("base_link");
            yield return Seconds(2f);
            float ground = soil.GetComponent<Terrain>().SampleHeight(hub.position) + soil.transform.position.y;
            Vector3 middle = hub.position + hub.forward * 5f;
            double before = soil.WorkingVolume(middle, 8f);
            blade.Preload(-1e-6);
            yield return DriveHolding(drive, hub, bladeDrive, blade, ground + 0.10f, 5f, 0.7f);
            yield return Seconds(1.5f);
            double after = soil.WorkingVolume(middle, 8f);
            Log($"L: a load of -1e-6 m3 over 5 m with the edge 0.10 m up: filled {blade.FilledM3:E2} m3, cut " +
                $"{blade.CutM3:E2} m3, the ground {after - before:E2} m3");
            Assert.Less(blade.FilledM3, 1e-3, "filled with no load");
            Assert.AreEqual(before, after, 1e-3, "the ground changed under an empty blade");
            Object.Destroy(dozer);
        }

        /// <summary>Drive the dozer along its heading for a distance at a speed, its edge held at a height of the world
        /// (CAP's machine control does this on the machine: a blade held to the body digs in where the front of the
        /// tracks dips into what it cut - the tracks now follow the ground as the twin's do).</summary>
        private static IEnumerator DriveHolding(CapTrackKinematic drive, Transform hub, CapBladeDrive bladeDrive,
                                                CapDozerBlade blade, float edgeY, float metres, float speed,
                                                System.Action each = null)
        {
            Vector3 p0 = hub.position;
            Vector3 ahead = hub.forward;
            ahead.y = 0f;
            ahead.Normalize();
            drive.SetCommand(speed, 0.0);
            float t0 = Time.time;
            while (Vector3.Dot(hub.position - p0, ahead) < metres && Time.time - t0 < metres / Mathf.Max(0.1f, speed) + 20f)
            {
                float edge = Mathf.Min(blade.edgeLeft.position.y, blade.edgeRight.position.y);
                // lift down positive: about -1.98 m of edge per radian (the contract's kinematics)
                double lift = bladeDrive.Position("lift_joint") + (edge - edgeY) / 1.98;
                bladeDrive.SetTarget("lift_joint", System.Math.Max(-0.41, System.Math.Min(0.187, lift)));
                yield return new WaitForFixedUpdate();
                each?.Invoke();
            }
            drive.SetCommand(0.0, 0.0);
        }

        // ---- the soil's checks: the working heightmap's floats (the terrain's own copy is 16-bit, 3 mm a step)

        /// <summary>The working heightmap's vertices whose lane coordinates - metres along `ahead` from `origin`, and
        /// across toward `side` - fall in [s0, s1] x [l0, l1]: their world points (y their height) and texture
        /// heights.</summary>
        private static List<(Vector3 at, float tex)> LaneVertices(SoilParticleSettings soil, Vector3 origin,
                                                                   Vector3 ahead, Vector3 side, float s0, float s1,
                                                                   float l0, float l1)
        {
            float tx0 = float.PositiveInfinity, ty0 = float.PositiveInfinity;
            float tx1 = float.NegativeInfinity, ty1 = float.NegativeInfinity;
            foreach (var c in new[] { origin + ahead * s0 + side * l0, origin + ahead * s0 + side * l1,
                                      origin + ahead * s1 + side * l0, origin + ahead * s1 + side * l1 })
            {
                var t = soil.ToTexel(c);
                tx0 = Mathf.Min(tx0, t.x);
                ty0 = Mathf.Min(ty0, t.y);
                tx1 = Mathf.Max(tx1, t.x);
                ty1 = Mathf.Max(ty1, t.y);
            }
            var size = soil.HeightmapSize;
            int x0 = Mathf.Clamp(Mathf.FloorToInt(tx0), 0, size.x - 1), x1 = Mathf.Clamp(Mathf.CeilToInt(tx1), 0, size.x - 1);
            int y0 = Mathf.Clamp(Mathf.FloorToInt(ty0), 0, size.y - 1), y1 = Mathf.Clamp(Mathf.CeilToInt(ty1), 0, size.y - 1);
            int w = x1 - x0 + 1, h = y1 - y0 + 1;
            var prev = RenderTexture.active;
            RenderTexture.active = soil.WorkingHeightmap;
            var tex = new Texture2D(w, h, TextureFormat.RGBAFloat, false, true);
            tex.ReadPixels(new Rect(x0, y0, w, h), 0, 0, false);
            tex.Apply(false, false);
            RenderTexture.active = prev;
            var px = tex.GetPixels();
            Object.Destroy(tex);
            var list = new List<(Vector3, float)>();
            for (int j = 0; j < h; j++)
            for (int i = 0; i < w; i++)
            {
                float th = px[j * w + i].r;
                Vector3 p = soil.FromTexel(x0 + i, y0 + j, th);
                Vector3 d = p - origin;
                float s = Vector3.Dot(d, ahead), l = Vector3.Dot(d, side);
                if (s < s0 || s > s1 || l < l0 || l > l1) continue;
                list.Add((p, th));
            }
            return list;
        }

        /// <summary>The soil over a lane's rectangle [m3 above the terrain's base].</summary>
        private static double LaneVolume(SoilParticleSettings soil, Vector3 origin, Vector3 ahead, Vector3 side,
                                         float s0, float s1, float l0, float l1)
        {
            double sum = 0.0;
            foreach (var (_, th) in LaneVertices(soil, origin, ahead, side, s0, s1, l0, l1)) sum += th;
            return sum * soil.TextureUnitM * soil.TexelArea;
        }

        /// <summary>The lowest and the highest vertex of a lane's rectangle [m, world].</summary>
        private static (float low, float high) LaneRange(SoilParticleSettings soil, Vector3 origin, Vector3 ahead,
                                                         Vector3 side, float s0, float s1, float l0, float l1)
        {
            float lo = float.PositiveInfinity, hi = float.NegativeInfinity;
            foreach (var (p, _) in LaneVertices(soil, origin, ahead, side, s0, s1, l0, l1))
            {
                lo = Mathf.Min(lo, p.y);
                hi = Mathf.Max(hi, p.y);
            }
            return (lo, hi);
        }

        /// <summary>The whole working heightmap's soil [m3 above the terrain's base].</summary>
        private static double MapVolume(SoilParticleSettings soil)
        {
            var terrain = soil.GetComponent<Terrain>();
            Vector3 centre = terrain.transform.position + 0.5f * terrain.terrainData.size;
            return soil.WorkingVolume(centre, 0.5f * Mathf.Max(terrain.terrainData.size.x, terrain.terrainData.size.z) + 1f);
        }

        private static Vector3 Level(Vector3 v)
        {
            v.y = 0f;
            return v.normalized;
        }

        /// <summary>The cutting edge's heights [m, world] while its middle was within [s0, s1] along a lane.</summary>
        private sealed class EdgeWatch
        {
            public float Low = float.PositiveInfinity, High = float.NegativeInfinity;
            public int Steps;
            private readonly CapDozerBlade blade;
            private readonly Vector3 origin, ahead;
            private readonly float s0, s1;

            public EdgeWatch(CapDozerBlade blade, Vector3 origin, Vector3 ahead, float s0, float s1)
            {
                this.blade = blade;
                this.origin = origin;
                this.ahead = ahead;
                this.s0 = s0;
                this.s1 = s1;
            }

            public void Step()
            {
                Vector3 a = blade.edgeLeft.position, b = blade.edgeRight.position;
                float s = Vector3.Dot(0.5f * (a + b) - origin, ahead);
                if (s < s0 || s > s1) return;
                Low = Mathf.Min(Low, Mathf.Min(a.y, b.y));
                High = Mathf.Max(High, Mathf.Max(a.y, b.y));
                Steps++;
            }
        }

        /// <summary>A lane's record for CAP's twin to run the same push on (its lane comparison script):
        /// the working heights over the lane before and after (world y, the vertices' grid), every physics step's
        /// commands, the hub's and the edge's poses (ROS axes) and the blade's joints and load, the ledger at the end.
        /// Written to $CAP_LANE_OUT (a directory) or Logs/cap_lanes/ under the project.</summary>
        private sealed class LaneRecord
        {
            private readonly SoilParticleSettings soil;
            private readonly int x0, y0, w, h;
            private float[] before;
            private readonly StringBuilder steps = new StringBuilder();
            private int n;
            private static readonly CultureInfo Inv = CultureInfo.InvariantCulture;

            public LaneRecord(SoilParticleSettings soil, Vector3 origin, Vector3 ahead, Vector3 side, float s0,
                              float s1, float l0, float l1)
            {
                this.soil = soil;
                float tx0 = float.PositiveInfinity, ty0 = float.PositiveInfinity;
                float tx1 = float.NegativeInfinity, ty1 = float.NegativeInfinity;
                foreach (var c in new[] { origin + ahead * s0 + side * l0, origin + ahead * s0 + side * l1,
                                          origin + ahead * s1 + side * l0, origin + ahead * s1 + side * l1 })
                {
                    var t = soil.ToTexel(c);
                    tx0 = Mathf.Min(tx0, t.x);
                    ty0 = Mathf.Min(ty0, t.y);
                    tx1 = Mathf.Max(tx1, t.x);
                    ty1 = Mathf.Max(ty1, t.y);
                }
                var size = soil.HeightmapSize;
                x0 = Mathf.Clamp(Mathf.FloorToInt(tx0), 0, size.x - 1);
                y0 = Mathf.Clamp(Mathf.FloorToInt(ty0), 0, size.y - 1);
                w = Mathf.Clamp(Mathf.CeilToInt(tx1), 0, size.x - 1) - x0 + 1;
                h = Mathf.Clamp(Mathf.CeilToInt(ty1), 0, size.y - 1) - y0 + 1;
            }

            private float[] Read()
            {
                var prev = RenderTexture.active;
                RenderTexture.active = soil.WorkingHeightmap;
                var tex = new Texture2D(w, h, TextureFormat.RGBAFloat, false, true);
                tex.ReadPixels(new Rect(x0, y0, w, h), 0, 0, false);
                tex.Apply(false, false);
                RenderTexture.active = prev;
                var px = tex.GetPixels();
                Object.Destroy(tex);
                var out_ = new float[w * h];
                for (int j = 0; j < h; j++)
                for (int i = 0; i < w; i++)
                    out_[j * w + i] = soil.FromTexel(x0 + i, y0 + j, px[j * w + i].r).y;
                return out_;
            }

            public void Begin() => before = Read();

            private static string F(double v) => v.ToString("R", Inv);

            private static void Vec(StringBuilder sb, Vector3 u)
            {
                Vector3 r = Ros(u);
                sb.Append(F(r.x)).Append(',').Append(F(r.y)).Append(',').Append(F(r.z));
            }

            /// <summary>One physics step: [t, v_cmd, w_cmd, speed, hub xyz, hub forward xyz, hub up xyz, lift, angle,
            /// tilt, edge left xyz, edge right xyz, load] (ROS axes).</summary>
            public void Step(CapTrackKinematic drive, Transform hub, CapBladeDrive bladeDrive, CapDozerBlade blade)
            {
                if (n++ > 0) steps.Append(',');
                steps.Append('[').Append(F(Time.time)).Append(',').Append(F(drive.LinearCmd)).Append(',')
                     .Append(F(drive.AngularCmd)).Append(',').Append(F(drive.Speed)).Append(',');
                Vec(steps, hub.position);
                steps.Append(',');
                Vec(steps, hub.forward);
                steps.Append(',');
                Vec(steps, hub.up);
                steps.Append(',').Append(F(bladeDrive.Position("lift_joint"))).Append(',')
                     .Append(F(bladeDrive.Position("angle_joint"))).Append(',')
                     .Append(F(bladeDrive.Position("tilt_joint"))).Append(',');
                Vec(steps, blade.edgeLeft.position);
                steps.Append(',');
                Vec(steps, blade.edgeRight.position);
                steps.Append(',').Append(F(blade.LoadM3)).Append(']');
            }

            private static void Heights(StringBuilder sb, float[] a)
            {
                sb.Append('[');
                for (int i = 0; i < a.Length; i++)
                {
                    if (i > 0) sb.Append(',');
                    sb.Append(a[i].ToString("F6", Inv));
                }
                sb.Append(']');
            }

            public string Write(string name, CapDozerBlade blade, string about)
            {
                float[] after = Read();
                var terrain = soil.GetComponent<Terrain>();
                Vector3 v0 = soil.FromTexel(x0, y0, 0f);
                var sb = new StringBuilder();
                sb.Append("{\"name\":\"").Append(name).Append("\",\"about\":\"").Append(about).Append("\",");
                sb.Append("\"cell_m\":").Append(F(Mathf.Sqrt(soil.TexelArea))).Append(',');
                // the vertices' grid: column i along Unity x (ROS -y), row j along Unity z (ROS +x)
                sb.Append("\"unity_x0\":").Append(F(v0.x)).Append(",\"unity_z0\":").Append(F(v0.z)).Append(',');
                sb.Append("\"nx\":").Append(w).Append(",\"ny\":").Append(h).Append(',');
                sb.Append("\"terrain_base_y\":").Append(F(terrain.transform.position.y)).Append(',');
                sb.Append("\"terrain_size_y\":").Append(F(terrain.terrainData.size.y)).Append(',');
                sb.Append("\"fixed_dt\":").Append(F(Time.fixedDeltaTime)).Append(',');
                sb.Append("\"ledger\":{\"cut\":").Append(F(blade.CutM3)).Append(",\"filled\":").Append(F(blade.FilledM3))
                  .Append(",\"spilled\":").Append(F(blade.SpilledM3)).Append(",\"rest\":").Append(F(blade.RestM3))
                  .Append(",\"load\":").Append(F(blade.LoadM3)).Append("},");
                sb.Append("\"before\":");
                Heights(sb, before);
                sb.Append(",\"after\":");
                Heights(sb, after);
                sb.Append(",\"steps\":[").Append(steps).Append("]}");
                string dir = System.Environment.GetEnvironmentVariable("CAP_LANE_OUT");
                if (string.IsNullOrEmpty(dir)) dir = Path.Combine(Application.dataPath, "..", "Logs", "cap_lanes");
                Directory.CreateDirectory(dir);
                string path = Path.Combine(dir, name + ".json");
                File.WriteAllText(path, sb.ToString());
                return path;
            }
        }


        [UnityTest, Timeout(900000)]
        public IEnumerator DozerBladeMovesSoilAndKeepsIt()
        {
            LogAssert.ignoreFailingMessages = true;
            SceneManager.sceneLoaded += NoParticles;
            yield return CapTestScene.LoadFresh(Scene);
            SceneManager.sceneLoaded -= NoParticles;
            yield return new WaitForSeconds(1.0f);
            var soil = SoilParticleSettings.instance;
            Assert.IsNotNull(soil);
            Assert.IsFalse(soil.particles);

            // ---- H: six dumped piles add their volume
            var pilesAt = new Vector3(20f, 0f, -38f);                 // open ground away from the dozer's lane
            double v0 = soil.WorkingVolume(pilesAt, 12f);
            double laid = 0.0;
            for (int i = 0; i < 6; i++)
                laid += CapSoilStamp.Cone(soil, pilesAt + new Vector3(-7.5f + 5f * (i % 3), 0f, -2.5f + 5f * (i / 3)), 5.5, 34f);
            yield return Seconds(0.5f);
            double v1 = soil.WorkingVolume(pilesAt, 12f);
            Log($"H: six piles laid {laid:F4} m3, the terrain gained {v1 - v0:F4} m3");
            Assert.AreEqual(33.0, laid, 0.001);
            Assert.AreEqual(33.0, v1 - v0, 1e-3, "the terrain did not gain what the piles laid (to 1e-3 m3)");

            // ---- I: the blade at the ground pushes a 1.5 m3 pile 12 m: soil kept, the pile gone from its place
            var dozer = Spawn(LaneA, 0f);
            var drive = dozer.GetComponent<CapTrackKinematic>();
            var bladeDrive = dozer.GetComponent<CapBladeDrive>();
            var blade = dozer.GetComponent<CapDozerBlade>();
            var hub = dozer.transform.Find("base_link");
            Assert.IsNotNull(blade, "no CapDozerBlade on the dozer");
            yield return Seconds(4f);
            Vector3 ahead = hub.forward;
            ahead.y = 0f;
            ahead.Normalize();
            Vector3 side = Vector3.Cross(Vector3.up, ahead);
            Vector3 origin = hub.position;
            Vector3 pile = hub.position + ahead * 8f;
            pile.y = 0f;
            Vector3 middle = hub.position + ahead * 9f;
            double before = soil.WorkingVolume(middle, 14f) + blade.LoadM3;
            CapSoilStamp.Cone(soil, pile, 1.5, 34f);
            bladeDrive.SetTarget("lift_joint", EdgeAtGround);
            yield return Seconds(1.5f);
            double withPile = soil.WorkingVolume(middle, 14f) + blade.LoadM3;
            float edgeY = Mathf.Min(blade.edgeLeft.position.y, blade.edgeRight.position.y);   // just above the ground
            var overPile = new EdgeWatch(blade, origin, ahead, 8f - 1.4f, 8f + 1.4f);
            yield return DriveHolding(drive, hub, bladeDrive, blade, edgeY, 12f, 0.7f, overPile.Step);
            yield return Seconds(2.5f);                              // stops; the load laid before the blade
            double after = soil.WorkingVolume(middle, 14f) + blade.LoadM3;
            float pileTop = soil.GetComponent<Terrain>().SampleHeight(pile) + soil.transform.position.y;
            // where the pile stood (inside its foot, 1.29 m): every vertex cut down to the edge that passed over it
            var (_, swept) = LaneRange(soil, origin, ahead, side, 8f - 1.0f, 8f + 1.0f, -1.0f, 1.0f);
            Log($"I: pile 1.5 m3 (+{withPile - before:F4}), pushed: cut {blade.CutM3:F3} filled {blade.FilledM3:F3} " +
                $"spilled {blade.SpilledM3:F3} laid {blade.RestM3:F3} load {blade.LoadM3:F3} m3; soil {withPile:F4} -> " +
                $"{after:F4} m3; where the pile stood the ground is at {pileTop:F3} m, its highest vertex {swept:F4} " +
                $"m under an edge at {overPile.Low:F4}..{overPile.High:F4} m");
            Assert.AreEqual(1.5, withPile - before, 1e-3, "the pile's volume");
            Assert.AreEqual(withPile, after, 1e-3, "soil appeared or vanished under the blade (to 1e-3 m3)");
            Assert.Greater(overPile.Steps, 20, "the edge never passed over the pile");
            Assert.LessOrEqual(swept, overPile.High + 0.001f, "a vertex the edge passed over stands above it");
            // The edge held at its height above the scene's ground: what stood above it was cut, what lay below it stays
            // (live 10-24 with a fixed lift: 1.131 of 1.5 m3 cut, all of it filled into the lane's hollows ahead)
            Assert.Greater(blade.CutM3, 0.9, "the blade cut too little of the pile");
            Assert.AreEqual(blade.CutM3, blade.FilledM3 + blade.SpilledM3 + blade.RestM3 + blade.LoadM3, 1e-4,
                "the blade's ledger: what it cut went into the ground, off its ends, before it or onto it");
            Assert.Less(pileTop, 0.15f, "the pile still stands where it was");
            Assert.AreEqual(0.0, blade.LoadM3, 1e-6, "the load was not laid down when it stopped");

            // ---- J: a pile past the blade's capacity spills off both ends, outside the blade (a fresh dozer, its own lane)
            Object.Destroy(dozer);
            yield return Seconds(0.5f);
            dozer = Spawn(LaneB, 0f);
            drive = dozer.GetComponent<CapTrackKinematic>();
            bladeDrive = dozer.GetComponent<CapBladeDrive>();
            blade = dozer.GetComponent<CapDozerBlade>();
            hub = dozer.transform.Find("base_link");
            yield return Seconds(4f);
            ahead = hub.forward;
            ahead.y = 0f;
            ahead.Normalize();
            pile = hub.position + ahead * 8f;
            pile.y = 0f;
            middle = hub.position + ahead * 9f;
            CapSoilStamp.Cone(soil, pile, 3.5, 34f);
            bladeDrive.SetTarget("lift_joint", EdgeAtGround);
            yield return Seconds(1.5f);
            origin = hub.position;
            side = Vector3.Cross(Vector3.up, ahead);
            double j0 = soil.WorkingVolume(middle, 14f) + blade.LoadM3;
            // beside the swept lane (the blade's ends 1.6 m out), the pile's own fringe counted before and after
            double left0 = LaneVolume(soil, origin, ahead, side, 4f, 20f, -3.2f, -1.65f);
            double right0 = LaneVolume(soil, origin, ahead, side, 4f, 20f, 1.65f, 3.2f);
            edgeY = Mathf.Min(blade.edgeLeft.position.y, blade.edgeRight.position.y);
            double most = 0.0;
            yield return DriveHolding(drive, hub, bladeDrive, blade, edgeY, 12f, 0.7f,
                                      () => most = System.Math.Max(most, blade.LoadM3));
            yield return Seconds(2.5f);
            double j1 = soil.WorkingVolume(middle, 14f) + blade.LoadM3;
            double leftGain = LaneVolume(soil, origin, ahead, side, 4f, 20f, -3.2f, -1.65f) - left0;
            double rightGain = LaneVolume(soil, origin, ahead, side, 4f, 20f, 1.65f, 3.2f) - right0;
            Log($"J: 3.5 m3 pushed: cut {blade.CutM3:F3} spilled {blade.SpilledM3:F3} laid {blade.RestM3:F3} m3, the " +
                $"load at most {most:F4} m3 (capacity {blade.capacityM3:F1}); beside the lane +{leftGain:F3} / " +
                $"+{rightGain:F3} m3; soil {j0:F4} -> {j1:F4} m3");
            Assert.Greater(blade.SpilledM3, 0.3, "a load past capacity did not spill");
            Assert.LessOrEqual(most, blade.capacityM3 + 1e-6, "the load passed the blade's capacity");
            Assert.Greater(leftGain, 0.25 * blade.SpilledM3, "no windrow off the left end");
            Assert.Greater(rightGain, 0.25 * blade.SpilledM3, "no windrow off the right end");
            Assert.AreEqual(j0, j1, 1e-3, "soil appeared or vanished in the spill (to 1e-3 m3)");
            Object.Destroy(dozer);
        }

        [UnityTest, Timeout(600000)]
        public IEnumerator DozerBladeFillsAHole()
        {
            // A blade carrying 1.5 m3 over a hole of 0.8 m3 - where its load pays, the hole filled
            // up to the edge; the soil (terrain and load) kept to 1e-3 m3
            LogAssert.ignoreFailingMessages = true;
            SceneManager.sceneLoaded += NoParticles;
            yield return CapTestScene.LoadFresh(Scene);
            SceneManager.sceneLoaded -= NoParticles;
            yield return new WaitForSeconds(1.0f);
            var soil = SoilParticleSettings.instance;
            var dozer = Spawn(LaneA, 0f);
            var drive = dozer.GetComponent<CapTrackKinematic>();
            var bladeDrive = dozer.GetComponent<CapBladeDrive>();
            var blade = dozer.GetComponent<CapDozerBlade>();
            var hub = dozer.transform.Find("base_link");
            yield return Seconds(4f);
            Vector3 ahead = Level(hub.forward), side = Vector3.Cross(Vector3.up, ahead);
            Vector3 origin = hub.position, middle = origin + ahead * 9f;
            // the hole: 1.6 m x 1.6 m, 8 m ahead of the hub, dug in units of the particle-free dig; its walls slide
            double v0 = soil.WorkingVolume(middle, 14f), dug = 0.0;
            for (float s = 7.3f; s <= 8.71f; s += 0.2f)
            for (float l = -0.7f; l <= 0.71f; l += 0.2f)
                dug += soil.DigUnit(origin + ahead * s + side * l, 1f);
            yield return Seconds(2f);
            double v1 = soil.WorkingVolume(middle, 14f);
            var (holeLow, _) = LaneRange(soil, origin, ahead, side, 7.4f, 8.6f, -0.6f, 0.6f);
            bladeDrive.SetTarget("lift_joint", EdgeAtGround);
            yield return Seconds(1.5f);
            // the edge held 5 mm over the ground (lift -0.02 puts it 7 cm up: filling that layer spent the load
            // before the hole), the load put on once the blade moves (one standing still lays its load at once)
            float ground = soil.GetComponent<Terrain>().SampleHeight(origin + ahead * 4f) + soil.transform.position.y;
            float edgeY = ground + 0.005f;
            double before = soil.WorkingVolume(middle, 14f) + blade.LoadM3 + 1.5;
            var watch = new EdgeWatch(blade, origin, ahead, 6.9f, 9.1f);
            bool loaded = false;
            yield return DriveHolding(drive, hub, bladeDrive, blade, edgeY, 12f, 0.7f, () =>
            {
                watch.Step();
                if (!loaded && drive.Speed > 0.2)
                {
                    blade.Preload(1.5);
                    loaded = true;
                }
            });
            yield return Seconds(2.5f);
            double after = soil.WorkingVolume(middle, 14f) + blade.LoadM3;
            var (lo, hi) = LaneRange(soil, origin, ahead, side, 7.4f, 8.6f, -0.6f, 0.6f);
            Log($"N: a hole of {v0 - v1:F4} m3 (dug {dug:F4}, its floor {holeLow - ground:F3} m below the ground), " +
                $"the edge {edgeY - ground:F3} m above the ground at {watch.Low:F3}..{watch.High:F3} m over it; with " +
                $"1.5 m3: filled {blade.FilledM3:F3} cut {blade.CutM3:F3} laid {blade.RestM3:F3} m3; soil {before:F4} " +
                $"-> {after:F4} m3; the hole's inside now {lo:F4}..{hi:F4} m");
            Assert.Greater(v0 - v1, 0.7, "the hole was not dug");
            Assert.IsTrue(loaded, "the load was never put on");
            Assert.Greater(watch.Steps, 20, "the edge never passed over the hole");
            Assert.Less(watch.High - watch.Low, 0.01f, "the edge was not held steady over the hole");
            Assert.AreEqual(before, after, 1e-3, "soil appeared or vanished filling the hole");
            Assert.GreaterOrEqual(lo, watch.Low - 0.001f, "the hole is not filled up to the edge");
            Assert.LessOrEqual(hi, watch.High + 0.001f, "filled above the edge");
            Assert.Greater(blade.FilledM3, 0.9 * (v0 - v1), "the blade filled less than the hole");
            Object.Destroy(dozer);
        }

        [UnityTest, Timeout(600000)]
        public IEnumerator DozerAngledBladeCastsToItsTrailingSide()
        {
            // The blade angled 24 deg (the angle joint's limit) pushes a 3 m3 pile - what it casts
            // and spills lands off its trailing end, 80 % of it or more on that side; the soil kept to 1e-3 m3
            LogAssert.ignoreFailingMessages = true;
            SceneManager.sceneLoaded += NoParticles;
            yield return CapTestScene.LoadFresh(Scene);
            SceneManager.sceneLoaded -= NoParticles;
            yield return new WaitForSeconds(1.0f);
            var soil = SoilParticleSettings.instance;
            var dozer = Spawn(LaneB, 0f);
            var drive = dozer.GetComponent<CapTrackKinematic>();
            var bladeDrive = dozer.GetComponent<CapBladeDrive>();
            var blade = dozer.GetComponent<CapDozerBlade>();
            var hub = dozer.transform.Find("base_link");
            yield return Seconds(4f);
            bladeDrive.SetTarget("angle_joint", 0.419);
            bladeDrive.SetTarget("lift_joint", EdgeAtGround);
            yield return Seconds(2.5f);
            Vector3 ahead = Level(hub.forward), side = Vector3.Cross(Vector3.up, ahead);
            Vector3 origin = hub.position, middle = origin + ahead * 9f;
            // the trailing end: the angle positive about up puts the left end back (as ROS) - checked, not assumed
            float behind = Vector3.Dot(blade.edgeRight.position - blade.edgeLeft.position, ahead);
            float trailSign = Mathf.Sign(Vector3.Dot(blade.edgeLeft.position - origin, side));
            Vector3 pile = origin + ahead * 8f;
            pile.y = 0f;
            CapSoilStamp.Cone(soil, pile, 3.0, 34f);
            yield return Seconds(0.5f);
            double v0 = soil.WorkingVolume(middle, 14f) + blade.LoadM3;
            // either side beyond the swept width (half of 3.21 m x cos 24 deg = 1.47 m)
            float tl0 = trailSign < 0f ? -4.5f : 1.5f, tl1 = trailSign < 0f ? -1.5f : 4.5f;
            double trail0 = LaneVolume(soil, origin, ahead, side, 4f, 20f, tl0, tl1);
            double lead0 = LaneVolume(soil, origin, ahead, side, 4f, 20f, -tl1, -tl0);
            float edgeY = Mathf.Min(blade.edgeLeft.position.y, blade.edgeRight.position.y);
            yield return DriveHolding(drive, hub, bladeDrive, blade, edgeY, 12f, 0.7f);
            yield return Seconds(2.5f);
            double v1 = soil.WorkingVolume(middle, 14f) + blade.LoadM3;
            double trail = LaneVolume(soil, origin, ahead, side, 4f, 20f, tl0, tl1) - trail0;
            double lead = LaneVolume(soil, origin, ahead, side, 4f, 20f, -tl1, -tl0) - lead0;
            double share = trail / System.Math.Max(1e-9, trail + System.Math.Max(0.0, lead));
            Log($"O: angled {bladeDrive.Position("angle_joint"):F3} rad (its left end {behind:F2} m behind), 3 m3 " +
                $"pushed: cut {blade.CutM3:F3} spilled {blade.SpilledM3:F3} laid {blade.RestM3:F3} m3; beside the " +
                $"lane: trailing +{trail:F3}, leading {lead:+0.000;-0.000} m3 ({share * 100.0:F1} % trailing); soil " +
                $"{v0:F4} -> {v1:F4} m3");
            Assert.Greater(behind, 1.0f, "the blade is not angled with its left end back");
            Assert.Greater(blade.SpilledM3, 0.5, "the angled blade cast too little");
            Assert.GreaterOrEqual(share, 0.8, "less than 80 % on the trailing side");
            Assert.AreEqual(v0, v1, 1e-3, "soil appeared or vanished");
            Object.Destroy(dozer);
        }

        [UnityTest, Timeout(600000)]
        public IEnumerator DozerBladeLaysItsLoadWhenItStops()
        {
            // A blade carrying 1.5 m3 stops - a second after it stands, its load lies before it as a
            // wedge: the load 0, the ground round the edge up by what it carried (1e-3 m3), nine tenths of it from
            // the blade's band on (the wedge's back face slides under the blade: 49 deg is what the slides hold)
            LogAssert.ignoreFailingMessages = true;
            SceneManager.sceneLoaded += NoParticles;
            yield return CapTestScene.LoadFresh(Scene);
            SceneManager.sceneLoaded -= NoParticles;
            yield return new WaitForSeconds(1.0f);
            var soil = SoilParticleSettings.instance;
            var dozer = Spawn(LaneA, 0f);
            var drive = dozer.GetComponent<CapTrackKinematic>();
            var bladeDrive = dozer.GetComponent<CapBladeDrive>();
            var blade = dozer.GetComponent<CapDozerBlade>();
            var hub = dozer.transform.Find("base_link");
            yield return Seconds(4f);
            bladeDrive.SetTarget("lift_joint", EdgeAtGround);
            yield return Seconds(1.5f);
            // the edge held 5 mm over the ground: what it fills while it slows is that layer, not lift -0.02's 7 cm
            float edgeY = soil.GetComponent<Terrain>().SampleHeight(hub.position + Level(hub.forward) * 4f)
                          + soil.transform.position.y + 0.005f;
            yield return DriveHolding(drive, hub, bladeDrive, blade, edgeY, 4f, 0.7f);   // ends commanding a stop
            blade.Preload(1.5);
            Vector3 fwd = Level(hub.forward), side = Vector3.Cross(Vector3.up, fwd);
            Vector3 last = hub.position, edgeAt = Vector3.zero;
            float tStill = -1f, tZero = -1f;
            double loadAtStill = 0.0, roundBefore = 0.0, aheadBefore = 0.0;
            float t0 = Time.time;
            while (Time.time - t0 < 8f && tZero < 0f)
            {
                yield return new WaitForFixedUpdate();
                float v = Vector3.Dot(hub.position - last, fwd) / Time.fixedDeltaTime;
                last = hub.position;
                if (tStill < 0f && Mathf.Abs(v) < 0.05f)
                {
                    tStill = Time.time;
                    loadAtStill = blade.LoadM3;
                    edgeAt = 0.5f * (blade.edgeLeft.position + blade.edgeRight.position);
                    roundBefore = soil.WorkingVolume(edgeAt + fwd * 1f, 4f);
                    aheadBefore = LaneVolume(soil, edgeAt, fwd, side, -0.6f, 2.5f, -1.8f, 1.8f);
                }
                if (tStill > 0f && blade.LoadM3 == 0.0) tZero = Time.time;
            }
            yield return Seconds(1f);
            double roundAfter = soil.WorkingVolume(edgeAt + fwd * 1f, 4f);
            double aheadGain = LaneVolume(soil, edgeAt, fwd, side, -0.6f, 2.5f, -1.8f, 1.8f) - aheadBefore;
            Log($"P: stood {tStill - t0:F2} s after the stop with {loadAtStill:F4} m3, laid {tZero - tStill:F2} s " +
                $"later (rest {blade.RestM3:F4} m3); round the edge +{roundAfter - roundBefore:F4} m3, before it " +
                $"+{aheadGain:F4} m3");
            Assert.Greater(tStill, 0f, "never stood");
            Assert.Greater(loadAtStill, 1.4, "the load was spent before it stood");
            Assert.Greater(tZero, 0f, "the load was never laid");
            Assert.AreEqual(1.0f, tZero - tStill, 0.2f, "not laid a second after it stood");
            Assert.AreEqual(loadAtStill, roundAfter - roundBefore, 1e-3, "the ground did not take the load");
            Assert.Greater(aheadGain, 0.9 * loadAtStill, "the load was not laid before the blade");
            Object.Destroy(dozer);
        }

        [UnityTest, Timeout(600000)]
        public IEnumerator SoilSlidesWithoutLosingVolume()
        {
            // The slides alone for 30 s (no machine at the heap) - a heap steeper than the soil holds
            // slumps, the heap's surroundings and the whole map keep their volume to 1e-3 m3. Thermal.compute reads
            // the working heightmap's slopes at half their height (its scale): a 30 deg setting holds up to about
            // 49 deg, so the heap is laid at 65 deg
            LogAssert.ignoreFailingMessages = true;
            SceneManager.sceneLoaded += NoParticles;
            yield return CapTestScene.LoadFresh(Scene);
            SceneManager.sceneLoaded -= NoParticles;
            yield return new WaitForSeconds(1.0f);
            var soil = SoilParticleSettings.instance;
            yield return Seconds(3f);                       // the scene's own slopes slide from its start
            Vector3 at = new Vector3(20f, 0f, -38f), x = Vector3.right, z = Vector3.forward;
            double local0 = soil.WorkingVolume(at, 6f), map0 = MapVolume(soil);
            double laid = CapSoilStamp.Cone(soil, at, 1.5, 65f);
            double local1 = soil.WorkingVolume(at, 6f), map1 = MapVolume(soil);
            var (_, peak0) = LaneRange(soil, at, z, x, -0.5f, 0.5f, -0.5f, 0.5f);
            yield return Seconds(30f);
            double local2 = soil.WorkingVolume(at, 6f), map2 = MapVolume(soil);
            var (_, peak1) = LaneRange(soil, at, z, x, -0.5f, 0.5f, -0.5f, 0.5f);
            Log($"Q: a 65 deg heap of {laid:F4} m3 (+{local1 - local0:F4} round it, +{map1 - map0:F4} on the map), its " +
                $"top {peak0:F3} -> {peak1:F3} m in 30 s; round it {local1:F4} -> {local2:F4} m3, the map {map1:F4} " +
                $"-> {map2:F4} m3");
            Assert.AreEqual(1.5, laid, 1e-6);
            Assert.AreEqual(1.5, local1 - local0, 1e-3);
            Assert.Greater(peak0 - peak1, 0.3f, "the heap did not slide");
            Assert.AreEqual(local1, local2, 1e-3, "the slides lost or made soil round the heap");
            Assert.AreEqual(map1, map2, 1e-3, "the slides changed the map's volume");
        }

        [UnityTest, Timeout(600000)]
        public IEnumerator DozerPushesAPileOnFixedJoints()
        {
            // The whole machine at fixed joint targets - the blade riding the body, the body the
            // ground under its tracks, the tracks what the blade leaves - pushes a 1.5 m3 pile 12 m: the soil kept to
            // 1e-3 m3 (under 0.1 % of the pile). The lane is written for CAP's twin to run the same push
            // (LaneRecord: the twin runs the lane again and the two are compared)
            LogAssert.ignoreFailingMessages = true;
            SceneManager.sceneLoaded += NoParticles;
            yield return CapTestScene.LoadFresh(Scene);
            SceneManager.sceneLoaded -= NoParticles;
            yield return new WaitForSeconds(1.0f);
            var soil = SoilParticleSettings.instance;
            var dozer = Spawn(LaneB, 0f);
            var drive = dozer.GetComponent<CapTrackKinematic>();
            var bladeDrive = dozer.GetComponent<CapBladeDrive>();
            var blade = dozer.GetComponent<CapDozerBlade>();
            var hub = dozer.transform.Find("base_link");
            yield return Seconds(4f);
            bladeDrive.SetTarget("lift_joint", EdgeAtGround);
            bladeDrive.SetTarget("angle_joint", 0.0);
            bladeDrive.SetTarget("tilt_joint", 0.0);
            yield return Seconds(1.5f);
            Vector3 ahead = Level(hub.forward), side = Vector3.Cross(Vector3.up, ahead);
            Vector3 origin = hub.position, middle = origin + ahead * 9f;
            Vector3 pile = origin + ahead * 8f;
            pile.y = 0f;
            CapSoilStamp.Cone(soil, pile, 1.5, 34f);
            yield return Seconds(0.5f);
            var record = new LaneRecord(soil, origin, ahead, side, -4f, 22f, -4f, 4f);
            record.Begin();
            record.Step(drive, hub, bladeDrive, blade);          // at rest: the state the twin starts from
            double before = soil.WorkingVolume(middle, 14f) + blade.LoadM3;
            float ground = soil.GetComponent<Terrain>().SampleHeight(origin) + soil.transform.position.y;
            float edge0 = Mathf.Min(blade.edgeLeft.position.y, blade.edgeRight.position.y);
            float lowest = float.PositiveInfinity, highest = float.NegativeInfinity;
            Vector3 p0 = hub.position;
            drive.SetCommand(0.7, 0.0);
            float t0 = Time.time;
            while (Vector3.Dot(hub.position - p0, ahead) < 12f && Time.time - t0 < 40f)
            {
                yield return new WaitForFixedUpdate();
                record.Step(drive, hub, bladeDrive, blade);
                float e = Mathf.Min(blade.edgeLeft.position.y, blade.edgeRight.position.y);
                lowest = Mathf.Min(lowest, e);
                highest = Mathf.Max(highest, e);
            }
            drive.SetCommand(0.0, 0.0);
            for (int k = 0; k < 125; k++)                    // 2.5 s: it stops, the load laid before the blade
            {
                yield return new WaitForFixedUpdate();
                record.Step(drive, hub, bladeDrive, blade);
            }
            double after = soil.WorkingVolume(middle, 14f) + blade.LoadM3;
            string path = record.Write("fixed_joints_1p5", blade, "a 1.5 m3 pile 8 m ahead, lift " +
                                       EdgeAtGround.ToString(CultureInfo.InvariantCulture) + " rad, 0.7 m/s for 12 m");
            Log($"R: fixed joints, 1.5 m3 pushed 12 m: the edge {edge0 - ground:F3} m above the ground at the start, " +
                $"{lowest - ground:F3}..{highest - ground:F3} m on the way (hub {hub.position.y - ground:F3} m up at " +
                $"the end); cut {blade.CutM3:F3} filled {blade.FilledM3:F3} spilled {blade.SpilledM3:F3} laid " +
                $"{blade.RestM3:F3} load {blade.LoadM3:F3} m3; soil {before:F4} -> {after:F4} m3; the lane -> {path}");
            Assert.Greater(blade.CutM3, 0.9, "the blade cut too little of the pile");
            Assert.AreEqual(before, after, 1e-3, "the soil changed by more than 1e-3 m3");
            Assert.AreEqual(0.0, blade.LoadM3, 1e-6, "the load was not laid down when it stopped");
            Object.Destroy(dozer);
        }

        [System.Serializable]
        private class DumpDone
        {
            public string id;
            public float x, y;
            public double volume_m3;
            public bool landed;
            public float waited_s;
        }

        [Test]
        public void DumpDoneIsJson()
        {
            // The dump truck's /scenario/dump_done as a reader parses it (10-26: "waited_s": F1} - composite formatting
            // took "{5:F1}}}" for the format "F1}" - and the Site dropped every load's done)
            string json = CapScenarioOps.DumpDoneJson("load-1", 25f, -32.5f, 5.5, false, 600f);
            Log($"S: {json}");
            Assert.IsFalse(json.Contains("F1"), json);
            var d = JsonUtility.FromJson<DumpDone>(json);
            Assert.AreEqual("load-1", d.id);
            Assert.AreEqual(25f, d.x, 1e-6f);
            Assert.AreEqual(-32.5f, d.y, 1e-6f);
            Assert.AreEqual(5.5, d.volume_m3, 1e-9);
            Assert.IsFalse(d.landed);
            Assert.AreEqual(600f, d.waited_s, 1e-6f);
        }
    }
}
#endif
