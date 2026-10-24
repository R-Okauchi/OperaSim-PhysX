#if UNITY_EDITOR
using System.Collections;
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
        // The edge just above the flat ground (about +8 mm with the dozer standing): it cuts the piles, not the ground -
        // lower, the front tracks run down into what the blade cut and it digs deeper as it goes (a blade held to the
        // body, not to the ground: CAP's machine control holds it to the design)
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

        [UnityTest, Timeout(900000)]
        public IEnumerator DozerDrivesAndMovesItsBlade()
        {
            LogAssert.ignoreFailingMessages = true;  // ROS-TCP has no endpoint here
            SceneManager.sceneLoaded += NoParticles;
            yield return CapTestScene.LoadFresh(Scene);
            SceneManager.sceneLoaded -= NoParticles;
            yield return new WaitForSeconds(1.0f);

            var dozer = Spawn(Start, 0f);
            var drive = dozer.GetComponent<DiffDriveController>();
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

        /// <summary>Drive the dozer along its heading for a distance at a speed (the blade as it is set).</summary>
        private static IEnumerator Drive(DiffDriveController drive, Transform hub, float metres, float speed)
        {
            Vector3 p0 = hub.position;
            Vector3 ahead = hub.forward;
            ahead.y = 0f;
            ahead.Normalize();
            drive.SetCommand(speed, 0.0);
            float t0 = Time.time;
            while (Vector3.Dot(hub.position - p0, ahead) < metres && Time.time - t0 < metres / Mathf.Max(0.1f, speed) + 20f)
                yield return new WaitForFixedUpdate();
            drive.SetCommand(0.0, 0.0);
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
            Assert.AreEqual(33.0, v1 - v0, 0.02);

            // ---- I: the blade at the ground pushes a 1.5 m3 pile 12 m: soil kept, the pile gone from its place
            var dozer = Spawn(LaneA, 0f);
            var drive = dozer.GetComponent<DiffDriveController>();
            var bladeDrive = dozer.GetComponent<CapBladeDrive>();
            var blade = dozer.GetComponent<CapDozerBlade>();
            var hub = dozer.transform.Find("base_link");
            Assert.IsNotNull(blade, "no CapDozerBlade on the dozer");
            yield return Seconds(4f);
            Vector3 ahead = hub.forward;
            ahead.y = 0f;
            ahead.Normalize();
            Vector3 pile = hub.position + ahead * 8f;
            pile.y = 0f;
            Vector3 middle = hub.position + ahead * 9f;
            double before = soil.WorkingVolume(middle, 14f) + blade.LoadM3;
            CapSoilStamp.Cone(soil, pile, 1.5, 34f);
            bladeDrive.SetTarget("lift_joint", EdgeAtGround);
            yield return Seconds(1.5f);
            double withPile = soil.WorkingVolume(middle, 14f) + blade.LoadM3;
            yield return Drive(drive, hub, 12f, 0.7f);
            yield return Seconds(2.5f);                              // stops; the load laid before the blade
            double after = soil.WorkingVolume(middle, 14f) + blade.LoadM3;
            float pileTop = soil.GetComponent<Terrain>().SampleHeight(pile) + soil.transform.position.y;
            Log($"I: pile 1.5 m3 (+{withPile - before:F4}), pushed: cut {blade.CutM3:F3} filled {blade.FilledM3:F3} " +
                $"spilled {blade.SpilledM3:F3} laid {blade.RestM3:F3} load {blade.LoadM3:F3} m3; soil {withPile:F4} -> " +
                $"{after:F4} m3; where the pile stood the ground is at {pileTop:F3} m");
            Assert.AreEqual(1.5, withPile - before, 0.005, "the pile's volume");
            Assert.AreEqual(withPile, after, 0.01, "soil appeared or vanished under the blade");
            // A fixed lift: the edge rides with the body over the scene's ground, a few centimetres above it here and
            // more where the ground dips under the pile - what stood above the edge was cut, what lay below it stays
            // (live 10-24: 1.131 of 1.5 m3 cut, all of it filled into the lane's hollows ahead)
            Assert.Greater(blade.CutM3, 0.9, "the blade cut too little of the pile");
            Assert.AreEqual(blade.CutM3, blade.FilledM3 + blade.SpilledM3 + blade.RestM3 + blade.LoadM3, 0.005,
                "the blade's ledger: what it cut went into the ground, off its ends, before it or onto it");
            Assert.Less(pileTop, 0.15f, "the pile still stands where it was");
            Assert.AreEqual(0.0, blade.LoadM3, 1e-6, "the load was not laid down when it stopped");

            // ---- J: a pile past the blade's capacity spills off both ends, outside the blade (a fresh dozer, its own lane)
            Object.Destroy(dozer);
            yield return Seconds(0.5f);
            dozer = Spawn(LaneB, 0f);
            drive = dozer.GetComponent<DiffDriveController>();
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
            double j0 = soil.WorkingVolume(middle, 14f) + blade.LoadM3;
            yield return Drive(drive, hub, 12f, 0.7f);
            yield return Seconds(2.5f);
            double j1 = soil.WorkingVolume(middle, 14f) + blade.LoadM3;
            Log($"J: 3.5 m3 pushed: cut {blade.CutM3:F3} spilled {blade.SpilledM3:F3} laid {blade.RestM3:F3} m3, soil " +
                $"{j0:F4} -> {j1:F4} m3");
            Assert.Greater(blade.SpilledM3, 0.3, "a load past capacity did not spill");
            Assert.AreEqual(j0, j1, 0.01, "soil appeared or vanished in the spill");
            Object.Destroy(dozer);
        }
    }
}
#endif
