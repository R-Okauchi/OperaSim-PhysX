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

        private static float YawChange(float from, float to) => Mathf.DeltaAngle(from, to);

        [UnityTest, Timeout(900000)]
        public IEnumerator DozerDrivesAndMovesItsBlade()
        {
            LogAssert.ignoreFailingMessages = true;  // ROS-TCP has no endpoint here
            SceneManager.sceneLoaded += NoParticles;
            yield return EditorSceneManager.LoadSceneAsyncInPlayMode(Scene, new LoadSceneParameters(LoadSceneMode.Single));
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
            Object.Destroy(dozer);
        }
    }
}
#endif
