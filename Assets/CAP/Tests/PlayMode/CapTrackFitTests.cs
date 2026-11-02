#if UNITY_EDITOR
using System.Collections;
using NUnit.Framework;
using UnityEditor;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.TestTools;

namespace CapKit.Tests
{
    /// <summary>
    /// The simulator's zx120 on kinematic tracks (CapTrackKinematic.Fit; CapCycleRecorder's placements'
    /// "kinematic_tracks"): its WheelColliders' drive could not climb a bench's ramp (r28, 11-02: it slid back down its
    /// foot). Fitted where it rests, it drives straight up a mesa's 16 deg ramp onto the top and stands level there,
    /// turns 60 deg in place without moving off the spot, and its arm goes with its body. On a copy of the scene's
    /// TerrainData (a test's Play would keep a change to the asset itself).
    ///   Unity -batchmode -projectPath ... -runTests -testPlatform PlayMode -testFilter CapTrackFit -testResults out.xml
    /// </summary>
    public class CapTrackFitTests
    {
        private const string Scene = "Assets/Scenes/SimpleScene.unity";

        private static void Log(string s) => Debug.Log("[CapTrackFitTest] " + s);

        /// <summary>As the scene loads (before its soil starts: its working heightmap is built from the terrain's heights
        /// then, and written back over them - as CapCycleRecorder shapes a run's): no particles, and chain5_yard's bench
        /// (harness_run.ZX120_MESA: its top 1.5 m up, its sides over a 5.2 m run, 16 deg) on a copy of the TerrainData.
        /// </summary>
        private static void BenchOnLoad(Scene scene, LoadSceneMode mode)
        {
            if (SoilParticleSettings.instance != null) SoilParticleSettings.instance.particles = false;
            var terrain = Terrain.activeTerrain;
            if (terrain == null) return;
            terrain.terrainData = Object.Instantiate(terrain.terrainData);
            var collider = terrain.GetComponent<TerrainCollider>();
            if (collider != null) collider.terrainData = terrain.terrainData;
            CapScenarioOps.ShapeTerrain(new CapScenarioOps.Config
            {
                terrain = new[]
                {
                    new CapScenarioOps.TerrainOp { op = "flatten", x0 = 12f, y0 = -56f, x1 = 46f, y1 = -32f, z = 0f },
                    new CapScenarioOps.TerrainOp { op = "mesa", x0 = 20f, y0 = -48f, x1 = 28f, y1 = -40f, z = 1.5f,
                                                   run_m = 5.2f },
                }
            });
        }

        private static IEnumerator Seconds(float s)
        {
            float until = Time.time + s;
            while (Time.time < until) yield return new WaitForFixedUpdate();
        }

        private static float RosX(Transform t) => t.position.z;
        private static float RosY(Transform t) => -t.position.x;

        [UnityTest, Timeout(600000)]
        public IEnumerator Zx120ClimbsAMesasRampAndTurnsOnItsTopItsArmWithIt()
        {
            LogAssert.ignoreFailingMessages = true;
            SceneManager.sceneLoaded += BenchOnLoad;
            try
            {
                yield return CapTestScene.LoadFresh(Scene);
            }
            finally
            {
                SceneManager.sceneLoaded -= BenchOnLoad;
            }
            yield return Seconds(1f);                                   // the soil started on the bench's ground
            var terrain = Terrain.activeTerrain;
            Assert.IsNotNull(terrain, "no terrain");
            float H(float rx, float ry) => terrain.SampleHeight(new Vector3(-ry, 0f, rx)) + terrain.transform.position.y;
            Assert.AreEqual(1.5f, H(24f, -44f), 0.02f, "the top");
            Assert.AreEqual(0.75f, H(28f + 2.6f, -44f), 0.06f, "half way up its ramp");
            var prefab = AssetDatabase.LoadAssetAtPath<GameObject>("Assets/Prefab/zx120.prefab");
            Assert.IsNotNull(prefab, "no Assets/Prefab/zx120.prefab");
            // on the flat 3 m east of the ramp's foot (ROS x 28 + 5.2), facing up it: ROS west, Unity's yaw 180
            var go = Object.Instantiate(prefab, new Vector3(44f, H(36.2f, -44f) + 1.0f, 36.2f),
                                        Quaternion.Euler(0f, 180f, 0f));
            go.name = "zx120";
            yield return Seconds(5f);                                   // at rest on its wheels
            var hub = go.transform;
            float rest = hub.position.y - H(RosX(hub), RosY(hub));
            Vector3 up0 = hub.up;
            // the upper structure (the articulation's root, fixed to the tracks' body) and the arm's end (its bucket)
            Transform upper = null, arm = null;
            foreach (var ab in go.GetComponentsInChildren<ArticulationBody>())
            {
                if (ab.isRoot && upper == null) upper = ab.transform;
                if (arm == null || ab.name.ToLowerInvariant().Contains("bucket")) arm = ab.transform;
            }
            Assert.IsNotNull(upper, "no articulation");
            Assert.IsNotNull(arm, "no arm");
            Vector3 upperInBody0 = hub.InverseTransformPoint(upper.position);
            Vector3 armInBody0 = hub.InverseTransformPoint(arm.position);
            var tracks = CapTrackKinematic.Fit(go, 1.79f, 1.0f, 0.6f);
            Assert.AreEqual(rest, tracks.hubAboveGround, 0.02f, "fitted at its height at rest");
            Log($"at rest {rest:F3} m over the ground at ROS ({RosX(hub):F2}, {RosY(hub):F2}); arm {arm.name}");
            yield return null;

            // straight up the ramp onto the top, to ROS x 24
            tracks.SetCommand(0.46, 0.0);
            float t0 = Time.time;
            while (RosX(hub) > 24f && Time.time - t0 < 60f) yield return new WaitForFixedUpdate();
            tracks.SetCommand(0.0, 0.0);
            yield return Seconds(1.5f);
            Log($"up in {Time.time - t0:F1} s at ROS ({RosX(hub):F2}, {RosY(hub):F2}), {hub.position.y:F3} m");
            Assert.LessOrEqual(RosX(hub), 24.5f, "not up on the top");
            Assert.AreEqual(1.5f, H(RosX(hub), RosY(hub)), 0.02f, "the top still under it");
            Assert.AreEqual(1.5f + rest, hub.position.y, 0.08f, "its hub over the top as over the flat");
            Assert.Less(Vector3.Angle(up0, hub.up), 2f, "not level on the top");
            Assert.AreEqual(-44f, RosY(hub), 0.15f, "slid across the ramp");

            // a turn in place on the top: 60 deg clockwise from above (ROS's negative)
            Vector3 at = hub.position;
            float yaw0 = hub.eulerAngles.y, t1 = Time.time;
            tracks.SetCommand(0.0, -0.6);
            while (Mathf.Abs(Mathf.DeltaAngle(yaw0, hub.eulerAngles.y)) < 60f && Time.time - t1 < 10f)
            {
                yield return new WaitForFixedUpdate();
            }
            float took = Time.time - t1;
            tracks.SetCommand(0.0, 0.0);
            yield return Seconds(1f);
            float moved = Vector2.Distance(new Vector2(at.x, at.z), new Vector2(hub.position.x, hub.position.z));
            Log($"turned {Mathf.DeltaAngle(yaw0, hub.eulerAngles.y):F1} deg in {took:F2} s, moved {moved:F3} m");
            Assert.Less(took, 2.5f, "a 60 deg turn at 0.6 rad/s takes 1.75 s");
            Assert.Less(moved, 0.1f, "moved while turning in place");

            // its upper structure with its body all the way (fixed to it); its arm's end too, but for what its joints
            // move it with nobody holding them here (a run's agent holds them in their travel posture)
            float upper_off = Vector3.Distance(upperInBody0, hub.InverseTransformPoint(upper.position));
            float arm_off = Vector3.Distance(armInBody0, hub.InverseTransformPoint(arm.position));
            Log($"the upper structure ({upper.name}) {upper_off:F3} m and the arm's end ({arm.name}) {arm_off:F3} m off " +
                "where they were on the body");
            Assert.Less(upper_off, 0.05f, "the upper structure did not go with the body");
            Assert.Less(arm_off, 1.0f, "the arm was left behind");
            Object.Destroy(go);
        }
    }
}
#endif
