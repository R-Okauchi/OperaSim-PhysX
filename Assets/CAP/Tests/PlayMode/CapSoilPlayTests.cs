#if UNITY_EDITOR
using System.Collections;
using System.Linq;
using NUnit.Framework;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.TestTools;

namespace CapKit.Tests
{
    /// <summary>
    /// 粒子なしの土 (SoilParticleSettings.particles = false) を SimpleScene で: 掘った量・こぼれた量・荷台の量・ダンプの量が
    /// 地形の体積と合うか (どこにも消えず、どこからも湧かない)。ROS 無し: 関節は ArticulationBody の目標を直に
    /// (com3 の位置の口と同じ、URDF の角度 [deg])。
    ///   Unity -batchmode -projectPath ... -runTests -testPlatform PlayMode -testFilter CapSoil -testResults out.xml
    /// 一つの試験で順に通す (地形のデータと SoilParticleSettings はシーンを読み直しても残る: DontDestroyOnLoad)。
    /// </summary>
    public class CapSoilPlayTests
    {
        private const string Scene = "Assets/Scenes/SimpleScene.unity";
        // CAP の zx200 の腕の姿勢 [rad] (boom, arm, bucket): skills/load.py の運ぶ姿勢と、_kinematics.solve_boom_arm で
        // 解いた浅い掘削 (爪先 7.5 m で地面の 0.3 m 下へ入れ、5.8 m まで引き、5.2 m で巻く)
        private static readonly float[] Lift = { -1.05f, 1.25f, 2.0f };
        private static readonly float[] Above = { -0.575f, 1.525f, 0.3f };
        private static readonly float[] Penetrate = { -0.368f, 1.371f, 0.5f };
        private static readonly float[] Drag = { -0.349f, 1.517f, 1.2f };
        private static readonly float[] Curl = { -0.200f, 1.422f, 2.0f };
        private const float Open = -0.5f;

        private static void Log(string s) => Debug.Log("[CapSoilTest] " + s);

        /// <summary>After Awake, before Start: no particles, and the terrains on copies of their data - the soil changes the
        /// TerrainData in place, and the editor saved the asset dug (2026-10-22: OnApplicationQuit's restore came too late).</summary>
        private static void NoParticles(Scene scene, LoadSceneMode mode)
        {
            if (SoilParticleSettings.instance != null)
                SoilParticleSettings.instance.particles = false;
            TerrainCopies(scene, mode);
        }

        [UnityTest, Timeout(900000)]
        public IEnumerator SoilWithoutParticles()
        {
            LogAssert.ignoreFailingMessages = true;              // ROS-TCP has no endpoint here
            SceneManager.sceneLoaded += NoParticles;              // after Awake, before Start
            yield return EditorSceneManager.LoadSceneAsyncInPlayMode(Scene, new LoadSceneParameters(LoadSceneMode.Single));
            SceneManager.sceneLoaded -= NoParticles;
            yield return new WaitForSeconds(1.0f);

            // ---- A: the mode, one dig's volume, the bucket's mouth, the bed
            var soil = SoilParticleSettings.instance;
            Assert.IsNotNull(soil, "no SoilParticleSettings");
            Assert.IsFalse(soil.particles, "particles still on: sceneLoaded came after Start?");
            var terrain = soil.GetComponent<Terrain>();
            Assert.AreEqual(0.01278f, soil.DigUnitM3, 0.0002f, "one dig's volume");
            var zx200 = GameObject.Find("zx200").transform;
            var bucket = zx200.GetComponentInChildren<BucketRocks>();
            Assert.AreEqual(0.8f, bucket.CapacityM3, 1e-6f);
            Assert.IsTrue(bucket.Mouth.fromRays && bucket.Mouth.escaped > 100, $"mouth {bucket.Mouth.escaped} rays out");
            var mst = GameObject.Find("mst110cr").transform;
            var bed = mst.GetComponentInChildren<CapBedLoad>();
            Assert.IsNotNull(bed, "no CapBedLoad on mst110cr's vessel");
            Assert.AreEqual(4.0f, bed.CapacityM3, 1e-6f);
            Assert.AreEqual(0, Object.FindObjectsOfType<RockObjectDetector>().Length, "particles made");
            foreach (var (name, x, z) in new[] { ("zx120", -40f, -40f), ("ic120", -40f, 40f), ("c30r", 40f, 40f) })
            {
                var go = GameObject.Find(name);
                if (go != null) Teleport(go.transform, new Vector3(x, 0f, z), go.transform.rotation);
            }
            yield return new WaitForSeconds(1.0f);

            // ---- B: digs and deposits at a far point move exactly their volume
            var far = new Vector3(-25f, 0f, -10f);
            float v0 = Volume(terrain, far, 20);
            for (int i = 0; i < 20; i++)
            {
                Vector2 o = Random.insideUnitCircle * 0.5f;
                Assert.AreEqual(soil.DigUnitM3, soil.DigUnit(far + new Vector3(o.x, 0f, o.y), 1f), 1e-7f);
            }
            yield return Steps(3);
            float v1 = Volume(terrain, far, 20);
            Log($"B: 20 digs {v0 - v1:F4} m3 out of the terrain, expected {20 * soil.DigUnitM3:F4}");
            Assert.AreEqual(20 * soil.DigUnitM3, v0 - v1, 0.02f * 20 * soil.DigUnitM3, "20 digs");
            soil.Deposit(far, 0.3f, 0.35f);
            yield return Steps(3);
            float v2 = Volume(terrain, far, 20);
            Log($"B: a 0.3 m3 deposit {v2 - v1:F4} m3 into the terrain");
            Assert.AreEqual(0.3f, v2 - v1, 0.006f, "deposit");

            // ---- E: the bucket over the bed, opened: all of it in the bed
            var joints = zx200.GetComponentsInChildren<ArticulationBody>().ToDictionary(a => a.name);
            Pose(joints, Lift);
            yield return new WaitForSeconds(5.0f);
            float liftDeg = Vector3.Angle(bucket.transform.parent.TransformDirection(bucket.Mouth.direction), Vector3.up);
            float holds = bucket.Holds();
            Log($"E: the carry (boom {Lift[0]}, arm {Lift[1]}, bucket {Lift[2]}): mouth {liftDeg:F1} deg from up, holds {holds:F3} m3");
            bucket.SetLoad(holds);
            float loaded = bucket.LoadM3;
            Vector3 spill = bucket.SpillPoint();
            Vector3 center = joints["body_link"].transform.position;
            Vector3 radial = Vector3.ProjectOnPlane(spill - center, Vector3.up).normalized;
            PlaceBedUnder(mst, bed, spill, radial);
            yield return new WaitForSeconds(3.0f);
            Vector3 bedCenter = bed.transform.TransformPoint((bed.Bed.min + bed.Bed.max) / 2f);
            float off = Vector3.ProjectOnPlane(bedCenter - spill, Vector3.up).magnitude;
            float tilt = Vector3.Angle(bed.transform.up, Vector3.up);
            Log($"E: the bed's centre {off:F2} m across from the spill point, tilted {tilt:F1} deg");
            Assert.Less(off, 0.5f, "the bed not under the bucket");
            Assert.Less(tilt, 5f, "the bed tilted where it was placed");
            float groundBefore = bucket.ToGroundM3, bedsBefore = bucket.ToBedsM3;
            Assert.AreEqual(loaded, bucket.LoadM3, 0.01f, "spilled while the bed was placed");
            SetBucket(joints, Open);
            yield return new WaitForSeconds(6.0f);
            float toBed = bucket.ToBedsM3 - bedsBefore, toGround = bucket.ToGroundM3 - groundBefore;
            Log($"E: opened: {toBed:F3} m3 into the bed, {toGround:F3} m3 to the ground, {bucket.LoadM3:F4} m3 left; the bed holds {bed.VolumeM3:F3} m3; " +
                $"the mound {(bed.MoundObject != null && bed.MoundObject.GetComponent<Collider>().enabled ? "shown" : "hidden")}");
            Assert.Less(bucket.LoadM3, 0.005f, "the bucket not emptied");
            Assert.Greater(toBed, 0.95f * loaded, "the bucket's load not in the bed");
            Assert.AreEqual(bed.VolumeM3, bed.ReceivedM3 - bed.ToGroundM3 - bed.LostM3, 1e-4f, "the bed's account");
            Assert.IsTrue(bed.MoundObject.GetComponent<Collider>().enabled, "no mound for the LiDAR");
            Assert.AreEqual(0f, bucket.LostM3, 1e-6f, "soil nowhere");

            // ---- F: the bed tipped: all of it behind the bed, on the ground
            float inBed = bed.VolumeM3;
            Vector3 rear = bed.transform.TransformPoint(new Vector3(0f, 0f, bed.Bed.min.z - 1.0f));
            float r0 = Volume(terrain, rear, 25);
            float bedGround0 = bed.ToGroundM3;
            var vessel = bed.GetComponent<ArticulationBody>();
            var drive = vessel.xDrive;
            drive.target = drive.lowerLimit;                       // -60 deg: the front up
            vessel.xDrive = drive;
            yield return new WaitForSeconds(6.0f);
            float tipped = Vector3.Angle(bed.transform.up, Vector3.up);
            float out1 = bed.ToGroundM3 - bedGround0;
            yield return Steps(3);
            float r1 = Volume(terrain, rear, 25);
            Log($"F: tipped {tipped:F0} deg: {out1:F3} m3 out of {inBed:F3}, the terrain behind +{r1 - r0:F3} m3; the bed holds {bed.VolumeM3:F4}");
            Assert.Greater(tipped, 40f, "the bed did not tip");
            Assert.Less(bed.VolumeM3, 0.01f, "the bed not emptied");
            Assert.AreEqual(inBed, out1, 0.01f, "the bed's soil not on the ground");
            Assert.AreEqual(out1, r1 - r0, 0.05f * out1 + 0.005f, "the ground behind the bed");

            // ---- G: overfilled when level: the excess over the rim, the bed at its capacity
            drive.target = 0f;
            vessel.xDrive = drive;
            yield return new WaitForSeconds(5.0f);
            float g0 = bed.ToGroundM3;
            bed.Add(bed.CapacityM3 + 0.5f);
            yield return new WaitForSeconds(2.0f);
            Log($"G: {bed.CapacityM3 + 0.5f:F2} m3 into a level bed: holds {bed.VolumeM3:F3}, {bed.ToGroundM3 - g0:F3} m3 over the rim");
            Assert.AreEqual(bed.CapacityM3, bed.VolumeM3, 0.02f, "the bed past its capacity");
            Assert.AreEqual(0.5f, bed.ToGroundM3 - g0, 0.02f, "the excess");

            // ---- H: full, on a slope: the load kept (the bed's tilt is its dump's, not the machine's: 10-23 review)
            float h0 = bed.ToGroundM3;
            Teleport(mst, mst.position + Vector3.up * 0.3f, Quaternion.AngleAxis(20f, mst.right) * mst.rotation);
            yield return Steps(3);                                   // judged before the machine settles back level
            float world = Vector3.Angle(bed.transform.up, Vector3.up);
            Log($"H: pitched {world:F1} deg: the bed's tilt {bed.TiltDeg():F1} deg, holds {bed.VolumeM3:F3}, out {bed.ToGroundM3 - h0:F4}");
            Assert.Greater(world, 8f, "the machine not pitched");
            Assert.Less(bed.TiltDeg(), 2f, "the dump's angle");
            Assert.AreEqual(h0, bed.ToGroundM3, 1e-4f, "the load spilled on a slope");
            Teleport(mst, new Vector3(40f, 0f, -40f), Quaternion.Euler(0f, mst.rotation.eulerAngles.y, 0f));

            // ---- C: a dig by the arm: what the bucket holds came out of the ground under it
            joints["body_link"].xDrive = WithTarget(joints["body_link"].xDrive, joints["body_link"].xDrive.target + 180f);
            Pose(joints, Above);
            yield return new WaitForSeconds(8.0f);
            Vector3 radialNow = Vector3.ProjectOnPlane(bucket.SpillPoint() - center, Vector3.up).normalized;
            Log($"C: swung {Vector3.SignedAngle(radial, radialNow, Vector3.up):F0} deg from the bed's side");
            Vector3 digAt = center + radialNow * 7.5f;               // the dig 5.2-7.5 m out, the pour under the carry
            float d0 = Volume(terrain, digAt, 40);
            float dug0 = bucket.DugM3, ground0 = bucket.ToGroundM3;
            var tip = joints["bucket_end_link"].transform;
            Log($"C: above: the tip {tip.position.y - GroundAt(tip.position):F2} m over the ground, the mouth {MouthDeg(bucket):F0} deg from up");
            // CAP's arm chain (OperaSim's older zx200.urdf) left the tip 0.12 m over this ground at the penetration: the
            // boom lowered from there until the bucket meets the ground, then 0.1 rad more, and the dig dragged and curled
            // at that boom
            Pose(joints, Penetrate);
            yield return new WaitForSeconds(3.0f);
            float boom = Penetrate[0];
            int touched = bucket.Contacts;
            while (bucket.Contacts < touched + 5 && boom < 0.6f)
            {
                boom += 0.03f;
                Pose(joints, new[] { boom, Penetrate[1], Penetrate[2] });
                yield return new WaitForSeconds(0.4f);
            }
            boom += 0.1f;
            var steps = new[] { (new[] { boom, Penetrate[1], Penetrate[2] }, "press"), (new[] { boom, Drag[1], Drag[2] }, "drag"),
                                (new[] { boom - 0.1f, Curl[1], Curl[2] }, "curl"), (Lift, "lift") };
            foreach (var (pose, name) in steps)
            {
                Pose(joints, pose);
                yield return new WaitForSeconds(3.0f);
                Log($"C: {name} (boom {pose[0]:F2}): the tip {tip.position.y - GroundAt(tip.position):F2} m over the ground, the mouth " +
                    $"{MouthDeg(bucket):F0} deg, holds {bucket.LoadM3:F3}; contacts {bucket.Contacts} (behind {bucket.Behind}, no room {bucket.NoRoom})");
            }
            yield return new WaitForSeconds(2.0f);
            float dug = bucket.DugM3 - dug0, spilled = bucket.ToGroundM3 - ground0;
            float d1 = Volume(terrain, digAt, 40);
            Log($"C: dug {dug:F3} m3 (holds {bucket.LoadM3:F3}, {spilled:F3} spilled on the lift), the ground {d1 - d0:+0.000;-0.000} m3");
            Assert.Greater(dug, 0.1f, "the arm dug nothing");
            Assert.AreEqual(-(dug - spilled), d1 - d0, 0.05f * dug + 0.005f, "the dig's volume");
            Assert.AreEqual(dug - spilled, bucket.LoadM3, 0.006f, "the bucket's account");

            // ---- D: opened over the ground: back into the ground, nothing lost
            SetBucket(joints, Open);
            yield return new WaitForSeconds(6.0f);
            yield return Steps(3);
            float d2 = Volume(terrain, digAt, 40);
            Log($"D: opened over the ground: {bucket.LoadM3:F4} m3 left, the ground {d2 - d0:+0.000;-0.000} m3 net of the dig");
            Assert.Less(bucket.LoadM3, 0.005f, "the bucket not emptied");
            Assert.AreEqual(0f, d2 - d0, 0.05f * dug + 0.005f, "the soil not back in the ground");
            Assert.AreEqual(0f, bucket.LostM3, 1e-6f, "soil nowhere");
            Log($"done: dug {bucket.DugM3:F3}, to beds {bucket.ToBedsM3:F3}, to the ground {bucket.ToGroundM3:F3}; the bed received " +
                $"{bed.ReceivedM3:F3}, flowed out {bed.ToGroundM3:F3}");
        }

        /// <summary>The scene's soil as upstream has it (particles on): a bucket pressed into the ground still makes particles,
        /// and no bed takes soil.</summary>
        [UnityTest, Timeout(600000)]
        public IEnumerator ParticlesStillWhenOn()
        {
            LogAssert.ignoreFailingMessages = true;
            SceneManager.sceneLoaded += TerrainCopies;
            yield return EditorSceneManager.LoadSceneAsyncInPlayMode(Scene, new LoadSceneParameters(LoadSceneMode.Single));
            SceneManager.sceneLoaded -= TerrainCopies;
            yield return new WaitForSeconds(1.0f);
            var soil = SoilParticleSettings.instance;
            Assert.IsTrue(soil.particles, "the scene's default is particles");
            Assert.AreEqual(0, Object.FindObjectsOfType<CapBedLoad>().Length, "a bed load with particles on");
            var zx200 = GameObject.Find("zx200").transform;
            var bucket = zx200.GetComponentInChildren<BucketRocks>();
            var joints = zx200.GetComponentsInChildren<ArticulationBody>().ToDictionary(a => a.name);
            Pose(joints, Penetrate);
            yield return new WaitForSeconds(3.0f);
            float boom = Penetrate[0];
            while (Object.FindObjectsOfType<RockObjectDetector>().Length == 0 && boom < 0.6f)
            {
                boom += 0.03f;
                Pose(joints, new[] { boom, Penetrate[1], Penetrate[2] });
                yield return new WaitForSeconds(0.4f);
            }
            int rocks = Object.FindObjectsOfType<RockObjectDetector>().Length;
            Log($"particles on: {rocks} particles by boom {boom:F2}; the bucket holds {bucket.LoadM3:F3} m3");
            Assert.Greater(rocks, 0, "no particles with particles on");
            Assert.AreEqual(0f, bucket.LoadM3, 1e-6f, "the bucket holds soil with particles on");
        }

        private static void TerrainCopies(Scene scene, LoadSceneMode mode)
        {
            foreach (var t in Object.FindObjectsOfType<Terrain>())
            {
                var copy = Object.Instantiate(t.terrainData);
                t.terrainData = copy;
                var collider = t.GetComponent<TerrainCollider>();
                if (collider != null) collider.terrainData = copy;
            }
        }

        private static int empties;

        /// <summary>SoilParticleSettings keeps itself (and its terrain) across scene loads: gone after each test, so the next
        /// test's scene makes its own - after the scene (its GNSS sensors read the terrain's transform).</summary>
        [UnityTearDown]
        public IEnumerator DropTheSoil()
        {
            LogAssert.ignoreFailingMessages = true;
            var loaded = SceneManager.GetSceneByPath(Scene);
            SceneManager.SetActiveScene(SceneManager.CreateScene($"CapSoilTestEmpty{++empties}"));
            if (loaded.IsValid() && loaded.isLoaded)
                yield return SceneManager.UnloadSceneAsync(loaded);
            if (SoilParticleSettings.instance != null)
            {
                Object.Destroy(SoilParticleSettings.instance.gameObject);
                SoilParticleSettings.instance = null;
            }
            yield return null;
        }

        private static float MouthDeg(BucketRocks b) =>
            Vector3.Angle(b.transform.parent.TransformDirection(b.Mouth.direction), Vector3.up);

        private static IEnumerator Steps(int n)
        {
            for (int i = 0; i < n; i++) yield return new WaitForFixedUpdate();
        }

        /// <summary>The terrain's volume [m3] above its base in a (2 half + 1) square of heightmap points around a point.</summary>
        private static float Volume(Terrain t, Vector3 world, int half)
        {
            var d = t.terrainData;
            int res = d.heightmapResolution;
            Vector3 rel = world - t.transform.position;
            int cx = Mathf.RoundToInt(rel.x / d.size.x * (res - 1)), cz = Mathf.RoundToInt(rel.z / d.size.z * (res - 1));
            int x0 = Mathf.Clamp(cx - half, 0, res - 1), z0 = Mathf.Clamp(cz - half, 0, res - 1);
            int w = Mathf.Min(2 * half + 1, res - x0), h = Mathf.Min(2 * half + 1, res - z0);
            float[,] heights = d.GetHeights(x0, z0, w, h);
            double sum = 0;
            foreach (var v in heights) sum += v;
            return (float)(sum * d.size.y * (d.size.x / (res - 1)) * (d.size.z / (res - 1)));
        }

        private static ArticulationDrive WithTarget(ArticulationDrive d, float deg)
        {
            d.target = deg;
            return d;
        }

        private static void Pose(System.Collections.Generic.Dictionary<string, ArticulationBody> j, float[] q)
        {
            j["boom_link"].xDrive = WithTarget(j["boom_link"].xDrive, q[0] * Mathf.Rad2Deg);
            j["arm_link"].xDrive = WithTarget(j["arm_link"].xDrive, q[1] * Mathf.Rad2Deg);
            j["bucket_link"].xDrive = WithTarget(j["bucket_link"].xDrive, q[2] * Mathf.Rad2Deg);
        }

        private static void SetBucket(System.Collections.Generic.Dictionary<string, ArticulationBody> j, float rad)
        {
            j["bucket_link"].xDrive = WithTarget(j["bucket_link"].xDrive, rad * Mathf.Rad2Deg);
        }

        private static float GroundAt(Vector3 p)
        {
            foreach (var t in Terrain.activeTerrains)
            {
                Vector3 o = t.transform.position, size = t.terrainData.size;
                if (p.x >= o.x && p.x <= o.x + size.x && p.z >= o.z && p.z <= o.z + size.z)
                    return t.SampleHeight(p) + o.y;
            }
            return 0f;
        }

        /// <summary>A machine moved (its ground clearance kept) and its articulations with it, at rest.</summary>
        private static void Teleport(Transform root, Vector3 to, Quaternion rotation)
        {
            Vector3 from = root.position;
            to.y = from.y + (GroundAt(to) - GroundAt(from));
            root.SetPositionAndRotation(to, rotation);
            foreach (var ab in root.GetComponentsInChildren<ArticulationBody>())
            {
                if (!ab.isRoot) continue;
                ab.TeleportRoot(ab.transform.position, ab.transform.rotation);
                ab.velocity = Vector3.zero;
                ab.angularVelocity = Vector3.zero;
            }
        }

        /// <summary>The carrier turned so its bed's length runs along ``along`` (its rear toward the excavator) and moved
        /// so the bed's centre is under ``point``.</summary>
        private static void PlaceBedUnder(Transform root, CapBedLoad bed, Vector3 point, Vector3 along)
        {
            Vector3 fwd = Vector3.ProjectOnPlane(bed.transform.forward, Vector3.up).normalized;
            Quaternion turn = Quaternion.AngleAxis(Vector3.SignedAngle(fwd, along, Vector3.up), Vector3.up);
            Vector3 bedCenter = bed.transform.TransformPoint((bed.Bed.min + bed.Bed.max) / 2f);
            Vector3 rel = turn * (bedCenter - root.position);
            Teleport(root, new Vector3(point.x - rel.x, 0f, point.z - rel.z), turn * root.rotation);
        }
    }
}
#endif
