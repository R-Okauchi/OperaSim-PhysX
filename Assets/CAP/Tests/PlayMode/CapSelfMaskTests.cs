#if UNITY_EDITOR
using System.Collections;
using System.Collections.Generic;
using System.IO;
using NUnit.Framework;
using UnityEditor;
using UnityEngine;
using UnityEngine.TestTools;

namespace CapKit.Tests
{
    /// <summary>
    /// What a swinging machine's kit sees of its own undercarriage, against its self mask (Assets/CAP/sensor_rig.json:
    /// the contract's kit.self_mask in the hub frame, grown by the margin CAP's kit mapper grows it by). The kit's
    /// LiDARs cast rays at colliders (UnitySensors' RaycastLiDARSensor, Physics.DefaultRaycastLayers): from each of the
    /// machine's LiDARs, at every swing angle, rays all round; where one first meets the undercarriage (every collider
    /// off the hub link: the upper and the arm swing with it), the point must lie in the mask, or the kit maps its own
    /// machine round it as its upper swings - its guard then stops on it (11-02, the chain's r29-r32: zx200's
    /// upper-mounted fr LiDAR mapped its body's box, 5 x 1 x 5 m round its hub, 3.0-3.6 m from its swing axis;
    /// CapBodyBoxes now takes the box out of the rays, as CapCycleRecorder does here).
    ///   Unity -batchmode -projectPath ... -runTests -testPlatform PlayMode -testFilter CapSelfMask -testResults out.xml
    /// </summary>
    public class CapSelfMaskTests
    {
        [System.Serializable] private sealed class RigFile { public RigMachine[] machines; }
        [System.Serializable] private sealed class RigMachine { public string machine_id; public string hub_link; public RigMask self_mask; public bool swing_free; }
        [System.Serializable] private sealed class RigMask { public RigHull hull; public RigBox undercarriage; public RigBox[] extra_boxes; }
        [System.Serializable] private sealed class RigHull { public float[] center_xy_m; public float radius_m, z_min_m, z_max_m; }
        [System.Serializable] private sealed class RigBox { public float[] center_xy_m; public float length_m, width_m, z_min_m, z_max_m; }

        private const float MarginM = 0.10f;      // CAP's kit mapper's margin round its self mask
        private const int Rays = 20000;           // all round from each LiDAR: some 1.4 deg apart
        private const float SwingStepDeg = 10f;

        private static void Log(string s) => Debug.Log("[CapSelfMaskTest] " + s);

        private static Transform Find(Transform root, string name)
        {
            foreach (var t in root.GetComponentsInChildren<Transform>(true)) if (t.name == name) return t;
            return null;
        }

        /// <summary>CAP's self mask's union for a swinging kit (swing_free, upper_swings): the hull's
        /// cylinder, the undercarriage's circle at its heights, the extra boxes, each grown by the margin.</summary>
        private static bool Masked(RigMask m, float x, float y, float z)
        {
            var h = m.hull;
            if (Sq(x - h.center_xy_m[0]) + Sq(y - h.center_xy_m[1]) <= Sq(h.radius_m + MarginM)
                && z >= h.z_min_m - MarginM && z <= h.z_max_m + MarginM) return true;
            var u = m.undercarriage;
            float r = new Vector2(Mathf.Abs(u.center_xy_m[0]) + u.length_m / 2f, Mathf.Abs(u.center_xy_m[1]) + u.width_m / 2f).magnitude;
            if (x * x + y * y <= Sq(r + MarginM) && z >= u.z_min_m - MarginM && z <= u.z_max_m + MarginM) return true;
            foreach (var e in m.extra_boxes ?? new RigBox[0])
            {
                if (Mathf.Abs(x - e.center_xy_m[0]) <= e.length_m / 2f + MarginM && Mathf.Abs(y - e.center_xy_m[1]) <= e.width_m / 2f + MarginM
                    && z >= e.z_min_m - MarginM && z <= e.z_max_m + MarginM) return true;
            }
            return false;
        }

        private static float Sq(float v) => v * v;

        private static Vector3[] AllRound(int n)
        {
            var d = new Vector3[n];
            float golden = Mathf.PI * (3f - Mathf.Sqrt(5f));
            for (int i = 0; i < n; i++)
            {
                float y = 1f - 2f * (i + 0.5f) / n, r = Mathf.Sqrt(1f - y * y);
                d[i] = new Vector3(Mathf.Cos(golden * i) * r, y, Mathf.Sin(golden * i) * r);
            }
            return d;
        }

        /// <summary>Points of the undercarriage the LiDARs see outside the mask, over a full swing: their count, and the
        /// one farthest from the axis.</summary>
        private static (int kept, string worst) Sweep(GameObject go, Transform hub, List<Transform> lidars, RigMask mask)
        {
            var dirs = AllRound(Rays);
            int kept = 0;
            float far = 0f;
            string worst = "none";
            Quaternion start = hub.localRotation;
            for (float a = 0f; a < 360f; a += SwingStepDeg)
            {
                hub.localRotation = start;
                hub.RotateAround(hub.position, hub.up, a);
                Physics.SyncTransforms();
                foreach (var lidar in lidars)
                {
                    foreach (var d in dirs)
                    {
                        if (!Physics.Raycast(lidar.position, d, out var hit, 30f, Physics.DefaultRaycastLayers)) continue;
                        var c = hit.collider.transform;
                        if (!c.IsChildOf(go.transform) || c.IsChildOf(hub)) continue;
                        // the hub frame, ROS: x forward (Unity z), y left (Unity -x), z up
                        Vector3 p = hub.InverseTransformPoint(hit.point);
                        float x = p.z, y = -p.x, z = p.y;
                        if (Masked(mask, x, y, z)) continue;
                        kept++;
                        float r = new Vector2(x, y).magnitude;
                        if (r > far)
                        {
                            far = r;
                            worst = $"{hit.collider.name} ({x:F2}, {y:F2}, {z:F2}) {r:F2} m from the axis, from {lidar.parent.name} at swing {a:F0} deg";
                        }
                    }
                }
            }
            hub.localRotation = start;
            Physics.SyncTransforms();
            return (kept, worst);
        }

        [UnityTest, Timeout(600000)]
        [TestCase("zx200", ExpectedResult = null)]
        [TestCase("zx120", ExpectedResult = null)]
        [TestCase("mst110cr", ExpectedResult = null)]
        public IEnumerator ItsLiDARsSeeNothingOfItsUndercarriageOutsideItsMaskAtAnySwing(string machine)
        {
            LogAssert.ignoreFailingMessages = true;
            var rig = JsonUtility.FromJson<RigFile>(File.ReadAllText("Assets/CAP/sensor_rig.json"));
            RigMachine spec = null;
            foreach (var m in rig.machines) if (m.machine_id == machine) spec = m;
            Assert.IsNotNull(spec, $"no {machine} in sensor_rig.json");
            Assert.IsTrue(spec.swing_free, $"{machine}'s kit is not on a swinging upper");
            var prefab = AssetDatabase.LoadAssetAtPath<GameObject>($"Assets/Prefab/{machine}.prefab");
            Assert.IsNotNull(prefab, $"no Assets/Prefab/{machine}.prefab");
            var isolated = new bool[32];
            for (int l = 0; l < 32; l++) isolated[l] = Physics.GetIgnoreLayerCollision(CapBodyBoxes.Layer, l);
            var go = Object.Instantiate(prefab, new Vector3(0f, 200f, 0f), Quaternion.identity);
            try
            {
                foreach (var ab in go.GetComponentsInChildren<ArticulationBody>()) ab.enabled = false;   // as built
                var rb = go.GetComponent<Rigidbody>();
                Assert.IsNotNull(rb, $"{machine}'s root has no Rigidbody");
                rb.isKinematic = true;
                yield return null;
                Physics.SyncTransforms();
                var hub = Find(go.transform, spec.hub_link);
                Assert.IsNotNull(hub, $"no {spec.hub_link} in {machine}");
                var lidars = new List<Transform>();
                foreach (var mb in go.GetComponentsInChildren<MonoBehaviour>(true))
                {
                    if (mb != null && mb.GetType().Name == "RaycastLiDARSensor") lidars.Add(mb.transform);
                }
                Assert.Greater(lidars.Count, 0, $"{machine} has no LiDAR");

                var box = go.GetComponent<BoxCollider>();
                Vector3 com = rb.centerOfMass, inertia = rb.inertiaTensor;
                var before = Sweep(go, hub, lidars, spec.self_mask);
                CapBodyBoxes.Isolate();
                Assert.IsTrue(CapBodyBoxes.Hide(rb), $"{machine}'s box not hidden");
                Physics.SyncTransforms();
                var after = Sweep(go, hub, lidars, spec.self_mask);
                Log($"{machine} ({lidars.Count} LiDARs, {Rays} rays each every {SwingStepDeg:F0} deg of swing): its " +
                    $"undercarriage seen outside its mask at {before.kept} points with its body's box (the farthest " +
                    $"{before.worst}), at {after.kept} with the box hidden (the farthest {after.worst})");

                // the box out of the rays only: the machine's mass and the soil's casts as they were
                Assert.IsTrue(CapBodyBoxes.MeetsNothing(box.gameObject.layer), "the box meets something on its new layer");
                Assert.That(Vector3.Distance(rb.centerOfMass, com), Is.LessThan(1e-4f), "its centre of mass moved");
                Assert.That(Vector3.Distance(rb.inertiaTensor, inertia), Is.LessThan(1e-3f * inertia.magnitude), "its inertia changed");
                Vector3 over = go.transform.TransformPoint(box.center + new Vector3(0.48f * box.size.x, box.size.y / 2f + 3f, 0.48f * box.size.z));
                Assert.IsFalse(Physics.Raycast(over, Vector3.down, out var past, 20f, Physics.DefaultRaycastLayers) && past.collider == box,
                               "a LiDAR's ray still meets the box");
                var down = new RaycastHit[64];
                int n = CapSoilGeometry.CastDown(over, 20f, ref down, QueryTriggerInteraction.Ignore);
                bool soil = false;
                for (int i = 0; i < n; i++) soil |= down[i].collider == box;
                Assert.IsTrue(soil, "the soil's cast no longer meets the box");

                Assert.AreEqual(0, after.kept, $"{machine}'s LiDARs see its undercarriage outside its mask: {after.worst}");
            }
            finally
            {
                Object.Destroy(go);
                for (int l = 0; l < 32; l++) Physics.IgnoreLayerCollision(CapBodyBoxes.Layer, l, isolated[l]);
            }
        }
    }
}
#endif
