using System.Collections;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;

namespace CapKit.Tests
{
    /// <summary>The scenario's terrain shaping (CapScenarioOps.ShapeTerrain): a ramp from one height to another along ROS
    /// x or y - a machine's way up onto a bench (11-02, CAP's chain starting from the machines' yard). On a copy of the
    /// scene's TerrainData: a test's Play would keep a change to the asset itself.</summary>
    public class CapScenarioShapeTests
    {
        private const string Scene = "Assets/Scenes/SimpleScene.unity";

        [UnityTest, Timeout(300000)]
        public IEnumerator ARampGoesFromItsNearHeightToItsFarOneBesideItsBench()
        {
            LogAssert.ignoreFailingMessages = true;
            yield return CapTestScene.LoadFresh(Scene);
            yield return null;
            var terrain = Terrain.activeTerrain;
            Assert.IsNotNull(terrain, "no terrain");
            terrain.terrainData = Object.Instantiate(terrain.terrainData);
            var collider = terrain.GetComponent<TerrainCollider>();
            if (collider != null) collider.terrainData = terrain.terrainData;
            var cfg = new CapScenarioOps.Config
            {
                terrain = new[]
                {
                    new CapScenarioOps.TerrainOp { op = "flatten", x0 = 18f, y0 = -48f, x1 = 30f, y1 = -40f, z = 1.5f },
                    new CapScenarioOps.TerrainOp { op = "ramp", x0 = 20f, y0 = -40f, x1 = 28f, y1 = -31.5f, z = 1.5f,
                                                   z_to = 0f, along = "y" },
                    new CapScenarioOps.TerrainOp { op = "ramp", x0 = 30f, y0 = -48f, x1 = 38.5f, y1 = -42f, z = 1.5f,
                                                   z_to = 0f },                            // along ROS x (its default)
                }
            };
            CapScenarioOps.ShapeTerrain(cfg);
            float H(float rx, float ry) => terrain.SampleHeight(new Vector3(-ry, 0f, rx)) + terrain.transform.position.y;
            Assert.AreEqual(1.5f, H(24f, -44f), 0.02f, "the bench");
            Assert.AreEqual(1.5f, H(24f, -39.9f), 0.05f, "the ramp's top, at its bench");
            Assert.AreEqual(0.75f, H(24f, -35.75f), 0.05f, "the ramp's middle");
            Assert.AreEqual(0.0f, H(24f, -31.6f), 0.05f, "the ramp's foot");
            Assert.AreEqual(H(21f, -35.75f), H(27f, -35.75f), 0.02f, "level across it");
            Assert.AreEqual(0.75f, H(34.25f, -45f), 0.05f, "a ramp along x: its middle");
            Assert.AreEqual(0.0f, H(38.4f, -45f), 0.05f, "a ramp along x: its foot");
        }

        [UnityTest, Timeout(300000)]
        public IEnumerator AMesaStandsFlatOnTopAndSlopesDownToTheGroundAllRound()
        {
            LogAssert.ignoreFailingMessages = true;
            yield return CapTestScene.LoadFresh(Scene);
            yield return null;
            var terrain = Terrain.activeTerrain;
            Assert.IsNotNull(terrain, "no terrain");
            terrain.terrainData = Object.Instantiate(terrain.terrainData);
            var collider = terrain.GetComponent<TerrainCollider>();
            if (collider != null) collider.terrainData = terrain.terrainData;
            float H(float rx, float ry) => terrain.SampleHeight(new Vector3(-ry, 0f, rx)) + terrain.transform.position.y;
            float side = H(28f + 2.8f, -45f), corner = H(28f + 2f, -42f + 2f), out1 = H(28f + 6f, -45f);
            CapScenarioOps.ShapeTerrain(new CapScenarioOps.Config
            {
                terrain = new[] { new CapScenarioOps.TerrainOp { op = "mesa", x0 = 20f, y0 = -48f, x1 = 28f, y1 = -42f,
                                                                 z = 1.5f, run_m = 5.6f } }
            });
            Assert.AreEqual(1.5f, H(24f, -45f), 0.02f, "its top");
            Assert.AreEqual(1.5f, H(27.9f, -42.1f), 0.03f, "its top's corner");
            Assert.AreEqual(Mathf.Max((1.5f + side) / 2f, side), H(28f + 2.8f, -45f), 0.05f, "half way down its side");
            float off = Mathf.Sqrt(8f) / 5.6f;                     // the corner: by the distance to the top
            Assert.AreEqual(Mathf.Max(Mathf.Lerp(1.5f, corner, off), corner), H(28f + 2f, -42f + 2f), 0.06f, "its corner");
            Assert.AreEqual(out1, H(28f + 6f, -45f), 0.01f, "past its side: the ground as it was");
        }
    }
}
