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
    }
}
