#if UNITY_EDITOR
using NUnit.Framework;
using UnityEngine;

namespace CapKit.Tests
{
    /// <summary>
    /// CapScenarioOps.RenameMachine (11-01, the chain's rehearsal 20): a clone of a machine of the scene holds the names
    /// its source resolved when it started ("mst110cr/global_pose"); renamed before it wakes, its topics, frames and
    /// robot name are its own - each segment equal to the source's name, never a name that only begins with it, and
    /// an array it shares with its source not changed under it.
    ///   Unity -batchmode -projectPath ... -runTests -testPlatform PlayMode -testFilter CapCloneRename -testResults out.xml
    /// </summary>
    public class CapCloneRenameTests
    {
        [Test]
        public void EachSegmentEqualToTheNameIsRenamed()
        {
            Assert.AreEqual("mst110cr_2/global_pose", CapScenarioOps.RenameSegment("mst110cr/global_pose", "mst110cr", "mst110cr_2"));
            Assert.AreEqual("/mst110cr_2/kit/lidar/roof/points",
                            CapScenarioOps.RenameSegment("/mst110cr/kit/lidar/roof/points", "mst110cr", "mst110cr_2"));
            Assert.AreEqual("mst110cr_2", CapScenarioOps.RenameSegment("mst110cr", "mst110cr", "mst110cr_2"));
            const string longer = "mst110cr_2/odom";
            Assert.AreSame(longer, CapScenarioOps.RenameSegment(longer, "mst110cr", "mst110cr_9"));
            const string body = "mst110cr-body";
            Assert.AreSame(body, CapScenarioOps.RenameSegment(body, "mst110cr", "mst110cr_2"));
            Assert.IsNull(CapScenarioOps.RenameSegment(null, "mst110cr", "mst110cr_2"));
        }

        [Test]
        public void AClonesComponentsHoldItsOwnNames()
        {
            var go = new GameObject("mst110cr_2");
            var child = new GameObject("body");
            child.transform.SetParent(go.transform);
            child.SetActive(false);                                    // asleep, as a clone is renamed
            var h = child.AddComponent<CapCloneRenameHolder>();
            var shared = h.topics;
            int n = CapScenarioOps.RenameMachine(go, "mst110cr", "mst110cr_2");
            Assert.AreEqual(5, n);
            Assert.AreEqual("mst110cr_2/global_pose", h.topic);
            Assert.AreEqual("/mst110cr_2/kit/imu", h.Frame);
            Assert.AreEqual("mst110cr_2", h.Robot);
            Assert.AreEqual("mst110cr_2/odom", h.longer);
            Assert.AreEqual("zx200/front_cmd", h.unrelated);
            Assert.AreEqual(new[] { "mst110cr_2/joint_states", "tf" }, h.topics);
            Assert.AreEqual("mst110cr/joint_states", shared[0]);       // the source's array not changed under it
            Assert.AreEqual("mst110cr_2/cmd_vel", h.list[0]);
            Object.DestroyImmediate(go);
        }
    }
}
#endif
