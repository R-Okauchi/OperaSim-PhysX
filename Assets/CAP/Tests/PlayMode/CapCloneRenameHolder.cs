using System.Collections.Generic;
using UnityEngine;

namespace CapKit.Tests
{
    /// <summary>A component holding names the way a machine's resolved them when it started (CapCloneRenameTests).</summary>
    public class CapCloneRenameHolder : MonoBehaviour
    {
        public string topic = "mst110cr/global_pose";
        [SerializeField] private string frame = "/mst110cr/kit/imu";
        private string robot = "mst110cr";
        public string longer = "mst110cr_2/odom";
        public string unrelated = "zx200/front_cmd";
        public string[] topics = { "mst110cr/joint_states", "tf" };
        public List<string> list = new List<string> { "mst110cr/cmd_vel" };

        public string Frame => frame;
        public string Robot => robot;
    }
}
