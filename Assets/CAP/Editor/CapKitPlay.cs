using System;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace CapKit.Editor
{
    /// <summary>
    /// Headless-friendly Play control for CI/dev loops without the MCP bridge:
    ///   Unity -projectPath ... -executeMethod CapKit.Editor.CapKitPlay.EnterPlay   (GUI editor stays open, Play running)
    /// Scene: env CAP_SCENE (default Assets/Scenes/SimpleScene.unity). Set CAP_ROS_IP to override the ROS-TCP endpoint
    /// address on the ROSConnectionPrefab in the loaded scene (dev machine vs. cluster).
    /// </summary>
    public static class CapKitPlay
    {
        public static void EnterPlay()
        {
            string scene = Environment.GetEnvironmentVariable("CAP_SCENE");
            if (string.IsNullOrWhiteSpace(scene)) scene = "Assets/Scenes/SimpleScene.unity";
            EditorSceneManager.OpenScene(scene, OpenSceneMode.Single);
            string ip = Environment.GetEnvironmentVariable("CAP_ROS_IP");
            if (!string.IsNullOrWhiteSpace(ip)) OverrideRosIp(ip);
            Debug.Log("[CAP Kit] EnterPlay scene=" + scene + " ros_ip=" + (ip ?? "(prefab default)"));
            EditorApplication.delayCall += () => EditorApplication.EnterPlaymode();
        }

        public static void ExitPlay() => EditorApplication.ExitPlaymode();

        private static void OverrideRosIp(string ip)
        {
            // ROSConnection lives on the ROSConnectionPrefab instance (or is created at runtime from Resources).
            foreach (var conn in Resources.FindObjectsOfTypeAll<Unity.Robotics.ROSTCPConnector.ROSConnection>())
            {
                var so = new SerializedObject(conn);
                var prop = so.FindProperty("m_RosIPAddress");
                if (prop == null) continue;
                prop.stringValue = ip;
                so.ApplyModifiedPropertiesWithoutUndo();
                Debug.Log("[CAP Kit] ROS IP override -> " + ip + " on " + conn.name);
            }
        }
    }
}
