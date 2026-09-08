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
            if (Environment.GetEnvironmentVariable("CAP_KEEP_VISUALIZERS") != "1") SilenceVisualizers();
            Debug.Log("[CAP Kit] EnterPlay scene=" + scene + " ros_ip=" + (ip ?? "(prefab default)"));
            // A bare delayCall can fire while scripts/assets are still (re)importing after a batch edit, and the
            // play request is then dropped silently. Poll until the editor is idle, then enter Play once.
            EditorApplication.update += EnterWhenIdle;
        }

        public static void ExitPlay() => EditorApplication.ExitPlaymode();

        private static void EnterWhenIdle()
        {
            if (EditorApplication.isCompiling || EditorApplication.isUpdating) return;
            EditorApplication.update -= EnterWhenIdle;
            if (EditorApplication.isPlayingOrWillChangePlaymode) return;
            Debug.Log("[CAP Kit] editor idle -> EnterPlaymode");
            EditorApplication.EnterPlaymode();
        }

        /// <summary>
        /// UnitySensors point-cloud visualizers draw MeshTopology.Points with a shader that has no PSIZE output on
        /// Metal; Unity logs one warning per draw (hundreds of lines per second with several LiDARs) and the draw
        /// itself costs frame time the 10 Hz kit LiDAR rate needs. Disable them in the loaded scene (in memory only,
        /// nothing is saved); set CAP_KEEP_VISUALIZERS=1 to keep them.
        /// </summary>
        private static void SilenceVisualizers()
        {
            int behaviours = 0, renderers = 0;
            foreach (var behaviour in Resources.FindObjectsOfTypeAll<Behaviour>())
            {
                if (behaviour == null || behaviour.gameObject.scene.rootCount == 0) continue;
                string ns = behaviour.GetType().Namespace;
                if (ns != null && ns.StartsWith("UnitySensors.Visualization", StringComparison.Ordinal) && behaviour.enabled)
                { behaviour.enabled = false; behaviours++; }
            }
            foreach (var renderer in Resources.FindObjectsOfTypeAll<Renderer>())
            {
                if (renderer == null || renderer.gameObject.scene.rootCount == 0 || !renderer.enabled) continue;
                var material = renderer.sharedMaterial;
                if (material != null && material.shader != null && material.shader.name.StartsWith("UnitySensors/", StringComparison.Ordinal))
                { renderer.enabled = false; renderers++; }
            }
            Debug.Log("[CAP Kit] silenced UnitySensors visualizers: behaviours=" + behaviours + " renderers=" + renderers);
        }

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
