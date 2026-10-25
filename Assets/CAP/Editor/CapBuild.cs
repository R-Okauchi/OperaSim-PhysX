using System;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEditor.Build.Reporting;
using UnityEngine;
using UnityEngine.Rendering;

namespace CapKit.Editor
{
    /// <summary>
    /// The simulator as a Linux player, for a host without the editor (the December plan's W2: Unity headless on a
    /// lab host, CAP's stack beside it):
    ///   Unity -batchmode -quit -projectPath ... -executeMethod CapKit.Editor.CapBuild.Linux
    /// env CAP_BUILD_OUT (default Builds/linux: ignored), CAP_BUILD_SCENES (comma-separated, default SimpleScene).
    /// The player is configured as the editor's runs are, by CapCycleRecorder's JSON (CAP_UNITY_RECORD_CONFIG: the
    /// ROS-TCP endpoint, the machines' places, the soil); the scene files and the project settings are not touched.
    /// The terrain's soil needs a GPU device (its compute shaders): run it with a display (Xvfb and Mesa will do), not
    /// with -nographics.
    /// </summary>
    public static class CapBuild
    {
        /// <summary>The simulator as a macOS player (Metal, fast cameras: the videos of a run in the cluster, its ROS-TCP to
        /// the cluster's bridge): Builds/mac/OperaSim.app by default.
        ///   Unity -batchmode -quit -buildTarget OSXUniversal -projectPath ... -executeMethod CapKit.Editor.CapBuild.Mac
        /// </summary>
        public static void Mac()
        {
            string output = Environment.GetEnvironmentVariable("CAP_BUILD_OUT");
            if (string.IsNullOrWhiteSpace(output)) output = "Builds/mac";
            var list = Scenes();
            Directory.CreateDirectory(output);
            var report = BuildPipeline.BuildPlayer(new BuildPlayerOptions
            {
                scenes = list,
                locationPathName = Path.Combine(output, "OperaSim.app"),
                target = BuildTarget.StandaloneOSX,
                targetGroup = BuildTargetGroup.Standalone,
                options = BuildOptions.None,
            });
            Finish(report, list);
        }

        private static string[] Scenes()
        {
            string scenes = Environment.GetEnvironmentVariable("CAP_BUILD_SCENES");
            if (string.IsNullOrWhiteSpace(scenes)) scenes = "Assets/Scenes/SimpleScene.unity";
            var list = scenes.Split(',').Select(s => s.Trim()).Where(s => s.Length > 0).ToArray();
            foreach (var s in list)
            {
                if (!File.Exists(s)) throw new FileNotFoundException($"[CapBuild] no scene {s}");
            }
            return list;
        }

        private static void Finish(BuildReport report, string[] list)
        {
            var summary = report.summary;
            Debug.Log($"[CapBuild] {summary.result}: {summary.outputPath}, {summary.totalSize / 1e6:F0} MB in " +
                      $"{summary.totalTime.TotalSeconds:F0} s, {summary.totalErrors} errors, {summary.totalWarnings} warnings " +
                      $"({string.Join(", ", list)})");
            if (Application.isBatchMode)
            {
                EditorApplication.Exit(summary.result == BuildResult.Succeeded ? 0 : 1);
            }
        }

        public static void Linux()
        {
            string output = Environment.GetEnvironmentVariable("CAP_BUILD_OUT");
            if (string.IsNullOrWhiteSpace(output)) output = "Builds/linux";
            var list = Scenes();
            Directory.CreateDirectory(output);
            var options = new BuildPlayerOptions
            {
                scenes = list,
                locationPathName = Path.Combine(output, "OperaSim.x86_64"),
                target = BuildTarget.StandaloneLinux64,
                targetGroup = BuildTargetGroup.Standalone,
                options = BuildOptions.None,
            };
            // OpenGL Core only: Unity 2022.3.4f1's Vulkan breaks the terrain's soil, and an automatic choice takes
            // Vulkan first on Linux - the project's own setting put back after the build
            bool automatic = PlayerSettings.GetUseDefaultGraphicsAPIs(BuildTarget.StandaloneLinux64);
            GraphicsDeviceType[] apis = PlayerSettings.GetGraphicsAPIs(BuildTarget.StandaloneLinux64);
            BuildReport report;
            try
            {
                PlayerSettings.SetUseDefaultGraphicsAPIs(BuildTarget.StandaloneLinux64, false);
                PlayerSettings.SetGraphicsAPIs(BuildTarget.StandaloneLinux64, new[] { GraphicsDeviceType.OpenGLCore });
                report = BuildPipeline.BuildPlayer(options);
            }
            finally
            {
                PlayerSettings.SetGraphicsAPIs(BuildTarget.StandaloneLinux64, apis);
                PlayerSettings.SetUseDefaultGraphicsAPIs(BuildTarget.StandaloneLinux64, automatic);
            }
            Finish(report, list);
        }
    }
}
