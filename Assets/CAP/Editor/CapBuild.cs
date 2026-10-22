using System;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEditor.Build.Reporting;
using UnityEngine;

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
        public static void Linux()
        {
            string output = Environment.GetEnvironmentVariable("CAP_BUILD_OUT");
            if (string.IsNullOrWhiteSpace(output)) output = "Builds/linux";
            string scenes = Environment.GetEnvironmentVariable("CAP_BUILD_SCENES");
            if (string.IsNullOrWhiteSpace(scenes)) scenes = "Assets/Scenes/SimpleScene.unity";
            var list = scenes.Split(',').Select(s => s.Trim()).Where(s => s.Length > 0).ToArray();
            foreach (var s in list)
            {
                if (!File.Exists(s)) throw new FileNotFoundException($"[CapBuild] no scene {s}");
            }
            Directory.CreateDirectory(output);
            var options = new BuildPlayerOptions
            {
                scenes = list,
                locationPathName = Path.Combine(output, "OperaSim.x86_64"),
                target = BuildTarget.StandaloneLinux64,
                targetGroup = BuildTargetGroup.Standalone,
                options = BuildOptions.None,
            };
            BuildReport report = BuildPipeline.BuildPlayer(options);
            var summary = report.summary;
            Debug.Log($"[CapBuild] {summary.result}: {summary.outputPath}, {summary.totalSize / 1e6:F0} MB in " +
                      $"{summary.totalTime.TotalSeconds:F0} s, {summary.totalErrors} errors, {summary.totalWarnings} warnings " +
                      $"({string.Join(", ", list)})");
            if (Application.isBatchMode)
            {
                EditorApplication.Exit(summary.result == BuildResult.Succeeded ? 0 : 1);
            }
        }
    }
}
