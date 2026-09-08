using System;
using UnityEditor;
using UnityEngine;
using Unity.Mathematics;
using UnitySensors.Sensor.LiDAR;

namespace CapKit.Editor
{
    public static class CapScanPatternGenerator
    {
        public const string Folder = "Assets/CAP/ScanPatterns";
        public const string Hemisphere = "Hemisphere360x90_gap32";
        public const string Flash = "Flash120x90_lowres";

        [MenuItem("CAP/Perception Kit/Generate Scan Patterns")]
        public static void GenerateScanPatterns()
        {
            CapKitRig.RunMenu("Generate Scan Patterns", report =>
            {
                foreach (string name in new[] { Hemisphere, Flash })
                {
                    try
                    {
                        Generate(name);
                        report.Info(name + ": saved 6912 directions.");
                    }
                    catch (Exception e) { report.Error(name + ": " + e.Message); }
                }
            });
        }

        public static ScanPattern GetOrCreate(string name)
        {
            ValidateName(name);
            var pattern = AssetDatabase.LoadAssetAtPath<ScanPattern>(Folder + "/" + name + ".asset");
            if (pattern == null) return Generate(name);
            if (pattern.scans == null || pattern.size <= 0 || pattern.scans.Length != pattern.size)
                throw new InvalidOperationException(name + " is invalid; run Generate Scan Patterns.");
            return pattern;
        }

        public static ScanPattern Generate(string name)
        {
            ValidateName(name);
            if (!AssetDatabase.IsValidFolder("Assets/CAP")) AssetDatabase.CreateFolder("Assets", "CAP");
            if (!AssetDatabase.IsValidFolder(Folder)) AssetDatabase.CreateFolder("Assets/CAP", "ScanPatterns");
            string path = Folder + "/" + name + ".asset";
            var pattern = AssetDatabase.LoadAssetAtPath<ScanPattern>(path);
            bool create = pattern == null;
            if (create && AssetDatabase.LoadMainAssetAtPath(path) != null)
                throw new InvalidOperationException("A non-ScanPattern asset already exists at " + path);
            if (create) pattern = ScriptableObject.CreateInstance<ScanPattern>();
            bool hemisphere = name == Hemisphere;
            Fill(pattern, hemisphere ? 0f : -60f, hemisphere ? 360f : 60f,
                hemisphere ? 0f : -45f, hemisphere ? 90f : 45f);
            // TODO: Airy's every-10th-frame 32-degree azimuth gap needs a temporal scan model.
            // A static pattern intentionally contains no frame-dependent gaps.
            if (create) AssetDatabase.CreateAsset(pattern, path);
            else EditorUtility.SetDirty(pattern);
            AssetDatabase.SaveAssetIfDirty(pattern);
            return pattern;
        }

        private static void Fill(ScanPattern scan, float minAzimuth, float maxAzimuth,
            float minZenith, float maxZenith)
        {
            const int azimuths = 96, elevations = 72;
            scan.size = azimuths * elevations;
            scan.scans = new float3[scan.size];
            scan.minAzimuthAngle = minAzimuth;
            scan.maxAzimuthAngle = maxAzimuth;
            scan.minZenithAngle = float.MaxValue;
            scan.maxZenithAngle = float.MinValue;
            for (int a = 0; a < azimuths; a++)
            {
                // Match GenerateFromSpecification: the upper azimuth bound is exclusive.
                float azimuth = Mathf.Lerp(minAzimuth, maxAzimuth, (float)a / azimuths);
                for (int e = 0; e < elevations; e++)
                {
                    float zenith = Mathf.Lerp(minZenith, maxZenith, (float)e / (elevations - 1));
                    Vector3 direction = Quaternion.Euler(-zenith, azimuth, 0f) * Vector3.forward;
                    scan.scans[a * elevations + e] = new float3(direction.x, direction.y, direction.z);
                    scan.minZenithAngle = Mathf.Min(scan.minZenithAngle, zenith);
                    scan.maxZenithAngle = Mathf.Max(scan.maxZenithAngle, zenith);
                }
            }
        }

        private static void ValidateName(string name)
        {
            if (name != Hemisphere && name != Flash)
                throw new ArgumentException("Unsupported CAP scan pattern: " + name);
        }
    }
}
