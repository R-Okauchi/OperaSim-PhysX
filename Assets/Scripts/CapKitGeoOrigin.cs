using System.Reflection;
using UnityEngine;
using UnitySensors.Sensor.GNSS;
using UnitySensors.Utils.GeoCoordinate;

namespace CapKit
{
    /// <summary>
    /// Binds the kit's GNSSSensor components to a site GeoCoordinateSystem at runtime.
    ///
    /// GNSSSensor needs a GeoCoordinateSystem, which must be a world-fixed scene object (it
    /// converts Unity world XZ into lat/lon relative to its own transform). A prefab cannot
    /// reference a scene object, so this component, placed on kit_hub by CapKitRig, finds the
    /// scene's GeoCoordinateSystem or creates one at the Unity world origin using the
    /// deployment's perception.site_origin_llh. Without it every GNSSSensor.Update throws.
    /// </summary>
    [DisallowMultipleComponent]
    [DefaultExecutionOrder(-100)]
    public sealed class CapKitGeoOrigin : MonoBehaviour
    {
        public const string OriginObjectName = "CapKitGeoOrigin";

        public double latitude = 36.073406033;   // overwritten by CapKitRig from the contract
        public double longitude = 140.041676353;
        public double altitude = 68.2;

        private static GeoCoordinateSystem s_system;
        private static readonly FieldInfo s_systemField =
            typeof(GNSSSensor).GetField("_coordinateSystem", BindingFlags.Instance | BindingFlags.NonPublic);
        private static readonly FieldInfo s_coordinateField =
            typeof(GeoCoordinateSystem).GetField("_coordinate", BindingFlags.Instance | BindingFlags.NonPublic);

        private void Awake()
        {
            var system = Resolve();
            if (system == null || s_systemField == null)
            {
                Debug.LogWarning("[CAP Kit] GeoCoordinateSystem unavailable; disabling kit GNSS sensors on " + name);
                foreach (var gnss in GetComponentsInChildren<GNSSSensor>(true)) gnss.enabled = false;
                return;
            }
            foreach (var gnss in GetComponentsInChildren<GNSSSensor>(true))
            {
                if (s_systemField.GetValue(gnss) == null) s_systemField.SetValue(gnss, system);
            }
        }

        private GeoCoordinateSystem Resolve()
        {
            if (s_system != null) return s_system;
            s_system = FindObjectOfType<GeoCoordinateSystem>();
            if (s_system != null) return s_system;

            // Create the origin inactive so the serialized coordinate is set before its Awake builds the converter.
            var go = new GameObject(OriginObjectName);
            go.SetActive(false);
            var system = go.AddComponent<GeoCoordinateSystem>();
            if (s_coordinateField != null)
                s_coordinateField.SetValue(system, new GeoCoordinate(latitude, longitude, altitude));
            go.SetActive(true);
            s_system = system;
            Debug.Log("[CAP Kit] Created " + OriginObjectName + " at Unity world origin (lat " + latitude +
                      ", lon " + longitude + ", alt " + altitude + ")");
            return s_system;
        }
    }
}
