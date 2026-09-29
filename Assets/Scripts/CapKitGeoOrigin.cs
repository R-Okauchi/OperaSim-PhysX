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
            // Always the kit's own system: the scene may carry another GeoCoordinateSystem (OperaSim's
            // own GNSS) at an arbitrary pose, and a kit sensor bound to it reports LLH about THAT pose.
            // Live 2026-09-14 21:24: every kit fix came out rotated 90 deg and offset (2.7, -9.6, 2.4) m
            // from the ROS world origin, so the estimator placed zx120 28.8 m and zx200 56 m off.
            foreach (var gnss in GetComponentsInChildren<GNSSSensor>(true)) s_systemField.SetValue(gnss, system);
            // The kit IMUs sample on the physics step (see CapKitFixedRateImu). From the scene root: the kit IMU
            // mounts hang under the machine's links, not under kit_hub; installing is idempotent across hubs.
            CapKitFixedRateImu.Install(transform.root.gameObject);
        }

        private GeoCoordinateSystem Resolve()
        {
            if (s_system != null) return s_system;
            var existing = GameObject.Find(OriginObjectName);
            if (existing != null && existing.TryGetComponent<GeoCoordinateSystem>(out var found))
            {
                s_system = found;
                return s_system;
            }
            var go = new GameObject(OriginObjectName);
            go.SetActive(false);
            // The kit's site frame is ENU with the ROS world origin as the survey origin: the
            // deployment's site_origin_llh is the LLH of ROS (0,0,0) and ROS x = east, y = north,
            // exactly what a real RTK deployment defines. UnitySensors maps the system's LOCAL x to
            // longitude (east) and local z to latitude (north); ROS-TCP maps ROS x to Unity z and
            // ROS y to Unity -x. Rotating the origin object -90 deg about Y makes local x = Unity z
            // (= ROS x = east) and local z = -Unity x (= ROS y = north). Without this the LLH frame
            // is a +90 deg yaw of the ROS world and no deployment field can honestly describe it.
            go.transform.SetPositionAndRotation(Vector3.zero, Quaternion.Euler(0f, -90f, 0f));
            var system = go.AddComponent<GeoCoordinateSystem>();
            if (s_coordinateField != null)
                s_coordinateField.SetValue(system, new GeoCoordinate(latitude, longitude, altitude));
            go.SetActive(true);
            s_system = system;
            Debug.Log("[CAP Kit] Created " + OriginObjectName + " at ROS world origin, ENU-aligned (lat " + latitude +
                      ", lon " + longitude + ", alt " + altitude + ")");
            return s_system;
        }
    }
}
