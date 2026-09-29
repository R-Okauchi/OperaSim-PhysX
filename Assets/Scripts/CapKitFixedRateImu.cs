using System.Reflection;
using RosMessageTypes.Sensor;
using Unity.Robotics.ROSTCPConnector;
using Unity.Robotics.ROSTCPConnector.ROSGeometry;
using UnityEngine;
using UnitySensors.ROS.Publisher.IMU;
using UnitySensors.Sensor.IMU;

namespace CapKit
{
    /// <summary>
    /// A kit IMU sampled on the physics step instead of once per rendered frame.
    ///
    /// UnitySensors' IMUSensor and IMUMsgPublisher sample and publish in Update, at most once per frame, so the
    /// kit IMU rate was the editor's frame rate: behind another window, or with the Mac loaded, the editor fell to
    /// 6-43 fps, the IMU from 100 Hz to 6 Hz, the kit estimate degraded past its gate, and the machines stopped
    /// (live 2026-09-29 run57, run59). The physics steps keep sim time whatever the frame rate, so here each
    /// FixedUpdate takes a sample (at most rateHz, by sim time) and publishes it at once, with its own step's stamp.
    /// At once, not queued to the frame's Update: the kit GNSS and heading publish in Update stamped with the frame
    /// time, which is ahead of the frame's physics steps; arriving first, they advanced the estimator past the
    /// frame's IMU samples, which it then dropped as late - at 4.5 fps every burst, and the estimate ran on
    /// extrapolation alone until it tripped the reference check (live 2026-09-29 run63). A frame's FixedUpdates
    /// all run before its Updates, so these go out first. The quantities are exactly IMUSensor's - the same finite
    /// differences, the same gravity term, the same FLU conversion as IMUMsgSerializer - so the Python kit shim's
    /// corrections apply unchanged; only the sampling instants move from frames to physics steps.
    ///
    /// Installed at runtime by CapKitGeoOrigin (on every kit_hub): it replaces each kit_* IMUMsgPublisher it finds
    /// under the machine and disables that publisher and its IMUSensor. PlayerPrefs "CapKitFixedRateImu" = 0 keeps
    /// the per-frame sensors.
    /// </summary>
    [DisallowMultipleComponent]
    public sealed class CapKitFixedRateImu : MonoBehaviour
    {
        public const string PrefsKey = "CapKitFixedRateImu";

        public string topic;
        public string frameId;
        public float rateHz = 100f;

        /// <summary>Samples published since play started (all kit IMUs): diagnostics.</summary>
        public static long Published;

        private ROSConnection _ros;
        private Vector3 _lastPosition;
        private Vector3 _lastVelocity;
        private Quaternion _lastRotation;
        private bool _primed;
        private double _nextSampleTime;

        private static readonly BindingFlags Private = BindingFlags.Instance | BindingFlags.NonPublic;

        /// <summary>Replace the per-frame kit IMUs under <paramref name="machineRoot"/> (idempotent).</summary>
        public static int Install(GameObject machineRoot)
        {
            if (PlayerPrefs.GetInt(PrefsKey, 1) == 0 || machineRoot == null) return 0;
            int installed = 0;
            foreach (var publisher in machineRoot.GetComponentsInChildren<IMUMsgPublisher>(true))
            {
                var go = publisher.gameObject;
                if (!go.name.StartsWith("kit_") || go.GetComponent<CapKitFixedRateImu>() != null) continue;
                string topicName = Read<string>(publisher, "_topicName");
                object serializer = Read<object>(publisher, "_serializer");
                object header = serializer != null ? Read<object>(serializer, "_header") : null;
                string frame = header != null ? Read<string>(header, "_frame_id") : null;
                if (string.IsNullOrEmpty(topicName) || string.IsNullOrEmpty(frame))
                {
                    Debug.LogWarning("[CAP Kit] " + go.name + ": IMU publisher without topic/frame; left per-frame");
                    continue;
                }
                float rate = Read<float>(publisher, "_frequency");
                publisher.enabled = false;
                var sensor = go.GetComponent<IMUSensor>();
                if (sensor != null) sensor.enabled = false;
                var imu = go.AddComponent<CapKitFixedRateImu>();
                imu.topic = topicName;
                imu.frameId = frame;
                imu.rateHz = rate > 0f ? rate : 100f;
                installed++;
            }
            if (installed > 0)
                Debug.Log("[CAP Kit] " + machineRoot.name + ": " + installed + " kit IMU(s) sampled on the physics step");
            return installed;
        }

        private static T Read<T>(object target, string field)
        {
            for (var type = target.GetType(); type != null; type = type.BaseType)
            {
                var info = type.GetField(field, Private);
                if (info != null) return (T)info.GetValue(target);
            }
            return default;
        }

        private void Start()
        {
            _ros = ROSConnection.GetOrCreateInstance();
            _ros.RegisterPublisher<ImuMsg>(topic);
        }

        private void FixedUpdate()
        {
            float dt = Time.fixedDeltaTime;
            Vector3 position = transform.position;
            Quaternion rotation = transform.rotation;
            if (!_primed)
            {
                _lastPosition = position;
                _lastVelocity = Vector3.zero;
                _lastRotation = rotation;
                _primed = true;
                _nextSampleTime = Time.fixedTimeAsDouble;
                return;
            }
            // IMUSensor.Update, with the physics step for the frame time.
            Vector3 velocity = (position - _lastPosition) / dt;
            Vector3 acceleration = (velocity - _lastVelocity) / dt;
            acceleration -= transform.InverseTransformDirection(Physics.gravity.normalized) * Physics.gravity.magnitude;
            Quaternion delta = Quaternion.Inverse(_lastRotation) * rotation;
            delta.ToAngleAxis(out float angle, out Vector3 axis);
            Vector3 angularVelocity = axis * (angle * Mathf.Deg2Rad / dt);
            _lastPosition = position;
            _lastVelocity = velocity;
            _lastRotation = rotation;

            double now = Time.fixedTimeAsDouble;
            if (now + 1e-6 < _nextSampleTime) return;
            _nextSampleTime = System.Math.Max(_nextSampleTime + 1.0 / rateHz, now);
            var msg = new ImuMsg();
            msg.header.frame_id = frameId;
#if ROS2
            int sec = (int)System.Math.Truncate(now);
#else
            uint sec = (uint)System.Math.Truncate(now);
#endif
            msg.header.stamp.sec = sec;
            msg.header.stamp.nanosec = (uint)((now - sec) * 1e+9);
            msg.linear_acceleration = acceleration.To<FLU>();
            msg.orientation = rotation.To<FLU>();
            msg.angular_velocity = angularVelocity.To<FLU>();
            if (_ros == null) return;
            _ros.Publish(topic, msg);
            Published++;
        }
    }
}
