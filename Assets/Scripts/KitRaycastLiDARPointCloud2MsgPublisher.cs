using UnityEngine;
using Unity.Robotics.Core;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Sensor;
using UnitySensors.Sensor.LiDAR;

namespace CapKit
{
    [DisallowMultipleComponent, RequireComponent(typeof(RaycastLiDARSensor))]
    public sealed class KitRaycastLiDARPointCloud2MsgPublisher : MonoBehaviour
    {
        [SerializeField] private float _frequency = 10f;
        [SerializeField] private string _topicName;
        [SerializeField] private KitPointCloud2MsgSerializer _serializer = new KitPointCloud2MsgSerializer();
        // Outgoing ROS-TCP queue per topic. The connector default (10) overflows when several machines
        // publish 128 KB clouds in the same frame ("Queue full! Messages are getting dropped!").
        [SerializeField] private int _queueSize = 40;

        private RaycastLiDARSensor _sensor;
        private ROSConnection _ros;
        private double _nextPublishTime;

        private void Start()
        {
            _sensor = GetComponent<RaycastLiDARSensor>();
            if (!(_frequency > 0f) || float.IsInfinity(_frequency) || string.IsNullOrWhiteSpace(_topicName)
                || !_sensor.enabled || !_sensor.pointCloud.points.IsCreated)
            {
                Debug.LogError("[CAP Kit] LiDAR publisher requires a configured sensor, topic and positive frequency.", this);
                enabled = false;
                return;
            }
            if (_serializer == null) _serializer = new KitPointCloud2MsgSerializer();
            _serializer.Init(_sensor);
            var marker = GetComponentInParent<CapKitFrameMarker>();
            _serializer.SetFrame(marker != null ? marker.transform : transform);
            // Kit sensors publish; they do not need the debug point-cloud visualizer (PSIZE warning per draw on Metal).
            var scope = marker != null ? marker.transform : transform;
            foreach (var behaviour in scope.GetComponentsInChildren<Behaviour>(true))
            {
                string ns = behaviour.GetType().Namespace;
                if (ns != null && ns.StartsWith("UnitySensors.Visualization", System.StringComparison.Ordinal)) behaviour.enabled = false;
            }
            foreach (var renderer in scope.GetComponentsInChildren<Renderer>(true))
            {
                var material = renderer.sharedMaterial;
                if (material != null && material.shader != null && material.shader.name.StartsWith("UnitySensors/", System.StringComparison.Ordinal))
                    renderer.enabled = false;
            }
            _ros = ROSConnection.GetOrCreateInstance();
            _ros.RegisterPublisher<PointCloud2Msg>(_topicName, Mathf.Max(1, _queueSize));
            _sensor.onSensorUpdated += PublishScan;
        }

        private void PublishScan()
        {
            if (!isActiveAndEnabled) return;
            double now = Clock.time;
            if (now + 0.000001 < _nextPublishTime) return;
            // Publish only completed scans. A faster publisher setting does not duplicate old scans.
            _ros.Publish(_topicName, _serializer.Serialize());
            _nextPublishTime = now + 1.0 / _frequency;
        }

        private void OnDestroy()
        {
            if (_sensor != null) _sensor.onSensorUpdated -= PublishScan;
        }
    }
}
