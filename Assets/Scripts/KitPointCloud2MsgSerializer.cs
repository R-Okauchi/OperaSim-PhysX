using System;
using UnityEngine;
using Unity.Robotics.Core;
using RosMessageTypes.Sensor;
using UnitySensors.ROS.Serializer;
using UnitySensors.Sensor.LiDAR;

namespace CapKit
{
    [Serializable]
    public sealed class KitPointCloud2MsgSerializer : RosMsgSerializer<RaycastLiDARSensor, PointCloud2Msg>
    {
        [SerializeField] private HeaderSerializer _header = new HeaderSerializer();
        private float[] _values;
        private Transform _frame;

        /// <summary>
        /// Frame the published points are expressed in. The contract mount describes this frame (the
        /// CapKitFrameMarker object); sensor prefabs may place their optical origin on a child with an
        /// offset (Mid-360: +0.047 m), so points are re-expressed here instead of in the sensor child.
        /// </summary>
        public void SetFrame(Transform frame) => _frame = frame;

        public override void Init(RaycastLiDARSensor sensor)
        {
            base.Init(sensor);
            if (_header == null) _header = new HeaderSerializer();
            _header.Init(sensor);
            _msg.height = 1;
            _msg.fields = new PointFieldMsg[4];
            string[] names = { "x", "y", "z", "intensity" };
            for (int i = 0; i < names.Length; i++)
                _msg.fields[i] = new PointFieldMsg
                {
                    name = names[i], offset = (uint)(4 * i), datatype = 7, count = 1
                };
            _msg.point_step = 16;
            _msg.is_bigendian = false;
            _msg.is_dense = true;
        }

        public override PointCloud2Msg Serialize()
        {
            var points = sensor.pointCloud.points;
            int count = points.IsCreated ? points.Length : 0;
            int bytes = checked(count * 16);
            if (_msg.data.Length != bytes) _msg.data = new byte[bytes];
            if (_values == null || _values.Length != count * 4) _values = new float[count * 4];
            _msg.width = (uint)count;
            _msg.row_step = (uint)bytes;
            _msg.header = _header.Serialize();
            // One simulation timestamp for this simultaneous cloud. No per-point time offsets.
            _msg.header.stamp = new TimeStamp(Clock.time);

            // sensor-child local -> kit frame local (identity when the sensor is the frame itself).
            Matrix4x4 toFrame = _frame != null && _frame != sensor.transform
                ? _frame.worldToLocalMatrix * sensor.transform.localToWorldMatrix
                : Matrix4x4.identity;
            bool reexpress = toFrame != Matrix4x4.identity;
            for (int i = 0; i < count; i++)
            {
                var point = points[i];
                var p = point.position;
                if (reexpress) p = toFrame.MultiplyPoint3x4(p);
                int offset = i * 4;
                // Mirror UnitySensors' IPointsToPointCloud2MsgJob: sensor-local Unity -> ROS FLU.
                bool valid = Finite(p.x) && Finite(p.y) && Finite(p.z) && Finite(point.intensity);
                _values[offset] = valid ? p.z : 0f;
                _values[offset + 1] = valid ? -p.x : 0f;
                _values[offset + 2] = valid ? p.y : 0f;
                _values[offset + 3] = valid ? point.intensity : 0f;
            }
            // Reuse managed buffers; no allocation per point and no native resources to dispose.
            Buffer.BlockCopy(_values, 0, _msg.data, 0, bytes);
            if (!BitConverter.IsLittleEndian)
                for (int i = 0; i < bytes; i += 4) Array.Reverse(_msg.data, i, 4);
            return _msg;
        }

        private static bool Finite(float value) => !float.IsNaN(value) && !float.IsInfinity(value);
    }
}
