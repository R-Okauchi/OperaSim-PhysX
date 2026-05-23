using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.BuiltinInterfaces;
using RosMessageTypes.Std;
using RosMessageTypes.Sensor;
using Unity.Robotics.Core;
using System.Collections.Generic;

/// <summary>
/// 簡易LiDARシミュレーション。Raycastベースで周囲の点群を生成しPointCloud2として送信。
/// キャビン上や屋根にマウントする。
///
/// 注意: UnitySensorsパッケージにはVelodyne/Livox等の高精度LiDARがあるが、
/// このスクリプトは軽量な代替として動作確認用に使う。
/// 本格的なLiDARが必要な場合はUnitySensorsのRaycastLiDARを使うこと。
/// </summary>
public class LiDARPublisher : MonoBehaviour
{
    ROSConnection ros;

    [Tooltip("LiDARデータを出力するROSトピック名")]
    public string topicName = "[robot_name]/lidar/front/points";
    private string preprocessedTopicName;

    [Tooltip("フレーム名")]
    public string frameName = "[robot_name]/lidar_link";
    private string preprocessedFrameName;

    [Tooltip("ROSメッセージの出力間隔(秒)")]
    public float publishMessageInterval = 0.1f; // 10Hz

    [Header("LiDAR Configuration")]
    [Tooltip("水平方向のレイ数")]
    public int horizontalRays = 360;

    [Tooltip("垂直方向のチャンネル数")]
    public int verticalChannels = 16;

    [Tooltip("最大検出距離 (m)")]
    public float maxRange = 50.0f;

    [Tooltip("最小検出距離 (m)")]
    public float minRange = 0.5f;

    [Tooltip("垂直FOV上限 (degrees)")]
    public float verticalFovUp = 15.0f;

    [Tooltip("垂直FOV下限 (degrees)")]
    public float verticalFovDown = -15.0f;

    private PointCloud2Msg message;
    private float timeElapsed;

    // Self-collision avoidance: the machine's own Colliders are disabled
    // during the raycast sweep so the LiDAR does not "see" its own boom,
    // arm, bucket, crawler tracks, dump bed, etc. This is the standard
    // approach in robotics simulators (Gazebo, CARLA, ISAAC Sim).
    private Collider[] selfColliders;

    /// <summary>
    /// 最新のRaycastヒットポイント(ワールド座標)。LiDARVisualizerが参照する。
    /// </summary>
    [HideInInspector]
    public Vector3[] hitPoints;
    [HideInInspector]
    public float[] hitIntensities;
    [HideInInspector]
    public int hitCount;

    void Start()
    {
        preprocessedTopicName = Utils.PreprocessNamespace(this.gameObject, topicName);
        preprocessedFrameName = Utils.PreprocessNamespace(this.gameObject, frameName);

        ros = ROSConnection.GetOrCreateInstance();
        ros.RegisterPublisher<PointCloud2Msg>(preprocessedTopicName);

        // Cache all Colliders belonging to this robot so we can disable them
        // during the raycast sweep. Walk up to the URDF root (the topmost
        // parent that still has child Colliders — typically the robot prefab
        // root) and collect every Collider underneath it.
        Transform robotRoot = FindRobotRoot(transform);
        selfColliders = robotRoot.GetComponentsInChildren<Collider>();
        Debug.Log($"[LiDARPublisher] {preprocessedTopicName}: cached {selfColliders.Length} self-colliders under '{robotRoot.name}' for raycast exclusion");

        // Initialize PointCloud2 message structure
        message = new PointCloud2Msg();
        message.header = new HeaderMsg();
        message.header.stamp = new TimeMsg();
        message.height = (uint)verticalChannels;
        message.width = (uint)horizontalRays;
        message.is_bigendian = false;
        message.is_dense = false;
        message.point_step = 16; // x(4) + y(4) + z(4) + intensity(4)
        message.row_step = (uint)(horizontalRays * 16);

        // Field descriptors: x, y, z, intensity
        message.fields = new PointFieldMsg[]
        {
            new PointFieldMsg { name = "x", offset = 0, datatype = PointFieldMsg.FLOAT32, count = 1 },
            new PointFieldMsg { name = "y", offset = 4, datatype = PointFieldMsg.FLOAT32, count = 1 },
            new PointFieldMsg { name = "z", offset = 8, datatype = PointFieldMsg.FLOAT32, count = 1 },
            new PointFieldMsg { name = "intensity", offset = 12, datatype = PointFieldMsg.FLOAT32, count = 1 },
        };
    }

    void FixedUpdate()
    {
        timeElapsed += Time.deltaTime;

        if (timeElapsed >= publishMessageInterval)
        {
            message.header.frame_id = preprocessedFrameName;
            message.header.stamp = new TimeStamp(Clock.time);

            // Disable self-colliders so the raycast doesn't hit our own
            // boom, arm, bucket, crawlers, dump bed, etc.
            SetSelfCollidersEnabled(false);

            // Generate point cloud via raycasting
            List<byte> pointData = new List<byte>();
            float verticalStep = (verticalFovUp - verticalFovDown) / Mathf.Max(verticalChannels - 1, 1);
            float horizontalStep = 360.0f / horizontalRays;

            int totalRays = horizontalRays * verticalChannels;
            if (hitPoints == null || hitPoints.Length != totalRays)
            {
                hitPoints = new Vector3[totalRays];
                hitIntensities = new float[totalRays];
            }
            hitCount = 0;

            for (int v = 0; v < verticalChannels; v++)
            {
                float vertAngle = verticalFovDown + v * verticalStep;

                for (int h = 0; h < horizontalRays; h++)
                {
                    float horizAngle = h * horizontalStep;

                    // Calculate ray direction in local frame
                    Vector3 direction = Quaternion.Euler(-vertAngle, horizAngle, 0) * Vector3.forward;
                    direction = transform.TransformDirection(direction);

                    RaycastHit hit;
                    float intensity = 0.0f;
                    float x = 0, y = 0, z = 0;

                    if (Physics.Raycast(transform.position, direction, out hit, maxRange))
                    {
                        if (hit.distance >= minRange)
                        {
                            // Hit point in sensor local frame
                            Vector3 localPoint = transform.InverseTransformPoint(hit.point);
                            // Unity local -> ROS: (z, -x, y)
                            x = localPoint.z;
                            y = -localPoint.x;
                            z = localPoint.y;
                            intensity = 1.0f - (hit.distance / maxRange); // distance-based intensity

                            // Store world-space hit point for visualization
                            hitPoints[hitCount] = hit.point;
                            hitIntensities[hitCount] = intensity;
                            hitCount++;
                        }
                    }

                    // Pack as float32: x, y, z, intensity
                    pointData.AddRange(System.BitConverter.GetBytes(x));
                    pointData.AddRange(System.BitConverter.GetBytes(y));
                    pointData.AddRange(System.BitConverter.GetBytes(z));
                    pointData.AddRange(System.BitConverter.GetBytes(intensity));
                }
            }

            // Re-enable self-colliders immediately after the sweep so
            // physics simulation (terrain contact, bucket-soil interaction,
            // inter-machine collision) continues to work normally.
            SetSelfCollidersEnabled(true);

            message.data = pointData.ToArray();
            message.width = (uint)horizontalRays;
            message.height = (uint)verticalChannels;
            message.row_step = (uint)(horizontalRays * 16);

            ros.Publish(preprocessedTopicName, message);
            timeElapsed = 0.0f;
        }
    }

    /// <summary>
    /// Walk up the hierarchy from the LiDAR sensor to find the robot's
    /// root Transform.  The URDF convention is:
    ///   robotRoot / base_link / body_link / ... / lidar_link
    /// We stop at the highest ancestor that is NOT the scene root.
    /// </summary>
    private static Transform FindRobotRoot(Transform sensor)
    {
        Transform current = sensor;
        while (current.parent != null && current.parent.parent != null)
        {
            current = current.parent;
        }
        return current;
    }

    /// <summary>
    /// Enable or disable all cached self-Colliders.
    /// Called around the raycast sweep to prevent self-detection.
    /// </summary>
    private void SetSelfCollidersEnabled(bool enabled)
    {
        if (selfColliders == null) return;
        for (int i = 0; i < selfColliders.Length; i++)
        {
            if (selfColliders[i] != null)
            {
                selfColliders[i].enabled = enabled;
            }
        }
    }
}
