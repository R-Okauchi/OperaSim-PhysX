using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.BuiltinInterfaces;
using RosMessageTypes.Std;
using RosMessageTypes.Sensor;
using Unity.Robotics.Core;

/// <summary>
/// IMUセンサシミュレーション。ArticulationBodyの姿勢・角速度・加速度をROSに送信。
/// マシンのbody_linkに付けることでbody IMUとして機能する。
/// boom_link, arm_link等に付ければ各関節のIMUになる。
/// </summary>
public class IMUPublisher : MonoBehaviour
{
    ROSConnection ros;

    [Tooltip("IMUデータを出力するROSトピック名")]
    public string topicName = "[robot_name]/imu/body";
    private string preprocessedTopicName;

    [Tooltip("フレーム名")]
    public string frameName = "[robot_name]/imu_body_link";
    private string preprocessedFrameName;

    [Tooltip("ROSメッセージの出力間隔(秒)")]
    public float publishMessageInterval = 0.02f; // 50Hz

    private ImuMsg message;
    private float timeElapsed;
    private Vector3 previousVelocity;
    private ArticulationBody ab;

    void Start()
    {
        message = new ImuMsg();
        message.header = new HeaderMsg();
        message.header.stamp = new TimeMsg();
        // Initialize covariance arrays (9 elements each, row-major)
        message.orientation_covariance = new double[9];
        message.angular_velocity_covariance = new double[9];
        message.linear_acceleration_covariance = new double[9];

        preprocessedTopicName = Utils.PreprocessNamespace(this.gameObject, topicName);
        preprocessedFrameName = Utils.PreprocessNamespace(this.gameObject, frameName);

        ros = ROSConnection.GetOrCreateInstance();
        ros.RegisterPublisher<ImuMsg>(preprocessedTopicName);

        ab = GetComponentInParent<ArticulationBody>();
        if (ab == null)
        {
            ab = GetComponent<ArticulationBody>();
        }
        previousVelocity = Vector3.zero;
    }

    void FixedUpdate()
    {
        timeElapsed += Time.deltaTime;

        if (timeElapsed >= publishMessageInterval)
        {
            message.header.frame_id = preprocessedFrameName;
            message.header.stamp = new TimeStamp(Clock.time);

            // Orientation: Unity(x,y,z,w) -> ROS(-z,x,-y,w)
            Quaternion rot = transform.rotation;
            message.orientation.x = -rot.z;
            message.orientation.y = rot.x;
            message.orientation.z = -rot.y;
            message.orientation.w = rot.w;

            if (ab != null)
            {
                // Angular velocity: Unity(x,y,z) -> ROS(z,-x,y)
                Vector3 angVel = ab.angularVelocity;
                message.angular_velocity.x = angVel.z;
                message.angular_velocity.y = -angVel.x;
                message.angular_velocity.z = angVel.y;

                // Linear acceleration: compute from velocity change
                Vector3 currentVelocity = ab.velocity;
                Vector3 acceleration = (currentVelocity - previousVelocity) / Time.fixedDeltaTime;
                // Add gravity in world frame
                acceleration += Physics.gravity;
                // Convert to ROS: Unity(x,y,z) -> ROS(z,-x,y)
                message.linear_acceleration.x = acceleration.z;
                message.linear_acceleration.y = -acceleration.x;
                message.linear_acceleration.z = acceleration.y;

                previousVelocity = currentVelocity;
            }

            ros.Publish(preprocessedTopicName, message);
            timeElapsed = 0.0f;
        }
    }
}
