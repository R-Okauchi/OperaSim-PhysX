using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.BuiltinInterfaces;
using RosMessageTypes.Std;
using RosMessageTypes.Sensor;
using Unity.Robotics.Core;

/// <summary>
/// GNSSセンサシミュレーション。Unity座標をGNSS位置（lat/lon/alt）に変換してROSに送信。
/// マシンのbody_linkまたはアンテナ位置に付ける。
/// </summary>
public class GNSSPublisher : MonoBehaviour
{
    ROSConnection ros;

    [Tooltip("GNSSデータを出力するROSトピック名")]
    public string topicName = "[robot_name]/gnss/fix";
    private string preprocessedTopicName;

    [Tooltip("フレーム名")]
    public string frameName = "[robot_name]/gnss_link";
    private string preprocessedFrameName;

    [Tooltip("ROSメッセージの出力間隔(秒)")]
    public float publishMessageInterval = 0.1f; // 10Hz

    [Header("GNSS Origin (site reference point)")]
    [Tooltip("サイト原点の緯度 (degrees)")]
    public double originLatitude = 36.0;

    [Tooltip("サイト原点の経度 (degrees)")]
    public double originLongitude = 140.0;

    [Tooltip("サイト原点の標高 (meters)")]
    public double originAltitude = 0.0;

    // 1度あたりのメートル (近似値: 緯度35-36度付近)
    private const double METERS_PER_DEG_LAT = 111320.0;
    private const double METERS_PER_DEG_LON = 91290.0; // cos(36deg) * 111320

    private NavSatFixMsg message;
    private float timeElapsed;

    void Start()
    {
        message = new NavSatFixMsg();
        message.header = new HeaderMsg();
        message.header.stamp = new TimeMsg();
        message.status = new NavSatStatusMsg();
        message.status.status = NavSatStatusMsg.STATUS_FIX; // 0 = fix
        message.status.service = NavSatStatusMsg.SERVICE_GPS;
        message.position_covariance = new double[9];
        // Set diagonal covariance (RTK-level accuracy: ~2cm)
        message.position_covariance[0] = 0.0004; // lat variance (0.02m)^2
        message.position_covariance[4] = 0.0004; // lon variance
        message.position_covariance[8] = 0.01;   // alt variance (0.1m)^2
        message.position_covariance_type = NavSatFixMsg.COVARIANCE_TYPE_DIAGONAL_KNOWN;

        preprocessedTopicName = Utils.PreprocessNamespace(this.gameObject, topicName);
        preprocessedFrameName = Utils.PreprocessNamespace(this.gameObject, frameName);

        ros = ROSConnection.GetOrCreateInstance();
        ros.RegisterPublisher<NavSatFixMsg>(preprocessedTopicName);
    }

    void FixedUpdate()
    {
        timeElapsed += Time.deltaTime;

        if (timeElapsed >= publishMessageInterval)
        {
            message.header.frame_id = preprocessedFrameName;
            message.header.stamp = new TimeStamp(Clock.time);

            // Unity position -> lat/lon
            // Unity(x,y,z) -> ROS(z,-x,y) => ROS x = Unity z, ROS y = -Unity x
            // lat offset from z (north), lon offset from -x (east)
            double northing = transform.position.z;  // meters north from origin
            double easting = -transform.position.x;  // meters east from origin

            message.latitude = originLatitude + northing / METERS_PER_DEG_LAT;
            message.longitude = originLongitude + easting / METERS_PER_DEG_LON;
            message.altitude = originAltitude + transform.position.y;

            ros.Publish(preprocessedTopicName, message);
            timeElapsed = 0.0f;
        }
    }
}
