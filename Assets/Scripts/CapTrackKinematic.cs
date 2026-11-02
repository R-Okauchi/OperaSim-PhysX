using RosMessageTypes.Geometry;
using RosMessageTypes.Nav;
using RosMessageTypes.Std;
using Unity.Robotics.Core;
using Unity.Robotics.ROSTCPConnector;
using UnityEngine;

/// <summary>
/// A tracked machine driven as CAP's twin drives it (twin/d37pxi_dozer.py), not through WheelColliders: cmd_vel's
/// speed reached after a first-order lag (lagS), its yaw rate at once, a unicycle on the terrain - the hub's height the
/// mean of the ground under the tracks' four ends, its pitch from the ground under their fronts and rears and its roll
/// from under their middles. A dozer's tracks climb what its blade leaves (a fill's 0.15-0.2 m edge, a pile's foot, a
/// windrow); six WheelColliders did not: on the fill's edge the front wheels' rays met its slanted triangles, which
/// pushed them back while the others slipped (live 10-25, M1a's first pile on the cluster: 0.67 -> 0 m/s with the
/// blade's load under 1 m3, the wheels slipping 0.4 at 170 N m).
/// The root Rigidbody is moved kinematically and the articulation fixed to it (the body and the blade) follows; the
/// WheelColliders and the DiffDriveController are switched off. Subscribes the DiffDriveController's cmd_vel topic and
/// publishes its odometry topic (from where the machine started, as it did); the emergency stop holds the machine.
/// </summary>
[DefaultExecutionOrder(-20)]
public class CapTrackKinematic : MonoBehaviour
{
    public Transform hub;                              // base_link
    public float lagS = 0.5f;                          // the tracks' speed's first-order lag (the twin's lag_s)
    public float maxLinear = 2.3f;                     // [m/s]
    public float maxAngular = 0.6f;                    // [rad/s]
    public float halfLength = 1.2f;                    // the tracks' contact either side of the hub (the twin's)
    public float halfGauge = 0.865f;                   // the tracks' centres
    public float hubAboveGround = 0.05f;               // base_link over the tracks' bottom
    public float publishInterval = 0.02f;

    public double LinearCmd { get; private set; }
    public double AngularCmd { get; private set; }
    public double Speed { get; private set; }          // the tracks' speed now [m/s]

    private Rigidbody body;
    private Terrain terrain;
    private ROSConnection ros;
    private EmergencyStop emergencyStop;
    private string odomTopic, frameId, childFrame;
    private float heading;                             // about Unity's up, degrees (Unity's yaw: clockwise from above)
    private Vector3 position;                          // the hub's ground point (x, z) and height
    private Vector3 startRos;
    private float startHeading;
    private float sinceSent;
    private Quaternion hubFromRoot = Quaternion.identity;
    private Vector3 hubInRoot;
    private bool started;

    /// <summary>A machine of the simulator's own (its WheelColliders and DiffDriveController) driven by this instead, as
    /// it stands now: its hub the root, as high over the ground under it as it is (a machine at rest on its wheels), its
    /// tracks' contact and its turn's rate as given (0: this component's). zx120's WheelColliders drive with a gain
    /// that cannot climb a bench's 16 deg ramp (750 N m per m/s on six wheels of 0.25 m: some 8 kN at its 0.46 m/s
    /// cruise against the 40 kN its 15 t needs there - r28, 11-02: it slid down the ramp's foot at up to 1.5 m/s) nor
    /// turn on the flat faster than about 0.07 rad/s. Awake switches the drive and the wheels off, before the drive's
    /// Start (a scene's machine as the scene has loaded: CapCycleRecorder's placements) or after it.</summary>
    public static CapTrackKinematic Fit(GameObject root, float halfLength, float halfGauge, float maxAngular)
    {
        Terrain t = SoilParticleSettings.instance != null ? SoilParticleSettings.instance.GetComponent<Terrain>()
                                                          : Terrain.activeTerrain;
        float ground = t != null ? t.SampleHeight(root.transform.position) + t.transform.position.y : float.NaN;
        var k = root.AddComponent<CapTrackKinematic>();
        if (halfLength > 0f) k.halfLength = halfLength;
        if (halfGauge > 0f) k.halfGauge = halfGauge;
        if (maxAngular > 0f) k.maxAngular = maxAngular;
        if (!float.IsNaN(ground)) k.hubAboveGround = root.transform.position.y - ground;
        return k;
    }

    /// <summary>The command a test gives in place of cmd_vel (ROS: linear forward [m/s], angular counter-clockwise from above
    /// [rad/s]).</summary>
    public void SetCommand(double linear, double angular)
    {
        LinearCmd = System.Math.Max(-maxLinear, System.Math.Min(maxLinear, linear));
        AngularCmd = System.Math.Max(-maxAngular, System.Math.Min(maxAngular, angular));
    }

    void Awake()
    {
        if (hub == null) hub = transform;
        body = GetComponent<Rigidbody>();
        var drive = GetComponent<DiffDriveController>();
        string twist = drive != null ? drive.TwistTopicName : "[robot_name]/tracks/cmd_vel";
        odomTopic = drive != null ? drive.OdomTopicName : "[robot_name]/odom_pose";
        childFrame = drive != null ? drive.childFrameName : "[robot_name]/base_link";
        if (drive != null) drive.enabled = false;      // before its Start: it never subscribes nor drives the wheels
        foreach (var w in GetComponentsInChildren<WheelCollider>(true)) w.enabled = false;
        if (body != null)
        {
            body.isKinematic = true;
            body.interpolation = RigidbodyInterpolation.None;
        }
        twistTopic = twist;
    }

    private string twistTopic;

    /// <summary>The cmd_vel topic it drives on, as Start resolves it (a clone's: its own name's).</summary>
    public string TwistTopic => Utils.PreprocessNamespace(gameObject, twistTopic ?? "");

    void Start()
    {
        terrain = SoilParticleSettings.instance != null ? SoilParticleSettings.instance.GetComponent<Terrain>()
                                                        : Terrain.activeTerrain;
        hubInRoot = Quaternion.Inverse(transform.rotation) * (hub.position - transform.position);
        hubFromRoot = Quaternion.Inverse(transform.rotation) * hub.rotation;
        heading = hub.eulerAngles.y;
        position = hub.position;
        startRos = Ros(hub.position);
        startHeading = heading;
        emergencyStop = EmergencyStop.GetEmergencyStop(gameObject);
        try
        {
            ros = ROSConnection.GetOrCreateInstance();
            string robot = Utils.PreprocessNamespace(gameObject, "[robot_name]");
            ros.Subscribe<TwistMsg>(Utils.PreprocessNamespace(gameObject, twistTopic),
                                    m => SetCommand(m.linear.x, m.angular.z));
            odomTopic = Utils.PreprocessNamespace(gameObject, odomTopic);
            ros.RegisterPublisher<OdometryMsg>(odomTopic);
            frameId = robot + "_tf/odom";
            childFrame = Utils.PreprocessNamespace(gameObject, childFrame);
        }
        catch (System.Exception e)
        {
            Debug.LogWarning($"[CapTrackKinematic] no ROS ({e.Message}): commands from SetCommand only");
            ros = null;
        }
        Place(Time.fixedDeltaTime, 0.0);
        started = true;
    }

    private static Vector3 Ros(Vector3 u) => new Vector3(u.z, -u.x, u.y);

    private float Ground(Vector3 p) => terrain != null ? terrain.SampleHeight(p) + terrain.transform.position.y : p.y;

    void FixedUpdate()
    {
        if (!started) return;
        float dt = Time.fixedDeltaTime;
        bool stop = emergencyStop != null && emergencyStop.isEmergencyStop;
        double v = stop ? 0.0 : LinearCmd, w = stop ? 0.0 : AngularCmd;
        Speed = stop ? 0.0 : Speed + (v - Speed) * (lagS > 0f ? 1.0 - System.Math.Exp(-dt / lagS) : 1.0);
        heading -= (float)(w * dt * Mathf.Rad2Deg);    // counter-clockwise from above is negative about Unity's up
        Vector3 f = Quaternion.Euler(0f, heading, 0f) * Vector3.forward;
        position += f * (float)(Speed * dt);
        Place(dt, w);
    }

    /// <summary>The hub on the ground under the tracks, the root moved so that it is there.</summary>
    private void Place(float dt, double w)
    {
        Quaternion yawOnly = Quaternion.Euler(0f, heading, 0f);
        Vector3 f = yawOnly * Vector3.forward, r = yawOnly * Vector3.right;
        Vector3 c = new Vector3(position.x, 0f, position.z);
        float front = Ground(c + f * halfLength), rear = Ground(c - f * halfLength);
        float left = Ground(c - r * halfGauge), right = Ground(c + r * halfGauge);
        float mean = 0.25f * (Ground(c + f * halfLength + r * halfGauge) + Ground(c + f * halfLength - r * halfGauge) +
                              Ground(c - f * halfLength + r * halfGauge) + Ground(c - f * halfLength - r * halfGauge));
        Vector3 f3 = (f * (2f * halfLength) + Vector3.up * (front - rear)).normalized;
        Vector3 r3 = (r * (2f * halfGauge) + Vector3.up * (right - left)).normalized;
        Vector3 up3 = Vector3.Cross(f3, r3).normalized;
        Quaternion hubRot = Quaternion.LookRotation(f3, up3);
        Vector3 hubPos = new Vector3(position.x, mean, position.z) + up3 * hubAboveGround;
        position.y = hubPos.y;
        Quaternion rootRot = hubRot * Quaternion.Inverse(hubFromRoot);
        Vector3 rootPos = hubPos - rootRot * hubInRoot;
        if (body != null)
        {
            body.MovePosition(rootPos);
            body.MoveRotation(rootRot);
        }
        else
        {
            transform.SetPositionAndRotation(rootPos, rootRot);
        }
        Publish(dt, hubPos, w);
    }

    private void Publish(float dt, Vector3 hubPos, double w)
    {
        sinceSent += dt;
        if (ros == null || sinceSent < publishInterval) return;
        sinceSent = 0f;
        // odometry from where the machine started, in its start's frame (as the DiffDriveController's)
        Vector3 d = Ros(hubPos) - startRos;
        float yaw0 = -startHeading * Mathf.Deg2Rad, yaw = -heading * Mathf.Deg2Rad;
        float c0 = Mathf.Cos(yaw0), s0 = Mathf.Sin(yaw0);
        var q = Quaternion.Euler(0f, 0f, (yaw - yaw0) * Mathf.Rad2Deg);
        var msg = new OdometryMsg
        {
            header = new HeaderMsg { frame_id = frameId, stamp = new RosMessageTypes.BuiltinInterfaces.TimeMsg() },
            child_frame_id = childFrame,
        };
        msg.header.stamp = new TimeStamp(Clock.time);
        msg.pose.pose.position.x = c0 * d.x + s0 * d.y;
        msg.pose.pose.position.y = -s0 * d.x + c0 * d.y;
        msg.pose.pose.orientation = new QuaternionMsg(q.x, q.y, q.z, q.w);
        msg.twist.twist.linear.x = Speed;
        msg.twist.twist.angular.z = w;
        ros.Publish(odomTopic, msg);
    }
}
