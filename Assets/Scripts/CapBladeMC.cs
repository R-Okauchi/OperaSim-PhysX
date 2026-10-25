using RosMessageTypes.Std;
using Unity.Robotics.ROSTCPConnector;
using UnityEngine;

/// <summary>
/// The dozer's own machine control, on the machine (as a real D37PXi's MC holds its blade to the design): CAP's skill
/// sends the design plane and the offset - [robot_name]/mc/target, [enabled, z0, gx, gy, x_ref, y_ref, offset] in the
/// map frame (metres) - and every physics step the tilt levels the cutting edge's two ends to the design and the lift
/// puts the lower one at design + offset, through the cylinders' drives (CapBladeDrive, at their rates). Its state goes
/// to [robot_name]/mc/state at 10 Hz: [enabled, edge_error_m, out_of_reach, tilt_saturated, ticks, lift_cmd, tilt_cmd]
/// (the counts since the machine started).
/// With nothing between the edge and its controller: CAP's MC beside the agent, over ROS-TCP, lagged 0.1-0.2 s, and every
/// pitch of the body was cut into the pad as a one-vertex groove or ridge (10-25, M1a on the cluster: 65 % within +-5 cm
/// where the twin, its MC ticked in its own steps, made 94 %). The emergency stop holds the blade where it is.
/// </summary>
[DefaultExecutionOrder(-15)]
public class CapBladeMC : MonoBehaviour
{
    public Transform edgeLeft;
    public Transform edgeRight;
    public CapBladeDrive drive;
    public string liftJoint = "lift_joint";
    public string tiltJoint = "tilt_joint";
    public float liftLower = -0.41f;               // [rad] the contract's (lift positive down)
    public float liftUpper = 0.187f;
    public float tiltLimit = 0.136f;
    public float edgePerLift = 1.98f;              // [m/rad] the edge's drop per radian of lift (the contract's kinematics)
    public float halfWidth = 1.607f;               // [m] the edge's ends either side of its middle
    public string targetTopic = "[robot_name]/mc/target";
    public string stateTopic = "[robot_name]/mc/state";
    public float stateInterval = 0.1f;

    public bool Engaged { get; private set; }
    public double EdgeErrorM { get; private set; }
    public int OutOfReach { get; private set; }
    public int TiltSaturated { get; private set; }
    public int Ticks { get; private set; }

    private double z0, gx, gy, xRef, yRef, offset;
    private double liftCmd, tiltCmd;
    private ROSConnection ros;
    private string state;
    private float sinceSent;
    private EmergencyStop emergencyStop;

    /// <summary>The design (map frame: z = z0 + gx (x - xRef) + gy (y - yRef)) and the offset held from the next step on;
    /// what a target message does, for a test without ROS.</summary>
    public void Engage(double z0m, double gxm, double gym, double xRefM, double yRefM, double offsetM)
    {
        z0 = z0m; gx = gxm; gy = gym; xRef = xRefM; yRef = yRefM; offset = offsetM;
        Engaged = true;
    }

    public void Disengage() => Engaged = false;

    void Start()
    {
        if (drive == null) drive = GetComponent<CapBladeDrive>();
        emergencyStop = EmergencyStop.GetEmergencyStop(gameObject);
        try
        {
            ros = ROSConnection.GetOrCreateInstance();
            ros.Subscribe<Float64MultiArrayMsg>(Utils.PreprocessNamespace(gameObject, targetTopic), OnTarget);
            state = Utils.PreprocessNamespace(gameObject, stateTopic);
            ros.RegisterPublisher<Float64MultiArrayMsg>(state);
        }
        catch (System.Exception e)
        {
            Debug.LogWarning($"[CapBladeMC] no ROS ({e.Message}): engaged from code only");
            ros = null;
        }
    }

    private void OnTarget(Float64MultiArrayMsg msg)
    {
        var d = msg.data;
        if (d == null || d.Length < 7) return;
        if (d[0] < 0.5)
        {
            Disengage();
            return;
        }
        Engage(d[1], d[2], d[3], d[4], d[5], d[6]);
    }

    /// <summary>The design plus the offset under a point of the world (Unity: x = -ROS y, y = ROS z, z = ROS x).</summary>
    private double Target(Vector3 u) => z0 + gx * (u.z - xRef) + gy * (-u.x - yRef) + offset;

    void FixedUpdate()
    {
        bool stopped = emergencyStop != null && emergencyStop.isEmergencyStop;
        if (Engaged && !stopped && drive != null && edgeLeft != null && edgeRight != null)
        {
            Vector3 a = edgeLeft.position, b = edgeRight.position;
            double ea = a.y - Target(a), eb = b.y - Target(b);
            // the tilt (positive raises the left end, about the blade's forward axis) levels the ends to the design;
            // the lift (positive down) puts the lower one at design + offset - from where the joints are now, every step
            double tilt = drive.Position(tiltJoint) - (ea - eb) / (2.0 * halfWidth);
            if (tilt < -tiltLimit || tilt > tiltLimit)
            {
                TiltSaturated++;
                tilt = System.Math.Max(-tiltLimit, System.Math.Min(tiltLimit, tilt));
            }
            double lift = drive.Position(liftJoint) + System.Math.Min(ea, eb) / edgePerLift;
            if (lift < liftLower || lift > liftUpper)
            {
                OutOfReach++;
                lift = System.Math.Max(liftLower, System.Math.Min(liftUpper, lift));
            }
            drive.SetTarget(tiltJoint, tilt);
            drive.SetTarget(liftJoint, lift);
            liftCmd = lift;
            tiltCmd = tilt;
            EdgeErrorM = System.Math.Min(ea, eb);
            Ticks++;
        }
        sinceSent += Time.fixedDeltaTime;
        if (ros != null && sinceSent >= stateInterval)
        {
            sinceSent = 0f;
            ros.Publish(state, new Float64MultiArrayMsg
            {
                data = new[] { Engaged ? 1.0 : 0.0, EdgeErrorM, OutOfReach, TiltSaturated, Ticks, liftCmd, tiltCmd },
            });
        }
    }
}
