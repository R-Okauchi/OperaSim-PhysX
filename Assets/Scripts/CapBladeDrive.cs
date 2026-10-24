using System;
using System.Collections.Generic;
using RosMessageTypes.Std;
using Unity.Robotics.ROSTCPConnector;
using UnityEngine;

/// <summary>
/// A dozer's blade joints (lift, angle, tilt), each from a Float64 target [rad] on its own topic
/// ([robot_name]/&lt;joint&gt;/cmd): the joint's drive target moves toward the command no faster than the joint's rate
/// (a hydraulic cylinder's speed, not the jump JointPosController makes), and stays where the joint is while the
/// emergency stop is on. The targets are kept inside the joint's limits.
/// </summary>
public class CapBladeDrive : MonoBehaviour
{
    [Serializable]
    public class BladeJoint
    {
        public string name;                    // the URDF joint's name (lift_joint, angle_joint, tilt_joint)
        public ArticulationBody body;
        public string topic = "[robot_name]/joint/cmd";
        public float rateRadPerS = 0.3f;
        public float initialRad;
        [NonSerialized] public double commandRad;
        [NonSerialized] public float targetDeg;
        [NonSerialized] public bool held;
    }

    public List<BladeJoint> joints = new List<BladeJoint>();

    private EmergencyStop emergencyStop;

    void Start()
    {
        emergencyStop = EmergencyStop.GetEmergencyStop(gameObject);
        var ros = ROSConnection.GetOrCreateInstance();
        foreach (var j in joints)
        {
            if (j.body == null)
            {
                Debug.LogError($"[CapBladeDrive] {name}: {j.name} has no joint");
                continue;
            }
            j.commandRad = Clamp(j, j.initialRad);
            j.targetDeg = (float)(j.commandRad * Mathf.Rad2Deg);
            var drive = j.body.xDrive;
            drive.target = j.targetDeg;
            j.body.xDrive = drive;
            var joint = j;                       // the topic's callback keeps its own joint
            ros.Subscribe<Float64Msg>(Utils.PreprocessNamespace(gameObject, j.topic), m => joint.commandRad = Clamp(joint, m.data));
        }
    }

    /// <summary>A command without ROS (tests, CAP's scenario operations): as a message on the joint's topic.</summary>
    public void SetTarget(string joint, double rad)
    {
        foreach (var j in joints)
        {
            if (j.name == joint) j.commandRad = Clamp(j, rad);
        }
    }

    /// <summary>The joint's position now [rad] (what joint_states says).</summary>
    public double Position(string joint)
    {
        foreach (var j in joints)
        {
            if (j.name == joint && j.body != null) return j.body.jointPosition[0];
        }
        return double.NaN;
    }

    void FixedUpdate()
    {
        bool stopped = emergencyStop != null && emergencyStop.isEmergencyStop;
        foreach (var j in joints)
        {
            if (j.body == null) continue;
            var drive = j.body.xDrive;
            if (stopped)
            {
                if (!j.held)
                {
                    j.targetDeg = j.body.jointPosition[0] * Mathf.Rad2Deg;   // where it is, not where it was going
                    j.held = true;
                }
            }
            else
            {
                if (j.held)
                {
                    j.commandRad = j.body.jointPosition[0];                    // after a stop: no jump to the old command
                    j.held = false;
                }
                float goal = (float)(j.commandRad * Mathf.Rad2Deg);
                float step = j.rateRadPerS * Mathf.Rad2Deg * Time.fixedDeltaTime;
                j.targetDeg = Mathf.MoveTowards(j.targetDeg, goal, step);
            }
            drive.target = j.targetDeg;
            j.body.xDrive = drive;
        }
    }

    static double Clamp(BladeJoint j, double rad)
    {
        if (j.body == null || double.IsNaN(rad)) return 0.0;
        var d = j.body.xDrive;
        double lo = d.lowerLimit * Mathf.Deg2Rad, hi = d.upperLimit * Mathf.Deg2Rad;
        if (hi <= lo) return rad;                // an unlimited joint
        return Math.Max(lo, Math.Min(hi, rad));
    }
}
