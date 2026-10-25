using System.Collections.Generic;
using RosMessageTypes.Std;
using Unity.Robotics.ROSTCPConnector;
using UnityEngine;

/// <summary>
/// CAP's dozer blade in the particle-free soil (CapBlade.compute on SoilParticleSettings' working heightmap; the same
/// model as CAP's twin, twin/blade_soil.py). Each physics step, before the soil's slides:
/// - the cutting edge (the two markers on the blade) is laid over the working heightmap; the vertices within
///   +-bandVertices of it along the travel are under it;
/// - everything above the edge is cut down to it into the load before the blade; every vertex below it is filled toward
///   it, all by the one fraction the load and this step's cut can pay;
/// - a load past capacityM3 spills off the ends (all off the trailing end of an angled blade), and an angled blade casts
///   some of its load to its trailing side as it moves;
/// - a blade that stands still for a second, backs up or is carried clear of the ground lays its load before it as a
///   wedge, so the terrain (and the survey) always shows the soil.
/// The ledger counts what the kernels wrote (quanta of 1e-8 m3): terrain and load together never change.
/// Publishes [robot_name]/blade/soil (load m3, cut m3/s, fill m3/s, spill m3/s) at 10 Hz.
/// </summary>
[DefaultExecutionOrder(-10)]
public class CapDozerBlade : MonoBehaviour
{
    public Transform edgeLeft;
    public Transform edgeRight;
    public Transform hub;                          // base_link: the travel's direction and speed
    public CapBladeDrive bladeDrive;               // the angle joint's position (side cast, spill side)
    public float capacityM3 = 2.0f;
    public float widthM = 3.21f;
    public float reposeDeg = 34f;
    public float bandVertices = 0.6f;
    public float clearM = 0.15f;
    public float restAfterS = 1.0f;
    public float angleMaxRad = 0.419f;
    public string soilTopic = "[robot_name]/blade/soil";
    public float publishInterval = 0.1f;
    [Tooltip("The blade's state in the log every this many seconds (0: never)")]
    public float logEveryS = 1f;
    [Tooltip("A step that would cut or fill more than this [m3] is not applied (and is logged): no blade does that in 20 ms")]
    public float stepLimitM3 = 1.5f;

    public double LoadM3 { get; private set; }
    public double CutM3 { get; private set; }
    public double FilledM3 { get; private set; }
    public double SpilledM3 { get; private set; }
    public double RestM3 { get; private set; }

    private SoilParticleSettings soil;
    private ComputeShader cs;
    private int kMeasure = -1, kApply = -1;
    private ComputeBuffer acc;
    private Vector3 lastHub;
    private float stillFor;
    private bool started;
    private ROSConnection ros;
    private string topic;
    private float sinceSent;
    private double cutWindow, fillWindow, spillWindow;
    private float sinceLog, lastAnomalyAt = -1e9f;
    private int anomalies;
    private WheelCollider[] wheels;
    private Rigidbody body;

    /// <summary>A load put onto the blade without the soil (tests: what a blade carries in).</summary>
    public void Preload(double m3) => LoadM3 += m3;

    void Start()
    {
        soil = SoilParticleSettings.instance;
        cs = Instantiate(Resources.Load<ComputeShader>("CapBlade"));
        kMeasure = cs.FindKernel("BladeMeasure");
        kApply = cs.FindKernel("BladeApply");
        acc = new ComputeBuffer(4, sizeof(uint));
        if (hub == null) hub = transform;
        lastHub = hub.position;
        wheels = GetComponentsInChildren<WheelCollider>();
        body = GetComponentInParent<Rigidbody>();
        try
        {
            ros = ROSConnection.GetOrCreateInstance();
            topic = Utils.PreprocessNamespace(gameObject, soilTopic);
            ros.RegisterPublisher<Float64MultiArrayMsg>(topic);
        }
        catch (System.Exception e)
        {
            Debug.LogWarning($"[CapDozerBlade] no ROS publisher ({e.Message}): the soil's numbers are not sent");
            ros = null;
        }
        started = true;
    }

    void OnDestroy()
    {
        acc?.Release();
    }

    void FixedUpdate()
    {
        if (!started || soil == null || soil.WorkingHeightmap == null || edgeLeft == null || edgeRight == null) return;
        float dt = Time.fixedDeltaTime;
        Vector3 forward = hub.forward;
        forward.y = 0f;
        forward.Normalize();
        float speed = Vector3.Dot(hub.position - lastHub, forward) / dt;
        lastHub = hub.position;
        stillFor = Mathf.Abs(speed) < 0.05f ? stillFor + dt : 0f;

        Vector3 a = edgeLeft.position, b = edgeRight.position;
        float angle = bladeDrive != null ? (float)bladeDrive.Position("angle_joint") : 0f;
        if (float.IsNaN(angle)) angle = 0f;
        var terrain = soil.GetComponent<Terrain>();
        float groundMax = float.NegativeInfinity;
        for (int i = 0; i <= 8; i++)
        {
            Vector3 p = Vector3.Lerp(a, b, i / 8f);
            groundMax = Mathf.Max(groundMax, terrain.SampleHeight(p) + terrain.transform.position.y);
        }
        bool clear = Mathf.Min(a.y, b.y) > groundMax + clearM;
        Status(dt, speed, a, b, groundMax, speed < -0.1f ? "back" : clear ? "clear" : stillFor > restAfterS ? "still" : "cut");

        if (speed < -0.1f || clear || stillFor > restAfterS)
        {
            Rest(a, b, forward);
            Publish(dt);
            return;
        }

        Vector3 ta = soil.ToTexel(a), tb = soil.ToTexel(b);
        // An edge off the terrain's heights (a glitch of the body's joints, a NaN) would carve the terrain to its bottom:
        // not applied, said (10-25: 37 vertices to -50 m under a blade on the cluster's player)
        const float topTexture = 32766f / 65535f;
        if (!Finite(a) || !Finite(b) || ta.z < 0f || tb.z < 0f || ta.z > topTexture || tb.z > topTexture)
        {
            Anomaly($"the edge off the terrain: a {a} b {b} texel {ta} {tb}");
            Publish(dt);
            return;
        }
        var size = soil.HeightmapSize;
        int pad = Mathf.CeilToInt(bandVertices) + 2;
        int x0 = Mathf.Max(0, Mathf.FloorToInt(Mathf.Min(ta.x, tb.x)) - pad);
        int y0 = Mathf.Max(0, Mathf.FloorToInt(Mathf.Min(ta.y, tb.y)) - pad);
        int x1 = Mathf.Min(size.x - 1, Mathf.CeilToInt(Mathf.Max(ta.x, tb.x)) + pad);
        int y1 = Mathf.Min(size.y - 1, Mathf.CeilToInt(Mathf.Max(ta.y, tb.y)) + pad);
        if (x1 < x0 || y1 < y0) return;
        var rect = new RectInt(x0, y0, x1 - x0 + 1, y1 - y0 + 1);
        Vector2 travel = new Vector2(forward.x, forward.z).normalized;   // vertex space runs along world x and z

        float qPerTex = CapSoilStamp.QuantaPerTexture(soil);
        cs.SetTexture(kMeasure, "heightmap", soil.WorkingHeightmap);
        cs.SetTexture(kApply, "heightmap", soil.WorkingHeightmap);
        cs.SetBuffer(kMeasure, "acc", acc);
        cs.SetBuffer(kApply, "acc", acc);
        cs.SetInts("rectMin", rect.x, rect.y);
        cs.SetInts("rectSize", rect.width, rect.height);
        cs.SetVector("edgeA", new Vector4(ta.x, ta.y, ta.z, 0f));
        cs.SetVector("edgeB", new Vector4(tb.x, tb.y, tb.z, 0f));
        cs.SetVector("travel", new Vector4(travel.x, travel.y, bandVertices, bandVertices));
        cs.SetFloat("qPerTex", qPerTex);
        int gx = (rect.width + 7) / 8, gy = (rect.height + 7) / 8;

        var got = new uint[4];
        acc.SetData(new uint[4]);
        cs.Dispatch(kMeasure, gx, gy, 1);
        acc.GetData(got);
        double cut = got[0] / CapSoilStamp.QuantaPerM3, demand = got[1] / CapSoilStamp.QuantaPerM3;
        if (cut > stepLimitM3 || demand > stepLimitM3)
        {
            Anomaly($"a step of cut {cut:F3} demand {demand:F3} m3 over {got[2]} vertices: edge a {a} b {b} texel {ta} " +
                    $"{tb} rect {rect} ground {groundMax:F3} speed {speed:F2}");
            Publish(dt);
            return;
        }
        // In [0, 1]: a load a rounding below zero made a negative fraction, the fill lowered the ground and its count
        // wrapped round (uint) - the load then hugely negative, and the next steps dug to the terrain's bottom (10-25:
        // "filled" 95.5 m3 of a cut of 11.3, the edge 9 m down, the dozer at 68 deg in the hole)
        float fill = demand > 0.0 ? (float)System.Math.Max(0.0, System.Math.Min(1.0, (LoadM3 + cut) / demand)) : 0f;

        acc.SetData(new uint[4]);
        cs.SetFloat("fillFrac", fill);
        cs.Dispatch(kApply, gx, gy, 1);
        acc.GetData(got);
        double cutDone = got[0] / CapSoilStamp.QuantaPerM3, filled = got[1] / CapSoilStamp.QuantaPerM3;
        LoadM3 += cutDone - filled;
        CutM3 += cutDone;
        FilledM3 += filled;
        cutWindow += cutDone;
        fillWindow += filled;
        soil.Commit(rect);

        // an angled blade casts part of its load off its trailing end as it moves; a load past capacity spills
        bool trailingLeft = angle > 0f;            // angled left end back (positive about the up axis, as ROS)
        float s = Mathf.Abs(Mathf.Sin(angle));
        if (s > 0f && LoadM3 > 0.0)
        {
            double cast = System.Math.Min(LoadM3, LoadM3 * s * Mathf.Abs(speed) * dt / widthM);
            Spill(a, b, cast, trailingLeft ? 1.0 : 0.0);
        }
        if (LoadM3 > capacityM3)
        {
            double over = LoadM3 - capacityM3;
            double leftShare = Mathf.Abs(angle) > 0.1f ? (trailingLeft ? 1.0 : 0.0)
                : 0.5 * (1.0 + Mathf.Sin(angle) / Mathf.Sin(angleMaxRad));
            Spill(a, b, over, leftShare);
        }
        Publish(dt);
    }

    private static bool Finite(Vector3 v) => !(float.IsNaN(v.x) || float.IsNaN(v.y) || float.IsNaN(v.z) ||
                                               float.IsInfinity(v.x) || float.IsInfinity(v.y) || float.IsInfinity(v.z));

    private void Anomaly(string what)
    {
        anomalies++;
        if (Time.time - lastAnomalyAt < 1f) return;        // once a second at most
        lastAnomalyAt = Time.time;
        Debug.LogWarning($"[CapDozerBlade] t {Time.time:F2} not applied ({anomalies} so far): {what}");
    }

    private void Status(float dt, float speed, Vector3 a, Vector3 b, float groundMax, string mode)
    {
        if (logEveryS <= 0f) return;
        sinceLog += dt;
        if (sinceLog < logEveryS) return;
        sinceLog = 0f;
        // the wheels (all the dozer's contact with the ground): how many touch, their slip, their torque
        int grounded = 0;
        float slip = 0f, torque = 0f, load = 0f;
        foreach (var w in wheels ?? new WheelCollider[0])
        {
            torque += w.motorTorque - w.brakeTorque;
            if (!w.GetGroundHit(out var hit)) continue;
            grounded++;
            slip += hit.forwardSlip;
            load += hit.force;
        }
        Vector3 e = hub.rotation.eulerAngles;
        float pitch = Mathf.DeltaAngle(0f, e.x), roll = Mathf.DeltaAngle(0f, e.z);
        string drag = body != null ? $" body v {body.velocity.magnitude:F2}" : "";
        Debug.Log($"[CapDozerBlade] t {Time.time:F1} {mode} speed {speed:F2} still {stillFor:F1} s edge {a.y:F3}/{b.y:F3} " +
                  $"ground {groundMax:F3} m; load {LoadM3:F3} cut {CutM3:F3} filled {FilledM3:F3} spilled {SpilledM3:F3} " +
                  $"laid {RestM3:F3} m3; wheels {grounded}/{wheels?.Length ?? 0} slip {(grounded > 0 ? slip / grounded : 0f):F2} " +
                  $"torque {torque:F0} N m load {load / 1000f:F0} kN; pitch {pitch:F1} roll {roll:F1} deg{drag}");
    }

    /// <summary>Spill a volume off the blade's ends, leftShare of it beyond the left end.</summary>
    private void Spill(Vector3 a, Vector3 b, double volume, double leftShare)
    {
        if (volume <= 0.0) return;
        Vector3 along = (b - a);
        along.y = 0f;
        along.Normalize();
        foreach (var (end, outward, share) in new[] { (a, -along, leftShare), (b, along, 1.0 - leftShare) })
        {
            double v = volume * share;
            if (v <= 0.0) continue;
            double landed = CapSoilStamp.Heap(soil, end + outward * 0.30f, v, 0.25f);
            LoadM3 -= landed;
            SpilledM3 += landed;
            spillWindow += landed;
        }
    }

    /// <summary>Lay the load before the blade as a wedge: its height from the load and the blade's width, its length to
    /// the repose.</summary>
    private void Rest(Vector3 a, Vector3 b, Vector3 forward)
    {
        if (LoadM3 <= 1e-9)
        {
            LoadM3 = 0.0;
            return;
        }
        float t = Mathf.Tan(reposeDeg * Mathf.Deg2Rad);
        float hp = Mathf.Min(0.9f, Mathf.Sqrt(2f * (float)(LoadM3 / widthM) * t));
        float cell = Mathf.Sqrt(soil.TexelArea);
        float length = Mathf.Max(hp / t, 2f * cell);
        float band = bandVertices * cell;
        Vector3 ta = soil.ToTexel(a), tb = soil.ToTexel(b);
        Vector2 ab = new Vector2(tb.x - ta.x, tb.y - ta.y);
        float l2 = ab.sqrMagnitude;
        if (l2 <= 0f) return;
        Vector2 travel = new Vector2(forward.x, forward.z).normalized;
        int reach = Mathf.CeilToInt((band + length) / cell) + 2;
        int x0 = Mathf.FloorToInt(Mathf.Min(ta.x, tb.x)) - reach, x1 = Mathf.CeilToInt(Mathf.Max(ta.x, tb.x)) + reach;
        int y0 = Mathf.FloorToInt(Mathf.Min(ta.y, tb.y)) - reach, y1 = Mathf.CeilToInt(Mathf.Max(ta.y, tb.y)) + reach;
        var at = new List<Vector2Int>();
        var w = new List<float>();
        for (int y = y0; y <= y1; y++)
        for (int x = x0; x <= x1; x++)
        {
            Vector2 q = new Vector2(x - ta.x, y - ta.y);
            float tt = Vector2.Dot(q, ab) / l2;
            if (tt < 0f || tt > 1f) continue;
            Vector2 c = new Vector2(ta.x, ta.y) + tt * ab;
            float sv = Vector2.Dot(new Vector2(x, y) - c, travel) * cell;      // metres ahead of the edge
            if (sv <= band || sv > band + length) continue;
            at.Add(new Vector2Int(x, y));
            w.Add(1f - (sv - band) / length);
        }
        double landed = CapSoilStamp.Stamp(soil, at, w, LoadM3);
        LoadM3 -= landed;
        RestM3 += landed;
        if (System.Math.Abs(LoadM3) < 1e-6) LoadM3 = 0.0;
    }

    private void Publish(float dt)
    {
        sinceSent += dt;
        if (ros == null || sinceSent < publishInterval) return;
        var msg = new Float64MultiArrayMsg
        {
            data = new[] { LoadM3, cutWindow / sinceSent, fillWindow / sinceSent, spillWindow / sinceSent },
        };
        ros.Publish(topic, msg);
        sinceSent = 0f;
        cutWindow = fillWindow = spillWindow = 0.0;
    }
}
