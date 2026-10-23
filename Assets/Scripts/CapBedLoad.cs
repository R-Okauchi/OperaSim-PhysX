using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// 粒子なしの土 (SoilParticleSettings.particles = false) の運搬車の荷台: バケットから落ちた量を持ち、荷台がダンプで
/// 傾くと (ダンプの関節の角が holdFullDeg までは満杯、holdNoneDeg で 0。車体の坂の姿勢は入れない) 余りを一番低い縁 (前の
/// 縁は除く) から外へ流す: 落ちる先は真下で最初に当たる物 (地面なら地形を上げ、他の機体なら失う)。縁から出る所は自分の
/// 車体の外まで (その真下に自分の車体が無い所まで) 外へ出す。
/// 平らなまま容量を越えた分は後ろか横の縁から。積んだ量は床の上の箱 (盛り) として見え、LiDAR の光線にも当たる: トリガーの当たり判定を
/// 荷台とは別の kinematic の体で動かす (荷台の ArticulationBody の質量・重心・慣性に触らない)。
/// SoilParticleSettings が Start で VesselController の付いたリンクに付ける。
/// </summary>
public class CapBedLoad : MonoBehaviour
{
    [Tooltip("容量 [m3] (平らに縁まで)。0 なら機体の名前から (CapSoilGeometry.BedOf)")]
    public float capacityM3 = 0f;

    [Tooltip("荷台の傾きがこの角度 [deg] までは容量いっぱいに持つ")]
    public float holdFullDeg = 8f;

    [Tooltip("荷台の傾きがこの角度 [deg] を越えると何も持たない (間は直線)")]
    public float holdNoneDeg = 45f;

    [Tooltip("流れ出る速さの上限 [m3/s]")]
    public float flowRateM3s = 3f;

    [Tooltip("積んだ量を盛り (箱) で見せ、光線に当てる")]
    public bool mound = true;

    /// <summary>今積んでいる土 [m3]</summary>
    public float VolumeM3 { get; private set; }
    public float CapacityM3 => bed.capacityM3;
    public CapSoilGeometry.Bed Bed => bed;
    public string MachineName => machine != null ? machine.name : name;
    /// <summary>入った・地面へ流れ出た・行き先なし [m3] (Start から)</summary>
    public float ReceivedM3 { get; private set; }
    public float ToGroundM3 { get; private set; }
    public float LostM3 { get; private set; }
    public GameObject MoundObject => moundGo;

    private static readonly Dictionary<Collider, CapBedLoad> moundColliders = new Dictionary<Collider, CapBedLoad>();

    private SoilParticleSettings soil;
    private Transform machine;
    private ArticulationBody joint;           // the bed's own joint: its dump
    private CapSoilGeometry.Bed bed;
    private GameObject moundGo;
    private Rigidbody moundBody;
    private BoxCollider moundCollider;
    private Renderer moundRenderer;
    private float moundHeight = -1f;
    private RaycastHit[] hits = new RaycastHit[128];             // grown when full (CapSoilGeometry.CastDown)
    private float sessionIn, sessionOut, sessionLost;
    private double lastIn = double.NegativeInfinity, lastOut = double.NegativeInfinity;

    /// <summary>当たり判定の持ち主の荷台 (荷台のリンクの下か、その盛り)。無ければ null。</summary>
    public static CapBedLoad Of(Collider c)
    {
        if (c == null)
            return null;
        if (moundColliders.TryGetValue(c, out var bed))
            return bed;
        return c.GetComponentInParent<CapBedLoad>();
    }

    private void Start()
    {
        soil = SoilParticleSettings.instance;
        machine = CapSoilGeometry.Machine(transform);
        joint = GetComponent<ArticulationBody>();
        bed = CapSoilGeometry.BedOf(machine.name, transform);
        if (capacityM3 > 0f)
            bed.capacityM3 = capacityM3;
        if (mound)
            MakeMound();
        Debug.Log($"[CapSoil] {machine.name} bed {bed.capacityM3:F2} m3, inside {bed.min.ToString("F2")}..{bed.max.ToString("F2")} " +
                  $"in {name}" + (bed.known ? "" : " (estimated from its colliders)"));
    }

    private void OnDestroy()
    {
        if (moundCollider != null)
            moundColliders.Remove(moundCollider);
        if (moundGo != null)
            Destroy(moundGo);
    }

    /// <summary>落ちてきた土 volume [m3] を積む</summary>
    public void Add(float volume)
    {
        if (volume <= 0f)
            return;
        VolumeM3 += volume;
        ReceivedM3 += volume;
        sessionIn += volume;
        lastIn = Time.timeAsDouble;
    }

    /// <summary>荷台の傾き [deg]: ダンプの関節の角 (車体に対して)。関節が無ければ親のリンクとの間の角。車体の坂の姿勢は
    /// 入れない: 世界に対する傾きでは 8° の坂を走るだけで流れ出た (10-23 のレビュー a846 P2-3)。</summary>
    public float TiltDeg()
    {
        if (joint != null && joint.jointType == ArticulationJointType.RevoluteJoint && joint.dofCount > 0)
            return Mathf.Abs(joint.jointPosition[0]) * Mathf.Rad2Deg;
        Transform parent = transform.parent;
        return Vector3.Angle(transform.up, parent != null ? parent.up : Vector3.up);
    }

    /// <summary>今の傾きで持てる量 [m3]</summary>
    public float Holds()
    {
        float tilt = TiltDeg();
        return bed.capacityM3 * Mathf.Clamp01((holdNoneDeg - tilt) / (holdNoneDeg - holdFullDeg));
    }

    private void FixedUpdate()
    {
        double now = Time.timeAsDouble;
        float excess = VolumeM3 - Holds();
        if (excess > 1e-6f)
        {
            float flow = Mathf.Min(excess, flowRateM3s * Time.fixedDeltaTime);
            VolumeM3 -= flow;
            OverTheRim(flow);
            lastOut = now;
        }
        UpdateMound();
        if (sessionIn > 0f && now - lastIn > 1.0)
        {
            Debug.Log($"[CapSoil] {MachineName} bed received {sessionIn:F3} m3; holds {VolumeM3:F3} of {bed.capacityM3:F2} m3");
            sessionIn = 0f;
        }
        if ((sessionOut > 0f || sessionLost > 0f) && now - lastOut > 0.5)
        {
            Debug.Log($"[CapSoil] {MachineName} bed flowed out {sessionOut:F3} m3 to the ground" +
                      (sessionLost > 0f ? $", {sessionLost:F3} m3 nowhere" : "") +
                      $"; holds {VolumeM3:F3} m3 at {TiltDeg():F0} deg");
            sessionOut = 0f;
            sessionLost = 0f;
        }
    }

    /// <summary>縁から外へ: 傾いていれば一番低い縁、平らなら縁の長さに比べた偶然の縁 - どちらも前の縁 (運転席の側) は
    /// 選ばない (前へ流すと車体の上に落ちた: 10-23 のレビュー a846 P2-2)。</summary>
    private void OverTheRim(float volume)
    {
        Vector3 lo = bed.min, hi = bed.max;
        float y = hi.y;
        var a = new[] { new Vector3(lo.x, y, lo.z), new Vector3(lo.x, y, hi.z), new Vector3(lo.x, y, lo.z), new Vector3(hi.x, y, lo.z) };
        var b = new[] { new Vector3(hi.x, y, lo.z), new Vector3(hi.x, y, hi.z), new Vector3(lo.x, y, hi.z), new Vector3(hi.x, y, hi.z) };
        var outward = new[] { Vector3.back, Vector3.forward, Vector3.left, Vector3.right };   // rear (the hinge's side), front, sides
        const int Front = 1;
        int pick = 0;
        if (Vector3.Angle(transform.up, Vector3.up) > 2f)
        {
            float lowest = float.MaxValue;
            for (int i = 0; i < 4; i++)
            {
                if (i == Front)
                    continue;
                float h = transform.TransformPoint((a[i] + b[i]) / 2f).y;
                if (h < lowest) { lowest = h; pick = i; }
            }
        }
        else
        {
            float sx = hi.x - lo.x, sz = hi.z - lo.z, r = Random.value * (sx + 2f * sz);
            pick = r < sx ? 0 : r < sx + sz ? 2 : 3;                // the rear or a side, by their lengths
        }
        float unit = soil != null && soil.DigUnitM3 > 0f ? soil.DigUnitM3 : volume;
        const float Spread = 0.15f;
        float reach = soil != null ? soil.DepositReach(Spread) : 0f;
        while (volume > 1e-7f)
        {
            float part = Mathf.Min(volume, unit);
            Vector3 local = Vector3.Lerp(a[pick], b[pick], Random.value) + outward[pick] * Random.Range(0.1f, 0.8f);
            float placed = 0f;
            if (soil != null && GroundBelow(ClearOfBody(local, outward[pick], reach), out Vector3 ground))
                placed = soil.Deposit(ground, part, Spread);      // off the terrain's edge or its heights: not all
            ToGroundM3 += placed;
            sessionOut += placed;
            LostM3 += part - placed;
            sessionLost += part - placed;
            volume -= part;
        }
    }

    /// <summary>Where the soil leaves the rim at ``local`` (the bed's frame), moved away from the bed level with the ground
    /// (``outward`` turned to the world and laid flat: tipped, it would sink instead) 0.25 m at a time, up to 3 m, until
    /// nothing of this machine but its bed lies below it, nor below the deposit's spread round it (8 points at ``reach``):
    /// the soil falls clear of its own body - the body let through, it went into the ground under the machine's own tracks
    /// (Codex soil #6); its spread let through, it reached 0.5 m under the body's edge, and by a corner beside it.</summary>
    private Vector3 ClearOfBody(Vector3 local, Vector3 outward, float reach)
    {
        Vector3 p0 = transform.TransformPoint(local);
        Vector3 away = Vector3.ProjectOnPlane(transform.TransformDirection(outward), Vector3.up);
        away = away.sqrMagnitude > 1e-6f ? away.normalized : Vector3.zero;
        for (int k = 0; k <= 12; k++)
        {
            Vector3 p = p0 + away * (0.25f * k);
            if (!OverOwnBody(p) && !SpreadOverOwnBody(p, reach))
                return p;
        }
        return p0 + away * 3f;
    }

    private bool SpreadOverOwnBody(Vector3 p, float reach)
    {
        if (reach <= 0f)
            return false;
        for (int i = 0; i < 8; i++)
        {
            float a = i * Mathf.PI / 4f;
            if (OverOwnBody(p + new Vector3(Mathf.Cos(a), 0f, Mathf.Sin(a)) * reach))
                return true;
        }
        return false;
    }

    /// <summary>Whether anything of this machine but its bed lies under ``p`` (from 3 m over it: a ray starting in a collider
    /// does not meet it), first or not: the ground heaped over the body's own box, or another's part between, the soil would
    /// go under it all the same (10-23: a mound left under the carrier's rear passed for the ground clear of it).</summary>
    private bool OverOwnBody(Vector3 p)
    {
        if (machine == null)
            return false;
        int n = CapSoilGeometry.CastDown(p + Vector3.up * 3f, 13f, ref hits, QueryTriggerInteraction.Ignore);
        for (int i = 0; i < n; i++)
        {
            Collider c = hits[i].collider;
            if (c != null && c.transform.IsChildOf(machine) && !c.transform.IsChildOf(transform))
                return true;                                       // the bed itself not the body
        }
        return false;
    }

    /// <summary>落ちる先: p の真下で最初に当たる物 (この機体の車体と荷台は抜ける。p より上の物 - 荷台の上の腕 - は落ちる
    /// 土に当たらない。地形は少し上でも: 穴の中の車)。地面なら true とその点、他の機体なら false (その上に落ちて地形には
    /// 入らない: 地形だけを探すと他の機体を素通りしてその車体の下に積もった、a846 P2-2)。</summary>
    private bool GroundBelow(Vector3 p, out Vector3 ground)
    {
        int n = CapSoilGeometry.CastDown(p + Vector3.up * 3f, 60f, ref hits, QueryTriggerInteraction.Ignore);
        float best = float.MaxValue;
        Collider first = null;
        ground = p;
        for (int i = 0; i < n; i++)
        {
            Collider c = hits[i].collider;
            if (c == null || hits[i].distance >= best || (machine != null && c.transform.IsChildOf(machine)))
                continue;
            if (!(c is TerrainCollider) && hits[i].point.y > p.y + 0.2f)
                continue;                                          // above where the soil leaves the rim
            best = hits[i].distance;
            first = c;
            ground = hits[i].point;
        }
        return first is TerrainCollider;
    }

    private void MakeMound()
    {
        moundGo = GameObject.CreatePrimitive(PrimitiveType.Cube);
        moundGo.name = $"{machine.name}_bed_load";
        moundCollider = moundGo.GetComponent<BoxCollider>();
        moundCollider.isTrigger = true;
        moundRenderer = moundGo.GetComponent<Renderer>();
        var template = soil != null && soil.rockPrefab != null ? soil.rockPrefab.GetComponent<Renderer>() : null;
        if (template != null)
            moundRenderer.sharedMaterial = template.sharedMaterial;
        moundBody = moundGo.AddComponent<Rigidbody>();
        moundBody.isKinematic = true;
        moundBody.useGravity = false;
        moundBody.interpolation = RigidbodyInterpolation.Interpolate;
        moundColliders[moundCollider] = this;
        moundRenderer.enabled = false;
        moundCollider.enabled = false;
        moundGo.transform.SetPositionAndRotation(transform.position, transform.rotation);
    }

    private void UpdateMound()
    {
        if (moundGo == null)
            return;
        bool show = VolumeM3 > 0.005f;
        if (moundRenderer.enabled != show)
        {
            moundRenderer.enabled = show;
            moundCollider.enabled = show;
        }
        if (!show)
            return;
        Vector3 size = bed.max - bed.min;
        float h = Mathf.Clamp(VolumeM3 / (size.x * size.z), 0.02f, 2f * size.y);
        if (Mathf.Abs(h - moundHeight) > 0.001f)
        {
            moundGo.transform.localScale = new Vector3(size.x, h, size.z);
            moundHeight = h;
        }
        var center = new Vector3((bed.min.x + bed.max.x) / 2f, bed.min.y + h / 2f, (bed.min.z + bed.max.z) / 2f);
        moundBody.MovePosition(transform.TransformPoint(center));
        moundBody.MoveRotation(transform.rotation);
    }
}
