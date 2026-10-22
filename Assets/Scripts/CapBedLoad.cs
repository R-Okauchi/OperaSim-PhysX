using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// 粒子なしの土 (SoilParticleSettings.particles = false) の運搬車の荷台: バケットから落ちた量を持ち、荷台が傾くと
/// (傾きが holdFullDeg までは満杯、holdNoneDeg で 0) 余りを一番低い縁から外の地面へ流す (ダンプ)。平らなまま容量を
/// 越えた分は縁のどこかから。積んだ量は床の上の箱 (盛り) として見え、LiDAR の光線にも当たる: トリガーの当たり判定を
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
    private CapSoilGeometry.Bed bed;
    private GameObject moundGo;
    private Rigidbody moundBody;
    private BoxCollider moundCollider;
    private Renderer moundRenderer;
    private float moundHeight = -1f;
    private readonly RaycastHit[] hits = new RaycastHit[128];
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

    /// <summary>今の傾きで持てる量 [m3]</summary>
    public float Holds()
    {
        float tilt = Vector3.Angle(transform.up, Vector3.up);
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
                      $"; holds {VolumeM3:F3} m3 at {Vector3.Angle(transform.up, Vector3.up):F0} deg");
            sessionOut = 0f;
            sessionLost = 0f;
        }
    }

    /// <summary>縁から外の地面へ: 傾いていれば一番低い縁、平らなら縁の長さに比べた偶然の縁。</summary>
    private void OverTheRim(float volume)
    {
        Vector3 lo = bed.min, hi = bed.max;
        float y = hi.y;
        var a = new[] { new Vector3(lo.x, y, lo.z), new Vector3(lo.x, y, hi.z), new Vector3(lo.x, y, lo.z), new Vector3(hi.x, y, lo.z) };
        var b = new[] { new Vector3(hi.x, y, lo.z), new Vector3(hi.x, y, hi.z), new Vector3(lo.x, y, hi.z), new Vector3(hi.x, y, hi.z) };
        var outward = new[] { Vector3.back, Vector3.forward, Vector3.left, Vector3.right };   // rear (the hinge's side), front, sides
        int pick = 0;
        if (Vector3.Angle(transform.up, Vector3.up) > 2f)
        {
            float lowest = float.MaxValue;
            for (int i = 0; i < 4; i++)
            {
                float h = transform.TransformPoint((a[i] + b[i]) / 2f).y;
                if (h < lowest) { lowest = h; pick = i; }
            }
        }
        else
        {
            float sx = hi.x - lo.x, sz = hi.z - lo.z, r = Random.value * 2f * (sx + sz);
            pick = r < sx ? 0 : r < 2f * sx ? 1 : r < 2f * sx + sz ? 2 : 3;
        }
        float unit = soil != null && soil.DigUnitM3 > 0f ? soil.DigUnitM3 : volume;
        while (volume > 1e-7f)
        {
            float part = Mathf.Min(volume, unit);
            Vector3 local = Vector3.Lerp(a[pick], b[pick], Random.value) + outward[pick] * Random.Range(0.1f, 0.8f);
            if (soil != null && GroundBelow(transform.TransformPoint(local), out Vector3 ground))
            {
                soil.Deposit(ground, part, 0.15f);
                ToGroundM3 += part;
                sessionOut += part;
            }
            else
            {
                LostM3 += part;
                sessionLost += part;
            }
            volume -= part;
        }
    }

    private bool GroundBelow(Vector3 p, out Vector3 ground)
    {
        int n = Physics.RaycastNonAlloc(p + Vector3.up * 3f, Vector3.down, hits, 60f, Physics.DefaultRaycastLayers,
                                        QueryTriggerInteraction.Ignore);
        float best = float.MaxValue;
        ground = p;
        for (int i = 0; i < n; i++)
        {
            if (hits[i].collider is TerrainCollider && hits[i].distance < best)
            {
                best = hits[i].distance;
                ground = hits[i].point;
            }
        }
        return best < float.MaxValue;
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
