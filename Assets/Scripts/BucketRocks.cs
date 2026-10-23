using System.Collections;
using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// バケットにアタッチして、地面との接触時に粒子を生成する。
/// 粒子なし (SoilParticleSettings.particles = false) のときは、掘った量をバケットが持つ: 口が上を向いている間は容量まで、
/// 口が下を向くにつれて持てる量が減り (上から holdFullDeg で満杯、holdNoneDeg で 0)、余りが口から真下へ落ちる
/// (荷台なら CapBedLoad へ、地面ならそこを上げる)。口の向きは Start で殻に光線を投げて測る (CapSoilGeometry)。
/// 60°・120° は実機の zx200 の 2026-01-29 の 58 分の作業 (人の運転、130 回の運搬) から: 運ぶ間の口は上から中央値 23°
/// (p10/p90 7/135)、掘り終わり 21°、放土で開いた所は中央値 121° (p10 79°)。CAP の scripts/bag_tools/zx200_bucket_mouth.py。
/// </summary>
public class BucketRocks : MonoBehaviour
{
    [Tooltip("粒子なしのときのバケットの容量 [m3] (山積み)。0 なら機体の名前から (zx200 0.8、zx120 0.5)")]
    public float capacityM3 = 0f;

    [Tooltip("口が上からこの角度 [deg] までは容量いっぱいに持つ")]
    public float holdFullDeg = 60f;

    [Tooltip("口が上からこの角度 [deg] を越えると何も持たない (間は直線)")]
    public float holdNoneDeg = 120f;

    [Tooltip("こぼれる速さの上限 [m3/s]")]
    public float pourRateM3s = 1.0f;

    [Tooltip("地面に落ちた土を広げる幅 (正規分布の σ) [m]")]
    public float spreadM = 0.35f;

    [Tooltip("掘るのは口の側の接触だけ: 殻の重心から口の向きにこの距離 [m] より後ろ (バケットの背) の接触では掘らない")]
    public float backReachM = 0.15f;

    [Tooltip("こぼれる所: 殻の重心から口の向きにこの距離 [m]")]
    public float mouthDepthM = 0.4f;

    /// <summary>今持っている土 [m3]</summary>
    public float LoadM3 { get; private set; }
    public float CapacityM3 => capacity;
    public CapSoilGeometry.Mouth Mouth => mouth;
    /// <summary>掘った・荷台へ・地面へ・行き先なし [m3] (Start から)</summary>
    public float DugM3 { get; private set; }
    public float ToBedsM3 { get; private set; }
    public float ToGroundM3 { get; private set; }
    public float LostM3 { get; private set; }
    /// <summary>地面との接触 (粒子なしで掘ろうとした回数)、そのうち口の後ろだったもの・満杯だったもの</summary>
    public int Contacts { get; private set; }
    public int Behind { get; private set; }
    public int NoRoom { get; private set; }

    private SoilParticleSettings soil;
    private Transform link;
    private Transform machine;
    private CapSoilGeometry.Mouth mouth;
    private float capacity;
    private double lastDig = double.NegativeInfinity;
    private readonly ContactPoint[] contacts = new ContactPoint[16];
    private RaycastHit[] hits = new RaycastHit[128];             // the excavator's own boxes come first (some 50): grown
    private readonly Dictionary<string, float> session = new Dictionary<string, float>();
    private float sessionDug;
    private double lastPour = double.NegativeInfinity;

    private void Start()
    {
        soil = SoilParticleSettings.instance;
        soil.RegisterBucket(gameObject);
        link = transform.parent != null ? transform.parent : transform;
        machine = CapSoilGeometry.Machine(transform);
        if (soil.particles)
            return;                                          // capacity 0: holds nothing
        capacity = capacityM3 > 0f ? capacityM3 : CapSoilGeometry.BucketCapacityM3(machine.name);
        mouth = CapSoilGeometry.BucketMouth(link, CapSoilGeometry.BucketShell(link));
        Debug.Log($"[CapSoil] {machine.name} bucket {capacity:F2} m3, mouth {mouth.direction.ToString("F3")} in {link.name} " +
                  (mouth.fromRays ? $"({mouth.escaped}/{mouth.cast} rays out, {mouth.spanDeg:F0} deg)" : "(no ray out: the pin-tip plane)") +
                  $", now {Vector3.Angle(link.TransformDirection(mouth.direction), Vector3.up):F0} deg from up");
    }

    private void OnDestroy()
    {
        if (SoilParticleSettings.instance != null)
        {
            SoilParticleSettings.instance.UnregisterBucket(gameObject);
        }
    }

    private void OnCollisionStay(Collision other)
    {
        // With particles, upstream's way: the manager itself, whether or not this one has started - collisions reach a
        // disabled MonoBehaviour, whose Start has not run (Codex soil #3: a bucket disabled at the start made no particle
        // ever). Without them the bucket's own state (its mouth, its capacity) is Start's: none dug before it.
        var manager = soil != null ? soil : SoilParticleSettings.instance;
        if (manager == null)
            return;
        if (manager.particles)
        {
            manager.OnBucketCollision(other, transform.position);
            return;
        }
        if (soil != null && isActiveAndEnabled)
            Dig(other);                      // disabled after its Start: no pour (FixedUpdate), so no dig either
    }

    /// <summary>今の口の向きで持てる量 [m3]</summary>
    public float Holds()
    {
        float fromUp = Vector3.Angle(link.TransformDirection(mouth.direction), Vector3.up);
        return capacity * Mathf.Clamp01((holdNoneDeg - fromUp) / (holdNoneDeg - holdFullDeg));
    }

    /// <summary>こぼれる所 (世界の座標)</summary>
    public Vector3 SpillPoint() => link.TransformPoint(mouth.centroid + mouth.direction * mouthDepthM);

    /// <summary>試験用: 持っている土を volume [m3] にする (掘らずに)</summary>
    public void SetLoad(float volume) => LoadM3 = Mathf.Clamp(volume, 0f, capacity);

    private void Dig(Collision other)
    {
        if (!soil.enable || other.gameObject.name != soil.terrainObjectName)
            return;
        double now = Time.timeAsDouble;
        if (now - lastDig <= 0.01)
            return;
        Contacts++;
        float room = Holds() - LoadM3;
        if (room <= 1e-6f)
        {
            NoRoom++;
            return;
        }
        int n = other.GetContacts(contacts);
        if (n == 0)
            return;
        Vector3 c = link.TransformPoint(mouth.centroid), m = link.TransformDirection(mouth.direction);
        int first = Random.Range(0, n);
        for (int i = 0; i < n; i++)
        {
            Vector3 p = contacts[(first + i) % n].point;
            if (Vector3.Dot(p - c, m) < -backReachM)
                continue;
            float took = soil.DigUnit(p, room);
            LoadM3 += took;
            DugM3 += took;
            sessionDug += took;
            lastDig = now;
            return;
        }
        Behind++;
    }

    private void FixedUpdate()
    {
        if (soil == null || soil.particles)
            return;
        double now = Time.timeAsDouble;
        if (LoadM3 > 0f)
        {
            float excess = LoadM3 - Holds();
            if (excess > 1e-6f)
            {
                float pour = Mathf.Min(excess, pourRateM3s * Time.fixedDeltaTime);
                LoadM3 -= pour;
                Drop(pour);
                lastPour = now;
            }
        }
        if (session.Count > 0 && now - lastPour > 0.5)
        {
            var parts = new List<string>();
            foreach (var kv in session) parts.Add($"{kv.Key} {kv.Value:F3}");
            Debug.Log($"[CapSoil] {machine.name} bucket poured: {string.Join(", ", parts)} m3; holds {LoadM3:F3} m3");
            session.Clear();
        }
        if (sessionDug > 0f && now - lastDig > 1.0)
        {
            Debug.Log($"[CapSoil] {machine.name} bucket dug {sessionDug:F3} m3; holds {LoadM3:F3} of {capacity:F2} m3");
            sessionDug = 0f;
        }
    }

    /// <summary>口から真下へ: 自分の機体の外で最初に当たる物へ - 荷台なら荷台、地面なら地面、他の機体や物なら行き先なし
    /// (その上に落ちて地形には入らない: 素通りすると運搬車の車体の下の地面が上がった)。光線は口の 2 m 上から (荷台の縁の
    /// 内に沈んだ口でも床に当たる) だが、口より MouthAboveM 上の物は素通り (Codex soil #4: 口の上の荷台の床へ移した)。
    /// トリガーは荷台の盛りの他は数えない。光線の当たりは全部を見てから決める (128 で切ると荷台や地面が抜けた: #5)。
    /// 地面に置けなかった分は行き先なし。</summary>
    private void Drop(float volume)
    {
        Vector3 spill = SpillPoint();
        int n = CapSoilGeometry.CastDown(spill + Vector3.up * 2f, 62f, ref hits, QueryTriggerInteraction.Collide);
        System.Array.Sort(hits, 0, n, HitDistance.Instance);
        for (int i = 0; i < n; i++)
        {
            var col = hits[i].collider;
            if (col.transform.IsChildOf(machine))
                continue;
            if (col is TerrainCollider)
            {
                // the ground (over a mouth sunk in a dig too: the soil back on it)
                float placed = soil.Deposit(hits[i].point, volume, spreadM);
                ToGroundM3 += placed;
                Count("ground", placed);
                if (volume - placed > 1e-7f)
                {
                    LostM3 += volume - placed;                 // off the terrain's edge or its heights
                    Count("nowhere", volume - placed);
                }
                return;
            }
            if (hits[i].point.y > spill.y + MouthAboveM)
                continue;                                      // above the mouth (a bed's floor too): the soil falls past it
            var bed = CapBedLoad.Of(col);
            if (bed == null && col.isTrigger)
                continue;                                      // a volume, not a thing: soil falls through it
            if (bed == null)
                break;                                         // on another machine or thing: nowhere
            bed.Add(volume);
            ToBedsM3 += volume;
            Count(bed.MachineName, volume);
            return;
        }
        LostM3 += volume;
        Count("nowhere", volume);
    }

    // A bed's hit this far above the mouth is over it (the mouth sunk this far below a rim still meets its floor)
    private const float MouthAboveM = 0.3f;

    private void Count(string where, float volume)
    {
        session.TryGetValue(where, out float v);
        session[where] = v + volume;
    }

    private sealed class HitDistance : IComparer<RaycastHit>
    {
        public static readonly HitDistance Instance = new HitDistance();
        public int Compare(RaycastHit a, RaycastHit b) => a.distance.CompareTo(b.distance);
    }
}
