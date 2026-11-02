using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// 粒子なしの土 (SoilParticleSettings.particles = false) が使う形: バケットの口の向きと荷台の内側の箱。
/// 動いているときの BucketRocks・CapBedLoad と、エディタの調べ (CapKit.Editor.CapSoilProbe) が同じ関数で測る。
/// </summary>
public static class CapSoilGeometry
{
    /// <summary>バケットの口: 殻の重心 (リンクの座標) と、そこから口へ向かう向き (リンクの座標の単位ベクトル)。</summary>
    public struct Mouth
    {
        public Vector3 centroid;
        public Vector3 direction;
        public int escaped;       // 殻に当たらず抜けた光線の数
        public int cast;          // 投げた光線の数
        public float spanDeg;     // ピンの面の中で抜けた向きの広がり [deg]
        public bool fromRays;     // false: 抜ける光線が無く、ピンの軸と爪先の面に垂直な向きで代えた
        public bool noShell;      // 殻の当たり判定が無く、リンクの上で代えた
    }

    /// <summary>荷台の内側 (荷台のリンクの座標、Unity の軸): 床 min.y、縁 max.y、ヒンジ側は min.z。容量 [m3] は平らに
    /// 縁まで。</summary>
    public struct Bed
    {
        public Vector3 min;
        public Vector3 max;
        public float capacityM3;
        public bool known;        // false: 当たり判定の外形からの見積もり
    }

    private static readonly string[] Kinds = { "mst110cr", "ic120", "c30r", "zx200", "zx120", "zx135u" };

    /// <summary>機体の型: シーンの名前の頭の型 ("zx200(Clone)"・"mst110cr_2" も。Codex soil #7: 複製の名前で容量が 0.8 から
    /// 0.5 に、荷台は見積もりに落ちた)。知らなければ名前のまま。</summary>
    public static string Kind(string machine)
    {
        string name = (machine ?? "").ToLowerInvariant();
        foreach (var kind in Kinds)
        {
            if (name.StartsWith(kind, System.StringComparison.Ordinal)) return kind;
        }
        return name;
    }

    /// <summary>機体ごとのバケットの容量 [m3] (山積み): CAP の機体の設定 (zx200 0.8、zx120 0.5) と同じ。</summary>
    public static float BucketCapacityM3(string machine)
    {
        switch (Kind(machine))
        {
            case "zx200": return 0.8f;
            case "zx120": return 0.5f;
            case "zx135u": return 0.5f;
            default: return 0.5f;
        }
    }

    /// <summary>
    /// 機体ごとの荷台の内側。CapSoilProbe で当たり判定を調べた値 (2026-10-22、SimpleScene):
    ///   mst110cr: 床の上面 y 0.19 (z 0.2–1.93)、横の壁の内側 x ±1.32・上端 y 0.81、後ろ (z −0.52–0.2) は縁の低い傾き
    ///             (y 0.34 まで)、前は z 1.9 から 2.7 へ上がる → z −0.45–2.3 を平らな箱に。縁まで 2.64 × 0.62 × 2.75 ≈ 4.5 m3
    ///             のうち後ろの傾きと前の斜面の分を引いて 4.0 m3
    ///   ic120:    床 y 0.54 (z 0.6–3.05)、壁の内側 x −0.90–0.87・上端 y 1.08、後ろは z 0.6 から −0.45 へ y 1.19 まで
    ///             上がる板 → z 0.1–3.05、1.77 × 0.54 × 2.95 ≈ 2.8 m3
    ///   c30r:     床の見た目は y 0.17 だが、当たり判定は荷台の上を y 0.93 で塞ぐ箱 (下向きの光線 9 本とも 0.935 で当たる)
    ///             → その上に載せる: 壁 x ±0.68、z −0.15–1.3、y 0.93–1.36 → 1.36 × 0.43 × 1.45 ≈ 0.85 m3
    /// 知らない荷台は当たり判定の外形から (幅と長さを 90 %、床を下から 15 %、縁を 60 % の高さ、容量はその箱の 80 %): 外形は
    /// 荷台の座標での当たり判定の点 (箱の 8 隅・メッシュの頂点: 世界の外形の対角 2 点では、回った荷台の幅がほぼ 0 になった)。
    /// </summary>
    public static Bed BedOf(string machine, Transform vessel)
    {
        switch (Kind(machine))
        {
            case "mst110cr":
                return new Bed { min = new Vector3(-1.32f, 0.19f, -0.45f), max = new Vector3(1.32f, 0.81f, 2.3f), capacityM3 = 4.0f, known = true };
            case "ic120":
                return new Bed { min = new Vector3(-0.90f, 0.54f, 0.10f), max = new Vector3(0.87f, 1.08f, 3.05f), capacityM3 = 2.8f, known = true };
            case "c30r":
                return new Bed { min = new Vector3(-0.68f, 0.93f, -0.15f), max = new Vector3(0.68f, 1.36f, 1.30f), capacityM3 = 0.85f, known = true };
        }
        bool any = false;
        Vector3 lo = Vector3.zero, hi = Vector3.zero;
        foreach (var c in vessel.GetComponentsInChildren<Collider>())
        {
            if (!c.enabled || c.isTrigger)
                continue;                                    // a disabled one's bounds are an empty box at the origin
            foreach (var q in OutlinePoints(c, vessel))
            {
                if (!any) { lo = q; hi = q; any = true; }
                lo = Vector3.Min(lo, q);
                hi = Vector3.Max(hi, q);
            }
        }
        if (!any)
        {
            return new Bed { min = new Vector3(-0.5f, 0f, -0.5f), max = new Vector3(0.5f, 0.5f, 0.5f), capacityM3 = 0.4f, known = false };
        }
        Vector3 c0 = (lo + hi) / 2f, size = hi - lo;
        var bed = new Bed
        {
            min = new Vector3(c0.x - 0.45f * size.x, lo.y + 0.15f * size.y, c0.z - 0.45f * size.z),
            max = new Vector3(c0.x + 0.45f * size.x, lo.y + 0.60f * size.y, c0.z + 0.45f * size.z),
            known = false,
        };
        Vector3 inner = bed.max - bed.min;
        bed.capacityM3 = 0.8f * inner.x * inner.y * inner.z;
        return bed;
    }

    /// <summary>バケットの殻: リンクの下の当たり判定のうち、BucketRocks の付いた内側と爪先のリンクを除いたもの。</summary>
    public static List<Collider> BucketShell(Transform link)
    {
        var tip = link.Find("bucket_end_link");
        var shell = new List<Collider>();
        foreach (var c in link.GetComponentsInChildren<Collider>(true))
        {
            if (c.GetComponentInParent<BucketRocks>() != null) continue;
            if (tip != null && c.transform.IsChildOf(tip)) continue;
            shell.Add(c);
        }
        return shell;
    }

    /// <summary>
    /// バケットの口の向き: 殻の重心から、ピンの面 (2° ごと) とそこから ±15° に光線を投げ、殻に当たらず抜けた向きの平均。
    /// 抜ける光線が無ければ、ピンの軸と爪先 (bucket_end_link) の面に垂直で重心と反対の向き。
    /// zx200 (2026-10-22): 抜けた 149/540・広がり 100°、リンクの座標で (0.002, −0.758, 0.652)。面の向きとは 7° 違う。
    /// </summary>
    public static Mouth BucketMouth(Transform link, IList<Collider> shell)
    {
        var points = new List<Vector3>();
        foreach (var c in shell)
        {
            foreach (var p in ColliderPoints(c, link)) points.Add(p);
        }
        var mouth = new Mouth();
        if (points.Count == 0)
        {
            mouth.direction = Vector3.up;
            mouth.noShell = true;
            return mouth;
        }
        Vector3 sum = Vector3.zero;
        foreach (var p in points) sum += p;
        mouth.centroid = sum / points.Count;
        var ab = link.GetComponent<ArticulationBody>();
        Vector3 axis = ab != null ? (ab.anchorRotation * Vector3.right).normalized : Vector3.right;
        Vector3 u = Vector3.Cross(axis, Vector3.up);
        u = u.sqrMagnitude > 1e-6f ? u.normalized : Vector3.Cross(axis, Vector3.forward).normalized;
        Vector3 w = Vector3.Cross(axis, u).normalized;
        Vector3 origin = link.TransformPoint(mouth.centroid);
        Vector3 escapedSum = Vector3.zero;
        var inPlane = new List<float>();
        foreach (float tilt in new[] { 0f, 15f, -15f })
        {
            float ct = Mathf.Cos(tilt * Mathf.Deg2Rad), st = Mathf.Sin(tilt * Mathf.Deg2Rad);
            for (int k = 0; k < 180; k++)
            {
                float a = k * 2f * Mathf.Deg2Rad;
                Vector3 d = (Mathf.Cos(a) * u + Mathf.Sin(a) * w) * ct + axis * st;
                var ray = new Ray(origin, link.TransformDirection(d).normalized);
                mouth.cast++;
                bool hit = false;
                foreach (var c in shell)
                {
                    if (c.Raycast(ray, out _, 4f)) { hit = true; break; }
                    // a ray from inside a convex piece does not meet it: the origin inside counts as a hit
                    if (c is MeshCollider mc && mc.convex && (c.ClosestPoint(origin) - origin).sqrMagnitude < 1e-8f) { hit = true; break; }
                }
                if (hit) continue;
                mouth.escaped++;
                escapedSum += d;
                if (tilt == 0f) inPlane.Add(k * 2f);
            }
        }
        if (inPlane.Count > 0)
        {
            // the largest circular gap between escaping angles lies outside the span
            inPlane.Sort();
            float gap = 0f;
            for (int i = 0; i < inPlane.Count; i++)
            {
                float next = i + 1 < inPlane.Count ? inPlane[i + 1] : inPlane[0] + 360f;
                gap = Mathf.Max(gap, next - inPlane[i]);
            }
            mouth.spanDeg = 360f - gap + 2f;
        }
        if (escapedSum.sqrMagnitude > 1e-6f)
        {
            mouth.direction = escapedSum.normalized;
            mouth.fromRays = true;
            return mouth;
        }
        var tipT = link.Find("bucket_end_link");
        Vector3 tip = tipT != null ? link.InverseTransformPoint(tipT.position) : Vector3.forward;
        Vector3 edge = tip - Vector3.Project(tip, axis);
        Vector3 normal = Vector3.Cross(axis, edge).normalized;
        Vector3 off = mouth.centroid - Vector3.Project(mouth.centroid, axis);
        mouth.direction = Vector3.Dot(off, normal) > 0 ? -normal : normal;
        return mouth;
    }

    /// <summary>from から真下への光線の当たり全部 (buffer は足りなければ広げる: 満杯の結果は近い順とは限らず、荷台や地面が
    /// 抜けた: Codex soil #5)。数を返す。</summary>
    public static int CastDown(Vector3 from, float distance, ref RaycastHit[] buffer, QueryTriggerInteraction triggers)
    {
        while (true)
        {
            // every layer: a machine's body box is on Ignore Raycast, out of the sensors' rays, and the soil meets it as
            // it did (CapBodyBoxes)
            int n = Physics.RaycastNonAlloc(from, Vector3.down, buffer, distance, Physics.AllLayers, triggers);
            if (n < buffer.Length)
                return n;
            if (buffer.Length >= 8192)
            {
                if (!castCapped)
                {
                    castCapped = true;
                    Debug.LogWarning($"[CapSoil] {n} colliders under {from}: some may be missed");
                }
                return n;
            }
            buffer = new RaycastHit[buffer.Length * 2];
        }
    }

    private static bool castCapped;

    /// <summary>当たり判定の点 (frame の座標): メッシュは頂点、箱は 8 隅、他は外形の 2 隅 (バケットの口の重心はこれで)。</summary>
    public static IEnumerable<Vector3> ColliderPoints(Collider c, Transform frame)
    {
        if (c is MeshCollider mc && mc.sharedMesh != null && mc.sharedMesh.isReadable)
        {
            foreach (var v in mc.sharedMesh.vertices) yield return frame.InverseTransformPoint(mc.transform.TransformPoint(v));
        }
        else if (c is BoxCollider bc)
        {
            for (int i = 0; i < 8; i++)
            {
                var o = new Vector3((i & 1) == 0 ? -0.5f : 0.5f, (i & 2) == 0 ? -0.5f : 0.5f, (i & 4) == 0 ? -0.5f : 0.5f);
                yield return frame.InverseTransformPoint(bc.transform.TransformPoint(bc.center + Vector3.Scale(o, bc.size)));
            }
        }
        else
        {
            var b = c.bounds;
            yield return frame.InverseTransformPoint(b.min);
            yield return frame.InverseTransformPoint(b.max);
        }
    }

    /// <summary>当たり判定の外形の点 (frame の座標): メッシュは頂点、箱は 8 隅、他は世界の外形の 8 隅 - 回った frame でも
    /// 外形を包む (対角 2 点は包まない)。</summary>
    public static IEnumerable<Vector3> OutlinePoints(Collider c, Transform frame)
    {
        if ((c is MeshCollider mc && mc.sharedMesh != null && mc.sharedMesh.isReadable) || c is BoxCollider)
        {
            foreach (var p in ColliderPoints(c, frame)) yield return p;
            yield break;
        }
        var b = c.bounds;
        for (int i = 0; i < 8; i++)
        {
            yield return frame.InverseTransformPoint(new Vector3((i & 1) == 0 ? b.min.x : b.max.x,
                                                                 (i & 2) == 0 ? b.min.y : b.max.y,
                                                                 (i & 4) == 0 ? b.min.z : b.max.z));
        }
    }

    /// <summary>"robot" の印の付いた一番近い祖先 (シーンの機体のルート)。無ければ一番上。</summary>
    public static Transform Machine(Transform t)
    {
        for (var x = t; x != null; x = x.parent)
        {
            if (x.CompareTag("robot")) return x;
        }
        return t.root;
    }

    /// <summary>平均 0・σ = sigma の 2 次元の正規分布 (2σ で切る)。</summary>
    public static Vector2 Gaussian2(float sigma)
    {
        if (sigma <= 0f) return Vector2.zero;
        for (int i = 0; i < 8; i++)
        {
            float u1 = Mathf.Max(Random.value, 1e-6f), u2 = Random.value;
            float r = Mathf.Sqrt(-2f * Mathf.Log(u1));
            var g = new Vector2(r * Mathf.Cos(2f * Mathf.PI * u2), r * Mathf.Sin(2f * Mathf.PI * u2)) * sigma;
            if (g.sqrMagnitude <= 4f * sigma * sigma) return g;
        }
        return Vector2.zero;
    }
}
