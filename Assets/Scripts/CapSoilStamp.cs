using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Soil laid onto the working heightmap in exact amounts (CapBlade.compute's StampAdd): a volume spread by weights over
/// vertices, the quanta's remainder to the heaviest, so what lands is what was asked to within one quantum. Used for a
/// blade's spills and its rest wedge, and for a scenario's piles (a dump truck's load as a cone at its repose).
/// The changed rect is committed to the terrain at once.
/// </summary>
public static class CapSoilStamp
{
    public const double QuantaPerM3 = 1e8;     // one quantum = 1e-8 m3 (the blade's ledger and the kernels count these)

    private static ComputeShader cs;
    private static int kStamp = -1;
    private static ComputeBuffer acc, texels, quanta;
    private const int MaxStamp = 16384;

    private static bool Ready(SoilParticleSettings soil)
    {
        if (soil == null || soil.WorkingHeightmap == null) return false;
        if (cs == null)
        {
            cs = Object.Instantiate(Resources.Load<ComputeShader>("CapBlade"));
            kStamp = cs.FindKernel("StampAdd");
            acc = new ComputeBuffer(4, sizeof(uint));
            texels = new ComputeBuffer(MaxStamp, sizeof(int) * 2);
            quanta = new ComputeBuffer(MaxStamp, sizeof(uint));
        }
        return true;
    }

    /// <summary>Quanta per texture-height unit per vertex (the kernels' qPerTex).</summary>
    public static float QuantaPerTexture(SoilParticleSettings soil) =>
        (float)(soil.TexelArea * soil.TextureUnitM * QuantaPerM3);

    /// <summary>Add volumeM3 at the vertices by their weights; returns the volume that landed [m3].</summary>
    public static double Stamp(SoilParticleSettings soil, IList<Vector2Int> at, IList<float> weights, double volumeM3)
    {
        if (!Ready(soil) || volumeM3 <= 0.0 || at.Count == 0) return 0.0;
        var size = soil.HeightmapSize;
        double total = 0.0;
        for (int i = 0; i < weights.Count; i++) total += Mathf.Max(0f, weights[i]);
        if (total <= 0.0) return 0.0;
        long q = (long)System.Math.Round(volumeM3 * QuantaPerM3);
        var px = new List<Vector2Int>();
        var pq = new List<uint>();
        long given = 0;
        int heaviest = -1;
        float most = -1f;
        for (int i = 0; i < at.Count && px.Count < MaxStamp; i++)
        {
            var p = at[i];
            float w = Mathf.Max(0f, weights[i]);
            if (w <= 0f || p.x < 0 || p.y < 0 || p.x >= size.x || p.y >= size.y) continue;
            uint share = (uint)System.Math.Floor(q * (w / total));
            px.Add(p);
            pq.Add(share);
            given += share;
            if (w > most) { most = w; heaviest = px.Count - 1; }
        }
        if (heaviest < 0) return 0.0;
        pq[heaviest] += (uint)System.Math.Max(0, q - given);
        var tx = new int[px.Count * 2];
        for (int i = 0; i < px.Count; i++) { tx[2 * i] = px[i].x; tx[2 * i + 1] = px[i].y; }
        texels.SetData(tx);
        quanta.SetData(pq.ToArray());
        acc.SetData(new uint[4]);
        cs.SetTexture(kStamp, "heightmap", soil.WorkingHeightmap);
        cs.SetBuffer(kStamp, "acc", acc);
        cs.SetBuffer(kStamp, "stampTexels", texels);
        cs.SetBuffer(kStamp, "stampQuanta", quanta);
        cs.SetInt("stampCount", px.Count);
        cs.SetFloat("qPerTex", QuantaPerTexture(soil));
        cs.Dispatch(kStamp, (px.Count + 63) / 64, 1, 1);
        var got = new uint[4];
        acc.GetData(got);
        int x0 = int.MaxValue, y0 = int.MaxValue, x1 = int.MinValue, y1 = int.MinValue;
        foreach (var p in px) { x0 = Mathf.Min(x0, p.x); y0 = Mathf.Min(y0, p.y); x1 = Mathf.Max(x1, p.x); y1 = Mathf.Max(y1, p.y); }
        soil.Commit(new RectInt(x0, y0, x1 - x0 + 1, y1 - y0 + 1));
        return got[3] / QuantaPerM3;
    }

    /// <summary>A Gaussian heap of volumeM3 centred at a world point (sigma [m], cut at two sigma).</summary>
    public static double Heap(SoilParticleSettings soil, Vector3 world, double volumeM3, float sigmaM)
    {
        if (!Ready(soil)) return 0.0;
        var c = soil.ToTexel(world);
        float cell = Mathf.Sqrt(soil.TexelArea);
        int r = Mathf.CeilToInt(2f * sigmaM / cell);
        var at = new List<Vector2Int>();
        var w = new List<float>();
        for (int dy = -r; dy <= r; dy++)
        for (int dx = -r; dx <= r; dx++)
        {
            int x = Mathf.RoundToInt(c.x) + dx, y = Mathf.RoundToInt(c.y) + dy;
            float d2 = ((x - c.x) * (x - c.x) + (y - c.y) * (y - c.y)) * cell * cell;
            if (d2 > 4f * sigmaM * sigmaM) continue;
            at.Add(new Vector2Int(x, y));
            w.Add(Mathf.Exp(-d2 / (2f * sigmaM * sigmaM)));
        }
        return Stamp(soil, at, w, volumeM3);
    }

    /// <summary>A dumped pile: a cone of volumeM3 at its repose, its peak over a world point.</summary>
    public static double Cone(SoilParticleSettings soil, Vector3 world, double volumeM3, float reposeDeg)
    {
        if (!Ready(soil)) return 0.0;
        float t = Mathf.Tan(reposeDeg * Mathf.Deg2Rad);
        float radius = Mathf.Pow((float)(3.0 * volumeM3 / (Mathf.PI * t)), 1f / 3f);
        var c = soil.ToTexel(world);
        float cell = Mathf.Sqrt(soil.TexelArea);
        int r = Mathf.CeilToInt(radius / cell) + 1;
        var at = new List<Vector2Int>();
        var w = new List<float>();
        for (int dy = -r; dy <= r; dy++)
        for (int dx = -r; dx <= r; dx++)
        {
            int x = Mathf.RoundToInt(c.x) + dx, y = Mathf.RoundToInt(c.y) + dy;
            float d = Mathf.Sqrt((x - c.x) * (x - c.x) + (y - c.y) * (y - c.y)) * cell;
            if (d >= radius) continue;
            at.Add(new Vector2Int(x, y));
            w.Add(radius - d);
        }
        return Stamp(soil, at, w, volumeM3);
    }
}
