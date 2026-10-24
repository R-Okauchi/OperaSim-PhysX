using System.Collections;
using UnityEngine;

/// <summary>
/// The soil's own check where the editor is not (a Linux player on a lab host, CapKit.Editor.CapBuild): a few seconds
/// after the scene starts, 20 digs and a 0.3 m3 deposit at a point far from the machines, the terrain's volume around
/// it measured before and after - the GPU's compute shaders (the dig kernel, the heightmap copy and its sync) doing
/// what CapSoilPlayTests saw in the editor. One line "[CapSoilSelfTest] PASS|FAIL ..." in the log, whatever happens in it;
/// then the player quits (code 0 or 3) if asked. CapCycleRecorder makes it on "soil_selftest": true
/// ("soil_selftest_quit": true). Not quit, it leaves its pit and its mound at the point (about +0.04 m3 on the whole).
/// The working heightmap is compared with the terrain in the first frame, before any physics step: the terrain's slopes
/// over the repose angle erode on the GPU alone from the first step on (CAP review a846 #5: 3 s later a point 3 texels
/// from a 39 deg slope could be 2 cm off, a false FAIL). Runs after SoilParticleSettings' Start (its working heightmap).
/// </summary>
[DefaultExecutionOrder(1000)]
public class CapSoilSelfTest : MonoBehaviour
{
    public Vector3 point = new Vector3(-25f, 0f, -10f);
    public bool quitAfter;

    private IEnumerator Start()
    {
        var soil = SoilParticleSettings.instance;
        if (soil != null && !soil.enable) soil = null;                   // disabled: nothing of it to check
        string result = "not run";
        bool pass = false;
        float gpu0 = 0f, cpu0 = 0f, worst = float.PositiveInfinity;
        var pairs = new System.Collections.Generic.List<string>();
        try
        {
            if (soil != null)
            {
                (gpu0, cpu0) = soil.HeightmapTexel(point);
                // the working heightmap against the terrain at different heights (SimpleScene's pit floor and its
                // rims, the open ground: ROS (x, y) as Unity (-y, ., x)) - a seed laid out other than the terrain is
                // (transposed, flipped, a texel off) would show
                worst = 0f;
                foreach (var ros in new[] { new Vector2(-20f, -10f), new Vector2(-8.3f, -10f), new Vector2(-15f, 5f), new Vector2(10f, 10f) })
                {
                    var (g, c) = soil.HeightmapTexel(new Vector3(-ros.y, 0f, ros.x));
                    worst = Mathf.Max(worst, Mathf.Abs(g - c));
                    pairs.Add($"({ros.x:F1},{ros.y:F1}) {g:F5}/{c:F5}");
                }
            }
        }
        catch (System.Exception e)
        {
            result = $"the heightmaps not compared: {e.GetType().Name}: {e.Message}";
            soil = null;
        }
        yield return new WaitForSeconds(3.0f);
        IEnumerator body = soil == null ? null : Digs(soil, gpu0, cpu0, worst, pairs);
        while (body != null)
        {
            object step;
            try
            {
                if (!body.MoveNext()) break;
                step = body.Current;
            }
            catch (System.Exception e)
            {
                // an exception ends the check, never its line or its quit (a846 #5: a run asked to quit ran on)
                outcome = (false, $"an exception in the check: {e.GetType().Name}: {e.Message}");
                break;
            }
            yield return step;
        }
        if (soil == null && result == "not run")
        {
            result = "no soil (SoilParticleSettings missing or disabled)";
        }
        else if (soil != null)
        {
            (pass, result) = outcome;
        }
        Debug.Log($"[CapSoilSelfTest] {(pass ? "PASS" : "FAIL")}: {result}");
        if (quitAfter)
        {
            yield return null;
            Application.Quit(pass ? 0 : 3);
        }
    }

    private (bool, string) outcome = (false, "the check did not end");

    private IEnumerator Digs(SoilParticleSettings soil, float gpu0, float cpu0, float worst,
                             System.Collections.Generic.List<string> pairs)
    {
        var terrain = soil.GetComponent<Terrain>();
        float v0 = Volume(terrain, point, 20);
        for (int i = 0; i < 20; i++)
        {
            Vector2 o = Random.insideUnitCircle * 0.5f;
            soil.DigUnit(point + new Vector3(o.x, 0f, o.y), 1f);
        }
        var (gpu1, cpu1) = soil.HeightmapTexel(point);     // the digs' texels: on the GPU, then copied to the terrain
        for (int i = 0; i < 3; i++) yield return new WaitForFixedUpdate();
        var (gpu2, cpu2) = soil.HeightmapTexel(point);
        float v1 = Volume(terrain, point, 20);
        soil.Deposit(point, 0.3f, 0.35f);
        for (int i = 0; i < 3; i++) yield return new WaitForFixedUpdate();
        float v2 = Volume(terrain, point, 20);
        float dug = v0 - v1, expected = 20f * soil.DigUnitM3, put = v2 - v1;
        bool pass = Mathf.Abs(dug - expected) <= 0.02f * expected && Mathf.Abs(put - 0.3f) <= 0.006f
                    && Mathf.Abs(gpu0 - cpu0) <= 1e-4f && worst <= 1e-4f;
        outcome = (pass, $"20 digs took {dug:F4} m3 of {expected:F4}, a 0.3 m3 deposit put {put:F4} m3, the working " +
                 $"heightmap {gpu0:F5} where the terrain is {cpu0:F5} in the first frame (after the digs " +
                 $"{gpu1:F5}/{cpu1:F5}, 3 steps on {gpu2:F5}/{cpu2:F5}; elsewhere {string.Join(", ", pairs)}) " +
                 $"(particles {(soil.particles ? "on" : "off")}, {SystemInfo.graphicsDeviceType} {SystemInfo.graphicsDeviceName}, " +
                 $"compute {(SystemInfo.supportsComputeShaders ? "yes" : "no")})");
    }

    /// <summary>The terrain's volume [m3] above its base in a (2 half + 1) square of heightmap points around a point.</summary>
    private static float Volume(Terrain t, Vector3 world, int half)
    {
        var d = t.terrainData;
        int res = d.heightmapResolution;
        Vector3 rel = world - t.transform.position;
        int cx = Mathf.RoundToInt(rel.x / d.size.x * (res - 1)), cz = Mathf.RoundToInt(rel.z / d.size.z * (res - 1));
        int x0 = Mathf.Clamp(cx - half, 0, res - 1), z0 = Mathf.Clamp(cz - half, 0, res - 1);
        int w = Mathf.Min(2 * half + 1, res - x0), h = Mathf.Min(2 * half + 1, res - z0);
        float[,] heights = d.GetHeights(x0, z0, w, h);
        double sum = 0;
        foreach (var v in heights) sum += v;
        return (float)(sum * d.size.y * (d.size.x / (res - 1)) * (d.size.z / (res - 1)));
    }
}
