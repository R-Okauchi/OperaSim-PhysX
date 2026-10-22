using System.Collections;
using UnityEngine;

/// <summary>
/// The soil's own check where the editor is not (a Linux player on a lab host, CapKit.Editor.CapBuild): a few seconds
/// after the scene starts, 20 digs and a 0.3 m3 deposit at a point far from the machines, the terrain's volume around
/// it measured before and after - the GPU's compute shaders (the dig kernel, the heightmap copy and its sync) doing
/// what CapSoilPlayTests saw in the editor. One line "[CapSoilSelfTest] PASS|FAIL ..." in the log; then the player
/// quits (code 0 or 3) if asked. CapCycleRecorder makes it on "soil_selftest": true ("soil_selftest_quit": true).
/// </summary>
public class CapSoilSelfTest : MonoBehaviour
{
    public Vector3 point = new Vector3(-25f, 0f, -10f);
    public bool quitAfter;

    private IEnumerator Start()
    {
        yield return new WaitForSeconds(3.0f);
        var soil = SoilParticleSettings.instance;
        string result;
        bool pass = false;
        if (soil == null || !soil.enable)
        {
            result = "no soil (SoilParticleSettings missing or disabled)";
        }
        else
        {
            var terrain = soil.GetComponent<Terrain>();
            var (gpu0, cpu0) = soil.HeightmapTexel(point);
            float v0 = Volume(terrain, point, 20);
            for (int i = 0; i < 20; i++)
            {
                Vector2 o = Random.insideUnitCircle * 0.5f;
                soil.DigUnit(point + new Vector3(o.x, 0f, o.y), 1f);
            }
            for (int i = 0; i < 3; i++) yield return new WaitForFixedUpdate();
            float v1 = Volume(terrain, point, 20);
            soil.Deposit(point, 0.3f, 0.35f);
            for (int i = 0; i < 3; i++) yield return new WaitForFixedUpdate();
            float v2 = Volume(terrain, point, 20);
            float dug = v0 - v1, expected = 20f * soil.DigUnitM3, put = v2 - v1;
            pass = Mathf.Abs(dug - expected) <= 0.02f * expected && Mathf.Abs(put - 0.3f) <= 0.006f
                   && Mathf.Abs(gpu0 - cpu0) <= 1e-4f;
            result = $"20 digs took {dug:F4} m3 of {expected:F4}, a 0.3 m3 deposit put {put:F4} m3, the working " +
                     $"heightmap {gpu0:F5} where the terrain is {cpu0:F5} " +
                     $"(particles {(soil.particles ? "on" : "off")}, {SystemInfo.graphicsDeviceType} {SystemInfo.graphicsDeviceName}, " +
                     $"compute {(SystemInfo.supportsComputeShaders ? "yes" : "no")})";
        }
        Debug.Log($"[CapSoilSelfTest] {(pass ? "PASS" : "FAIL")}: {result}");
        if (quitAfter)
        {
            yield return null;
            Application.Quit(pass ? 0 : 3);
        }
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
