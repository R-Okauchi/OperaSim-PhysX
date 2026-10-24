using System;
using System.Globalization;
using System.Text;
using RosMessageTypes.Std;
using Unity.Robotics.ROSTCPConnector;
using UnityEngine;

/// <summary>
/// The ground's survey for CAP, as a drone survey would give it: the soil's working heightmap (the float heights, the
/// blade's and the bucket's changes in them this very step) over a ROS rectangle, at its own vertices (no resampling,
/// no half-cell shift), in metres to 0.1 mm, as JSON on /terrain/dem_grid - every period and whenever a request comes
/// on /terrain/dem_grid/request ({"id": ..., "x0": ..., "y0": ..., "x1": ..., "y1": ...}: answered with that region and
/// its id, so a Site can wait for a survey newer than the work it just saw end).
///   {"format": "grid_v1", "frame": "map", "source": "unity_working_heightmap", "surveyed": true, "sigma_m": 0.0005,
///    "id": ..., "stamp": ..., "x0": ..., "y0": ..., "cell_m": ..., "nx": ..., "ny": ..., "heights_m": [...]}
/// heights_m[j * nx + i] is at (x0 + i cell, y0 + j cell). /terrain/dem (the older cells format) is left as it is.
/// </summary>
public class CapTerrainDEMPublisher : MonoBehaviour
{
    public float x0 = 5f, y0 = -45f, x1 = 35f, y1 = -25f;    // the region sent every period (ROS)
    public float periodS = 3f;
    public string topic = "/terrain/dem_grid";
    public string requestTopic = "/terrain/dem_grid/request";

    private ROSConnection ros;
    private float since;
    private string pending;                                  // a request's id and region, answered at the next frame

    public class HeightGrid
    {
        public float x0, y0, cell;
        public int nx, ny;
        public float[] heights;                              // [j * nx + i], metres (world)
    }

    [Serializable]
    private class Request
    {
        public string id;
        public float x0, y0, x1, y1;
    }

    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();
        ros.RegisterPublisher<StringMsg>(topic);
        ros.Subscribe<StringMsg>(requestTopic, m => pending = m.data);
        Debug.Log($"[{nameof(CapTerrainDEMPublisher)}] {topic}: ({x0}, {y0})..({x1}, {y1}) every {periodS} s and on {requestTopic}");
    }

    void Update()
    {
        if (pending != null)
        {
            string req = pending;
            pending = null;
            try
            {
                var r = JsonUtility.FromJson<Request>(req);
                Send(r.id ?? "", r.x0, r.y0, r.x1, r.y1);
            }
            catch (Exception e)
            {
                Debug.LogWarning($"[{nameof(CapTerrainDEMPublisher)}] request unreadable ({e.Message}): {req}");
            }
        }
        since += Time.deltaTime;
        if (periodS > 0f && since >= periodS)
        {
            since = 0f;
            Send("", x0, y0, x1, y1);
        }
    }

    private void Send(string id, float ax, float ay, float bx, float by)
    {
        var g = Grid(ax, ay, bx, by);
        if (g == null) return;
        var sb = new StringBuilder(g.heights.Length * 9 + 256);
        sb.Append(string.Format(CultureInfo.InvariantCulture,
            "{{\"format\": \"grid_v1\", \"frame\": \"map\", \"source\": \"unity_working_heightmap\", \"surveyed\": true, " +
            "\"sigma_m\": 0.0005, \"id\": \"{0}\", \"stamp\": {1:F3}, \"x0\": {2:F4}, \"y0\": {3:F4}, \"cell_m\": {4:F7}, " +
            "\"nx\": {5}, \"ny\": {6}, \"heights_m\": [", id, Time.timeAsDouble, g.x0, g.y0, g.cell, g.nx, g.ny));
        for (int k = 0; k < g.heights.Length; k++)
        {
            if (k > 0) sb.Append(',');
            sb.Append(g.heights[k].ToString("F4", CultureInfo.InvariantCulture));
        }
        sb.Append("]}");
        ros.Publish(topic, new StringMsg(sb.ToString()));
    }

    /// <summary>The working heightmap over a ROS rectangle, at its vertices: ROS x along the texel rows (Unity z), ROS y
    /// along the texel columns reversed (Unity x = -ROS y). Null without the soil.</summary>
    public static HeightGrid Grid(float ax, float ay, float bx, float by)
    {
        var soil = SoilParticleSettings.instance;
        if (soil == null || soil.WorkingHeightmap == null) return null;
        var terrain = soil.GetComponent<Terrain>();
        Vector3 p = terrain.transform.position, size = terrain.terrainData.size;
        var res = soil.HeightmapSize;
        float cell = size.x / (res.x - 1);
        // ROS x in [ax, bx] -> Unity z -> texel y; ROS y in [ay, by] -> Unity x = -y -> texel x
        int ty0 = Mathf.Clamp(Mathf.CeilToInt((Mathf.Min(ax, bx) - p.z) / cell), 0, res.y - 1);
        int ty1 = Mathf.Clamp(Mathf.FloorToInt((Mathf.Max(ax, bx) - p.z) / cell), 0, res.y - 1);
        int tx0 = Mathf.Clamp(Mathf.CeilToInt((-Mathf.Max(ay, by) - p.x) / cell), 0, res.x - 1);
        int tx1 = Mathf.Clamp(Mathf.FloorToInt((-Mathf.Min(ay, by) - p.x) / cell), 0, res.x - 1);
        int w = tx1 - tx0 + 1, h = ty1 - ty0 + 1;
        if (w <= 0 || h <= 0) return null;
        var prev = RenderTexture.active;
        RenderTexture.active = soil.WorkingHeightmap;
        var tex = new Texture2D(w, h, TextureFormat.RGBAFloat, false, true);
        tex.ReadPixels(new Rect(tx0, ty0, w, h), 0, 0, false);
        tex.Apply(false, false);
        RenderTexture.active = prev;
        var px = tex.GetPixels();
        UnityEngine.Object.Destroy(tex);
        var g = new HeightGrid
        {
            cell = cell, nx = h, ny = w,
            x0 = p.z + ty0 * cell,                            // the first row's ROS x
            y0 = -(p.x + tx1 * cell),                         // the last column's ROS y (ROS y rises as texel x falls)
            heights = new float[w * h],
        };
        float scale = size.y / (32766f / 65535f);
        for (int j = 0; j < w; j++)                           // ROS y rising: texel x from tx1 down
        for (int i = 0; i < h; i++)                           // ROS x rising: texel y from ty0 up
        {
            int u = (tx1 - j) - tx0, v = i;
            g.heights[j * g.nx + i] = p.y + px[v * w + u].r * scale;
        }
        return g;
    }
}
