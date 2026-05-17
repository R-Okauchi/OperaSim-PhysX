using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Std;
using System.Globalization;
using System.Text;

/// <summary>
/// Unity Terrain の heightmap をグリッドサンプリングして ROS topic で publish する。
/// ドローン SfM 測量で得られる DEM（数値標高モデル）をシミュレートする。
///
/// 初回 publish: TerrainTiler 初期化を待つために publishDelaySec 秒後。
/// その後 republishIntervalSec ごとに再 publish。
/// DDS (rclpy) の Bridge は late joiner なので、定期再 publish しないと
/// 起動順や Bridge 再起動で DEM を取りこぼす（VOLATILE durability のため）。
///
/// TerrainTiler で分割されたタイルにも対応。
/// </summary>
public class TerrainDEMPublisher : MonoBehaviour
{
    ROSConnection ros;

    [Tooltip("DEM を publish する ROS トピック名")]
    public string topicName = "/terrain/dem";

    [Tooltip("サンプリング解像度 (m)。0.5 = 50cm 間隔")]
    public float cellSizeM = 0.5f;

    [Tooltip("初回 publish までの待機秒数 (TerrainTiler の初期化を待つ)")]
    public float publishDelaySec = 3.0f;

    [Tooltip("再 publish 間隔 (秒)。0 で one-shot。late-joining Bridge を救う")]
    public float republishIntervalSec = 15.0f;

    private bool initialPublished = false;
    private float timeSinceLastPublish = 0f;

    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();
        ros.RegisterPublisher<StringMsg>(topicName);
    }

    void Update()
    {
        if (!initialPublished)
        {
            publishDelaySec -= Time.deltaTime;
            if (publishDelaySec > 0) return;

            initialPublished = true;
            timeSinceLastPublish = 0f;
            PublishDEM();
            return;
        }

        if (republishIntervalSec <= 0f) return;

        timeSinceLastPublish += Time.deltaTime;
        if (timeSinceLastPublish >= republishIntervalSec)
        {
            timeSinceLastPublish = 0f;
            PublishDEM();
        }
    }

    void PublishDEM()
    {
        // Collect all active terrains (TerrainTiler splits into tiles)
        Terrain[] terrains = Terrain.activeTerrains;
        if (terrains == null || terrains.Length == 0)
        {
            Debug.LogWarning("[TerrainDEMPublisher] No active terrains found");
            return;
        }

        // Compute world-space bounding box of all terrains
        float worldMinX = float.MaxValue, worldMaxX = float.MinValue;
        float worldMinZ = float.MaxValue, worldMaxZ = float.MinValue;
        foreach (var t in terrains)
        {
            Vector3 pos = t.transform.position;
            Vector3 size = t.terrainData.size;
            worldMinX = Mathf.Min(worldMinX, pos.x);
            worldMaxX = Mathf.Max(worldMaxX, pos.x + size.x);
            worldMinZ = Mathf.Min(worldMinZ, pos.z);
            worldMaxZ = Mathf.Max(worldMaxZ, pos.z + size.z);
        }

        Debug.Log($"[TerrainDEMPublisher] Terrain bounds: Unity X=[{worldMinX},{worldMaxX}], Z=[{worldMinZ},{worldMaxZ}], cellSize={cellSizeM}m");

        // Grid sampling in Unity world space, then convert to ROS frame
        // Unity (x, y, z) -> ROS (z, -x, y)
        var sb = new StringBuilder();
        sb.Append("{\"cells\":[");
        int cellCount = 0;
        bool first = true;

        for (float ux = worldMinX + cellSizeM * 0.5f; ux < worldMaxX; ux += cellSizeM)
        {
            for (float uz = worldMinZ + cellSizeM * 0.5f; uz < worldMaxZ; uz += cellSizeM)
            {
                // Sample height from whichever terrain contains this point
                float unityHeight = SampleTerrainHeight(terrains, ux, uz);

                // Unity -> ROS coordinate conversion
                // Unity (x, y, z) -> ROS (z, -x, y)
                float rosX = uz;
                float rosY = -ux;
                float rosZ = unityHeight;  // height

                if (!first) sb.Append(",");
                first = false;
                string xStr = rosX.ToString("F2", CultureInfo.InvariantCulture);
                string yStr = rosY.ToString("F2", CultureInfo.InvariantCulture);
                string hStr = rosZ.ToString("F2", CultureInfo.InvariantCulture);
                sb.Append("{\"x\":").Append(xStr)
                  .Append(",\"y\":").Append(yStr)
                  .Append(",\"height_m\":").Append(hStr).Append("}");
                cellCount++;
            }
        }

        sb.Append("],\"cell_size_m\":")
          .Append(cellSizeM.ToString("F2", CultureInfo.InvariantCulture))
          .Append(",\"cell_count\":").Append(cellCount).Append("}");

        var msg = new StringMsg(sb.ToString());
        ros.Publish(topicName, msg);

        Debug.Log($"[TerrainDEMPublisher] Published DEM: {cellCount} cells at {cellSizeM}m resolution");
    }

    /// <summary>
    /// Sample terrain height at Unity world position (x, z).
    /// Checks all active terrain tiles and returns the height from the
    /// tile that contains the point.
    /// </summary>
    float SampleTerrainHeight(Terrain[] terrains, float worldX, float worldZ)
    {
        foreach (var t in terrains)
        {
            Vector3 pos = t.transform.position;
            Vector3 size = t.terrainData.size;

            if (worldX >= pos.x && worldX <= pos.x + size.x &&
                worldZ >= pos.z && worldZ <= pos.z + size.z)
            {
                return t.SampleHeight(new Vector3(worldX, 0, worldZ));
            }
        }

        // Fallback: try SampleHeight on first terrain (may extrapolate)
        if (terrains.Length > 0)
        {
            return terrains[0].SampleHeight(new Vector3(worldX, 0, worldZ));
        }
        return 0f;
    }
}
