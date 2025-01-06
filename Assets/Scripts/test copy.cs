using UnityEngine;
using System.Collections;
using System.Collections.Generic;

public class TerrainSunRandomizer : MonoBehaviour
{
    [Tooltip("配置するプレハブのリスト")]
    public List<GameObject> prefabs;

    private void Start()
    {
        // ---- 1. Terrainの生成 ----
        TerrainData terrainData = new TerrainData();
        terrainData.heightmapResolution = 513;
        float[,] heights = new float[terrainData.heightmapResolution, terrainData.heightmapResolution];
        for (int y = 0; y < terrainData.heightmapResolution; y++)
        {
            for (int x = 0; x < terrainData.heightmapResolution; x++)
            {
                float perlin = Mathf.PerlinNoise(x * 0.01f, y * 0.01f) * 0.025f;
                heights[y, x] = perlin;
            }
        }
        terrainData.SetHeights(0, 0, heights);
        // 幅100, 最大高さ100, 奥行100
        terrainData.size = new Vector3(100, 100, 100);
        
        // Terrainオブジェクトを生成
        GameObject terrainGO = Terrain.CreateTerrainGameObject(terrainData);
        terrainGO.name = "ProceduralTerrain100x100";
        terrainGO.transform.position = new Vector3(-terrainData.size.x * 0.5f, 0f, -terrainData.size.z * 0.5f);

        // ---- 2. 日光(Directional Light)のランダム化 ----
        // シーン内に"Directional Light"という名前で置かれている想定
        GameObject sun = GameObject.Find("Directional Light");
        if (sun != null)
        {
            // ライトコンポーネントを取得
            Light sunLight = sun.GetComponent<Light>();

            // 角度をランダムにする（X軸、Y軸をランダム回転）
            // ※ Z回転もランダムにすると太陽が横倒しになるため、不要なら0固定が普通です
            float randomHour = Random.Range(9f, 16f);
            // 例: 9時で約40度、16時で約10度にするなど適当に変換
            float altitude = Mathf.Lerp(40f, 10f, (randomHour - 9f) / 7f);
            float randomY = Random.Range(0f, 360f);
            sun.transform.rotation = Quaternion.Euler(altitude, randomY, 0f);

            // 光の色をランダムに(0.5〜1.0の範囲で少し淡くするなど工夫してもよい)
            sunLight.color = new Color(Random.value, Random.value, Random.value);

            // 光の強度(intensity)をランダムに(0.5〜2.0の範囲など)
            sunLight.intensity = Random.Range(0.5f, 2f);
        }
        else
        {
            Debug.LogWarning("Directional Lightが見つかりませんでした。シーン内に配置してください。");
        }

        PlaceCamerasInGrid();
        StartCoroutine(SaveAllCameraImages());
        PlacePrefabsRandomly(terrainData);
    }

    private void PlaceCamerasInGrid()
    {
        int gridCount = 5;
        float spacing = 20f;
        float halfExtent = (gridCount - 1) * spacing * 0.5f;

        for (int i = 0; i < gridCount; i++)
        {
            for (int j = 0; j < gridCount; j++)
            {
                float x = i * spacing - halfExtent;
                float z = j * spacing - halfExtent;
                Vector3 camPos = new Vector3(x, 20f, z);
                GameObject camGO = new GameObject($"Camera_{i}_{j}");
                camGO.transform.position = camPos;
                // 真下を向く
                camGO.transform.rotation = Quaternion.Euler(90f, 0f, 0f);
                camGO.AddComponent<Camera>();
            }
        }
    }

    private IEnumerator SaveAllCameraImages()
    {
        yield return new WaitForEndOfFrame();
        Camera[] cameras = FindObjectsOfType<Camera>();
        string directoryPath = Application.dataPath + "/images/";
        if (!System.IO.Directory.Exists(directoryPath))
        {
            System.IO.Directory.CreateDirectory(directoryPath);
        }

        foreach (Camera cam in cameras)
        {
            // RenderTextureを使ってカメラごとの画像を取得
            RenderTexture rt = new RenderTexture(Screen.width, Screen.height, 24);
            cam.targetTexture = rt;
            Texture2D screenShot = new Texture2D(Screen.width, Screen.height, TextureFormat.RGB24, false);
            cam.Render();
            RenderTexture.active = rt;
            screenShot.ReadPixels(new Rect(0, 0, Screen.width, Screen.height), 0, 0);
            cam.targetTexture = null;
            RenderTexture.active = null;
            Destroy(rt);

            // PNGとして保存
            byte[] bytes = screenShot.EncodeToPNG();
            string filename = directoryPath + $"{cam.name}.png";
            System.IO.File.WriteAllBytes(filename, bytes);
            Debug.Log($"Saved {filename}");

            // カメラの外部パラメータ（位置と回転）を保存
            string externalParams = $"Position: {cam.transform.position}\nRotation: {cam.transform.rotation}\n";
            System.IO.File.WriteAllText(directoryPath + $"{cam.name}_external.txt", externalParams);

            // カメラの内部パラメータ（視野角など）を保存
            string internalParams = $"Field of View: {cam.fieldOfView}\nAspect Ratio: {cam.aspect}\nNear Clip Plane: {cam.nearClipPlane}\nFar Clip Plane: {cam.farClipPlane}\n";
            System.IO.File.WriteAllText(directoryPath + $"{cam.name}_internal.txt", internalParams);

            // 深度情報を保存
            RenderTexture depthRT = new RenderTexture(Screen.width, Screen.height, 24, RenderTextureFormat.Depth);
            cam.targetTexture = depthRT;
            Texture2D depthTexture = new Texture2D(Screen.width, Screen.height, TextureFormat.RFloat, false);
            cam.Render();
            RenderTexture.active = depthRT;
            depthTexture.ReadPixels(new Rect(0, 0, Screen.width, Screen.height), 0, 0);
            cam.targetTexture = null;
            RenderTexture.active = null;
            Destroy(depthRT);

            byte[] depthBytes = depthTexture.EncodeToEXR();
            string depthFilename = directoryPath + $"{cam.name}_depth.exr";
            System.IO.File.WriteAllBytes(depthFilename, depthBytes);
            Debug.Log($"Saved {depthFilename}");
        }
    }

    private void PlacePrefabsRandomly(TerrainData terrainData)
    {
        if (prefabs == null || prefabs.Count == 0)
        {
            Debug.LogWarning("プレハブが設定されていません。");
            return;
        }

        Vector3 terrainPosition = new Vector3(-terrainData.size.x * 0.5f, 0f, -terrainData.size.z * 0.5f);

        for (int i = 0; i < prefabs.Count; i++)
        {
            Vector3 randomPosition = new Vector3(
                Random.Range(0, terrainData.size.x),
                5,
                Random.Range(0, terrainData.size.z)
            );

            float terrainHeight = terrainData.GetHeight(
                (int)(randomPosition.x / terrainData.size.x * terrainData.heightmapResolution),
                (int)(randomPosition.z / terrainData.size.z * terrainData.heightmapResolution)
            );

            randomPosition.y = terrainHeight;

            // Terrainの位置に合わせてランダムな位置を調整
            randomPosition += terrainPosition;

            Instantiate(prefabs[i], randomPosition, Quaternion.identity);
        }
    }
}
