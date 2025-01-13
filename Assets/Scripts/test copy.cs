using UnityEngine;
using System.Collections;
using System.Collections.Generic;
using System.IO;

public class TerrainSunRandomizerCopy : MonoBehaviour
{
    [Header("TerrainLayer 設定")]
    [SerializeField] private TerrainLayer[] terrainLayers;

    [Header("Prefab 設定")]
    public List<GameObject> prefabs;

    // 新しく追加: DepthToColor マテリアル (先ほど作成した Material をアサイン)
    [Header("Depth Capture Settings")]
    [Tooltip("DepthToColor.shader を使ったマテリアル")]
    [SerializeField] private Material depthToColorMaterial;

    // 生成したプレハブへの参照を保持
    private List<GameObject> spawnedPrefabs = new List<GameObject>();

    [SerializeField] private int numIterations = 10;
    private List<Camera> cameras = new List<Camera>();

    private void Start()
    {
        // 1. カメラをあらかじめ配置しておく
        PlaceCamerasInGrid();

        // 2. コルーチンで複数回の処理を実行
        StartCoroutine(RunMultipleIterations());
    }

    /// <summary>
    /// numIterations 回繰り返し実行
    /// </summary>
    private IEnumerator RunMultipleIterations()
    {
        for (int i = 0; i < numIterations; i++)
        {
            Debug.Log($"=== イテレーション {i} 開始 ===");
            // 1. 既にあるオブジェクトのクリーンアップ
            foreach (var go in spawnedPrefabs)
            {
                if (go != null) Destroy(go);
            }
            spawnedPrefabs.Clear();
            yield return null;

            GameObject oldTerrain = GameObject.Find("ProceduralTerrain100x100");
            if (oldTerrain != null)
            {
                Destroy(oldTerrain);
            }
            yield return null;

            // 2. Terrain や Sun をランダム生成 → 撮影
            yield return StartCoroutine(GenerateSceneAndCapture(i));

            Debug.Log($"=== イテレーション {i} 終了 ===");
        }

        Debug.Log("すべてのイテレーションが完了しました。");
    }

    /// <summary>
    /// 一連の流れを実行 (Terrain / Sun ランダム化、プレハブ配置、画像保存)
    /// </summary>
    private IEnumerator GenerateSceneAndCapture(int iterationIndex)
    {
        // ---- 1. Terrainの生成 ----
        TerrainData terrainData = new TerrainData();
        terrainData.heightmapResolution = 513;
        terrainData.size = new Vector3(100, 100, 100);

        // ---- 1-2. Perlin Noise で適当な起伏を作る ----
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

        // ---- 2. TerrainLayer を TerrainData に登録 ----
        if (terrainLayers != null && terrainLayers.Length > 0)
        {
            terrainData.terrainLayers = terrainLayers;
        }
        else
        {
            Debug.LogWarning("terrainLayers が設定されていません。");
        }

        // ---- 3. Terrain オブジェクトを生成 ----
        GameObject terrainGO = Terrain.CreateTerrainGameObject(terrainData);
        terrainGO.name = "ProceduralTerrain100x100";
        terrainGO.transform.position = new Vector3(-terrainData.size.x * 0.5f, 0f, -terrainData.size.z * 0.5f);

        // ---- 4. Terrain のアルファマップを設定して、ランダム or ノイズでレイヤーを塗り分ける ----
        ApplyRandomAlphaMap(terrainData);

        // ---- 5. 日光(Directional Light)のランダム化 ----
        RandomizeSun();

        // ---- 6. プレハブを配置 ----
        PlacePrefabsRandomly(terrainData);

        // ---- 7. カメラですべて撮影して PNG & EXR 保存 ----
        yield return StartCoroutine(SaveAllCameraImages(iterationIndex, terrainData));

        // 後処理があればここに
    }

    /// <summary>
    /// TerrainData のアルファマップをランダムに塗り分ける例
    /// </summary>
    private void ApplyRandomAlphaMap(TerrainData terrainData)
    {
        if (terrainLayers == null || terrainLayers.Length < 2) return;

        int alphaWidth = terrainData.alphamapWidth;
        int alphaHeight = terrainData.alphamapHeight;
        int layerCount = terrainData.alphamapLayers;

        float[,,] alphaMap = new float[alphaHeight, alphaWidth, layerCount];

        for (int y = 0; y < alphaHeight; y++)
        {
            for (int x = 0; x < alphaWidth; x++)
            {
                // デフォルトはレイヤー0のみ 100%
                for (int l = 0; l < layerCount; l++)
                {
                    alphaMap[y, x, l] = 0f;
                }
                alphaMap[y, x, 0] = 1f;

                // ノイズによってレイヤー1を混ぜる
                float noise = Mathf.PerlinNoise(x * 0.005f, y * 0.005f);
                if (noise > 0.3f)
                {
                    float mixRatio = (noise - 0.5f) * 2f; // 0 ~ 1
                    float layer1Amount = 1.0f * mixRatio;
                    float layer0Amount = 1f - layer1Amount;
                    alphaMap[y, x, 0] = layer0Amount;
                    alphaMap[y, x, 1] = layer1Amount;
                }
            }
        }

        terrainData.SetAlphamaps(0, 0, alphaMap);
    }

    /// <summary>
    /// シーン内の "Directional Light" を見つけて、ランダムに色や回転を変える
    /// </summary>
    private void RandomizeSun()
    {
        GameObject sun = GameObject.Find("Directional Light");
        if (sun != null)
        {
            Light sunLight = sun.GetComponent<Light>();
            float randomHour = Random.Range(9f, 16f);  // 9時～16時
            float altitude = Mathf.Lerp(40f, 10f, (randomHour - 9f) / 7f);
            float randomY = Random.Range(0f, 360f);
            sun.transform.rotation = Quaternion.Euler(altitude, randomY, 0f);

            float r = Random.Range(0.8f, 1f);
            float g = Random.Range(0.7f, 1f);
            float b = Random.Range(0.5f, 0.8f);
            sunLight.color = new Color(r, g, b);
            sunLight.intensity = Random.Range(0.5f, 2f);
        }
        else
        {
            Debug.LogWarning("Directional Lightが見つかりませんでした。シーン内に配置してください。");
        }
    }

    /// <summary>
    /// TerrainData にあわせてプレハブをランダム配置
    /// </summary>
    private void PlacePrefabsRandomly(TerrainData terrainData)
    {
        if (prefabs == null || prefabs.Count == 0)
        {
            Debug.LogWarning("プレハブが設定されていません。");
            return;
        }

        Vector3 terrainPosition = new Vector3(-terrainData.size.x * 0.5f, 0f, -terrainData.size.z * 0.5f);

        // プレハブ数ぶんだけ配置
        for (int i = 0; i < prefabs.Count; i++)
        {
            Vector3 randomPosition = new Vector3(
                Random.Range(20, terrainData.size.x - 20),
                0,
                Random.Range(20, terrainData.size.z - 20)
            );

            float terrainHeight = terrainData.GetHeight(
                (int)(randomPosition.x / terrainData.size.x * terrainData.heightmapResolution),
                (int)(randomPosition.z / terrainData.size.z * terrainData.heightmapResolution)
            );

            randomPosition.y = terrainHeight;
            randomPosition += terrainPosition;

            float randomYRotation = Random.Range(0f, 360f);

            GameObject spawned = Instantiate(prefabs[i], randomPosition, Quaternion.Euler(0f, randomYRotation, 0f));
            spawnedPrefabs.Add(spawned);
        }
    }

    /// <summary>
    /// 5x5 のグリッド状にカメラを配置 (先に一度だけ呼ぶ想定)
    /// </summary>
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
                Vector3 camPos = new Vector3(x, 50f, z);

                GameObject camGO = new GameObject($"Camera_{i}_{j}");
                camGO.transform.position = camPos;
                camGO.transform.rotation = Quaternion.Euler(90f, 0f, 0f); // 真下

                Camera cam = camGO.AddComponent<Camera>();
                cam.depthTextureMode = DepthTextureMode.Depth;
                cameras.Add(cam); // リストに覚えておく
            }
        }
    }

    /// <summary>
    /// すべてのカメラに対してスクショ (PNG, EXR) を撮り、指定フォルダに保存する
    /// </summary>
    private IEnumerator SaveAllCameraImages(int iterationIndex, TerrainData terrainData)
    {

        yield return null;

        string directoryPath = $"data_images/iteration_{iterationIndex}/";
        if (!Directory.Exists(directoryPath))
        {
            Directory.CreateDirectory(directoryPath);
        }

        // === 1. Terrain & 全プレハブが表示された状態で撮影 ===
        foreach (Camera cam in cameras)
        {
            SaveCameraImage(cam, directoryPath);
        }

        // === 2. Terrain を非表示にして撮影 ===
        GameObject terrainGO = GameObject.Find("ProceduralTerrain100x100");
        if (terrainGO != null)
        {
            terrainGO.SetActive(false);

            // 2-1. 全プレハブが表示された状態
            foreach (Camera cam in cameras)
            {
                SaveCameraImage(cam, directoryPath, "_noTerrain");
            }

            // 2-2. プレハブを一つずつだけ表示して撮影
            foreach (var prefabGO in spawnedPrefabs)
            {
                // 全部OFF
                foreach (var go in spawnedPrefabs)
                {
                    go.SetActive(false);
                }

                // 今回撮りたいプレハブだけON
                prefabGO.SetActive(true);

                // カメラごとに撮影
                foreach (Camera cam in cameras)
                {
                    SaveCameraImage(cam, directoryPath, $"_noTerrain_only_{prefabGO.name}");
                }
            }

            // 全プレハブを再度ONに戻す
            foreach (var go in spawnedPrefabs)
            {
                go.SetActive(true);
            }

            // Terrain もONに戻す
            terrainGO.SetActive(true);
        }
    }

    /// <summary>
    /// カメラからスクショを撮って PNG/EXR 等を保存する
    /// </summary>
    private void SaveCameraImage(Camera cam, string directoryPath, string suffix = "")
    {
        // ---(1) カラー画像を取得・保存 (PNG) ---
        RenderTexture colorRT = new RenderTexture(Screen.width, Screen.height, 24, RenderTextureFormat.Default);
        cam.targetTexture = colorRT;

        Texture2D screenShot = new Texture2D(Screen.width, Screen.height, TextureFormat.RGB24, false);

        // カメラ描画 → ReadPixels → PNG
        cam.Render();
        RenderTexture.active = colorRT;
        screenShot.ReadPixels(new Rect(0, 0, Screen.width, Screen.height), 0, 0);
        screenShot.Apply();

        cam.targetTexture = null;
        RenderTexture.active = null;
        colorRT.Release();
        Destroy(colorRT);

        byte[] colorBytes = screenShot.EncodeToPNG();
        string colorFilename = Path.Combine(directoryPath, $"{cam.name}{suffix}.png");
        File.WriteAllBytes(colorFilename, colorBytes);


        // --- カメラ外部パラメータを保存 (お好みで) ---
        string externalParams = $"Position: {cam.transform.position}\nRotation: {cam.transform.rotation}\n";
        File.WriteAllText(Path.Combine(directoryPath, $"{cam.name}{suffix}_external.txt"), externalParams);

        // --- カメラ内部パラメータを保存 (お好みで) ---
        string internalParams =
            $"Field of View: {cam.fieldOfView}\n" +
            $"Aspect Ratio: {cam.aspect}\n" +
            $"Near Clip Plane: {cam.nearClipPlane}\n" +
            $"Far Clip Plane: {cam.farClipPlane}\n";
        File.WriteAllText(Path.Combine(directoryPath, $"{cam.name}{suffix}_internal.txt"), internalParams);


        // ---(2) 深度画像をカスタムシェーダで取得・保存 (EXR)---

        // ▼ まず、カラー用の一時RTを作って再度カメラ描画 (カメラのカラー出力を受け取る)
        RenderTexture tempRT = new RenderTexture(Screen.width, Screen.height, 24, RenderTextureFormat.Default);
        cam.targetTexture = tempRT;
        cam.Render();

        // ▼ 深度→カラー化用のRT
        //   (浮動小数点フォーマットにしておいた方が、後段でEXR出力がスムーズ)
        RenderTexture depthRT = new RenderTexture(Screen.width, Screen.height, 0, RenderTextureFormat.ARGBFloat);

        // ▼ DepthToColor マテリアルを使って Blit
        //    _CameraDepthTexture をサンプリング → グレースケール出力
        Graphics.Blit(tempRT, depthRT, depthToColorMaterial);

        // ▼ ReadPixels
        Texture2D depthTex = new Texture2D(Screen.width, Screen.height, TextureFormat.RGBAFloat, false);
        RenderTexture.active = depthRT;
        depthTex.ReadPixels(new Rect(0, 0, Screen.width, Screen.height), 0, 0);
        depthTex.Apply();

        // ▼ EXR で書き出し
        byte[] depthBytes = depthTex.EncodeToEXR(Texture2D.EXRFlags.OutputAsFloat);
        string depthFilename = Path.Combine(directoryPath, $"{cam.name}{suffix}_depth.exr");
        File.WriteAllBytes(depthFilename, depthBytes);

        // ▼ 後始末
        cam.targetTexture = null;
        RenderTexture.active = null;
        tempRT.Release();
        depthRT.Release();
        Destroy(tempRT);
        Destroy(depthRT);
    }
}
