using UnityEngine;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using SD = System.Diagnostics; // 追加

[System.Serializable]
public class PrefabSpawnInfo
{
    public GameObject prefab;
    [Tooltip("このプレハブがシーン内に同時に出現できる最大数")]
    public int maxSpawnCount = 3;

    public float overlapMargin = 0.1f;
}

public class MakeDataset : MonoBehaviour
{
    // ▼ ここを修正：候補となる TerrainLayer を大量に登録しておく
    [Header("TerrainLayer 候補をすべて登録")]
    [SerializeField] private TerrainLayer[] allTerrainLayers;

    [Header("TerrainLayer のランダム選択数")]
    [Tooltip("1回の生成で選ぶ最小枚数")]
    [SerializeField] private int minLayersToUse = 2;
    [Tooltip("1回の生成で選ぶ最大枚数")]
    [SerializeField] private int maxLayersToUse = 4;

    [Header("Prefab 設定 (上限付き)")]
    public List<PrefabSpawnInfo> prefabsWithLimit;

    // DepthToColor 用のマテリアル
    [Header("Depth Capture Settings")]
    [Tooltip("DepthToColor.shader を使ったマテリアル")]
    [SerializeField] private Material depthToColorMaterial;

    // 生成した全プレハブインスタンス
    private List<GameObject> spawnedPrefabs = new List<GameObject>();

    // 元のプレハブごとにリスト化した生成インスタンス
    //   Key: 元プレハブ、Value: そのプレハブから生成されたインスタンス一覧
    private Dictionary<GameObject, List<GameObject>> spawnedPrefabInstances
        = new Dictionary<GameObject, List<GameObject>>();

    [SerializeField] private int numIterations = 1000;
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
            // 前回のイテレーションフォルダが消えるまで待機する等、不要なら削除可
            if (i > 0)
            {
                string prevDir = $"data_images/iteration_{i-1}";
                while (Directory.Exists(prevDir))
                {
                    Debug.Log($"前のイテレーションフォルダ {prevDir} がまだ存在します。削除されるまで待機します。");
                    yield return new WaitForSeconds(1f);
                }
            }
            if (i > 0)
            {
                string errorDir = $"data_images/iteration_{i-1}_error";
                if (Directory.Exists(errorDir))
                {
                    Debug.Log($"エラーフォルダが検出されました。再び{i-1}のイテレーションを実行します。");
                    Directory.Delete(errorDir, true);
                    i -= 1;
                }
            }
            Debug.Log($"=== イテレーション {i} 開始 ===");

            // メモリ使用量をチェック (お好みで)
            SD.Process currentProcess = SD.Process.GetCurrentProcess();
            long memoryUsage = currentProcess.PrivateMemorySize64;
            Debug.Log($"現在のメモリ使用量: {memoryUsage / (1024 * 1024)} MB");

            // 1. 既存オブジェクトのクリーンアップ
            foreach (var go in spawnedPrefabs)
            {
                if (go != null) Destroy(go);
            }
            spawnedPrefabs.Clear();
            spawnedPrefabInstances.Clear();

            yield return null;

            // 古い Terrain を削除
            GameObject oldTerrain = GameObject.Find("ProceduralTerrain100x100");
            if (oldTerrain != null)
            {
                Destroy(oldTerrain);
            }
            yield return null;

            // 2. Terrain や Sun をランダム生成 → 撮影
            yield return StartCoroutine(GenerateSceneAndCapture(i));

            Debug.Log($"=== イテレーション {i} 終了 ===");

            // メモリ解放
            Resources.UnloadUnusedAssets();
            System.GC.Collect();
        }

        Debug.Log("すべてのイテレーションが完了しました。");

        Time.timeScale = 0f; // ゲームを一時停止
        // Application.Quit();
    }

    /// <summary>
    /// シーン生成から撮影まで一連の処理
    /// </summary>
    private IEnumerator GenerateSceneAndCapture(int iterationIndex)
    {
        // ---- 1. Terrainの生成 ----
        TerrainData terrainData = new TerrainData();
        terrainData.heightmapResolution = 513;
        terrainData.size = new Vector3(50, 100, 70);

        // ---- 1-2. Perlin Noise で適当な起伏を作る ----
        float[,] heights = new float[terrainData.heightmapResolution, terrainData.heightmapResolution];
        float randomSeed = Random.Range(0f, 100f); // ランダムなシード
        for (int y = 0; y < terrainData.heightmapResolution; y++)
        {
            for (int x = 0; x < terrainData.heightmapResolution; x++)
            {
                float perlin = Mathf.PerlinNoise((x + randomSeed) * 0.01f, (y + randomSeed) * 0.01f) * 0.025f;
                heights[y, x] = perlin;
            }
        }

        // ---- 1-3. ランダムに台地を追加 (オプション) ----
        if (Random.value > 0.8f) // 20% の確率で台地を追加
        {
            int plateauX = Random.Range(0, terrainData.heightmapResolution);
            int plateauY = Random.Range(0, terrainData.heightmapResolution);
            int plateauWidth = Random.Range(30, 100);
            int plateauHeight = Random.Range(30, 100);
            float plateauMaxHeight = 0.05f;

            for (int y = plateauY - plateauHeight; y < plateauY + plateauHeight && y < terrainData.heightmapResolution; y++)
            {
                for (int x = plateauX - plateauWidth; x < plateauX + plateauWidth && x < terrainData.heightmapResolution; x++)
                {
                    if (x >= 0 && y >= 0)
                    {
                        float distanceToCenterX = Mathf.Abs(x - plateauX) / (float)plateauWidth;
                        float distanceToCenterY = Mathf.Abs(y - plateauY) / (float)plateauHeight;
                        float distanceToCenter = Mathf.Sqrt(distanceToCenterX * distanceToCenterX + distanceToCenterY * distanceToCenterY);
                        float fadeFactor = Mathf.SmoothStep(0f, 1f, 1f - distanceToCenter);
                        heights[y, x] += plateauMaxHeight * fadeFactor;
                    }
                }
            }
        }
        terrainData.SetHeights(0, 0, heights);

        // ---- 2. ランダムな TerrainLayer セットを TerrainData に登録 ----
        SetupRandomTerrainLayers(terrainData);

        // ---- 3. Terrain オブジェクトを生成 ----
        GameObject terrainGO = Terrain.CreateTerrainGameObject(terrainData);
        terrainGO.name = "ProceduralTerrain100x100";
        terrainGO.transform.position = new Vector3(
            -terrainData.size.x * 0.5f,
            0f,
            -terrainData.size.z * 0.5f
        );

        // ---- 4. AlphaMap をノイズなどで塗り分け ----
        ApplyRandomAlphaMap(terrainData);

        // ---- 5. Directional Light (Sun) をランダム化 ----
        RandomizeSun();

        // ---- 6. プレハブを配置 (上限数を考慮したランダム配置) ----
        PlacePrefabsRandomly(terrainData);

        // ---- 7. カメラでスクリーンショットを撮ってファイル出力 ----
        yield return StartCoroutine(SaveAllCameraImages(iterationIndex, terrainData));
    }

    /// <summary>
    /// (1) allTerrainLayers からランダムな数だけレイヤーを選んで
    /// (2) terrainData.terrainLayers にセットする
    /// </summary>
    private void SetupRandomTerrainLayers(TerrainData terrainData)
    {
        if (allTerrainLayers == null || allTerrainLayers.Length == 0)
        {
            Debug.LogWarning("TerrainLayer の候補が設定されていません。");
            return;
        }

        // 使うレイヤー枚数をランダムに決定
        int layerCountToUse = Random.Range(minLayersToUse, maxLayersToUse + 1);

        // 重複を避けたい場合は Shuffle & Take
        List<TerrainLayer> shuffled = new List<TerrainLayer>(allTerrainLayers);
        // ランダム順にシャッフル
        for (int i = 0; i < shuffled.Count; i++)
        {
            int r = Random.Range(i, shuffled.Count);
            var tmp = shuffled[i];
            shuffled[i] = shuffled[r];
            shuffled[r] = tmp;
        }
        // 先頭から layerCountToUse 枚を選ぶ
        List<TerrainLayer> selectedLayers = new List<TerrainLayer>();
        for (int i = 0; i < layerCountToUse; i++)
        {
            selectedLayers.Add(shuffled[i]);
        }

        // 選択したレイヤーを TerrainData に登録
        terrainData.terrainLayers = selectedLayers.ToArray();
    }

    /// <summary>
    /// TerrainData のアルファマップを「複数レイヤー」向けにノイズやランダムで塗り分ける
    /// </summary>
    private void ApplyRandomAlphaMap(TerrainData terrainData)
    {
        if (terrainData.terrainLayers == null || terrainData.terrainLayers.Length == 0) return;

        int alphaWidth = terrainData.alphamapWidth;
        int alphaHeight = terrainData.alphamapHeight;
        int layerCount = terrainData.alphamapLayers;

        float[,,] alphaMap = new float[alphaHeight, alphaWidth, layerCount];

        // ノイズスケールをランダム化するなどお好みで
        float noiseScale = Random.Range(0.003f, 0.01f);
        float noiseOffsetX = Random.Range(0f, 100f);
        float noiseOffsetY = Random.Range(0f, 100f);

        for (int y = 0; y < alphaHeight; y++)
        {
            for (int x = 0; x < alphaWidth; x++)
            {
                // 各レイヤーごとにノイズ値を取得
                float total = 0f;
                float[] noiseValues = new float[layerCount];

                for (int l = 0; l < layerCount; l++)
                {
                    // レイヤー毎にちょっとノイズ座標をずらすテクニック
                    float nx = (x + noiseOffsetX + l * 13.4f) * noiseScale;
                    float ny = (y + noiseOffsetY + l * 7.9f) * noiseScale;

                    float val = Mathf.PerlinNoise(nx, ny);
                    // 0～1のままだと重なりが少ないので、多少パラメータ調整可
                    noiseValues[l] = Mathf.Pow(val, 1.2f); 
                    total += noiseValues[l];
                }

                // 合計が0なら全部0に…は避けたいので一応対策
                if (total < 0.0001f) total = 0.0001f;

                // 正規化して書き込み
                for (int l = 0; l < layerCount; l++)
                {
                    alphaMap[y, x, l] = noiseValues[l] / total;
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
    /// TerrainData にあわせてプレハブをランダム配置 (各プレハブの最大数を考慮)。
    /// 生成したインスタンスは spawnedPrefabs と spawnedPrefabInstances に登録。
    /// </summary>
    private void PlacePrefabsRandomly(TerrainData terrainData)
    {
        if (prefabsWithLimit == null || prefabsWithLimit.Count == 0)
        {
            Debug.LogWarning("プレハブが設定されていません。");
            return;
        }

        Vector3 terrainPosition = new Vector3(
            -terrainData.size.x * 0.5f,
            0f,
            -terrainData.size.z * 0.5f
        );

        // プレハブ単位で処理
        foreach (var info in prefabsWithLimit)
        {
            if (info.prefab == null || info.maxSpawnCount <= 0) continue;

            // 0～maxSpawnCountの間でランダムに個数を決める
            int spawnCount = Random.Range(0, info.maxSpawnCount + 1);

            // まだ登録がなければ初期化
            if (!spawnedPrefabInstances.ContainsKey(info.prefab))
            {
                spawnedPrefabInstances[info.prefab] = new List<GameObject>();
            }

            // ▼ BoxCollider を取得しておく
            BoxCollider prefabCollider = info.prefab.GetComponent<BoxCollider>();
            if (prefabCollider == null)
            {
                Debug.LogWarning($"Prefab {info.prefab.name} にBoxColliderがありません。重なりチェックできません。");
            }

            // spawnCount回 配置を試みる
            for (int instanceIndex = 0; instanceIndex < spawnCount; instanceIndex++)
            {
                bool placed = false;
                int maxAttempts = 100;

                for (int attempt = 0; attempt < maxAttempts; attempt++)
                {
                    // ランダム座標
                    Vector3 randomPosition = new Vector3(
                        Random.Range(5, terrainData.size.x - 5),
                        0,
                        Random.Range(5, terrainData.size.z - 5)
                    );
                    // 高さをTerrainからサンプル
                    float terrainHeight = terrainData.GetHeight(
                        (int)(randomPosition.x / terrainData.size.x * terrainData.heightmapResolution),
                        (int)(randomPosition.z / terrainData.size.z * terrainData.heightmapResolution)
                    );
                    randomPosition.y = terrainHeight;
                    randomPosition += terrainPosition;

                    // ランダム回転
                    Quaternion randomRotation = Quaternion.Euler(0f, Random.Range(0f, 360f), 0f);

                    // 重なりチェック
                    if (prefabCollider != null)
                    {
                        // 中心
                        Vector3 worldCenter = randomPosition + randomRotation * prefabCollider.center;
                        Vector3 halfExtents = (prefabCollider.size * 0.5f) + Vector3.one * info.overlapMargin;

                        Collider[] hits = Physics.OverlapBox(
                            worldCenter,
                            halfExtents,
                            randomRotation
                        );

                        if (hits.Length == 0)
                        {
                            // 配置OK
                            GameObject instance = Instantiate(info.prefab, randomPosition, randomRotation);
                            spawnedPrefabs.Add(instance);
                            spawnedPrefabInstances[info.prefab].Add(instance);
                            placed = true;
                            break;
                        }
                    }
                    else
                    {
                        // BoxCollider が無いプレハブの場合のフォールバック
                        GameObject tempInstance = Instantiate(info.prefab, randomPosition, randomRotation);

                        Bounds newBounds = tempInstance.GetComponentInChildren<Renderer>().bounds;
                        bool overlaps = false;
                        foreach (var existingGO in spawnedPrefabs)
                        {
                            if (existingGO == null) continue;
                            Bounds existingBounds = existingGO.GetComponentInChildren<Renderer>().bounds;
                            if (newBounds.Intersects(existingBounds))
                            {
                                overlaps = true;
                                break;
                            }
                        }

                        if (!overlaps)
                        {
                            spawnedPrefabs.Add(tempInstance);
                            spawnedPrefabInstances[info.prefab].Add(tempInstance);
                            placed = true;
                            break;
                        }
                        else
                        {
                            Destroy(tempInstance);
                        }
                    }
                }

                if (!placed)
                {
                    Debug.LogWarning($"{info.prefab.name} の配置（インスタンス {instanceIndex + 1}）に失敗しました。");
                }
            }
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
                Vector3 camPos = new Vector3(x, 30f, z);

                GameObject camGO = new GameObject($"Camera_{i}_{j}");
                camGO.transform.position = camPos;
                camGO.transform.rotation = Quaternion.Euler(90f, 0f, 0f); // 真下を向く

                Camera cam = camGO.AddComponent<Camera>();
                cam.depthTextureMode = DepthTextureMode.Depth;
                cameras.Add(cam);
            }
        }
    }

    /// <summary>
    /// カメラでカラー/深度を撮影してファイルに保存
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

        // === 2. Terrainを非表示にして撮影 ===
        GameObject terrainGO = GameObject.Find("ProceduralTerrain100x100");
        if (terrainGO != null)
        {
            // 2-1. Terrain OFF & 全プレハブ ON
            terrainGO.SetActive(false);
            foreach (Camera cam in cameras)
            {
                SaveCameraImage(cam, directoryPath, "_noTerrain");
            }

            // 2-2. プレハブの種類ごとに撮影 (他をOFFにしてその種類だけON)
            foreach (var kvp in spawnedPrefabInstances)
            {
                GameObject prefabType = kvp.Key;
                List<GameObject> instances = kvp.Value;

                // 全OFF
                foreach (var go in spawnedPrefabs)
                {
                    if (go != null) go.SetActive(false);
                }
                // 今回撮りたい種類だけ ON
                foreach (var inst in instances)
                {
                    if (inst != null) inst.SetActive(true);
                }
                // カメラごとに撮影
                foreach (Camera cam in cameras)
                {
                    SaveCameraImage(cam, directoryPath, $"_noTerrain_only_{prefabType.name}");
                }
            }
            // 終わったら元に戻す
            foreach (var go in spawnedPrefabs)
            {
                if (go != null) go.SetActive(true);
            }
            terrainGO.SetActive(true);
        }
    }

    /// <summary>
    /// カメラから画像を取得して保存する処理（カラーPNG & 深度EXR）
    /// </summary>
    private void SaveCameraImage(Camera cam, string directoryPath, string suffix = "")
    {
        // ---(1) カラー画像を取得・保存 (PNG) ---
        RenderTexture colorRT = new RenderTexture(Screen.width, Screen.height, 24, RenderTextureFormat.Default);
        cam.targetTexture = colorRT;

        Texture2D screenShot = new Texture2D(Screen.width, Screen.height, TextureFormat.RGB24, false);

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

        Destroy(screenShot);

        // カメラ外部パラメータ（お好みで）
        string externalParams = $"Position: {cam.transform.position}\nRotation: {cam.transform.rotation}\n";
        File.WriteAllText(Path.Combine(directoryPath, $"{cam.name}{suffix}_external.txt"), externalParams);

        // カメラ内部パラメータ（お好みで）
        string internalParams =
            $"Field of View: {cam.fieldOfView}\n" +
            $"Aspect Ratio: {cam.aspect}\n" +
            $"Near Clip Plane: {cam.nearClipPlane}\n" +
            $"Far Clip Plane: {cam.farClipPlane}\n";
        File.WriteAllText(Path.Combine(directoryPath, $"{cam.name}{suffix}_internal.txt"), internalParams);

        // ---(2) 深度画像をカスタムシェーダ(DepthToColor)で取得・保存 (EXR)---
        RenderTexture tempRT = new RenderTexture(Screen.width, Screen.height, 24, RenderTextureFormat.Default);
        cam.targetTexture = tempRT;
        cam.Render();

        RenderTexture depthRT = new RenderTexture(Screen.width, Screen.height, 0, RenderTextureFormat.ARGBFloat);

        // DepthToColor マテリアルを使って Blit
        Graphics.Blit(tempRT, depthRT, depthToColorMaterial);

        Texture2D depthTex = new Texture2D(Screen.width, Screen.height, TextureFormat.RGBAFloat, false);
        RenderTexture.active = depthRT;
        depthTex.ReadPixels(new Rect(0, 0, Screen.width, Screen.height), 0, 0);
        depthTex.Apply();

        byte[] depthBytes = depthTex.EncodeToEXR(Texture2D.EXRFlags.OutputAsFloat);
        string depthFilename = Path.Combine(directoryPath, $"{cam.name}{suffix}_depth.exr");
        File.WriteAllBytes(depthFilename, depthBytes);

        cam.targetTexture = null;
        RenderTexture.active = null;
        tempRT.Release();
        depthRT.Release();
        Destroy(tempRT);
        Destroy(depthRT);
        Destroy(depthTex);
    }
}
