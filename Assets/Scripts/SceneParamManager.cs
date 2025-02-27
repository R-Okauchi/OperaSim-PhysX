// using UnityEngine;
// using UnityEngine.Networking;
// using System.Collections;

// public class SceneParamManager : MonoBehaviour
// {
//     // TerrainLayer の候補（インスペクタで全部登録しておく）
//     public TerrainLayer[] terrainLayers;

//     // Prefab設定 (上限付き) 
//     public List<PrefabSpawnInfo> prefabsWithLimit;

//     // Directional Light の参照（シーン内にあるライトをアサインしておく）
//     [SerializeField] private Light sunLight;

//     /// <summary>
//     /// Pythonサーバ (FastAPI) からシーンパラメータを取得し、それを元にTerrainやPrefab等をセットアップする。
//     /// </summary>
//     public IEnumerator FetchAndApplySceneParams()
//     {
//         string url = "http://127.0.0.1:8000/getSceneParams"; 
//         using(UnityWebRequest req = UnityWebRequest.Get(url))
//         {
//             yield return req.SendWebRequest();
//             if (req.result != UnityWebRequest.Result.Success)
//             {
//                 Debug.LogError($"Error fetching scene params: {req.error}");
//                 yield break;
//             }

//             // JSONを文字列で取得
//             string jsonText = req.downloadHandler.text;
//             // パース
//             SceneParams sceneParams = JsonUtility.FromJson<SceneParams>(jsonText);
//             if(sceneParams == null)
//             {
//                 Debug.LogError("Failed to parse JSON.");
//                 yield break;
//             }

//             // ---- 以下、パラメータを使ってシーンを構築する処理 ----

//             // 1. Terrain生成
//             TerrainData terrainData = GenerateTerrain(sceneParams);

//             // 2. サン(Sun)の設定
//             ApplySunSettings(sceneParams);

//             // 3. Prefabの配置
//             PlacePrefabs(sceneParams, terrainData);

//             Debug.Log("Scene setup complete from external API.");
//         }
//     }

//     private TerrainData GenerateTerrain(SceneParams sceneParams)
//     {
//         // 適宜、ランダムでやっていたところを  sceneParams.randomSeed  に差し替え
//         // 例えばPerlinNoiseに sceneParams.randomSeed を混ぜて同じマップになるように
//         // （もしPython側で高さデータそのものを送ってきてもいいが、例ではseedだけ受け取るとする）

//         TerrainData terrainData = new TerrainData();
//         terrainData.heightmapResolution = 513;
//         terrainData.size = new Vector3(50, 100, 70);

//         float[,] heights = new float[terrainData.heightmapResolution, terrainData.heightmapResolution];
//         float randomSeed = sceneParams.randomSeed; 
//         for (int y = 0; y < terrainData.heightmapResolution; y++)
//         {
//             for (int x = 0; x < terrainData.heightmapResolution; x++)
//             {
//                 float perlin = Mathf.PerlinNoise((x + randomSeed) * 0.01f, (y + randomSeed) * 0.01f) * 0.025f;
//                 heights[y, x] = perlin;
//             }
//         }
//         terrainData.SetHeights(0, 0, heights);

//         // TerrainLayerを反映 (Pythonからインデックスだけもらっている想定)
//         // sceneParams.terrainLayers の数だけ割り当てる
//         if(sceneParams.terrainLayers != null && sceneParams.terrainLayers.Length > 0)
//         {
//             TerrainLayer[] chosen = new TerrainLayer[sceneParams.terrainLayers.Length];
//             for(int i=0; i<sceneParams.terrainLayers.Length; i++)
//             {
//                 int index = sceneParams.terrainLayers[i].layerIndex;
//                 if(index >= 0 && index < terrainLayers.Length)
//                 {
//                     chosen[i] = terrainLayers[index];
//                 }
//             }
//             terrainData.terrainLayers = chosen;
//         }
//         else
//         {
//             // デフォルトで全部突っ込むか、警告出すか
//             terrainData.terrainLayers = terrainLayers;
//         }

//         // 既存Terrainを削除する場合、ここで行う

//         // GameObject生成
//         GameObject terrainGO = Terrain.CreateTerrainGameObject(terrainData);
//         terrainGO.name = "ProceduralTerrain100x100";
//         terrainGO.transform.position = new Vector3(-terrainData.size.x*0.5f, 0f, -terrainData.size.z*0.5f);

//         // ※ AlphaMapの塗り分けなども、Pythonからマスクを送ってもらうか、Unity側で生成するかはお好みで

//         return terrainData;
//     }

//     private void ApplySunSettings(SceneParams sceneParams)
//     {
//         if (sunLight == null) 
//         {
//             // シーンを検索してもいいが、事前にInspectorで設定しておくのが楽
//             Debug.LogWarning("Sun Light not assigned.");
//             return;
//         }

//         // 例: AltitudeはX回転、Y回転は方角
//         sunLight.transform.rotation = Quaternion.Euler(sceneParams.sunAltitude, sceneParams.sunRotationY, 0f);

//         if(sceneParams.sunColor != null && sceneParams.sunColor.Length >= 3)
//         {
//             sunLight.color = new Color(sceneParams.sunColor[0], sceneParams.sunColor[1], sceneParams.sunColor[2]);
//         }
//         sunLight.intensity = sceneParams.sunIntensity;
//     }

//     private void PlacePrefabs(SceneParams sceneParams, TerrainData terrainData)
//     {
//         // 今の PlacePrefabsRandomly() の仕組みを改変して、
//         // Pythonから受け取った `prefabs` 配列に従って配置する。
//         // すでにシーンにあるPrefabは削除しておくかどうかは要件次第。

//         if (prefabsWithLimit == null || prefabsWithLimit.Count == 0) return;

//         Vector3 terrainPos = new Vector3(-terrainData.size.x*0.5f, 0f, -terrainData.size.z*0.5f);

//         foreach (var prefabParam in sceneParams.prefabs)
//         {
//             // JSON上の prefabName と、Unityの PrefabSpawnInfo の prefab.name が一致するものを探す
//             PrefabSpawnInfo spawnInfo = prefabsWithLimit.Find(info => info.prefab != null && info.prefab.name == prefabParam.prefabName);
//             if(spawnInfo == null)
//             {
//                 Debug.LogWarning($"PrefabSpawnInfoが見つかりません: {prefabParam.prefabName}");
//                 continue;
//             }

//             // Pythonがくれたインスタンス情報に従い配置
//             foreach (var inst in prefabParam.instances)
//             {
//                 // Terrainの高さサンプリング
//                 float terrainHeight = terrainData.GetHeight(
//                     (int)(inst.posX / terrainData.size.x * terrainData.heightmapResolution),
//                     (int)(inst.posZ / terrainData.size.z * terrainData.heightmapResolution)
//                 );

//                 Vector3 position = new Vector3(inst.posX, terrainHeight, inst.posZ) + terrainPos;
//                 Quaternion rotation = Quaternion.Euler(0f, inst.rotationY, 0f);

//                 // もしOverlap確認とかを続行したい場合は今のロジックを流用
//                 // ↓単純に配置するだけの例
//                 GameObject go = Instantiate(spawnInfo.prefab, position, rotation);
//             }
//         }
//     }
// }
