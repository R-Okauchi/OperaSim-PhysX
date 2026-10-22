# Linux 版 (CAP)

エディタの無いホスト (研究室の GPU の node) で模擬を回すための Linux のプレイヤー。

```
Unity -batchmode -quit -buildTarget Linux64 -projectPath . -executeMethod CapKit.Editor.CapBuild.Linux
```

- `CAP_BUILD_OUT` (既定 `Builds/linux`、git の外)、`CAP_BUILD_SCENES` (既定 SimpleScene)。Unity の「Linux Build Support (Mono)」が
  要る (Unity Hub の `install-modules -m linux-mono`)。
- **UnitySensors (b1a50503bb) のエディタ用の assembly は全部の platform 向け** (`UnitySensorsEditor.asmdef` の `includePlatforms` が空)
  で、その中の 2 つの inspector がプレイヤーの書き出しで失敗する (UnityEditor が無い)。上流は 2025-07 の構成替え (66b5d7db7f) で
  直したが、同時に API が変わり CAP のキットのスクリプトが通らない。なので書き出しの前 (Unity を起動する前) に、パッケージの
  キャッシュの `Library/PackageCache/com.frj.unity-sensors@*/Editor/UnitySensorsEditor.asmdef` を `"includePlatforms": ["Editor"]` に
  する (Unity は "immutable asset(s) were unexpectedly altered" と言って書き出す)。CAP の `scripts/unity_linux_build.sh` がこれと書き出しと
  イメージの作りまでを通す。
- 設定はエディタの通しと同じ `CapCycleRecorder` の JSON (`CAP_UNITY_RECORD_CONFIG`: ROS-TCP の接続先、機体の置き直し、
  `soil_particles`)。シーンのファイルとプロジェクトの設定は変えない。
- 地形の土は計算 shader を使うので、GPU の device が要る: `-nographics` にしない。`-batchmode -force-vulkan` (GPU)、
  画面の無い所は Xvfb と Mesa のソフトの OpenGL。
- 自己試験: 設定に `"soil_selftest": true, "soil_selftest_quit": true` → 起動の 3 s 後に遠くで 20 回掘り 0.3 m3 置いて地形の体積で
  確かめ、ログに `[CapSoilSelfTest] PASS|FAIL ...` (GPU の名前と計算 shader の有無も)、合格 0・不合格 3 で終わる (`CapSoilSelfTest`)。
