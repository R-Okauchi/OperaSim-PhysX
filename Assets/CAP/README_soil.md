# 粒子なしの土 (CAP)

`SoilParticleSettings.particles = false` で、掘った土を粒子にせず、量として運ぶ。地形の変化はそのまま
(掘った所が下がり、こぼれた所とダンプの後ろが上がる。安息角の崩れも同じ)。シーンのファイルは変えない:
CAP の実行は `CapCycleRecorder` の設定の JSON (`CAP_UNITY_RECORD_CONFIG`) に `"soil_particles": "off"` を書く。

| 部品 | すること |
|---|---|
| `BucketRocks` (バケット) | 地面に触れると 1 回分 (`DigUnitM3`) ずつ地形を下げて量を持つ (容量 zx200 0.8・zx120 0.5 m3)。持てる量は口の向きで決まる: 上から 60° までは満杯、120° で 0。余りは口から真下へ (荷台か地面)。口の向きは殻に光線を投げて測る |
| `CapBedLoad` (荷台、`VesselController` の付いたリンクに Start で付く) | 落ちてきた量を持つ (容量 mst110cr 4.0・ic120 2.8・c30r 0.85 m3)。傾きが 8° までは満杯、45° で 0、余りは一番低い縁から外の地面へ。積んだ量は床の上の箱 (盛り) として見え、LiDAR の光線にも当たる (荷台とは別の kinematic の体、トリガー) |
| `SoilParticleSettings` | `DigUnit`・`Deposit` (地形を下げる・上げる)、`DigUnitM3` (1 回の掘削の量: SimpleScene で 0.0128 m3) |
| `CapSoilGeometry` | 口の向き・荷台の内側の箱・容量 (動くときと調べるときで同じ関数) |

60° と 120° は実機の zx200 の 2026-01-29 の 58 分の作業 (人の運転、130 回の運搬) から: 運ぶ間の口は上から中央値 23°、
放土で開いた所は中央値 121°。CAP の運ぶ姿勢 (boom −1.05・arm 1.25・bucket 2.0 rad) は模擬で 65° (実機の URDF では 55°:
OperaSim の zx200 の腕の基準は実機と約 10° 違う) なので、模擬では持ち上げで 1–7 % こぼれる。

## 確かめ

```
Unity -batchmode -projectPath . -runTests -testPlatform PlayMode -testFilter CapSoil -testResults out.xml
Unity -batchmode -quit -projectPath . -executeMethod CapKit.Editor.CapSoilProbe.Run    # 形の調べ (Temp/cap_soil_probe.json)
```

`CapSoilPlayTests` (SimpleScene、ROS 無し、地形はデータの複製で: 試験で asset を書き換えない):
- 遠くで 20 回掘る → 地形から 0.2554 m3 (見込み 0.2557)、0.3 m3 置く → 0.2996 m3
- 運ぶ姿勢のバケットの下に mst110cr の荷台を置いて開く → 0.62 m3 がそのまま荷台へ、盛りが出る
- 荷台を 60° 傾ける → 全部が後ろの地面へ (地形 +0.622 m3)。平らな荷台に容量 + 0.5 m3 → 0.5 m3 が縁から
- 腕で掘る → 0.64–0.67 m3、地面の減りと一致 (持ち上げでこぼれた分を除く)。地面の上で開く → 差し引き ±0.001 m3
- 粒子あり (シーンのまま) では元どおり粒子が出て、バケットも荷台も量を持たない

荷台の内側の箱は当たり判定を調べた値 (`CapSoilProbe` の下向きの光線で床の高さを確かめる)。c30r は当たり判定の箱が荷台の上を
y 0.93 で塞いでいるので、その上に載せる。
