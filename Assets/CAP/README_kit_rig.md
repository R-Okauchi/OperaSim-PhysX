# CAP Perception Kit rig (Unity side)

Single source of truth = cap-pangaea contract (`cap_pangaea/contract/machines/*.json` `sensors[]` + `kit`).

```
cd cap-pangaea
PYTHONPATH=../cooperative-agent-protocol/cap-spec/gen/python python scripts/export_sensor_rig.py \
    --deployment utens-operasim-kit --out ../OperaSim-PhysX/Assets/CAP/sensor_rig.json
```
Then in Unity: `CAP/Perception Kit/Generate Scan Patterns` (once), `CAP/Perception Kit/Apply From Contract`.
`CAP/Perception Kit/Remove From Prefabs` deletes everything the tool created (objects carry `CapKitFrameMarker.managedByCapKit`).

## What the tool changes (permanently, in `Assets/Prefab/<machine>.prefab`)
- `kit_hub` under the hub link (`body_link` for excavators/MST110CR, `base_link` for IC120/C30R):
  `GroundTruthPublisher` → `/<robot>/kit/odom` (nav_msgs/Odometry, frame `map`, child `<robot>/kit_hub`, 50 Hz) —
  temporary stand-in for the real kit localization; `QuaternionStampedPublisher` → `/<robot>/kit/gnss/heading` (5 Hz);
  `kit_gnss_a/b` at ±baseline/2 along forward with `GNSSSensor` + `NavSatFixMsgPublisher` (5 Hz).
- `kit_<sensor_id>` under `parent_link` for every declared frame (roof, fl, fr, rl, rr). Only `populated: true`
  sensors get a device: UnitySensors `Mid-360` prefab (inverted by the contract rpy [180,0,0]) or a `RaycastLiDARSensor`
  with a generated `ScanPattern` (`Assets/CAP/ScanPatterns/*.asset`), plus `KitRaycastLiDARPointCloud2MsgPublisher`
  (x,y,z,intensity float32, 16 B/point, ONE timestamp per cloud — no fabricated per-point time) and, if declared,
  `IMUSensor` + `IMUMsgPublisher` on the same rigid mount.

## Frame convention (from sensor_rig.json `frame_convention`)
ROS (x fwd, y left, z up; rpy deg) → Unity local position `(-y, z, x)`, euler `(pitch, -yaw, -roll)`.
Pod tilt sign must be verified in P0 (the JSON keeps the ROS values so the tool can be corrected without touching cap).

## Manual steps
- Assign the scene `GeoCoordinateSystem` to each `kit_gnss_*` `GNSSSensor` (scene object, cannot live in a prefab).
- UnitySensors serialized fields touched: `LiDARSensor._scanPattern/_pointsNumPerScan/_minRange/_maxRange/_gaussianNoiseSigma`,
  `UnitySensor._frequency`, `RosMsgPublisher._frequency/_topicName/_serializer._header._frame_id`. Re-check after a package upgrade.
- Known limits: UnitySensors IMU has no noise model and publishes world-frame acceleration; GNSS has zero covariance;
  the raycaster uses one transform per scan. The cap-side `kit_sensor_shim` corrects/injects these — the simulator does not.
