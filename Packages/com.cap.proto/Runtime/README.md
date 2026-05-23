# CAP Proto Bindings — Runtime

Auto-generated C# protobuf bindings from `cap-spec/proto/cap/v0/*.proto`.

## Why this directory is empty in git

The `*.cs` files here are **build artefacts** of `buf generate`, not
hand-written code. They live in `cap-spec/gen/csharp/` (gitignored)
on the cap repo side. Phase 1e-1 wired the C# protoc plugin into
`cap-spec/buf.gen.yaml`; Phase 1e-2 copies / symlinks the output
into this Unity package.

## Sync step

From the cap repo root:

```bash
cd cap-spec && buf generate
rsync -a --delete cap-spec/gen/csharp/ \
  ../OperaSim-PhysX/Packages/com.cap.proto/Runtime/
```

Re-run after any change to `cap-spec/proto/cap/v0/*.proto`.

## Generated files (Phase 1e-1 verified)

```
Alo.cs                Common.cs             Dialogue.cs
Error.cs              Events.cs             Intent.cs
Lifecycle.cs          MachineAgent.cs       MachineBridgeConfig.cs
MachineSpec.cs        MeanField.cs          Runtime.cs
SiteAgent.cs          Skill.cs              WorldModel.cs
```

`MachineSpec.cs` and `MachineBridgeConfig.cs` are the two
specifically wired into Phase 1 SSOT. The rest become available
for future Unity-side consumers (telemetry serialisation, HAL
contract, etc.).

## NuGet dependency

The generated `.cs` imports `Google.Protobuf` at compile time.
Unity does not ship the NuGet package by default; install it via
[NuGet for Unity](https://github.com/GlitchEnzo/NuGetForUnity)
before this package compiles:

1. Install NuGet for Unity (`Window → Package Manager → + →
   Add package from git URL`):
   `https://github.com/GlitchEnzo/NuGetForUnity.git?path=/src/NuGetForUnity#v4.x`
2. `Window → NuGet → Manage NuGet Packages → Search
   "Google.Protobuf" → Install` (use the Unity-compatible 3.x
   line, not 4.x — Unity 2021/2022 ships .NET Standard 2.1, the
   3.x line is the last that targets it cleanly).
3. After NuGet install, re-run the sync step above; Unity's
   compile pass should pick up both the generated `.cs` and the
   `Google.Protobuf.dll`.

## Phase 1e-2 next-session checklist

- [ ] NuGet for Unity installed
- [ ] `Google.Protobuf` v3.x installed
- [ ] `rsync` from `cap-spec/gen/csharp/` succeeds
- [ ] Unity compiles `Packages/com.cap.proto/Runtime/MachineSpec.cs`
      without errors
- [ ] `Assets/Scripts/CapMachineSpec.cs` ScriptableObject wrapper
      added (proto-derived, surfaces fields to Inspector)
- [ ] One prefab (zx120) migrated to ScriptableObject reference
      as proof-of-concept; rest follow once green
- [ ] `DiffDriveControllerV2` skeleton added (Phase 3 skid physics
      plan); old `DiffDriveController` kept until V2 verified

## Verification

```bash
# from cap repo root, after sync:
ls -la ../OperaSim-PhysX/Packages/com.cap.proto/Runtime/
# expect: MachineSpec.cs + 14 other .cs files
```
