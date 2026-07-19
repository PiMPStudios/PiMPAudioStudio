# Repository Layout

```text
PiMPAudioStudio/
  PiMPAudioStudio.xcodeproj/     Generated/maintained Xcode project
  project.yml                    XcodeGen source of truth for project settings
  PiMPAudioStudio/               macOS app target
  PiMPAudioCore/                 Swift package (inference, dataset, LoRA, registry)
  PiMPAudioStudioTests/          App-level unit tests
  remote-worker/                 Portable CUDA/Side-Step worker
  scripts/                       MCP smoke and package test helpers
  Spikes/                        Isolated experiments (not production)
  Vendor/                        Vendored deps (e.g. demucs-mlx-swift)
  docs/                          User, developer, plans, investigations
  CHANGELOG.md
  README.md
```

## App Target (`PiMPAudioStudio/`)

| Path | Responsibility |
| --- | --- |
| `PiMPAudioStudioApp.swift` | App entry, environment object wiring |
| `AppState.swift` | First-launch downloads, registry, background downloads |
| `UI/MainShellView.swift` | Sidebar shell: Generate/Edit/Datasets/Training/History/Presets |
| `UI/Components/GenerateView.swift` | Generation UX |
| `UI/Editor/` | Editor UI + state |
| `UI/Training/` | Dataset review, training sheet, RunPod panel |
| `UI/Settings/` | Models, Adapters, Intelligence, MCP, community |
| `Services/` | Orchestration, MCP, training, remote, lyrics |
| `DSP/` | Editor DSP implementations |
| `Models/` | App-level generation/history/trace types |
| `PiMPAudioStudio.entitlements` | Sandbox entitlements |

## Core Package (`PiMPAudioCore/Sources/PiMPAudioCore/`)

| Path | Responsibility |
| --- | --- |
| `ModelRegistry/` | Built-in specs, downloads, active model tracking |
| `Inference/` | Music + SFX pipelines, VAE, DiT, diffusion, editor latent tools |
| `Dataset/` | Package v2, validation, native ACE annotation |
| `LoRA/` | LoRA/LoKR configs, injectors, trainers, merge, schedules |
| `LLM/` | Local LLM actor used by intelligence features |
| `Community/` | Community adapter metadata models |
| `Quality/` | Audio/tensor parity metrics |

Tests live in `PiMPAudioCore/Tests/PiMPAudioCoreTests/`.

## Remote Worker (`remote-worker/`)

| Path | Responsibility |
| --- | --- |
| `pimp-remote-worker` | CLI entry |
| `install.sh` | Pod install into managed venv |
| `bootstrap_remote_worker.sh` | Bootstrap helper |
| `tests/` | Path/safety unit tests |
| `README.md` | Operator guide |

## Docs

| Path | Audience |
| --- | --- |
| `docs/user/` | End users; future in-app Help source |
| `docs/developer/` | Engineers packaging/extending the app |
| `docs/superpowers/` | Historical phase plans/specs |
| `docs/debug/` | Incident writeups |
| other `docs/*.md` | Contracts, roadmaps, investigations |

## Scripts

| Script | Purpose |
| --- | --- |
| `scripts/smoke-pimpaudio-mcp.sh` | Local MCP session smoke test |
| `scripts/test-pimpaudiocore-serial.sh` | Serial Core test helper |

## Configuration Anchors

- Bundle ID: `com.pimpstudios.PiMPAudioStudio`
- Marketing version / build: see `project.yml` (`0.8.0` / `8` at time of writing)
- Minimum macOS: 15.0
- Hardened runtime + app sandbox: enabled
