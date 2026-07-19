# PiMP Audio Studio Developer Documentation

These docs are for people building, packaging, extending, or automating PiMP
Audio Studio. End-user instructions live under `docs/user/`.

## Start Here

| Topic | Guide |
| --- | --- |
| System architecture and boundaries | [Architecture](architecture.md) |
| Repo map and module ownership | [Repository Layout](repository-layout.md) |
| Build, test, XcodeGen | [Build And Test](build-and-test.md) |
| Inference pipelines and models | [Inference And Pipelines](inference-and-pipelines.md) |
| Dataset packages and training loops | [Dataset And Training Internals](dataset-and-training-internals.md) |
| Adapters, LoRA/LoKR, merge | [Adapters And LoRA](adapters-and-lora.md) |
| Portable CUDA worker | [Remote Worker](remote-worker.md) |
| Local MCP server | [MCP Server](mcp-server.md) |
| Paths, sandbox, secrets | [Storage, Sandbox, Security](storage-sandbox-security.md) |
| Bundling user docs for Help | [Documentation Integration](documentation-integration.md) |
| Signing, notarization, DMG | [Release And Packaging](release-packaging.md) |

## Related Top-Level Docs

Engineering notes that remain outside the polished developer hub:

| Path | Role |
| --- | --- |
| `docs/mcp-local.md` | Detailed MCP transport and tool reference |
| `docs/training-datasets.md` | Package file layout and enrichment details |
| `docs/dataset-package-v2.md` | Package-v2 contract |
| `docs/remote-worker-contract.md` | Remote artifact/job contracts |
| `docs/remote-sidestep-backend-plan.md` | Planned Side-Step source-job backend |
| `docs/superpowers/` | Historical phase specs/plans |
| `docs/debug/` | Incident analyses |

Do not ship raw investigation notes as end-user Help without editing.

## Architecture Snapshot

```text
PiMPAudioStudio (SwiftUI app)
  UI/ Services/ DSP/ Models/
        |
        v
PiMPAudioCore (Swift package)
  Inference/ Dataset/ LoRA/ ModelRegistry/ LLM/ Community/
        |
        +--> MLX / MLXNN (local Apple Silicon)
        |
remote-worker/ (Python, cloud only)
  Side-Step / ACE-Step / PyTorch / CUDA
```

Hard boundaries:

- The macOS app stays native Swift/SwiftUI/MLX — no Python runtime in the app.
- Remote workers may use Python/CUDA on user-approved cloud hardware.
- Secrets stay in Keychain or user environment variables, never in job zips.
