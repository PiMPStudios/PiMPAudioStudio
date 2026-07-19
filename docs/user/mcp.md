# MCP For Users

PiMP Audio Studio can expose a **localhost-only** MCP (Model Context Protocol)
endpoint for local agents and scripts. The server is **off by default**.

This page is the user-facing overview. Developers should also read
`docs/developer/mcp-server.md` and `docs/mcp-local.md`.

## Enable MCP

1. Open **Settings → MCP**.
2. Turn on **Enable MCP Server**.
3. Copy the endpoint shown in Settings.

Default endpoint:

```text
http://127.0.0.1:38765/mcp
```

Optional bearer token is disabled by default for localhost-only use. If you
enable it, clients must send:

```text
Authorization: Bearer <token>
```

## What You Can Automate

High-level tool groups:

| Area | Examples |
| --- | --- |
| Models & adapters | `list_models`, `list_adapters`, `apply_adapter` |
| Generation | `generate`, `generate_song`, `generation_status` |
| History & traces | `list_history`, `get_trace`, `compare_traces`, `mark_preferred_take` |
| Intelligence | `enhance_prompt`, `write_lyrics` |
| Editor | `load_editor_audio`, `submit_editor_inpaint`, `apply_editor_effect_chain`, `render_editor_audio` |
| Local training | `training_preflight`, `start_training`, `stop_training`, inspect/compare tools |
| RunPod | `runpod_status`, `launch_runpod_pod`, `stop_runpod_pod` |
| Operations | `list_operations`, `get_operation`, `cancel_operation` |
| Cleanup | `release_memory` |

Long-running work returns an `operationID` you can poll. Mutating tools default
to safe modes (`dryRun: true` where applicable) and require explicit confirmation
or MCP form elicitation before billable/destructive actions.

## Safety Model

- Bind to **127.0.0.1** only — not a public server.
- Prefer `dryRun: true` first.
- Billable RunPod launches require confirmation.
- Training start/stop, editor mutations, and memory release are confirmation
  gated.
- Audio resources return **paths/metadata**, not raw PCM streams.
- Sandbox still applies: history/audio IDs are safer than arbitrary absolute
  paths for editor loads.

## Resources And Workflows

MCP also exposes resources such as models, adapters, history, traces, editor
state, operations, presets, and workflow recipes (`improve_chorus`,
`create_mix_variants`, and others). Workflow prompts do not mutate the app by
themselves; they guide a client through tools.

## Smoke Test

From a developer checkout:

```bash
scripts/smoke-pimpaudio-mcp.sh
```

With token mode:

```bash
PIMP_AUDIO_MCP_TOKEN='<token>' scripts/smoke-pimpaudio-mcp.sh
```

## When To Use MCP

- Local coding agents driving generation/training
- Batch dry-runs of training configs
- Automated A/B labs with fixed seeds
- CI-like smoke checks on a developer Mac

When not to use it:

- Untrusted networks
- Shared machines without a token
- Expecting remote multi-user access (not supported)

## Related

- [Local CLI Access](local-cli-access.md)
- [Remote GPU Training](remote-gpu-training.md)
- [Diagnostics And Logs](diagnostics-and-logs.md)
