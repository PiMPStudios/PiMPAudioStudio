# MCP For Users

PiMP Audio Studio can expose a **localhost-only** MCP (Model Context Protocol)
endpoint for local agents and scripts. The server is **off by default**.

This page is the user-facing overview. Developers should also read
`docs/developer/mcp-server.md` and `docs/mcp-local.md`.

## Enable MCP

1. Open **Settings → MCP**.
2. Turn on **Enable MCP Server**.
3. Optionally set **Bind address** and **Port**.
4. Copy the endpoint shown in Settings.

Default bind and endpoint:

```text
Bind: 127.0.0.1:38765
http://127.0.0.1:38765/mcp
```

### Bind address

| Setting | Meaning |
| --- | --- |
| Localhost (`127.0.0.1`) | Only this Mac can connect (default) |
| All interfaces (`0.0.0.0`) | Other machines on your LAN can connect |

When you switch to all interfaces, a bearer token is **required** and cannot be
turned off until you bind localhost again. Prefer trusted networks only.
Settings shows a suggested **Network endpoint** using this Mac's LAN IP when
available; remote clients should use that host (or your Mac's real IP), not
literally `0.0.0.0`.

### Port

Default port is **38765**. Allowed range is **1024–65535**. Change the port field
and click **Apply**. Changing bind address or port restarts the server when it
is enabled.

Optional bearer token is disabled by default for localhost-only use. If you
enable it, or bind to all interfaces (which forces the token on), clients must
send:

```text
Authorization: Bearer <token>
```

## What You Can Automate

High-level tool groups:

| Area | Examples |
| --- | --- |
| Models & adapters | `list_models`, `list_adapters`, `apply_adapter`, `import_adapter`, `delete_adapter`, `merge_adapters`, `download_model`, `set_active_model` |
| Generation | `generate`, `generate_song`, `generation_status`, `cancel_generation`, `run_adapter_audition` |
| History & traces | `list_history`, `get_trace`, `compare_traces`, `mark_preferred_take`, `delete_history_entries` |
| Intelligence | `enhance_prompt`, `write_lyrics` |
| Editor | `load_editor_audio`, `submit_editor_inpaint`, `apply_editor_effect_chain`, `render_editor_audio` |
| Datasets | `build_training_dataset`, `enrich_training_dataset`, `update_training_dataset_review`, inspect tools |
| Local training | `training_preflight`, `start_training`, `stop_training`, inspect/compare tools |
| Remote GPU | `runpod_status`, `launch_runpod_pod`, `create_remote_training_job`, `upload_remote_training_job`, `start_remote_training_job`, `remote_training_job_status`, `pull_remote_training_job`, `stop_runpod_pod` |
| Library / diagnostics | `save_generation_preset`, `delete_generation_preset`, `set_verbose_diagnostics`, `tail_diagnostics_log` |
| Operations | `list_operations`, `get_operation`, `cancel_operation` (confirm) |
| Cleanup | `release_memory` |

Long-running work returns an `operationID` you can poll. Mutating tools default
to safe modes (`dryRun: true` where applicable) and require `confirm: true` or
MCP form elicitation before billable/destructive actions. Clients that cannot
prompt must send `confirm: true`. The optional bearer token is stored in the
Keychain. API keys and SSH private keys stay UI/Keychain-managed and are not
written through MCP.

## Safety Model

- Bind to **127.0.0.1** only — not a public server.
- Prefer `dryRun: true` first.
- Generation, editor load/export, lyrics, adapter selection, take marking,
  and cancelling a generation or operation are confirmation-gated. `generate` /
  `generate_song` / `load_editor_audio` / `render_editor_audio` default to
  dry-run.
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
