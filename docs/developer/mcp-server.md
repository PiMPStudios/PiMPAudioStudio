# MCP Server

## Overview

PiMP Audio Studio embeds a localhost-only MCP server using the official Swift
MCP SDK for lifecycle, tool/resource dispatch, structured output, and Streamable
HTTP sessions. The app still owns the listener and optional token gate.

Detailed transport reference: `docs/mcp-local.md`.
User overview: `docs/user/mcp.md`.

## Enablement

Settings → MCP. Defaults:

```text
Bind: 127.0.0.1:38765
Endpoint: http://127.0.0.1:38765/mcp
```

Persisted keys (app defaults namespace):

- `mcp.enabled`
- `mcp.bindAddress` (`127.0.0.1` or `0.0.0.0`)
- `mcp.port` (`1024`–`65535`, default `38765`)
- `mcp.tokenRequired` / `mcp.token`

`MCPHTTPServer` accepts `bindHost` + `port`. Binding `0.0.0.0` relaxes Origin
checks for LAN browser clients and auto-enables the bearer token from Settings.
Changing bind/port restarts a running server. Token compare is constant-time.
Responses (including SSE) advertise `Connection: close` because the server
closes the socket after the body.

## Code Map

| File | Role |
| --- | --- |
| `PiMPAudioMCPHTTPServer.swift` | Localhost HTTP + SSE transport |
| `PiMPAudioMCPServer.swift` | Server wiring |
| `PiMPAudioMCPToolCatalog.swift` | Tool schemas |
| `PiMPAudioCommandFacade*.swift` | Implementation surface for tools/resources |

## Transport

- `POST /mcp` — JSON-RPC; may return SSE
- `initialize` returns `MCP-Session-Id`; clients must echo it
- `GET /mcp` — standalone SSE for server-initiated messages / progress
- `DELETE /mcp` — session teardown; transport rebuilds for next client
- Abandoned sessions also rebuild on next `initialize`

Smoke script:

```bash
scripts/smoke-pimpaudio-mcp.sh
```

## Tool Catalog (Current)

### Read / inspect

- `list_models`, `list_adapters`
- `list_history`, `get_trace`, `compare_traces`
- `training_status`, `list_training_datasets`, `list_training_runs`
- `inspect_training_dataset`, `inspect_training_run`, `compare_training_runs`
- `tail_training_log`
- `generation_status`, `editor_status`
- `list_operations`, `get_operation`
- `runpod_status`

### Intelligence (text only)

- `enhance_prompt`
- `write_lyrics`

### Generation

- `apply_adapter`
- `generate`
- `generate_song`
- `mark_preferred_take`

### Editor

- `load_editor_audio`
- `submit_editor_inpaint`
- `apply_editor_effect_chain`
- `render_editor_audio`

Supported DSP effect names: `trimToSelection`, `deleteSelection`, `fadeIn`,
`fadeOut`, `normalize`, `eq`, `reverb`, `timeStretch`, `pitchShift`.

### Training / datasets

- `training_preflight`
- `start_training` (default `dryRun: true`)
- `stop_training`
- `build_training_dataset` (default dry-run; operation when submitted)
- `enrich_training_dataset` (default dry-run; operation when submitted)
- `update_training_dataset_review` (confirm)

### Adapters / models / library

- `import_adapter`, `delete_adapter`, `merge_adapters`
- `download_model` (default dry-run; operation when submitted)
- `set_active_model`
- `delete_history_entries`
- `save_generation_preset`, `delete_generation_preset`
- `set_verbose_diagnostics`, `tail_diagnostics_log`
- `cancel_generation`, `run_adapter_audition`

### Remote GPU lifecycle

- `refresh_runpod_launch_options`
- `launch_runpod_pod` (default dry-run; optional `region` / `dataCenterIDs`)
- `stop_runpod_pod` (DELETE pod, not pause)
- `create_remote_training_job` (default dry-run)
- `upload_remote_training_job` (confirm; uses app-configured SSH key or `sshKeyPath`)
- `start_remote_training_job` (confirm)
- `remote_training_job_status` (read-only monitor)
- `pull_remote_training_job` (confirm; optional adapter import)

### Cleanup

- `release_memory`

Implementation files for the expansion:

- `PiMPAudioCommandFacade+RemoteJobs.swift`
- `PiMPAudioCommandFacade+MCPExtensions.swift`
- `PiMPAudioMCPToolCatalog.swift`
- `PiMPAudioMCPServer.swift`

Legacy `anvil.` / `anvil_` prefixes are not part of the public surface.

## Safety Patterns

1. Prefer dry-run defaults for mutations and billable actions.
2. Require `confirm: true` or MCP form elicitation (`approved: true`).
3. Return `operationID` for long-running work; clients poll
   `get_operation` / `list_operations`.
4. Progress tokens emit request-scoped `notifications/progress` on GET stream.
5. History/audio IDs preferred over absolute paths under sandbox.
6. Default editor renders write under `MCPExports` beside source audio.

## Resources And Prompts

Resources use `pimpaudio://...` URIs for models, adapters, history, takes,
traces, editor, operations, presets, workflows, and client roots.

Workflow prompts (non-mutating recipes):

- `improve_chorus`
- `create_mix_variants`
- `clean_up_region`
- `prepare_stems_for_review`
- `production_ready_prompt`

Completions cover prompt args and resource template ids.

## Extending MCP

1. Add schema entry in `PiMPAudioMCPToolCatalog`.
2. Implement handler in the command façade with dry-run/confirm patterns.
3. Add/extend resources if the data should be readable without a tool call.
4. Update `docs/mcp-local.md`, `docs/user/mcp.md`, smoke script expectations,
   and CHANGELOG.
5. Never return secrets in tool payloads.
