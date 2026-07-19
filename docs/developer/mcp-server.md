# MCP Server

## Overview

PiMP Audio Studio embeds a localhost-only MCP server using the official Swift
MCP SDK for lifecycle, tool/resource dispatch, structured output, and Streamable
HTTP sessions. The app still owns the listener and optional token gate.

Detailed transport reference: `docs/mcp-local.md`.
User overview: `docs/user/mcp.md`.

## Enablement

Settings → MCP. Default endpoint:

```text
http://127.0.0.1:38765/mcp
```

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

### Training / remote

- `training_preflight`
- `start_training` (default `dryRun: true`)
- `stop_training`
- `refresh_runpod_launch_options`
- `launch_runpod_pod` (default dry-run; billable when confirmed)
- `stop_runpod_pod` (DELETE pod, not pause)

### Cleanup

- `release_memory`

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
