# MCP Integration — Implementation Plan

> **Codex planning note (2026-05-19):** This plan records the agreed local MCP
> direction. Implement in small slices. Do not expose remote/LAN access in v1.
>
> **Standards update (2026-05-20):** Future MCP work should bias toward the
> official protocol shape and the official Swift SDK where practical. Keep
> `AnvilCommandFacade` as the app boundary and move transport/lifecycle/schema
> concerns out of hand-rolled code over time. See
> `docs/mcp-standards-roadmap.md`.

## Goal

Add an optional localhost-only MCP server for Anvil so local agents and scripts
can inspect models, adapters, history, and generation traces, then later trigger
safe app actions like prompt enhancement, lyric writing, generation, preferred
take marking, and memory release.

## Guardrails

- MCP is off by default.
- v1 binds only to `127.0.0.1` / `localhost`.
- Localhost token/pairing is optional.
- Token becomes mandatory only for future LAN/remote/tunnel/headless modes.
- Expensive/destructive operations require explicit confirmation, either
  `confirm: true` or MCP form elicitation approval when the client advertises
  support.
- No raw tensors or model internals in v1.
- MCP must call app services through a command facade, not SwiftUI views.
- Public tool names are plain snake_case. Legacy `anvil.` and `anvil_` aliases
  can remain callable, but docs and future tools should use snake_case first.

## Phase 1 — Read-Only Local MCP

**Deliverable:** A running localhost MCP endpoint that exposes low-risk read
tools and can be toggled from Settings.

- [x] Create an `AnvilMCPServer` service with start/stop/status lifecycle.
- [x] Add a small `AnvilCommandFacade` for MCP-safe reads.
- [x] Add Settings controls:
  - [x] Enable MCP server toggle.
  - [x] Running/stopped/failed state.
  - [x] Local endpoint display.
  - [x] Optional token enable/copy/regenerate controls.
- [x] Implement read-only tools:
  - [x] `list_models`
  - [x] `list_adapters`
  - [x] `list_history`
  - [x] `get_trace`
  - [x] `compare_traces`
- [x] Ensure every response is DTO/JSON oriented and stable across app changes.
- [x] Add docs for connecting a local MCP client.
- [x] Add a `scripts/smoke-mcp.sh` localhost smoke test for health,
      initialize, tool discovery, history, trace reads, resources, and compare.

Implementation note: Phase 1 now ships as an app-hosted Streamable HTTP-style
JSON-RPC endpoint at `http://127.0.0.1:38765/mcp`. It is off by default,
configured from Settings > MCP, and uses the shared Generation Analyzer
comparison grouping through `GenerationTraceComparator`.

## Phase 2 — Structured Resources

**Deliverable:** MCP clients can browse Anvil artifacts without guessing file
paths.

- [x] Add `anvil://models`.
- [x] Add `anvil://model/{id}`.
- [x] Add `anvil://adapters`.
- [x] Add `anvil://adapter/{id}`.
- [x] Add `anvil://history`.
- [x] Add `anvil://history/{id}`.
- [x] Add `anvil://trace/{id}`.
- [x] Add `anvil://audio/{id}` metadata with local file path, not streamed audio
      by default.
- [x] Add `anvil://project/current`.
- [x] Add `anvil://track/current`.
- [x] Add `anvil://clip/current`.
- [x] Add `anvil://region/current-selection`.
- [x] Add `anvil://takes`.
- [x] Add `anvil://take/{id}`.
- [x] Add `anvil://presets`.
- [x] Add `anvil://preset/{id}`.
- [x] Add `anvil://workflows`.
- [x] Add `anvil://workflow/{id}`.
- [x] Add `resources/templates/list` for parameterized model, adapter,
      history, take, trace, audio, preset, workflow, and operation resources.

## Phase 3 — Controlled Local Actions

**Deliverable:** A local agent can run the same safe workflows the UI can run,
with explicit safeguards.

- [x] `enhance_prompt`
- [x] `write_lyrics`
- [x] `mark_preferred_take`
- [x] `apply_adapter`
- [x] `generate_song`
- [x] `generate`
- [x] `generation_status`
- [x] `editor_status`
- [x] `load_editor_audio`
- [x] `submit_editor_inpaint`
- [x] `apply_editor_effect_chain`
- [x] `render_editor_audio`
- [x] `list_operations`
- [x] `get_operation`
- [x] `cancel_operation`
- [x] `release_memory` with required confirmation

Implementation note: `enhance_prompt` and `write_lyrics` are
return-only action tools. They reuse the same `PromptEnhancer` and
`LyricWriterManager` objects as the Generate view, but they do not write into
the UI prompt field, lyric field, history, trace sidecars, or generated audio.

Implementation note: `mark_preferred_take` is the first controlled write
surface. It is intentionally limited to review metadata already supported by
the Generation Analyzer: preferred flag plus optional listening note on an
existing `.trace.json`, mirrored into `history.json`.

Implementation note: `apply_adapter` changes adapter selection state but
does not load model weights or start generation. It validates installed
ACE-Step-compatible adapters, supports the existing three-adapter stack limit,
persists the primary adapter, updates `GenerationService`, and notifies the
Generate view so open UI state does not overwrite MCP state on the next run.

Implementation note: `generate_song` composes the existing return-only prompt
and lyric helpers, optional adapter selection, and generation submission into a
single agent-friendly orchestration tool. `dryRun: true` validates the resolved
request without mutating app state or applying adapters.

Implementation note: `generate` submits work through the same
`GenerationService.generate(request:)` path as the Generate button, but returns
immediately instead of blocking on a full render. `generation_status`
returns the current state and latest completed result; `dryRun: true` validates
and echoes the resolved request without starting generation.

Implementation note: MCP-submitted generation now registers a pollable operation
record with `operationID`, status, progress, result/error, and cancellation
metadata. `cancel_operation` currently cancels active generation operations
through `GenerationService.cancel()`. This gives agents a stable operation
contract before the transport grows protocol-level progress notifications.
Operation progress is now normalized across the full generation lifecycle:
`progress` / `total` describe whole-operation completion, while
`stageProgress` / `stageTotal` carry stage-local counters when available.

Implementation note: `editor_status` exposes the shared editor coordinator
readiness, loaded audio metadata, supported editor action descriptors, and any
active editor operation. `load_editor_audio` validates a history/audio id or
readable local path, registers `editor.loadAudio`, loads the file into the
shared editor coordinator, and records loaded-audio metadata when latent
encoding completes. `submit_editor_inpaint` extends the same operation shape to
confirmed selected-region editor mutations, with `dryRun` validation and
`editor.inpaint` polling/cancellation. `apply_editor_effect_chain` applies
confirmed DSP chains through the existing editor DSP processor, re-encodes
editable latents, and reports as `editor.applyEffectChain`. `render_editor_audio`
bounces the current editor buffer or a selected time range to WAV/AIFF using
`editor.renderAudio`.

Implementation note: `release_memory` is the first guarded cleanup
action. It calls the same consolidated `GenerationService.unloadAllWeights()`
path as Settings, requires `confirm: true` or MCP form elicitation approval,
and returns before/after MLX active, cache, and peak memory counters so an MCP
client can verify cleanup.

Implementation note: guarded MCP mutations now use the SDK-backed
`elicitation/create` form flow when the initialized client advertises form
elicitation support and omits `confirm: true`. Decline/cancel responses return
a no-op structured result. Clients without elicitation support keep the
compatibility path and must call again with `confirm: true` after explicit user
approval.

Implementation note: common editing/generation workflows are now exposed through
MCP prompts instead of broad generic tools. Initial prompt names are
`improve_chorus`, `create_mix_variants`, `clean_up_region`,
`prepare_stems_for_review`, and `production_ready_prompt`; matching
`anvil://workflow/{id}` resources document their arguments, related resources,
and tool dependencies.

Implementation note: MCP completions now cover workflow prompt arguments and
resource-template ids. `completion/complete` suggests take/history ids, current
region times, common editor issue/direction values, preset/workflow ids, and
recent operation ids from the existing local metadata surface.

Implementation note: MCP client roots are now tracked through the Swift SDK
roots primitive when the initialized client advertises roots. The bridge exposes
`anvil://client-roots` with the current root list and path-access policy, and
path completions can suggest root paths while still preferring history/audio ids
for editor loads.

## Phase 4 — Standards Hardening

**Deliverable:** Decide whether the official Swift MCP SDK can replace the
custom protocol shell, or document the exact spec gaps that remain in the
custom server.

- [x] Repair stale package tests so MCP changes have a clean baseline via
      `scripts/test-anvilcore-serial.sh`.
- [x] Update user-facing MCP docs to primary snake_case names.
- [x] Inventory Swift 6 / strict-concurrency warnings before changing language
      mode.
- [x] Spike `modelcontextprotocol/swift-sdk` in an isolated branch or target.
- [x] Re-expose a tiny read-only set through the SDK:
  - [x] `list_history`
  - [x] `resources/list`
  - [x] one `resources/read` path
- [x] Compare SDK output against the `scripts/smoke-mcp.sh` method shape and an
      SDK client. Full app smoke remains part of the replacement slice.
- [x] Decide:
  - [x] Replace `AnvilMCPServer` protocol/transport internals with SDK
        primitives, now using `StatefulHTTPServerTransport`, or
  - [ ] Keep the custom server temporarily and track explicit spec divergences
        in `docs/mcp-standards-roadmap.md`.

Spike note: `Spikes/MCPServerSDKSpike` confirms the SDK can own tool/resource
schemas, lifecycle, structured tool output, and HTTP transport semantics while
Anvil keeps `AnvilCommandFacade` and the localhost listener.

Implementation note: `AnvilMCPServer` now uses the official Swift MCP SDK for
tool/resource registration, lifecycle handling, structured tool results,
JSON-RPC dispatch, and stateful Streamable HTTP handling. The app keeps its
existing localhost listener, Settings controls, optional token gate, and
`AnvilCommandFacade` service boundary. The smoke script now initializes a
session, parses POST SSE responses, verifies `GET /mcp`, and checks a
request-scoped `notifications/progress` event.

## Phase 5 — Deferred Heavy/Destructive Actions

Only start this after read-only and controlled-action tools are reliable.

- [ ] `train_lora_from_history`
- [ ] `resume_lora_training`
- [ ] `set_active_model`
- [ ] `delete_history_entry`
- [ ] `delete_model`

Each of these needs stricter confirmation and clearer UI feedback than the
read-only phase.

## Test Plan

- [ ] With MCP disabled, no localhost listener is active.
- [x] Enabling MCP starts a localhost-only listener.
- [x] Read-only tools work without a token when token mode is off.
- [ ] When optional token mode is on, clients without the token are rejected.
- [x] `list_history` and `get_trace` agree with the GUI History/Analyzer.
- [x] `compare_traces` returns the same changed-field grouping as Analyzer.
- [ ] `enhance_prompt` returns enhanced text without mutating the Generate form.
- [ ] `write_lyrics` returns sanitized duration-aware lyrics without mutating
      the Generate form.
- [ ] `mark_preferred_take` updates trace/history review metadata and preserves
      existing notes when no note argument is provided.
- [x] `apply_adapter` can clear adapter selection and refuses missing adapters.
- [x] `generate` supports dry-run request validation without starting audio.
- [ ] `generate` can submit a real request and `generation_status` follows it
      to completion.
- [x] `list_operations` exposes MCP operation state and `cancel_operation`
      refuses non-running operation ids without cancelling unrelated work.
- [x] `GET /mcp` opens a stateful SSE stream and receives request-scoped
      `notifications/progress` when `_meta.progressToken` is supplied.
- [x] `prompts/list` and `prompts/get` expose the workflow prompt catalog.
- [x] Pollable operation payloads expose normalized whole-operation progress.
- [x] `editor_status`, `load_editor_audio`, `submit_editor_inpaint`, and
      `apply_editor_effect_chain`, and `render_editor_audio` expose editor
      operations with confirmation on destructive edit paths.
- [x] `release_memory` refuses to run without `confirm: true` or MCP form
      elicitation approval.
- [x] `resources/templates/list` and `completion/complete` are SDK-backed and
      covered by the localhost smoke script.
- [x] `anvil://client-roots` exposes client-advertised roots and the local path
      policy.
- [x] App quit stops the MCP server cleanly.
