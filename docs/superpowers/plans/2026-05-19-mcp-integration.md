# MCP Integration — Implementation Plan

> **Codex planning note (2026-05-19):** This plan records the agreed local MCP
> direction. Implement in small slices. Do not expose remote/LAN access in v1.

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
- Expensive/destructive operations require explicit `confirm: true`.
- No raw tensors or model internals in v1.
- MCP must call app services through a command facade, not SwiftUI views.

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
  - [x] `anvil.list_models`
  - [x] `anvil.list_adapters`
  - [x] `anvil.list_history`
  - [x] `anvil.get_trace`
  - [x] `anvil.compare_traces`
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

## Phase 3 — Controlled Local Actions

**Deliverable:** A local agent can run the same safe workflows the UI can run,
with explicit safeguards.

- [x] `anvil.enhance_prompt`
- [x] `anvil.write_lyrics`
- [x] `anvil.mark_preferred_take`
- [ ] `anvil.generate`
- [x] `anvil.apply_adapter`
- [x] `anvil.release_memory` with required `confirm: true`

Implementation note: `anvil.enhance_prompt` and `anvil.write_lyrics` are
return-only action tools. They reuse the same `PromptEnhancer` and
`LyricWriterManager` objects as the Generate view, but they do not write into
the UI prompt field, lyric field, history, trace sidecars, or generated audio.

Implementation note: `anvil.mark_preferred_take` is the first controlled write
surface. It is intentionally limited to review metadata already supported by
the Generation Analyzer: preferred flag plus optional listening note on an
existing `.trace.json`, mirrored into `history.json`.

Implementation note: `anvil.apply_adapter` changes adapter selection state but
does not load model weights or start generation. It validates installed
ACE-Step-compatible adapters, supports the existing three-adapter stack limit,
persists the primary adapter, updates `GenerationService`, and notifies the
Generate view so open UI state does not overwrite MCP state on the next run.

Implementation note: `anvil.release_memory` is the first guarded cleanup
action. It calls the same consolidated `GenerationService.unloadAllWeights()`
path as Settings, requires `confirm: true`, and returns before/after MLX active,
cache, and peak memory counters so an MCP client can verify cleanup.

## Phase 4 — Deferred Heavy/Destructive Actions

Only start this after read-only and controlled-action tools are reliable.

- [ ] `anvil.train_lora_from_history`
- [ ] `anvil.resume_lora_training`
- [ ] `anvil.set_active_model`
- [ ] `anvil.delete_history_entry`
- [ ] `anvil.delete_model`

Each of these needs stricter confirmation and clearer UI feedback than the
read-only phase.

## Test Plan

- [ ] With MCP disabled, no localhost listener is active.
- [ ] Enabling MCP starts a localhost-only listener.
- [ ] Read-only tools work without a token when token mode is off.
- [ ] When optional token mode is on, clients without the token are rejected.
- [ ] `list_history` and `get_trace` agree with the GUI History/Analyzer.
- [ ] `compare_traces` returns the same changed-field grouping as Analyzer.
- [ ] `enhance_prompt` returns enhanced text without mutating the Generate form.
- [ ] `write_lyrics` returns sanitized duration-aware lyrics without mutating
      the Generate form.
- [ ] `mark_preferred_take` updates trace/history review metadata and preserves
      existing notes when no note argument is provided.
- [x] `apply_adapter` can clear adapter selection and refuses missing adapters.
- [x] `release_memory` refuses to run without `confirm: true`.
- [ ] App quit stops the MCP server cleanly.
