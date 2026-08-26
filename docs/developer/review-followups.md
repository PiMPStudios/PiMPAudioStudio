# Review Follow-ups

Ranked leftovers from a 2026-08 repo review. Pick from the top. Severity is
security / correctness / release / maintainability — not “lines of code.”

Internal 0.8.0 use on a trusted Mac is still reasonable. Several of these should
land before a wider or paid build.

A `[x]` means the product change landed. It does **not** mean a test file on
disk is compiled or run in CI unless that target is in `project.pbxproj` (see
completed **#22**).

## Completed

- [x] **#1 MCP generate / lyrics / adapters / editor load-export confirmation**
  (security) — `generate`, `generate_song`, `load_editor_audio`, and
  `render_editor_audio` default to dry-run and need `confirm` to mutate.
  `write_lyrics`, `apply_adapter`, `mark_preferred_take`, and
  `cancel_generation` require confirm / elicitation.
- [x] **#2 All-interfaces token + CORS** (security) — Binding `0.0.0.0` forces
  a bearer token that cannot be turned off. CORS echoes a trusted Origin and
  never sends `*`.
- [x] **#3 YouTube URL allowlist** (security) — Dataset import only accepts
  https YouTube hosts and sanitizes yt-dlp video IDs before building watch URLs.
  Tests live in `URLAllowlistTests.swift` (in the Xcode target as of **#22**).
- [x] **#4 Hugging Face adapter import host + streaming** (security) — Import
  requires huggingface.co / hf.co https and streams safetensors to disk via
  `DownloadClient`. Covered by `URLAllowlistTests.swift` (in the Xcode target as
  of **#22**).
- [x] **#6 Keychain write status + accessibility** (security / correctness) —
  Writes check `SecItemUpdate`/`SecItemAdd`, pin
  `WhenUnlockedThisDeviceOnly`, and surface failures in Settings / RunPod.
  Tests live in `KeychainStoreTests.swift` (in the Xcode target as of **#22**).
- [x] **#8 Community adapter catalog URL** (correctness) —
  Catalog now loads from `PiMPStudios/pimp-adapters`. Pre-rebrand identity
  migration (old Application Support folder, Keychain service, MCP scheme, JSON
  field names) was removed in the same pass because no shipped users needed it.
  Commit `aebf72d`. Still need a public `pimp-adapters` repo with `manifest.json`
  or the Community tab stays empty on 404. Catalog URL test is in
  `CommunityServiceTests.swift` (in the Xcode target as of **#22**).
- [x] **#5 Built-in model SHA-256 pins** (security / supply chain) —
  ACE-Step built-in files pin Hugging Face SHA-256 values. Verification still
  **fail-open**: `nil`/empty `sha256` skips the checksum (gated Stable Audio)
  instead of refusing the download.
- [x] **#22 Stale XcodeGen project** (quality / process) — Regenerated
  `project.pbxproj` so `URLAllowlistTests`, `KeychainStoreTests`, and
  `CommunityServiceTests` compile in the app test target. Optional later: CI
  `xcodegen generate && git diff --exit-code`.
- [x] **#23 MCP `cancel_operation` confirmation** (security / policy) —
  Same confirm / elicitation gate as `cancel_generation`. Facade also requires
  `confirm: true`. `set_verbose_diagnostics` still ungated (lower stakes).
- [x] **#24 Operation trim must not cancel live work** (correctness) —
  Evicts oldest terminal records only; pinned / non-terminal ids stay even if
  the cap is exceeded.

## Open — ranked most important first

25. **SFX unconditional `print()` + MLX `.item()` GPU syncs** (perf / hygiene)
    `SoundFXPipeline`, `SFXDiffusionActor`, `SFXDiTActor` print even when
    `AudioDiagnostics` is off. `.item(Float.self)` in those arguments forces
    device→host syncs on the generate path. Finder-launched `.app` has no
    stdout, so the probes also vanish in shipping. Route through
    `AudioDiagnostics`; drop Python-parity scaffolding that always prints.

7. **Sandbox exceptions cover all of Homebrew plus `~/.codex` / `~/.grok` / `~/.gemini`** (release)
   App Store / paid-release blocker if those tree-wide entitlements stay.
   Prefer security-scoped bookmarks for the exact binaries. Not in scope for
   this non-App-Store 0.8.0 cut.

8. ~~Community catalog URL~~ — done.

9. **CI doesn’t run remote-worker Python tests (ruff only) and only compiles Core** (quality)
   Path/zip guards can regress unnoticed. Add
   `python3 -m unittest discover -s remote-worker/tests -v` to the Ubuntu job.

10. **Community / HF adapters are unsigned** (security)
    LoRA deltas into DiT with no manifest signature. Pin hashes in the catalog
    and refuse install on mismatch.

11. **`ytDlpVersion()` blocks the main thread** (UX / correctness)
    Import sheet can freeze on PATH lookup. Make the version probe async.

12. **`ENABLE_TESTABILITY: YES` on Release** (release)
    Extra attack surface in shipped builds. Debug only.

13. **God files** (maintainability)
    `RunPodService` ~3k, enrichment, package writer, MCP catalog/server. This is
    how `generate` missed the confirm list, and how **#23** `cancel_operation`
    missed it again. Eventual shape: one tool declaration with Codable input
    plus `requiresConfirmation`, schema/dispatch derived from that.

14. **Three near-identical `CapturedProcess` implementations** (maintainability)
    Timeout/pipe/cancel bugs will keep duplicating. One shared helper.

15. **MCP HTTP is IPv4-only** (correctness)
    `localhost` → `::1` clients miss the server. Dual-stack or document
    `127.0.0.1` only (the UI already advertises that).

16. **History index keeps unsafe filenames as ghost rows** (correctness)
    UI shows entries that cannot be opened or deleted cleanly. Skip/quarantine
    at load.

17. **MCP `merge_adapters` doesn’t clamp `alpha`** (correctness)
    Garbage blend weights. Clamp to 0…1.

18. **`find_remote_job` takes the first nested `remote_job.json`** (security / correctness)
    Extra JSON in a zip can win. Require the file at package root.

19. **SSH `accept-new` known_hosts should stay per-pod in the container** (security)
    Mostly already true; don’t let it drift to the user file.

20. **Empty `NSHumanReadableCopyright`** (release nit)
    Fill before shipping.

21. **Polish / future Swift 6** (nits)
    `print()` in YouTube builder, UUID MCP token vs `SecRandomCopyBytes`,
    non-constant-time SHA compare, Swift 5.10 vs CI 6.2.

26. **Small MCP / download nits** (correctness) — keep, do not boil:
    RFC 7235 `Authorization` scheme is case-insensitive (`bearer` 401s);
    `Transfer-Encoding: chunked` becomes an empty body (reject 501);
    `preconditionFailure` in `DownloadClient` / `NativeModelPackInstaller` is
    reachable from Add Model (throw instead);
    `send` on EINTR should retry.

## Highest-leverage next picks

If only a few: **#9** (Ubuntu `python3 -m unittest` for remote-worker — cheap
on Linux minutes). **#25** is a local SFX hygiene pass (no CI needed).
**#7** only if App Store is soon.

## Declined or parked (do not re-open from later reviews)

- **`@Observable` rewrite** of `AppState` / `GenerationService` — real SwiftUI
  cost, prior skip as a facade-wide migration
- **PiMPAudioCore tests on GitHub Actions** — `swift test` cannot load MLX
  `default.metallib` on Actions; compile-only Core CI is intentional
- **Swift 6 language mode / `SWIFT_STRICT_CONCURRENCY` as errors** — toolchain
  is already 6.2; 5.10 mode is intentional
- **Tree-wide SwiftFormat or SwiftLint in CI** — parked until Actions minutes
  reset, then a dedicated format pass (not a surprise while quota is tight)

## What’s already in good shape

Do not “fix” these while working the list — they are intentional:

- Path containment for History, adapters, and remote-worker zips, with tests
- Secrets in Keychain; job zips documented as token-free
- Model downloads verify SHA-256 when `ModelFileSpec.sha256` is set, and skip
  verification when it is missing — a missing pin must not block install
- MCP socket details: constant-time token compare, FD generation, 2 MB cap,
  dry-run defaults on billable tools
- Immutable generation submission context; cancel waits; `.releasing` gate
- Process argv, not shell (`ssh` / `scp` / `ditto` / `env yt-dlp`)
- Hardened runtime + app sandbox
