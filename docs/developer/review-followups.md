# Review Follow-ups

Ranked leftovers from a 2026-08 repo review. Pick from the top. Severity is
security / correctness / release / maintainability — not “lines of code.”

Internal 0.8.0 use on a trusted Mac is still reasonable. Several of these should
land before a wider or paid build.

A `[x]` means the product change landed. It does **not** mean a test file on
disk is compiled or run in CI. After the 2026-08 Qwen review, **#22** tracks
three security test files that were never added to `project.pbxproj`.

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
  Tests live in `URLAllowlistTests.swift` but are **not in the Xcode target**
  until **#22**.
- [x] **#4 Hugging Face adapter import host + streaming** (security) — Import
  requires huggingface.co / hf.co https and streams safetensors to disk via
  `DownloadClient`. Same uncompiled `URLAllowlistTests.swift` as **#3**.
- [x] **#6 Keychain write status + accessibility** (security / correctness) —
  Writes check `SecItemUpdate`/`SecItemAdd`, pin
  `WhenUnlockedThisDeviceOnly`, and surface failures in Settings / RunPod.
  Tests live in `KeychainStoreTests.swift` but are **not in the Xcode target**
  until **#22**.
- [x] **#8 Community adapter catalog URL** (correctness) —
  Catalog now loads from `PiMPStudios/pimp-adapters`. Pre-rebrand identity
  migration (old Application Support folder, Keychain service, MCP scheme, JSON
  field names) was removed in the same pass because no shipped users needed it.
  Commit `aebf72d`. Still need a public `pimp-adapters` repo with `manifest.json`
  or the Community tab stays empty on 404. Catalog URL test is in
  `CommunityServiceTests.swift` — also **not in the Xcode target** until **#22**.
- [x] **#5 Built-in model SHA-256 pins** (security / supply chain) —
  ACE-Step built-in files pin Hugging Face SHA-256 values. Verification still
  **fail-open**: `nil`/empty `sha256` skips the checksum (gated Stable Audio)
  instead of refusing the download.

## Open — ranked most important first

22. **Stale XcodeGen project — three app tests never compile** (quality / process)
    `project.yml` already has `sources: [PiMPAudioStudioTests]`. Committed
    `project.pbxproj` was last generated in `01e4951`. Disk has 10 test files;
    the target Sources phase has 7. Missing: `URLAllowlistTests.swift`,
    `KeychainStoreTests.swift`, `CommunityServiceTests.swift`. CI
    `xcodebuild -only-testing:PiMPAudioStudioTests` goes green without them.
    Fix: `xcodegen generate` and commit the pbxproj. Optional later: CI
    `xcodegen generate && git diff --exit-code -- PiMPAudioStudio.xcodeproj`.
    Do not stop committing the pbxproj without a workflow decision.

23. **MCP `cancel_operation` bypasses the `cancel_generation` confirm gate**
    (security / policy) — `requiresElicitableConfirmation` includes
    `cancel_generation` but not `cancel_operation`. The latter can still call
    `generationService?.cancel()` for the active generation id. Same
    string-list vs dispatch hole as `generate` missing confirm. Gate
    `cancel_operation` (and consider `set_verbose_diagnostics`). Longer term:
    `requiresConfirmation` on the tool declaration plus a test that every
    mutating catalog tool is gated — folds into **#13**.

24. **`trimOperationRecords()` can cancel a live operation** (correctness) —
    New ops `insert(id, at: 0)`; the cap keeps the 50 newest and
    `operationTasks[id]?.cancel()` on the rest with no terminal/active check.
    A live generate is evicted if 50 later `registerOperation` calls pile up
    (generate / editor / download / training / audition — not list/status).
    Evict oldest **terminal** records first; never cancel
    `activeGenerationOperationID` / `activeEditorOperationID`.

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

If only a few: **#22** (`xcodegen generate` so the three tests actually run),
then **#23** (gate `cancel_operation`). **#9** is the cheapest CI add (Ubuntu
Python unittests, not macOS 10× minutes). **#7** only if App Store is soon.

Hold **#25** for an SFX hygiene pass. Do **#24** when touching the operation
registry.

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
