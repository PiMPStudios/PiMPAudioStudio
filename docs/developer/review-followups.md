# Review Follow-ups

Ranked leftovers from a 2026-08 repo review. Pick from the top. Severity is
security / correctness / release / maintainability — not “lines of code.”

Internal 0.8.0 use on a trusted Mac is still reasonable. Ranked leftovers are
done or parked.

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
  Commit `aebf72d`. PiMP will not publish adapters; the Community tab stays
  empty on 404 unless someone else hosts that repo. Hugging Face import is
  the supported download path. Catalog URL test is in
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
- [x] **#25 SFX diagnostics** (perf / hygiene) — Unconditional `print()` /
  `.item()` probes now go through `AudioDiagnostics` (AUDIODIAG files when
  verbose is on). DiT first-two-call probes also honor that flag.
- [x] **#9 Remote-worker Python tests in CI** (quality) — Ubuntu job runs
  `python3 -m unittest discover -s remote-worker/tests -v` after ruff. Core
  tests on Actions stay compile-only (metallib).
- [x] **#26 Small MCP / download nits** (correctness) — Bearer scheme is
  case-insensitive; chunked transfer encoding → 501; `send` retries `EINTR`;
  Hugging Face URL builders throw instead of `preconditionFailure`.
- [x] **#17 MCP `merge_adapters` clamps `alpha`** (correctness) — Blend weight
  is clamped to 0…1 (non-finite → 0.5) in `AdapterRegistry` and the MCP
  facade.
- [x] **#11 `ytDlpVersion()` is async** (UX) — Import sheet probes yt-dlp off
  the main thread via `.task`.
- [x] **#12 `ENABLE_TESTABILITY` is Debug-only** (release) — Release/shipped
  builds no longer enable testability.
- [x] **#16 History skips unsafe filenames at load** (correctness) — Ghost rows
  with traversal names are dropped from the index instead of staying in the UI.
- [x] **#20 `NSHumanReadableCopyright`** (release nit) — Set to
  `© 2026 PiMP Studios`.
- [x] **#15 MCP HTTP is IPv4-only** (correctness) — Documented: bind
  `127.0.0.1`, not `localhost`/`::1`. Dual-stack / IPv6 parked.
- [x] **#18 `remote_job.json` must sit at the package/zip root** (security) —
  No more `rglob` first-match. Allowed: file at extract root, or one wrapper
  folder that directly contains it. Deeper or competing copies fail.
- [x] **#19 Per-pod SSH `known_hosts`** (security) — Launch already uses
  `UserKnownHostsFile=known_hosts-<pod-id>` next to the key, not
  `~/.ssh/known_hosts`. Test asserts that.

## Open — ranked most important first

8. ~~Community catalog URL~~ — done.

9. ~~Remote-worker Python tests in CI~~ — done.

10. ~~Community / HF adapters unsigned~~ — parked: no PiMP catalog; Hub
    import is user-initiated. Optional SHA-256 still fail-open.

11. ~~`ytDlpVersion()` on the main thread~~ — done.

12. ~~`ENABLE_TESTABILITY` on Release~~ — done.

13. ~~God files / typed MCP catalog~~ — parked: confirm-list misses already
    fixed (**#1**, **#23**). No split or catalog rewrite without a new bug.

14. ~~Three `CapturedProcess` copies~~ — parked: wrappers have diverged
    (timeout, stdin, SIGKILL). Do not unify unless a hang shows up.

15. ~~MCP HTTP IPv4-only~~ — documented: use `127.0.0.1`, not `::1`. Dual-stack
    / IPv6 parked.

16. ~~History unsafe filenames as ghost rows~~ — done.

17. ~~`merge_adapters` alpha clamp~~ — done.

18. ~~`find_remote_job` nested `remote_job.json`~~ — done.

19. ~~SSH per-pod `known_hosts`~~ — done.

20. ~~Empty `NSHumanReadableCopyright`~~ — done.

21. ~~Polish / future Swift 6 nits~~ — parked: YouTube `print()`, UUID MCP
    token, SHA string compare, and 5.10 language mode are not user-facing
    bugs.

## Highest-leverage next picks

Ranked list is empty. Do not reopen parked rewrites from later reviews.

## Declined or parked (do not re-open from later reviews)

- **`@Observable` rewrite** of `AppState` / `GenerationService` — real SwiftUI
  cost, prior skip as a facade-wide migration
- **PiMPAudioCore tests on GitHub Actions** — `swift test` cannot load MLX
  `default.metallib` on Actions; compile-only Core CI is intentional
- **Swift 6 language mode / `SWIFT_STRICT_CONCURRENCY` as errors** — toolchain
  is already 6.2; 5.10 mode is intentional
- **SwiftLint in CI** — not enabled. SwiftFormat 0.62.1 is the format gate;
  do not add SwiftLint in the same pass.
- **#7 Homebrew / `~/.codex` / `~/.grok` / `~/.gemini` sandbox exceptions** —
  App Store entitlement cleanup. This product is not going to the App Store;
  keep the exceptions for local CLI tools.
- **#10 Unsigned community / HF adapters** — PiMP will not publish a signed
  catalog. Users train locally or import a Hub URL they chose. Optional
  SHA-256 on downloads stays fail-open (same as built-in models). Do not
  refuse install for a missing hash.
- **MCP dual-stack / IPv6** — listener stays IPv4 (`127.0.0.1` / `0.0.0.0`).
  Clients must not use `localhost` if that resolves to `::1`.
- **#14 Shared `CapturedProcess`** — YouTube, RunPod, and provider CLI
  wrappers stay separate. They share a pipe/cancel shape but differ on
  timeout, stdin, and SIGKILL. Unify only if a real hang or timeout bug
  shows up.
- **#13 God files / typed MCP catalog** — `generate` and `cancel_operation`
  confirm misses are fixed. File size and a single-declaration catalog are
  not current bugs. Do not split RunPod/enrichment/MCP unless a new miss
  shows up.
- **#21 Polish / Swift 6 nits** — YouTube skip `print()`, UUID MCP token,
  non-constant-time SHA hex compare, and Swift 5.10 language mode stay as
  they are. Token compare is already constant-time; SHA is a file checksum
  not a secret.

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
