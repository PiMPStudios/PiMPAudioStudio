# Model Download Restructure — Design Spec

> Replaces the broken placeholder repo IDs in `ModelRegistry.builtInSpecs` with the real `ACE-Step/Ace-Step1.5` HuggingFace layout, adds Turbo + Full variant support, and adds bearer-token plumbing for gated repos like Stable Audio Open.

**Scope:** Three phases, three commits. AnvilCore + AnvilApp. Zero changes to `MusicPipeline`, `SoundFXPipeline`, `WeightConverter`, or any view outside `ModelsSettingsView`.

---

## Problem Statement

`ModelRegistry.builtInSpecs` declares four model specs with placeholder HuggingFace repo IDs:

| Current Spec ID | Current Repo (placeholder, doesn't exist) |
|---|---|
| `ace-step-dit` | `ACEStep/ACE-Step-v1-5-DiT` |
| `ace-step-vae` | `ACEStep/ACE-Step-v1-5-VAE` |
| `stable-audio-open` | `stabilityai/stable-audio-open-1.0` (real but gated) |

Every download attempt returns HTTP 401. The real ACE-Step models live in **one repo** (`ACE-Step/Ace-Step1.5`) with components in subdirectories. Stable Audio is a real gated repo requiring an `Authorization: Bearer <token>` header.

Verified upstream layout (HF API, 2026-05-07):

**Repo `ACE-Step/Ace-Step1.5`** (bundle — Turbo + VAE + tokenizer):

| Path | Size (bytes) |
|---|---|
| `acestep-v15-turbo/model.safetensors` | 4,787,825,604 |
| `acestep-v15-turbo/config.json` | 1,968 |
| `vae/diffusion_pytorch_model.safetensors` | 337,431,388 |
| `vae/config.json` | 425 |
| `Qwen3-Embedding-0.6B/tokenizer.json` | 11,423,705 |

**Repo `ACE-Step/acestep-v15-base`** (separate repo — Full variant):

| Path | Size (bytes) |
|---|---|
| `model.safetensors` | 4,787,825,604 |
| `config.json` | 1,940 |

**Repo `stabilityai/stable-audio-open-1.0`** (gated):

| Path | Size (bytes) |
|---|---|
| `model.safetensors` | 4,853,889,016 |
| `model_config.json` | 4,172 |
| `vae_model.ckpt` | 624,474,947 |
| `vae_model_config.json` | 3,608 |

---

## Architecture

### `ModelFileSpec` — explicit remote→local mapping

Per-file spec replaces the flat `files: [String]` array on `ModelSpec`. The `localName` keeps consumer code (`ACEStepGenerationService`) unchanged: it still loads `"model.safetensors"`, `"tokenizer.json"`, `"vae.safetensors"`. The `remotePath` carries the subdirectory.

```swift
public struct ModelFileSpec: Sendable, Codable, Hashable {
    public let remotePath: String   // "acestep-v15-turbo/model.safetensors"
    public let localName: String    // "model.safetensors"
    public let sha256: String?
    public let sizeBytes: Int64
}
```

Local layout stays flat: `<baseDirectory>/<spec.id>/<localName>`. Remote subdirectory structure is invisible to consumers.

### `ModelSpec` extensions

- `variantFamily: String?` — groups Turbo + Full under one UI section
- `requiresAuth: Bool` — drives bearer-token attachment and pre-flight gate
- `displayName: String` — human-readable label for Settings UI

### `DownloadClient` — auth-aware

- Switch from `URLSession.download(from: URL)` to `URLSession.download(for: URLRequest)` so headers can be set
- Add `bearerToken: String?` parameter (default nil)
- When non-nil, attach `Authorization: Bearer <token>` to every request
- Map HTTP 401/403 with token present to `DownloadError.authForbidden(repo:)`

### New built-in specs

| ID | Repo | Files (remote → local) | Category | Variant Family | Auth |
|---|---|---|---|---|---|
| `ace-step-dit-turbo` | `ACE-Step/Ace-Step1.5` | `acestep-v15-turbo/model.safetensors`→`model.safetensors`, `acestep-v15-turbo/config.json`→`config.json` | aceStepDiT | `ace-step-1.5` | no |
| `ace-step-dit-base` | `ACE-Step/acestep-v15-base` | `model.safetensors`→`model.safetensors`, `config.json`→`config.json` | aceStepDiT | `ace-step-1.5` | no |
| `ace-step-vae` | `ACE-Step/Ace-Step1.5` | `vae/diffusion_pytorch_model.safetensors`→`vae.safetensors`, `vae/config.json`→`config.json`, `Qwen3-Embedding-0.6B/tokenizer.json`→`tokenizer.json` | aceStepVAE | `ace-step-1.5` | no |
| `stable-audio-open` | `stabilityai/stable-audio-open-1.0` | `model.safetensors`→`model.safetensors`, `model_config.json`→`config.json`, `vae_model.ckpt`→`vae_model.ckpt`, `vae_model_config.json`→`vae_config.json` | stableAudio | — | **yes** |

> **Tokenizer location decision:** Neither DiT variant repo contains a `tokenizer.json` — upstream pulls it from `ACE-Step/Ace-Step1.5/Qwen3-Embedding-0.6B/`. We bundle `tokenizer.json` into the **VAE spec** since the VAE is required by both variants and lives in the bundle repo. `ACEStepGenerationService.loadDiTWeights` is patched in Phase 2 to look up `tokenizer.json` from the active VAE spec's local directory rather than the DiT spec's. This avoids architectural churn (spec stays "one repo per spec") and keeps the tokenizer as a single shared download.

Default active DiT: `ace-step-dit-turbo`. Essential model IDs: `["ace-step-dit-turbo", "ace-step-vae"]`.

---

## Phased Plan

### Phase 1 — Type system + DownloadClient auth plumbing

Pure refactor. No behavioral change. `builtInSpecs` keeps placeholder values until Phase 2.

**Files:**
- `AnvilCore/Sources/AnvilCore/ModelRegistry/ModelSpec.swift` — add `ModelFileSpec`, extend `ModelSpec`
- `AnvilCore/Sources/AnvilCore/ModelRegistry/ModelRegistry.swift` — update download orchestration
- `AnvilCore/Sources/AnvilCore/ModelRegistry/DownloadClient.swift` — add `bearerToken` param, switch to `URLRequest`
- `AnvilCore/Tests/AnvilCoreTests/ModelSpecTests.swift` — update tests
- `AnvilCore/Tests/AnvilCoreTests/DownloadClientTests.swift` — update tests

**Commit:** `refactor(core): ModelSpec gains per-file remote/local mapping + auth metadata`

### Phase 2 — Real ACE-Step v1.5 paths + Turbo/Full variants

**Files:**
- `AnvilCore/Sources/AnvilCore/ModelRegistry/ModelRegistry.swift` — rewrite `builtInSpecs`, update `defaultActiveModels`, `essentialModelIDs`, add migration shim
- `AnvilApp/Services/ACEStepGenerationService.swift` — change VAE filename lookup from `contains("vae")` to deterministic `"vae.safetensors"`
- `AnvilApp/UI/Settings/ModelsSettingsView.swift` — group DiT specs by `variantFamily`, render Turbo/Full as sibling rows
- `AnvilCore/Tests/AnvilCoreTests/ModelRegistryTests.swift` — update spec ID assertions
- `CHANGELOG.md`

**Commit:** `feat(models): real ACE-Step v1.5 repo paths + Turbo/Full variants`

### Phase 3 — HF token UI + bearer-token wiring

**Files:**
- `AnvilApp/Services/APIKeyStore.swift` — add `huggingFace` provider case
- `AnvilApp/UI/Settings/ModelsSettingsView.swift` — token entry section
- `AnvilCore/Sources/AnvilCore/ModelRegistry/ModelRegistry.swift` — token-provider closure injected into init; pre-flight gate when `spec.requiresAuth && token == nil`
- `AnvilApp/AnvilAppApp.swift` — wire token provider to `APIKeyStore` lookup at registry construction
- `CHANGELOG.md`

**Commit:** `feat(downloads): HuggingFace bearer token support for gated models`

---

## Migration

Placeholder repo IDs never resolved — every existing user got HTTP 401 and downloaded nothing. There are no real on-disk weights to migrate, only stale spec IDs in persisted active-model state.

**One-shot migration on first launch with new code:**

1. If persisted active DiT ID is the legacy `"ace-step-dit"`, remap to `"ace-step-dit-turbo"`.
2. Sweep `baseDirectory` for orphan subdirectories whose names don't match any current spec ID; remove them.
3. Gate behind `UserDefaults` flag `com.pimp.AnvilApp.modelMigration.v1` so this runs exactly once and never deletes user-imported model directories.

---

## Multi-Variant UX

- `ModelsSettingsView` groups specs by `spec.variantFamily ?? spec.category.displayName`.
- Within "ACE-Step v1.5" group, Turbo and Full appear as sibling rows. Each shows its own download status, size, and Activate button.
- Only one DiT variant is active at a time within `aceStepDiT` category. Activating the other auto-deactivates the previous via existing `setActiveModel` flow.
- Both variants can be downloaded concurrently (~9 GB total).
- `ACEStepGenerationService` already detects active-model-ID changes and reloads weights on the next `generate()` call. Variant hot-swap is automatic.
- `LoRACoordinator.ensureAdapter` already does its defensive base-model self-check — variant swap auto-clears any injected adapter.

---

## Bearer Token Details

### Storage

- Reuse `APIKeyStore`. Add `huggingFace` to its provider enum.
- Keychain item: `com.pimp.AnvilApp.hfToken`. Local only (no iCloud sync).
- Read accessor: `APIKeyStore.huggingFaceToken() async -> String?` returns `nil` when unset.

### Header attachment

- Build `URLRequest` from `huggingFaceURL(repo:file:)`.
- Set `Authorization: Bearer <token>` only when `bearerToken` is non-nil and non-empty.
- Use `URLSession.download(for: URLRequest)` (not the URL-only overload).

### Pre-flight gate

```swift
if spec.requiresAuth {
    guard let token = await tokenProvider(), !token.isEmpty else {
        throw DownloadError.authRequired(repo: spec.huggingFaceRepo)
    }
    // pass token into DownloadClient
}
```

### Error mapping

```swift
public enum DownloadError: Error, Sendable, LocalizedError {
    case httpError(statusCode: Int)
    case checksumMismatch(file: String)
    case authRequired(repo: String)
    case authForbidden(repo: String)

    public var errorDescription: String? {
        switch self {
        case .authRequired(let repo):
            return "\(repo) is a gated model. Add a HuggingFace access token in Settings → Models, and accept the model's license on huggingface.co."
        case .authForbidden(let repo):
            return "HuggingFace rejected the access token for \(repo). Verify the token is valid and that the model license has been accepted."
        // existing cases retained
        }
    }
}
```

### UI

`ModelsSettingsView` adds a section above the model list:

```
HuggingFace Access Token (optional)
[Secure field..............] [Save]
Required for gated models like Stable Audio Open.
Get a token: https://huggingface.co/settings/tokens
```

---

## Testing & Verification

### Unit (AnvilCore via Xcode test runner)

- `ModelSpec` JSONEncoder/Decoder round-trip
- `ModelSpec.localPath(for: localName, base:)` resolves to `<base>/<id>/<localName>` regardless of `remotePath` shape
- `DownloadClient.huggingFaceURL` builds correct URLs for nested file paths
- `DownloadClient` attaches `Authorization: Bearer …` when token provided, omits when nil
- `DownloadError.authRequired` and `.authForbidden` produce expected `errorDescription`

### Manual smoke (on device)

**After Phase 2:**
- Fresh install → first-launch downloads Turbo DiT + VAE successfully (~5 GB)
- Files land at `<appsupport>/.../ace-step-dit-turbo/model.safetensors`, etc.
- Generate a 10s clip end-to-end
- Settings → Models: switch active DiT to "ACE-Step v1.5 Full" → triggers Full download (~9 GB)
- After Full download, generate → weight swap visible (`.loadingWeights` state), produces audio
- Switch back to Turbo → fast swap (already cached), produces audio
- LoRA + variant swap → adapter clears automatically
- "Release GPU Memory" between swaps → no crash, next generation reloads

**After Phase 3:**
- No token, attempt Stable Audio → pre-flight error: "Stable Audio Open 1.0 is a gated model…"
- Bad token → `authForbidden` error
- Valid token + license accepted → download succeeds
- Token persists across relaunches (Keychain)
- Removing token → next gated download fails with pre-flight error

### Negative paths

- No internet → networking error, not opaque HTTP-0
- Bad SHA256 → retries once, then `checksumMismatch`
- HF 503 → maps to `httpError(503)` with "Server error" description
- Cancel mid-download → clean cancellation, no orphaned tempfile

---

## Rollback

Each phase is independently revertable.

- Reverting Phase 3: removes token UI, non-gated downloads still work.
- Reverting Phase 2: restores Phase 1's placeholder values; type system still works (unit tests pass) but downloads return 401 again.
- Reverting Phase 1: full rollback to current state.

`git revert <commit-sha>` per phase.
