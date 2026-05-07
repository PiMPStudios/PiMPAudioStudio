# Model Download Restructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace broken placeholder repo IDs in `ModelRegistry.builtInSpecs` with real `ACE-Step/Ace-Step1.5` HuggingFace layout, add Turbo + Full DiT variants, and add bearer-token plumbing for gated repos like Stable Audio Open.

**Architecture:** Three phases — type-system change first (no behavior delta), then real repo paths + variants, then HF token UI/wiring. Each phase is a single commit and independently revertable. `ModelFileSpec` introduces explicit remote→local path mapping so consumer code (`ACEStepGenerationService`) keeps loading flat filenames (`model.safetensors`, `tokenizer.json`, `vae.safetensors`) while the registry handles HF subdirectory paths.

**Tech Stack:** Swift, AnvilCore (SPM lib), AnvilApp (Xcode app), URLSession, Keychain via existing `APIKeyStore`. Verified via Xcode build — no SPM test runner because Metal shaders don't compile in `swift test`.

---

## Reference: Verified HF Layout

Probed via `https://huggingface.co/api/models/<repo>/tree/main/<subdir>` on 2026-05-07.

**Repo `ACE-Step/Ace-Step1.5`** (Turbo + VAE + tokenizer):

| Path | Size (bytes) |
|---|---|
| `acestep-v15-turbo/model.safetensors` | 4,787,825,604 |
| `acestep-v15-turbo/config.json` | 1,968 |
| `vae/diffusion_pytorch_model.safetensors` | 337,431,388 |
| `vae/config.json` | 425 |
| `Qwen3-Embedding-0.6B/tokenizer.json` | 11,423,705 |

**Repo `ACE-Step/acestep-v15-base`** (Full DiT, separate repo):

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

# Phase 1 — Type System + DownloadClient Auth Plumbing

Pure refactor. No behavioral change. `builtInSpecs` keeps placeholders until Phase 2.

## Task 1.1: Add `ModelFileSpec` and extend `ModelSpec`

**Files:**
- Modify: `AnvilCore/Sources/AnvilCore/ModelRegistry/ModelSpec.swift`

- [ ] **Step 1: Replace the entire `ModelSpec.swift` file with the new shape**

```swift
import Foundation

public enum ModelCategory: String, Codable, Sendable, Hashable {
    case aceStepDiT
    case aceStepVAE
    case stableAudio
    case lyricLLM
}

public struct ModelFileSpec: Sendable, Codable, Hashable {
    /// Path inside the HuggingFace repo, e.g. "acestep-v15-turbo/model.safetensors".
    public let remotePath: String
    /// Flat filename used on disk and by consumer code, e.g. "model.safetensors".
    /// Defaults to the basename of `remotePath` when not specified.
    public let localName: String
    public let sha256: String?
    public let sizeBytes: Int64

    public init(
        remotePath: String,
        localName: String? = nil,
        sha256: String? = nil,
        sizeBytes: Int64 = 0
    ) {
        self.remotePath = remotePath
        self.localName = localName ?? (remotePath as NSString).lastPathComponent
        self.sha256 = sha256
        self.sizeBytes = sizeBytes
    }
}

public struct ModelSpec: Sendable, Identifiable, Codable {
    public let id: String
    public let displayName: String
    public let huggingFaceRepo: String
    public let files: [ModelFileSpec]
    public let category: ModelCategory
    /// Groups variants of the same model family for UI presentation, e.g. "ace-step-1.5".
    public let variantFamily: String?
    /// When true, downloads attach `Authorization: Bearer <token>` and gate on token presence.
    public let requiresAuth: Bool

    public init(
        id: String,
        displayName: String,
        huggingFaceRepo: String,
        files: [ModelFileSpec],
        category: ModelCategory,
        variantFamily: String? = nil,
        requiresAuth: Bool = false
    ) {
        self.id = id
        self.displayName = displayName
        self.huggingFaceRepo = huggingFaceRepo
        self.files = files
        self.category = category
        self.variantFamily = variantFamily
        self.requiresAuth = requiresAuth
    }

    public var totalSizeBytes: Int64 {
        files.reduce(0) { $0 + $1.sizeBytes }
    }

    public var totalSizeFormatted: String {
        let gb = Double(totalSizeBytes) / 1_000_000_000
        return String(format: "%.1f GB", gb)
    }

    public func localDirectory(base: URL) -> URL {
        base.appending(path: id)
    }

    /// Resolves to `<base>/<id>/<localName>`. Code that loads weights uses the
    /// flat localName ("model.safetensors") and is unaware of the remote path.
    public func localPath(for localName: String, base: URL) -> URL {
        localDirectory(base: base).appending(path: localName)
    }
}
```

- [ ] **Step 2: Build AnvilCore**

Run: `cd ~/Developer/GitRepos/AnvilApp && xcodebuild -scheme AnvilApp -destination 'platform=macOS' build 2>&1 | grep -E "error:|BUILD"`

Expected: many compile errors in `ModelRegistry.swift`, test files, and `ACEStepGenerationService.swift` because the old `files: [String]` API is gone, `displayName` is now stored not computed, and the init signature changed. Those are fixed in subsequent tasks. Don't try to fix them yet — proceed.

## Task 1.2: Update `ModelRegistry` download orchestration to use `ModelFileSpec`

**Files:**
- Modify: `AnvilCore/Sources/AnvilCore/ModelRegistry/ModelRegistry.swift` (download loop and status check only — leave `builtInSpecs` placeholders alone for this phase)

- [ ] **Step 1: Update `status(for:)` to iterate `ModelFileSpec.localName`**

Find the block (around line 109):

```swift
let allFilesExist = spec.files.allSatisfy { file in
    FileManager.default.fileExists(
        atPath: spec.localPath(for: file, base: baseDirectory).path()
    )
}
```

Replace with:

```swift
let allFilesExist = spec.files.allSatisfy { fileSpec in
    FileManager.default.fileExists(
        atPath: spec.localPath(for: fileSpec.localName, base: baseDirectory).path()
    )
}
```

- [ ] **Step 2: Update `downloadModels(_:onProgress:)` to iterate `ModelFileSpec`**

Find the block (around line 132):

```swift
for file in spec.files {
    let destination = spec.localPath(for: file, base: baseDirectory)
    let expectedHash = spec.sha256[file]

    do {
        try await client.downloadAndVerify(
            repo: spec.huggingFaceRepo,
            file: file,
            to: destination,
            expectedHash: expectedHash,
            onProgress: { [id = spec.id] progress in
                Task { [weak self] in
                    await self?.setStatus(id, to: .downloading(progress: progress))
                }
                onProgress(id, progress)
            },
            onRetry: { [id = spec.id] in
                logger.warning("Verification failed for \(file) in \(id), retrying...")
            }
        )
    } catch {
        // ... existing catch
    }
}
```

Replace with:

```swift
for fileSpec in spec.files {
    let destination = spec.localPath(for: fileSpec.localName, base: baseDirectory)

    do {
        try await client.downloadAndVerify(
            repo: spec.huggingFaceRepo,
            file: fileSpec.remotePath,
            to: destination,
            expectedHash: fileSpec.sha256,
            bearerToken: nil,
            onProgress: { [id = spec.id] progress in
                Task { [weak self] in
                    await self?.setStatus(id, to: .downloading(progress: progress))
                }
                onProgress(id, progress)
            },
            onRetry: { [id = spec.id, remote = fileSpec.remotePath] in
                logger.warning("Verification failed for \(remote) in \(id), retrying...")
            }
        )
    } catch {
        let message: String
        if case DownloadError.checksumMismatch = error {
            message = "Model verification failed. The file may be corrupted or tampered with."
        } else {
            message = error.localizedDescription
        }
        statuses[spec.id] = .failed(message)
        throw error
    }
}
```

> Note: `bearerToken: nil` is a placeholder — Phase 3 wires it to a token provider. Phase 1 keeps the new parameter slot present so the DownloadClient signature is stable.

- [ ] **Step 3: Build**

Expected: `builtInSpecs` and tests still fail (handled in next tasks). The download path itself should compile.

## Task 1.3: Update `DownloadClient` to accept bearer tokens

**Files:**
- Modify: `AnvilCore/Sources/AnvilCore/ModelRegistry/DownloadClient.swift`

- [ ] **Step 1: Add `bearerToken` parameter to `downloadAndVerify` and internal helpers**

Replace the existing `downloadAndVerify` signature:

```swift
public func downloadAndVerify(
    repo: String,
    file: String,
    to destination: URL,
    expectedHash: String?,
    onProgress: @Sendable @escaping (Double) -> Void,
    onRetry: @Sendable @escaping () -> Void
) async throws {
    try await downloadFile(repo: repo, file: file, to: destination, onProgress: onProgress)
    // ...
}
```

With:

```swift
public func downloadAndVerify(
    repo: String,
    file: String,
    to destination: URL,
    expectedHash: String?,
    bearerToken: String? = nil,
    onProgress: @Sendable @escaping (Double) -> Void,
    onRetry: @Sendable @escaping () -> Void
) async throws {
    try await downloadFile(
        repo: repo,
        file: file,
        to: destination,
        bearerToken: bearerToken,
        onProgress: onProgress
    )

    guard let hash = expectedHash, !hash.isEmpty else { return }

    let verified = await verifySHA256(file: destination, expected: hash)
    if verified { return }

    try? FileManager.default.removeItem(at: destination)
    onRetry()

    try await downloadFile(
        repo: repo,
        file: file,
        to: destination,
        bearerToken: bearerToken,
        onProgress: onProgress
    )

    let retryVerified = await verifySHA256(file: destination, expected: hash)
    if retryVerified { return }

    try? FileManager.default.removeItem(at: destination)
    throw DownloadError.checksumMismatch(file: file)
}
```

- [ ] **Step 2: Update `download(_:_:_:_:)` public helper to also accept `bearerToken`**

Replace:

```swift
public func download(
    repo: String,
    file: String,
    to destination: URL,
    onProgress: @Sendable @escaping (Double) -> Void
) async throws {
    try await downloadFile(repo: repo, file: file, to: destination, onProgress: onProgress)
}
```

With:

```swift
public func download(
    repo: String,
    file: String,
    to destination: URL,
    bearerToken: String? = nil,
    onProgress: @Sendable @escaping (Double) -> Void
) async throws {
    try await downloadFile(
        repo: repo,
        file: file,
        to: destination,
        bearerToken: bearerToken,
        onProgress: onProgress
    )
}
```

- [ ] **Step 3: Switch internal `downloadFile` to URLRequest-based API and attach the header**

Replace:

```swift
private func downloadFile(
    repo: String,
    file: String,
    to destination: URL,
    onProgress: @Sendable @escaping (Double) -> Void
) async throws {
    let url = huggingFaceURL(repo: repo, file: file)
    let delegate = DownloadProgressDelegate(onProgress: onProgress)
    let session = URLSession(configuration: .default, delegate: delegate, delegateQueue: nil)

    try await withTaskCancellationHandler {
        defer { session.invalidateAndCancel() }
        let (tempURL, response) = try await session.download(from: url)
        // ...
    } onCancel: {
        session.invalidateAndCancel()
    }
}
```

With:

```swift
private func downloadFile(
    repo: String,
    file: String,
    to destination: URL,
    bearerToken: String?,
    onProgress: @Sendable @escaping (Double) -> Void
) async throws {
    var request = URLRequest(url: huggingFaceURL(repo: repo, file: file))
    if let token = bearerToken, !token.isEmpty {
        request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
    }

    let delegate = DownloadProgressDelegate(onProgress: onProgress)
    let session = URLSession(configuration: .default, delegate: delegate, delegateQueue: nil)

    try await withTaskCancellationHandler {
        defer { session.invalidateAndCancel() }
        let (tempURL, response) = try await session.download(for: request)

        guard let httpResponse = response as? HTTPURLResponse,
              (200...299).contains(httpResponse.statusCode) else {
            throw DownloadError.httpError(
                statusCode: (response as? HTTPURLResponse)?.statusCode ?? -1
            )
        }

        let destDir = destination.deletingLastPathComponent()
        try FileManager.default.createDirectory(at: destDir, withIntermediateDirectories: true)
        if FileManager.default.fileExists(atPath: destination.path()) {
            try FileManager.default.removeItem(at: destination)
        }
        try FileManager.default.moveItem(at: tempURL, to: destination)
    } onCancel: {
        session.invalidateAndCancel()
    }
}
```

- [ ] **Step 4: Add new error cases for auth**

Replace the `DownloadError` enum:

```swift
public enum DownloadError: Error, Sendable, LocalizedError {
    case httpError(statusCode: Int)
    case checksumMismatch(file: String)
    case authRequired(repo: String)
    case authForbidden(repo: String)

    public var errorDescription: String? {
        switch self {
        case .httpError(let statusCode):
            let meaning: String
            switch statusCode {
            case 401: meaning = "Unauthorized — authentication or model access may be required"
            case 403: meaning = "Forbidden — you may need to accept the model's license on HuggingFace"
            case 404: meaning = "Not Found — the model repository or file does not exist"
            case 429: meaning = "Too Many Requests — rate limited, please try again later"
            case 500...599: meaning = "Server error — HuggingFace may be experiencing issues"
            case -1: meaning = "Unknown — no HTTP response received"
            default: meaning = HTTPURLResponse.localizedString(forStatusCode: statusCode)
            }
            return "Download failed (HTTP \(statusCode)): \(meaning)"
        case .checksumMismatch(let file):
            return "Checksum verification failed for '\(file)' — the file may be corrupted"
        case .authRequired(let repo):
            return "\(repo) is a gated model. Add a HuggingFace access token in Settings → Models, and accept the model's license on huggingface.co."
        case .authForbidden(let repo):
            return "HuggingFace rejected the access token for \(repo). Verify the token is valid and that the model license has been accepted."
        }
    }
}
```

- [ ] **Step 5: Build**

Expected: `ModelRegistry` test files and `builtInSpecs` may still error. DownloadClient itself compiles.

## Task 1.4: Update `builtInSpecs` placeholders to use the new shape (still placeholders!)

**Files:**
- Modify: `AnvilCore/Sources/AnvilCore/ModelRegistry/ModelRegistry.swift` (only the `builtInSpecs` array)

- [ ] **Step 1: Rewrite `builtInSpecs` keeping placeholder values but using new init**

Replace the existing `builtInSpecs`:

```swift
public static let builtInSpecs: [ModelSpec] = [
    ModelSpec(
        id: "ace-step-dit",
        huggingFaceRepo: "ACEStep/ACE-Step-v1-5-DiT",
        files: ["model.safetensors", "tokenizer.json", "config.json"],
        sizeBytes: 3_200_000_000,
        sha256: [:],
        category: .aceStepDiT
    ),
    // ... etc
]
```

With:

```swift
public static let builtInSpecs: [ModelSpec] = [
    ModelSpec(
        id: "ace-step-dit",
        displayName: "ACE-Step DiT (placeholder)",
        huggingFaceRepo: "ACEStep/ACE-Step-v1-5-DiT",
        files: [
            ModelFileSpec(remotePath: "model.safetensors", sizeBytes: 3_000_000_000),
            ModelFileSpec(remotePath: "tokenizer.json", sizeBytes: 11_000_000),
            ModelFileSpec(remotePath: "config.json", sizeBytes: 2_000)
        ],
        category: .aceStepDiT
    ),
    ModelSpec(
        id: "ace-step-vae",
        displayName: "ACE-Step VAE (placeholder)",
        huggingFaceRepo: "ACEStep/ACE-Step-v1-5-VAE",
        files: [
            ModelFileSpec(remotePath: "vae.safetensors", sizeBytes: 800_000_000),
            ModelFileSpec(remotePath: "config.json", sizeBytes: 500)
        ],
        category: .aceStepVAE
    ),
    ModelSpec(
        id: "stable-audio-open",
        displayName: "Stable Audio Open 1.0 (placeholder)",
        huggingFaceRepo: "stabilityai/stable-audio-open-1.0",
        files: [
            ModelFileSpec(remotePath: "model.safetensors", sizeBytes: 1_200_000_000),
            ModelFileSpec(remotePath: "config.json", sizeBytes: 4_000)
        ],
        category: .stableAudio
    ),
    ModelSpec(
        id: "lyric-llm",
        displayName: "Phi-3 Mini 4k Instruct",
        huggingFaceRepo: "microsoft/Phi-3-mini-4k-instruct",
        files: [
            ModelFileSpec(remotePath: "model.safetensors", sizeBytes: 2_400_000_000),
            ModelFileSpec(remotePath: "tokenizer.json", sizeBytes: 2_000_000),
            ModelFileSpec(remotePath: "config.json", sizeBytes: 1_000)
        ],
        category: .lyricLLM
    ),
]
```

- [ ] **Step 2: Build**

Expected: `ACEStepGenerationService.swift` (and possibly `SFXModelManager.swift`) still error — they reference `spec.files` as `[String]` and call `spec.files.first(where: { $0.contains("vae") })`. Fix in next task.

## Task 1.5: Update `ACEStepGenerationService` and `SFXModelManager` for new spec shape

**Files:**
- Modify: `AnvilApp/Services/ACEStepGenerationService.swift`
- Modify: `AnvilApp/Services/SFXModelManager.swift` (only if it touches `spec.files` directly)

- [ ] **Step 1: Patch `loadVAEWeights` filename lookup**

Find the block in `ACEStepGenerationService.swift`:

```swift
let vaeFile = spec.files.first(where: { $0.contains("vae") }) ?? "vae.safetensors"
let weightsPath = spec.localPath(for: vaeFile, base: baseDir)
```

Replace with:

```swift
let weightsPath = spec.localPath(for: "vae.safetensors", base: baseDir)
```

> The VAE spec's `ModelFileSpec` now uses `localName: "vae.safetensors"` deterministically, so we can address it by flat name without searching.

- [ ] **Step 2: Find and fix any other `spec.files` String-array assumptions**

Search the codebase:

```bash
cd ~/Developer/GitRepos/AnvilApp && grep -rn "spec.files\|\.files\.first\|\.files\.contains\|\.files\.allSatisfy" --include="*.swift" | grep -v "ModelRegistry.swift\|ModelFileSpec\|fileSpec"
```

Triage each hit. Most likely:
- `SFXModelManager` uses `spec.files` for ready checks — change `for file in spec.files` to `for fileSpec in spec.files` and use `fileSpec.localName` in path resolution.
- Any test fixtures that build a `ModelSpec` literal need to use `ModelFileSpec`.

- [ ] **Step 3: Build**

Expected: `BUILD SUCCEEDED` if AnvilApp + AnvilCore compile cleanly. Test files may still fail — handle in next task.

## Task 1.6: Update unit tests

**Files:**
- Modify: `AnvilCore/Tests/AnvilCoreTests/ModelSpecTests.swift`
- Modify: `AnvilCore/Tests/AnvilCoreTests/DownloadClientTests.swift`
- Modify: `AnvilCore/Tests/AnvilCoreTests/ModelRegistryTests.swift`

- [ ] **Step 1: Update `ModelSpecTests` to construct specs with `ModelFileSpec`**

Replace any `ModelSpec(... files: ["model.safetensors"], sizeBytes: ..., ...)` literal with `ModelSpec(... files: [ModelFileSpec(remotePath: "model.safetensors", sizeBytes: ...)], ...)` and add the new `displayName` parameter. Use `"placeholder"` strings where the test doesn't care.

- [ ] **Step 2: Update `DownloadClientTests` if any test asserts on signatures**

The url builder signature is unchanged, but if a test references `downloadAndVerify` directly, add `bearerToken: nil` (or omit, default works).

- [ ] **Step 3: Update `ModelRegistryTests` for new spec IDs and shapes**

Tests assert `specs.contains(where: { $0.id == "ace-step-dit" })` etc. — those still pass because Phase 1 keeps the old IDs in `builtInSpecs` placeholders. No test ID changes needed yet.

- [ ] **Step 4: Build**

Expected: `BUILD SUCCEEDED` for the whole AnvilApp scheme.

## Task 1.7: Commit Phase 1

- [ ] **Step 1: Stage and commit**

```bash
cd ~/Developer/GitRepos/AnvilApp
git add AnvilCore/Sources/AnvilCore/ModelRegistry/ModelSpec.swift \
        AnvilCore/Sources/AnvilCore/ModelRegistry/ModelRegistry.swift \
        AnvilCore/Sources/AnvilCore/ModelRegistry/DownloadClient.swift \
        AnvilApp/Services/ACEStepGenerationService.swift \
        AnvilApp/Services/SFXModelManager.swift \
        AnvilCore/Tests/AnvilCoreTests/
git commit -m "refactor(core): ModelSpec gains per-file remote/local mapping + auth metadata

Introduces ModelFileSpec with explicit remotePath/localName so a single
ModelSpec can represent files that live in HuggingFace subdirectories
while consumer code keeps loading flat filenames. Adds variantFamily
and requiresAuth to ModelSpec. Adds bearerToken plumbing to DownloadClient
(URLRequest-based, defaults nil). DownloadError gains authRequired and
authForbidden cases.

No behavioral change yet — builtInSpecs keep placeholder values that still
return HTTP 401 on download. Phase 2 swaps them for real ACE-Step v1.5
repo paths.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

- [ ] **Step 2: Verify build still passes**

```bash
xcodebuild -scheme AnvilApp -destination 'platform=macOS' build 2>&1 | grep -E "error:|BUILD"
```

Expected: `BUILD SUCCEEDED`.

---

# Phase 2 — Real ACE-Step v1.5 Repo Paths + Turbo/Full Variants

This is the phase that makes downloads actually work.

## Task 2.1: Rewrite `builtInSpecs` with real values

**Files:**
- Modify: `AnvilCore/Sources/AnvilCore/ModelRegistry/ModelRegistry.swift`

- [ ] **Step 1: Replace `builtInSpecs` with the verified real-world layout**

```swift
public static let builtInSpecs: [ModelSpec] = [
    ModelSpec(
        id: "ace-step-dit-turbo",
        displayName: "ACE-Step v1.5 Turbo",
        huggingFaceRepo: "ACE-Step/Ace-Step1.5",
        files: [
            ModelFileSpec(
                remotePath: "acestep-v15-turbo/model.safetensors",
                localName: "model.safetensors",
                sizeBytes: 4_787_825_604
            ),
            ModelFileSpec(
                remotePath: "acestep-v15-turbo/config.json",
                localName: "config.json",
                sizeBytes: 1_968
            )
        ],
        category: .aceStepDiT,
        variantFamily: "ace-step-1.5"
    ),
    ModelSpec(
        id: "ace-step-dit-base",
        displayName: "ACE-Step v1.5 Full",
        huggingFaceRepo: "ACE-Step/acestep-v15-base",
        files: [
            ModelFileSpec(
                remotePath: "model.safetensors",
                localName: "model.safetensors",
                sizeBytes: 4_787_825_604
            ),
            ModelFileSpec(
                remotePath: "config.json",
                localName: "config.json",
                sizeBytes: 1_940
            )
        ],
        category: .aceStepDiT,
        variantFamily: "ace-step-1.5"
    ),
    ModelSpec(
        id: "ace-step-vae",
        displayName: "ACE-Step VAE + Tokenizer",
        huggingFaceRepo: "ACE-Step/Ace-Step1.5",
        files: [
            ModelFileSpec(
                remotePath: "vae/diffusion_pytorch_model.safetensors",
                localName: "vae.safetensors",
                sizeBytes: 337_431_388
            ),
            ModelFileSpec(
                remotePath: "vae/config.json",
                localName: "config.json",
                sizeBytes: 425
            ),
            ModelFileSpec(
                remotePath: "Qwen3-Embedding-0.6B/tokenizer.json",
                localName: "tokenizer.json",
                sizeBytes: 11_423_705
            )
        ],
        category: .aceStepVAE,
        variantFamily: "ace-step-1.5"
    ),
    ModelSpec(
        id: "stable-audio-open",
        displayName: "Stable Audio Open 1.0",
        huggingFaceRepo: "stabilityai/stable-audio-open-1.0",
        files: [
            ModelFileSpec(
                remotePath: "model.safetensors",
                localName: "model.safetensors",
                sizeBytes: 4_853_889_016
            ),
            ModelFileSpec(
                remotePath: "model_config.json",
                localName: "config.json",
                sizeBytes: 4_172
            ),
            ModelFileSpec(
                remotePath: "vae_model.ckpt",
                localName: "vae_model.ckpt",
                sizeBytes: 624_474_947
            ),
            ModelFileSpec(
                remotePath: "vae_model_config.json",
                localName: "vae_config.json",
                sizeBytes: 3_608
            )
        ],
        category: .stableAudio,
        variantFamily: nil,
        requiresAuth: true
    ),
    ModelSpec(
        id: "lyric-llm",
        displayName: "Phi-3 Mini 4k Instruct",
        huggingFaceRepo: "microsoft/Phi-3-mini-4k-instruct",
        files: [
            ModelFileSpec(remotePath: "model.safetensors", sizeBytes: 2_400_000_000),
            ModelFileSpec(remotePath: "tokenizer.json", sizeBytes: 2_000_000),
            ModelFileSpec(remotePath: "config.json", sizeBytes: 1_000)
        ],
        category: .lyricLLM
    ),
]
```

- [ ] **Step 2: Update `defaultActiveModels` and `essentialModelIDs`**

Find:

```swift
private static let defaultActiveModels: [ModelCategory: String] = [
    .aceStepDiT: "ace-step-dit",
    .aceStepVAE: "ace-step-vae",
    // ...
]

public static let essentialModelIDs: Set<String> = [
    "ace-step-dit",
    "ace-step-vae"
]
```

Replace with:

```swift
private static let defaultActiveModels: [ModelCategory: String] = [
    .aceStepDiT: "ace-step-dit-turbo",
    .aceStepVAE: "ace-step-vae",
    // keep other categories as-is
]

public static let essentialModelIDs: Set<String> = [
    "ace-step-dit-turbo",
    "ace-step-vae"
]
```

(Verify the existing `defaultActiveModels` has any other category entries and preserve them.)

## Task 2.2: Add one-shot migration shim for legacy active-model IDs

**Files:**
- Modify: `AnvilCore/Sources/AnvilCore/ModelRegistry/ModelRegistry.swift`

- [ ] **Step 1: Add a private migration method**

Find the place where active models are loaded from persistence (likely `init` or `loadActiveModels()`). Add a migration method:

```swift
/// One-shot migration from Phase-1 placeholder spec IDs to Phase-2 real IDs.
/// Runs exactly once (gated on UserDefaults flag).
private func migrateLegacyModelIDsIfNeeded() {
    let migrationKey = "com.pimp.AnvilApp.modelMigration.v1"
    guard !UserDefaults.standard.bool(forKey: migrationKey) else { return }

    // Map legacy "ace-step-dit" → "ace-step-dit-turbo"
    if activeModels[.aceStepDiT] == "ace-step-dit" {
        activeModels[.aceStepDiT] = "ace-step-dit-turbo"
        logger.info("Migrated legacy active DiT 'ace-step-dit' → 'ace-step-dit-turbo'")
    }

    // Sweep orphan directories whose names don't match any current spec ID.
    let validIDs = Set(allSpecs.map(\.id))
    if let entries = try? FileManager.default.contentsOfDirectory(
        at: baseDirectory, includingPropertiesForKeys: nil
    ) {
        for entry in entries {
            let name = entry.lastPathComponent
            // Skip files (only sweep directories) and the user-models metadata file.
            var isDir: ObjCBool = false
            FileManager.default.fileExists(atPath: entry.path(), isDirectory: &isDir)
            guard isDir.boolValue, name != "user-models.json" else { continue }
            if !validIDs.contains(name) {
                try? FileManager.default.removeItem(at: entry)
                logger.info("Removed orphan model directory: \(name)")
            }
        }
    }

    UserDefaults.standard.set(true, forKey: migrationKey)
}
```

- [ ] **Step 2: Call the migration once after specs and active models are loaded**

In the registry's init or wherever active models are settled at construction, add:

```swift
migrateLegacyModelIDsIfNeeded()
```

- [ ] **Step 3: Build**

```bash
xcodebuild -scheme AnvilApp -destination 'platform=macOS' build 2>&1 | grep -E "error:|BUILD"
```

Expected: `BUILD SUCCEEDED` (the test files may need updates next).

## Task 2.3: Patch `ACEStepGenerationService` to look up tokenizer from VAE spec

**Files:**
- Modify: `AnvilApp/Services/ACEStepGenerationService.swift`

The current code looks for `tokenizer.json` inside the DiT spec's directory. Now `tokenizer.json` lives in the VAE spec's directory.

- [ ] **Step 1: Update the tokenizer-upgrade path in `loadDiTWeights`**

Find the block:

```swift
// Upgrade stub ConditioningActor to one backed by the real downloaded tokenizer.
let tokenizerPath = spec.localPath(for: "tokenizer.json", base: baseDir)
if FileManager.default.fileExists(atPath: tokenizerPath.path()) {
    if let real = try? ConditioningActor.make(tokenizerURL: tokenizerPath) {
        conditioning = real
    } else {
        logger.warning("Could not upgrade ConditioningActor from tokenizer at \(tokenizerPath.path()) — using stub")
    }
}
```

Replace with:

```swift
// Upgrade stub ConditioningActor to one backed by the real downloaded tokenizer.
// tokenizer.json lives in the VAE spec's directory (shared between DiT variants).
if let vaeSpec = await registry.activeSpec(for: .aceStepVAE) {
    let tokenizerPath = vaeSpec.localPath(for: "tokenizer.json", base: baseDir)
    if FileManager.default.fileExists(atPath: tokenizerPath.path()) {
        if let real = try? ConditioningActor.make(tokenizerURL: tokenizerPath) {
            conditioning = real
        } else {
            logger.warning("Could not upgrade ConditioningActor from tokenizer at \(tokenizerPath.path()) — using stub")
        }
    } else {
        logger.info("tokenizer.json not yet downloaded at \(tokenizerPath.path()) — keeping stub conditioning actor")
    }
}
```

## Task 2.4: Group variants in `ModelsSettingsView`

**Files:**
- Modify: `AnvilApp/UI/Settings/ModelsSettingsView.swift`

- [ ] **Step 1: Read the current view to understand its grouping structure**

```bash
cd ~/Developer/GitRepos/AnvilApp && grep -n "Section\|aceStepDiT\|builtIn" AnvilApp/UI/Settings/ModelsSettingsView.swift | head -20
```

- [ ] **Step 2: Replace the existing single "ACE-Step (Music)" section with variant-grouped rendering**

Inside the `body` of `ModelsSettingsView`, where the ACE-Step section is built, change the model row iteration to group `aceStepDiT` specs by `variantFamily`:

```swift
Section("ACE-Step (Music)") {
    let ditSpecs = registry.allSpecs.filter { $0.category == .aceStepDiT }
    let groupedByFamily = Dictionary(grouping: ditSpecs, by: { $0.variantFamily ?? "ungrouped" })

    ForEach(groupedByFamily.keys.sorted(), id: \.self) { family in
        if let variants = groupedByFamily[family], variants.count > 1 {
            // Render as a sub-grouped block with a header for the family
            VStack(alignment: .leading, spacing: 4) {
                Text(family.replacingOccurrences(of: "-", with: " ").capitalized)
                    .font(.headline)
                ForEach(variants.sorted(by: { $0.id < $1.id }), id: \.id) { spec in
                    ModelRowView(spec: spec)
                }
            }
        } else if let single = variants?.first {
            ModelRowView(spec: single)
        }
    }

    // VAE row(s)
    let vaeSpecs = registry.allSpecs.filter { $0.category == .aceStepVAE }
    ForEach(vaeSpecs, id: \.id) { spec in
        ModelRowView(spec: spec)
    }
}
```

> If `ModelRowView` doesn't exist by that name, adapt to whatever the existing per-spec row component is called. The point is: under "ACE-Step (Music)", show Turbo and Full as sibling rows under a "Ace step 1.5" sub-header, then VAE as its own row.

- [ ] **Step 3: Build and visually inspect**

```bash
xcodebuild -scheme AnvilApp -destination 'platform=macOS' build 2>&1 | grep -E "error:|BUILD"
```

Expected: `BUILD SUCCEEDED`. Run the app, open Settings → Models, verify both Turbo and Full appear under ACE-Step v1.5.

## Task 2.5: Update unit tests for new spec IDs

**Files:**
- Modify: `AnvilCore/Tests/AnvilCoreTests/ModelRegistryTests.swift`

- [ ] **Step 1: Update spec ID assertions**

Find:

```swift
#expect(specs.contains(where: { $0.id == "ace-step-dit" }))
```

Replace with:

```swift
#expect(specs.contains(where: { $0.id == "ace-step-dit-turbo" }))
#expect(specs.contains(where: { $0.id == "ace-step-dit-base" }))
```

- [ ] **Step 2: Build**

Expected: `BUILD SUCCEEDED`.

## Task 2.6: Update CHANGELOG and commit Phase 2

**Files:**
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Add a new entry under "[Unreleased] — GenerationService Split" or as a new section**

Add a new "[Unreleased] — Model Download Restructure" section at the top of CHANGELOG.md (above the GenerationService Split section), with at least:

```markdown
## [Unreleased] — Model Download Restructure (Phase 1 + 2)

### Added

- **`ModelFileSpec`** — explicit `remotePath`/`localName`/`sha256`/`sizeBytes` per file in a model spec, replacing the flat `files: [String]` array. Lets one HuggingFace repo's subdirectory layout map to flat local filenames that consumer code already expects.
- **`ModelSpec.variantFamily`** — groups related specs (e.g. ACE-Step Turbo and Full) under one UI section.
- **`ModelSpec.requiresAuth`** — flags gated repos so downloads attach `Authorization: Bearer` headers and pre-flight on token presence.
- **`ace-step-dit-base`** — Full ACE-Step v1.5 DiT variant (4.8 GB, higher quality, slower).
- **One-shot migration shim** — maps legacy active-model ID `ace-step-dit` to `ace-step-dit-turbo` and sweeps orphan model directories. Gated on `UserDefaults` flag `com.pimp.AnvilApp.modelMigration.v1`.

### Changed

- **`ModelRegistry.builtInSpecs`** — placeholder repo IDs replaced with verified `ACE-Step/Ace-Step1.5` and `ACE-Step/acestep-v15-base` paths and exact file sizes from the HF API.
- **`ace-step-dit` → `ace-step-dit-turbo` (default)** — Turbo variant is now the out-of-the-box DiT model. Lower memory, 8-step diffusion. Full variant available in Settings → Models.
- **`ace-step-vae` now bundles `tokenizer.json`** — shared between DiT variants. `ACEStepGenerationService` looks up the tokenizer from the active VAE spec's local directory.
- **`DownloadClient`** — switched to `URLRequest`-based download API to support per-request headers; new `bearerToken: String?` parameter.

### Fixed

- **HTTP 401 "DownloadError error 0" on first launch** — placeholder repos `ACEStep/ACE-Step-v1-5-DiT` and `ACEStep/ACE-Step-v1-5-VAE` did not exist on HuggingFace. Now resolves to real repos.
```

- [ ] **Step 2: Commit Phase 2**

```bash
cd ~/Developer/GitRepos/AnvilApp
git add AnvilCore/Sources/AnvilCore/ModelRegistry/ModelRegistry.swift \
        AnvilApp/Services/ACEStepGenerationService.swift \
        AnvilApp/UI/Settings/ModelsSettingsView.swift \
        AnvilCore/Tests/AnvilCoreTests/ \
        CHANGELOG.md
git commit -m "feat(models): real ACE-Step v1.5 repo paths + Turbo/Full variants

builtInSpecs now uses the verified ACE-Step/Ace-Step1.5 (Turbo + VAE +
tokenizer) and ACE-Step/acestep-v15-base (Full) repos. Default DiT is
'ace-step-dit-turbo' (4.8 GB, 8-step). Full variant ships as a sibling
spec under variantFamily 'ace-step-1.5' so users can opt into the larger
model from Settings → Models.

Tokenizer.json moves to the VAE spec (shared across DiT variants);
ACEStepGenerationService looks it up from the active VAE spec's directory.

Includes a one-shot migration that remaps legacy 'ace-step-dit' active
ID to 'ace-step-dit-turbo' and sweeps orphan directories from the
placeholder era. Gated on UserDefaults flag.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

- [ ] **Step 3: Smoke test on device (manual — required before Phase 3)**

Run the app from Xcode. Verify:
- First-launch download succeeds for Turbo + VAE
- Files land at expected paths: `<appsupport>/.../ace-step-dit-turbo/model.safetensors`, `<appsupport>/.../ace-step-vae/vae.safetensors`, `<appsupport>/.../ace-step-vae/tokenizer.json`
- Generate a 10s clip — succeeds
- Settings → Models shows Turbo and Full under "ACE-Step v1.5"
- Switch active DiT to Full — triggers Full download, then weights swap on next generation

---

# Phase 3 — HuggingFace Token UI + Bearer Wiring

## Task 3.1: Add HuggingFace token storage to `APIKeyStore`

**Files:**
- Modify: `AnvilApp/Services/APIKeyStore.swift`

- [ ] **Step 1: Read the current `APIKeyStore` to understand its enum + accessor pattern**

```bash
cd ~/Developer/GitRepos/AnvilApp && cat AnvilApp/Services/APIKeyStore.swift | head -80
```

- [ ] **Step 2: Add `huggingFace` provider case**

Add to the existing provider enum (it's likely a string-backed enum like `case openAI = "openai"`). Add:

```swift
case huggingFace = "huggingface"
```

Add a convenience accessor:

```swift
public func huggingFaceToken() -> String? {
    return token(for: .huggingFace)
}
```

(Adapt to whatever the existing accessor pattern is — `token(for:)`, `apiKey(for:)`, etc.)

- [ ] **Step 3: Build**

Expected: `BUILD SUCCEEDED`.

## Task 3.2: Wire token provider into `ModelRegistry`

**Files:**
- Modify: `AnvilCore/Sources/AnvilCore/ModelRegistry/ModelRegistry.swift`

- [ ] **Step 1: Add an optional token-provider closure to `ModelRegistry`**

Add a new private property and init parameter:

```swift
private let tokenProvider: (@Sendable () async -> String?)?

public init(
    baseDirectory: URL,
    client: DownloadClient = DownloadClient(),
    tokenProvider: (@Sendable () async -> String?)? = nil
) {
    self.baseDirectory = baseDirectory
    self.client = client
    self.tokenProvider = tokenProvider
    // ... existing init body
}
```

- [ ] **Step 2: Resolve and use the token in `downloadModels`**

Inside the `for spec in specs { ... }` loop in `downloadModels`, before the `for fileSpec in spec.files { ... }` loop, add:

```swift
let bearerToken: String?
if spec.requiresAuth {
    bearerToken = await tokenProvider?()
    guard let token = bearerToken, !token.isEmpty else {
        statuses[spec.id] = .failed("Token required")
        throw DownloadError.authRequired(repo: spec.huggingFaceRepo)
    }
    _ = token  // used below as bearerToken
} else {
    bearerToken = nil
}
```

Then update the inner `client.downloadAndVerify` call to pass `bearerToken: bearerToken` instead of `bearerToken: nil`.

- [ ] **Step 3: Map HTTP 401/403 with token to `authForbidden`**

Inside the inner `do { ... } catch { ... }` block, before the `let message: String` assignment, intercept auth-related HTTP errors when a token was supplied:

```swift
if let httpError = error as? DownloadError,
   case .httpError(let code) = httpError,
   (code == 401 || code == 403),
   bearerToken != nil {
    statuses[spec.id] = .failed("Token rejected")
    throw DownloadError.authForbidden(repo: spec.huggingFaceRepo)
}
```

- [ ] **Step 4: Build**

Expected: `BUILD SUCCEEDED`.

## Task 3.3: Wire `APIKeyStore` to `ModelRegistry` at app startup

**Files:**
- Modify: `AnvilApp/AnvilAppApp.swift` (or wherever `ModelRegistry` is constructed — likely `AppState`)

- [ ] **Step 1: Find the `ModelRegistry` construction site**

```bash
cd ~/Developer/GitRepos/AnvilApp && grep -rn "ModelRegistry(" --include="*.swift"
```

- [ ] **Step 2: Pass the token provider into the registry**

At the call site (likely in `AppState.init` or `AnvilAppApp.init`), add the closure:

```swift
let apiKeyStore = APIKeyStore()
let registry = ModelRegistry(
    baseDirectory: <existingBaseDir>,
    tokenProvider: { [weak apiKeyStore] in
        apiKeyStore?.huggingFaceToken()
    }
)
```

If `APIKeyStore` is a `@MainActor` class and `tokenProvider` is `@Sendable`, you may need to await an actor hop:

```swift
tokenProvider: { [apiKeyStore] in
    await MainActor.run { apiKeyStore.huggingFaceToken() }
}
```

(Adapt to actual concurrency model.)

- [ ] **Step 3: Build**

Expected: `BUILD SUCCEEDED`.

## Task 3.4: Add HF token entry UI to `ModelsSettingsView`

**Files:**
- Modify: `AnvilApp/UI/Settings/ModelsSettingsView.swift`

- [ ] **Step 1: Add a new `Section` above the model list**

Inside `body`, before the model-list sections, add:

```swift
Section("HuggingFace Access Token (optional)") {
    SecureField("hf_…", text: $huggingFaceToken)
    HStack {
        Button("Save") {
            apiKeyStore.setToken(huggingFaceToken, for: .huggingFace)
        }
        .disabled(huggingFaceToken.isEmpty)
        Spacer()
        Link("Get a token", destination: URL(string: "https://huggingface.co/settings/tokens")!)
    }
    Text("Required for gated models like Stable Audio Open. Token is stored in the macOS Keychain.")
        .font(.caption)
        .foregroundStyle(.secondary)
}
```

(Adapt the API key store call to whatever the actual store method name is.)

- [ ] **Step 2: Add `@State private var huggingFaceToken: String = ""` at the top of the view**

Initialize from existing token if any:

```swift
@State private var huggingFaceToken: String = ""

// In .onAppear or .task:
.task {
    huggingFaceToken = apiKeyStore.huggingFaceToken() ?? ""
}
```

- [ ] **Step 3: Build and run**

Verify:
- Section appears above model list
- Pasting a token and tapping Save persists across relaunches
- "Get a token" link opens HF in browser

## Task 3.5: Update CHANGELOG and commit Phase 3

**Files:**
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Append to the existing "[Unreleased] — Model Download Restructure" section**

Under `### Added`:

```markdown
- **HuggingFace access token storage** — `APIKeyStore` gains a `huggingFace` provider; new section in Settings → Models for entering and persisting the token in the macOS Keychain. Required for gated repos like `stabilityai/stable-audio-open-1.0`.
- **`DownloadClient` bearer-token attachment** — when a model spec has `requiresAuth: true`, downloads attach `Authorization: Bearer <token>` and pre-flight on token presence.
- **`DownloadError.authRequired` and `.authForbidden`** — distinct error cases for "no token configured for gated repo" vs "token rejected by HF".
```

- [ ] **Step 2: Commit Phase 3**

```bash
cd ~/Developer/GitRepos/AnvilApp
git add AnvilApp/Services/APIKeyStore.swift \
        AnvilCore/Sources/AnvilCore/ModelRegistry/ModelRegistry.swift \
        AnvilApp/AnvilAppApp.swift \
        AnvilApp/UI/Settings/ModelsSettingsView.swift \
        CHANGELOG.md
git commit -m "feat(downloads): HuggingFace bearer token support for gated models

Adds a 'HuggingFace Access Token (optional)' section to Settings → Models
backed by the existing APIKeyStore (Keychain). ModelRegistry takes an
optional tokenProvider closure; specs flagged requiresAuth=true gate the
download on token presence (DownloadError.authRequired) and map 401/403
responses with token present to DownloadError.authForbidden.

Unblocks downloading stabilityai/stable-audio-open-1.0 once the user has
a valid HF token and has accepted the model's license on huggingface.co.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

- [ ] **Step 3: Smoke test (manual)**

- No HF token configured + attempt Stable Audio download → pre-flight error: "Stable Audio Open 1.0 is a gated model…"
- Bad token + attempt → `authForbidden` error
- Valid token + license accepted on huggingface.co + attempt → succeeds
- Token persists across relaunches

---

# Final Step: Push and Finish Branch

After Phase 3 commit lands and smoke tests pass:

- [ ] **Step 1: Push all three phase commits**

```bash
cd ~/Developer/GitRepos/AnvilApp && git push origin main
```

- [ ] **Step 2: Verify build is still green**

```bash
xcodebuild -scheme AnvilApp -destination 'platform=macOS' build 2>&1 | grep -E "error:|BUILD"
```

Expected: `BUILD SUCCEEDED`.

---

## Summary of Commits

| Commit (expected) | Phase | Files Changed |
|---|---|---|
| `refactor(core): ModelSpec gains per-file remote/local mapping + auth metadata` | 1 | `ModelSpec.swift`, `ModelRegistry.swift`, `DownloadClient.swift`, `ACEStepGenerationService.swift`, `SFXModelManager.swift`, tests |
| `feat(models): real ACE-Step v1.5 repo paths + Turbo/Full variants` | 2 | `ModelRegistry.swift`, `ACEStepGenerationService.swift`, `ModelsSettingsView.swift`, tests, `CHANGELOG.md` |
| `feat(downloads): HuggingFace bearer token support for gated models` | 3 | `APIKeyStore.swift`, `ModelRegistry.swift`, `AnvilAppApp.swift`, `ModelsSettingsView.swift`, `CHANGELOG.md` |
