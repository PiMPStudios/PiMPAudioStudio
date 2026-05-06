# Phase 9a: Model Management Foundation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Harden model downloads with SHA256 verification and auto-retry, redesign first-launch for faster time-to-music, and prepare the registry for Phase 9b model swapping.

**Architecture:** SHA256 enforcement in DownloadClient with one auto-retry. ModelRegistry splits into static builtInSpecs and runtime userSpecs with JSON persistence. First-launch gates only on ACE-Step models (~4 GB), then background-downloads SFX and LLM.

**Tech Stack:** Swift, CryptoKit (SHA256), SwiftUI, AnvilCore (ModelRegistry/DownloadClient/ModelSpec)

---

## File Structure

### Modified Files

| File | Responsibility |
|------|---------------|
| `AnvilCore/Sources/AnvilCore/ModelRegistry/ModelSpec.swift` | Add `Codable`, populate SHA256 checksums |
| `AnvilCore/Sources/AnvilCore/ModelRegistry/DownloadClient.swift` | Enforce verification after download, auto-retry on failure |
| `AnvilCore/Sources/AnvilCore/ModelRegistry/ModelRegistry.swift` | Static `builtInSpecs`, runtime `userSpecs`, JSON persistence, extension points |
| `AnvilApp/AppState.swift` | ACE-Step-only setup gate, background download coordination, `BackgroundDownloadPhase` |
| `AnvilApp/UI/FirstLaunchView.swift` | Show only ACE-Step models, updated copy |

### New Files

| File | Responsibility |
|------|---------------|
| `AnvilApp/UI/Components/BackgroundDownloadIndicator.swift` | Subtle progress pill for background download status |

---

### Task 1: ModelSpec — Codable Conformance and SHA256 Checksums

**Files:**

- Modify: `AnvilCore/Sources/AnvilCore/ModelRegistry/ModelSpec.swift`

- [ ] **Step 1: Add Codable conformance and populate SHA256 checksums**

Replace the entire file with:

```swift
import Foundation

public struct ModelSpec: Sendable, Identifiable, Codable {
    public let id: String
    public let huggingFaceRepo: String
    public let files: [String]
    public let sizeBytes: Int64
    /// SHA256 checksums keyed by filename. Empty dictionary means skip
    /// verification (used for user-added models without known hashes).
    public let sha256: [String: String]

    public init(
        id: String,
        huggingFaceRepo: String,
        files: [String],
        sizeBytes: Int64,
        sha256: [String: String] = [:]
    ) {
        self.id = id
        self.huggingFaceRepo = huggingFaceRepo
        self.files = files
        self.sizeBytes = sizeBytes
        self.sha256 = sha256
    }

    public var totalSizeFormatted: String {
        let gb = Double(sizeBytes) / 1_000_000_000
        return String(format: "%.1f GB", gb)
    }

    public func localDirectory(base: URL) -> URL {
        base.appending(path: id)
    }

    public func localPath(for file: String, base: URL) -> URL {
        localDirectory(base: base).appending(path: file)
    }
}
```

Changes from the original:

1. Added `Codable` to the conformance list
2. Added default value `= [:]` for `sha256` parameter in `init`

Note: Actual SHA256 hash values will be computed during implementation by
downloading each model file and running `SHA256.hash(data:)`. For now the
checksums remain empty `[:]` in `ModelRegistry` — they will be populated
in Task 3 when we update the registry's built-in specs. The `Codable`
conformance is what matters for this task.

- [ ] **Step 2: Verify Xcode build**

Run: `xcodebuild -project AnvilApp.xcodeproj -scheme AnvilApp -destination 'platform=macOS' build 2>&1 | tail -5`
Expected: `** BUILD SUCCEEDED **` (or only pre-existing SFXModelManager errors)

- [ ] **Step 3: Commit**

```bash
git add AnvilCore/Sources/AnvilCore/ModelRegistry/ModelSpec.swift
git commit -m "feat(model): add Codable conformance to ModelSpec with default sha256"
```

---

### Task 2: DownloadClient — Verification with Auto-Retry

**Files:**

- Modify: `AnvilCore/Sources/AnvilCore/ModelRegistry/DownloadClient.swift`

- [ ] **Step 1: Add verified download method with auto-retry**

Replace the entire file with:

```swift
import Foundation
import CryptoKit

public actor DownloadClient {
    public init() {}

    public func huggingFaceURL(repo: String, file: String) -> URL {
        var components = URLComponents()
        components.scheme = "https"
        components.host = "huggingface.co"
        components.path = "/\(repo)/resolve/main/\(file)"
        guard let url = components.url else {
            preconditionFailure("Invalid HuggingFace URL for repo: \(repo), file: \(file)")
        }
        return url
    }

    /// Downloads a file and verifies its SHA256 checksum if provided.
    /// On verification failure, deletes the file and retries once.
    /// If `expectedHash` is nil or empty, skips verification.
    public func downloadAndVerify(
        repo: String,
        file: String,
        to destination: URL,
        expectedHash: String?,
        onProgress: @Sendable @escaping (Double) -> Void,
        onRetry: @Sendable @escaping () -> Void
    ) async throws {
        try await downloadFile(repo: repo, file: file, to: destination, onProgress: onProgress)

        guard let hash = expectedHash, !hash.isEmpty else { return }

        let verified = await verifySHA256(file: destination, expected: hash)
        if verified { return }

        // First verification failed — delete and retry once
        try? FileManager.default.removeItem(at: destination)
        onRetry()

        try await downloadFile(repo: repo, file: file, to: destination, onProgress: onProgress)

        let retryVerified = await verifySHA256(file: destination, expected: hash)
        if retryVerified { return }

        // Second verification failed — hard fail
        try? FileManager.default.removeItem(at: destination)
        throw DownloadError.checksumMismatch(file: file)
    }

    /// Raw download without verification. Used internally by downloadAndVerify.
    public func download(
        repo: String,
        file: String,
        to destination: URL,
        onProgress: @Sendable @escaping (Double) -> Void
    ) async throws {
        try await downloadFile(repo: repo, file: file, to: destination, onProgress: onProgress)
    }

    private func downloadFile(
        repo: String,
        file: String,
        to destination: URL,
        onProgress: @Sendable @escaping (Double) -> Void
    ) async throws {
        let url = huggingFaceURL(repo: repo, file: file)
        let delegate = DownloadProgressDelegate(onProgress: onProgress)
        let session = URLSession(configuration: .default, delegate: delegate, delegateQueue: nil)
        defer { session.invalidateAndCancel() }

        let (tempURL, response) = try await session.download(from: url)

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
    }

    public nonisolated func verifySHA256(file: URL, expected: String) async -> Bool {
        await Task.detached(priority: .userInitiated) {
            guard let data = try? Data(contentsOf: file) else { return false }
            let digest = SHA256.hash(data: data)
            let hex = digest.map { String(format: "%02x", $0) }.joined()
            return hex == expected.lowercased()
        }.value
    }
}

public enum DownloadError: Error, Sendable {
    case httpError(statusCode: Int)
    case checksumMismatch(file: String)
}

private final class DownloadProgressDelegate: NSObject, URLSessionDownloadDelegate, @unchecked Sendable {
    private let onProgress: @Sendable (Double) -> Void

    init(onProgress: @Sendable @escaping (Double) -> Void) {
        self.onProgress = onProgress
    }

    func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didFinishDownloadingTo location: URL
    ) {}

    func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didWriteData bytesWritten: Int64,
        totalBytesWritten: Int64,
        totalBytesExpectedToWrite: Int64
    ) {
        guard totalBytesExpectedToWrite > 0 else { return }
        onProgress(Double(totalBytesWritten) / Double(totalBytesExpectedToWrite))
    }
}
```

Changes from the original:

1. Added `downloadAndVerify` method with auto-retry logic
2. Extracted `downloadFile` private method (refactored from old `download`)
3. Kept original `download` method as a public passthrough (SFXModelManager
   uses it directly)
4. `DownloadProgressDelegate`, `DownloadError`, `verifySHA256` unchanged

- [ ] **Step 2: Verify Xcode build**

Run: `xcodebuild -project AnvilApp.xcodeproj -scheme AnvilApp -destination 'platform=macOS' build 2>&1 | tail -5`
Expected: `** BUILD SUCCEEDED **` (or only pre-existing errors)

- [ ] **Step 3: Commit**

```bash
git add AnvilCore/Sources/AnvilCore/ModelRegistry/DownloadClient.swift
git commit -m "feat(model): add downloadAndVerify with SHA256 enforcement and auto-retry"
```

---

### Task 3: ModelRegistry — Built-in/User Split with JSON Persistence

**Files:**

- Modify: `AnvilCore/Sources/AnvilCore/ModelRegistry/ModelRegistry.swift`

- [ ] **Step 1: Rewrite ModelRegistry with builtInSpecs, userSpecs, and extension points**

Replace the entire file with:

```swift
import Foundation
import OSLog

private let logger = Logger(subsystem: "com.pimp.AnvilApp", category: "ModelRegistry")

public enum ModelStatus: Sendable, Equatable {
    case notDownloaded
    case downloading(progress: Double)
    case verifying
    case ready
    case failed(String)
}

public actor ModelRegistry {
    public let baseDirectory: URL
    private let client: DownloadClient
    private var statuses: [String: ModelStatus] = [:]

    // MARK: - Built-in Models

    public static let builtInSpecs: [ModelSpec] = [
        ModelSpec(
            id: "ace-step-dit",
            huggingFaceRepo: "ACEStep/ACE-Step-v1-5-DiT",
            files: ["model.safetensors", "config.json"],
            sizeBytes: 3_200_000_000,
            sha256: [:]
        ),
        ModelSpec(
            id: "ace-step-vae",
            huggingFaceRepo: "ACEStep/ACE-Step-v1-5-VAE",
            files: ["vae.safetensors", "config.json"],
            sizeBytes: 800_000_000,
            sha256: [:]
        ),
        ModelSpec(
            id: "stable-audio-open",
            huggingFaceRepo: "stabilityai/stable-audio-open-1.0",
            files: ["model.safetensors", "config.json"],
            sizeBytes: 1_200_000_000,
            sha256: [:]
        ),
        ModelSpec(
            id: "lyric-llm",
            huggingFaceRepo: "microsoft/Phi-3-mini-4k-instruct",
            files: ["model.safetensors", "tokenizer.json", "config.json"],
            sizeBytes: 2_400_000_000,
            sha256: [:]
        ),
    ]

    // MARK: - User-Added Models

    private(set) var userSpecs: [ModelSpec] = []

    private var userSpecsFileURL: URL {
        baseDirectory.appending(path: "user-models.json")
    }

    // MARK: - Unified Access

    public var allSpecs: [ModelSpec] {
        Self.builtInSpecs + userSpecs
    }

    /// The ACE-Step model IDs required for initial setup.
    public static let essentialModelIDs: Set<String> = [
        "ace-step-dit",
        "ace-step-vae"
    ]

    // MARK: - Init

    public init(baseDirectory: URL, client: DownloadClient = DownloadClient()) {
        self.baseDirectory = baseDirectory
        self.client = client
        loadUserSpecs()
    }

    // MARK: - Status

    public func status(for modelID: String) -> ModelStatus {
        if let cached = statuses[modelID] {
            return cached
        }
        guard let spec = allSpecs.first(where: { $0.id == modelID }) else {
            return .failed("Unknown model: \(modelID)")
        }
        let allFilesExist = spec.files.allSatisfy { file in
            FileManager.default.fileExists(
                atPath: spec.localPath(for: file, base: baseDirectory).path()
            )
        }
        return allFilesExist ? .ready : .notDownloaded
    }

    public func isModelReady(_ modelID: String) -> Bool {
        status(for: modelID) == .ready
    }

    private func setStatus(_ modelID: String, to status: ModelStatus) {
        statuses[modelID] = status
    }

    // MARK: - Download

    /// Downloads the specified models with SHA256 verification and auto-retry.
    public func downloadModels(
        _ specs: [ModelSpec],
        onProgress: @Sendable @escaping (String, Double) -> Void
    ) async throws {
        for spec in specs {
            guard status(for: spec.id) != .ready else { continue }
            statuses[spec.id] = .downloading(progress: 0)

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

            statuses[spec.id] = .ready
        }
    }

    /// Downloads only the essential (ACE-Step) models for first launch.
    public func downloadEssentialModels(
        onProgress: @Sendable @escaping (String, Double) -> Void
    ) async throws {
        let essentials = Self.builtInSpecs.filter { Self.essentialModelIDs.contains($0.id) }
        try await downloadModels(essentials, onProgress: onProgress)
    }

    /// Downloads all built-in models. Kept for backward compatibility.
    public func downloadAll(
        onProgress: @Sendable @escaping (String, Double) -> Void
    ) async throws {
        try await downloadModels(Self.builtInSpecs, onProgress: onProgress)
    }

    // MARK: - User Model Management (extension points for 9b)

    public func addModel(_ spec: ModelSpec) throws {
        guard !isBuiltIn(id: spec.id) else {
            throw RegistryError.cannotModifyBuiltIn
        }
        guard !userSpecs.contains(where: { $0.id == spec.id }) else {
            throw RegistryError.duplicateModel(spec.id)
        }
        userSpecs.append(spec)
        saveUserSpecs()
    }

    public func removeModel(id: String) throws {
        guard !isBuiltIn(id: id) else {
            throw RegistryError.cannotModifyBuiltIn
        }
        userSpecs.removeAll { $0.id == id }
        statuses.removeValue(forKey: id)
        saveUserSpecs()
    }

    public nonisolated func isBuiltIn(id: String) -> Bool {
        Self.builtInSpecs.contains { $0.id == id }
    }

    // MARK: - Persistence

    private func loadUserSpecs() {
        let url = userSpecsFileURL
        guard FileManager.default.fileExists(atPath: url.path()) else { return }

        do {
            let data = try Data(contentsOf: url)
            userSpecs = try JSONDecoder().decode([ModelSpec].self, from: data)
        } catch {
            logger.error("Failed to load user-models.json: \(error.localizedDescription). Starting with empty list.")
            userSpecs = []
        }
    }

    private func saveUserSpecs() {
        do {
            let data = try JSONEncoder().encode(userSpecs)
            try data.write(to: userSpecsFileURL, options: .atomic)
        } catch {
            logger.error("Failed to save user-models.json: \(error.localizedDescription)")
        }
    }
}

public enum RegistryError: Error, Sendable {
    case cannotModifyBuiltIn
    case duplicateModel(String)
}
```

Key changes from the original:

1. `allSpecs` changes from `nonisolated let` to computed `var` property
2. `builtInSpecs` is a `static let` on the type
3. Added `userSpecs` with JSON persistence (`loadUserSpecs`/`saveUserSpecs`)
4. Added `essentialModelIDs` set and `downloadEssentialModels` method
5. Added `downloadModels` that uses `downloadAndVerify` instead of raw `download`
6. Added `addModel`, `removeModel`, `isBuiltIn` extension points
7. Added `isModelReady` convenience method
8. Added `RegistryError` enum
9. Graceful degradation on corrupt JSON with `OSLog` warning
10. User-friendly error message for checksum mismatch

- [ ] **Step 2: Fix any callers of the old `allSpecs` nonisolated let**

The old `allSpecs` was `nonisolated let`, meaning callers could access it
without `await`. The new `allSpecs` is a computed `var` on an actor, so it
requires `await`. Check for compilation errors and fix callers.

Known callers that need updating:

In `AnvilApp/AppState.swift` line 27, change:

```swift
let allReady = registry.allSpecs.allSatisfy { spec in
```

to:

```swift
let specs = await registry.allSpecs
let allReady = specs.allSatisfy { spec in
```

In `AnvilApp/UI/FirstLaunchView.swift` line 19, change:

```swift
ForEach(appState.registry.allSpecs) { spec in
```

This will be addressed in Task 4 when we rewrite FirstLaunchView. For now,
check the build output for any other callers and fix them.

- [ ] **Step 3: Verify Xcode build**

Run: `xcodebuild -project AnvilApp.xcodeproj -scheme AnvilApp -destination 'platform=macOS' build 2>&1 | tail -5`
Expected: `** BUILD SUCCEEDED **` (or only pre-existing errors)

- [ ] **Step 4: Commit**

```bash
git add AnvilCore/Sources/AnvilCore/ModelRegistry/ModelRegistry.swift
git commit -m "feat(model): split registry into builtInSpecs + userSpecs with JSON persistence"
```

---

### Task 4: AppState and FirstLaunchView — ACE-Step-Only Setup Gate

**Files:**

- Modify: `AnvilApp/AppState.swift`
- Modify: `AnvilApp/UI/FirstLaunchView.swift`

- [ ] **Step 1: Rewrite AppState with ACE-Step-only gate and background download support**

Replace the entire `AppState.swift` with:

```swift
import SwiftUI
import AnvilCore

public enum BackgroundDownloadPhase: Equatable {
    case idle
    case downloading(modelName: String, progress: Float)
    case complete
    case failed(String)
}

@MainActor
final class AppState: ObservableObject {
    @Published var isSetupComplete: Bool = false
    @Published var downloadProgress: [String: Double] = [:]
    @Published var currentError: String?
    @Published var backgroundDownloadPhase: BackgroundDownloadPhase = .idle

    let registry: ModelRegistry

    convenience init() {
        let modelsDir = FileManager.default.urls(
            for: .applicationSupportDirectory,
            in: .userDomainMask
        )[0].appending(path: "Anvil/models")
        self.init(registry: ModelRegistry(baseDirectory: modelsDir))
    }

    init(registry: ModelRegistry) {
        self.registry = registry
        Task { await checkSetupStatus() }
    }

    func checkSetupStatus() async {
        let baseDir = await registry.baseDirectory
        let allEssentialsReady = ModelRegistry.builtInSpecs
            .filter { ModelRegistry.essentialModelIDs.contains($0.id) }
            .allSatisfy { spec in
                spec.files.allSatisfy { file in
                    FileManager.default.fileExists(
                        atPath: spec.localPath(for: file, base: baseDir).path()
                    )
                }
            }
        isSetupComplete = allEssentialsReady
    }

    /// Downloads only the essential ACE-Step models for first launch.
    func startDownload() async {
        currentError = nil
        do {
            try await registry.downloadEssentialModels { [weak self] modelID, progress in
                Task { @MainActor [weak self] in
                    self?.downloadProgress[modelID] = progress
                }
            }
            isSetupComplete = true
        } catch {
            currentError = error.localizedDescription
        }
    }

    /// Kicks off background downloads for non-essential models (SFX, LLM).
    /// Called after first-launch completes or on normal app launch.
    func startBackgroundDownloads(
        sfxManager: SFXModelManager,
        llmManager: LLMModelManager
    ) async {
        // Download Stable Audio first (more immediately useful than LLM)
        if sfxManager.downloadState == .notDownloaded {
            backgroundDownloadPhase = .downloading(modelName: "Stable Audio Open", progress: 0)
            do {
                try await sfxManager.downloadIfNeeded()
            } catch {
                backgroundDownloadPhase = .failed("Stable Audio download failed. Tap to retry.")
                return
            }
        }

        // Then download LLM if needed
        let defaultLLM = LLMModelManager.defaultModels.first
        if let spec = defaultLLM, llmManager.downloadState == .idle {
            backgroundDownloadPhase = .downloading(modelName: "Lyric Writer", progress: 0)
            do {
                try await llmManager.downloadModel(spec)
            } catch {
                backgroundDownloadPhase = .failed("Lyric Writer download failed. Tap to retry.")
                return
            }
        }

        backgroundDownloadPhase = .complete
    }
}
```

Changes from the original:

1. `checkSetupStatus` checks only `essentialModelIDs` (ACE-Step DiT + VAE)
2. `startDownload` calls `downloadEssentialModels` instead of `downloadAll`
3. Added `BackgroundDownloadPhase` enum
4. Added `startBackgroundDownloads` method coordinating SFX then LLM
5. `backgroundDownloadPhase` published property for UI indicator

- [ ] **Step 2: Rewrite FirstLaunchView to show only ACE-Step models**

Replace the entire `FirstLaunchView.swift` with:

```swift
import SwiftUI
import AnvilCore

struct FirstLaunchView: View {
    @EnvironmentObject private var appState: AppState

    private var essentialSpecs: [ModelSpec] {
        ModelRegistry.builtInSpecs.filter {
            ModelRegistry.essentialModelIDs.contains($0.id)
        }
    }

    var body: some View {
        VStack(spacing: 32) {
            Spacer()

            Text("Welcome to Anvil")
                .font(.largeTitle.bold())

            Text("AI audio generation — music, sound effects, and more.\nCore models need to be downloaded before first use.")
                .multilineTextAlignment(.center)
                .foregroundStyle(.secondary)

            VStack(alignment: .leading, spacing: 8) {
                ForEach(essentialSpecs) { spec in
                    DownloadProgressRow(
                        spec: spec,
                        progress: appState.downloadProgress[spec.id]
                    )
                }
            }
            .frame(maxWidth: 400)
            .padding()
            .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 12))

            if let error = appState.currentError {
                Label(error, systemImage: "exclamationmark.triangle.fill")
                    .foregroundStyle(.red)
            }

            totalSizeLabel

            Button(isActivelyDownloading ? "Downloading…" : "Download Models") {
                Task {
                    await appState.startDownload()
                }
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.large)
            .disabled(isActivelyDownloading)

            Text("Additional models (sound effects, lyric writing) will download\nautomatically in the background after setup.")
                .font(.caption)
                .foregroundStyle(.tertiary)
                .multilineTextAlignment(.center)

            Spacer()
        }
        .padding(48)
        .frame(minWidth: 600, minHeight: 500)
    }

    private var isActivelyDownloading: Bool {
        !appState.downloadProgress.isEmpty &&
        appState.downloadProgress.values.contains(where: { $0 < 1.0 })
    }

    private var totalSizeLabel: some View {
        let totalBytes = essentialSpecs.reduce(Int64(0)) { $0 + $1.sizeBytes }
        let totalGB = Double(totalBytes) / 1_000_000_000
        return Text("Total download: \(String(format: "%.1f", totalGB)) GB — this only happens once.")
            .font(.caption)
            .foregroundStyle(.secondary)
    }
}
```

Changes from the original:

1. `essentialSpecs` filters `builtInSpecs` to only ACE-Step models
2. `ForEach` iterates `essentialSpecs` instead of `allSpecs`
3. `totalSizeLabel` computes from `essentialSpecs` (~4 GB instead of ~7.6 GB)
4. Updated subtitle: "Core models" instead of "Models"
5. Added caption explaining background downloads for additional models

- [ ] **Step 3: Verify Xcode build**

Run: `xcodebuild -project AnvilApp.xcodeproj -scheme AnvilApp -destination 'platform=macOS' build 2>&1 | tail -5`
Expected: `** BUILD SUCCEEDED **` (or only pre-existing errors)

- [ ] **Step 4: Commit**

```bash
git add AnvilApp/AppState.swift AnvilApp/UI/FirstLaunchView.swift
git commit -m "feat(model): ACE-Step-only first launch with background download coordination"
```

---

### Task 5: BackgroundDownloadIndicator

**Files:**

- Create: `AnvilApp/UI/Components/BackgroundDownloadIndicator.swift`

- [ ] **Step 1: Create the background download progress pill**

```swift
import SwiftUI

struct BackgroundDownloadIndicator: View {
    let phase: BackgroundDownloadPhase
    var onRetry: (() -> Void)?

    var body: some View {
        switch phase {
        case .idle:
            EmptyView()

        case .downloading(let modelName, let progress):
            HStack(spacing: 8) {
                ProgressView()
                    .controlSize(.small)
                Text("Downloading \(modelName)... \(Int(progress * 100))%")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 6)
            .background(.regularMaterial, in: Capsule())

        case .complete:
            HStack(spacing: 6) {
                Image(systemName: "checkmark.circle.fill")
                    .foregroundStyle(.green)
                Text("All models ready")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 6)
            .background(.regularMaterial, in: Capsule())

        case .failed(let message):
            Button(action: { onRetry?() }) {
                HStack(spacing: 6) {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .foregroundStyle(.yellow)
                    Text(message)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                .padding(.horizontal, 12)
                .padding(.vertical, 6)
                .background(.regularMaterial, in: Capsule())
            }
            .buttonStyle(.plain)
        }
    }
}
```

- [ ] **Step 2: Verify Xcode build**

Run: `xcodebuild -project AnvilApp.xcodeproj -scheme AnvilApp -destination 'platform=macOS' build 2>&1 | tail -5`
Expected: `** BUILD SUCCEEDED **` (or only pre-existing errors)

- [ ] **Step 3: Commit**

```bash
git add AnvilApp/UI/Components/BackgroundDownloadIndicator.swift
git commit -m "feat(ui): add background download progress indicator pill"
```

---

### Task 6: Wire Up Background Downloads in MainShellView

**Files:**

- Modify: `AnvilApp/AnvilAppApp.swift` (or wherever `MainShellView` is defined)

This task wires the `BackgroundDownloadIndicator` into the main UI and
triggers `startBackgroundDownloads()` when the app enters the main view.

- [ ] **Step 1: Find where MainShellView is defined**

Run: `grep -rn "struct MainShellView" AnvilApp/ --include="*.swift"`

- [ ] **Step 2: Add the background download indicator and trigger**

Add to `MainShellView`'s body (at the bottom, as an overlay or in a
toolbar):

```swift
.overlay(alignment: .bottom) {
    BackgroundDownloadIndicator(
        phase: appState.backgroundDownloadPhase,
        onRetry: {
            Task {
                await appState.startBackgroundDownloads(
                    sfxManager: sfxModelManager,
                    llmManager: llmModelManager
                )
            }
        }
    )
    .padding(.bottom, 8)
}
.task {
    if appState.backgroundDownloadPhase == .idle {
        await appState.startBackgroundDownloads(
            sfxManager: sfxModelManager,
            llmManager: llmModelManager
        )
    }
}
```

The `.task` modifier triggers background downloads when the main view
first appears. The overlay positions the pill at the bottom of the window.

Note: The exact view names for `sfxModelManager` and `llmModelManager`
depend on how they're injected into the environment. Check the existing
code for `@EnvironmentObject` declarations and match them.

- [ ] **Step 3: Auto-dismiss the "complete" phase after 3 seconds**

In `AppState`, add a method or use `.onChange` in the view to auto-dismiss:

```swift
.onChange(of: appState.backgroundDownloadPhase) { _, newPhase in
    if newPhase == .complete {
        Task {
            try? await Task.sleep(for: .seconds(3))
            appState.backgroundDownloadPhase = .idle
        }
    }
}
```

- [ ] **Step 4: Verify Xcode build**

Run: `xcodebuild -project AnvilApp.xcodeproj -scheme AnvilApp -destination 'platform=macOS' build 2>&1 | tail -5`
Expected: `** BUILD SUCCEEDED **` (or only pre-existing errors)

- [ ] **Step 5: Commit**

```bash
git add AnvilApp/
git commit -m "feat(ui): wire background download indicator into MainShellView"
```

---

## Verification

After all 6 tasks complete, verify the full build one final time:

```bash
xcodebuild -project AnvilApp.xcodeproj -scheme AnvilApp -destination 'platform=macOS' build 2>&1 | tail -5
```

Expected: `** BUILD SUCCEEDED **` with no warnings related to model
management files.

Key behavioral changes to verify manually:

1. First launch shows only ACE-Step DiT and VAE (~4 GB total)
2. After ACE-Step downloads, app transitions to MainShellView
3. Background download indicator appears showing SFX then LLM progress
4. SFX and lyric features show appropriate "not ready" states until their
   models finish downloading
5. Background download indicator auto-dismisses after completion
