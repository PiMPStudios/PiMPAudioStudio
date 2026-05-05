# AnvilApp Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the AnvilApp project skeleton with a working model download pipeline and first-launch UI — the foundation everything else builds on.

**Architecture:** AnvilCore is a local Swift package containing all non-UI logic (ModelRegistry, download client, verification). The app target is an Xcode project generated via xcodegen that depends on AnvilCore. First-launch flow downloads model weight stubs (real weights in later phases) and transitions to a basic three-panel shell.

**Tech Stack:** Swift 5.10+, SwiftUI (macOS 14+), CryptoKit (SHA256), URLSession (async downloads with progress), Swift Testing framework, xcodegen

---

## File Structure

```
AnvilApp/
├── project.yml                          ← xcodegen spec
├── AnvilCore/
│   ├── Package.swift
│   ├── Sources/
│   │   └── AnvilCore/
│   │       ├── ModelRegistry/
│   │       │   ├── ModelSpec.swift
│   │       │   ├── ModelRegistry.swift
│   │       │   └── DownloadClient.swift
│   │       └── AnvilCore.swift          ← public re-exports
│   └── Tests/
│       └── AnvilCoreTests/
│           ├── ModelSpecTests.swift
│           ├── ModelRegistryTests.swift
│           └── DownloadClientTests.swift
├── AnvilApp/
│   ├── AnvilAppApp.swift                ← @main entry point
│   ├── AppState.swift                   ← ObservableObject root state
│   ├── UI/
│   │   ├── FirstLaunchView.swift
│   │   ├── MainShellView.swift
│   │   └── Components/
│   │       └── DownloadProgressRow.swift
│   └── Assets.xcassets/
└── AnvilAppTests/
    └── AppStateTests.swift
```

---

### Task 1: Create AnvilCore Swift Package

**Files:**
- Create: `AnvilCore/Package.swift`
- Create: `AnvilCore/Sources/AnvilCore/AnvilCore.swift`
- Create: `AnvilCore/Tests/AnvilCoreTests/AnvilCoreTests.swift`

- [ ] **Step 1: Initialize the package**

```bash
cd ~/Developer/GitRepos/AnvilApp
mkdir -p AnvilCore
cd AnvilCore
swift package init --name AnvilCore --type library
```

- [ ] **Step 2: Set the platform and Swift version in Package.swift**

Replace the generated `Package.swift` with:

```swift
// swift-tools-version: 5.10
import PackageDescription

let package = Package(
    name: "AnvilCore",
    platforms: [.macOS(.v14)],
    products: [
        .library(name: "AnvilCore", targets: ["AnvilCore"]),
    ],
    targets: [
        .target(name: "AnvilCore"),
        .testTarget(name: "AnvilCoreTests", dependencies: ["AnvilCore"]),
    ]
)
```

- [ ] **Step 3: Create the public entry point**

`AnvilCore/Sources/AnvilCore/AnvilCore.swift`:

```swift
public enum AnvilCore {
    public static let version = "0.1.0"
}
```

- [ ] **Step 4: Write a smoke test**

`AnvilCore/Tests/AnvilCoreTests/AnvilCoreTests.swift`:

```swift
import Testing
@testable import AnvilCore

@Test func versionExists() {
    #expect(AnvilCore.version == "0.1.0")
}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd ~/Developer/GitRepos/AnvilApp/AnvilCore && swift test`
Expected: All tests passed

- [ ] **Step 6: Commit**

```bash
cd ~/Developer/GitRepos/AnvilApp
git add AnvilCore/
git commit -m "feat: initialize AnvilCore Swift package"
```

---

### Task 2: Define ModelSpec

**Files:**
- Create: `AnvilCore/Sources/AnvilCore/ModelRegistry/ModelSpec.swift`
- Create: `AnvilCore/Tests/AnvilCoreTests/ModelSpecTests.swift`

- [ ] **Step 1: Write the failing test**

`AnvilCore/Tests/AnvilCoreTests/ModelSpecTests.swift`:

```swift
import Testing
@testable import AnvilCore

@Test func modelSpecStoresProperties() {
    let spec = ModelSpec(
        id: "ace-step-dit",
        huggingFaceRepo: "ACEStep/ACE-Step-v1-5-DiT",
        files: ["model.safetensors", "config.json"],
        sizeBytes: 3_200_000_000,
        sha256: ["model.safetensors": "abc123", "config.json": "def456"]
    )

    #expect(spec.id == "ace-step-dit")
    #expect(spec.files.count == 2)
    #expect(spec.sha256["model.safetensors"] == "abc123")
}

@Test func modelSpecTotalSizeFormatted() {
    let spec = ModelSpec(
        id: "test",
        huggingFaceRepo: "test/repo",
        files: ["a.bin"],
        sizeBytes: 3_200_000_000,
        sha256: [:]
    )

    #expect(spec.totalSizeFormatted == "3.2 GB")
}

@Test func modelSpecLocalDirectory() {
    let spec = ModelSpec(
        id: "ace-step-dit",
        huggingFaceRepo: "ACEStep/ACE-Step-v1-5-DiT",
        files: [],
        sizeBytes: 0,
        sha256: [:]
    )

    let baseURL = URL(filePath: "/tmp/models")
    #expect(spec.localDirectory(base: baseURL).path().contains("ace-step-dit"))
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/Developer/GitRepos/AnvilApp/AnvilCore && swift test`
Expected: FAIL — `ModelSpec` not defined

- [ ] **Step 3: Write minimal implementation**

`AnvilCore/Sources/AnvilCore/ModelRegistry/ModelSpec.swift`:

```swift
import Foundation

public struct ModelSpec: Sendable, Codable, Identifiable {
    public let id: String
    public let huggingFaceRepo: String
    public let files: [String]
    public let sizeBytes: Int64
    public let sha256: [String: String]

    public init(
        id: String,
        huggingFaceRepo: String,
        files: [String],
        sizeBytes: Int64,
        sha256: [String: String]
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

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/Developer/GitRepos/AnvilApp/AnvilCore && swift test`
Expected: All tests passed

- [ ] **Step 5: Commit**

```bash
cd ~/Developer/GitRepos/AnvilApp
git add AnvilCore/Sources/AnvilCore/ModelRegistry/ModelSpec.swift \
        AnvilCore/Tests/AnvilCoreTests/ModelSpecTests.swift
git commit -m "feat: add ModelSpec type for model weight metadata"
```

---

### Task 3: Implement DownloadClient

**Files:**
- Create: `AnvilCore/Sources/AnvilCore/ModelRegistry/DownloadClient.swift`
- Create: `AnvilCore/Tests/AnvilCoreTests/DownloadClientTests.swift`

- [ ] **Step 1: Write the failing test**

`AnvilCore/Tests/AnvilCoreTests/DownloadClientTests.swift`:

```swift
import Testing
import Foundation
@testable import AnvilCore

@Test func downloadClientBuildsCorrectURL() {
    let client = DownloadClient()
    let url = client.huggingFaceURL(repo: "ACEStep/ACE-Step-v1-5-DiT", file: "model.safetensors")

    #expect(url.absoluteString == "https://huggingface.co/ACEStep/ACE-Step-v1-5-DiT/resolve/main/model.safetensors")
}

@Test func downloadClientVerifiesSHA256() throws {
    let client = DownloadClient()
    let tempDir = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
    try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: tempDir) }

    let testData = Data("hello world".utf8)
    let testFile = tempDir.appending(path: "test.bin")
    try testData.write(to: testFile)

    // SHA256 of "hello world"
    let expectedHash = "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
    let result = client.verifySHA256(file: testFile, expected: expectedHash)
    #expect(result == true)

    let badResult = client.verifySHA256(file: testFile, expected: "0000")
    #expect(badResult == false)
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/Developer/GitRepos/AnvilApp/AnvilCore && swift test`
Expected: FAIL — `DownloadClient` not defined

- [ ] **Step 3: Write minimal implementation**

`AnvilCore/Sources/AnvilCore/ModelRegistry/DownloadClient.swift`:

```swift
import Foundation
import CryptoKit

public actor DownloadClient {
    public init() {}

    public func huggingFaceURL(repo: String, file: String) -> URL {
        URL(string: "https://huggingface.co/\(repo)/resolve/main/\(file)")!
    }

    public func download(
        repo: String,
        file: String,
        to destination: URL,
        onProgress: @Sendable @escaping (Double) -> Void
    ) async throws {
        let url = huggingFaceURL(repo: repo, file: file)
        let (tempURL, response) = try await URLSession.shared.download(for: URLRequest(url: url), delegate: ProgressDelegate(handler: onProgress))

        guard let httpResponse = response as? HTTPURLResponse,
              httpResponse.statusCode == 200 else {
            throw DownloadError.httpError(statusCode: (response as? HTTPURLResponse)?.statusCode ?? -1)
        }

        let destDir = destination.deletingLastPathComponent()
        try FileManager.default.createDirectory(at: destDir, withIntermediateDirectories: true)

        if FileManager.default.fileExists(atPath: destination.path()) {
            try FileManager.default.removeItem(at: destination)
        }
        try FileManager.default.moveItem(at: tempURL, to: destination)
    }

    public nonisolated func verifySHA256(file: URL, expected: String) -> Bool {
        guard let data = try? Data(contentsOf: file) else { return false }
        let digest = SHA256.hash(data: data)
        let hex = digest.map { String(format: "%02x", $0) }.joined()
        return hex == expected.lowercased()
    }
}

public enum DownloadError: Error, Sendable {
    case httpError(statusCode: Int)
    case checksumMismatch(file: String)
}

private final class ProgressDelegate: NSObject, URLSessionTaskDelegate, Sendable {
    private let handler: @Sendable (Double) -> Void

    init(handler: @Sendable @escaping (Double) -> Void) {
        self.handler = handler
    }

    func urlSession(
        _ session: URLSession,
        didCreateTask task: URLSessionTask
    ) {
        task.progress.observe(\.fractionCompleted) { [handler] progress, _ in
            handler(progress.fractionCompleted)
        }
    }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/Developer/GitRepos/AnvilApp/AnvilCore && swift test`
Expected: All tests passed

- [ ] **Step 5: Commit**

```bash
cd ~/Developer/GitRepos/AnvilApp
git add AnvilCore/Sources/AnvilCore/ModelRegistry/DownloadClient.swift \
        AnvilCore/Tests/AnvilCoreTests/DownloadClientTests.swift
git commit -m "feat: add DownloadClient with HuggingFace URL building and SHA256 verification"
```

---

### Task 4: Implement ModelRegistry

**Files:**
- Create: `AnvilCore/Sources/AnvilCore/ModelRegistry/ModelRegistry.swift`
- Create: `AnvilCore/Tests/AnvilCoreTests/ModelRegistryTests.swift`

- [ ] **Step 1: Write the failing test**

`AnvilCore/Tests/AnvilCoreTests/ModelRegistryTests.swift`:

```swift
import Testing
import Foundation
@testable import AnvilCore

@Test func registryReturnsAllSpecs() async {
    let registry = ModelRegistry(baseDirectory: FileManager.default.temporaryDirectory)
    let specs = registry.allSpecs

    #expect(specs.isEmpty == false)
    #expect(specs.contains(where: { $0.id == "ace-step-dit" }))
    #expect(specs.contains(where: { $0.id == "ace-step-vae" }))
    #expect(specs.contains(where: { $0.id == "lyric-llm" }))
}

@Test func registryDetectsModelNotDownloaded() async {
    let tempDir = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
    let registry = ModelRegistry(baseDirectory: tempDir)

    let status = await registry.status(for: "ace-step-dit")
    #expect(status == .notDownloaded)
}

@Test func registryDetectsModelReady() async throws {
    let tempDir = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
    try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: tempDir) }

    let registry = ModelRegistry(baseDirectory: tempDir)
    let spec = registry.allSpecs.first(where: { $0.id == "ace-step-dit" })!

    let modelDir = spec.localDirectory(base: tempDir)
    try FileManager.default.createDirectory(at: modelDir, withIntermediateDirectories: true)
    for file in spec.files {
        try Data("stub".utf8).write(to: spec.localPath(for: file, base: tempDir))
    }

    let status = await registry.status(for: "ace-step-dit")
    #expect(status == .ready)
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/Developer/GitRepos/AnvilApp/AnvilCore && swift test`
Expected: FAIL — `ModelRegistry` not defined

- [ ] **Step 3: Write minimal implementation**

`AnvilCore/Sources/AnvilCore/ModelRegistry/ModelRegistry.swift`:

```swift
import Foundation

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

    public nonisolated let allSpecs: [ModelSpec] = [
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

    private var statuses: [String: ModelStatus] = [:]

    public init(baseDirectory: URL, client: DownloadClient = DownloadClient()) {
        self.baseDirectory = baseDirectory
        self.client = client
    }

    public func status(for modelID: String) -> ModelStatus {
        if let cached = statuses[modelID] {
            return cached
        }

        guard let spec = allSpecs.first(where: { $0.id == modelID }) else {
            return .failed("Unknown model: \(modelID)")
        }

        let allFilesExist = spec.files.allSatisfy { file in
            FileManager.default.fileExists(atPath: spec.localPath(for: file, base: baseDirectory).path())
        }

        return allFilesExist ? .ready : .notDownloaded
    }

    public func downloadAll(onProgress: @Sendable @escaping (String, Double) -> Void) async throws {
        for spec in allSpecs {
            guard status(for: spec.id) != .ready else { continue }
            statuses[spec.id] = .downloading(progress: 0)

            for file in spec.files {
                let destination = spec.localPath(for: file, base: baseDirectory)
                try await client.download(
                    repo: spec.huggingFaceRepo,
                    file: file,
                    to: destination,
                    onProgress: { progress in
                        onProgress(spec.id, progress)
                    }
                )

                if let expectedHash = spec.sha256[file], !expectedHash.isEmpty {
                    statuses[spec.id] = .verifying
                    guard client.verifySHA256(file: destination, expected: expectedHash) else {
                        statuses[spec.id] = .failed("Checksum mismatch: \(file)")
                        throw DownloadError.checksumMismatch(file: file)
                    }
                }
            }

            statuses[spec.id] = .ready
        }
    }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/Developer/GitRepos/AnvilApp/AnvilCore && swift test`
Expected: All tests passed

- [ ] **Step 5: Commit**

```bash
cd ~/Developer/GitRepos/AnvilApp
git add AnvilCore/Sources/AnvilCore/ModelRegistry/ModelRegistry.swift \
        AnvilCore/Tests/AnvilCoreTests/ModelRegistryTests.swift
git commit -m "feat: add ModelRegistry with status tracking and download orchestration"
```

---

### Task 5: Set Up Xcode Project via xcodegen

**Files:**
- Create: `project.yml`
- Create: `AnvilApp/AnvilAppApp.swift`
- Create: `AnvilApp/AppState.swift`

- [ ] **Step 1: Install xcodegen if not present**

```bash
brew list xcodegen || brew install xcodegen
```

- [ ] **Step 2: Write project.yml**

`~/Developer/GitRepos/AnvilApp/project.yml`:

```yaml
name: AnvilApp
options:
  bundleIdPrefix: net.pimp.anvil
  deploymentTarget:
    macOS: "14.0"
  xcodeVersion: "16.0"
  minimumXcodeGenVersion: "2.38"

packages:
  AnvilCore:
    path: ./AnvilCore

targets:
  AnvilApp:
    type: application
    platform: macOS
    sources: [AnvilApp]
    dependencies:
      - package: AnvilCore
    settings:
      base:
        PRODUCT_BUNDLE_IDENTIFIER: net.pimp.anvil.AnvilApp
        MARKETING_VERSION: "0.1.0"
        CURRENT_PROJECT_VERSION: 1
        SWIFT_VERSION: "5.10"
        MACOSX_DEPLOYMENT_TARGET: "14.0"
        INFOPLIST_KEY_LSApplicationCategoryType: "public.app-category.music"
    entitlements:
      path: AnvilApp/AnvilApp.entitlements
      properties:
        com.apple.security.app-sandbox: true
        com.apple.security.network.client: true
        com.apple.security.files.user-selected.read-write: true

  AnvilAppTests:
    type: bundle.unit-test
    platform: macOS
    sources: [AnvilAppTests]
    dependencies:
      - target: AnvilApp
    settings:
      base:
        SWIFT_VERSION: "5.10"
```

- [ ] **Step 3: Create the app entry point**

`~/Developer/GitRepos/AnvilApp/AnvilApp/AnvilAppApp.swift`:

```swift
import SwiftUI
import AnvilCore

@main
struct AnvilAppApp: App {
    @StateObject private var appState = AppState()

    var body: some Scene {
        WindowGroup {
            if appState.isSetupComplete {
                MainShellView()
                    .environmentObject(appState)
            } else {
                FirstLaunchView()
                    .environmentObject(appState)
            }
        }
        .windowStyle(.titleBar)
        .defaultSize(width: 1200, height: 800)

        Settings {
            Text("Settings coming soon")
                .frame(width: 400, height: 300)
        }
    }
}
```

- [ ] **Step 4: Create AppState**

`~/Developer/GitRepos/AnvilApp/AnvilApp/AppState.swift`:

```swift
import SwiftUI
import AnvilCore

@MainActor
final class AppState: ObservableObject {
    @Published var isSetupComplete: Bool = false
    @Published var downloadProgress: [String: Double] = [:]
    @Published var currentError: String?

    let registry: ModelRegistry

    init() {
        let modelsDir = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appending(path: "Anvil/models")
        self.registry = ModelRegistry(baseDirectory: modelsDir)

        Task { await checkSetupStatus() }
    }

    func checkSetupStatus() async {
        let allReady = registry.allSpecs.allSatisfy { spec in
            // Sync check via file existence
            spec.files.allSatisfy { file in
                FileManager.default.fileExists(
                    atPath: spec.localPath(for: file, base: registry.baseDirectory).path()
                )
            }
        }
        isSetupComplete = allReady
    }

    func startDownload() async {
        currentError = nil
        do {
            try await registry.downloadAll { [weak self] modelID, progress in
                Task { @MainActor in
                    self?.downloadProgress[modelID] = progress
                }
            }
            isSetupComplete = true
        } catch {
            currentError = error.localizedDescription
        }
    }
}
```

- [ ] **Step 5: Create entitlements file**

`~/Developer/GitRepos/AnvilApp/AnvilApp/AnvilApp.entitlements`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>com.apple.security.app-sandbox</key>
    <true/>
    <key>com.apple.security.network.client</key>
    <true/>
    <key>com.apple.security.files.user-selected.read-write</key>
    <true/>
</dict>
</plist>
```

- [ ] **Step 6: Generate the Xcode project**

```bash
cd ~/Developer/GitRepos/AnvilApp
xcodegen generate
```

Expected: `Generated AnvilApp.xcodeproj`

- [ ] **Step 7: Verify it builds**

```bash
cd ~/Developer/GitRepos/AnvilApp
xcodebuild -project AnvilApp.xcodeproj -scheme AnvilApp -destination "platform=macOS" build 2>&1 | tail -5
```

Expected: `BUILD SUCCEEDED`

- [ ] **Step 8: Commit**

```bash
cd ~/Developer/GitRepos/AnvilApp
echo "AnvilApp.xcodeproj/xcuserdata/" >> .gitignore
echo ".build/" >> .gitignore
echo "DerivedData/" >> .gitignore
git add project.yml AnvilApp/ AnvilAppTests/ .gitignore
git add AnvilApp.xcodeproj/project.pbxproj AnvilApp.xcodeproj/project.xcworkspace/
git commit -m "feat: set up AnvilApp Xcode project with SwiftUI shell"
```

---

### Task 6: First-Launch Download UI

**Files:**
- Create: `AnvilApp/UI/FirstLaunchView.swift`
- Create: `AnvilApp/UI/Components/DownloadProgressRow.swift`

- [ ] **Step 1: Create DownloadProgressRow component**

`~/Developer/GitRepos/AnvilApp/AnvilApp/UI/Components/DownloadProgressRow.swift`:

```swift
import SwiftUI
import AnvilCore

struct DownloadProgressRow: View {
    let spec: ModelSpec
    let progress: Double?

    var body: some View {
        HStack {
            VStack(alignment: .leading, spacing: 2) {
                Text(spec.id)
                    .font(.body.monospaced())
                Text(spec.totalSizeFormatted)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Spacer()

            if let progress {
                if progress >= 1.0 {
                    Image(systemName: "checkmark.circle.fill")
                        .foregroundStyle(.green)
                } else {
                    ProgressView(value: progress)
                        .frame(width: 120)
                }
            } else {
                Text("Waiting…")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .padding(.vertical, 4)
    }
}
```

- [ ] **Step 2: Create FirstLaunchView**

`~/Developer/GitRepos/AnvilApp/AnvilApp/UI/FirstLaunchView.swift`:

```swift
import SwiftUI
import AnvilCore

struct FirstLaunchView: View {
    @EnvironmentObject private var appState: AppState
    @State private var isDownloading = false

    var body: some View {
        VStack(spacing: 32) {
            Spacer()

            Text("Welcome to Anvil")
                .font(.largeTitle.bold())

            Text("AI audio generation — music, sound effects, and more.\nModels need to be downloaded before first use.")
                .multilineTextAlignment(.center)
                .foregroundStyle(.secondary)

            VStack(alignment: .leading, spacing: 8) {
                ForEach(appState.registry.allSpecs) { spec in
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

            Button(isDownloading ? "Downloading…" : "Download Models") {
                isDownloading = true
                Task {
                    await appState.startDownload()
                    isDownloading = false
                }
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.large)
            .disabled(isDownloading)

            Spacer()
        }
        .padding(48)
        .frame(minWidth: 600, minHeight: 500)
    }

    private var totalSizeLabel: some View {
        let totalBytes = appState.registry.allSpecs.reduce(Int64(0)) { $0 + $1.sizeBytes }
        let totalGB = Double(totalBytes) / 1_000_000_000
        return Text("Total download: \(String(format: "%.1f", totalGB)) GB — this only happens once.")
            .font(.caption)
            .foregroundStyle(.secondary)
    }
}
```

- [ ] **Step 3: Verify build**

```bash
cd ~/Developer/GitRepos/AnvilApp
xcodegen generate
xcodebuild -project AnvilApp.xcodeproj -scheme AnvilApp -destination "platform=macOS" build 2>&1 | tail -5
```

Expected: `BUILD SUCCEEDED`

- [ ] **Step 4: Commit**

```bash
cd ~/Developer/GitRepos/AnvilApp
git add AnvilApp/UI/
git commit -m "feat: add first-launch download UI with progress tracking"
```

---

### Task 7: Main Shell View (Three-Panel Layout)

**Files:**
- Create: `AnvilApp/UI/MainShellView.swift`

- [ ] **Step 1: Create the three-panel shell**

`~/Developer/GitRepos/AnvilApp/AnvilApp/UI/MainShellView.swift`:

```swift
import SwiftUI

struct MainShellView: View {
    @EnvironmentObject private var appState: AppState
    @State private var selectedSidebarItem: SidebarItem = .generate

    var body: some View {
        NavigationSplitView {
            sidebar
        } content: {
            canvas
        } detail: {
            inspector
        }
        .navigationSplitViewStyle(.balanced)
    }

    private var sidebar: some View {
        List(selection: $selectedSidebarItem) {
            Section("Create") {
                Label("Generate", systemImage: "waveform")
                    .tag(SidebarItem.generate)
            }
            Section("Library") {
                Label("History", systemImage: "clock")
                    .tag(SidebarItem.history)
                Label("Presets", systemImage: "slider.horizontal.3")
                    .tag(SidebarItem.presets)
            }
        }
        .listStyle(.sidebar)
        .navigationTitle("Anvil")
    }

    private var canvas: some View {
        VStack(spacing: 16) {
            Picker("Mode", selection: .constant("auto")) {
                Text("Auto").tag("auto")
                Text("Music").tag("music")
                Text("Sound FX").tag("fx")
                Text("Ambient").tag("ambient")
            }
            .pickerStyle(.segmented)
            .padding(.horizontal)

            TextEditor(text: .constant(""))
                .font(.body)
                .frame(minHeight: 100)
                .overlay(
                    RoundedRectangle(cornerRadius: 8)
                        .stroke(.quaternary)
                )
                .padding(.horizontal)

            Spacer()

            HStack {
                Button("Write Lyrics") {}
                    .buttonStyle(.bordered)
                Button("Generate") {}
                    .buttonStyle(.borderedProminent)
            }
            .padding()
        }
        .padding(.top)
    }

    private var inspector: some View {
        Form {
            Section("Parameters") {
                LabeledContent("BPM") { TextField("120", text: .constant("")) }
                LabeledContent("Key") { TextField("C Major", text: .constant("")) }
                LabeledContent("Duration") { TextField("30s", text: .constant("")) }
                LabeledContent("Seed") { TextField("Random", text: .constant("")) }
            }
            Section("Model") {
                Text("ACE-Step DiT v1.5")
                    .foregroundStyle(.secondary)
            }
        }
        .formStyle(.grouped)
        .frame(minWidth: 250)
    }
}

enum SidebarItem: Hashable {
    case generate
    case history
    case presets
}
```

- [ ] **Step 2: Verify build**

```bash
cd ~/Developer/GitRepos/AnvilApp
xcodegen generate
xcodebuild -project AnvilApp.xcodeproj -scheme AnvilApp -destination "platform=macOS" build 2>&1 | tail -5
```

Expected: `BUILD SUCCEEDED`

- [ ] **Step 3: Commit**

```bash
cd ~/Developer/GitRepos/AnvilApp
git add AnvilApp/UI/MainShellView.swift
git commit -m "feat: add three-panel main shell with sidebar, canvas, and inspector"
```

---

### Task 8: AppState Tests

**Files:**
- Create: `AnvilAppTests/AppStateTests.swift`

- [ ] **Step 1: Write tests for AppState setup detection**

`~/Developer/GitRepos/AnvilApp/AnvilAppTests/AppStateTests.swift`:

```swift
import Testing
import Foundation
@testable import AnvilApp
@testable import AnvilCore

@Test @MainActor func appStateDetectsIncompleteSetup() async {
    let tempDir = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
    let registry = ModelRegistry(baseDirectory: tempDir)
    let state = AppState(registry: registry)

    await state.checkSetupStatus()
    #expect(state.isSetupComplete == false)
}

@Test @MainActor func appStateDetectsCompleteSetup() async throws {
    let tempDir = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
    try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: tempDir) }

    let registry = ModelRegistry(baseDirectory: tempDir)
    for spec in registry.allSpecs {
        let dir = spec.localDirectory(base: tempDir)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        for file in spec.files {
            try Data("stub".utf8).write(to: spec.localPath(for: file, base: tempDir))
        }
    }

    let state = AppState(registry: registry)
    await state.checkSetupStatus()
    #expect(state.isSetupComplete == true)
}
```

- [ ] **Step 2: Update AppState to accept injected registry (for testing)**

Modify `AnvilApp/AppState.swift` — add an `init(registry:)` overload:

```swift
import SwiftUI
import AnvilCore

@MainActor
final class AppState: ObservableObject {
    @Published var isSetupComplete: Bool = false
    @Published var downloadProgress: [String: Double] = [:]
    @Published var currentError: String?

    let registry: ModelRegistry

    convenience init() {
        let modelsDir = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appending(path: "Anvil/models")
        self.init(registry: ModelRegistry(baseDirectory: modelsDir))
    }

    init(registry: ModelRegistry) {
        self.registry = registry
        Task { await checkSetupStatus() }
    }

    func checkSetupStatus() async {
        let allReady = registry.allSpecs.allSatisfy { spec in
            spec.files.allSatisfy { file in
                FileManager.default.fileExists(
                    atPath: spec.localPath(for: file, base: registry.baseDirectory).path()
                )
            }
        }
        isSetupComplete = allReady
    }

    func startDownload() async {
        currentError = nil
        do {
            try await registry.downloadAll { [weak self] modelID, progress in
                Task { @MainActor in
                    self?.downloadProgress[modelID] = progress
                }
            }
            isSetupComplete = true
        } catch {
            currentError = error.localizedDescription
        }
    }
}
```

- [ ] **Step 3: Run tests**

```bash
cd ~/Developer/GitRepos/AnvilApp
xcodegen generate
xcodebuild -project AnvilApp.xcodeproj -scheme AnvilAppTests -destination "platform=macOS" test 2>&1 | tail -10
```

Expected: All tests passed

- [ ] **Step 4: Commit**

```bash
cd ~/Developer/GitRepos/AnvilApp
git add AnvilAppTests/AppStateTests.swift AnvilApp/AppState.swift
git commit -m "test: add AppState tests with injectable ModelRegistry"
```

---

### Task 9: Add .gitignore and Clean Up

**Files:**
- Modify: `.gitignore`
- Create: `AnvilCore/Sources/AnvilCore/AnvilCore.swift` (update re-exports)

- [ ] **Step 1: Finalize .gitignore**

`~/Developer/GitRepos/AnvilApp/.gitignore`:

```
# Xcode
AnvilApp.xcodeproj/xcuserdata/
AnvilApp.xcodeproj/project.xcworkspace/xcuserdata/
DerivedData/

# Swift Package Manager
.build/
.swiftpm/

# macOS
.DS_Store

# Models (never commit weights)
*.safetensors
*.bin
*.gguf
```

- [ ] **Step 2: Update AnvilCore public re-exports**

`AnvilCore/Sources/AnvilCore/AnvilCore.swift`:

```swift
// All public types are already exported by being `public` in their own files.
// This file exists only for top-level module metadata.

public enum AnvilCoreInfo {
    public static let version = "0.1.0"
}
```

- [ ] **Step 3: Final build check**

```bash
cd ~/Developer/GitRepos/AnvilApp/AnvilCore && swift test
cd ~/Developer/GitRepos/AnvilApp && xcodegen generate
xcodebuild -project AnvilApp.xcodeproj -scheme AnvilApp -destination "platform=macOS" build 2>&1 | tail -5
```

Expected: All tests pass, build succeeds

- [ ] **Step 4: Commit and push**

```bash
cd ~/Developer/GitRepos/AnvilApp
git add .gitignore AnvilCore/Sources/AnvilCore/AnvilCore.swift
git commit -m "chore: finalize gitignore and AnvilCore public API exports"
git push origin main
```

---

## Summary

After completing all 9 tasks:
- AnvilCore package with `ModelSpec`, `DownloadClient`, `ModelRegistry` — fully tested
- AnvilApp Xcode project targeting macOS 14+ with sandbox entitlements
- First-launch download flow with per-model progress UI
- Three-panel main shell (sidebar, canvas, inspector) as placeholder
- AppState managing the setup/ready lifecycle transition
- All builds succeed, all tests pass, App Store sandbox entitlements in place

**Next plan:** Phase 2 — Core Inference (VAEModel, DiffusionSampler, DITModel, MusicPipeline)
