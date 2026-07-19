import Foundation
import XCTest
@testable import PiMPAudioCore

final class NativeModelPackInstallerTests: XCTestCase {
    func testProductionCatalogPinsPublishedRepositories() {
        XCTAssertEqual(
            NativeModelPackCatalog.captioner.repositoryID,
            "DaRealDahoodie/pimp-audio-ace-captioner-4bit-v1"
        )
        XCTAssertEqual(
            NativeModelPackCatalog.captioner.revision,
            "e47dac2f8fd9a6a2bfe2b3f158a1850770fcc1ec"
        )
        XCTAssertEqual(
            NativeModelPackCatalog.transcriber.repositoryID,
            "DaRealDahoodie/pimp-audio-ace-transcriber-4bit-v1"
        )
        XCTAssertEqual(
            NativeModelPackCatalog.transcriber.revision,
            "528f8addaf9298e855bb4deb11c7cd30443ed775"
        )
    }

    func testInstallerDownloadsValidatesAndCommitsPack() async throws {
        let fixture = try makePackDirectory(task: .caption)
        let manifest = try NativeModelPackValidator.loadManifestMetadata(
            at: fixture
        )
        let modelsDirectory = try makeTemporaryDirectory()
        let downloader = FixtureNativeModelPackDownloader(
            sourceDirectory: fixture
        )
        let installer = NativeModelPackInstaller(
            modelsDirectory: modelsDirectory,
            downloader: downloader
        )
        let source = NativeModelPackCatalogEntry(
            task: .caption,
            displayName: "Fixture Captioner",
            repositoryID: "fixture/captioner",
            revision: "fixture-revision",
            packID: manifest.packID,
            destinationDirectoryName: "fixture-captioner",
            expectedSizeBytes: manifest.expectedSizeBytes
        )
        let progress = NativeModelPackProgressRecorder()

        let installedManifest = try await installer.install(source) { event in
            progress.append(event)
        }

        XCTAssertEqual(installedManifest, manifest)
        let destination = modelsDirectory.appendingPathComponent(
            source.destinationDirectoryName,
            isDirectory: true
        )
        XCTAssertEqual(
            try NativeModelPackValidator.loadAndValidate(at: destination),
            manifest
        )

        let requests = await downloader.requests()
        XCTAssertEqual(requests.count, manifest.files.count + 1)
        XCTAssertTrue(
            requests.allSatisfy {
                $0.repositoryID == source.repositoryID
                    && $0.revision == source.revision
            }
        )
        XCTAssertEqual(
            requests.first?.file,
            NativeModelPackManifest.filename
        )

        let events = progress.events()
        XCTAssertEqual(events.first?.stage, .preparing)
        XCTAssertTrue(events.contains { $0.stage == .verifying })
        XCTAssertTrue(events.contains { $0.stage == .installing })
        XCTAssertEqual(events.last?.stage, .complete)
        XCTAssertEqual(events.last?.fractionCompleted, 1)
        XCTAssertEqual(
            events.map(\.fractionCompleted),
            events.map(\.fractionCompleted).sorted()
        )
        XCTAssertFalse(
            try FileManager.default.contentsOfDirectory(
                atPath: modelsDirectory.path
            ).contains {
                $0.hasPrefix(".\(source.destinationDirectoryName).")
            }
        )
    }

    func testInstallerRejectsWrongTaskWithoutReplacingExistingPack() async throws {
        let fixture = try makePackDirectory(task: .caption)
        let manifest = try NativeModelPackValidator.loadManifestMetadata(
            at: fixture
        )
        let modelsDirectory = try makeTemporaryDirectory()
        let destinationName = "fixture-transcriber"
        let destination = modelsDirectory.appendingPathComponent(
            destinationName,
            isDirectory: true
        )
        try FileManager.default.createDirectory(
            at: destination,
            withIntermediateDirectories: true
        )
        let sentinel = destination.appendingPathComponent("keep-me.txt")
        try Data("existing".utf8).write(to: sentinel)

        let installer = NativeModelPackInstaller(
            modelsDirectory: modelsDirectory,
            downloader: FixtureNativeModelPackDownloader(
                sourceDirectory: fixture
            )
        )
        let wrongTaskSource = NativeModelPackCatalogEntry(
            task: .transcribe,
            displayName: "Wrong Task",
            repositoryID: "fixture/wrong-task",
            revision: "fixture-revision",
            packID: manifest.packID,
            destinationDirectoryName: destinationName,
            expectedSizeBytes: manifest.expectedSizeBytes
        )

        do {
            _ = try await installer.install(wrongTaskSource)
            XCTFail("Expected the installer to reject a caption pack.")
        } catch let error as NativeModelPackInstallError {
            guard case .taskMismatch = error else {
                return XCTFail("Unexpected installer error: \(error)")
            }
        }

        XCTAssertEqual(
            try String(contentsOf: sentinel, encoding: .utf8),
            "existing"
        )
        XCTAssertEqual(
            try FileManager.default.contentsOfDirectory(
                atPath: modelsDirectory.path
            ),
            [destinationName]
        )
    }

    private func makeTemporaryDirectory() throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(
            at: directory,
            withIntermediateDirectories: true
        )
        addTeardownBlock {
            try? FileManager.default.removeItem(at: directory)
        }
        return directory
    }

    private func makePackDirectory(
        task: OmniAnnotationTask
    ) throws -> URL {
        let directory = try makeTemporaryDirectory()
        let index: [String: Any] = [
            "weight_map": [
                "thinker.model.embed_tokens.weight":
                    "thinker.safetensors",
                "thinker.audio_tower.conv1.weight":
                    "audio-tower.safetensors",
            ],
        ]
        try JSONSerialization.data(
            withJSONObject: index,
            options: [.prettyPrinted, .sortedKeys]
        ).write(
            to: directory.appendingPathComponent(
                "model.safetensors.index.json"
            ),
            options: .atomic
        )

        for path in NativeModelPackValidator.requiredFiles.values
            where path != "model.safetensors.index.json"
        {
            let data: Data
            if path == "annotation-prompts.json" {
                data = try JSONSerialization.data(
                    withJSONObject: [
                        "schemaVersion": 1,
                        "caption": OmniAnnotationTask.caption.prompt,
                        "transcribe": OmniAnnotationTask.transcribe.prompt,
                    ],
                    options: [.prettyPrinted, .sortedKeys]
                )
            } else {
                data = Data(path.utf8)
            }
            try data.write(
                to: directory.appendingPathComponent(path),
                options: .atomic
            )
        }

        let files = try NativeModelPackValidator.requiredFiles
            .map { role, path in
                try NativeModelPackValidator.record(
                    role: role,
                    path: path,
                    in: directory
                )
            }
            .sorted { $0.role < $1.role }
        let modelID: String
        let revision: String
        let packID: String
        switch task {
        case .caption:
            modelID = "ACE-Step/acestep-captioner"
            revision = OmniWeightLoader.captionerRevision
            packID = "fixture-captioner"
        case .transcribe:
            modelID = "ACE-Step/acestep-transcriber"
            revision = OmniWeightLoader.transcriberRevision
            packID = "fixture-transcriber"
        }
        let manifest = NativeModelPackManifest(
            packID: packID,
            task: task,
            upstreamModelID: modelID,
            upstreamRevision: revision,
            upstreamLicense: "MIT",
            quantization: .init(bits: 4, groupSize: 64),
            tensorCounts: .init(thinker: 1, audioTower: 1),
            expectedSizeBytes: files.reduce(0) {
                $0 + $1.byteCount
            },
            excludedComponents: ["talker", "token2wav", "vision"],
            files: files
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(manifest).write(
            to: directory.appendingPathComponent(
                NativeModelPackManifest.filename
            ),
            options: .atomic
        )
        return directory
    }
}

private struct NativeModelPackDownloadRequest: Sendable {
    let repositoryID: String
    let revision: String
    let file: String
}

private actor FixtureNativeModelPackDownloader:
    NativeModelPackDownloading
{
    private let sourceDirectory: URL
    private var capturedRequests: [NativeModelPackDownloadRequest] = []

    init(sourceDirectory: URL) {
        self.sourceDirectory = sourceDirectory
    }

    func downloadAndVerify(
        repo: String,
        revision: String,
        file: String,
        to destination: URL,
        expectedHash: String?,
        bearerToken: String?,
        onProgress: @Sendable @escaping (Double) -> Void,
        onRetry: @Sendable @escaping () -> Void
    ) async throws {
        capturedRequests.append(
            NativeModelPackDownloadRequest(
                repositoryID: repo,
                revision: revision,
                file: file
            )
        )
        onProgress(0)
        try FileManager.default.createDirectory(
            at: destination.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try FileManager.default.copyItem(
            at: sourceDirectory.appendingPathComponent(file),
            to: destination
        )
        if let expectedHash {
            let actualHash = try NativeModelPackValidator.sha256(
                of: destination
            )
            guard actualHash == expectedHash else {
                throw DownloadError.checksumMismatch(file: file)
            }
        }
        onProgress(1)
    }

    func requests() -> [NativeModelPackDownloadRequest] {
        capturedRequests
    }
}

private final class NativeModelPackProgressRecorder:
    @unchecked Sendable
{
    private let lock = NSLock()
    private var capturedEvents: [NativeModelPackInstallProgress] = []

    func append(_ event: NativeModelPackInstallProgress) {
        lock.lock()
        capturedEvents.append(event)
        lock.unlock()
    }

    func events() -> [NativeModelPackInstallProgress] {
        lock.lock()
        defer { lock.unlock() }
        return capturedEvents
    }
}
