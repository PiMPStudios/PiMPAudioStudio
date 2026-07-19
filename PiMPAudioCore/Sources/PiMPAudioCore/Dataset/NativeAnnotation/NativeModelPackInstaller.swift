import Foundation

public struct NativeModelPackCatalogEntry: Equatable, Sendable {
    public let task: DatasetAnnotationTask
    public let displayName: String
    public let repositoryID: String
    public let revision: String
    public let packID: String
    public let destinationDirectoryName: String
    public let expectedSizeBytes: Int64

    public init(
        task: DatasetAnnotationTask,
        displayName: String,
        repositoryID: String,
        revision: String,
        packID: String,
        destinationDirectoryName: String,
        expectedSizeBytes: Int64
    ) {
        self.task = task
        self.displayName = displayName
        self.repositoryID = repositoryID
        self.revision = revision
        self.packID = packID
        self.destinationDirectoryName = destinationDirectoryName
        self.expectedSizeBytes = expectedSizeBytes
    }

    public var modelPageURL: URL {
        guard let url = URL(
            string: "https://huggingface.co/\(repositoryID)"
        ) else {
            preconditionFailure(
                "Invalid Hugging Face repository ID: \(repositoryID)"
            )
        }
        return url
    }
}

public enum NativeModelPackCatalog {
    public static let captioner = NativeModelPackCatalogEntry(
        task: .caption,
        displayName: "ACE Captioner 4-bit",
        repositoryID:
            "DaRealDahoodie/pimp-audio-ace-captioner-4bit-v1",
        revision:
            "e47dac2f8fd9a6a2bfe2b3f158a1850770fcc1ec",
        packID: "ace-caption-4bit-v1",
        destinationDirectoryName:
            NativeACEDatasetAnnotationBackend.captionerDirectoryName,
        expectedSizeBytes: 4_668_775_088
    )

    public static let transcriber = NativeModelPackCatalogEntry(
        task: .transcribe,
        displayName: "ACE Transcriber 4-bit",
        repositoryID:
            "DaRealDahoodie/pimp-audio-ace-transcriber-4bit-v1",
        revision:
            "528f8addaf9298e855bb4deb11c7cd30443ed775",
        packID: "ace-transcribe-4bit-v1",
        destinationDirectoryName:
            NativeACEDatasetAnnotationBackend.transcriberDirectoryName,
        expectedSizeBytes: 4_668_775_115
    )

    public static let all: [NativeModelPackCatalogEntry] = [
        captioner,
        transcriber,
    ]

    public static func entry(
        for task: DatasetAnnotationTask
    ) -> NativeModelPackCatalogEntry {
        switch task {
        case .caption:
            return captioner
        case .transcribe:
            return transcriber
        }
    }
}

public enum NativeModelPackInstallStage: Equatable, Sendable {
    case preparing
    case downloading
    case verifying
    case installing
    case complete
}

public struct NativeModelPackInstallProgress: Equatable, Sendable {
    public let stage: NativeModelPackInstallStage
    public let fractionCompleted: Double
    public let currentFile: String?

    public init(
        stage: NativeModelPackInstallStage,
        fractionCompleted: Double,
        currentFile: String? = nil
    ) {
        self.stage = stage
        self.fractionCompleted = min(max(fractionCompleted, 0), 1)
        self.currentFile = currentFile
    }
}

protocol NativeModelPackDownloading: Sendable {
    func downloadAndVerify(
        repo: String,
        revision: String,
        file: String,
        to destination: URL,
        expectedHash: String?,
        bearerToken: String?,
        onProgress: @Sendable @escaping (Double) -> Void,
        onRetry: @Sendable @escaping () -> Void
    ) async throws
}

public actor NativeModelPackInstaller {
    private static let diskSafetyMarginBytes: Int64 = 256 * 1_024 * 1_024

    private let modelsDirectory: URL
    private let downloader: any NativeModelPackDownloading

    public init(modelsDirectory: URL) {
        self.modelsDirectory = modelsDirectory
        self.downloader = DownloadClient()
    }

    init(
        modelsDirectory: URL,
        downloader: any NativeModelPackDownloading
    ) {
        self.modelsDirectory = modelsDirectory
        self.downloader = downloader
    }

    @discardableResult
    public func install(
        task: DatasetAnnotationTask,
        onProgress: @Sendable @escaping (
            NativeModelPackInstallProgress
        ) -> Void = { _ in }
    ) async throws -> NativeModelPackManifest {
        try await install(
            NativeModelPackCatalog.entry(for: task),
            onProgress: onProgress
        )
    }

    public func remove(task: DatasetAnnotationTask) throws {
        try Task.checkCancellation()
        let source = NativeModelPackCatalog.entry(for: task)
        let destination = modelsDirectory.appendingPathComponent(
            source.destinationDirectoryName,
            isDirectory: true
        )
        guard FileManager.default.fileExists(atPath: destination.path) else {
            return
        }
        try FileManager.default.removeItem(at: destination)
    }

    @discardableResult
    func install(
        _ source: NativeModelPackCatalogEntry,
        onProgress: @Sendable @escaping (
            NativeModelPackInstallProgress
        ) -> Void = { _ in }
    ) async throws -> NativeModelPackManifest {
        try Task.checkCancellation()
        onProgress(
            NativeModelPackInstallProgress(
                stage: .preparing,
                fractionCompleted: 0
            )
        )

        try FileManager.default.createDirectory(
            at: modelsDirectory,
            withIntermediateDirectories: true
        )
        let stagingDirectory = modelsDirectory.appendingPathComponent(
            ".\(source.destinationDirectoryName).downloading-"
                + UUID().uuidString,
            isDirectory: true
        )
        try FileManager.default.createDirectory(
            at: stagingDirectory,
            withIntermediateDirectories: true
        )
        defer {
            try? FileManager.default.removeItem(at: stagingDirectory)
        }

        let manifestURL = stagingDirectory.appendingPathComponent(
            NativeModelPackManifest.filename
        )
        try await downloader.downloadAndVerify(
            repo: source.repositoryID,
            revision: source.revision,
            file: NativeModelPackManifest.filename,
            to: manifestURL,
            expectedHash: nil,
            bearerToken: nil,
            onProgress: { _ in
                onProgress(
                    NativeModelPackInstallProgress(
                        stage: .downloading,
                        fractionCompleted: 0,
                        currentFile: NativeModelPackManifest.filename
                    )
                )
            },
            onRetry: {}
        )

        try Task.checkCancellation()
        let manifest = try NativeModelPackValidator.loadManifestMetadata(
            at: stagingDirectory
        )
        try validate(manifest, against: source)
        try ensureDiskCapacity(for: manifest.expectedSizeBytes)

        var completedBytes: Int64 = 0
        for record in manifest.files {
            try Task.checkCancellation()
            let bytesBeforeFile = completedBytes
            let destination = stagingDirectory.appendingPathComponent(
                record.path
            )
            try await downloader.downloadAndVerify(
                repo: source.repositoryID,
                revision: source.revision,
                file: record.path,
                to: destination,
                expectedHash: record.sha256,
                bearerToken: nil,
                onProgress: { fileProgress in
                    let downloadedBytes = Double(bytesBeforeFile)
                        + Double(record.byteCount) * fileProgress
                    onProgress(
                        NativeModelPackInstallProgress(
                            stage: .downloading,
                            fractionCompleted:
                                downloadedBytes
                                / Double(manifest.expectedSizeBytes),
                            currentFile: record.path
                        )
                    )
                },
                onRetry: {}
            )
            completedBytes += record.byteCount
        }

        try Task.checkCancellation()
        onProgress(
            NativeModelPackInstallProgress(
                stage: .verifying,
                fractionCompleted: 1
            )
        )
        let validatedManifest = try NativeModelPackValidator.loadAndValidate(
            at: stagingDirectory,
            cancellationCheck: {
                try Task.checkCancellation()
            }
        )

        try Task.checkCancellation()
        onProgress(
            NativeModelPackInstallProgress(
                stage: .installing,
                fractionCompleted: 1
            )
        )
        let destination = modelsDirectory.appendingPathComponent(
            source.destinationDirectoryName,
            isDirectory: true
        )
        try commit(
            stagingDirectory: stagingDirectory,
            destination: destination
        )
        onProgress(
            NativeModelPackInstallProgress(
                stage: .complete,
                fractionCompleted: 1
            )
        )
        return validatedManifest
    }

    private func validate(
        _ manifest: NativeModelPackManifest,
        against source: NativeModelPackCatalogEntry
    ) throws {
        let expectedTask: OmniAnnotationTask
        switch source.task {
        case .caption:
            expectedTask = .caption
        case .transcribe:
            expectedTask = .transcribe
        }
        guard manifest.task == expectedTask else {
            throw NativeModelPackInstallError.taskMismatch(
                expected: source.task,
                actual: manifest.task.rawValue
            )
        }
        guard manifest.packID == source.packID else {
            throw NativeModelPackInstallError.packIDMismatch(
                expected: source.packID,
                actual: manifest.packID
            )
        }
        guard manifest.expectedSizeBytes == source.expectedSizeBytes else {
            throw NativeModelPackInstallError.packSizeMismatch(
                expected: source.expectedSizeBytes,
                actual: manifest.expectedSizeBytes
            )
        }
    }

    private func ensureDiskCapacity(for packSizeBytes: Int64) throws {
        let values = try modelsDirectory.resourceValues(
            forKeys: [
                .volumeAvailableCapacityForImportantUsageKey,
                .volumeAvailableCapacityKey,
            ]
        )
        let availableBytes = values.volumeAvailableCapacityForImportantUsage
            ?? values.volumeAvailableCapacity.map(Int64.init)
        guard let availableBytes else { return }

        let (requiredBytes, overflow) = packSizeBytes.addingReportingOverflow(
            Self.diskSafetyMarginBytes
        )
        guard !overflow, availableBytes >= requiredBytes else {
            throw NativeModelPackInstallError.insufficientDiskSpace(
                requiredBytes: overflow ? Int64.max : requiredBytes,
                availableBytes: availableBytes
            )
        }
    }

    private func commit(
        stagingDirectory: URL,
        destination: URL
    ) throws {
        let fileManager = FileManager.default
        guard fileManager.fileExists(atPath: destination.path) else {
            try fileManager.moveItem(
                at: stagingDirectory,
                to: destination
            )
            return
        }

        let backup = modelsDirectory.appendingPathComponent(
            ".\(destination.lastPathComponent).backup-\(UUID().uuidString)",
            isDirectory: true
        )
        try fileManager.moveItem(at: destination, to: backup)
        do {
            try fileManager.moveItem(
                at: stagingDirectory,
                to: destination
            )
        } catch {
            let installError = error
            do {
                try fileManager.moveItem(at: backup, to: destination)
            } catch {
                throw NativeModelPackInstallError.replacementFailed(
                    installError: installError.localizedDescription,
                    restoreError: error.localizedDescription
                )
            }
            throw installError
        }
        try? fileManager.removeItem(at: backup)
    }
}

public enum NativeModelPackInstallError: LocalizedError {
    case taskMismatch(expected: DatasetAnnotationTask, actual: String)
    case packIDMismatch(expected: String, actual: String)
    case packSizeMismatch(expected: Int64, actual: Int64)
    case insufficientDiskSpace(
        requiredBytes: Int64,
        availableBytes: Int64
    )
    case replacementFailed(installError: String, restoreError: String)

    public var errorDescription: String? {
        switch self {
        case .taskMismatch(let expected, let actual):
            return "Downloaded \(actual) pack cannot be installed as the "
                + "\(expected.rawValue) model."
        case .packIDMismatch(let expected, let actual):
            return "Downloaded native model pack ID \(actual) does not match "
                + "the catalog ID \(expected)."
        case .packSizeMismatch(let expected, let actual):
            return "Downloaded native model pack declares \(actual) bytes; "
                + "the catalog expects \(expected) bytes."
        case .insufficientDiskSpace(let required, let available):
            return "Not enough disk space for the native model pack. "
                + "\(Self.format(required)) is required and "
                + "\(Self.format(available)) is available."
        case .replacementFailed(let installError, let restoreError):
            return "Could not install the native model pack "
                + "(\(installError)) or restore the previous copy "
                + "(\(restoreError))."
        }
    }

    private static func format(_ bytes: Int64) -> String {
        ByteCountFormatter.string(
            fromByteCount: bytes,
            countStyle: .file
        )
    }
}
