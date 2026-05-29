import Foundation

struct LocalFilesDatasetConfig {
    var clipLength: TimeInterval = 35
    var clipsPerTrack: Int = 8
    var maxTotalClips: Int = 40
}

struct LocalFilesDatasetResult {
    let outputDirectory: URL
    let clips: [TrainingImportedFile]
    var clipCount: Int { clips.count }
}

enum LocalAudioFileDiscovery {
    static let supportedExtensions = Set(["aac", "aif", "aiff", "flac", "m4a", "mp3", "ogg", "opus", "wav", "webm"])

    static func audioFiles(in directories: [URL]) -> [URL] {
        directories.flatMap { audioFiles(in: $0) }.deduplicatedByFilePath()
    }

    static func audioFiles(in directory: URL) -> [URL] {
        let didStartAccessing = directory.startAccessingSecurityScopedResource()
        defer {
            if didStartAccessing {
                directory.stopAccessingSecurityScopedResource()
            }
        }

        let keys: [URLResourceKey] = [.isRegularFileKey]
        guard let enumerator = FileManager.default.enumerator(
            at: directory,
            includingPropertiesForKeys: keys,
            options: [.skipsHiddenFiles, .skipsPackageDescendants]
        ) else {
            return []
        }

        var files: [URL] = []
        for case let fileURL as URL in enumerator {
            guard supportedExtensions.contains(fileURL.pathExtension.lowercased()) else { continue }
            let values = try? fileURL.resourceValues(forKeys: Set(keys))
            guard values?.isRegularFile != false else { continue }
            files.append(fileURL)
        }

        return files.sorted { lhs, rhs in
            lhs.path.localizedStandardCompare(rhs.path) == .orderedAscending
        }
    }
}

private extension Array where Element == URL {
    func deduplicatedByFilePath() -> [URL] {
        var seen = Set<String>()
        return filter { url in
            seen.insert(url.standardizedFileURL.path).inserted
        }
    }
}

enum LocalFilesImportPhase: Equatable {
    case idle
    case chunking(index: Int, total: Int, name: String)
    case complete(outputDirectory: URL, clipCount: Int)
    case failed(String)

    var isRunning: Bool {
        if case .chunking = self { return true }
        return false
    }
}

enum LocalFilesDatasetError: LocalizedError {
    case noFilesSelected
    case noClipsProduced

    var errorDescription: String? {
        switch self {
        case .noFilesSelected:
            return "Select at least one local audio file."
        case .noClipsProduced:
            return "No clips were produced from the selected files."
        }
    }
}

@MainActor
final class LocalFilesDatasetBuilder: ObservableObject {
    @Published var phase: LocalFilesImportPhase = .idle
    @Published var resultClips: [TrainingImportedFile] = []

    private var task: Task<Void, Never>?

    func start(
        sourceFiles: [URL],
        sourceDirectories: [URL] = [],
        config: LocalFilesDatasetConfig,
        outputBase: URL,
        styleTemplate: String = ""
    ) {
        cancel()
        resultClips = []
        phase = .chunking(index: 0, total: sourceFiles.count, name: "Preparing")

        task = Task { [weak self] in
            guard let self else { return }
            do {
                let result = try await Self.run(
                    sourceFiles: sourceFiles,
                    sourceDirectories: sourceDirectories,
                    config: config,
                    outputBase: outputBase,
                    styleTemplate: styleTemplate,
                    onPhase: { [weak self] p in await MainActor.run { self?.phase = p } }
                )
                self.resultClips = result.clips
                self.phase = .complete(outputDirectory: result.outputDirectory, clipCount: result.clipCount)
            } catch is CancellationError {
                self.phase = .idle
            } catch {
                self.phase = .failed(error.localizedDescription)
            }
        }
    }

    func cancel() {
        task?.cancel()
        task = nil
    }

    static func run(
        sourceFiles: [URL],
        sourceDirectories: [URL] = [],
        config: LocalFilesDatasetConfig,
        outputBase: URL,
        styleTemplate: String = "",
        onPhase: @escaping (LocalFilesImportPhase) async -> Void
    ) async throws -> LocalFilesDatasetResult {
        guard !sourceFiles.isEmpty else { throw LocalFilesDatasetError.noFilesSelected }

        let scopedDirectories = sourceDirectories.filter { $0.startAccessingSecurityScopedResource() }
        defer {
            scopedDirectories.forEach { $0.stopAccessingSecurityScopedResource() }
        }

        let folderTag = datasetFolderTag(for: sourceFiles.first)
        let outputDir = outputBase
            .appendingPathComponent("TrainingDatasets", isDirectory: true)
            .appendingPathComponent("\(folderTag)_local_\(timestampTag())", isDirectory: true)
        let clipsDir = outputDir.appendingPathComponent("clips", isDirectory: true)
        try FileManager.default.createDirectory(at: clipsDir, withIntermediateDirectories: true)

        var allClips: [TrainingImportedFile] = []
        let trimmedStyle = styleTemplate.trimmingCharacters(in: .whitespacesAndNewlines)

        for (idx, sourceURL) in sourceFiles.enumerated() {
            try Task.checkCancellation()
            guard allClips.count < config.maxTotalClips else { break }

            let title = sourceURL.deletingPathExtension().lastPathComponent
            await onPhase(.chunking(index: idx + 1, total: sourceFiles.count, name: sourceURL.lastPathComponent))

            let didStartAccessing = sourceURL.startAccessingSecurityScopedResource()
            defer {
                if didStartAccessing {
                    sourceURL.stopAccessingSecurityScopedResource()
                }
            }

            do {
                let remaining = config.maxTotalClips - allClips.count
                let chunks = try await YouTubeDatasetBuilder.chunkAudio(
                    from: sourceURL,
                    title: title,
                    videoIndex: idx + 1,
                    clipLength: config.clipLength,
                    maxClips: min(config.clipsPerTrack, remaining),
                    styleTemplate: trimmedStyle,
                    outputDirectory: clipsDir,
                    filenamePrefix: "f"
                )
                allClips.append(contentsOf: chunks.map {
                    TrainingImportedFile(url: $0.url, description: $0.caption)
                })
            } catch is CancellationError {
                throw CancellationError()
            } catch {
                print("[LocalFilesDatasetBuilder] Skipping \(sourceURL.lastPathComponent): \(error.localizedDescription)")
            }
        }

        guard !allClips.isEmpty else { throw LocalFilesDatasetError.noClipsProduced }

        try TrainingDatasetPackageWriter.writeLocalPackage(
            outputDirectory: outputDir,
            clips: allClips,
            sourceFiles: sourceFiles,
            config: config,
            styleTemplate: trimmedStyle
        )
        return LocalFilesDatasetResult(outputDirectory: outputDir, clips: allClips)
    }

    private static func datasetFolderTag(for url: URL?) -> String {
        let name = url?.deletingPathExtension().lastPathComponent ?? "local-dataset"
        let sanitized = YouTubeDatasetBuilder.sanitizeName(name)
        return sanitized.isEmpty ? "local-dataset" : sanitized
    }

    private static func timestampTag() -> String {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.dateFormat = "yyyy-MM-dd-HHmmss"
        return formatter.string(from: Date())
    }
}
