import Foundation
import PiMPAudioCore

/// Audio file selected or resolved for LoRA training.
struct TrainingImportedFile: Codable, Hashable, Sendable {
    /// Absolute file URL used for training.
    var url: URL
    /// Caption / style description used as text conditioning.
    var description: String
    /// Optional lyric/transcript conditioning, used mainly for vocal stem datasets.
    var lyrics: String
    /// Optional reusable preprocess cache target, present when imported from an
    /// PiMP Audio Studio dataset bundle that declares `preprocess_cache.json`.
    var preprocessCache: TrainingPreprocessCacheReference?
    /// Optional source timing metadata for clips cut from larger local files.
    var sourceClip: TrainingSourceClipMetadata?

    init(
        url: URL,
        description: String,
        lyrics: String = "",
        preprocessCache: TrainingPreprocessCacheReference? = nil,
        sourceClip: TrainingSourceClipMetadata? = nil
    ) {
        self.url = url
        self.description = description
        self.lyrics = lyrics
        self.preprocessCache = preprocessCache
        self.sourceClip = sourceClip
    }

    enum CodingKeys: String, CodingKey {
        case url
        case description
        case lyrics
        case preprocessCache
        case sourceClip
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        url = try container.decode(URL.self, forKey: .url)
        description = try container.decode(String.self, forKey: .description)
        lyrics = try container.decodeIfPresent(String.self, forKey: .lyrics) ?? ""
        preprocessCache = try container.decodeIfPresent(TrainingPreprocessCacheReference.self, forKey: .preprocessCache)
        sourceClip = try container.decodeIfPresent(TrainingSourceClipMetadata.self, forKey: .sourceClip)
    }
}

struct TrainingSourceClipMetadata: Codable, Hashable, Sendable {
    var fileName: String?
    var startSeconds: Double?
    var startFrame: Int64?
    var frameCount: Int64?
    var sampleRate: Double?

    init(
        fileName: String? = nil,
        startSeconds: Double? = nil,
        startFrame: Int64? = nil,
        frameCount: Int64? = nil,
        sampleRate: Double? = nil
    ) {
        self.fileName = fileName
        self.startSeconds = startSeconds
        self.startFrame = startFrame
        self.frameCount = frameCount
        self.sampleRate = sampleRate
    }
}

struct TrainingPreprocessCacheReference: Codable, Hashable, Sendable {
    var datasetDirectory: URL
    var bundleFileName: String
    var bundleName: String
    var cacheDirectory: String
    var clipID: String
    var sourceAudioPath: String
    var fingerprint: String

    var indexFile: String {
        "\(cacheDirectory)/index.json"
    }

    var sampleFile: String {
        "\(cacheDirectory)/samples/\(sampleFileName)"
    }

    var sampleFileName: String {
        "\(Self.slug(clipID))_\(String(fingerprint.prefix(12))).safetensors"
    }

    init(
        datasetDirectory: URL,
        bundleFileName: String,
        bundleName: String,
        cacheDirectory: String,
        clipID: String,
        sourceAudioPath: String,
        prompt: String,
        lyrics: String
    ) {
        self.datasetDirectory = datasetDirectory
        self.bundleFileName = bundleFileName
        self.bundleName = bundleName
        self.cacheDirectory = cacheDirectory
        self.clipID = clipID
        self.sourceAudioPath = sourceAudioPath
        self.fingerprint = Self.fingerprint([
            bundleFileName,
            bundleName,
            clipID,
            sourceAudioPath,
            prompt,
            lyrics,
        ])
    }

    private static func fingerprint(_ parts: [String]) -> String {
        let input = parts.joined(separator: "\n")
        var hash: UInt64 = 1_469_598_103_934_665_603
        for byte in input.utf8 {
            hash ^= UInt64(byte)
            hash &*= 1_099_511_628_211
        }
        return String(format: "%016llx", hash)
    }

    private static func slug(_ value: String) -> String {
        let allowed = CharacterSet.alphanumerics.union(CharacterSet(charactersIn: "-_"))
        let scalars = value.lowercased().unicodeScalars.map { scalar -> Character in
            allowed.contains(scalar) ? Character(scalar) : "-"
        }
        let collapsed = String(scalars).split(separator: "-").joined(separator: "-")
        return collapsed.isEmpty ? "clip" : collapsed
    }
}

/// Records the training data sources for a LoRA adapter so the session can be
/// reloaded later for retraining with the same (or expanded) dataset.
///
/// Saved as `training_manifest.json` alongside `adapter.json` in the adapter
/// directory. Absence of the file is gracefully handled — adapters created
/// before this feature was added simply have no manifest.
struct TrainingManifest: Codable {
    /// IDs of `HistoryEntry` items used as training data.
    let historyEntryIDs: [UUID]

    /// Imported audio files (not from history) with their training captions.
    let importedFiles: [TrainingImportedFile]
}

enum TrainingRunStatus: String, Codable, Equatable {
    case running
    case completed
    case paused
    case cancelled
    case failed
}

struct TrainingRunProgressEntry: Codable, Equatable {
    var timestamp: Date
    var epoch: Int
    var totalEpochs: Int
    var step: Int
    var loss: Float
    var learningRate: Float
    var lossMovingAverage: Float?
    var gradientNorm: Float?
    var parameterNorm: Float?
    var updateNorm: Float?
    var validationLoss: Float? = nil
    var elapsed: TimeInterval
    var estimatedRemaining: TimeInterval?
}

struct TrainingRunEvent: Codable, Equatable {
    var timestamp: Date
    var status: TrainingRunStatus
    var message: String
}

/// Durable sidecar written beside each trained adapter for debugging,
/// support, and reproducible local training history.
struct TrainingRunLog: Codable, Equatable {
    static let currentSchemaVersion = "1.1"

    var schemaVersion: String
    var adapterID: String
    var adapterName: String
    var adapterKind: AdapterKind
    var status: TrainingRunStatus
    var baseModelID: String
    var sampleCount: Int
    var config: LoRAConfig
    var lokrConfig: LoKRConfig?
    var resumedFromCheckpoint: Bool
    var startedAt: Date
    var updatedAt: Date
    var finishedAt: Date?
    var progress: [TrainingRunProgressEntry]
    var lossHistory: [Float]
    var validationHistory: [TrainingValidationPoint]?
    var events: [TrainingRunEvent]
    var finalLoss: Float?
    var trainedPaths: [String]
    var trainedLayerIndices: [Int]
    var weightsTensorCount: Int?
    var errorMessage: String?

    init(
        adapterID: String,
        adapterName: String,
        adapterKind: AdapterKind,
        baseModelID: String,
        sampleCount: Int,
        config: LoRAConfig,
        lokrConfig: LoKRConfig?,
        resumedFromCheckpoint: Bool,
        startedAt: Date = Date()
    ) {
        self.schemaVersion = Self.currentSchemaVersion
        self.adapterID = adapterID
        self.adapterName = adapterName
        self.adapterKind = adapterKind
        self.status = .running
        self.baseModelID = baseModelID
        self.sampleCount = sampleCount
        self.config = config
        self.lokrConfig = lokrConfig
        self.resumedFromCheckpoint = resumedFromCheckpoint
        self.startedAt = startedAt
        self.updatedAt = startedAt
        self.finishedAt = nil
        self.progress = []
        self.lossHistory = []
        self.validationHistory = []
        self.events = [
            TrainingRunEvent(
                timestamp: startedAt,
                status: .running,
                message: resumedFromCheckpoint ? "Training resumed from checkpoint." : "Training started."
            )
        ]
        self.finalLoss = nil
        self.trainedPaths = []
        self.trainedLayerIndices = []
        self.weightsTensorCount = nil
        self.errorMessage = nil
    }

    mutating func recordProgress(
        _ entry: TrainingRunProgressEntry,
        lossHistory: [Float],
        validationHistory: [TrainingValidationPoint]
    ) {
        progress.append(entry)
        self.lossHistory = lossHistory
        self.validationHistory = validationHistory
        updatedAt = entry.timestamp
    }

    mutating func finish(
        status: TrainingRunStatus,
        finalLoss: Float? = nil,
        trainedPaths: [String] = [],
        trainedLayerIndices: [Int] = [],
        weightsTensorCount: Int? = nil,
        message: String,
        errorMessage: String? = nil,
        finishedAt: Date = Date()
    ) {
        self.status = status
        self.finalLoss = finalLoss ?? self.finalLoss
        self.trainedPaths = trainedPaths.isEmpty ? self.trainedPaths : trainedPaths
        self.trainedLayerIndices = trainedLayerIndices.isEmpty ? self.trainedLayerIndices : trainedLayerIndices
        self.weightsTensorCount = weightsTensorCount ?? self.weightsTensorCount
        self.errorMessage = errorMessage
        self.finishedAt = finishedAt
        self.updatedAt = finishedAt
        self.events.append(TrainingRunEvent(
            timestamp: finishedAt,
            status: status,
            message: message
        ))
    }
}

enum TrainingDatasetImportError: LocalizedError {
    case unsupportedJSON(String)
    case noUsableAudioFiles(URL)

    var errorDescription: String? {
        switch self {
        case .unsupportedJSON(let name):
            return "\(name) is not a supported PiMP Audio Studio training dataset JSON file."
        case .noUsableAudioFiles(let url):
            return "No readable audio files were found from \(url.lastPathComponent)."
        }
    }
}

enum TrainingDatasetImporter {
    private struct CaptionEntry {
        let description: String?
        let lyrics: String?
    }

    static func importFiles(from jsonURL: URL) throws -> [TrainingImportedFile] {
        let didStartAccessing = jsonURL.startAccessingSecurityScopedResource()
        defer {
            if didStartAccessing {
                jsonURL.stopAccessingSecurityScopedResource()
            }
        }

        let data = try Data(contentsOf: jsonURL)
        let object = try JSONSerialization.jsonObject(with: data)
        let fileName = jsonURL.lastPathComponent
        let jsonBaseDirectory = jsonURL.deletingLastPathComponent()

        let imported: [TrainingImportedFile]
        if let clips = object as? [[String: Any]] {
            imported = files(
                from: clips,
                include: includeList(from: [:], fileName: fileName),
                baseDirectory: jsonBaseDirectory
            )
        } else if let dict = object as? [String: Any] {
            imported = try files(from: dict, baseDirectory: jsonBaseDirectory, fileName: fileName)
        } else {
            throw TrainingDatasetImportError.unsupportedJSON(fileName)
        }

        guard !imported.isEmpty else {
            throw TrainingDatasetImportError.noUsableAudioFiles(jsonURL)
        }
        return imported
    }

    private static func files(
        from dict: [String: Any],
        baseDirectory: URL,
        fileName: String
    ) throws -> [TrainingImportedFile] {
        if let clips = dict["clips"] as? [[String: Any]] {
            let include = includeList(from: dict, fileName: fileName)
            let datasetBase = datasetBaseDirectory(from: dict, fallback: baseDirectory)
            let enrichedClips = clipsByApplyingTopLevelSeparation(clips, from: dict)
            let cacheContext = preprocessCacheContext(
                from: dict,
                fileName: fileName,
                packageDirectory: baseDirectory,
                include: include
            )
            return files(
                from: enrichedClips,
                include: include,
                baseDirectory: datasetBase,
                cacheContext: cacheContext
            )
        }

        if let datasets = dict["datasets"] as? [[String: Any]] {
            return filesFromDatasetConfig(datasets: datasets, baseDirectory: baseDirectory)
        }

        throw TrainingDatasetImportError.unsupportedJSON(fileName)
    }

    private static func files(
        from clips: [[String: Any]],
        include: [String],
        baseDirectory: URL,
        cacheContext: PreprocessCacheContext? = nil
    ) -> [TrainingImportedFile] {
        clips.compactMap { clip in
            guard let audioPath = selectedAudioPath(from: clip, include: include) else {
                return nil
            }
            let audioURL = resolve(audioPath, relativeTo: baseDirectory)
            guard FileManager.default.fileExists(atPath: audioURL.path(percentEncoded: false)) else {
                return nil
            }

            let description = firstString(clip, keys: ["prompt", "caption"])
                ?? tagsDescription(from: clip)
                ?? audioURL.deletingPathExtension().lastPathComponent
            let lyrics = shouldUseLyrics(for: audioPath, include: include)
                ? transcript(from: clip)
                : ""
            let clipID = firstString(clip, keys: ["id"])
                ?? URL(fileURLWithPath: audioPath).deletingPathExtension().lastPathComponent
            let cache = cacheContext.map {
                TrainingPreprocessCacheReference(
                    datasetDirectory: $0.packageDirectory,
                    bundleFileName: $0.bundleFileName,
                    bundleName: $0.bundleName,
                    cacheDirectory: $0.cacheDirectory,
                    clipID: clipID,
                    sourceAudioPath: audioPath,
                    prompt: description,
                    lyrics: lyrics
                )
            }
            return TrainingImportedFile(
                url: audioURL,
                description: description,
                lyrics: lyrics,
                preprocessCache: cache,
                sourceClip: sourceClipMetadata(from: clip)
            )
        }
    }

    private struct PreprocessCacheContext {
        let packageDirectory: URL
        let bundleFileName: String
        let bundleName: String
        let cacheDirectory: String
    }

    private static func preprocessCacheContext(
        from dict: [String: Any],
        fileName: String,
        packageDirectory: URL,
        include: [String]
    ) -> PreprocessCacheContext? {
        let declaresCache = firstString(dict, keys: ["preprocess_cache"]) != nil ||
            firstString(dict, keys: ["pimp_audio_training_bundle_version", "anvil_training_bundle_version"]) != nil ||
            fileName.lowercased().hasPrefix("training_bundle")
        guard declaresCache else { return nil }
        let bundleName = bundleCacheName(include: include, fileName: fileName)
        return PreprocessCacheContext(
            packageDirectory: packageDirectory,
            bundleFileName: fileName,
            bundleName: bundleName,
            cacheDirectory: "preprocessed/\(bundleName)"
        )
    }

    private static func bundleCacheName(include: [String], fileName: String) -> String {
        let normalized = include.map { $0.lowercased() }
        if normalized.contains(where: { ["vocal", "vocals", "voice", "voices"].contains($0) }) {
            return "vocals"
        }
        if normalized.contains(where: { ["instrumental", "instrumentals", "music", "backing"].contains($0) }) {
            return "instrumental"
        }
        let lowerName = fileName.lowercased()
        if lowerName.contains("vocal") {
            return "vocals"
        }
        if lowerName.contains("instrumental") {
            return "instrumental"
        }
        return "full_mix"
    }

    private static func filesFromDatasetConfig(
        datasets: [[String: Any]],
        baseDirectory: URL
    ) -> [TrainingImportedFile] {
        let supportedExtensions = Set(["wav", "m4a", "mp3", "aiff", "aif", "aac", "flac", "ogg", "opus", "webm"])
        let defaultCaptionLookup = captionLookup(
            at: baseDirectory.appendingPathComponent("captions.json"),
            baseDirectory: baseDirectory
        )
        var imported: [TrainingImportedFile] = []

        for dataset in datasets {
            guard let path = dataset["path"] as? String else { continue }
            let directory = resolve(path, relativeTo: baseDirectory)
            let captions = captionsForDatasetConfig(
                dataset,
                baseDirectory: baseDirectory,
                defaultLookup: defaultCaptionLookup
            )
            guard let enumerator = FileManager.default.enumerator(
                at: directory,
                includingPropertiesForKeys: nil,
                options: [.skipsHiddenFiles]
            ) else { continue }

            for case let fileURL as URL in enumerator {
                guard supportedExtensions.contains(fileURL.pathExtension.lowercased()) else { continue }
                let captionEntry = captionEntry(
                    for: fileURL,
                    datasetDirectory: directory,
                    baseDirectory: baseDirectory,
                    lookup: captions
                )
                imported.append(TrainingImportedFile(
                    url: fileURL,
                    description: captionEntry?.description ?? fileURL.deletingPathExtension().lastPathComponent,
                    lyrics: shouldUseLyrics(for: fileURL.path(percentEncoded: false), include: [])
                        ? captionEntry?.lyrics ?? ""
                        : ""
                ))
            }
        }

        return imported.sorted { $0.url.path < $1.url.path }
    }

    private static func captionsForDatasetConfig(
        _ dataset: [String: Any],
        baseDirectory: URL,
        defaultLookup: [String: CaptionEntry]
    ) -> [String: CaptionEntry] {
        guard let captionPath = dataset["caption_file"] as? String, !captionPath.isEmpty else {
            return defaultLookup
        }
        return captionLookup(at: resolve(captionPath, relativeTo: baseDirectory), baseDirectory: baseDirectory)
    }

    private static func captionEntry(
        for fileURL: URL,
        datasetDirectory: URL,
        baseDirectory: URL,
        lookup: [String: CaptionEntry]
    ) -> CaptionEntry? {
        for key in captionLookupKeys(for: fileURL, datasetDirectory: datasetDirectory, baseDirectory: baseDirectory) {
            if let entry = lookup[key] {
                return entry
            }
        }
        return nil
    }

    private static func captionLookup(
        at url: URL,
        baseDirectory: URL
    ) -> [String: CaptionEntry] {
        guard FileManager.default.fileExists(atPath: url.path(percentEncoded: false)),
              let data = try? Data(contentsOf: url),
              let object = try? JSONSerialization.jsonObject(with: data) else {
            return [:]
        }

        if let clips = object as? [[String: Any]] {
            return captionLookup(from: clips, baseDirectory: baseDirectory)
        }

        if let dict = object as? [String: Any] {
            if let clips = dict["clips"] as? [[String: Any]] {
                return captionLookup(from: clips, baseDirectory: baseDirectory)
            }
            return captionLookup(from: dict, baseDirectory: baseDirectory)
        }

        return [:]
    }

    private static func captionLookup(
        from clips: [[String: Any]],
        baseDirectory: URL
    ) -> [String: CaptionEntry] {
        var lookup: [String: CaptionEntry] = [:]
        for clip in clips {
            guard let path = firstString(clip, keys: ["file", "path", "audio"]) else { continue }
            addCaptionEntry(
                CaptionEntry(description: firstString(clip, keys: ["prompt", "caption"]), lyrics: transcript(from: clip)),
                path: path,
                baseDirectory: baseDirectory,
                to: &lookup
            )
        }
        return lookup
    }

    private static func captionLookup(
        from dict: [String: Any],
        baseDirectory: URL
    ) -> [String: CaptionEntry] {
        var lookup: [String: CaptionEntry] = [:]
        for (path, value) in dict {
            if let caption = value as? String {
                addCaptionEntry(
                    CaptionEntry(description: caption, lyrics: nil),
                    path: path,
                    baseDirectory: baseDirectory,
                    to: &lookup
                )
            } else if let clip = value as? [String: Any] {
                addCaptionEntry(
                    CaptionEntry(description: firstString(clip, keys: ["prompt", "caption"]), lyrics: transcript(from: clip)),
                    path: path,
                    baseDirectory: baseDirectory,
                    to: &lookup
                )
            }
        }
        return lookup
    }

    private static func addCaptionEntry(
        _ entry: CaptionEntry,
        path: String,
        baseDirectory: URL,
        to lookup: inout [String: CaptionEntry]
    ) {
        for key in captionLookupKeys(for: path, baseDirectory: baseDirectory) {
            lookup[key] = entry
        }
    }

    private static func includeList(from dict: [String: Any], fileName: String) -> [String] {
        if let include = dict["include"] as? [String], !include.isEmpty {
            return include
        }
        let lowerName = fileName.lowercased()
        if lowerName.contains("vocal") || lowerName.contains("voice") {
            return ["vocals"]
        }
        if lowerName.contains("instrumental") || lowerName.contains("backing") {
            return ["instrumental"]
        }
        if lowerName.contains("full_mix") || lowerName.contains("full-mix") {
            return ["full_mix"]
        }
        return []
    }

    private static func clipsByApplyingTopLevelSeparation(
        _ clips: [[String: Any]],
        from dict: [String: Any]
    ) -> [[String: Any]] {
        guard let separation = dict["separation"] as? [String: Any],
              let separationClips = separation["clips"] as? [[String: Any]] else {
            return clips
        }

        var separationByFile: [String: [String: Any]] = [:]
        for clip in separationClips {
            guard let file = firstString(clip, keys: ["file"]) else { continue }
            separationByFile[file] = clip
        }
        return clips.map { clip in
            guard clip["separation"] == nil,
                  let file = firstString(clip, keys: ["file"]),
                  let separationClip = separationByFile[file],
                  let stems = separationClip["stems"] as? [String: Any] else {
                return clip
            }
            var enriched = clip
            enriched["separation"] = [
                "stems": stems,
                "separation_file": separationClip["separation_file"] as? String ?? "",
            ]
            return enriched
        }
    }

    private static func datasetBaseDirectory(from dict: [String: Any], fallback: URL) -> URL {
        guard let datasetDir = dict["dataset_dir"] as? String, !datasetDir.isEmpty else {
            return fallback
        }
        return resolve(datasetDir, relativeTo: fallback)
    }

    private static func selectedAudioPath(from clip: [String: Any], include: [String]) -> String? {
        let normalizedIncludes = include.map { $0.lowercased() }
        if let assets = clip["assets"] as? [String: Any],
           let asset = selectedAsset(from: assets, include: normalizedIncludes) {
            return asset
        }

        if let separation = clip["separation"] as? [String: Any],
           let stems = separation["stems"] as? [String: Any],
           let stem = selectedAsset(from: stems, include: normalizedIncludes) {
            return stem
        }

        return firstString(clip, keys: ["file", "path", "audio"])
    }

    private static func selectedAsset(from assets: [String: Any], include: [String]) -> String? {
        for key in preferredAssetKeys(include: include) {
            if let value = assets[key] as? String {
                return value
            }
            if let matchedKey = assets.keys.first(where: { $0.caseInsensitiveCompare(key) == .orderedSame }),
               let value = assets[matchedKey] as? String {
                return value
            }
        }
        return nil
    }

    private static func preferredAssetKeys(include: [String]) -> [String] {
        var keys: [String] = []
        for item in include {
            switch item {
            case "vocal", "vocals", "voice", "voices":
                keys.append(contentsOf: ["vocals", "voice", "vocal"])
            case "instrumental", "instrumentals", "music", "backing":
                keys.append(contentsOf: ["instrumental", "instrumentals", "music", "backing"])
            case "full_mix", "fullmix", "full mix", "mix":
                keys.append(contentsOf: ["full_mix", "fullmix", "mix"])
            default:
                keys.append(item)
            }
        }
        keys.append(contentsOf: ["full_mix", "file", "audio"])
        var seen = Set<String>()
        return keys.filter { seen.insert($0).inserted }
    }

    private static func shouldUseLyrics(for audioPath: String, include: [String]) -> Bool {
        let lowerPath = audioPath.lowercased()
        if lowerPath.contains("instrumental") {
            return false
        }
        if include.contains(where: { ["vocal", "vocals", "voice", "voices"].contains($0.lowercased()) }) {
            return true
        }
        return lowerPath.contains("vocal") || lowerPath.contains("voice")
    }

    private static func transcript(from clip: [String: Any]) -> String {
        if let lyrics = firstString(clip, keys: ["lyrics", "transcript"]) {
            return lyrics
        }
        if let transcription = clip["transcription"] as? [String: Any],
           let text = transcription["text"] as? String {
            return text
        }
        return ""
    }

    private static func sourceClipMetadata(from clip: [String: Any]) -> TrainingSourceClipMetadata? {
        let source = clip["source"] as? [String: Any] ?? [:]
        let fileName = firstString(clip, keys: ["source_file_name"])
            ?? firstString(source, keys: ["file_name"])
        let startSeconds = firstDouble(clip, keys: ["source_start_seconds"])
            ?? firstDouble(source, keys: ["source_start_seconds", "start_seconds"])
        let startFrame = firstInt64(clip, keys: ["source_start_frame"])
            ?? firstInt64(source, keys: ["source_start_frame", "start_frame"])
        let frameCount = firstInt64(clip, keys: ["source_frame_count"])
            ?? firstInt64(source, keys: ["source_frame_count", "frame_count"])
        let sampleRate = firstDouble(clip, keys: ["source_sample_rate"])
            ?? firstDouble(source, keys: ["source_sample_rate", "sample_rate"])

        guard fileName != nil ||
            startSeconds != nil ||
            startFrame != nil ||
            frameCount != nil ||
            sampleRate != nil else {
            return nil
        }
        return TrainingSourceClipMetadata(
            fileName: fileName,
            startSeconds: startSeconds,
            startFrame: startFrame,
            frameCount: frameCount,
            sampleRate: sampleRate
        )
    }

    private static func tagsDescription(from clip: [String: Any]) -> String? {
        guard let tags = clip["tags"] as? [String], !tags.isEmpty else { return nil }
        return tags.joined(separator: ", ")
    }

    private static func firstString(_ dict: [String: Any], keys: [String]) -> String? {
        for key in keys {
            if let value = dict[key] as? String {
                let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
                if !trimmed.isEmpty {
                    return trimmed
                }
            }
        }
        return nil
    }

    private static func firstDouble(_ dict: [String: Any], keys: [String]) -> Double? {
        for key in keys {
            switch dict[key] {
            case let value as Double:
                return value
            case let value as Float:
                return Double(value)
            case let value as Int:
                return Double(value)
            case let value as Int64:
                return Double(value)
            case let value as NSNumber:
                return value.doubleValue
            case let value as String:
                if let parsed = Double(value) {
                    return parsed
                }
            default:
                continue
            }
        }
        return nil
    }

    private static func firstInt64(_ dict: [String: Any], keys: [String]) -> Int64? {
        for key in keys {
            switch dict[key] {
            case let value as Int64:
                return value
            case let value as Int:
                return Int64(value)
            case let value as NSNumber:
                return value.int64Value
            case let value as String:
                if let parsed = Int64(value) {
                    return parsed
                }
            default:
                continue
            }
        }
        return nil
    }

    private static func resolve(_ path: String, relativeTo baseDirectory: URL) -> URL {
        let expanded = (path as NSString).expandingTildeInPath
        if expanded.hasPrefix("/") {
            return URL(fileURLWithPath: expanded).standardizedFileURL
        }
        return baseDirectory.appendingPathComponent(expanded).standardizedFileURL
    }

    private static func captionLookupKeys(
        for path: String,
        baseDirectory: URL
    ) -> [String] {
        let cleanedPath = path.hasPrefix("./") ? String(path.dropFirst(2)) : path
        let resolved = resolve(cleanedPath, relativeTo: baseDirectory)
        return uniqueStrings([
            path,
            cleanedPath,
            resolved.path(percentEncoded: false),
            URL(fileURLWithPath: cleanedPath).lastPathComponent,
        ])
    }

    private static func captionLookupKeys(
        for fileURL: URL,
        datasetDirectory: URL,
        baseDirectory: URL
    ) -> [String] {
        uniqueStrings([
            fileURL.standardizedFileURL.path,
            relativePath(for: fileURL, from: baseDirectory),
            relativePath(for: fileURL, from: datasetDirectory),
            fileURL.lastPathComponent,
        ].compactMap { $0 })
    }

    private static func relativePath(for url: URL, from baseDirectory: URL) -> String? {
        var basePath = baseDirectory.standardizedFileURL.path
        if !basePath.hasSuffix("/") {
            basePath.append("/")
        }
        let filePath = url.standardizedFileURL.path
        guard filePath.hasPrefix(basePath) else { return nil }
        return String(filePath.dropFirst(basePath.count))
    }

    private static func uniqueStrings(_ values: [String]) -> [String] {
        var seen = Set<String>()
        return values.filter { !$0.isEmpty && seen.insert($0).inserted }
    }
}
