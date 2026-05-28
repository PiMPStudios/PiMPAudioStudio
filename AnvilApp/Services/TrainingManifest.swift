import Foundation

/// Audio file selected or resolved for LoRA training.
struct TrainingImportedFile: Codable, Hashable, Sendable {
    /// Absolute file URL used for training.
    var url: URL
    /// Caption / style description used as text conditioning.
    var description: String
    /// Optional lyric/transcript conditioning, used mainly for vocal stem datasets.
    var lyrics: String

    init(url: URL, description: String, lyrics: String = "") {
        self.url = url
        self.description = description
        self.lyrics = lyrics
    }

    enum CodingKeys: String, CodingKey {
        case url
        case description
        case lyrics
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        url = try container.decode(URL.self, forKey: .url)
        description = try container.decode(String.self, forKey: .description)
        lyrics = try container.decodeIfPresent(String.self, forKey: .lyrics) ?? ""
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

enum TrainingDatasetImportError: LocalizedError {
    case unsupportedJSON(String)
    case noUsableAudioFiles(URL)

    var errorDescription: String? {
        switch self {
        case .unsupportedJSON(let name):
            return "\(name) is not a supported Anvil training dataset JSON file."
        case .noUsableAudioFiles(let url):
            return "No readable audio files were found from \(url.lastPathComponent)."
        }
    }
}

enum TrainingDatasetImporter {
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
            imported = files(from: clips, include: [], baseDirectory: jsonBaseDirectory)
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
            let include = (dict["include"] as? [String]) ?? []
            let datasetBase = datasetBaseDirectory(from: dict, fallback: baseDirectory)
            return files(from: clips, include: include, baseDirectory: datasetBase)
        }

        if let datasets = dict["datasets"] as? [[String: Any]] {
            return filesFromDatasetConfig(datasets: datasets, baseDirectory: baseDirectory)
        }

        throw TrainingDatasetImportError.unsupportedJSON(fileName)
    }

    private static func files(
        from clips: [[String: Any]],
        include: [String],
        baseDirectory: URL
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
            return TrainingImportedFile(url: audioURL, description: description, lyrics: lyrics)
        }
    }

    private static func filesFromDatasetConfig(
        datasets: [[String: Any]],
        baseDirectory: URL
    ) -> [TrainingImportedFile] {
        let supportedExtensions = Set(["wav", "m4a", "mp3", "aiff", "aif", "aac", "flac", "ogg", "opus"])
        var imported: [TrainingImportedFile] = []

        for dataset in datasets {
            guard let path = dataset["path"] as? String else { continue }
            let directory = resolve(path, relativeTo: baseDirectory)
            guard let enumerator = FileManager.default.enumerator(
                at: directory,
                includingPropertiesForKeys: nil,
                options: [.skipsHiddenFiles]
            ) else { continue }

            for case let fileURL as URL in enumerator {
                guard supportedExtensions.contains(fileURL.pathExtension.lowercased()) else { continue }
                imported.append(TrainingImportedFile(
                    url: fileURL,
                    description: fileURL.deletingPathExtension().lastPathComponent
                ))
            }
        }

        return imported.sorted { $0.url.path < $1.url.path }
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
            default:
                keys.append(item)
            }
        }
        keys.append(contentsOf: ["file", "audio"])
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

    private static func resolve(_ path: String, relativeTo baseDirectory: URL) -> URL {
        let expanded = (path as NSString).expandingTildeInPath
        if expanded.hasPrefix("/") {
            return URL(fileURLWithPath: expanded)
        }
        return baseDirectory.appendingPathComponent(expanded)
    }
}
