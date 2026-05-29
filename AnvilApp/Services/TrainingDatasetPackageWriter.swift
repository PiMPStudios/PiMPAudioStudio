import Foundation

enum TrainingDatasetPackageWriter {
    static let captionsFileName = "captions.json"
    static let datasetConfigFileName = "dataset_config.json"
    static let datasetManifestFileName = "dataset_manifest.json"
    static let trainingBundleFileName = "training_bundle.json"
    static let fullMixBundleFileName = "training_bundle_full_mix.json"
    static let remotePackageFileName = "remote_training_package.json"

    static func writeLocalPackage(
        outputDirectory: URL,
        clips: [TrainingImportedFile],
        sourceFiles: [URL],
        config: LocalFilesDatasetConfig,
        styleTemplate: String,
        createdAt: Date = Date()
    ) throws {
        try FileManager.default.createDirectory(at: outputDirectory, withIntermediateDirectories: true)

        try writeCaptionsFile(
            clips: clips.map { (url: $0.url, caption: normalizedCaption(for: $0)) },
            to: outputDirectory
        )

        let clipEntries = packageClips(
            from: clips,
            sourceFiles: sourceFiles,
            outputDirectory: outputDirectory,
            config: config
        )
        let createdAtString = ISO8601DateFormatter().string(from: createdAt)
        let trimmedStyle = styleTemplate.trimmingCharacters(in: .whitespacesAndNewlines)

        try writeJSONObject(
            datasetConfig(),
            to: outputDirectory.appendingPathComponent(datasetConfigFileName)
        )
        try writeJSONObject(
            datasetManifest(
                outputDirectory: outputDirectory,
                clips: clipEntries,
                sourceFiles: sourceFiles,
                config: config,
                styleTemplate: trimmedStyle,
                createdAt: createdAtString
            ),
            to: outputDirectory.appendingPathComponent(datasetManifestFileName)
        )

        let bundle = trainingBundle(clips: clipEntries, createdAt: createdAtString)
        try writeJSONObject(bundle, to: outputDirectory.appendingPathComponent(trainingBundleFileName))
        try writeJSONObject(bundle, to: outputDirectory.appendingPathComponent(fullMixBundleFileName))
        try writeJSONObject(
            remotePackageDescriptor(
                outputDirectory: outputDirectory,
                clipCount: clips.count,
                createdAt: createdAtString
            ),
            to: outputDirectory.appendingPathComponent(remotePackageFileName)
        )
    }

    private static func datasetConfig() -> [String: Any] {
        [
            "anvil_dataset_config_version": "1.0",
            "dataset_type": "audio_dir",
            "datasets": [
                [
                    "path": "clips",
                    "caption_file": captionsFileName,
                    "audio_role": "full_mix",
                ],
            ],
            "random_crop": true,
        ]
    }

    private static func datasetManifest(
        outputDirectory: URL,
        clips: [[String: Any]],
        sourceFiles: [URL],
        config: LocalFilesDatasetConfig,
        styleTemplate: String,
        createdAt: String
    ) -> [String: Any] {
        [
            "anvil_dataset_version": "1.0",
            "name": outputDirectory.lastPathComponent,
            "created_at": createdAt,
            "source": [
                "kind": "local_files",
                "files": sourceFiles.map { ["file_name": $0.lastPathComponent] },
            ],
            "caption_mode": styleTemplate.isEmpty ? "filename" : "style_template",
            "style_hint": styleTemplate,
            "clip_count": clips.count,
            "clip_length_seconds": seconds(config.clipLength),
            "estimated_total_seconds": seconds(Double(clips.count) * config.clipLength),
            "audio": [
                "format": "m4a",
                "role": "full_mix",
                "sample_rate": 0,
                "channels": 0,
            ],
            "separation": [
                "enabled": false,
                "status": "not_run",
                "available_assets": ["full_mix"],
            ],
            "transcription": [
                "enabled": false,
                "status": "not_run",
            ],
            "analysis": [
                "enabled": false,
                "status": "not_run",
            ],
            "training": [
                "default_bundle": trainingBundleFileName,
                "full_mix_bundle": fullMixBundleFileName,
                "dataset_config": datasetConfigFileName,
            ],
            "remote_packaging": [
                "status": "ready_for_upload",
                "descriptor": remotePackageFileName,
                "provider_targets": ["runpod", "digitalocean_gpu"],
            ],
            "clips": clips,
        ]
    }

    private static func trainingBundle(clips: [[String: Any]], createdAt: String) -> [String: Any] {
        [
            "anvil_training_bundle_version": "1.0",
            "created_at": createdAt,
            "dataset_dir": ".",
            "include": ["full_mix"],
            "training_lyrics": "[Instrumental]",
            "training_lyrics_source": "constant",
            "clip_count": clips.count,
            "clips": clips,
        ]
    }

    private static func remotePackageDescriptor(
        outputDirectory: URL,
        clipCount: Int,
        createdAt: String
    ) -> [String: Any] {
        [
            "anvil_remote_package_version": "1.0",
            "created_at": createdAt,
            "package_name": outputDirectory.lastPathComponent,
            "package_root": ".",
            "clip_count": clipCount,
            "default_bundle": fullMixBundleFileName,
            "bundles": [
                [
                    "name": "full_mix",
                    "path": fullMixBundleFileName,
                    "include": ["full_mix"],
                    "training_lyrics": "[Instrumental]",
                    "training_lyrics_source": "constant",
                ],
            ],
            "provider_targets": ["runpod", "digitalocean_gpu"],
            "copy_rules": [
                "include": [
                    "clips/**",
                    captionsFileName,
                    datasetConfigFileName,
                    datasetManifestFileName,
                    trainingBundleFileName,
                    fullMixBundleFileName,
                    remotePackageFileName,
                ],
                "exclude": [],
            ],
            "worker_contract": [
                "resolve_relative_paths_from": "package_root",
                "entry_bundle": fullMixBundleFileName,
                "secrets_required": [],
                "expected_gpu_class": "h100_or_h200",
            ],
            "notes": [
                "This first package is local-file/full-mix only; stem separation, transcription, BPM, and key analysis have not been run.",
                "Remote workers should copy the package root intact and resolve clip assets relative to the selected bundle file.",
            ],
        ]
    }

    private static func packageClips(
        from clips: [TrainingImportedFile],
        sourceFiles: [URL],
        outputDirectory: URL,
        config: LocalFilesDatasetConfig
    ) -> [[String: Any]] {
        clips.enumerated().map { index, clip in
            let relativePath = relativeClipPath(for: clip.url, from: outputDirectory)
            let caption = normalizedCaption(for: clip)
            return [
                "id": String(format: "clip_%04d", index + 1),
                "file": relativePath,
                "assets": [
                    "full_mix": relativePath,
                ],
                "caption": caption,
                "prompt": caption,
                "lyrics": "",
                "transcript": "",
                "transcription": [
                    "status": "not_run",
                    "text": "",
                ],
                "analysis": [
                    "status": "not_run",
                ],
                "clip_length_seconds": seconds(config.clipLength),
                "source": sourceMetadata(for: clip.url, sourceFiles: sourceFiles),
                "tags": [],
                "negative_tags": [],
            ]
        }
    }

    private static func normalizedCaption(for clip: TrainingImportedFile) -> String {
        let trimmed = clip.description.trimmingCharacters(in: .whitespacesAndNewlines)
        if !trimmed.isEmpty {
            return trimmed
        }
        return clip.url.deletingPathExtension().lastPathComponent
    }

    private static func sourceMetadata(for clipURL: URL, sourceFiles: [URL]) -> [String: Any] {
        var metadata: [String: Any] = [
            "kind": "local_file",
        ]
        if let source = sourceFileName(for: clipURL, sourceFiles: sourceFiles) {
            metadata["file_name"] = source
        }
        return metadata
    }

    private static func sourceFileName(for clipURL: URL, sourceFiles: [URL]) -> String? {
        guard !sourceFiles.isEmpty else { return nil }
        if let index = sourceIndex(from: clipURL.lastPathComponent),
           sourceFiles.indices.contains(index - 1) {
            return sourceFiles[index - 1].lastPathComponent
        }
        return sourceFiles.count == 1 ? sourceFiles[0].lastPathComponent : nil
    }

    private static func sourceIndex(from fileName: String) -> Int? {
        guard fileName.first == "f" else { return nil }
        let digits = fileName.dropFirst().prefix { $0.isNumber }
        guard !digits.isEmpty else { return nil }
        return Int(digits)
    }

    private static func relativeClipPath(for url: URL, from baseDirectory: URL) -> String {
        let basePath = normalizedDirectoryPath(baseDirectory)
        let filePath = url.standardizedFileURL.path
        if filePath.hasPrefix(basePath) {
            return String(filePath.dropFirst(basePath.count))
        }
        return "clips/\(url.lastPathComponent)"
    }

    private static func normalizedDirectoryPath(_ url: URL) -> String {
        var path = url.standardizedFileURL.path
        if !path.hasSuffix("/") {
            path.append("/")
        }
        return path
    }

    private static func seconds(_ value: TimeInterval) -> Double {
        (value * 1000).rounded() / 1000
    }

    private static func writeCaptionsFile(clips: [(url: URL, caption: String)], to outputDir: URL) throws {
        let entries = clips.map {
            [
                "file": "clips/\($0.url.lastPathComponent)",
                "caption": $0.caption,
            ]
        }
        try writeJSONObjectArray(entries, to: outputDir.appendingPathComponent(captionsFileName))
    }

    private static func writeJSONObject(_ object: [String: Any], to url: URL) throws {
        let data = try JSONSerialization.data(withJSONObject: object, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: url, options: .atomic)
    }

    private static func writeJSONObjectArray(_ object: [[String: Any]], to url: URL) throws {
        let data = try JSONSerialization.data(withJSONObject: object, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: url, options: .atomic)
    }
}
