import Testing
import Foundation
@testable import AnvilApp
@testable import AnvilCore

private func silentWAVData(sampleRate: Int = 22_050, channels: Int = 1, frameCount: Int = 2_205) -> Data {
    let bitsPerSample = 16
    let blockAlign = channels * bitsPerSample / 8
    let byteRate = sampleRate * blockAlign
    let dataSize = frameCount * blockAlign
    var data = Data()

    func appendString(_ value: String) {
        data.append(Data(value.utf8))
    }

    func appendUInt16(_ value: UInt16) {
        var littleEndian = value.littleEndian
        withUnsafeBytes(of: &littleEndian) { data.append(contentsOf: $0) }
    }

    func appendUInt32(_ value: UInt32) {
        var littleEndian = value.littleEndian
        withUnsafeBytes(of: &littleEndian) { data.append(contentsOf: $0) }
    }

    appendString("RIFF")
    appendUInt32(UInt32(36 + dataSize))
    appendString("WAVE")
    appendString("fmt ")
    appendUInt32(16)
    appendUInt16(1)
    appendUInt16(UInt16(channels))
    appendUInt32(UInt32(sampleRate))
    appendUInt32(UInt32(byteRate))
    appendUInt16(UInt16(blockAlign))
    appendUInt16(UInt16(bitsPerSample))
    appendString("data")
    appendUInt32(UInt32(dataSize))
    data.append(Data(repeating: 0, count: dataSize))
    return data
}

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
    for spec in await registry.allSpecs {
        let dir = spec.localDirectory(base: tempDir)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        for file in spec.files {
            try Data("stub".utf8).write(to: spec.localPath(for: file.localName, base: tempDir))
        }
    }

    let state = AppState(registry: registry)
    await state.checkSetupStatus()
    #expect(state.isSetupComplete == true)
}

@Test func trainingDatasetImporterUsesVocalStemTranscriptAsLyrics() throws {
    let datasetDir = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString, isDirectory: true)
    let stemDir = datasetDir
        .appendingPathComponent("stems", isDirectory: true)
        .appendingPathComponent("clip_0001", isDirectory: true)
    try FileManager.default.createDirectory(at: stemDir, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: datasetDir) }

    let vocalURL = stemDir.appendingPathComponent("vocals.wav")
    try Data("stub".utf8).write(to: vocalURL)

    let bundleURL = datasetDir.appendingPathComponent("training_bundle_vocals.json")
    let json = """
    {
      "dataset_dir": "\(datasetDir.path)",
      "include": ["vocals"],
      "clips": [
        {
          "assets": { "vocals": "stems/clip_0001/vocals.wav" },
          "prompt": "dark blues vocal",
          "transcription": { "text": "I sing from the shadow" }
        }
      ]
    }
    """
    try Data(json.utf8).write(to: bundleURL)

    let files = try TrainingDatasetImporter.importFiles(from: bundleURL)

    #expect(files.count == 1)
    #expect(files[0].url.path == vocalURL.path)
    #expect(files[0].description == "dark blues vocal")
    #expect(files[0].lyrics == "I sing from the shadow")
}

@Test func trainingDatasetImporterLeavesInstrumentalLyricsEmpty() throws {
    let datasetDir = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString, isDirectory: true)
    let stemDir = datasetDir
        .appendingPathComponent("stems", isDirectory: true)
        .appendingPathComponent("clip_0001", isDirectory: true)
    try FileManager.default.createDirectory(at: stemDir, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: datasetDir) }

    let instrumentalURL = stemDir.appendingPathComponent("instrumental.wav")
    try Data("stub".utf8).write(to: instrumentalURL)

    let bundleURL = datasetDir.appendingPathComponent("training_bundle_instrumental.json")
    let json = """
    {
      "dataset_dir": "\(datasetDir.path)",
      "include": ["instrumental"],
      "clips": [
        {
          "assets": { "instrumental": "stems/clip_0001/instrumental.wav" },
          "caption": "dark blues backing track",
          "transcript": "This lyric should not condition the instrumental stem"
        }
      ]
    }
    """
    try Data(json.utf8).write(to: bundleURL)

    let files = try TrainingDatasetImporter.importFiles(from: bundleURL)

    #expect(files.count == 1)
    #expect(files[0].url.path == instrumentalURL.path)
    #expect(files[0].description == "dark blues backing track")
    #expect(files[0].lyrics.isEmpty)
}

@Test func trainingDatasetImporterUsesAdjacentCaptionsForDatasetConfig() throws {
    let datasetDir = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString, isDirectory: true)
    let clipsDir = datasetDir.appendingPathComponent("clips", isDirectory: true)
    try FileManager.default.createDirectory(at: clipsDir, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: datasetDir) }

    let clipURL = clipsDir.appendingPathComponent("clip_0001.wav")
    try Data("stub".utf8).write(to: clipURL)

    let captionsJSON = """
    [
      {
        "file": "clips/clip_0001.wav",
        "caption": "smoky blues full mix",
        "transcription": { "text": "ignore full mix lyric" }
      }
    ]
    """
    try Data(captionsJSON.utf8).write(to: datasetDir.appendingPathComponent("captions.json"))

    let configJSON = """
    {
      "dataset_type": "audio_dir",
      "datasets": [
        { "path": "clips" }
      ]
    }
    """
    let configURL = datasetDir.appendingPathComponent("dataset_config.json")
    try Data(configJSON.utf8).write(to: configURL)

    let files = try TrainingDatasetImporter.importFiles(from: configURL)

    #expect(files.count == 1)
    #expect(files[0].url.path == clipURL.path)
    #expect(files[0].description == "smoky blues full mix")
    #expect(files[0].lyrics.isEmpty)
}

@Test func trainingDatasetImporterUsesDatasetConfigCaptionFileForVocalLyrics() throws {
    let datasetDir = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString, isDirectory: true)
    let stemDir = datasetDir
        .appendingPathComponent("stems", isDirectory: true)
        .appendingPathComponent("clip_0001", isDirectory: true)
    let metadataDir = datasetDir.appendingPathComponent("metadata", isDirectory: true)
    try FileManager.default.createDirectory(at: stemDir, withIntermediateDirectories: true)
    try FileManager.default.createDirectory(at: metadataDir, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: datasetDir) }

    let vocalURL = stemDir.appendingPathComponent("vocals.wav")
    try Data("stub".utf8).write(to: vocalURL)

    let captionsJSON = """
    [
      {
        "file": "stems/clip_0001/vocals.wav",
        "prompt": "dark blues vocal stem",
        "transcription": { "text": "I kept singing in the dark" }
      }
    ]
    """
    try Data(captionsJSON.utf8).write(to: metadataDir.appendingPathComponent("captions.json"))

    let configJSON = """
    {
      "dataset_type": "audio_dir",
      "datasets": [
        {
          "path": "stems",
          "caption_file": "metadata/captions.json"
        }
      ]
    }
    """
    let configURL = datasetDir.appendingPathComponent("dataset_config.json")
    try Data(configJSON.utf8).write(to: configURL)

    let files = try TrainingDatasetImporter.importFiles(from: configURL)

    #expect(files.count == 1)
    #expect(files[0].url.path == vocalURL.path)
    #expect(files[0].description == "dark blues vocal stem")
    #expect(files[0].lyrics == "I kept singing in the dark")
}

@Test func trainingManifestDecodesImportedFilesWithoutLyrics() throws {
    let json = """
    {
      "historyEntryIDs": [],
      "importedFiles": [
        {
          "url": "file:///tmp/example.wav",
          "description": "legacy caption"
        }
      ]
    }
    """

    let manifest = try JSONDecoder().decode(TrainingManifest.self, from: Data(json.utf8))

    #expect(manifest.importedFiles.count == 1)
    #expect(manifest.importedFiles[0].description == "legacy caption")
    #expect(manifest.importedFiles[0].lyrics == "")
}

@Test func localDatasetPackageWriterEmitsImportableFullMixBundle() throws {
    let datasetDir = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString, isDirectory: true)
    let clipsDir = datasetDir.appendingPathComponent("clips", isDirectory: true)
    try FileManager.default.createDirectory(at: clipsDir, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: datasetDir) }

    let sourceURL = datasetDir.appendingPathComponent("source.wav")
    let clipURL = clipsDir.appendingPathComponent("f001_clip_001.m4a")
    try Data("source".utf8).write(to: sourceURL)
    try Data("clip".utf8).write(to: clipURL)

    try TrainingDatasetPackageWriter.writeLocalPackage(
        outputDirectory: datasetDir,
        clips: [TrainingImportedFile(url: clipURL, description: "dark blues full mix")],
        sourceFiles: [sourceURL],
        config: LocalFilesDatasetConfig(clipLength: 30, clipsPerTrack: 1, maxTotalClips: 1),
        styleTemplate: "dark blues full mix",
        createdAt: Date(timeIntervalSince1970: 0)
    )

    for fileName in [
        "captions.json",
        "dataset_config.json",
        "dataset_manifest.json",
        "training_bundle.json",
        "training_bundle_full_mix.json",
        "remote_training_package.json",
        "package_validation.json",
    ] {
        let fileURL = datasetDir.appendingPathComponent(fileName)
        #expect(FileManager.default.fileExists(atPath: fileURL.path))
    }

    let validation = TrainingDatasetPackageWriter.validatePackage(at: datasetDir)
    #expect(validation.isValid)
    #expect(validation.clipCount == 1)
    #expect(validation.importableClipCount == 1)

    let imported = try TrainingDatasetImporter.importFiles(
        from: datasetDir.appendingPathComponent("training_bundle_full_mix.json")
    )
    #expect(imported.count == 1)
    #expect(imported[0].url.path == clipURL.path)
    #expect(imported[0].description == "dark blues full mix")
    #expect(imported[0].lyrics.isEmpty)

    let remoteData = try Data(contentsOf: datasetDir.appendingPathComponent("remote_training_package.json"))
    let remoteJSON = try JSONSerialization.jsonObject(with: remoteData) as? [String: Any]
    let providerTargets = remoteJSON?["provider_targets"] as? [String] ?? []
    #expect(providerTargets.contains("runpod"))
    #expect(providerTargets.contains("digitalocean_gpu"))
    #expect(remoteJSON?["default_bundle"] as? String == "training_bundle_full_mix.json")
    let remoteBundles = remoteJSON?["bundles"] as? [[String: Any]] ?? []
    let remoteBundleNames = Set(remoteBundles.compactMap { $0["name"] as? String })
    #expect(remoteBundleNames.contains("full_mix"))
    #expect(remoteBundleNames.contains("vocals"))
    #expect(remoteBundleNames.contains("instrumental"))
    let copyRules = remoteJSON?["copy_rules"] as? [String: Any]
    let includedPaths = copyRules?["include"] as? [String] ?? []
    #expect(includedPaths.contains("stems/**"))
    #expect(includedPaths.contains("training_bundle_vocals.json"))
    #expect(includedPaths.contains("training_bundle_instrumental.json"))

    let manifestData = try Data(contentsOf: datasetDir.appendingPathComponent("dataset_manifest.json"))
    let manifestJSON = try JSONSerialization.jsonObject(with: manifestData) as? [String: Any]
    let audioJSON = manifestJSON?["audio"] as? [String: Any]
    #expect(audioJSON?["metadata_status"] as? String == "audio_properties_unavailable")
    #expect(audioJSON?["format"] as? String == "m4a")

    let sourceJSON = manifestJSON?["source"] as? [String: Any]
    let sourceFilesJSON = sourceJSON?["files"] as? [[String: Any]]
    #expect(sourceFilesJSON?.first?["file_name"] as? String == "source.wav")
    #expect((sourceFilesJSON?.first?["byte_size"] as? NSNumber)?.intValue == 6)

    let clipsJSON = manifestJSON?["clips"] as? [[String: Any]]
    let clipAudioJSON = clipsJSON?.first?["audio"] as? [String: Any]
    #expect(clipAudioJSON?["format"] as? String == "m4a")
    #expect(clipAudioJSON?["metadata_status"] as? String == "unreadable_audio_header")
    #expect((clipAudioJSON?["byte_size"] as? NSNumber)?.intValue == 4)
}

@Test func localDatasetPackageWriterReadsAudioHeadersWhenAvailable() throws {
    let datasetDir = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString, isDirectory: true)
    let clipsDir = datasetDir.appendingPathComponent("clips", isDirectory: true)
    try FileManager.default.createDirectory(at: clipsDir, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: datasetDir) }

    let sourceURL = datasetDir.appendingPathComponent("source.wav")
    let clipURL = clipsDir.appendingPathComponent("f001_clip_001.wav")
    try silentWAVData(sampleRate: 22_050, channels: 1, frameCount: 2_205).write(to: sourceURL)
    try silentWAVData(sampleRate: 22_050, channels: 1, frameCount: 2_205).write(to: clipURL)

    try TrainingDatasetPackageWriter.writeLocalPackage(
        outputDirectory: datasetDir,
        clips: [TrainingImportedFile(url: clipURL, description: "readable wav")],
        sourceFiles: [sourceURL],
        config: LocalFilesDatasetConfig(clipLength: 30, clipsPerTrack: 1, maxTotalClips: 1),
        styleTemplate: "",
        createdAt: Date(timeIntervalSince1970: 0)
    )

    let manifestData = try Data(contentsOf: datasetDir.appendingPathComponent("dataset_manifest.json"))
    let manifestJSON = try JSONSerialization.jsonObject(with: manifestData) as? [String: Any]
    let audioJSON = manifestJSON?["audio"] as? [String: Any]
    let clipsJSON = manifestJSON?["clips"] as? [[String: Any]]
    let clipAudioJSON = clipsJSON?.first?["audio"] as? [String: Any]

    #expect(audioJSON?["metadata_status"] as? String == "audio_properties_collected")
    #expect(audioJSON?["format"] as? String == "wav")
    #expect((audioJSON?["sample_rate"] as? NSNumber)?.intValue == 22_050)
    #expect((audioJSON?["channels"] as? NSNumber)?.intValue == 1)
    #expect(clipAudioJSON?["metadata_status"] as? String == "ok")
    #expect((clipAudioJSON?["sample_rate"] as? NSNumber)?.intValue == 22_050)
    #expect((clipAudioJSON?["channels"] as? NSNumber)?.intValue == 1)
    let duration = (clipAudioJSON?["duration_seconds"] as? NSNumber)?.doubleValue ?? 0
    #expect(abs(duration - 0.1) < 0.001)
}

@Test func localDatasetPackageWriterPreservesEnrichmentMetadataOnCaptionRefresh() throws {
    let datasetDir = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString, isDirectory: true)
    let clipsDir = datasetDir.appendingPathComponent("clips", isDirectory: true)
    try FileManager.default.createDirectory(at: clipsDir, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: datasetDir) }

    let sourceURL = datasetDir.appendingPathComponent("source.wav")
    let clipURL = clipsDir.appendingPathComponent("f001_clip_001.wav")
    try silentWAVData().write(to: sourceURL)
    try silentWAVData().write(to: clipURL)

    try TrainingDatasetPackageWriter.writeLocalPackage(
        outputDirectory: datasetDir,
        clips: [TrainingImportedFile(url: clipURL, description: "old caption")],
        sourceFiles: [sourceURL],
        config: LocalFilesDatasetConfig(clipLength: 30, clipsPerTrack: 1, maxTotalClips: 1),
        styleTemplate: "old caption",
        createdAt: Date(timeIntervalSince1970: 0)
    )

    let enrichedCaptions = """
    [
      {
        "file": "clips/f001_clip_001.wav",
        "caption": "old caption",
        "transcript": "I keep the line",
        "transcription": { "status": "complete", "text": "I keep the line" },
        "analysis": { "tempo_bpm_estimate": 72 },
        "separation": {
          "mode": "instrumental",
          "stems": {
            "vocals": "stems/f001_clip_001/vocals.wav",
            "instrumental": "stems/f001_clip_001/instrumental.wav"
          },
          "separation_file": "stems/f001_clip_001/separation.json"
        }
      }
    ]
    """
    try Data(enrichedCaptions.utf8).write(to: datasetDir.appendingPathComponent("captions.json"))

    try TrainingDatasetPackageWriter.writeLocalPackage(
        outputDirectory: datasetDir,
        clips: [TrainingImportedFile(url: clipURL, description: "new caption")],
        sourceFiles: [sourceURL],
        config: LocalFilesDatasetConfig(clipLength: 30, clipsPerTrack: 1, maxTotalClips: 1),
        styleTemplate: "new caption",
        createdAt: Date(timeIntervalSince1970: 1)
    )

    let captionsData = try Data(contentsOf: datasetDir.appendingPathComponent("captions.json"))
    let captions = try JSONSerialization.jsonObject(with: captionsData) as? [[String: Any]]
    let record = captions?.first
    #expect(record?["caption"] as? String == "new caption")
    #expect(record?["transcript"] as? String == "I keep the line")
    let transcription = record?["transcription"] as? [String: Any]
    #expect(transcription?["text"] as? String == "I keep the line")
    let separation = record?["separation"] as? [String: Any]
    let stems = separation?["stems"] as? [String: Any]
    #expect(stems?["vocals"] as? String == "stems/f001_clip_001/vocals.wav")

    let bundleData = try Data(contentsOf: datasetDir.appendingPathComponent("training_bundle_full_mix.json"))
    let bundle = try JSONSerialization.jsonObject(with: bundleData) as? [String: Any]
    let bundleClips = bundle?["clips"] as? [[String: Any]]
    let bundleClip = bundleClips?.first
    #expect(bundleClip?["caption"] as? String == "new caption")
    #expect((bundleClip?["separation"] as? [String: Any])?["mode"] as? String == "instrumental")
}

@Test func localDatasetPackageValidationRefreshesDerivedEnrichmentManifestMetadata() throws {
    let datasetDir = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString, isDirectory: true)
    let clipsDir = datasetDir.appendingPathComponent("clips", isDirectory: true)
    let stemsDir = datasetDir
        .appendingPathComponent("stems", isDirectory: true)
        .appendingPathComponent("f001_clip_001", isDirectory: true)
    try FileManager.default.createDirectory(at: clipsDir, withIntermediateDirectories: true)
    try FileManager.default.createDirectory(at: stemsDir, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: datasetDir) }

    let sourceURL = datasetDir.appendingPathComponent("source.wav")
    let clipURL = clipsDir.appendingPathComponent("f001_clip_001.wav")
    let vocalURL = stemsDir.appendingPathComponent("vocals.wav")
    let instrumentalURL = stemsDir.appendingPathComponent("instrumental.wav")
    try silentWAVData().write(to: sourceURL)
    try silentWAVData().write(to: clipURL)
    try silentWAVData().write(to: vocalURL)
    try silentWAVData().write(to: instrumentalURL)

    try TrainingDatasetPackageWriter.writeLocalPackage(
        outputDirectory: datasetDir,
        clips: [TrainingImportedFile(url: clipURL, description: "dark blues")],
        sourceFiles: [sourceURL],
        config: LocalFilesDatasetConfig(clipLength: 30, clipsPerTrack: 1, maxTotalClips: 1),
        styleTemplate: "dark blues",
        createdAt: Date(timeIntervalSince1970: 0)
    )

    let separationJSON = """
    {
      "stems": {
        "vocals": {
          "file": "stems/f001_clip_001/vocals.wav",
          "analysis": { "tempo_bpm_estimate": 72 }
        },
        "instrumental": {
          "file": "stems/f001_clip_001/instrumental.wav",
          "analysis": { "tempo_bpm_estimate": 70 }
        }
      }
    }
    """
    try Data(separationJSON.utf8).write(to: stemsDir.appendingPathComponent("separation.json"))

    let captionsJSON = """
    [
      {
        "file": "clips/f001_clip_001.wav",
        "caption": "dark blues",
        "lyrics": "I keep singing low",
        "transcript": "I keep singing low",
        "transcription": {
          "status": "complete",
          "text": "I keep singing low",
          "backend": "test-qwen3-asr",
          "model": "fake-qwen3-asr"
        },
        "separation": {
          "mode": "instrumental",
          "stems": {
            "vocals": "stems/f001_clip_001/vocals.wav",
            "instrumental": "stems/f001_clip_001/instrumental.wav"
          },
          "separation_file": "stems/f001_clip_001/separation.json"
        }
      }
    ]
    """
    try Data(captionsJSON.utf8).write(to: datasetDir.appendingPathComponent("captions.json"))
    _ = try LocalDatasetEnrichmentService.exportStemTrainingBundles(datasetDirectory: datasetDir)

    try TrainingDatasetPackageWriter.refreshValidationReport(
        at: datasetDir,
        checkedAt: Date(timeIntervalSince1970: 1)
    )

    let manifestData = try Data(contentsOf: datasetDir.appendingPathComponent("dataset_manifest.json"))
    let manifest = try JSONSerialization.jsonObject(with: manifestData) as? [String: Any]
    let separation = manifest?["separation"] as? [String: Any]
    #expect(separation?["status"] as? String == "complete")
    let availableAssets = separation?["available_assets"] as? [String] ?? []
    #expect(availableAssets.contains("vocals"))
    #expect(availableAssets.contains("instrumental"))
    let transcription = manifest?["transcription"] as? [String: Any]
    #expect(transcription?["status"] as? String == "complete")
    let models = transcription?["models"] as? [String] ?? []
    #expect(models.contains("fake-qwen3-asr"))
    let analysis = manifest?["analysis"] as? [String: Any]
    #expect(analysis?["bpm_status"] as? String == "estimated_from_stems")
    let enrichment = manifest?["enrichment"] as? [String: Any]
    #expect(enrichment?["status"] as? String == "complete")
    #expect((enrichment?["separated_clip_count"] as? NSNumber)?.intValue == 1)
    let training = manifest?["training"] as? [String: Any]
    #expect(training?["vocal_bundle"] as? String == "training_bundle_vocals.json")
    #expect(training?["instrumental_bundle"] as? String == "training_bundle_instrumental.json")
}

@Test func localDatasetEnrichmentUsesNativeDemucsOnly() {
    #expect(DemucsMLXSourceSeparator.backendName == "demucs-mlx-swift")
    #expect(DemucsMLXSourceSeparator.defaultModelName == "htdemucs")
}

@Test func localDatasetEnrichmentTranscribesCachedVocalStems() async throws {
    let datasetDir = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString, isDirectory: true)
    let clipsDir = datasetDir.appendingPathComponent("clips", isDirectory: true)
    let stemsDir = datasetDir
        .appendingPathComponent("stems", isDirectory: true)
        .appendingPathComponent("f001_clip_001", isDirectory: true)
    try FileManager.default.createDirectory(at: clipsDir, withIntermediateDirectories: true)
    try FileManager.default.createDirectory(at: stemsDir, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: datasetDir) }

    let clipURL = clipsDir.appendingPathComponent("f001_clip_001.wav")
    let vocalURL = stemsDir.appendingPathComponent("vocals.wav")
    let instrumentalURL = stemsDir.appendingPathComponent("instrumental.wav")
    try silentWAVData().write(to: clipURL)
    try silentWAVData().write(to: vocalURL)
    try silentWAVData().write(to: instrumentalURL)

    let separationJSON = """
    {
      "stems": {
        "vocals": {
          "file": "stems/f001_clip_001/vocals.wav",
          "analysis": { "tempo_bpm_estimate": 72 }
        },
        "instrumental": {
          "file": "stems/f001_clip_001/instrumental.wav",
          "analysis": { "tempo_bpm_estimate": 72 }
        }
      }
    }
    """
    try Data(separationJSON.utf8).write(to: stemsDir.appendingPathComponent("separation.json"))

    let captionsJSON = """
    [
      {
        "file": "clips/f001_clip_001.wav",
        "caption": "dark blues stem package",
        "separation": {
          "mode": "instrumental",
          "stems": {
            "vocals": "stems/f001_clip_001/vocals.wav",
            "instrumental": "stems/f001_clip_001/instrumental.wav"
          },
          "separation_file": "stems/f001_clip_001/separation.json"
        }
      }
    ]
    """
    try Data(captionsJSON.utf8).write(to: datasetDir.appendingPathComponent("captions.json"))

    let lyricsTranscriber = FakeDatasetLyricsTranscriber()
    let result = try await LocalDatasetEnrichmentService.run(
        datasetDirectory: datasetDir,
        lyricsTranscriber: lyricsTranscriber,
        onStep: { _ in }
    )

    #expect(result.transcribedClipCount == 1)
    #expect(await lyricsTranscriber.evictionCount == 1)

    let captionsData = try Data(contentsOf: datasetDir.appendingPathComponent("captions.json"))
    let captions = try JSONSerialization.jsonObject(with: captionsData) as? [[String: Any]]
    let record = captions?.first
    #expect(record?["lyrics"] as? String == "I keep singing low")
    #expect(record?["transcript"] as? String == "I keep singing low")
    let transcription = record?["transcription"] as? [String: Any]
    #expect(transcription?["backend"] as? String == "test-qwen3-asr")
    #expect(transcription?["model"] as? String == "fake-qwen3-asr")
    #expect(transcription?["lyrics_file"] as? String == "stems/f001_clip_001/lyrics.txt")
    let assets = record?["assets"] as? [String: Any]
    #expect(assets?["timed_lyrics"] as? String == "stems/f001_clip_001/lyrics.timed.json")

    let vocalImported = try TrainingDatasetImporter.importFiles(
        from: datasetDir.appendingPathComponent("training_bundle_vocals.json")
    )
    #expect(vocalImported.count == 1)
    #expect(vocalImported[0].lyrics == "I keep singing low")

    let manifestData = try Data(contentsOf: datasetDir.appendingPathComponent("dataset_manifest.json"))
    let manifest = try JSONSerialization.jsonObject(with: manifestData) as? [String: Any]
    let manifestTranscription = manifest?["transcription"] as? [String: Any]
    #expect(manifestTranscription?["status"] as? String == "complete")
    let models = manifestTranscription?["models"] as? [String] ?? []
    #expect(models.contains("fake-qwen3-asr"))
}

@Test func localDatasetEnrichmentExportsStemBundlesNatively() throws {
    let datasetDir = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString, isDirectory: true)
    let clipsDir = datasetDir.appendingPathComponent("clips", isDirectory: true)
    let stemsDir = datasetDir
        .appendingPathComponent("stems", isDirectory: true)
        .appendingPathComponent("f001_clip_001", isDirectory: true)
    try FileManager.default.createDirectory(at: clipsDir, withIntermediateDirectories: true)
    try FileManager.default.createDirectory(at: stemsDir, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: datasetDir) }

    let clipURL = clipsDir.appendingPathComponent("f001_clip_001.wav")
    let vocalURL = stemsDir.appendingPathComponent("vocals.wav")
    let instrumentalURL = stemsDir.appendingPathComponent("instrumental.wav")
    try Data("clip".utf8).write(to: clipURL)
    try Data("vocals".utf8).write(to: vocalURL)
    try Data("instrumental".utf8).write(to: instrumentalURL)

    let separationJSON = """
    {
      "stems": {
        "vocals": {
          "file": "stems/f001_clip_001/vocals.wav",
          "analysis": { "tempo_bpm_estimate": 72 }
        },
        "instrumental": {
          "file": "stems/f001_clip_001/instrumental.wav",
          "analysis": { "tempo_bpm_estimate": 72 }
        }
      }
    }
    """
    try Data(separationJSON.utf8).write(to: stemsDir.appendingPathComponent("separation.json"))

    let captionsJSON = """
    [
      {
        "file": "clips/f001_clip_001.wav",
        "caption": "dark blues stem package",
        "transcription": { "status": "complete", "text": "I keep singing low" },
        "separation": {
          "mode": "instrumental",
          "stems": {
            "vocals": "stems/f001_clip_001/vocals.wav",
            "instrumental": "stems/f001_clip_001/instrumental.wav"
          },
          "separation_file": "stems/f001_clip_001/separation.json"
        }
      }
    ]
    """
    try Data(captionsJSON.utf8).write(to: datasetDir.appendingPathComponent("captions.json"))

    let counts = try LocalDatasetEnrichmentService.exportStemTrainingBundles(datasetDirectory: datasetDir)

    #expect(counts.vocalCount == 1)
    #expect(counts.instrumentalCount == 1)

    let vocalImported = try TrainingDatasetImporter.importFiles(
        from: datasetDir.appendingPathComponent("training_bundle_vocals.json")
    )
    #expect(vocalImported.count == 1)
    #expect(vocalImported[0].url.path == vocalURL.path)
    #expect(vocalImported[0].description == "dark blues stem package")
    #expect(vocalImported[0].lyrics == "I keep singing low")

    let instrumentalImported = try TrainingDatasetImporter.importFiles(
        from: datasetDir.appendingPathComponent("training_bundle_instrumental.json")
    )
    #expect(instrumentalImported.count == 1)
    #expect(instrumentalImported[0].url.path == instrumentalURL.path)
    #expect(instrumentalImported[0].description == "dark blues stem package")
    #expect(instrumentalImported[0].lyrics.isEmpty)
}

@Test @MainActor func localFilesDatasetBuilderWritesWAVClipsByDefault() async throws {
    let tempRoot = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString, isDirectory: true)
    try FileManager.default.createDirectory(at: tempRoot, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: tempRoot) }

    let sourceURL = tempRoot.appendingPathComponent("source.wav")
    try silentWAVData(sampleRate: 22_050, channels: 1, frameCount: 6_615).write(to: sourceURL)

    let result = try await LocalFilesDatasetBuilder.run(
        sourceFiles: [sourceURL],
        config: LocalFilesDatasetConfig(clipLength: 0.1, clipsPerTrack: 2, maxTotalClips: 2),
        outputBase: tempRoot,
        styleTemplate: "lossless training test",
        onPhase: { _ in }
    )

    #expect(result.clipCount == 2)
    #expect(result.clips.allSatisfy { $0.url.pathExtension == "wav" })

    let validation = TrainingDatasetPackageWriter.validatePackage(at: result.outputDirectory)
    #expect(validation.isValid)
    #expect(validation.importableClipCount == 2)

    let bundleURL = result.outputDirectory.appendingPathComponent("training_bundle_full_mix.json")
    let imported = try TrainingDatasetImporter.importFiles(from: bundleURL)
    #expect(imported.count == 2)
    #expect(imported.allSatisfy { $0.url.pathExtension == "wav" })

    let manifestData = try Data(contentsOf: result.outputDirectory.appendingPathComponent("dataset_manifest.json"))
    let manifestJSON = try JSONSerialization.jsonObject(with: manifestData) as? [String: Any]
    let audioJSON = manifestJSON?["audio"] as? [String: Any]
    #expect(audioJSON?["format"] as? String == "wav")
    #expect(audioJSON?["metadata_status"] as? String == "audio_properties_collected")
}

@Test func localDatasetPackageValidationReportsMissingClipFiles() throws {
    let datasetDir = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString, isDirectory: true)
    let clipsDir = datasetDir.appendingPathComponent("clips", isDirectory: true)
    try FileManager.default.createDirectory(at: clipsDir, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: datasetDir) }

    let sourceURL = datasetDir.appendingPathComponent("source.wav")
    let clipURL = clipsDir.appendingPathComponent("f001_clip_001.m4a")
    try Data("source".utf8).write(to: sourceURL)
    try Data("clip".utf8).write(to: clipURL)

    try TrainingDatasetPackageWriter.writeLocalPackage(
        outputDirectory: datasetDir,
        clips: [TrainingImportedFile(url: clipURL, description: "dark blues full mix")],
        sourceFiles: [sourceURL],
        config: LocalFilesDatasetConfig(clipLength: 30, clipsPerTrack: 1, maxTotalClips: 1),
        styleTemplate: "dark blues full mix",
        createdAt: Date(timeIntervalSince1970: 0)
    )
    try FileManager.default.removeItem(at: clipURL)

    let validation = TrainingDatasetPackageWriter.validatePackage(at: datasetDir)

    #expect(!validation.isValid)
    #expect(validation.clipCount == 1)
    #expect(validation.importableClipCount == 0)
    #expect(validation.missingClipFiles == ["clips/f001_clip_001.m4a"])
}

@Test func localAudioFileDiscoveryFindsNestedSupportedAudioFiles() throws {
    let rootDir = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString, isDirectory: true)
    let nestedDir = rootDir.appendingPathComponent("nested", isDirectory: true)
    try FileManager.default.createDirectory(at: nestedDir, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: rootDir) }

    let wavURL = rootDir.appendingPathComponent("take-a.wav")
    let m4aURL = nestedDir.appendingPathComponent("take-b.m4a")
    let ignoredURL = nestedDir.appendingPathComponent("notes.txt")
    try Data("wav".utf8).write(to: wavURL)
    try Data("m4a".utf8).write(to: m4aURL)
    try Data("notes".utf8).write(to: ignoredURL)

    let discovered = LocalAudioFileDiscovery.audioFiles(in: [rootDir])
    let paths = discovered.map { $0.standardizedFileURL.path }
    let expectedPaths = [
        wavURL.standardizedFileURL.path,
        m4aURL.standardizedFileURL.path,
    ].sorted { $0.localizedStandardCompare($1) == .orderedAscending }

    #expect(paths == expectedPaths)
}

@Test func lyricSanitizerRemovesCommentaryAndChordPrefixes() {
    let raw = """
    Here's a 140-second ambient blues-inspired soundscape with a slow tempo.

    [Verse 1]
    (Am) Longing for a love that's lost
    (G) In the shadows, I'm left to cross

    [Chorus]
    (C) I'm drifting on a tide of pain
    (F) Trying to find my way back to love again

    This song has 16 lines, with a total duration of approximately 140 seconds.
    """

    let sanitized = LyricOutputSanitizer.sanitize(raw)

    #expect(sanitized == """
    [Verse 1]
    Longing for a love that's lost
    In the shadows, I'm left to cross

    [Chorus]
    I'm drifting on a tide of pain
    Trying to find my way back to love again
    """)
}

@Test func lyricSanitizerDropsPromptMetadataBeforeLyrics() {
    let raw = """
    Key: A minor
    Tempo: 65 BPM
    Lyrics:

    [Hook]
    Come down slow
    Let the dark wind carry me
    """

    let sanitized = LyricOutputSanitizer.sanitize(raw)

    #expect(sanitized == """
    [Hook]
    Come down slow
    Let the dark wind carry me
    """)
}

@Test func lyricSanitizerRemovesTimestampsAndPerformanceDirections() {
    let raw = """
    [Verse 1]
    [0:00 - 0:10]
    Shadows creep, darkness falls
    [0:10 - 0:20]
    Ghosts of love and memories
    [Ambient fade-out]
    """

    let sanitized = LyricOutputSanitizer.sanitize(raw)

    #expect(sanitized == """
    [Verse 1]
    Shadows creep, darkness falls
    Ghosts of love and memories
    """)
}

@Test func lyricSanitizerWrapsTimestampOnlyLyricsInVerse() {
    let raw = """
    [0:00]
    In the shadows, I wander alone
    [0:05]
    My heart is heavy, my soul is stone

    [0:10]
    I'm searching for a light, a guiding tone
    """

    let sanitized = LyricOutputSanitizer.sanitize(raw)

    #expect(sanitized == """
    [Verse]
    In the shadows, I wander alone
    My heart is heavy, my soul is stone
    I'm searching for a light, a guiding tone
    """)
}

@Test func longLyricPlanRequiresExplicitSongSections() {
    let plan = LyricWritingPlan.make(duration: 200)

    #expect(plan.sectionPlan.contains("[Verse 1]"))
    #expect(plan.sectionPlan.contains("[Chorus]"))
    #expect(plan.sectionPlan.contains("[Verse 2]"))
    #expect(plan.sectionPlan.contains("[Bridge]"))
    #expect(plan.systemPrompt.contains("Do not collapse the whole song into one [Verse]"))
}

@Test func lyricBackendProvidersExcludeHuggingFace() {
    let providerIDs = LyricWriterManager.remoteBackendProviders.map(\.id)

    #expect(providerIDs.contains("anthropic"))
    #expect(providerIDs.contains("openai"))
    #expect(providerIDs.contains("google"))
    #expect(providerIDs.contains("xai"))
    #expect(!providerIDs.contains("huggingface"))
}

private actor FakeDatasetLyricsTranscriber: DatasetLyricsTranscribing, DatasetLyricsTranscriptionEvicting {
    private(set) var evictionCount = 0

    func evictModels() async {
        evictionCount += 1
    }

    func transcribe(
        vocalsURL: URL,
        outputDirectory: URL,
        force: Bool
    ) async throws -> DatasetLyricsTranscriptionArtifact {
        let lyricsURL = outputDirectory.appendingPathComponent("lyrics.txt")
        let timedLyricsURL = outputDirectory.appendingPathComponent("lyrics.timed.json")
        let lrcURL = outputDirectory.appendingPathComponent("lyrics.lrc")
        try "I keep singing low".write(to: lyricsURL, atomically: true, encoding: .utf8)
        try Data("""
        {
          "text": "I keep singing low",
          "words": [
            { "text": "I", "start": 0.1, "end": 0.2 },
            { "text": "keep", "start": 0.2, "end": 0.4 },
            { "text": "singing", "start": 0.4, "end": 0.7 },
            { "text": "low", "start": 0.7, "end": 1.0 }
          ]
        }
        """.utf8).write(to: timedLyricsURL)
        try "[00:00.10]I keep singing low\n".write(to: lrcURL, atomically: true, encoding: .utf8)

        return DatasetLyricsTranscriptionArtifact(
            text: "I keep singing low",
            source: "test",
            backend: "test-qwen3-asr",
            model: "fake-qwen3-asr",
            alignerModel: "fake-aligner",
            lyricsURL: lyricsURL,
            timedLyricsURL: timedLyricsURL,
            lrcURL: lrcURL,
            wordCount: 4,
            elapsedSeconds: 0.25
        )
    }
}

@Test @MainActor func lyricWriterManagerRegistersAuthenticatedCLIBackend() {
    let providerAccessStore = ProviderAccessStore(statuses: [
        "xai": ProviderCLIStatus(
            providerID: "xai",
            displayName: "Grok (xAI)",
            command: "grok",
            version: "grok 0.1.213",
            state: .authenticated,
            detail: "Signed in with grok.com",
            lastChecked: Date()
        )
    ])
    let manager = LyricWriterManager(
        apiKeyStore: APIKeyStore(),
        providerAccessStore: providerAccessStore
    )

    manager.refreshRemoteBackends()

    #expect(manager.availableBackends().contains { $0.id == "xai" })
    #expect(manager.backendRequirementMessage(for: "xai") == nil)
}

@Test func lyricCLIResponseParserReadsProviderJSONResults() {
    let claude = ProviderCommandResult(
        command: "claude",
        arguments: [],
        exitStatus: 0,
        stdout: #"{"result":"[Verse]\nLights up"}"#,
        stderr: ""
    )
    let codex = ProviderCommandResult(
        command: "codex",
        arguments: [],
        exitStatus: 0,
        stdout: """
        {"type":"thread.started","thread_id":"abc"}
        {"type":"item.completed","item":{"type":"agent_message","text":"[Verse]\\nClean room glow"}}
        """,
        stderr: "WARN ignored\n"
    )
    let gemini = ProviderCommandResult(
        command: "gemini",
        arguments: [],
        exitStatus: 0,
        stdout: #"{"response":"[Hook]\nSoft static"}"#,
        stderr: ""
    )
    let grok = ProviderCommandResult(
        command: "grok",
        arguments: [],
        exitStatus: 0,
        stdout: """
        A new version is available.
        {
          "text": "[Chorus]\\nSignal comes alive"
        }
        """,
        stderr: ""
    )

    #expect(LyricCLIResponseParser.responseText(from: claude, providerID: "anthropic") == "[Verse]\nLights up")
    #expect(LyricCLIResponseParser.responseText(from: codex, providerID: "openai") == "[Verse]\nClean room glow")
    #expect(LyricCLIResponseParser.responseText(from: gemini, providerID: "google") == "[Hook]\nSoft static")
    #expect(LyricCLIResponseParser.responseText(from: grok, providerID: "xai") == "[Chorus]\nSignal comes alive")
}

@Test func lyricCLIInvocationsDoNotExposePromptTextInArguments() throws {
    let secret = "PRIVATE_DELTA_BLUES_PROMPT"
    let fullPrompt = "System instructions\n\nSong description: \(secret)"
    let modes: [ProviderCLILyricMode] = [.claude, .codex, .gemini, .grok]

    for mode in modes {
        let invocation = try mode.invocation(
            systemPrompt: "System instructions",
            userPrompt: "Song description: \(secret)",
            fullPrompt: fullPrompt
        )
        defer { invocation.cleanup() }

        let visibleArguments = invocation.arguments.joined(separator: " ")
        #expect(!visibleArguments.contains(secret))
        #expect(!visibleArguments.contains(fullPrompt))

        switch mode {
        case .grok:
            guard let url = invocation.temporaryPromptFileURL else {
                #expect(Bool(false))
                continue
            }
            #expect(invocation.standardInput == nil)
            #expect(try String(contentsOf: url, encoding: .utf8).contains(secret))
            invocation.cleanup()
            #expect(!FileManager.default.fileExists(atPath: url.path))
        default:
            #expect(invocation.standardInput == fullPrompt)
            #expect(invocation.temporaryPromptFileURL == nil)
        }
    }
}

@Test func lyricSanitizerRestructuresOversizedSingleVerseForLongSongs() {
    let raw = """
    [Verse]
    In the shadows, I'll be waiting
    For the ghosts that I've made
    A slow descent into the night
    With the weight of my mistakes
    I'll follow the echoes of my heart
    Through the emptiness I'll depart
    Lost in the haze of a love so cold
    I'll search for a light to hold
    In the dark, I'll find my way
    Through the shards of a broken day
    And I'll whisper my name aloud
    A prayer to a love that's lost
    In the silence, I'll confess
    My heart is a heavy mess
    A skeleton of what used to be
    A ghost of love that's lost its sea
    And I'll wander, lost and alone
    Through the shadows, I'll call home
    """

    let sanitized = LyricOutputSanitizer.sanitize(raw, targetDuration: 200)

    #expect(sanitized == """
    [Verse 1]
    In the shadows, I'll be waiting
    For the ghosts that I've made
    A slow descent into the night
    With the weight of my mistakes

    [Chorus]
    I'll follow the echoes of my heart
    Through the emptiness I'll depart
    Lost in the haze of a love so cold
    I'll search for a light to hold

    [Verse 2]
    In the dark, I'll find my way
    Through the shards of a broken day
    And I'll whisper my name aloud
    A prayer to a love that's lost

    [Chorus]
    In the silence, I'll confess
    My heart is a heavy mess
    A skeleton of what used to be

    [Bridge]
    A ghost of love that's lost its sea
    And I'll wander, lost and alone
    Through the shadows, I'll call home
    """)
}

@Test func traceCompareIncludesCoverSourceAndStrength() {
    let left = traceForComparison(
        sourceAudioFileName: "InDaClub.m4a",
        coverStrength: 0.7
    )
    let right = traceForComparison(
        sourceAudioFileName: "InDaClub-DryRun.m4a",
        coverStrength: 0.35
    )

    let groups = GenerationTraceComparator.groups(left: left, right: right)
    let samplingRows = groups
        .first { $0.title == "Sampling" }?
        .rows
        .reduce(into: [String: TraceComparisonRow]()) { rows, row in
            rows[row.label] = row
        }

    #expect(samplingRows?["Source Audio"]?.left == "InDaClub.m4a")
    #expect(samplingRows?["Source Audio"]?.right == "InDaClub-DryRun.m4a")
    #expect(samplingRows?["Cover Strength"]?.left == "70%")
    #expect(samplingRows?["Cover Strength"]?.right == "35%")
}

private func traceForComparison(
    sourceAudioFileName: String?,
    coverStrength: Float
) -> GenerationTrace {
    GenerationTrace(
        historyEntryID: UUID(),
        createdAt: Date(timeIntervalSince1970: 0),
        audioFileName: "test.wav",
        traceFileName: "test.trace.json",
        pipeline: sourceAudioFileName == nil ? "aceStep" : "coverSong",
        prompt: "dark blues cover",
        originalPrompt: nil,
        lyrics: "",
        request: GenerationTraceRequestSnapshot(
            duration: 30,
            seed: 1234,
            method: "ode",
            guidanceScale: 6,
            guidancePreset: "softLate",
            guidancePresetName: "Soft Late",
            cfgIntervalStart: 0.3,
            cfgIntervalEnd: 1.0,
            precision: "float32",
            stepsOverride: nil,
            shiftOverride: nil,
            resolvedMode: "music",
            pipelineOverride: "auto",
            coverStrength: coverStrength,
            referenceAudioFileName: sourceAudioFileName
        ),
        primaryModel: HistoryModelSnapshot(
            category: "aceStepDiT",
            id: "ace-step-test",
            displayName: "ACE-Step Test"
        ),
        supportingModels: [],
        adapters: [],
        featureFlags: GenerationTraceFeatureFlags(
            lmThinkingEnabled: false,
            cotMetadataEnabled: false,
            cotCaptionEnabled: false,
            dcwEnabled: false
        ),
        timing: GenerationTraceTimingSnapshot(
            diffusionTime: 1,
            perStepTime: 0.1,
            totalTime: 2
        ),
        audio: GenerationTraceAudioSnapshot(
            sampleRate: 48000,
            duration: 30,
            sampleShape: [1, 1, 2],
            absMean: 0.1,
            absMax: 0.4,
            waveformPeaks: [],
            perSecondAbsMean: [],
            perSecondAbsMax: []
        ),
        listeningNote: nil
    )
}
