import Testing
import Foundation
import AVFoundation
import MLX
import MLXNN
import MLXOptimizers
@testable import PiMPAudioStudio
@testable import PiMPAudioCore

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

private func pcm16WAVData(sampleRate: Int = 22_050, samples: [Float]) -> Data {
    let bitsPerSample = 16
    let channels = 1
    let blockAlign = channels * bitsPerSample / 8
    let byteRate = sampleRate * blockAlign
    let dataSize = samples.count * blockAlign
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

    func appendInt16(_ value: Int16) {
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

    for sample in samples {
        let clamped = max(-1, min(1, sample))
        appendInt16(Int16(clamped * Float(Int16.max)))
    }
    return data
}

private func zipEntries(in archiveURL: URL) throws -> [String] {
    let process = Process()
    let stdout = Pipe()
    let stderr = Pipe()
    process.executableURL = URL(fileURLWithPath: "/usr/bin/zipinfo")
    process.arguments = ["-1", archiveURL.path(percentEncoded: false)]
    process.standardOutput = stdout
    process.standardError = stderr
    try process.run()
    process.waitUntilExit()
    if process.terminationStatus != 0 {
        let error = String(data: stderr.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
        throw NSError(
            domain: "PiMPAudioStudioTests.ZipInfo",
            code: Int(process.terminationStatus),
            userInfo: [NSLocalizedDescriptionKey: error]
        )
    }
    let output = String(data: stdout.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
    return output.split(separator: "\n").map(String.init)
}

private func writeFixture(_ value: String, to url: URL) throws {
    try FileManager.default.createDirectory(
        at: url.deletingLastPathComponent(),
        withIntermediateDirectories: true
    )
    try Data(value.utf8).write(to: url)
}

@MainActor
private func makeTestCommandFacade(root: URL) -> PiMPAudioCommandFacade {
    let historyStore = HistoryStore(directory: root.appendingPathComponent("History", isDirectory: true))
    return PiMPAudioCommandFacade(
        registry: ModelRegistry(baseDirectory: root.appendingPathComponent("Models", isDirectory: true)),
        historyStore: historyStore,
        promptEnhancer: PromptEnhancer(llm: nil),
        lyricWriterManager: LyricWriterManager(
            apiKeyStore: APIKeyStore(),
            providerAccessStore: ProviderAccessStore()
        )
    )
}

private extension JSONDecoder {
    static var withISO8601Dates: JSONDecoder {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return decoder
    }
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

@Test func appStorageLocationsResolveRealUserHomeOutsideSandboxContainer() {
    let home = AppStorageLocations.realUserHomeDirectory(environment: [
        "HOME": "/Users/testuser/Library/Containers/com.pimpstudios.PiMPAudioStudio/Data",
        "USER": "testuser",
    ])

    #expect(home.path == "/Users/testuser")
}

@Test func appRuntimeMigrationMovesLegacyDataDirectoriesAndWritesReport() throws {
    let tempRoot = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString, isDirectory: true)
    let legacyRoot = tempRoot.appendingPathComponent("Application Support/Anvil", isDirectory: true)
    let destinationRoot = tempRoot.appendingPathComponent("Application Support/PiMP Audio Studio", isDirectory: true)
    let reportURL = destinationRoot.appendingPathComponent("runtime_migration.json")
    defer { try? FileManager.default.removeItem(at: tempRoot) }

    try writeFixture("legacy model", to: legacyRoot.appendingPathComponent("models/ace-step/model.safetensors"))
    try writeFixture("legacy adapter", to: legacyRoot.appendingPathComponent("models/adapters/dark/adapter.json"))
    try writeFixture("legacy preset", to: legacyRoot.appendingPathComponent("presets/dark-blues.json"))
    try writeFixture("legacy llm", to: legacyRoot.appendingPathComponent("LLM/qwen/model.safetensors"))
    try writeFixture("legacy import", to: legacyRoot.appendingPathComponent("SourceAudioImports/source.m4a"))
    try writeFixture("legacy dataset", to: legacyRoot.appendingPathComponent("TrainingDatasets/pkg/captions.json"))
    try writeFixture("legacy mcp import", to: legacyRoot.appendingPathComponent("MCPImports/imported.wav"))
    try writeFixture("existing model", to: destinationRoot.appendingPathComponent("models/existing/keep.txt"))

    let report = AppRuntimeMigration.migrateDataDirectories(
        from: [legacyRoot],
        into: destinationRoot,
        reportURL: reportURL,
        migratedAt: Date(timeIntervalSince1970: 0)
    )

    #expect(FileManager.default.fileExists(
        atPath: destinationRoot.appendingPathComponent("models/ace-step/model.safetensors").path
    ))
    #expect(FileManager.default.fileExists(
        atPath: destinationRoot.appendingPathComponent("models/adapters/dark/adapter.json").path
    ))
    #expect(FileManager.default.fileExists(
        atPath: destinationRoot.appendingPathComponent("presets/dark-blues.json").path
    ))
    #expect(FileManager.default.fileExists(
        atPath: destinationRoot.appendingPathComponent("LLM/qwen/model.safetensors").path
    ))
    #expect(FileManager.default.fileExists(
        atPath: destinationRoot.appendingPathComponent("SourceAudioImports/source.m4a").path
    ))
    #expect(FileManager.default.fileExists(
        atPath: destinationRoot.appendingPathComponent("TrainingDatasets/pkg/captions.json").path
    ))
    #expect(FileManager.default.fileExists(
        atPath: destinationRoot.appendingPathComponent("MCPImports/imported.wav").path
    ))
    #expect(
        try String(
            contentsOf: destinationRoot.appendingPathComponent("models/existing/keep.txt"),
            encoding: .utf8
        ) == "existing model"
    )

    let migratedPaths = Set(report.entries.map(\.relativePath))
    #expect(migratedPaths == ["models", "presets", "LLM", "SourceAudioImports", "TrainingDatasets", "MCPImports"])
    #expect(report.entries.first(where: { $0.relativePath == "models" })?.action == "merged")
    #expect(FileManager.default.fileExists(atPath: reportURL.path))

    let reportData = try Data(contentsOf: reportURL)
    let decoded = try JSONDecoder.withISO8601Dates.decode(AppDataDirectoryMigrationReport.self, from: reportData)
    #expect(decoded.schemaVersion == "1.0")
    #expect(decoded.entries.count == report.entries.count)
}

@Test func appRuntimeMigrationDoesNotOverwriteExistingFiles() throws {
    let tempRoot = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString, isDirectory: true)
    let legacyRoot = tempRoot.appendingPathComponent("Application Support/Anvil", isDirectory: true)
    let destinationRoot = tempRoot.appendingPathComponent("Application Support/PiMP Audio Studio", isDirectory: true)
    defer { try? FileManager.default.removeItem(at: tempRoot) }

    let destinationPreset = destinationRoot.appendingPathComponent("presets/shared.json")
    try writeFixture("new preset", to: destinationPreset)
    try writeFixture("old preset", to: legacyRoot.appendingPathComponent("presets/shared.json"))

    _ = AppRuntimeMigration.migrateDataDirectories(from: [legacyRoot], into: destinationRoot)
    _ = AppRuntimeMigration.migrateDataDirectories(from: [legacyRoot], into: destinationRoot)

    #expect(try String(contentsOf: destinationPreset, encoding: .utf8) == "new preset")
}

@Test func appRuntimeMigrationTreatsMissingLegacyRootsAsNoOp() throws {
    let tempRoot = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString, isDirectory: true)
    let missingLegacyRoot = tempRoot.appendingPathComponent("missing/Anvil", isDirectory: true)
    let destinationRoot = tempRoot.appendingPathComponent("Application Support/PiMP Audio Studio", isDirectory: true)
    let reportURL = destinationRoot.appendingPathComponent("runtime_migration.json")
    defer { try? FileManager.default.removeItem(at: tempRoot) }

    let report = AppRuntimeMigration.migrateDataDirectories(
        from: [missingLegacyRoot],
        into: destinationRoot,
        reportURL: reportURL
    )

    #expect(report.entries.isEmpty)
    #expect(FileManager.default.fileExists(atPath: reportURL.path))
}

@Test func appRuntimeMigrationLeavesFailedDataMigrationRetryable() {
    let failedReport = AppDataDirectoryMigrationReport(
        schemaVersion: "1.0",
        migratedAt: Date(timeIntervalSince1970: 0),
        sourceRoots: ["/legacy/Anvil"],
        destinationRoot: "/new/PiMP Audio Studio",
        entries: [
            .init(
                relativePath: "models",
                sourcePath: "/legacy/Anvil/models",
                destinationPath: "/new/PiMP Audio Studio/models",
                action: "failed",
                message: "Permission denied"
            ),
        ]
    )
    let cleanReport = AppDataDirectoryMigrationReport(
        schemaVersion: "1.0",
        migratedAt: Date(timeIntervalSince1970: 0),
        sourceRoots: ["/legacy/Anvil"],
        destinationRoot: "/new/PiMP Audio Studio",
        entries: [
            .init(
                relativePath: "models",
                sourcePath: "/legacy/Anvil/models",
                destinationPath: "/new/PiMP Audio Studio/models",
                action: "copied",
                message: nil
            ),
        ]
    )

    #expect(failedReport.hasFailures)
    #expect(AppRuntimeMigration.shouldMarkDataDirectoryMigrationComplete(failedReport) == false)
    #expect(AppRuntimeMigration.shouldMarkDataDirectoryMigrationComplete(cleanReport))
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

@Test func datasetPackageEditorImportsSidecarsAndPersistsReview() throws {
    let datasetDir = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString, isDirectory: true)
    let clipsDir = datasetDir.appendingPathComponent("clips", isDirectory: true)
    try FileManager.default.createDirectory(at: clipsDir, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: datasetDir) }

    let clipURL = clipsDir.appendingPathComponent("clip_0001.wav")
    try silentWAVData().write(to: clipURL)

    try TrainingDatasetPackageWriter.writeLocalPackage(
        outputDirectory: datasetDir,
        clips: [TrainingImportedFile(url: clipURL, description: "old caption")],
        sourceFiles: [clipURL],
        config: LocalFilesDatasetConfig(),
        styleTemplate: "old caption"
    )

    try Data("smoky minor key blues trio".utf8)
        .write(to: clipsDir.appendingPathComponent("clip_0001.caption.txt"))
    try Data("""
    [ti:Sidecar Test]
    [00:01.00]I found the groove
    [00:03.00][00:04.00]after midnight
    <00:06.50>still moving
    """.utf8)
        .write(to: clipsDir.appendingPathComponent("clip_0001.lyrics.lrc"))
    try Data("""
    file,bpm,keyscale,language,timesignature
    clips/clip_0001.wav,78,E minor,en,4/4
    """.utf8)
        .write(to: datasetDir.appendingPathComponent("metadata.csv"))

    var document = try TrainingDatasetPackageEditor.loadPackage(at: datasetDir)
    let changed = TrainingDatasetPackageEditor.importSidecars(into: &document)

    #expect(changed >= 5)
    #expect(document.clips[0].caption == "smoky minor key blues trio")
    #expect(document.clips[0].lyrics == "I found the groove\nafter midnight\nstill moving")
    #expect(document.clips[0].captionSource == "sidecar")
    #expect(document.clips[0].lyricsSource == "sidecar")
    #expect(document.clips[0].bpm == "78")
    #expect(document.clips[0].key == "E minor")
    #expect(document.clips[0].language == "en")

    document.clips[0].isApproved = true
    try TrainingDatasetPackageEditor.save(document)

    let reloaded = try TrainingDatasetPackageEditor.loadPackage(at: datasetDir)
    #expect(reloaded.clips[0].isApproved)
    #expect(reloaded.clips[0].caption == "smoky minor key blues trio")
    #expect(reloaded.clips[0].lyrics == "I found the groove\nafter midnight\nstill moving")
    #expect(reloaded.clips[0].captionSource == "sidecar")
    #expect(reloaded.clips[0].lyricsSource == "sidecar")
    #expect(reloaded.clips[0].reviewStatus == "approved")
    #expect(reloaded.clips[0].timeSignature == "4/4")
    #expect(reloaded.isTrainingReady)

    let reviewedManifestData = try Data(contentsOf: datasetDir.appendingPathComponent("dataset_manifest.json"))
    let reviewedManifest = try JSONSerialization.jsonObject(with: reviewedManifestData) as? [String: Any]
    let reviewedAnalysis = reviewedManifest?["analysis"] as? [String: Any]
    #expect(reviewedAnalysis?["key_status"] as? String == "available_from_metadata")

    let backupRoot = datasetDir.appendingPathComponent("review_backups", isDirectory: true)
    let backups = try FileManager.default.contentsOfDirectory(
        at: backupRoot,
        includingPropertiesForKeys: nil
    )
    #expect(backups.count == 1)
    #expect(FileManager.default.fileExists(atPath: backups[0].appendingPathComponent("captions.json").path))

    let captionsData = try Data(contentsOf: datasetDir.appendingPathComponent("captions.json"))
    let captions = try JSONSerialization.jsonObject(with: captionsData) as? [[String: Any]]
    let annotation = captions?.first?["annotation"] as? [String: Any]
    #expect(annotation?["caption_source"] as? String == "sidecar")
    #expect(annotation?["lyrics_source"] as? String == "sidecar")
    #expect(annotation?["review_status"] as? String == "approved")

    let exported = try TrainingDatasetPackageEditor.exportSidecars(from: reloaded)
    #expect(exported == 1)
    let exportedCaption = try String(
        contentsOf: clipsDir.appendingPathComponent("clip_0001.caption.txt"),
        encoding: .utf8
    )
    #expect(exportedCaption == "smoky minor key blues trio")
    let exportedJSONData = try Data(contentsOf: clipsDir.appendingPathComponent("clip_0001.json"))
    let exportedJSON = try JSONSerialization.jsonObject(with: exportedJSONData) as? [String: Any]
    #expect(exportedJSON?["keyscale"] as? String == "E minor")
    let metadataCSV = try String(
        contentsOf: datasetDir.appendingPathComponent("metadata.csv"),
        encoding: .utf8
    )
    #expect(metadataCSV.contains("clips/clip_0001.wav"))
    #expect(metadataCSV.contains("E minor"))

    let preprocessData = try Data(contentsOf: datasetDir.appendingPathComponent("preprocess_cache.json"))
    let preprocess = try JSONSerialization.jsonObject(with: preprocessData) as? [String: Any]
    let adapterModes = preprocess?["adapter_modes"] as? [String] ?? []
    #expect(adapterModes.contains("lora"))
    #expect(adapterModes.contains("lokr"))
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

@Test func trainingRunLogRoundTripsNativeTrainingMetadata() throws {
    var log = TrainingRunLog(
        adapterID: "darkgentlemen-lokr",
        adapterName: "Dark Gentlemen LoKR",
        adapterKind: .lokr,
        baseModelID: "ace-step-dit-xl",
        sampleCount: 12,
        config: LoRAConfig(rank: 64, maxEpochs: 50, trainingLayerCount: 8),
        lokrConfig: LoKRConfig(linearDim: 64, maxEpochs: 50, trainingLayerCount: 8),
        resumedFromCheckpoint: false,
        startedAt: Date(timeIntervalSince1970: 0)
    )
    log.recordProgress(
        TrainingRunProgressEntry(
            timestamp: Date(timeIntervalSince1970: 10),
            epoch: 2,
            totalEpochs: 50,
            step: 25,
            loss: 1.25,
            learningRate: 0.00008,
            lossMovingAverage: 1.375,
            gradientNorm: 0.125,
            parameterNorm: 12.5,
            updateNorm: 0.03125,
            elapsed: 10,
            estimatedRemaining: 240
        ),
        lossHistory: [1.5, 1.25]
    )
    log.finish(
        status: .completed,
        finalLoss: 0.75,
        trainedPaths: [
            "layers.28.selfAttn.qProj",
            "layers.29.crossAttn.vProj",
        ],
        trainedLayerIndices: [28, 29],
        weightsTensorCount: 4,
        message: "Training completed.",
        finishedAt: Date(timeIntervalSince1970: 20)
    )

    let encoder = JSONEncoder()
    encoder.dateEncodingStrategy = .iso8601
    let data = try encoder.encode(log)
    let decoder = JSONDecoder()
    decoder.dateDecodingStrategy = .iso8601
    let decoded = try decoder.decode(TrainingRunLog.self, from: data)

    #expect(decoded.schemaVersion == TrainingRunLog.currentSchemaVersion)
    #expect(decoded.adapterKind == .lokr)
    #expect(decoded.status == .completed)
    #expect(decoded.lokrConfig?.linearDim == 64)
    #expect(decoded.progress.count == 1)
    #expect(decoded.progress.first?.lossMovingAverage == 1.375)
    #expect(decoded.progress.first?.gradientNorm == 0.125)
    #expect(decoded.progress.first?.parameterNorm == 12.5)
    #expect(decoded.progress.first?.updateNorm == 0.03125)
    #expect(decoded.lossHistory == [1.5, 1.25])
    #expect(decoded.finalLoss == 0.75)
    #expect(decoded.trainedLayerIndices == [28, 29])
    #expect(decoded.weightsTensorCount == 4)
}

@Test func trainingMetricSnapshotCapturesOptimizerParameterMovement() {
    let model = Linear(2, 1)
    let optimizer = AdamW(learningRate: 0.01)
    let trainable = model.trainableParameters().flattened()
    let keys = trainable.map(\.0)
    let before = TrainingMetricMath.materializedSnapshot(trainable.map(\.1))
    let gradients = trainable.map { _, value in MLXArray.ones(like: value) }

    let gradientPairs = zip(keys, gradients).map { ($0.0, $0.1) }
    optimizer.update(model: model, gradients: ModuleParameters.unflattened(gradientPairs))
    eval(model.trainableParameters(), optimizer.innerState())

    let after = model.trainableParameters().flattened().map(\.1)
    let updateNorm = TrainingMetricMath.l2NormOfDeltas(before: before, after: after)

    #expect(updateNorm != nil)
    #expect((updateNorm ?? 0) > 0)
}

@Test func datasetAnnotationMetadataBuilderUsesSharedBackendDescriptor() throws {
    let metadata = DatasetAnnotationMetadata(
        updatedAt: Date(timeIntervalSince1970: 10),
        backend: DatasetAnnotationBackendDescriptor(
            id: "native-qwen3-asr",
            version: "1.0",
            modelID: "qwen3-asr",
            modelRevision: "local"
        ),
        captionSource: "manual",
        lyricsSource: "native-qwen3-asr",
        language: "en",
        caption: "slow dark blues",
        lyrics: "I keep singing low",
        reviewStatus: "approved",
        rawOutputFiles: ["timed_lyrics": "lyrics.timed.json"],
        rawLyricsOutputFile: "lyrics.txt"
    )

    let object = metadata.persistedObject
    let rawOutputs = object["raw_output_files"] as? [String: String]

    #expect(object["schema_version"] as? String == TrainingDatasetPackageWriter.annotationSchemaVersion)
    #expect(object["annotation_backend"] as? String == "native-qwen3-asr")
    #expect(object["annotation_backend_version"] as? String == "1.0")
    #expect(object["model_id"] as? String == "qwen3-asr")
    #expect(object["caption_present"] as? Bool == true)
    #expect(object["lyrics_present"] as? Bool == true)
    #expect(rawOutputs?["timed_lyrics"] == "lyrics.timed.json")
    #expect(rawOutputs?["lyrics"] == "lyrics.txt")
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
        "preprocess_cache.json",
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
    #expect(imported[0].preprocessCache?.bundleName == "full_mix")
    #expect(imported[0].preprocessCache?.sampleFile.contains("preprocessed/full_mix/samples") == true)

    let cacheIndexURL = datasetDir.appendingPathComponent("preprocessed/full_mix/index.json")
    try FileManager.default.createDirectory(
        at: cacheIndexURL.deletingLastPathComponent(),
        withIntermediateDirectories: true
    )
    try """
    {
      "pimp_audio_preprocess_index_version": "1.0",
      "entries": [
        {
          "clip_id": "f001_clip_001",
          "file": "preprocessed/full_mix/samples/f001_clip_001.safetensors"
        }
      ]
    }
    """.write(to: cacheIndexURL, atomically: true, encoding: .utf8)
    try TrainingDatasetPackageWriter.refreshPreprocessCacheManifest(at: datasetDir)

    let refreshedPreprocessData = try Data(contentsOf: datasetDir.appendingPathComponent("preprocess_cache.json"))
    let refreshedPreprocess = try JSONSerialization.jsonObject(with: refreshedPreprocessData) as? [String: Any]
    #expect(refreshedPreprocess?["pimp_audio_preprocess_cache_version"] as? String == "1.0")
    #expect(refreshedPreprocess?["anvil_preprocess_cache_version"] == nil)
    #expect(refreshedPreprocess?["status"] as? String == "partial")
    let refreshedBundles = refreshedPreprocess?["bundles"] as? [[String: Any]] ?? []
    let fullMixBundle = refreshedBundles.first { ($0["name"] as? String) == "full_mix" }
    #expect(fullMixBundle?["status"] as? String == "complete")
    #expect((fullMixBundle?["cached_clip_count"] as? NSNumber)?.intValue == 1)

    let remoteData = try Data(contentsOf: datasetDir.appendingPathComponent("remote_training_package.json"))
    let remoteJSON = try JSONSerialization.jsonObject(with: remoteData) as? [String: Any]
    #expect(remoteJSON?["pimp_audio_remote_package_version"] as? String == "1.0")
    #expect(remoteJSON?["anvil_remote_package_version"] == nil)
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
    #expect(includedPaths.contains("annotations/**"))
    #expect(includedPaths.contains("preprocessed/**"))
    #expect(includedPaths.contains("preprocess_cache.json"))
    #expect(includedPaths.contains("training_bundle_vocals.json"))
    #expect(includedPaths.contains("training_bundle_instrumental.json"))
    let workerContract = remoteJSON?["worker_contract"] as? [String: Any]
    #expect(workerContract?["job_descriptor"] as? String == "remote_job.json")
    #expect(workerContract?["result_descriptor"] as? String == "remote_artifacts/result.json")
    #expect(workerContract?["logs_file"] as? String == "remote_artifacts/logs.jsonl")
    #expect(workerContract?["metrics_file"] as? String == "remote_artifacts/metrics.jsonl")

    let descriptor = try RemoteTrainingJobCodec.decoder.decode(
        RemoteTrainingPackageDescriptor.self,
        from: remoteData
    )
    #expect(descriptor.pimpAudioRemotePackageVersion == "1.0")
    var legacyRemoteJSON = remoteJSON ?? [:]
    legacyRemoteJSON["anvil_remote_package_version"] = legacyRemoteJSON.removeValue(
        forKey: "pimp_audio_remote_package_version"
    )
    let legacyRemoteData = try JSONSerialization.data(withJSONObject: legacyRemoteJSON)
    let legacyDescriptor = try RemoteTrainingJobCodec.decoder.decode(
        RemoteTrainingPackageDescriptor.self,
        from: legacyRemoteData
    )
    #expect(legacyDescriptor.pimpAudioRemotePackageVersion == "1.0")

    let job = RemoteTrainingJobRequest.makeTemplate(
        from: descriptor,
        jobID: "job-test-1",
        createdAt: Date(timeIntervalSince1970: 0),
        maxRuntimeSeconds: 7_200,
        maxCostUSD: 12.50
    )
    #expect(job.pimpAudioRemoteJobVersion == "1.0")
    #expect(job.task == .trainLoRA)
    #expect(job.package.entryBundle == "training_bundle_full_mix.json")
    #expect(job.package.selectedBundle.name == "full_mix")
    #expect(job.training?.adapterMode == .lora)
    #expect(job.training?.baseModelID == "ace-step-xl-sft")
    #expect(job.runtime.gpuClass == "h100_or_h200")
    #expect(job.runtime.maxRuntimeSeconds == 7_200)
    #expect(job.runtime.maxCostUSD == 12.50)
    #expect(job.outputs.resultFile == "remote_artifacts/result.json")
    #expect(job.safety.uploadRequiresUserApproval)
    #expect(job.safety.writeProviderSecretsToPackage == false)

    let encodedJob = try RemoteTrainingJobCodec.encoder.encode(job)
    let encodedJobJSON = try JSONSerialization.jsonObject(with: encodedJob) as? [String: Any]
    #expect(encodedJobJSON?["pimp_audio_remote_job_version"] as? String == "1.0")
    #expect(encodedJobJSON?["anvil_remote_job_version"] == nil)
    let decodedJob = try RemoteTrainingJobCodec.decoder.decode(RemoteTrainingJobRequest.self, from: encodedJob)
    #expect(decodedJob.jobID == "job-test-1")
    #expect(decodedJob.package.copyRules.include.contains("preprocessed/**"))
    var legacyJobJSON = encodedJobJSON ?? [:]
    legacyJobJSON["anvil_remote_job_version"] = legacyJobJSON.removeValue(
        forKey: "pimp_audio_remote_job_version"
    )
    let legacyJobData = try JSONSerialization.data(withJSONObject: legacyJobJSON)
    let legacyJob = try RemoteTrainingJobCodec.decoder.decode(RemoteTrainingJobRequest.self, from: legacyJobData)
    #expect(legacyJob.pimpAudioRemoteJobVersion == "1.0")

    let backupDirectory = datasetDir
        .appendingPathComponent("review_backups", isDirectory: true)
        .appendingPathComponent("old-save", isDirectory: true)
    try FileManager.default.createDirectory(at: backupDirectory, withIntermediateDirectories: true)
    try Data("backup".utf8).write(to: backupDirectory.appendingPathComponent("captions.json"))
    try Data("appledouble".utf8).write(to: clipsDir.appendingPathComponent("._f001_clip_001.m4a"))
    try Data("finder".utf8).write(to: datasetDir.appendingPathComponent(".DS_Store"))

    let archiveURL = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString)
        .appendingPathExtension("zip")
    defer { try? FileManager.default.removeItem(at: archiveURL) }
    try TrainingDatasetPackageWriter.exportPackageArchive(outputDirectory: datasetDir, to: archiveURL)
    let archiveEntries = try zipEntries(in: archiveURL)
    let archiveRoot = "\(datasetDir.lastPathComponent)/"
    #expect(archiveEntries.contains("\(archiveRoot)captions.json"))
    #expect(archiveEntries.contains("\(archiveRoot)clips/f001_clip_001.m4a"))
    #expect(archiveEntries.contains("\(archiveRoot)preprocessed/full_mix/index.json"))
    #expect(!archiveEntries.contains { $0.contains("/review_backups/") })
    #expect(!archiveEntries.contains { $0.components(separatedBy: "/").last?.hasPrefix("._") == true })
    #expect(!archiveEntries.contains { $0.components(separatedBy: "/").last == ".DS_Store" })

    let lokrJob = RemoteTrainingJobRequest.makeTemplate(
        from: descriptor,
        task: .trainLoKR,
        selectedBundleName: "instrumental",
        jobID: "job-test-2",
        createdAt: Date(timeIntervalSince1970: 0)
    )
    #expect(lokrJob.training?.adapterMode == .lokr)
    #expect(lokrJob.package.selectedBundle.name == "instrumental")

    let manifestData = try Data(contentsOf: datasetDir.appendingPathComponent("dataset_manifest.json"))
    let manifestJSON = try JSONSerialization.jsonObject(with: manifestData) as? [String: Any]
    #expect(manifestJSON?["pimp_audio_dataset_version"] as? String == "1.0")
    #expect(manifestJSON?["anvil_dataset_version"] == nil)
    let audioJSON = manifestJSON?["audio"] as? [String: Any]
    #expect(audioJSON?["metadata_status"] as? String == "audio_properties_unavailable")
    #expect(audioJSON?["format"] as? String == "m4a")
    let annotationJSON = manifestJSON?["annotation"] as? [String: Any]
    #expect(annotationJSON?["schema_version"] as? String == "1.0")
    let trainingJSON = manifestJSON?["training"] as? [String: Any]
    #expect(trainingJSON?["preprocess_cache"] as? String == "preprocess_cache.json")

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
    #expect(clipsJSON?.first?["caption_source"] as? String == "filename")
    let annotation = clipsJSON?.first?["annotation"] as? [String: Any]
    #expect(annotation?["caption_source"] as? String == "filename")
    let duration = (clipAudioJSON?["duration_seconds"] as? NSNumber)?.doubleValue ?? 0
    #expect(abs(duration - 0.1) < 0.001)
}

@Test func localDatasetEnrichmentAnalyzesTempoAndKeyFromAudio() throws {
    let directory = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString, isDirectory: true)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }

    let sampleRate = 22_050
    let frameCount = sampleRate * 8
    var samples = [Float](repeating: 0, count: frameCount)
    for index in samples.indices {
        let time = Double(index) / Double(sampleRate)
        let chord =
            0.24 * sin(2 * Double.pi * 82.407 * time) +
            0.14 * sin(2 * Double.pi * 123.471 * time) +
            0.14 * sin(2 * Double.pi * 164.814 * time) +
            0.08 * sin(2 * Double.pi * 246.942 * time)
        samples[index] = Float(chord)
    }

    let beatInterval = sampleRate / 2
    for beatStart in stride(from: 0, to: frameCount, by: beatInterval) {
        for tap in 0..<256 where beatStart + tap < frameCount {
            let envelope = 1 - (Float(tap) / 256)
            samples[beatStart + tap] += 0.6 * envelope
        }
    }

    let url = directory.appendingPathComponent("e_minor_120bpm.wav")
    try pcm16WAVData(sampleRate: sampleRate, samples: samples).write(to: url)

    let analysis = LocalDatasetEnrichmentService.analyzeAudioFile(url, role: "instrumental")
    let bpm = (analysis["tempo_bpm_estimate"] as? NSNumber)?.doubleValue ?? 0
    #expect(abs(bpm - 120) < 4)
    #expect(analysis["key_status"] as? String == "estimated_from_audio")
    #expect((analysis["keyscale"] as? String)?.isEmpty == false)
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
        "caption_source": "manual",
        "transcript": "I keep the line",
        "lyrics_source": "native-qwen3-asr",
        "transcription": { "status": "complete", "text": "I keep the line" },
        "annotation": {
          "schema_version": "1.0",
          "caption_source": "manual",
          "lyrics_source": "native-qwen3-asr",
          "review_status": "needs_review"
        },
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
    #expect(record?["caption_source"] as? String == "manual")
    #expect(record?["lyrics_source"] as? String == "native-qwen3-asr")
    let preservedAnnotation = record?["annotation"] as? [String: Any]
    #expect(preservedAnnotation?["lyrics_source"] as? String == "native-qwen3-asr")
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
        "keyscale": "E minor",
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
    #expect(analysis?["key_status"] as? String == "available_from_metadata")
    let enrichment = manifest?["enrichment"] as? [String: Any]
    #expect(enrichment?["status"] as? String == "complete")
    #expect((enrichment?["separated_clip_count"] as? NSNumber)?.intValue == 1)
    let training = manifest?["training"] as? [String: Any]
    #expect(training?["vocal_bundle"] as? String == "training_bundle_vocals.json")
    #expect(training?["instrumental_bundle"] as? String == "training_bundle_instrumental.json")
    #expect(training?["preprocess_cache"] as? String == "preprocess_cache.json")
    let annotation = manifest?["annotation"] as? [String: Any]
    #expect(annotation?["status"] as? String == "manual_review_required")
    let manifestClips = manifest?["clips"] as? [[String: Any]]
    #expect(manifestClips?.first?["caption"] as? String == "dark blues")
    #expect(manifestClips?.first?["lyrics"] as? String == "I keep singing low")

    let preprocessData = try Data(contentsOf: datasetDir.appendingPathComponent("preprocess_cache.json"))
    let preprocess = try JSONSerialization.jsonObject(with: preprocessData) as? [String: Any]
    #expect(preprocess?["status"] as? String == "ready_for_preprocess")
}

@Test func localDatasetEnrichmentUsesNativeDemucsOnly() {
    #expect(DemucsMLXSourceSeparator.backendName == "demucs-mlx-swift")
    #expect(DemucsMLXSourceSeparator.defaultModelName == "htdemucs")
}

@Test func trainingBaseModelLabelNamesActiveAceVariant() {
    let xlSFT = ModelSpec(
        id: "ace-step-dit-xl-sft",
        displayName: "ACE-Step v1.5 XL SFT",
        huggingFaceRepo: "ACE-Step/acestep-v15-xl-sft",
        files: [],
        category: .aceStepDiT,
        variantFamily: "ace-step-1.5-xl"
    )
    let xlTurbo = ModelSpec(
        id: "ace-step-dit-xl-turbo",
        displayName: "ACE-Step v1.5 XL Turbo",
        huggingFaceRepo: "ACE-Step/acestep-v15-xl-turbo",
        files: [],
        category: .aceStepDiT,
        variantFamily: "ace-step-1.5-xl"
    )
    let xlBase = ModelSpec(
        id: "ace-step-dit-xl-base",
        displayName: "ACE-Step v1.5 XL Base",
        huggingFaceRepo: "ACE-Step/acestep-v15-xl-base",
        files: [],
        category: .aceStepDiT,
        variantFamily: "ace-step-1.5-xl"
    )
    let turbo = ModelSpec(
        id: "ace-step-dit-turbo",
        displayName: "ACE-Step v1.5 Turbo",
        huggingFaceRepo: "ACE-Step/Ace-Step1.5",
        files: [],
        category: .aceStepDiT,
        variantFamily: "ace-step-1.5"
    )

    #expect(TrainingService.baseModelDisplayName(for: xlSFT) == "ACE-Step XL SFT")
    #expect(TrainingService.baseModelDisplayName(for: xlTurbo) == "ACE-Step XL Turbo")
    #expect(TrainingService.baseModelDisplayName(for: xlBase) == "ACE-Step XL Base")
    #expect(TrainingService.baseModelDisplayName(for: turbo) == "ACE-Step Turbo")
}

@Test func peftBaseModelInferencePreservesXLVariant() {
    #expect(
        PEFTKeyRemapper.resolveBaseModelID(from: "ACE-Step/acestep-v15-xl-turbo")
            == "ace-step-dit-xl-turbo"
    )
    #expect(
        PEFTKeyRemapper.resolveBaseModelID(from: "ACE-Step/acestep-v15-xl-base")
            == "ace-step-dit-xl-base"
    )
    #expect(
        PEFTKeyRemapper.resolveBaseModelID(from: "ACE-Step/acestep-v15-xl-sft")
            == "ace-step-dit-xl-sft"
    )
    #expect(
        PEFTKeyRemapper.resolveBaseModelID(from: "ACE-Step/acestep-v15-xl")
            == "ace-step-dit-xl"
    )
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
          "analysis": {
            "tempo_bpm_estimate": 72,
            "key": "E minor",
            "keyscale": "E minor",
            "key_status": "estimated_from_audio",
            "key_source": "native-dsp-instrumental",
            "key_confidence": 0.82
          }
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
    #expect(result.autoCaptionedClipCount == 1)
    #expect(await lyricsTranscriber.evictionCount == 1)

    let captionsData = try Data(contentsOf: datasetDir.appendingPathComponent("captions.json"))
    let captions = try JSONSerialization.jsonObject(with: captionsData) as? [[String: Any]]
    let record = captions?.first
    #expect(record?["caption"] as? String == "dark blues stem package, very slow 72 BPM, E minor, full mix with vocals and instrumental backing")
    #expect(record?["prompt"] as? String == "dark blues stem package, very slow 72 BPM, E minor, full mix with vocals and instrumental backing")
    #expect(record?["caption_source"] as? String == "native")
    #expect(record?["lyrics"] as? String == "I keep singing low")
    #expect(record?["transcript"] as? String == "I keep singing low")
    let transcription = record?["transcription"] as? [String: Any]
    #expect(transcription?["backend"] as? String == "test-qwen3-asr")
    #expect(transcription?["model"] as? String == "fake-qwen3-asr")
    #expect(transcription?["lyrics_file"] as? String == "stems/f001_clip_001/lyrics.txt")
    #expect(record?["lyrics_source"] as? String == "native-qwen3-asr")
    let annotation = record?["annotation"] as? [String: Any]
    #expect(annotation?["caption_source"] as? String == "native")
    #expect(annotation?["caption_backend"] as? String == "native-metadata-captioner")
    #expect(annotation?["caption_model_id"] as? String == "pimp-audio-metadata-caption-v1")
    #expect(annotation?["caption_seed"] as? String == "dark blues stem package")
    #expect(annotation?["lyrics_source"] as? String == "native-qwen3-asr")
    #expect(annotation?["model_id"] as? String == "fake-qwen3-asr")
    let assets = record?["assets"] as? [String: Any]
    #expect(assets?["timed_lyrics"] as? String == "stems/f001_clip_001/lyrics.timed.json")
    #expect((record?["bpm"] as? NSNumber)?.doubleValue == 72)
    #expect(record?["keyscale"] as? String == "E minor")
    let analysis = record?["analysis"] as? [String: Any]
    #expect(analysis?["bpm_status"] as? String == "estimated_from_stems")
    #expect(analysis?["key_status"] as? String == "estimated_from_audio")

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
    let manifestAnnotation = manifest?["annotation"] as? [String: Any]
    let lyricSources = manifestAnnotation?["lyrics_sources"] as? [String] ?? []
    #expect(lyricSources.contains("native-qwen3-asr"))
    let captionSources = manifestAnnotation?["caption_sources"] as? [String] ?? []
    #expect(captionSources.contains("native"))
    let manifestAnalysis = manifest?["analysis"] as? [String: Any]
    #expect(manifestAnalysis?["bpm_status"] as? String == "estimated_from_stems")
    #expect(manifestAnalysis?["key_status"] as? String == "estimated_from_audio")
    let manifestClips = manifest?["clips"] as? [[String: Any]]
    #expect(
        manifestClips?.first?["caption"] as? String ==
            "dark blues stem package, very slow 72 BPM, E minor, full mix with vocals and instrumental backing"
    )
    #expect(manifestClips?.first?["lyrics_source"] as? String == "native-qwen3-asr")
}

@Test func localDatasetEnrichmentPreservesImportedCaptions() async throws {
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
          "analysis": { "tempo_bpm_estimate": 90 }
        },
        "instrumental": {
          "file": "stems/f001_clip_001/instrumental.wav",
          "analysis": {
            "tempo_bpm_estimate": 90,
            "key": "A minor",
            "keyscale": "A minor",
            "key_status": "estimated_from_audio",
            "key_source": "native-dsp-instrumental"
          }
        }
      }
    }
    """
    try Data(separationJSON.utf8).write(to: stemsDir.appendingPathComponent("separation.json"))

    let captionsJSON = """
    [
      {
        "file": "clips/f001_clip_001.wav",
        "caption": "imported human caption",
        "caption_source": "sidecar",
        "annotation": {
          "caption_source": "sidecar"
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

    let result = try await LocalDatasetEnrichmentService.run(
        datasetDirectory: datasetDir,
        lyricsTranscriber: FakeDatasetLyricsTranscriber(),
        onStep: { _ in }
    )

    #expect(result.autoCaptionedClipCount == 0)

    let captionsData = try Data(contentsOf: datasetDir.appendingPathComponent("captions.json"))
    let captions = try JSONSerialization.jsonObject(with: captionsData) as? [[String: Any]]
    let record = captions?.first
    #expect(record?["caption"] as? String == "imported human caption")
    #expect(record?["caption_source"] as? String == "sidecar")
    let annotation = record?["annotation"] as? [String: Any]
    #expect(annotation?["caption_source"] as? String == "sidecar")
}

@Test func localDatasetEnrichmentReplacesFilenameCaptionPlaceholders() async throws {
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
          "analysis": { "tempo_bpm_estimate": 90 }
        },
        "instrumental": {
          "file": "stems/f001_clip_001/instrumental.wav",
          "analysis": {
            "tempo_bpm_estimate": 90,
            "key": "A minor",
            "keyscale": "A minor",
            "key_status": "estimated_from_audio",
            "key_source": "native-dsp-instrumental"
          }
        }
      }
    }
    """
    try Data(separationJSON.utf8).write(to: stemsDir.appendingPathComponent("separation.json"))

    let captionsJSON = """
    [
      {
        "file": "clips/f001_clip_001.wav",
        "caption": "source song (clip 1)",
        "caption_source": "filename",
        "source": {
          "kind": "local_file",
          "file_name": "source song.wav"
        },
        "annotation": {
          "caption_source": "filename"
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

    let result = try await LocalDatasetEnrichmentService.run(
        datasetDirectory: datasetDir,
        lyricsTranscriber: FakeDatasetLyricsTranscriber(),
        onStep: { _ in }
    )

    #expect(result.autoCaptionedClipCount == 1)

    let captionsData = try Data(contentsOf: datasetDir.appendingPathComponent("captions.json"))
    let captions = try JSONSerialization.jsonObject(with: captionsData) as? [[String: Any]]
    let record = captions?.first
    #expect(record?["caption"] as? String == "slow 90 BPM, A minor, full mix with vocals and instrumental backing")
    #expect(record?["prompt"] as? String == "slow 90 BPM, A minor, full mix with vocals and instrumental backing")
    #expect(record?["caption_source"] as? String == "native")
    let annotation = record?["annotation"] as? [String: Any]
    #expect(annotation?["caption_seed"] as? String == "")
    let inputs = annotation?["caption_inputs"] as? [String] ?? []
    #expect(!inputs.contains("caption_seed"))

    let imported = try TrainingDatasetImporter.importFiles(
        from: datasetDir.appendingPathComponent(TrainingDatasetPackageWriter.fullMixBundleFileName)
    )
    #expect(imported.first?.description == "slow 90 BPM, A minor, full mix with vocals and instrumental backing")
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
    let vocalBundleData = try Data(contentsOf: datasetDir.appendingPathComponent("training_bundle_vocals.json"))
    let vocalBundle = try JSONSerialization.jsonObject(with: vocalBundleData) as? [String: Any]
    #expect(vocalBundle?["preprocess_cache"] as? String == "preprocess_cache.json")
    let vocalClips = vocalBundle?["clips"] as? [[String: Any]] ?? []
    #expect(vocalClips.first?["annotation_schema_version"] as? String == "1.0")

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
    let samplingJSON = manifestJSON?["clip_sampling"] as? [String: Any]
    #expect(samplingJSON?["mode"] as? String == LocalDatasetClipSamplingMode.evenlySpread.rawValue)
}

@Test @MainActor func localFilesDatasetBuilderUsesCustomDatasetName() async throws {
    let tempRoot = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString, isDirectory: true)
    try FileManager.default.createDirectory(at: tempRoot, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: tempRoot) }

    let sourceURL = tempRoot.appendingPathComponent("first-source-file.wav")
    try silentWAVData(sampleRate: 22_050, channels: 1, frameCount: 2_205).write(to: sourceURL)

    let result = try await LocalFilesDatasetBuilder.run(
        sourceFiles: [sourceURL],
        config: LocalFilesDatasetConfig(clipLength: 0.1, clipsPerTrack: 1, maxTotalClips: 1),
        outputBase: tempRoot,
        datasetName: "My Custom Dataset!",
        styleTemplate: "custom dataset naming test",
        onPhase: { _ in }
    )

    #expect(result.outputDirectory.lastPathComponent.hasPrefix("my-custom-dataset_local_"))

    let manifestData = try Data(contentsOf: result.outputDirectory.appendingPathComponent("dataset_manifest.json"))
    let manifestJSON = try JSONSerialization.jsonObject(with: manifestData) as? [String: Any]
    #expect((manifestJSON?["name"] as? String)?.hasPrefix("my-custom-dataset_local_") == true)
}

@Test func localClipSamplerEvenlySpreadsLongSources() throws {
    let starts = LocalDatasetClipSampler.startTimes(
        totalSeconds: 90.0 * 60.0,
        clipLength: 35,
        maxClips: 8,
        mode: .evenlySpread,
        seed: 1337,
        sourceIdentifier: "long-source.wav"
    )

    #expect(starts.count == 8)
    #expect(starts.first == 0)
    #expect(abs((starts.last ?? 0) - ((90.0 * 60.0) - 35.0)) < 0.001)
    #expect(starts[1] > 35)
}

@Test func localClipSamplerRandomSeedIsDeterministicAndSorted() throws {
    let first = LocalDatasetClipSampler.startTimes(
        totalSeconds: 90.0 * 60.0,
        clipLength: 35,
        maxClips: 8,
        mode: .randomSeeded,
        seed: 42,
        sourceIdentifier: "long-source.wav"
    )
    let second = LocalDatasetClipSampler.startTimes(
        totalSeconds: 90.0 * 60.0,
        clipLength: 35,
        maxClips: 8,
        mode: .randomSeeded,
        seed: 42,
        sourceIdentifier: "long-source.wav"
    )
    let differentSeed = LocalDatasetClipSampler.startTimes(
        totalSeconds: 90.0 * 60.0,
        clipLength: 35,
        maxClips: 8,
        mode: .randomSeeded,
        seed: 43,
        sourceIdentifier: "long-source.wav"
    )

    #expect(first == second)
    #expect(first != differentSeed)
    #expect(first == first.sorted())
    #expect(first.contains { $0 > 35.0 * 8.0 })
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

@MainActor
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
            source: "qwen3-asr",
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

@Test func editorAudioExporterNormalizesDestinationExtensions() {
    let baseURL = URL(fileURLWithPath: "/tmp/edited-render")
    let wrongExtensionURL = URL(fileURLWithPath: "/tmp/edited-render.caf")

    #expect(EditorAudioExporter.normalizedDestinationURL(for: baseURL, format: .wav).lastPathComponent == "edited-render.wav")
    #expect(EditorAudioExporter.normalizedDestinationURL(for: wrongExtensionURL, format: .m4a).lastPathComponent == "edited-render.m4a")
}

@Test func editorAudioExporterWritesReadableWAV() async throws {
    let directory = FileManager.default.temporaryDirectory
        .appendingPathComponent("EditorAudioExporterTests-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }

    let buffer = editorExportTestBuffer()
    let destination = directory.appendingPathComponent("edited-render")
    let exportedURL = try await EditorAudioExporter.export(
        buffer: buffer,
        format: .wav,
        to: destination
    )

    #expect(exportedURL.lastPathComponent == "edited-render.wav")
    #expect(FileManager.default.fileExists(atPath: exportedURL.path(percentEncoded: false)))

    let audioFile = try AVAudioFile(forReading: exportedURL)
    #expect(Int(audioFile.fileFormat.sampleRate.rounded()) == buffer.sampleRate)
    #expect(Int(audioFile.fileFormat.channelCount) == buffer.channels)
    #expect(Int(audioFile.length) == buffer.frameCount)
}

private func editorExportTestBuffer(
    sampleRate: Int = 22_050,
    channels: Int = 1,
    frameCount: Int = 2_205
) -> PiMPAudioStudio.AudioBuffer {
    var samples = [Int16](repeating: 0, count: frameCount * channels)
    for index in samples.indices {
        samples[index] = Int16((index % 128) * 128 - 8_192)
    }
    let data = samples.withUnsafeBufferPointer { Data(buffer: $0) }
    return PiMPAudioStudio.AudioBuffer(data: data, sampleRate: sampleRate, channels: channels)
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

@Test func trainingPlanRecommendationScalesLargeLoRADatasets() {
    let recommendation = TrainingPlanRecommendation.make(
        adapterKind: .lora,
        sampleCount: 200,
        knownDuration: 6_000
    )

    #expect(recommendation?.rank == 64)
    #expect(recommendation?.epochs == 20)
    #expect(recommendation?.layerCount == 12)
    #expect(recommendation?.gradientAccumulationSteps == 2)
    #expect(recommendation?.estimatedOptimizerUpdates == 2_000)
    #expect(recommendation?.summary.contains("~100m audio") == true)
}

@Test func trainingPlanRecommendationDownshiftsShortAudioDatasets() {
    let recommendation = TrainingPlanRecommendation.make(
        adapterKind: .lora,
        sampleCount: 200,
        knownDuration: 600
    )

    #expect(recommendation?.rank == 16)
    #expect(recommendation?.epochs == 10)
    #expect(recommendation?.layerCount == 8)
    #expect(recommendation?.gradientAccumulationSteps == 2)
    #expect(recommendation?.estimatedOptimizerUpdates == 1_000)
}

@Test func trainingPlanRecommendationIgnoresSparseDurationCoverage() {
    let recommendation = TrainingPlanRecommendation.make(
        adapterKind: .lora,
        profile: TrainingDatasetProfile(
            sampleCount: 200,
            knownDuration: 600,
            knownDurationSampleCount: 20
        )
    )

    #expect(recommendation?.rank == 64)
    #expect(recommendation?.epochs == 20)
    #expect(recommendation?.layerCount == 12)
    #expect(recommendation?.summary.contains("known audio from 20/200") == true)
}

@Test func trainingPlanRecommendationKeepsLoKRTinyRunsNarrow() {
    let recommendation = TrainingPlanRecommendation.make(
        adapterKind: .lokr,
        sampleCount: 8
    )

    #expect(recommendation?.rank == 16)
    #expect(recommendation?.epochs == 75)
    #expect(recommendation?.layerCount == 4)
    #expect(recommendation?.gradientAccumulationSteps == 1)
    #expect(recommendation?.estimatedOptimizerUpdates == 600)
}

@Test @MainActor func mcpTrainingResourcesAreAdvertisedAndReadable() async throws {
    let tempRoot = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString, isDirectory: true)
    defer { try? FileManager.default.removeItem(at: tempRoot) }

    let facade = makeTestCommandFacade(root: tempRoot)
    let resources = await facade.listResources()
    let uris = Set(resources.compactMap { $0["uri"] as? String })

    #expect(uris.contains("pimpaudio://training"))
    #expect(uris.contains("pimpaudio://training-datasets"))
    #expect(uris.contains("pimpaudio://training-runs"))
    #expect(uris.contains("pimpaudio://training-dataset/latest"))
    #expect(uris.contains("pimpaudio://training-run/latest"))

    let directStatus = facade.trainingStatus()
    #expect(directStatus["available"] as? Bool == false)

    let resourceStatus = try await facade.readResource(uri: "pimpaudio://training")
    #expect(resourceStatus["available"] as? Bool == false)
}

@Test @MainActor func mcpTrainingDatasetListingUsesOutputFolderPackages() throws {
    let tempRoot = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString, isDirectory: true)
    defer { try? FileManager.default.removeItem(at: tempRoot) }

    let packageDirectory = tempRoot
        .appendingPathComponent("History", isDirectory: true)
        .appendingPathComponent("TrainingDatasets", isDirectory: true)
        .appendingPathComponent("mcp-dataset", isDirectory: true)
    try writeFixture(
        """
        [
          {
            "id": "clip-1",
            "file": "clips/f001_clip_001.wav",
            "caption": "dark test clip"
          }
        ]
        """,
        to: packageDirectory.appendingPathComponent(TrainingDatasetPackageWriter.captionsFileName)
    )
    try writeFixture(
        """
        {
          "name": "MCP Dataset",
          "created_at": "2026-06-07T00:00:00Z",
          "clip_count": 1,
          "enrichment": { "status": "ready_for_preprocess" }
        }
        """,
        to: packageDirectory.appendingPathComponent(TrainingDatasetPackageWriter.datasetManifestFileName)
    )

    let facade = makeTestCommandFacade(root: tempRoot)
    let payload = facade.listTrainingDatasets(arguments: ["limit": 50])
    let datasets = payload["datasets"] as? [[String: Any]] ?? []
    let dataset = datasets.first {
        $0["path"] as? String == packageDirectory.path(percentEncoded: false)
    }

    #expect(dataset?["name"] as? String == "MCP Dataset")
    #expect(dataset?["clipCount"] as? Int == 1)
    #expect(dataset?["enrichmentStatus"] as? String == "ready_for_preprocess")
}

@Test @MainActor func mcpTrainingDatasetInspectionReportsCoverageAndRecommendations() async throws {
    let tempRoot = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString, isDirectory: true)
    defer { try? FileManager.default.removeItem(at: tempRoot) }

    let packageDirectory = tempRoot
        .appendingPathComponent("History", isDirectory: true)
        .appendingPathComponent("TrainingDatasets", isDirectory: true)
        .appendingPathComponent("mcp-dataset", isDirectory: true)
    try writeFixture(
        """
        [
          {
            "id": "clip-1",
            "file": "clips/f001_clip_001.wav",
            "caption": "dark test clip",
            "lyrics": "line one\\nline two",
            "audio": { "duration_seconds": 60 },
            "analysis": { "bpm": 110, "key": "D minor" },
            "training_review": { "approved": true, "status": "approved" }
          }
        ]
        """,
        to: packageDirectory.appendingPathComponent(TrainingDatasetPackageWriter.captionsFileName)
    )
    try writeFixture(
        """
        {
          "name": "MCP Dataset",
          "created_at": "2026-06-07T00:00:00Z",
          "clip_count": 1,
          "enrichment": { "status": "ready_for_preprocess" }
        }
        """,
        to: packageDirectory.appendingPathComponent(TrainingDatasetPackageWriter.datasetManifestFileName)
    )

    let facade = makeTestCommandFacade(root: tempRoot)
    let detail = try facade.inspectTrainingDataset(arguments: [
        "path": packageDirectory.path(percentEncoded: false),
        "sampleLimit": 1
    ])
    let profile = detail["profile"] as? [String: Any]
    let annotations = detail["annotations"] as? [String: Any]
    let recommendations = detail["recommendations"] as? [String: Any]
    let loraRecommendation = recommendations?["lora"] as? [String: Any]
    let sampleClips = detail["sampleClips"] as? [[String: Any]]

    #expect(profile?["sampleCount"] as? Int == 1)
    #expect(profile?["knownDurationSampleCount"] as? Int == 1)
    #expect(annotations?["captionCount"] as? Int == 1)
    #expect(annotations?["lyricsCount"] as? Int == 1)
    #expect(annotations?["approvedCount"] as? Int == 1)
    #expect(loraRecommendation?["adapterKind"] as? String == "lora")
    #expect(sampleClips?.first?["lyricsLineCount"] as? Int == 2)

    let resourceDetail = try await facade.readResource(uri: "pimpaudio://training-dataset/latest")
    let summary = resourceDetail["summary"] as? [String: Any]
    #expect(summary?["name"] as? String == "MCP Dataset")
}
