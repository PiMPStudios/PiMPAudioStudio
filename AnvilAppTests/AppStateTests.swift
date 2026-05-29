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
