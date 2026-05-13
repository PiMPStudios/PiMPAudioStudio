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
