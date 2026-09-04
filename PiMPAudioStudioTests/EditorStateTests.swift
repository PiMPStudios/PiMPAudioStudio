import Foundation
import PiMPAudioCore
@testable import PiMPAudioStudio
import Testing

@Suite(.serialized)
@MainActor
struct EditorStateTests {
    @Test
    func openingNewAudioDuringDSPKeepsNewAudio() async throws {
        let state = EditorState()
        let old = AudioBuffer(
            data: Data(repeating: 1, count: 4_000_000),
            sampleRate: 48000,
            channels: 2
        )
        let new = AudioBuffer(
            data: Data(repeating: 2, count: 400),
            sampleRate: 48000,
            channels: 2
        )
        state.audioBuffer = old
        state.fileName = "old.wav"
        state.duration = old.duration
        state.isLoaded = true

        state.applyDSPEffect(.normalize)
        state.applyLoadedBuffer(new, fileName: "new.wav")
        try await Task.sleep(for: .milliseconds(50))

        for _ in 0 ..< 200 {
            try await Task.sleep(for: .milliseconds(10))
            if !state.isDSPBusy {
                break
            }
        }

        #expect(state.fileName == "new.wav")
        #expect(state.audioBuffer?.frameCount == new.frameCount)
    }

    @Test
    func undoDuringDSPDoesNotCommitStaleResult() async throws {
        let state = EditorState()
        let original = AudioBuffer(
            data: Data(repeating: 3, count: 400),
            sampleRate: 48000,
            channels: 2
        )
        let current = AudioBuffer(
            data: Data(repeating: 1, count: 4_000_000),
            sampleRate: 48000,
            channels: 2
        )
        state.audioBuffer = original
        state.saveUndoSnapshot()
        state.audioBuffer = current
        state.noteBufferMutation()
        state.fileName = "current.wav"
        state.isLoaded = true

        state.applyDSPEffect(.normalize)
        state.undoDSP()

        for _ in 0 ..< 200 {
            try await Task.sleep(for: .milliseconds(10))
            if !state.isDSPBusy {
                break
            }
        }

        #expect(state.audioBuffer?.frameCount == original.frameCount)
        #expect(state.audioBuffer?.data == original.data)
    }
}
