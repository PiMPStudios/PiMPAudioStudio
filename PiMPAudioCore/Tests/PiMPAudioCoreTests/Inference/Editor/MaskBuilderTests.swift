import Testing
import MLX
@testable import PiMPAudioCore

// MARK: - MaskBuilder Tests

@Suite("MaskBuilder")
struct MaskBuilderTests {

    // Common test parameters matching ACE-Step v1.5 defaults.
    let sampleRate = 48_000
    let hopSize    = 1_920
    let latentDim  = 8

    private func latentFrames(for duration: Double) -> Int {
        Int((duration * Double(sampleRate) / Double(hopSize)).rounded(.up))
    }

    private func startFrame(for time: Double) -> Int {
        Int((time * Double(sampleRate) / Double(hopSize)).rounded(.down))
    }

    private func endFrame(for time: Double) -> Int {
        Int((time * Double(sampleRate) / Double(hopSize)).rounded(.up))
    }

    @Test("ACE-Step frame range uses 1920-hop 25 Hz timing")
    func aceStepFrameRangeUses1920HopTiming() {
        let totalDuration = 30.0
        let latentFrames = MusicPipeline.latentFrames(forDuration: totalDuration)
        let range = MaskBuilder.frameRange(
            selection: InpaintSelection(startTime: 0, endTime: totalDuration),
            totalDuration: totalDuration,
            sampleRate: MusicPipeline.defaultSampleRate,
            latentFrames: latentFrames,
            hopSize: MusicPipeline.hopLength
        )

        #expect(latentFrames == 750)
        #expect(range?.lowerBound == 0)
        #expect(range?.upperBound == 750)
    }

    @Test("Frame range covers tiny non-zero selections")
    func frameRangeCoversTinyNonZeroSelections() {
        let totalDuration = 5.0
        let latentFrames = latentFrames(for: totalDuration)
        let tinyDelta = 0.00001
        let range = MaskBuilder.frameRange(
            selection: InpaintSelection(startTime: 2.0, endTime: 2.0 + tinyDelta),
            totalDuration: totalDuration,
            sampleRate: sampleRate,
            latentFrames: latentFrames,
            hopSize: hopSize
        )

        #expect(range?.lowerBound == startFrame(for: 2.0))
        #expect(range?.upperBound == startFrame(for: 2.0) + 1)
    }

    @Test("Frame range rejects inverted selections")
    func frameRangeRejectsInvertedSelections() {
        let totalDuration = 5.0
        let latentFrames = latentFrames(for: totalDuration)
        let range = MaskBuilder.frameRange(
            selection: InpaintSelection(startTime: 4.0, endTime: 2.0),
            totalDuration: totalDuration,
            sampleRate: sampleRate,
            latentFrames: latentFrames,
            hopSize: hopSize
        )

        #expect(range == nil)
    }

    // MARK: 1 — Full mask (entire duration) → all 1s, shape [1, T, 1]

    @Test("Full selection produces all-ones mask with shape [1, T, 1]")
    func fullSelectionAllOnes() {
        let totalDuration = 10.0  // seconds
        let latentFrames = latentFrames(for: totalDuration)

        let selection = InpaintSelection(startTime: 0, endTime: totalDuration)
        let mask = MaskBuilder.buildMask(
            selection: selection,
            totalDuration: totalDuration,
            sampleRate: sampleRate,
            latentFrames: latentFrames,
            latentDim: latentDim,
            hopSize: hopSize
        )

        eval(mask)
        #expect(mask.shape == [1, latentFrames, 1])

        let flat = mask.reshaped([-1]).asArray(Float.self)
        #expect(flat.allSatisfy { $0 == 1.0 })
    }

    // MARK: 2 — Partial mask → correct number of 1s

    @Test("Partial selection produces correct number of masked frames")
    func partialSelectionCorrectFrameCount() {
        let totalDuration = 10.0
        let latentFrames  = latentFrames(for: totalDuration)

        // Select 2 – 5 seconds
        let startTime = 2.0
        let endTime   = 5.0
        let expectedStart = startFrame(for: startTime)
        let expectedEnd   = endFrame(for: endTime)
        let expectedOnes  = expectedEnd - expectedStart

        let selection = InpaintSelection(startTime: startTime, endTime: endTime)
        let mask = MaskBuilder.buildMask(
            selection: selection,
            totalDuration: totalDuration,
            sampleRate: sampleRate,
            latentFrames: latentFrames,
            latentDim: latentDim,
            hopSize: hopSize
        )

        eval(mask)
        #expect(mask.shape == [1, latentFrames, 1])

        let flat = mask.reshaped([-1]).asArray(Float.self)
        let onesCount = flat.filter { $0 == 1.0 }.count
        #expect(onesCount == expectedOnes)
    }

    // MARK: 3 — Binary values only (0 or 1)

    @Test("Mask contains only binary values (0.0 or 1.0)")
    func binaryValuesOnly() {
        let totalDuration = 8.0
        let latentFrames  = latentFrames(for: totalDuration)

        let selection = InpaintSelection(startTime: 1.0, endTime: 6.5)
        let mask = MaskBuilder.buildMask(
            selection: selection,
            totalDuration: totalDuration,
            sampleRate: sampleRate,
            latentFrames: latentFrames,
            latentDim: latentDim,
            hopSize: hopSize
        )

        eval(mask)
        let flat = mask.reshaped([-1]).asArray(Float.self)
        #expect(flat.allSatisfy { $0 == 0.0 || $0 == 1.0 })
    }

    // MARK: 4 — Empty selection (start == end) → zero mask

    @Test("Empty selection (startTime == endTime) returns zero mask")
    func emptySelectionZeroMask() {
        let totalDuration = 5.0
        let latentFrames  = latentFrames(for: totalDuration)

        let selection = InpaintSelection(startTime: 2.0, endTime: 2.0)
        let mask = MaskBuilder.buildMask(
            selection: selection,
            totalDuration: totalDuration,
            sampleRate: sampleRate,
            latentFrames: latentFrames,
            latentDim: latentDim,
            hopSize: hopSize
        )

        eval(mask)
        // Shape should be [1, T, 1] (no frequency band)
        #expect(mask.shape == [1, latentFrames, 1])

        let flat = mask.reshaped([-1]).asArray(Float.self)
        #expect(flat.allSatisfy { $0 == 0.0 })
    }

    // MARK: 5 — Frequency band → shape [1, T, latentDim], sum in [0, T*D]

    @Test("Frequency band produces [1, T, latentDim] mask with sum in valid range")
    func frequencyBandShape() {
        let totalDuration = 6.0
        let latentFrames  = latentFrames(for: totalDuration)
        let band = FrequencyBand(lowCutHz: 200, highCutHz: 8_000)

        let selection = InpaintSelection(
            startTime: 0.5,
            endTime: 4.0,
            frequencyBand: band
        )
        let mask = MaskBuilder.buildMask(
            selection: selection,
            totalDuration: totalDuration,
            sampleRate: sampleRate,
            latentFrames: latentFrames,
            latentDim: latentDim,
            hopSize: hopSize
        )

        eval(mask)
        #expect(mask.shape == [1, latentFrames, latentDim])

        let flat = mask.reshaped([-1]).asArray(Float.self)
        let total = Float(latentFrames * latentDim)
        let sum   = flat.reduce(0, +)
        #expect(sum >= 0)
        #expect(sum <= total)
        // At least some channels should be active given the wide band
        #expect(sum > 0)
    }

    // MARK: 6 — Clamp: selection beyond duration is clamped, not crashed

    @Test("Selection beyond total duration is clamped to valid frame range")
    func clampsBeyondDuration() {
        let totalDuration = 5.0
        let latentFrames  = latentFrames(for: totalDuration)

        // Selection extends well beyond the audio
        let selection = InpaintSelection(startTime: 3.0, endTime: 999.0)
        let mask = MaskBuilder.buildMask(
            selection: selection,
            totalDuration: totalDuration,
            sampleRate: sampleRate,
            latentFrames: latentFrames,
            latentDim: latentDim,
            hopSize: hopSize
        )

        eval(mask)
        #expect(mask.shape == [1, latentFrames, 1])

        let flat = mask.reshaped([-1]).asArray(Float.self)
        // No out-of-bounds; all values must be binary
        #expect(flat.allSatisfy { $0 == 0.0 || $0 == 1.0 })

        // Frames from startFrame to latentFrames should be 1
        let expectedStart = startFrame(for: 3.0)
        let expectedOnes  = latentFrames - expectedStart
        let onesCount = flat.filter { $0 == 1.0 }.count
        #expect(onesCount == expectedOnes)
    }

    // MARK: 7 — tiny non-zero range covers the touched frame

    @Test("Tiny non-zero selection covers at least one touched frame")
    func tinySelectionCoversTouchedFrame() {
        let totalDuration = 5.0
        let latentFrames  = latentFrames(for: totalDuration)

        let tinyDelta = 0.00001  // much less than one hop
        let selection = InpaintSelection(startTime: 2.0, endTime: 2.0 + tinyDelta)
        let mask = MaskBuilder.buildMask(
            selection: selection,
            totalDuration: totalDuration,
            sampleRate: sampleRate,
            latentFrames: latentFrames,
            latentDim: latentDim,
            hopSize: hopSize
        )

        eval(mask)
        let flat = mask.reshaped([-1]).asArray(Float.self)
        #expect(flat.filter { $0 == 1.0 }.count == 1)
    }

    // MARK: 8 — startFrame > endFrame (inverted range) → zero mask

    @Test("Inverted selection returns zero mask")
    func invertedSelectionZeroMask() {
        let totalDuration = 5.0
        let latentFrames = latentFrames(for: totalDuration)

        let selection = InpaintSelection(startTime: 4.0, endTime: 2.0)
        let mask = MaskBuilder.buildMask(
            selection: selection,
            totalDuration: totalDuration,
            sampleRate: sampleRate,
            latentFrames: latentFrames,
            latentDim: latentDim,
            hopSize: hopSize
        )

        eval(mask)
        let flat = mask.reshaped([-1]).asArray(Float.self)
        #expect(flat.allSatisfy { $0 == 0.0 })
    }
}
