import Testing
import MLX
@testable import AnvilCore

// MARK: - MaskBuilder Tests

@Suite("MaskBuilder")
struct MaskBuilderTests {

    // Common test parameters matching ACE-Step defaults
    let sampleRate = 44_100
    let hopSize    = 2_048
    let latentDim  = 8

    // MARK: 1 — Full mask (entire duration) → all 1s, shape [1, T, 1]

    @Test("Full selection produces all-ones mask with shape [1, T, 1]")
    func fullSelectionAllOnes() {
        let totalDuration = 10.0  // seconds
        // latentFrames = floor(10.0 * 44100) / 2048 = 441000 / 2048 = 215
        let latentFrames = Int(totalDuration * Double(sampleRate)) / hopSize

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
        let latentFrames  = Int(totalDuration * Double(sampleRate)) / hopSize

        // Select 2 – 5 seconds
        let startTime = 2.0
        let endTime   = 5.0
        let expectedStart = Int(startTime * Double(sampleRate)) / hopSize
        let expectedEnd   = Int(endTime   * Double(sampleRate)) / hopSize
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
        let latentFrames  = Int(totalDuration * Double(sampleRate)) / hopSize

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
        let latentFrames  = Int(totalDuration * Double(sampleRate)) / hopSize

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
        let latentFrames  = Int(totalDuration * Double(sampleRate)) / hopSize
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
        let latentFrames  = Int(totalDuration * Double(sampleRate)) / hopSize

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
        let expectedStart = Int(3.0 * Double(sampleRate)) / hopSize
        let expectedOnes  = latentFrames - expectedStart
        let onesCount = flat.filter { $0 == 1.0 }.count
        #expect(onesCount == expectedOnes)
    }

    // MARK: 7 — startFrame > endFrame (inverted range) → zero mask

    @Test("Inverted selection (start > end after clamping) returns zero mask")
    func invertedSelectionZeroMask() {
        let totalDuration = 5.0
        let latentFrames  = Int(totalDuration * Double(sampleRate)) / hopSize

        // startTime > endTime in sample space but both get clamped to same frame
        // Use an extremely tiny window that collapses to the same frame
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
        // startFrame == endFrame → zero mask
        #expect(flat.allSatisfy { $0 == 0.0 })
    }
}
