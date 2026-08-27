import Foundation
import UniformTypeIdentifiers

// MARK: - AudioBuffer

struct AudioBuffer: Equatable, Sendable {
    var data: Data
    var sampleRate: Int
    var channels: Int

    var frameCount: Int {
        data.count / (MemoryLayout<Int16>.size * channels)
    }

    var duration: TimeInterval {
        Double(frameCount) / Double(sampleRate)
    }
}

// MARK: - DSPEffect

enum DSPEffect: String, CaseIterable, Sendable {
    case trimToSelection
    case deleteSelection
    case fadeIn
    case fadeOut
    case normalize
    case trimSilence
    case highPass
    case compress
    case limit
    case loudness
    case softGate
    case polishForKeep
    case eq
    case reverb
    case timeStretch
    case pitchShift

    var requiresParameters: Bool {
        switch self {
        case .trimToSelection, .deleteSelection, .fadeIn, .fadeOut:
            false
        case .normalize, .trimSilence, .highPass, .compress, .limit, .loudness,
             .softGate, .polishForKeep, .eq, .reverb, .timeStretch, .pitchShift:
            true
        }
    }

    var requiresSelection: Bool {
        switch self {
        case .trimToSelection, .deleteSelection, .fadeIn, .fadeOut:
            true
        case .normalize, .trimSilence, .highPass, .compress, .limit, .loudness,
             .softGate, .polishForKeep, .eq, .reverb, .timeStretch, .pitchShift:
            false
        }
    }

    var displayName: String {
        switch self {
        case .trimToSelection: "Trim to Selection"
        case .deleteSelection: "Delete Selection"
        case .fadeIn: "Fade In"
        case .fadeOut: "Fade Out"
        case .normalize: "Normalize"
        case .trimSilence: "Trim Silence"
        case .highPass: "High-Pass"
        case .compress: "Compress"
        case .limit: "Limit"
        case .loudness: "Loudness"
        case .softGate: "Soft Gate"
        case .polishForKeep: "Polish for Keep"
        case .eq: "Parametric EQ"
        case .reverb: "Reverb"
        case .timeStretch: "Time Stretch"
        case .pitchShift: "Pitch Shift"
        }
    }

    var isTransform: Bool {
        switch self {
        case .timeStretch, .pitchShift: true
        default: false
        }
    }
}

// MARK: - ApplyTarget

enum ApplyTarget: Equatable, Sendable {
    case selection(start: TimeInterval, end: TimeInterval)
    case wholeTrack
}

// MARK: - DSPParameters

enum DSPParameters: Equatable, Sendable {
    case none
    case normalize(NormalizeParameters)
    case trimSilence(TrimSilenceParameters)
    case highPass(HighPassParameters)
    case compress(CompressorParameters)
    case limit(LimiterParameters)
    case loudness(LoudnessParameters)
    case softGate(SoftGateParameters)
    case polishForKeep(PolishForKeepRecipe)
    case eq(EQParameters)
    case reverb(ReverbParameters)
    case timeStretch(TimeStretchParameters)
    case pitchShift(PitchShiftParameters)
}

// MARK: - EQBand + EQParameters

struct EQBand: Equatable, Sendable {
    var frequency: Float
    var gain: Float
    var q: Float
}

struct EQParameters: Equatable, Sendable {
    var bands: [EQBand]

    static let `default` = EQParameters(bands: [
        EQBand(frequency: 200, gain: 0, q: 1.0),
        EQBand(frequency: 1200, gain: 0, q: 1.4),
        EQBand(frequency: 8000, gain: 0, q: 0.8),
    ])
}

// MARK: - ReverbPreset + ReverbParameters

enum ReverbPreset: String, CaseIterable {
    case room, hall, plate

    var parameters: ReverbParameters {
        switch self {
        case .room: .room
        case .hall: .hall
        case .plate: .plate
        }
    }

    var displayName: String {
        rawValue.capitalized
    }
}

struct ReverbParameters: Equatable, Sendable {
    var decayTime: Float
    var wetDryMix: Float
    var preDelay: Float
    var damping: Float

    static let room = ReverbParameters(decayTime: 0.8, wetDryMix: 0.25, preDelay: 10, damping: 0.5)
    static let hall = ReverbParameters(decayTime: 2.4, wetDryMix: 0.35, preDelay: 20, damping: 0.6)
    static let plate = ReverbParameters(decayTime: 1.5, wetDryMix: 0.40, preDelay: 5, damping: 0.3)
    static let `default` = ReverbParameters.hall
}

// MARK: - TimeStretchParameters

struct TimeStretchParameters: Equatable, Sendable {
    var rate: Float // 0.25-4.0 (1.0 = original speed)
    static let `default` = TimeStretchParameters(rate: 1.0)
}

// MARK: - PitchShiftParameters

struct PitchShiftParameters: Equatable, Sendable {
    var pitch: Float // -2400 to +2400 cents (-24 to +24 semitones)
    static let `default` = PitchShiftParameters(pitch: 0)
}

// MARK: - Normalize

/// Peak-target normalize (default −1 dBFS headroom) or RMS target.
struct NormalizeParameters: Equatable, Sendable {
    /// Peak target in dBFS. `0` is full scale; recommended polish defaults are −1 or −3.
    var targetPeakDBFS: Float

    static let `default` = NormalizeParameters(targetPeakDBFS: -1.0)
    static let minusThree = NormalizeParameters(targetPeakDBFS: -3.0)
    static let fullScale = NormalizeParameters(targetPeakDBFS: 0.0)

    static let allowedRange: ClosedRange<Float> = -12 ... 0
}

struct TrimSilenceParameters: Equatable, Sendable {
    /// Level below which leading/trailing frames count as silence (dBFS).
    var thresholdDBFS: Float
    /// Maximum seconds to remove from the start.
    var maxLeadingSeconds: Float
    /// Maximum seconds to remove from the end.
    var maxTrailingSeconds: Float
    /// Linear edge fade after trim (milliseconds). `0` disables fades.
    var edgeFadeMilliseconds: Float

    static let `default` = TrimSilenceParameters(
        thresholdDBFS: -50,
        maxLeadingSeconds: 5,
        maxTrailingSeconds: 5,
        edgeFadeMilliseconds: 30
    )
}

struct HighPassParameters: Equatable, Sendable {
    /// Butterworth-ish 2-pole cutoff in Hz.
    var cutoffHz: Float

    static let `default` = HighPassParameters(cutoffHz: 40)
    static let allowedRange: ClosedRange<Float> = 20 ... 200
}

// MARK: - Dynamics (P1)

struct CompressorParameters: Equatable, Sendable {
    /// Threshold in dBFS. Signal above this is reduced.
    var thresholdDBFS: Float
    /// Compression ratio (e.g. 3 = 3:1).
    var ratio: Float
    /// Attack time in milliseconds.
    var attackMilliseconds: Float
    /// Release time in milliseconds.
    var releaseMilliseconds: Float
    /// Soft-knee width in dB (0 = hard knee).
    var kneeDB: Float
    /// Makeup gain in dB applied after compression.
    var makeupGainDB: Float

    static let light = CompressorParameters(
        thresholdDBFS: -18,
        ratio: 2.5,
        attackMilliseconds: 15,
        releaseMilliseconds: 120,
        kneeDB: 6,
        makeupGainDB: 2
    )

    static let medium = CompressorParameters(
        thresholdDBFS: -16,
        ratio: 3.5,
        attackMilliseconds: 10,
        releaseMilliseconds: 100,
        kneeDB: 4,
        makeupGainDB: 3
    )

    static let `default` = light
}

struct LimiterParameters: Equatable, Sendable {
    /// Output ceiling in dBFS (sample peak).
    var ceilingDBFS: Float
    /// Release time in milliseconds.
    var releaseMilliseconds: Float

    static let streaming = LimiterParameters(ceilingDBFS: -1.0, releaseMilliseconds: 50)
    static let louder = LimiterParameters(ceilingDBFS: -0.5, releaseMilliseconds: 40)
    static let `default` = streaming

    static let ceilingRange: ClosedRange<Float> = -6 ... 0
}

struct LoudnessParameters: Equatable, Sendable {
    /// Approximate target loudness (ungated LUFS-style, same as meter).
    var targetLUFS: Float
    /// Peak ceiling enforced after gain (via limiter).
    var ceilingDBFS: Float
    /// Maximum upward gain in dB (safety).
    var maxBoostDB: Float

    static let streaming = LoudnessParameters(
        targetLUFS: -14,
        ceilingDBFS: -1.0,
        maxBoostDB: 18
    )

    static let louder = LoudnessParameters(
        targetLUFS: -10,
        ceilingDBFS: -0.5,
        maxBoostDB: 20
    )

    static let `default` = streaming

    static let targetRange: ClosedRange<Float> = -24 ... -6
}

/// Downward expander for light noise-floor cleanup. Easy to overuse on full mixes —
/// defaults stay gentle (high threshold floor, modest ratio, limited range).
struct SoftGateParameters: Equatable, Sendable {
    /// Below this level (dBFS), expansion begins.
    var thresholdDBFS: Float
    /// Expansion ratio (e.g. 2 = 2:1 downward expand).
    var ratio: Float
    /// Attack in ms (how fast gate opens when signal returns).
    var attackMilliseconds: Float
    /// Release in ms (how fast gate closes into silence).
    var releaseMilliseconds: Float
    /// Maximum extra attenuation in dB (caps how deep the gate goes).
    var rangeDB: Float

    static let gentle = SoftGateParameters(
        thresholdDBFS: -48,
        ratio: 2.0,
        attackMilliseconds: 5,
        releaseMilliseconds: 80,
        rangeDB: 12
    )

    static let `default` = gentle

    static let thresholdRange: ClosedRange<Float> = -70 ... -24
}

// MARK: - NormalizeMode

enum NormalizeMode: Equatable {
    /// Scale so the sample peak reaches `targetDBFS` (clamped to full scale).
    case peak(targetDBFS: Float)
    case rms(targetDB: Float)

    static let peakDefault = NormalizeMode.peak(targetDBFS: -1.0)
}

// MARK: - ExportFormat

/// Audio formats the editor can export to using native macOS AVFoundation.
/// MP3, FLAC, and OGG are intentionally excluded — they require third-party
/// libraries (LAME, libFLAC, libvorbis) not available natively on macOS.
enum ExportFormat: String, CaseIterable {
    case wav = "WAV"
    case aiff = "AIFF"
    case m4a = "M4A"

    var fileExtension: String {
        switch self {
        case .wav: "wav"
        case .aiff: "aiff"
        case .m4a: "m4a"
        }
    }

    var contentType: UTType {
        switch self {
        case .wav: .wav
        case .aiff: .aiff
        case .m4a: UTType("public.mpeg-4-audio") ?? .audio
        }
    }

    var menuLabel: String {
        switch self {
        case .wav: "Export as WAV (lossless)…"
        case .aiff: "Export as AIFF (lossless)…"
        case .m4a: "Export as M4A (AAC compressed)…"
        }
    }
}

// MARK: - DSPOperation

struct DSPOperation: Equatable {
    let effect: DSPEffect
    let target: ApplyTarget
    let description: String
}
