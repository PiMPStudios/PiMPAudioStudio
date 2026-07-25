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
    case eq
    case reverb
    case timeStretch
    case pitchShift

    var requiresParameters: Bool {
        switch self {
        case .trimToSelection, .deleteSelection, .fadeIn, .fadeOut:
            return false
        case .normalize, .trimSilence, .highPass, .eq, .reverb, .timeStretch, .pitchShift:
            return true
        }
    }

    var requiresSelection: Bool {
        switch self {
        case .trimToSelection, .deleteSelection, .fadeIn, .fadeOut:
            return true
        case .normalize, .trimSilence, .highPass, .eq, .reverb, .timeStretch, .pitchShift:
            return false
        }
    }

    var displayName: String {
        switch self {
        case .trimToSelection: return "Trim to Selection"
        case .deleteSelection: return "Delete Selection"
        case .fadeIn: return "Fade In"
        case .fadeOut: return "Fade Out"
        case .normalize: return "Normalize"
        case .trimSilence: return "Trim Silence"
        case .highPass: return "High-Pass"
        case .eq: return "Parametric EQ"
        case .reverb: return "Reverb"
        case .timeStretch: return "Time Stretch"
        case .pitchShift: return "Pitch Shift"
        }
    }

    var isTransform: Bool {
        switch self {
        case .timeStretch, .pitchShift: return true
        default: return false
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
        case .room: return .room
        case .hall: return .hall
        case .plate: return .plate
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
    var rate: Float          // 0.25-4.0 (1.0 = original speed)
    static let `default` = TimeStretchParameters(rate: 1.0)
}

// MARK: - PitchShiftParameters

struct PitchShiftParameters: Equatable, Sendable {
    var pitch: Float         // -2400 to +2400 cents (-24 to +24 semitones)
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

    static let allowedRange: ClosedRange<Float> = -12...0
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
    static let allowedRange: ClosedRange<Float> = 20...200
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
    case wav  = "WAV"
    case aiff = "AIFF"
    case m4a  = "M4A"

    var fileExtension: String {
        switch self {
        case .wav:  return "wav"
        case .aiff: return "aiff"
        case .m4a:  return "m4a"
        }
    }

    var contentType: UTType {
        switch self {
        case .wav:  return .wav
        case .aiff: return .aiff
        case .m4a:  return UTType("public.mpeg-4-audio") ?? .audio
        }
    }

    var menuLabel: String {
        switch self {
        case .wav:  return "Export as WAV (lossless)…"
        case .aiff: return "Export as AIFF (lossless)…"
        case .m4a:  return "Export as M4A (AAC compressed)…"
        }
    }
}

// MARK: - DSPOperation

struct DSPOperation: Equatable {
    let effect: DSPEffect
    let target: ApplyTarget
    let description: String
}
