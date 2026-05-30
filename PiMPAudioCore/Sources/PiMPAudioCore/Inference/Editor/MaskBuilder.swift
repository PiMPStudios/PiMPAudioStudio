import MLX

// MARK: - MaskBuilder

/// Converts a user's time selection (and optional frequency band) into a
/// latent-space binary mask suitable for inpainting.
///
/// ACE-Step VAE hop size is 2048 samples. At 44100 Hz one latent frame ≈ 0.0464 s.
/// The Nyquist frequency assumed for channel mapping is 22 050 Hz.
public enum MaskBuilder {

    // MARK: - Constants

    /// Nyquist frequency used for linear Hz → latent-channel mapping.
    static let nyquistHz: Float = 22_050

    // MARK: - Public API

    /// Build a binary latent-space mask from an ``InpaintSelection``.
    ///
    /// - Parameters:
    ///   - selection:      The user's time/frequency selection.
    ///   - totalDuration:  Total audio duration in seconds.
    ///   - sampleRate:     Audio sample rate in Hz (e.g. 44 100).
    ///   - latentFrames:   Number of latent time frames (T dimension).
    ///   - latentDim:      Number of latent channels (D dimension).
    ///   - hopSize:        VAE hop size in samples (e.g. 2048 for ACE-Step).
    /// - Returns: MLXArray of shape `[1, latentFrames, 1]` (broadcastable mask)
    ///   or `[1, latentFrames, latentDim]` when a `frequencyBand` is present.
    ///   Values are 1.0 (regenerate) or 0.0 (preserve).
    public static func buildMask(
        selection: InpaintSelection,
        totalDuration: Double,
        sampleRate: Int,
        latentFrames: Int,
        latentDim: Int,
        hopSize: Int
    ) -> MLXArray {

        // Convert time → latent frame indices
        let rawStart = Int(selection.startTime * Double(sampleRate)) / hopSize
        let rawEnd   = Int(selection.endTime   * Double(sampleRate)) / hopSize

        let startFrame = max(0, min(rawStart, latentFrames))
        let endFrame   = max(0, min(rawEnd,   latentFrames))

        // Degenerate selection: return an all-zero mask
        guard startFrame < endFrame else {
            if selection.frequencyBand != nil {
                return MLXArray.zeros([1, latentFrames, latentDim])
            } else {
                return MLXArray.zeros([1, latentFrames, 1])
            }
        }

        if let band = selection.frequencyBand {
            return buildFrequencyMask(
                startFrame: startFrame,
                endFrame: endFrame,
                latentFrames: latentFrames,
                latentDim: latentDim,
                band: band
            )
        } else {
            return buildTimeMask(
                startFrame: startFrame,
                endFrame: endFrame,
                latentFrames: latentFrames
            )
        }
    }

    // MARK: - Private helpers

    /// Build a `[1, latentFrames, 1]` time-only mask.
    private static func buildTimeMask(
        startFrame: Int,
        endFrame: Int,
        latentFrames: Int
    ) -> MLXArray {
        var data = [Float](repeating: 0, count: latentFrames)
        for i in startFrame..<endFrame {
            data[i] = 1.0
        }
        return MLXArray(data, [1, latentFrames, 1])
    }

    /// Build a `[1, latentFrames, latentDim]` mask with per-channel attenuation
    /// derived from a ``FrequencyBand``.
    ///
    /// Channels are linearly mapped from 0 Hz (channel 0) to `nyquistHz`
    /// (channel `latentDim - 1`). Channels inside the band are 1.0;
    /// channels outside are 0.0. Time frames outside the selection stay 0.0.
    private static func buildFrequencyMask(
        startFrame: Int,
        endFrame: Int,
        latentFrames: Int,
        latentDim: Int,
        band: FrequencyBand
    ) -> MLXArray {

        // Compute per-channel weights (0 or 1)
        var channelWeights = [Float](repeating: 0, count: latentDim)
        for ch in 0..<latentDim {
            let chHz = Float(ch) / Float(max(latentDim - 1, 1)) * nyquistHz
            if chHz >= band.lowCutHz && chHz <= band.highCutHz {
                channelWeights[ch] = 1.0
            }
        }

        // Build flat [latentFrames * latentDim] data
        var data = [Float](repeating: 0, count: latentFrames * latentDim)
        for t in startFrame..<endFrame {
            for ch in 0..<latentDim {
                data[t * latentDim + ch] = channelWeights[ch]
            }
        }

        return MLXArray(data, [1, latentFrames, latentDim])
    }
}
