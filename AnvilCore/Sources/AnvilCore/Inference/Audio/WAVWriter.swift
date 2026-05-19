import Foundation
import MLX

public struct WAVWriter {
    public static func write(
        samples: MLXArray,
        sampleRate: Int = 44100,
        to url: URL
    ) throws {
        // The MLX VAE decoder returns NLC layout [batch, length, channels].
        // Some training/inference paths return [channels, frames] (CL) or
        // a flat 1-D mono buffer. Normalize to a single shape we understand.
        let channels: Int
        let numFrames: Int
        let normalized: MLXArray

        switch samples.ndim {
        case 1:
            // Mono buffer of audio samples.
            channels = 1
            numFrames = samples.shape[0]
            normalized = samples
        case 2:
            // [channels, frames] — keep as-is; interleave below.
            channels = samples.shape[0]
            numFrames = samples.shape[1]
            normalized = samples
        case 3:
            // [batch, length, channels] from the MLX VAE decoder. We always
            // generate batch=1, so take batch 0 to land on [length, channels].
            channels = samples.shape[2]
            numFrames = samples.shape[1]
            normalized = samples[0]
        default:
            channels = 0
            numFrames = 0
            normalized = samples
            assertionFailure("WAVWriter received samples of unexpected rank \(samples.ndim)")
        }

        // The MLX AutoencoderOobleck VAE often produces audio samples that
        // exceed the [-1, 1] range (we've observed peaks ~1.15 for Turbo
        // and ~1.78 for the Full DiT variant). Hard-clipping to [-1, 1]
        // produces audible distortion — sounds like sizzle or harshness on
        // top of the actual music. Upstream sidesteps this by writing
        // 32-bit float WAVs, but we want compatible 16-bit PCM output.
        //
        // Peak-normalize when the peak exceeds 1.0 so the loudest sample
        // lands exactly at 1.0. When the peak is already ≤ 1.0 the scale
        // factor is 1.0 (no-op). This preserves the waveform shape exactly
        // — only the absolute loudness changes, and only when the source
        // would have clipped.
        let absMax = MLX.abs(normalized).max().item(Float.self)
        let scale: Float = absMax > 1.0 ? (1.0 / absMax) : 1.0
        let leveled = scale != 1.0 ? normalized * scale : normalized
        let clamped = clip(leveled, min: -1.0, max: 1.0)
        let scaled = (clamped * Float(32767)).asType(.int16)
        eval(scaled)

        let interleaved: MLXArray
        switch samples.ndim {
        case 1:
            // Already a flat mono buffer.
            interleaved = scaled.reshaped([-1])
        case 2:
            // [channels, frames] -> transpose to [frames, channels] -> flatten LRLR…
            interleaved = scaled.transposed(0, 1).reshaped([-1])
        default:
            // [length, channels] (after the [0] indexing above) -> row-major flatten
            // already produces LRLR… interleaving.
            interleaved = scaled.reshaped([-1])
        }
        eval(interleaved)

        var pcmData = interleaved.asData(access: .copy).data

        // Adaptive tail fade: ACE-Step (and other diffusion models) commonly
        // produce near-zero latents for the last 1–4 seconds of the requested
        // window, resulting in an abrupt silence cutoff mid-beat. Scan the
        // PCM backwards in 0.25 s windows to find where content actually ends,
        // then apply a 0.75 s linear fade-out that ramps from full amplitude
        // to silence right at that transition. Keeps the file at the original
        // requested duration (no trimming) but eliminates the jarring dropoff.
        applyTailFadeInPlace(
            to: &pcmData,
            numFrames: numFrames,
            channels: channels,
            sampleRate: sampleRate
        )

        let bitsPerSample: UInt16 = 16
        let blockAlign = UInt16(channels) * (bitsPerSample / 8)
        let byteRate = UInt32(sampleRate) * UInt32(blockAlign)
        let dataSize = UInt32(pcmData.count)
        let fileSize = 36 + dataSize

        var header = Data(capacity: 44)
        header.append(contentsOf: "RIFF".utf8)
        header.append(littleEndian: fileSize)
        header.append(contentsOf: "WAVE".utf8)
        header.append(contentsOf: "fmt ".utf8)
        header.append(littleEndian: UInt32(16))
        header.append(littleEndian: UInt16(1))
        header.append(littleEndian: UInt16(channels))
        header.append(littleEndian: UInt32(sampleRate))
        header.append(littleEndian: byteRate)
        header.append(littleEndian: blockAlign)
        header.append(littleEndian: bitsPerSample)
        header.append(contentsOf: "data".utf8)
        header.append(littleEndian: dataSize)

        try write(header: header, pcmData: pcmData, to: url)
    }

    private static func write(header: Data, pcmData: Data, to url: URL) throws {
        let fileManager = FileManager.default
        let tempURL = url.deletingLastPathComponent()
            .appending(path: ".\(url.lastPathComponent).\(UUID().uuidString).tmp")
        defer { try? fileManager.removeItem(at: tempURL) }

        try header.write(to: tempURL)
        let handle = try FileHandle(forWritingTo: tempURL)
        try handle.seekToEnd()
        handle.write(pcmData)
        try handle.close()

        if fileManager.fileExists(atPath: url.path(percentEncoded: false)) {
            try fileManager.removeItem(at: url)
        }
        try fileManager.moveItem(at: tempURL, to: url)
    }

    // MARK: - Tail fade

    /// Scan interleaved 16-bit PCM backwards to find where audio content drops to
    /// near-silence, then apply a linear fade-out ending at that point.
    ///
    /// - Parameters:
    ///   - pcmData: Interleaved little-endian Int16 samples (L0 R0 L1 R1 …).
    ///   - numFrames: Total number of audio frames (samples per channel).
    ///   - channels: Number of channels (1 = mono, 2 = stereo).
    ///   - sampleRate: Sample rate in Hz (e.g. 48000).
    ///   - scanWindow: Duration of each backward-scan window (default 0.25 s).
    ///   - fadeDuration: Length of the linear fade applied before content end (default 0.75 s).
    ///   - rmsThreshold: Int16-scale RMS below which a window is considered silent (default 150 ≈ −47 dBFS).
    /// - Returns: A new `Data` with the fade applied; same byte count as `pcmData`.
    static func applyTailFade(
        to pcmData: Data,
        numFrames: Int,
        channels: Int,
        sampleRate: Int,
        scanWindow: Double = 0.25,
        fadeDuration: Double = 0.75,
        rmsThreshold: Double = 150.0,
        maxSilentTailDuration: Double = 1.25
    ) -> Data {
        var faded = pcmData
        applyTailFadeInPlace(
            to: &faded,
            numFrames: numFrames,
            channels: channels,
            sampleRate: sampleRate,
            scanWindow: scanWindow,
            fadeDuration: fadeDuration,
            rmsThreshold: rmsThreshold,
            maxSilentTailDuration: maxSilentTailDuration
        )
        return faded
    }

    static func applyTailFadeInPlace(
        to pcmData: inout Data,
        numFrames: Int,
        channels: Int,
        sampleRate: Int,
        scanWindow: Double = 0.25,
        fadeDuration: Double = 0.75,
        rmsThreshold: Double = 150.0,
        maxSilentTailDuration: Double = 1.25
    ) {
        guard numFrames > 0, channels > 0, pcmData.count == numFrames * channels * 2 else {
            return
        }

        let bytesPerFrame = channels * 2
        let scanFrames = max(1, Int(Double(sampleRate) * scanWindow))
        let fadeFrames  = max(1, Int(Double(sampleRate) * fadeDuration))
        let maxSilentTailFrames = max(1, Int(Double(sampleRate) * maxSilentTailDuration))
        let byteCount = pcmData.count

        pcmData.withUnsafeMutableBytes { rawBuffer in
            guard let rawBase = rawBuffer.baseAddress else { return }
            let bytes = rawBase.assumingMemoryBound(to: UInt8.self)

            // ── 1. Find where audio content ends ──────────────────────────────
            // Walk backwards in scanWindow-sized chunks until a chunk whose RMS
            // exceeds rmsThreshold is found. That chunk's end is "content end".
            var contentEndFrame = numFrames
            let numChunks = (numFrames + scanFrames - 1) / scanFrames
            scan: for chunk in stride(from: numChunks - 1, through: 0, by: -1) {
                let startF = chunk * scanFrames
                let endF   = min(startF + scanFrames, numFrames)
                var sumSq: Int64 = 0
                for f in startF..<endF {
                    let base = f * bytesPerFrame
                    for c in 0..<channels {
                        let off = base + c * 2
                        let lo  = UInt16(bytes[off])
                        let hi  = UInt16(bytes[off + 1])
                        let raw = Int16(bitPattern: lo | (hi << 8))
                        let s   = Int64(raw)
                        sumSq += s * s
                    }
                }
                let rms = (Double(sumSq) / Double((endF - startF) * channels)).squareRoot()
                if rms > rmsThreshold {
                    contentEndFrame = endF
                    break scan
                }
            }

            let silentTailFrames = numFrames - contentEndFrame
            guard silentTailFrames > 0 else {
                return
            }
            guard silentTailFrames <= maxSilentTailFrames else {
                return
            }

            // ── 2. Apply linear fade from (contentEnd - fadeDuration) to contentEnd
            let fadeStart = max(0, contentEndFrame - fadeFrames)
            let fadeLen   = contentEndFrame - fadeStart
            for f in fadeStart..<contentEndFrame {
                let alpha = Float(contentEndFrame - 1 - f) / Float(max(fadeLen - 1, 1))
                let base  = f * bytesPerFrame
                for c in 0..<channels {
                    let off = base + c * 2
                    let lo  = UInt16(bytes[off])
                    let hi  = UInt16(bytes[off + 1])
                    let raw = Int16(bitPattern: lo | (hi << 8))
                    let faded = Int16(clamping: Int32(Float(raw) * alpha))
                    let u = UInt16(bitPattern: faded)
                    bytes[off]     = UInt8(u & 0xFF)
                    bytes[off + 1] = UInt8((u >> 8) & 0xFF)
                }
            }

            // ── 3. Hard-zero any remaining frames past content end ─────────────
            // These are already near-silent (that's how contentEndFrame was chosen),
            // but zeroing them removes any low-level hiss that could confuse downstream
            // tools and avoids any quantisation noise artefacts from the scan threshold.
            if contentEndFrame < numFrames {
                let zeroOffset = contentEndFrame * bytesPerFrame
                memset(bytes.advanced(by: zeroOffset), 0, byteCount - zeroOffset)
            }
        }
    }
}

private extension Data {
    mutating func append(littleEndian value: UInt16) {
        var v = value.littleEndian
        append(Data(bytes: &v, count: 2))
    }

    mutating func append(littleEndian value: UInt32) {
        var v = value.littleEndian
        append(Data(bytes: &v, count: 4))
    }
}
