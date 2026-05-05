import Foundation
import MLX

public enum VAETiledDecode {
    public static let chunkSize = 2048
    public static let overlap = 64
    public static let stride = chunkSize - 2 * overlap

    public static func computeTileCount(latentFrames: Int) -> Int {
        if latentFrames <= chunkSize { return 1 }
        return Int(ceil(Double(latentFrames) / Double(stride)))
    }

    public static func decode(_ latents: MLXArray, using vae: AutoEncoderOobleck) -> MLXArray {
        let latentFrames = latents.shape[1]
        if latentFrames <= chunkSize {
            return vae.decode(latents)
        }

        let numSteps = computeTileCount(latentFrames: latentFrames)
        var decodedParts: [MLXArray] = []
        var upsampleFactor: Double?

        for idx in 0..<numSteps {
            let coreStart = idx * stride
            let coreEnd = min(coreStart + stride, latentFrames)
            let winStart = max(0, coreStart - overlap)
            let winEnd = min(latentFrames, coreEnd + overlap)

            let chunk = latents[0..., winStart..<winEnd, 0...]
            let audioChunk = vae.decode(chunk)
            eval(audioChunk)

            if upsampleFactor == nil {
                upsampleFactor = Double(audioChunk.shape[1]) / Double(chunk.shape[1])
            }
            let factor = upsampleFactor!

            let trimStart = Int((Double(coreStart - winStart) * factor).rounded())
            let trimEnd = Int((Double(winEnd - coreEnd) * factor).rounded())
            let audioLen = audioChunk.shape[1]
            let endIdx = trimEnd > 0 ? audioLen - trimEnd : audioLen
            decodedParts.append(audioChunk[0..., trimStart..<endIdx, 0...])
        }

        return concatenated(decodedParts, axis: 1)
    }
}
