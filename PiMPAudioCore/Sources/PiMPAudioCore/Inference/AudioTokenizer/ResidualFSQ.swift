// PiMPAudioCore/Sources/PiMPAudioCore/Inference/AudioTokenizer/ResidualFSQ.swift
import Foundation
import MLX
import MLXNN

/// Finite Scalar Quantization with a Linear bottleneck on each side.
///
/// Mirrors upstream's `vector_quantize_pytorch.ResidualFSQ` configured with
/// `num_quantizers=1`. The quantizer projects `hidden_size`-dim continuous
/// features down to `dim` (= len(levels)) dims, rounds each scalar to the
/// nearest of its `levels[i]` evenly-spaced bins, then projects back up.
///
/// The "residual" prefix is structurally a no-op when `num_quantizers=1`
/// but kept for naming parity with upstream and the safetensors keys
/// (`tokenizer.quantizer.project_in.*`, `tokenizer.quantizer.project_out.*`).
public class ResidualFSQ: Module {
    /// Per-axis quantization granularity. For ACE-Step v1.5 this is
    /// `[8, 8, 8, 5, 5, 5]` from `config.fsq_input_levels`. Total codebook
    /// size = product = 64000.
    public let levels: [Int]
    /// `project_in: Linear(hidden_size, len(levels))` — typically Linear(2048, 6).
    public var projectIn: Linear
    /// `project_out: Linear(len(levels), hidden_size)` — Linear(6, 2048).
    public var projectOut: Linear

    public init(hiddenSize: Int, levels: [Int]) {
        self.levels = levels
        self.projectIn = Linear(hiddenSize, levels.count, bias: true)
        self.projectOut = Linear(levels.count, hiddenSize, bias: true)
    }

    /// Run FSQ on continuous input.
    /// - Parameter hidden: `[B, T, hidden_size]`.
    /// - Returns: `(quantized, indices)` where:
    ///   - `quantized`: `[B, T, hidden_size]` — same shape as input, run through the bottleneck.
    ///   - `indices`:   `[B, T, 1]` — single integer per token (residual depth = 1).
    public func callAsFunction(_ hidden: MLXArray) -> (quantized: MLXArray, indices: MLXArray) {
        let z = projectIn(hidden)                       // [B, T, dim]
        let (zq, idx) = quantize(z)                     // [B, T, dim], [B, T]
        let out = projectOut(zq)                        // [B, T, hidden_size]
        return (out, idx.expandedDimensions(axis: -1))  // [B, T, 1]
    }

    /// Snap each scalar to the nearest of `levels[i]` evenly-spaced bins in [-1, 1],
    /// and pack the per-axis bin indices into a single integer per token.
    /// Mirrors upstream `lucidrains/vector-quantize-pytorch` FSQ.quantize logic.
    public func quantize(_ z: MLXArray) -> (quantized: MLXArray, indices: MLXArray) {
        let bsz = z.shape[0]
        let seqL = z.shape[1]

        // bound input via tanh and round to nearest bin per axis
        let zBounded = MLX.tanh(z)                                        // [B, T, dim] in (-1, 1)
        let levelArr = MLXArray(levels.map { Float($0) })                 // [dim]
        // half-step grid: bin i in 0..<level corresponds to value
        //   ((2*i + 1) / level) - 1   ∈ (-1, 1)
        // To pick: i = round((zBounded + 1) * level / 2 - 0.5),
        // clamped to [0, level-1].
        let halfLevels = levelArr / 2.0
        let scaled = (zBounded + 1.0) * halfLevels - 0.5                  // [B, T, dim]
        let rounded = MLX.round(scaled)
        let clamped = MLX.clip(rounded, min: MLXArray.zeros(like: rounded), max: levelArr - 1.0)
        let idxPerAxis = clamped.asType(.int32)                            // [B, T, dim]

        // recover quantized continuous values from bin indices
        let zq = (idxPerAxis.asType(.float32) + 0.5) * 2.0 / levelArr - 1.0 // [B, T, dim]

        // pack per-axis indices into a single integer:
        // idx = sum_i (idxPerAxis[i] * prod(levels[i+1..]))
        var multiplier = 1
        var packed = MLXArray.zeros([bsz, seqL], dtype: .int32)
        for i in (0..<levels.count).reversed() {
            let axis = idxPerAxis[0..., 0..., i] * multiplier
            packed = packed + axis
            multiplier *= levels[i]
        }
        return (zq, packed)
    }

    /// Inverse of `quantize` for the per-token packed integer index path.
    /// Used during cover generation when `audio_codes` are supplied directly:
    /// `lm_hints_5Hz = quantizer.get_output_from_indices(audio_codes)`.
    public func getOutputFromIndices(_ packed: MLXArray) -> MLXArray {
        let codes = packed.ndim == 3 && packed.shape[2] == 1
            ? packed[0..., 0..., 0]
            : packed
        let bsz = codes.shape[0]
        let seqL = codes.shape[1]
        let dim = levels.count

        let idxPerAxis = MLXArray.zeros([bsz, seqL, dim], dtype: .int32)
        var remaining = codes
        for i in (0..<dim).reversed() {
            let level = levels[i]
            let axis = remaining % level
            remaining = remaining / level
            // place `axis` at slot i along axis 2
            idxPerAxis[0..., 0..., i] = axis
        }

        let levelArr = MLXArray(levels.map { Float($0) })
        let zq = (idxPerAxis.asType(.float32) + 0.5) * 2.0 / levelArr - 1.0  // [B, T, dim]
        return projectOut(zq)                                                 // [B, T, hidden_size]
    }
}
