// PiMPAudioCore/Sources/PiMPAudioCore/Inference/StableAudio/SFXDiTModel.swift
//
// Stable Audio Open 1.0 Diffusion Transformer.
//
// Architecture (from actual HF weight inspection of
// stabilityai/stable-audio-open-1.0 model.safetensors — 745 keys):
//
//   hiddenSize          = 1536   numLayers = 24
//   numHeads            = 24     headDim   = 64
//   numKVHeads          = 12     (cross-attn GQA 2:1)
//   latentDim           = 64     conditioningDim = 768 (T5-base)
//   ffIntermediateDim   = 6144   (SwiGLU up-branch; combined proj = 12288)
//   rotaryDim           = 32     (partial RoPE, first 32 of 64 head dims)
//   timestepFeaturesDim = 256    (FourierFeatures output = 2 × 128)
//   globalCondDim       = 1536   (concat startEmbed[768] + totalEmbed[768])
//
// Forward pass (prepend-style conditioning, matches mlx-audiogen DiT):
//   1. preprocess_conv  Conv1d(latentDim→latentDim, k=1)   — 64→64, identity channel count
//   2. timestepFeatures.encode(t)      → [B, 256] Fourier features
//   3. toTimestepEmbed MLP(256→H→H)    → [B, H] tEmb
//   4. toGlobalEmbed MLP(1536→H→H)(g) → [B, H]  added to tEmb → gEmb [B, 1, H] (prepend token)
//   5. toCondEmbed MLP(768→768→768)(crossCond) → projected cross-attn tokens
//   6. RoPE for seqLen+1 positions (prepend at pos 0, latents at 1..T)
//   7. transformer: project_in → prepend gEmb → 24 × SAODiTLayer → remove prepend → project_out
//   8. postprocess_conv Conv1d(latentDim→latentDim, k=1)   — 64→64, identity channel count
//
// Swift module paths mirror SFXDiTNameMap output (HF keys → module paths).

import Foundation
import MLX
import MLXFast
import MLXNN

// MARK: - Learned Fourier features for the diffusion timestep

/// Wraps the learned `timestep_features.weight` buffer [halfDim, 1].
/// stable_audio_tools uses `register_buffer` so the bank is saved in the
/// checkpoint and must be loaded, not recomputed from scratch.
public final class SAOFourierFeatures: Module {
    /// Frequency bank, shape [halfDim, 1].
    public var weight: MLXArray

    public init(halfDim: Int = 128) {
        weight = MLXArray.zeros([halfDim, 1])
    }

    /// Encode scalar timestep(s) → [B, 2·halfDim].
    /// Mirrors: `f = 2π · t @ weight.T;  cat([cos(f), sin(f)], dim=-1)`.
    public func encode(_ t: MLXArray) -> MLXArray {
        let bT = t.reshaped([-1, 1]).asType(.float32) // [B, 1]
        let angles = 2 * Float.pi * matmul(bT, weight.transposed()) // [B, halfDim]
        return concatenated([cos(angles), sin(angles)], axis: -1) // [B, 2·halfDim]
    }
}

// MARK: - Two-layer MLP with bias and SiLU

//
// Used for to_timestep_embed / to_cond_embed / to_global_embed.
// Upstream: nn.Sequential(Linear+bias, SiLU, Linear+bias).
// SFXDiTNameMap maps index 0 → fc1 and index 2 → fc2.

public final class SAOBiasMLP: Module, UnaryLayer {
    public var fc1: Linear
    public var fc2: Linear

    public init(inputDim: Int, hiddenDim: Int, outputDim: Int? = nil) {
        let out = outputDim ?? hiddenDim
        fc1 = Linear(inputDim, hiddenDim, bias: true)
        fc2 = Linear(hiddenDim, out, bias: true)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        fc2(saoSiLU(fc1(x)))
    }
}

// MARK: - Self-attention with partial RoPE

//
// RoPE is applied only to the first `rotaryDim` channels per head;
// the remaining headDim − rotaryDim channels pass through unchanged.

public final class SAODiTSelfAttention: Module {
    let numHeads: Int
    let headDim: Int
    let rotaryDim: Int

    @ModuleInfo(key: "to_qkv") public var toQKV: Linear
    @ModuleInfo(key: "to_out") public var toOut: Linear

    public init(hiddenSize: Int, numHeads: Int, headDim: Int, rotaryDim: Int) {
        self.numHeads = numHeads
        self.headDim = headDim
        self.rotaryDim = rotaryDim
        let inner = numHeads * headDim
        _toQKV.wrappedValue = Linear(hiddenSize, inner * 3, bias: false)
        _toOut.wrappedValue = Linear(inner, hiddenSize, bias: false)
    }

    public func callAsFunction(
        _ x: MLXArray,
        cosEmb: MLXArray,
        sinEmb: MLXArray
    ) -> MLXArray {
        let B = x.shape[0]
        let L = x.shape[1]
        let qkvOut = toQKV(x).reshaped([B, L, 3, numHeads, headDim])
        var q = qkvOut[0..., 0..., 0].transposed(0, 2, 1, 3) // [B, H, L, D]
        var k = qkvOut[0..., 0..., 1].transposed(0, 2, 1, 3)
        let v = qkvOut[0..., 0..., 2].transposed(0, 2, 1, 3)

        (q, k) = applyPartialRoPE(q: q, k: k, cosEmb: cosEmb, sinEmb: sinEmb)

        let scale = Float(1.0 / sqrt(Float(headDim)))
        let ctx = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: nil
        )
        return toOut(ctx.transposed(0, 2, 1, 3).reshaped([B, L, -1]))
    }

    private func applyPartialRoPE(
        q: MLXArray,
        k: MLXArray,
        cosEmb: MLXArray, // [1, 1, L, rotaryDim]
        sinEmb: MLXArray // [1, 1, L, rotaryDim]
    ) -> (MLXArray, MLXArray) {
        let qRot = q[0..., 0..., 0..., 0 ..< rotaryDim]
        let qPass = q[0..., 0..., 0..., rotaryDim...]
        let kRot = k[0..., 0..., 0..., 0 ..< rotaryDim]
        let kPass = k[0..., 0..., 0..., rotaryDim...]
        let qFull = concatenated([rotateVec(qRot, cos: cosEmb, sin: sinEmb), qPass], axis: -1)
        let kFull = concatenated([rotateVec(kRot, cos: cosEmb, sin: sinEmb), kPass], axis: -1)
        return (qFull, kFull)
    }

    /// Standard RoPE: out = x * cos + rotate_half(x) * sin.
    private func rotateVec(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let half = rotaryDim / 2
        let x1 = x[0..., 0..., 0..., 0 ..< half]
        let x2 = x[0..., 0..., 0..., half...]
        return x * cos + concatenated([-x2, x1], axis: -1) * sin
    }
}

// MARK: - Cross-attention with GQA (24 Q heads, 12 KV heads)

public final class SAODiTCrossAttention: Module {
    let numQHeads: Int
    let numKVHeads: Int
    let headDim: Int

    @ModuleInfo(key: "to_q") public var toQ: Linear
    @ModuleInfo(key: "to_kv") public var toKV: Linear
    @ModuleInfo(key: "to_out") public var toOut: Linear

    public init(
        hiddenSize: Int,
        numQHeads: Int,
        numKVHeads: Int,
        headDim: Int,
        condDim: Int
    ) {
        self.numQHeads = numQHeads
        self.numKVHeads = numKVHeads
        self.headDim = headDim
        _toQ.wrappedValue = Linear(hiddenSize, numQHeads * headDim, bias: false)
        _toKV.wrappedValue = Linear(condDim, numKVHeads * headDim * 2, bias: false)
        _toOut.wrappedValue = Linear(numQHeads * headDim, hiddenSize, bias: false)
    }

    public func callAsFunction(_ x: MLXArray, conditioning: MLXArray) -> MLXArray {
        let B = x.shape[0]
        let L = x.shape[1]
        let condLen = conditioning.shape[1]

        let q = toQ(x).reshaped([B, L, numQHeads, headDim]).transposed(0, 2, 1, 3)
        let kv = toKV(conditioning).reshaped([B, condLen, 2, numKVHeads, headDim])
        var k = kv[0..., 0..., 0].transposed(0, 2, 1, 3) // [B, numKVHeads, condLen, D]
        var v = kv[0..., 0..., 1].transposed(0, 2, 1, 3)

        // Expand each KV head to cover ratio Q heads.
        let ratio = numQHeads / numKVHeads
        if ratio > 1 {
            k = repeated(k, count: ratio, axis: 1)
            v = repeated(v, count: ratio, axis: 1)
        }

        let scale = Float(1.0 / sqrt(Float(headDim)))
        let ctx = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: nil
        )
        return toOut(ctx.transposed(0, 2, 1, 3).reshaped([B, L, -1]))
    }
}

// MARK: - SwiGLU feed-forward network

//
// Upstream (mlx-audiogen dit.py + diffusers activation_fn="swiglu"):
//   FeedForward { proj → split → up * silu(gate) } → Linear(inner, dim)
// Both projections have bias=True (PyTorch default).
// Weight keys after SFXDiTNameMap:
//   ff.ff.0.proj.weight/bias  → ff.gate_up.weight/bias  [2·intermediate, hidden] / [2·intermediate]
//   ff.ff.2.weight/bias       → ff.out.weight/bias       [hidden, intermediate]   / [hidden]
//
// NOTE: gate activation is SiLU (x·σ(x)), NOT GELU.  Using GELU here
// produces wrong velocity predictions and causes the ODE to diverge.

public final class SAODiTFFN: Module {
    @ModuleInfo(key: "gate_up") public var gateUp: Linear
    @ModuleInfo(key: "out") public var projOut: Linear

    public init(hiddenSize: Int, intermediateDim: Int) {
        _gateUp.wrappedValue = Linear(hiddenSize, intermediateDim * 2, bias: true)
        _projOut.wrappedValue = Linear(intermediateDim, hiddenSize, bias: true)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let proj = gateUp(x)
        let half = proj.shape.last! / 2
        let up = proj[0..., 0..., 0 ..< half]
        let gate = proj[0..., 0..., half...]
        return projOut(up * saoSiLU(gate)) // SwiGLU: gate = SiLU(z), NOT GELU
    }
}

// MARK: - Single transformer layer (pre-norm: self-attn → cross-attn → FFN)

public final class SAODiTLayer: Module {
    @ModuleInfo(key: "pre_norm") public var norm1: LayerNorm
    @ModuleInfo(key: "self_attn") public var selfAttn: SAODiTSelfAttention
    @ModuleInfo(key: "cross_attend_norm") public var norm2: LayerNorm
    @ModuleInfo(key: "cross_attn") public var crossAttn: SAODiTCrossAttention
    @ModuleInfo(key: "ff_norm") public var norm3: LayerNorm
    @ModuleInfo(key: "ff") public var ff: SAODiTFFN

    public init(config: SFXDiTConfig) {
        _norm1.wrappedValue = LayerNorm(dimensions: config.hiddenSize)
        _selfAttn.wrappedValue = SAODiTSelfAttention(
            hiddenSize: config.hiddenSize,
            numHeads: config.numHeads,
            headDim: config.headDim,
            rotaryDim: config.rotaryDim
        )
        _norm2.wrappedValue = LayerNorm(dimensions: config.hiddenSize)
        _crossAttn.wrappedValue = SAODiTCrossAttention(
            hiddenSize: config.hiddenSize,
            numQHeads: config.numHeads,
            numKVHeads: config.numKVHeads,
            headDim: config.headDim,
            condDim: config.conditioningDim
        )
        _norm3.wrappedValue = LayerNorm(dimensions: config.hiddenSize)
        _ff.wrappedValue = SAODiTFFN(
            hiddenSize: config.hiddenSize,
            intermediateDim: config.ffIntermediateDim
        )
    }

    public func callAsFunction(
        _ x: MLXArray,
        crossCond: MLXArray,
        cosEmb: MLXArray,
        sinEmb: MLXArray
    ) -> MLXArray {
        var h = x
        h = h + selfAttn(norm1(h), cosEmb: cosEmb, sinEmb: sinEmb)
        h = h + crossAttn(norm2(h), conditioning: crossCond)
        h = h + ff(norm3(h))
        return h
    }
}

// MARK: - Transformer stack

//
// Actual checkpoint keys inside transformer:
//   project_in.weight   [1536, 64]   — expand latentDim → hiddenSize
//   project_out.weight  [64, 1536]   — contract hiddenSize → latentDim
//   layers.*            — 24 SAODiTLayer blocks
// No final LayerNorm (no transformer.norm.* key in checkpoint).

public final class SAODiTTransformer: Module {
    @ModuleInfo(key: "project_in") public var projectIn: Linear
    @ModuleInfo(key: "project_out") public var projectOut: Linear
    public var layers: [SAODiTLayer]

    public init(config: SFXDiTConfig) {
        _projectIn.wrappedValue = Linear(config.latentDim, config.hiddenSize, bias: false)
        _projectOut.wrappedValue = Linear(config.hiddenSize, config.latentDim, bias: false)
        layers = (0 ..< config.numLayers).map { _ in SAODiTLayer(config: config) }
    }

    /// - Parameters:
    ///   - x:            `[B, T, latentDim]` preprocessed latents
    ///   - prependToken: `[B, 1, hiddenSize]` conditioning token prepended to sequence.
    ///                   Attended by all sequence positions via self-attention, then removed.
    ///   - crossCond:    `[B, S, condDim]` projected conditioning tokens (T5 hidden states only)
    public func callAsFunction(
        _ x: MLXArray,
        prependToken: MLXArray,
        crossCond: MLXArray,
        cosEmb: MLXArray,
        sinEmb: MLXArray
    ) -> MLXArray {
        var h = projectIn(x) // [B, T, H]
        h = concatenated([prependToken, h], axis: 1) // [B, T+1, H]
        for layer in layers {
            h = layer(h, crossCond: crossCond, cosEmb: cosEmb, sinEmb: sinEmb)
        }
        // Remove the prepended conditioning token.
        let seqWithPrepend = h.shape[1]
        h = h[0..., 1 ..< seqWithPrepend, 0...] // [B, T, H]
        return projectOut(h) // [B, T, latentDim]
    }
}

// MARK: - Top-level diffusion transformer

public final class SFXDiTModel: Module {
    public let config: SFXDiTConfig

    @ModuleInfo(key: "preprocess_conv") public var preprocessConv: Conv1d
    @ModuleInfo(key: "timestep_features") public var timestepFeatures: SAOFourierFeatures
    @ModuleInfo(key: "to_timestep_embed") public var toTimestepEmbed: SAOBiasMLP
    @ModuleInfo(key: "to_cond_embed") public var toCondEmbed: SAOBiasMLP
    @ModuleInfo(key: "to_global_embed") public var toGlobalEmbed: SAOBiasMLP
    public var transformer: SAODiTTransformer
    @ModuleInfo(key: "postprocess_conv") public var postprocessConv: Conv1d

    public init(config: SFXDiTConfig = .sao) {
        self.config = config
        let halfDim = config.timestepFeaturesDim / 2
        _preprocessConv.wrappedValue = Conv1d(
            inputChannels: config.latentDim,
            outputChannels: config.latentDim,
            kernelSize: 1,
            bias: false
        )
        _timestepFeatures.wrappedValue = SAOFourierFeatures(halfDim: halfDim)
        _toTimestepEmbed.wrappedValue = SAOBiasMLP(
            inputDim: config.timestepFeaturesDim,
            hiddenDim: config.hiddenSize
        )
        _toCondEmbed.wrappedValue = SAOBiasMLP(
            inputDim: config.condTokenDim,
            hiddenDim: config.condTokenDim
        )
        _toGlobalEmbed.wrappedValue = SAOBiasMLP(
            inputDim: config.globalCondDim,
            hiddenDim: config.hiddenSize
        )
        transformer = SAODiTTransformer(config: config)
        _postprocessConv.wrappedValue = Conv1d(
            inputChannels: config.latentDim,
            outputChannels: config.latentDim,
            kernelSize: 1,
            bias: false
        )
    }

    /// Predict the velocity field for the given noisy latents.
    ///
    /// Forward pass (mirrors mlx-audiogen DiT):
    ///   1. preprocess_conv (k=1, identity channel count)
    ///   2. Timestep Fourier features → MLP → tEmb
    ///   3. Global embed: tEmb + toGlobalEmbed(globalCond) → prepend token [B,1,H]
    ///   4. Project cross-attention conditioning: toCondEmbed(crossCond) [B,S,768→768]
    ///   5. RoPE for seqLen+1 positions (prepend token at pos 0, latent frames at 1..T)
    ///   6. Transformer: prepend gEmb → 24 blocks → remove prepend → project_out
    ///   7. postprocess_conv (k=1, identity channel count)
    ///
    /// - Parameters:
    ///   - latents:    `[B, T, latentDim]` noisy latent frames
    ///   - timestep:   `[B]` or `[1]` continuous diffusion time ∈ (0, 1]
    ///   - crossCond:  `[B, S, condTokenDim]` T5 hidden states [B, 128, 768] (no time tokens)
    ///   - globalCond: `[B, globalCondDim]` concat(totalEmbed, startEmbed)
    ///                 from `SAONumberConditioner`
    /// - Returns: `[B, T, latentDim]` predicted velocity
    /// - Parameter diagEnabled: when `true` emit `[AUDIODIAG]` probes for every
    ///   internal stage (pass only on the first 1-2 calls per generation to avoid
    ///   flooding the console with 100+ duplicate log lines).
    public func forward(
        latents: MLXArray,
        timestep: MLXArray,
        crossCond: MLXArray,
        globalCond: MLXArray,
        diagEnabled: Bool = false
    ) -> MLXArray {
        let seqLen = latents.shape[1]

        // 1. Preprocess: kernel=1 conv applied as a residual correction.
        //    Python reference: x_t = x_t + self.preprocess_conv(x_t)
        let h = latents + preprocessConv(latents) // [B, T, latentDim]
        if diagEnabled {
            AudioDiagnostics.tensor("SFX_DIT_PREPROC", h)
        }

        // 2. Timestep Fourier encoding → MLP.
        let tFeats = timestepFeatures.encode(timestep) // [B, timestepFeaturesDim]
        let tEmb = toTimestepEmbed(tFeats) // [B, hiddenSize]
        if diagEnabled {
            AudioDiagnostics.tensor("SFX_DIT_TIMESTEP", timestep)
            AudioDiagnostics.tensor("SFX_DIT_TEMB", tEmb)
        }

        // 3. Global conditioning → add to timestep embed → prepend token.
        //    SAO uses prepend-style conditioning: global_cond+t_embed is prepended
        //    as a single extra token and removed after the transformer blocks.
        let gEmb = (tEmb + toGlobalEmbed(globalCond)) // [B, hiddenSize]
            .expandedDimensions(axis: 1) // [B, 1, hiddenSize]
        if diagEnabled {
            AudioDiagnostics.tensor("SFX_DIT_GLOBAL_EMB", gEmb)
        }

        // 4. Project cross-attention conditioning (T5 tokens → same 768 dim).
        //    to_cond_embed: 768→768 non-linear projection (project_cond_tokens=False).
        let projCross = toCondEmbed(crossCond) // [B, S, condTokenDim]
        if diagEnabled {
            AudioDiagnostics.tensor("SFX_DIT_PROJ_CROSS", projCross)
        }

        // 5. RoPE for seqLen+1 positions (prepend token at pos 0, latent frames at 1..T).
        let (cosEmb, sinEmb) = SFXDiTModel.computeRoPE(
            seqLen: seqLen + 1,
            rotaryDim: config.rotaryDim
        )

        // 6. Transformer: prepend gEmb → 24 blocks → remove prepend → project_out.
        let out = transformer(
            h,
            prependToken: gEmb,
            crossCond: projCross,
            cosEmb: cosEmb,
            sinEmb: sinEmb
        ) // [B, T, latentDim]
        if diagEnabled {
            AudioDiagnostics.tensor("SFX_DIT_TRANSFORMER_OUT", out)
        }

        // 7. Postprocess: kernel=1 conv applied as a residual correction.
        //    Python reference: x_t = x_t + self.postprocess_conv(x_t)
        let finalOut = out + postprocessConv(out) // [B, T, latentDim]
        if diagEnabled {
            AudioDiagnostics.tensor("SFX_DIT_OUT", finalOut)
        }
        return finalOut
    }

    // MARK: - RoPE helper

    /// Compute cos/sin position embeddings for partial RoPE.
    /// Returns `[1, 1, seqLen, rotaryDim]` tensors for broadcasting over heads and batch.
    public static func computeRoPE(
        seqLen: Int,
        rotaryDim: Int,
        base: Float = 10000.0
    ) -> (MLXArray, MLXArray) {
        let halfDim = rotaryDim / 2
        let invFreq = MLXArray((0 ..< halfDim).map { i -> Float in
            1.0 / pow(base, Float(2 * i) / Float(rotaryDim))
        }) // [halfDim]
        let pos = MLXArray((0 ..< seqLen).map { Float($0) }) // [seqLen]
        let angles = pos.reshaped([seqLen, 1]) * invFreq.reshaped([1, halfDim]) // [seqLen, halfDim]
        // Duplicate for the rotate-half pattern → [seqLen, rotaryDim].
        let doubled = concatenated([angles, angles], axis: -1)
        let cosE = cos(doubled).reshaped([1, 1, seqLen, rotaryDim])
        let sinE = sin(doubled).reshaped([1, 1, seqLen, rotaryDim])
        return (cosE, sinE)
    }
}

// MARK: - Private activation helpers

/// SiLU / Swish: x · σ(x).  Used as the gate in SAODiTFFN (SwiGLU).
private func saoSiLU(_ x: MLXArray) -> MLXArray {
    x * sigmoid(x)
}
