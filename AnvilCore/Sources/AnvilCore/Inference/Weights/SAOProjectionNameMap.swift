// AnvilCore/Sources/AnvilCore/Inference/Weights/SAOProjectionNameMap.swift
//
// Stable Audio Open's projection_model has a small flat namespace. The
// safetensors keys are typically already `text_projection.weight` /
// `timing_projection.weight`, matching our `SAOProjectionModel` ModuleInfo
// keys directly. This map exists for symmetry with T5/CLAP and to handle
// any HF-side naming surprises (e.g., `text_proj.weight` variants) once
// we crack the gated repo and inspect the actual key layout.

import Foundation

public enum SAOProjectionNameMap {
    public static func mapKey(_ key: String) -> String? {
        // Pass-through for the canonical names. Add aliasing branches here
        // when we discover the actual safetensors layout.
        if key == "text_projection.weight" { return key }
        if key == "timing_projection.weight" { return key }

        // Common alias: HF often names projection layers as just `proj_*`.
        if key == "text_proj.weight" { return "text_projection.weight" }
        if key == "timing_proj.weight" { return "timing_projection.weight" }

        return nil
    }
}
