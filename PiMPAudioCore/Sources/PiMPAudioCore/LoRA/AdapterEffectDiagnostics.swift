import Foundation
import MLX

/// Emits one high-signal adapter-effect line per adapter per generation.
///
/// This is intentionally separate from the existing tensor previews: those prove
/// weights were loaded, while this proves the adapter changed an actual layer
/// output and reports how large that change was relative to the base output.
public enum AdapterEffectDiagnostics {
    private static let lock = NSLock()
    private static var generationID = 0
    private static var emittedKeys = Set<String>()

    public static func beginGeneration() {
        lock.lock()
        generationID += 1
        emittedKeys.removeAll()
        lock.unlock()
    }

    static func contribution(
        kind: AdapterKind,
        adapterID: String,
        path: String?,
        strength: Float,
        scale: Float,
        baseOut: MLXArray,
        deltaOut: MLXArray
    ) {
        guard AudioDiagnostics.isEnabled else { return }

        let generation: Int
        lock.lock()
        generation = generationID
        let key = "\(generation)|\(kind.rawValue)|\(adapterID)"
        guard !emittedKeys.contains(key) else {
            lock.unlock()
            return
        }
        emittedKeys.insert(key)
        lock.unlock()

        let baseAbs = MLX.abs(baseOut.asType(.float32))
        let deltaAbs = MLX.abs(deltaOut.asType(.float32))
        let baseMeanArray = baseAbs.mean()
        let deltaMeanArray = deltaAbs.mean()
        let deltaMaxArray = deltaAbs.max()
        eval(baseMeanArray, deltaMeanArray, deltaMaxArray)

        let baseMean = baseMeanArray.item(Float.self)
        let deltaMean = deltaMeanArray.item(Float.self)
        let deltaMax = deltaMaxArray.item(Float.self)
        let ratio = deltaMean / max(baseMean, 1e-8)

        AudioDiagnostics.note(
            "adapter/effect",
            "generation=\(generation) kind=\(kind.rawValue) adapter=\(adapterID) " +
                "path=\(path ?? "unknown") strength=\(String(format: "%.3f", strength)) " +
                "layerScale=\(String(format: "%.3f", scale)) " +
                "baseAbsMean=\(String(format: "%.5g", baseMean)) " +
                "deltaAbsMean=\(String(format: "%.5g", deltaMean)) " +
                "deltaAbsMax=\(String(format: "%.5g", deltaMax)) " +
                "deltaToBase=\(String(format: "%.5g", ratio)) " +
                "deltaToBasePercent=\(String(format: "%.3f", ratio * 100.0)) " +
                "verdict=\(verdict(for: ratio)) meaning=\(meaning(for: ratio))"
        )
    }

    private static func verdict(for ratio: Float) -> String {
        switch ratio {
        case ..<0.005:
            "active_tiny"
        case ..<0.03:
            "active_subtle"
        case ..<0.15:
            "active_audible"
        default:
            "active_strong"
        }
    }

    private static func meaning(for ratio: Float) -> String {
        switch ratio {
        case ..<0.005:
            "adapter_active_but_likely_inaudible"
        case ..<0.03:
            "adapter_active_but_subtle"
        case ..<0.15:
            "adapter_active_at_audible_scale"
        default:
            "adapter_active_strong"
        }
    }
}
