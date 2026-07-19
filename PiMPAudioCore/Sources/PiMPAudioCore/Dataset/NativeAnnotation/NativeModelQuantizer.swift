import MLXNN

public enum NativeModelQuantizer {
    public static func quantize(
        model: Module,
        using configuration: NativeModelPackManifest.Quantization
    ) {
        MLXNN.quantize(
            model: model,
            groupSize: configuration.groupSize,
            bits: configuration.bits,
            mode: .affine
        )
    }
}
