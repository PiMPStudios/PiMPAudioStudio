// swift-tools-version: 5.10
import PackageDescription

let package = Package(
    name: "PiMPAudioCore",
    platforms: [.macOS("15.0")],
    products: [
        .library(name: "PiMPAudioCore", targets: ["PiMPAudioCore"]),
        .executable(name: "QualityEval", targets: ["QualityEval"]),
        .executable(name: "DiTProbeEval", targets: ["DiTProbeEval"]),
        .executable(name: "MLXDropoutVJPProbe", targets: ["MLXDropoutVJPProbe"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.21.0"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.1.6"),
        .package(url: "https://github.com/soniqo/speech-swift.git", revision: "72a20dbb142d000b73d395b7bc62599fef8387e2"),
        .package(path: "../Vendor/demucs-mlx-swift"),
    ],
    targets: [
        .target(
            name: "PiMPAudioCore",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXOptimizers", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "Hub", package: "swift-transformers"),
                .product(name: "Tokenizers", package: "swift-transformers"),
                .product(name: "DemucsMLX", package: "demucs-mlx-swift"),
                .product(name: "Qwen3ASR", package: "speech-swift"),
                .product(name: "AudioCommon", package: "speech-swift"),
            ]
        ),
        .executableTarget(
            name: "QualityEval",
            dependencies: [
                "PiMPAudioCore",
                .product(name: "MLX", package: "mlx-swift"),
            ]
        ),
        .executableTarget(
            name: "DiTProbeEval",
            dependencies: [
                "PiMPAudioCore",
                .product(name: "MLX", package: "mlx-swift"),
            ]
        ),
        .executableTarget(
            name: "MLXDropoutVJPProbe",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
            ]
        ),
        .testTarget(name: "PiMPAudioCoreTests", dependencies: ["PiMPAudioCore"]),
    ]
)
