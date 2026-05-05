// swift-tools-version: 5.10
import PackageDescription

let package = Package(
    name: "AnvilCore",
    platforms: [.macOS(.v14)],
    products: [
        .library(name: "AnvilCore", targets: ["AnvilCore"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.21.0"),
    ],
    targets: [
        .target(
            name: "AnvilCore",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
            ]
        ),
        .testTarget(name: "AnvilCoreTests", dependencies: ["AnvilCore"]),
    ]
)
