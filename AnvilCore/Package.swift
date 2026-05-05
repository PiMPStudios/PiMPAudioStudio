// swift-tools-version: 5.10
import PackageDescription

let package = Package(
    name: "AnvilCore",
    platforms: [.macOS(.v14)],
    products: [
        .library(name: "AnvilCore", targets: ["AnvilCore"]),
    ],
    targets: [
        .target(name: "AnvilCore"),
        .testTarget(name: "AnvilCoreTests", dependencies: ["AnvilCore"]),
    ]
)
