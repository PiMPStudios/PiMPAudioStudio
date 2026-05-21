// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "MCPServerSDKSpike",
    platforms: [.macOS(.v14)],
    products: [
        .library(name: "AnvilMCPSDKSpike", targets: ["AnvilMCPSDKSpike"]),
        .executable(name: "anvil-mcp-sdk-probe", targets: ["AnvilMCPSDKProbe"]),
    ],
    dependencies: [
        .package(url: "https://github.com/modelcontextprotocol/swift-sdk.git", from: "0.11.0"),
    ],
    targets: [
        .target(
            name: "AnvilMCPSDKSpike",
            dependencies: [
                .product(name: "MCP", package: "swift-sdk"),
            ]
        ),
        .executableTarget(
            name: "AnvilMCPSDKProbe",
            dependencies: [
                "AnvilMCPSDKSpike",
                .product(name: "MCP", package: "swift-sdk"),
            ]
        ),
        .testTarget(
            name: "AnvilMCPSDKSpikeTests",
            dependencies: [
                "AnvilMCPSDKSpike",
                .product(name: "MCP", package: "swift-sdk"),
            ]
        ),
    ]
)
