import AnvilMCPSDKSpike
import Foundation

@main
struct AnvilMCPSDKProbe {
    static func main() async throws {
        let provider = StaticAnvilHistoryProvider()
        let payload = try await AnvilSDKServerFactory.historyPayload(
            historyProvider: provider,
            limit: 25
        )

        print("tool=list_history")
        print("resource=anvil://history")
        print(payload)
    }
}
