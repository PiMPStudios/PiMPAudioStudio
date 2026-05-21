import Foundation

public struct AnvilHistoryRow: Codable, Equatable, Sendable {
    public let id: String
    public let fileName: String
    public let prompt: String
    public let modelID: String
    public let seed: Int?
    public let hasTrace: Bool
    public let createdAt: String

    public init(
        id: String,
        fileName: String,
        prompt: String,
        modelID: String,
        seed: Int?,
        hasTrace: Bool,
        createdAt: String
    ) {
        self.id = id
        self.fileName = fileName
        self.prompt = prompt
        self.modelID = modelID
        self.seed = seed
        self.hasTrace = hasTrace
        self.createdAt = createdAt
    }
}

public protocol AnvilHistoryProviding: Sendable {
    func listHistory(limit: Int) async throws -> [AnvilHistoryRow]
}

public struct AnvilHistoryEnvelope: Codable, Equatable, Sendable {
    public let count: Int
    public let history: [AnvilHistoryRow]

    public init(history: [AnvilHistoryRow]) {
        self.count = history.count
        self.history = history
    }
}

public struct StaticAnvilHistoryProvider: AnvilHistoryProviding {
    private let rows: [AnvilHistoryRow]

    public init(rows: [AnvilHistoryRow] = Self.fixtureRows) {
        self.rows = rows
    }

    public func listHistory(limit: Int) async throws -> [AnvilHistoryRow] {
        Array(rows.prefix(max(1, min(limit, 200))))
    }

    public static let fixtureRows: [AnvilHistoryRow] = [
        AnvilHistoryRow(
            id: "86B72111-6F3D-49E4-BF05-4D9B7E7B48C1",
            fileName: "2026-05-20_sparse-dark-guitar.wav",
            prompt: "Sparse dark guitar loop with tape wobble",
            modelID: "ace-step-v1.5",
            seed: 4281,
            hasTrace: true,
            createdAt: "2026-05-20T23:40:00Z"
        ),
    ]
}
