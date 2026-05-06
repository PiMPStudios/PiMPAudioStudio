import Foundation

struct HistoryEntry: Identifiable, Codable, Sendable, Equatable {
    let id: UUID
    let createdAt: Date
    let prompt: String
    let lyrics: String
    let duration: TimeInterval
    let seed: Int
    let method: String
    let guidanceScale: Float
    let precision: String
    let fileName: String

    static func fileName(prompt: String, date: Date = Date()) -> String {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyyMMdd_HHmmss"
        let timestamp = formatter.string(from: date)

        let words = prompt
            .lowercased()
            .components(separatedBy: .whitespaces)
            .prefix(5)
            .joined(separator: "-")
            .replacingOccurrences(
                of: "[^a-z0-9\\-]",
                with: "",
                options: .regularExpression
            )
            .prefix(50)

        return "anvil_\(timestamp)_\(words).wav"
    }
}
