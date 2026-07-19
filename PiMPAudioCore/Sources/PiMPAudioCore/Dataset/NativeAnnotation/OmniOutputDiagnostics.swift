import Foundation

public struct OmniOutputDiagnostics: Codable, Equatable, Sendable {
    public let isEmpty: Bool
    public let lineCount: Int
    public let uniqueLineCount: Int
    public let maximumRepeatedLineCount: Int
    public let instrumentalLineCount: Int
    public let uniqueLineRatio: Double
    public let isPathologicalRepetition: Bool

    public init(text: String) {
        let normalizedLines = text
            .components(separatedBy: .newlines)
            .map {
                $0.trimmingCharacters(in: .whitespacesAndNewlines)
                    .lowercased()
            }
            .filter { !$0.isEmpty }
        let frequencies = Dictionary(
            normalizedLines.map { ($0, 1) },
            uniquingKeysWith: +
        )
        let maximumRepeatedLineCount = frequencies.values.max() ?? 0
        let uniqueLineCount = frequencies.count
        let lineCount = normalizedLines.count
        let uniqueLineRatio =
            lineCount == 0 ? 0 : Double(uniqueLineCount) / Double(lineCount)
        let instrumentalLineCount = normalizedLines.filter {
            $0.contains("instrumental")
        }.count
        let isPathologicalRepetition =
            maximumRepeatedLineCount >= 8
            || instrumentalLineCount >= 8
            || (lineCount >= 12 && uniqueLineRatio < 0.25)

        self.isEmpty = text.trimmingCharacters(
            in: .whitespacesAndNewlines
        ).isEmpty
        self.lineCount = lineCount
        self.uniqueLineCount = uniqueLineCount
        self.maximumRepeatedLineCount = maximumRepeatedLineCount
        self.instrumentalLineCount = instrumentalLineCount
        self.uniqueLineRatio = uniqueLineRatio
        self.isPathologicalRepetition = isPathologicalRepetition
    }
}
