import Foundation

/// Block-level Markdown subset used by offline Help (`docs/user`).
///
/// SwiftUI `Text(AttributedString(markdown:))` collapses presentation intents into
/// a single run with no inter-block spacing — this parser preserves structure.
enum HelpMarkdownBlock: Equatable, Sendable {
    case heading(level: Int, text: String)
    case paragraph(String)
    case unorderedList([String])
    case orderedList([String])
    case code(language: String?, code: String)
    case table(headers: [String], rows: [[String]])
    case thematicBreak
}

enum HelpMarkdown {
    /// Parse user-doc Markdown into layout blocks.
    static func parse(_ source: String) -> [HelpMarkdownBlock] {
        let normalized = source
            .replacingOccurrences(of: "\r\n", with: "\n")
            .replacingOccurrences(of: "\r", with: "\n")
        let lines = normalized.split(separator: "\n", omittingEmptySubsequences: false)
            .map(String.init)

        var blocks: [HelpMarkdownBlock] = []
        var index = 0

        while index < lines.count {
            let line = lines[index]
            let trimmed = line.trimmingCharacters(in: .whitespaces)

            if trimmed.isEmpty {
                index += 1
                continue
            }

            if trimmed == "---" || trimmed == "***" || trimmed == "___" {
                blocks.append(.thematicBreak)
                index += 1
                continue
            }

            if trimmed.hasPrefix("```") {
                let language = String(trimmed.dropFirst(3)).trimmingCharacters(in: .whitespaces)
                index += 1
                var codeLines: [String] = []
                while index < lines.count {
                    let codeLine = lines[index]
                    if codeLine.trimmingCharacters(in: .whitespaces).hasPrefix("```") {
                        index += 1
                        break
                    }
                    codeLines.append(codeLine)
                    index += 1
                }
                blocks.append(.code(
                    language: language.isEmpty ? nil : language,
                    code: codeLines.joined(separator: "\n")
                ))
                continue
            }

            if let heading = parseHeading(trimmed) {
                blocks.append(heading)
                index += 1
                continue
            }

            if isTableRow(trimmed),
               index + 1 < lines.count,
               isTableSeparator(lines[index + 1].trimmingCharacters(in: .whitespaces)) {
                let headers = splitTableCells(trimmed)
                index += 2
                var rows: [[String]] = []
                while index < lines.count {
                    let rowTrimmed = lines[index].trimmingCharacters(in: .whitespaces)
                    if !isTableRow(rowTrimmed) {
                        break
                    }
                    rows.append(splitTableCells(rowTrimmed))
                    index += 1
                }
                blocks.append(.table(headers: headers, rows: rows))
                continue
            }

            if isUnorderedItem(trimmed) {
                var items: [String] = []
                while index < lines.count {
                    let itemTrimmed = lines[index].trimmingCharacters(in: .whitespaces)
                    if itemTrimmed.isEmpty {
                        break
                    }
                    guard let item = unorderedItemText(itemTrimmed) else { break }
                    items.append(item)
                    index += 1
                }
                if !items.isEmpty {
                    blocks.append(.unorderedList(items))
                }
                continue
            }

            if isOrderedItem(trimmed) {
                var items: [String] = []
                while index < lines.count {
                    let itemTrimmed = lines[index].trimmingCharacters(in: .whitespaces)
                    if itemTrimmed.isEmpty {
                        break
                    }
                    guard let item = orderedItemText(itemTrimmed) else { break }
                    items.append(item)
                    index += 1
                }
                if !items.isEmpty {
                    blocks.append(.orderedList(items))
                }
                continue
            }

            // Paragraph: soft-wrap consecutive non-blank, non-block lines.
            var paragraphLines: [String] = [trimmed]
            index += 1
            while index < lines.count {
                let next = lines[index]
                let nextTrimmed = next.trimmingCharacters(in: .whitespaces)
                if nextTrimmed.isEmpty {
                    break
                }
                if nextTrimmed.hasPrefix("```") {
                    break
                }
                if parseHeading(nextTrimmed) != nil {
                    break
                }
                if isUnorderedItem(nextTrimmed) || isOrderedItem(nextTrimmed) {
                    break
                }
                if isTableRow(nextTrimmed),
                   index + 1 < lines.count,
                   isTableSeparator(lines[index + 1].trimmingCharacters(in: .whitespaces)) {
                    break
                }
                if nextTrimmed == "---" || nextTrimmed == "***" || nextTrimmed == "___" {
                    break
                }
                paragraphLines.append(nextTrimmed)
                index += 1
            }
            let text = paragraphLines.joined(separator: " ")
            if !text.isEmpty {
                blocks.append(.paragraph(text))
            }
        }

        return blocks
    }

    /// Drop a leading H1 when it duplicates the navigation title.
    static func droppingRedundantTitle(from blocks: [HelpMarkdownBlock], pageTitle: String) -> [HelpMarkdownBlock] {
        guard let first = blocks.first,
              case let .heading(level, text) = first,
              level == 1 else {
            return blocks
        }
        let normalizedHeading = normalizeTitle(text)
        let normalizedPage = normalizeTitle(pageTitle)
        if normalizedHeading == normalizedPage || text.caseInsensitiveCompare(pageTitle) == .orderedSame {
            return Array(blocks.dropFirst())
        }
        return blocks
    }

    // MARK: - Line helpers

    private static func parseHeading(_ line: String) -> HelpMarkdownBlock? {
        guard line.hasPrefix("#") else { return nil }
        var level = 0
        for ch in line {
            if ch == "#" {
                level += 1
            } else {
                break
            }
        }
        guard level >= 1, level <= 6 else { return nil }
        let rest = line.dropFirst(level)
        guard rest.first == " " || rest.isEmpty else { return nil }
        let text = rest.trimmingCharacters(in: .whitespaces)
        guard !text.isEmpty else { return nil }
        return .heading(level: level, text: text)
    }

    private static func isUnorderedItem(_ line: String) -> Bool {
        unorderedItemText(line) != nil
    }

    private static func unorderedItemText(_ line: String) -> String? {
        for prefix in ["- ", "* ", "+ "] {
            if line.hasPrefix(prefix) {
                return String(line.dropFirst(prefix.count))
            }
        }
        return nil
    }

    private static func isOrderedItem(_ line: String) -> Bool {
        orderedItemText(line) != nil
    }

    private static func orderedItemText(_ line: String) -> String? {
        guard let dot = line.firstIndex(of: ".") else { return nil }
        let number = line[..<dot]
        guard !number.isEmpty, number.allSatisfy(\.isNumber) else { return nil }
        let after = line[line.index(after: dot)...]
        guard after.first == " " else { return nil }
        return String(after.dropFirst()).trimmingCharacters(in: .whitespaces)
    }

    private static func isTableRow(_ line: String) -> Bool {
        line.contains("|") && !isTableSeparator(line)
    }

    private static func isTableSeparator(_ line: String) -> Bool {
        guard line.contains("|"), line.contains("-") else { return false }
        let stripped = line
            .replacingOccurrences(of: "|", with: "")
            .replacingOccurrences(of: ":", with: "")
            .replacingOccurrences(of: "-", with: "")
            .trimmingCharacters(in: .whitespaces)
        return stripped.isEmpty
    }

    private static func splitTableCells(_ line: String) -> [String] {
        var raw = line.trimmingCharacters(in: .whitespaces)
        if raw.hasPrefix("|") {
            raw.removeFirst()
        }
        if raw.hasSuffix("|") {
            raw.removeLast()
        }
        return raw.split(separator: "|", omittingEmptySubsequences: false)
            .map { $0.trimmingCharacters(in: .whitespaces) }
    }

    private static func normalizeTitle(_ value: String) -> String {
        value
            .lowercased()
            .replacingOccurrences(of: "-", with: " ")
            .split(whereSeparator: { $0.isWhitespace })
            .joined(separator: " ")
    }
}
