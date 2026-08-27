import Foundation

/// Offline user documentation loaded from the app bundle (`Resources/Help`).
///
/// Authoring source of truth remains `docs/user/`; the target **Sync User Help
/// Docs** build phase copies pages into the app bundle on every build.
struct HelpPage: Identifiable, Hashable, Sendable {
    /// File stem, e.g. `getting-started`.
    var id: String {
        fileName
    }

    /// Display title for the TOC.
    let title: String
    /// Bundled file name including `.md`.
    let fileName: String
}

enum HelpCatalog {
    /// Preferred reading order from `docs/user/README.md` (fallback if README parse fails).
    private static let defaultOrder: [(title: String, file: String)] = [
        ("Getting Started", "getting-started.md"),
        ("First Launch And Model Setup", "model-setup.md"),
        ("Generating Music", "generating.md"),
        ("Sound Effects", "sound-effects.md"),
        ("Audio Editor", "editor.md"),
        ("History And Presets", "history-and-presets.md"),
        ("Datasets And Training", "datasets-and-training.md"),
        ("Adapters", "adapters.md"),
        ("Remote GPU Training", "remote-gpu-training.md"),
        ("Intelligence And Lyrics", "intelligence.md"),
        ("Local CLI Access", "local-cli-access.md"),
        ("MCP For Users", "mcp.md"),
        ("Diagnostics And Logs", "diagnostics-and-logs.md"),
        ("Keyboard Shortcuts", "keyboard-shortcuts.md"),
        ("Troubleshooting", "troubleshooting.md"),
        ("Bug Report Checklist", "bug-report-checklist.md"),
    ]

    /// Root directory containing `*.md` help pages.
    static func helpDirectoryURL() -> URL? {
        if let bundled = Bundle.main.resourceURL?.appendingPathComponent("Help", isDirectory: true),
           FileManager.default.fileExists(atPath: bundled.path(percentEncoded: false)) {
            return bundled
        }

        #if DEBUG
            // Fallback when running from Xcode before the Help folder is copied into the bundle.
            let source = URL(fileURLWithPath: #filePath)
                .deletingLastPathComponent() // Services
                .deletingLastPathComponent() // PiMPAudioStudio
                .deletingLastPathComponent() // repo root
                .appendingPathComponent("docs/user", isDirectory: true)
            if FileManager.default.fileExists(atPath: source.path(percentEncoded: false)) {
                return source
            }
        #endif

        return nil
    }

    static func pages() -> [HelpPage] {
        guard let directory = helpDirectoryURL() else { return [] }
        let available = Set(
            (try? FileManager.default.contentsOfDirectory(atPath: directory.path(percentEncoded: false)))?
                .filter { $0.hasSuffix(".md") } ?? []
        )

        var ordered: [HelpPage] = []
        var seen = Set<String>()

        // Prefer README-derived titles when present.
        if let readme = loadMarkdown(named: "README.md") {
            for (title, file) in parseIndexLinks(from: readme) where available.contains(file) {
                ordered.append(HelpPage(title: title, fileName: file))
                seen.insert(file)
            }
        }

        for item in defaultOrder where available.contains(item.file) && !seen.contains(item.file) {
            ordered.append(HelpPage(title: item.title, fileName: item.file))
            seen.insert(item.file)
        }

        // Any leftover pages not listed above.
        for file in available.sorted() where !seen.contains(file) && file != "README.md" {
            let stem = (file as NSString).deletingPathExtension
            let title = stem
                .replacingOccurrences(of: "-", with: " ")
                .capitalized
            ordered.append(HelpPage(title: title, fileName: file))
        }

        return ordered
    }

    static func loadMarkdown(named fileName: String) -> String? {
        guard let directory = helpDirectoryURL() else { return nil }
        let url = directory.appendingPathComponent(fileName)
        return try? String(contentsOf: url, encoding: .utf8)
    }

    static func loadMarkdown(for page: HelpPage) -> String? {
        loadMarkdown(named: page.fileName)
    }

    /// Parse `[Title](file.md)` links from the user docs index.
    static func parseIndexLinks(from markdown: String) -> [(title: String, file: String)] {
        let pattern = #"\[([^\]]+)\]\(([^)]+\.md)\)"#
        guard let regex = try? NSRegularExpression(pattern: pattern) else { return [] }
        let ns = markdown as NSString
        let matches = regex.matches(in: markdown, range: NSRange(location: 0, length: ns.length))
        return matches.compactMap { match in
            guard match.numberOfRanges >= 3,
                  let titleRange = Range(match.range(at: 1), in: markdown),
                  let fileRange = Range(match.range(at: 2), in: markdown) else {
                return nil
            }
            let title = String(markdown[titleRange])
            let file = String(markdown[fileRange])
            // Skip self-references that aren't content pages.
            if file == "README.md" {
                return nil
            }
            return (title, file)
        }
    }
}
