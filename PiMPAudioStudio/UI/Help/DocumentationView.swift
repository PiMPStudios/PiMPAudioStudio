import SwiftUI

/// Offline user documentation browser (TOC + Markdown).
struct DocumentationView: View {
    @State private var pages: [HelpPage] = []
    @State private var selectedPageID: String?
    @State private var searchText = ""

    private var filteredPages: [HelpPage] {
        let query = searchText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !query.isEmpty else { return pages }
        return pages.filter { page in
            if page.title.localizedCaseInsensitiveContains(query) { return true }
            if page.fileName.localizedCaseInsensitiveContains(query) { return true }
            if let body = HelpCatalog.loadMarkdown(for: page),
               body.localizedCaseInsensitiveContains(query) {
                return true
            }
            return false
        }
    }

    private var selectedPage: HelpPage? {
        if let id = selectedPageID {
            return pages.first(where: { $0.id == id }) ?? filteredPages.first
        }
        return filteredPages.first
    }

    var body: some View {
        NavigationSplitView {
            List(selection: $selectedPageID) {
                if pages.isEmpty {
                    ContentUnavailableView(
                        "Help Unavailable",
                        systemImage: "book.closed",
                        description: Text("User documentation was not found in the app bundle. Rebuild so the Sync User Help Docs phase can copy docs/user.")
                    )
                } else {
                    Section("Topics") {
                        ForEach(filteredPages) { page in
                            Text(page.title)
                                .tag(page.id)
                        }
                    }
                }
            }
            .listStyle(.sidebar)
            .navigationTitle("Help")
            .searchable(text: $searchText, prompt: "Search help")
        } detail: {
            if let page = selectedPage,
               let markdown = HelpCatalog.loadMarkdown(for: page) {
                ScrollView {
                    VStack(alignment: .leading, spacing: 16) {
                        Text(page.title)
                            .font(.largeTitle.weight(.semibold))
                        MarkdownDocumentView(markdown: markdown, pageTitle: page.title)
                    }
                    .frame(maxWidth: 820, alignment: .leading)
                    .padding(24)
                    .frame(maxWidth: .infinity, alignment: .leading)
                }
                .background(Color(nsColor: .textBackgroundColor))
                .environment(\.openURL, OpenURLAction { url in
                    handleHelpLink(url)
                })
            } else if pages.isEmpty {
                ContentUnavailableView(
                    "No Help Pages",
                    systemImage: "doc.text.magnifyingglass",
                    description: Text("Bundle docs/user as Resources/Help and rebuild.")
                )
            } else {
                ContentUnavailableView(
                    "Select a Topic",
                    systemImage: "sidebar.left",
                    description: Text("Choose a page from the topics list.")
                )
            }
        }
        .onAppear {
            reload()
        }
        .onReceive(NotificationCenter.default.publisher(for: .openHelpPage)) { notification in
            reload()
            if let fileName = notification.userInfo?["fileName"] as? String {
                selectPage(fileName: fileName)
            }
        }
    }

    private func reload() {
        pages = HelpCatalog.pages()
        if selectedPageID == nil {
            selectedPageID = pages.first?.id
        } else if let id = selectedPageID, !pages.contains(where: { $0.id == id }) {
            selectedPageID = pages.first?.id
        }
    }

    private func selectPage(fileName: String) {
        if let match = pages.first(where: { $0.fileName == fileName || $0.id == fileName }) {
            selectedPageID = match.id
        }
    }

    /// Internal `.md` links jump to another Help topic; other URLs open normally.
    private func handleHelpLink(_ url: URL) -> OpenURLAction.Result {
        let fileName: String?
        if url.isFileURL {
            fileName = url.lastPathComponent
        } else if url.scheme == nil || url.scheme == "help" {
            fileName = url.path.isEmpty ? url.absoluteString : url.lastPathComponent
        } else if url.host == nil, url.path.hasSuffix(".md") {
            fileName = url.lastPathComponent
        } else {
            // Relative markdown often becomes a non-file URL with path "model-setup.md".
            let candidate = url.lastPathComponent
            fileName = candidate.hasSuffix(".md") ? candidate : nil
        }

        if let fileName, fileName.hasSuffix(".md") {
            selectPage(fileName: fileName)
            return .handled
        }

        // AbsoluteString fallback: "model-setup.md" or "./model-setup.md"
        let raw = url.absoluteString
        if raw.hasSuffix(".md") {
            let name = (raw as NSString).lastPathComponent
            selectPage(fileName: name)
            return .handled
        }

        return .systemAction
    }
}

/// Renders Help Markdown with real block layout (headings, lists, tables, code).
struct MarkdownDocumentView: View {
    let markdown: String
    var pageTitle: String = ""

    private var blocks: [HelpMarkdownBlock] {
        let parsed = HelpMarkdown.parse(markdown)
        return HelpMarkdown.droppingRedundantTitle(from: parsed, pageTitle: pageTitle)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            ForEach(Array(blocks.enumerated()), id: \.offset) { _, block in
                HelpMarkdownBlockView(block: block)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .textSelection(.enabled)
    }
}

// MARK: - Block views

private struct HelpMarkdownBlockView: View {
    let block: HelpMarkdownBlock

    var body: some View {
        switch block {
        case .heading(let level, let text):
            HelpInlineMarkdownText(text)
                .font(headingFont(level))
                .foregroundStyle(.primary)
                .padding(.top, level <= 2 ? 8 : 4)

        case .paragraph(let text):
            HelpInlineMarkdownText(text)
                .font(.body)
                .foregroundStyle(.primary)
                .fixedSize(horizontal: false, vertical: true)

        case .unorderedList(let items):
            VStack(alignment: .leading, spacing: 6) {
                ForEach(Array(items.enumerated()), id: \.offset) { _, item in
                    HStack(alignment: .firstTextBaseline, spacing: 8) {
                        Text("•")
                            .font(.body)
                            .foregroundStyle(.secondary)
                            .frame(width: 14, alignment: .center)
                        HelpInlineMarkdownText(item)
                            .font(.body)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }
            }
            .padding(.leading, 4)

        case .orderedList(let items):
            VStack(alignment: .leading, spacing: 6) {
                ForEach(Array(items.enumerated()), id: \.offset) { index, item in
                    HStack(alignment: .firstTextBaseline, spacing: 8) {
                        Text("\(index + 1).")
                            .font(.body.monospacedDigit())
                            .foregroundStyle(.secondary)
                            .frame(width: 24, alignment: .trailing)
                        HelpInlineMarkdownText(item)
                            .font(.body)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }
            }
            .padding(.leading, 4)

        case .code(_, let code):
            ScrollView(.horizontal, showsIndicators: false) {
                Text(code)
                    .font(.system(.callout, design: .monospaced))
                    .foregroundStyle(.primary)
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .padding(12)
            .background(
                RoundedRectangle(cornerRadius: 8, style: .continuous)
                    .fill(Color(nsColor: .controlBackgroundColor))
            )
            .overlay(
                RoundedRectangle(cornerRadius: 8, style: .continuous)
                    .strokeBorder(Color(nsColor: .separatorColor).opacity(0.6), lineWidth: 1)
            )

        case .table(let headers, let rows):
            HelpMarkdownTableView(headers: headers, rows: rows)

        case .thematicBreak:
            Divider()
                .padding(.vertical, 4)
        }
    }

    private func headingFont(_ level: Int) -> Font {
        switch level {
        case 1: return .title.weight(.semibold)
        case 2: return .title2.weight(.semibold)
        case 3: return .title3.weight(.semibold)
        default: return .headline
        }
    }
}

/// Inline Markdown (bold, code, links) for a single line/paragraph.
private struct HelpInlineMarkdownText: View {
    let source: String

    init(_ source: String) {
        self.source = source
    }

    var body: some View {
        Text(inlineAttributed)
            .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var inlineAttributed: AttributedString {
        let options = AttributedString.MarkdownParsingOptions(
            interpretedSyntax: .inlineOnlyPreservingWhitespace
        )
        if let attributed = try? AttributedString(markdown: source, options: options) {
            return attributed
        }
        return AttributedString(source)
    }
}

private struct HelpMarkdownTableView: View {
    let headers: [String]
    let rows: [[String]]

    private var columnCount: Int {
        max(headers.count, rows.map(\.count).max() ?? 0)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            if !headers.isEmpty {
                tableRow(cells: padded(headers), isHeader: true)
                Divider()
            }
            ForEach(Array(rows.enumerated()), id: \.offset) { index, row in
                tableRow(cells: padded(row), isHeader: false)
                if index < rows.count - 1 {
                    Divider().opacity(0.5)
                }
            }
        }
        .padding(10)
        .background(
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .fill(Color(nsColor: .controlBackgroundColor).opacity(0.55))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .strokeBorder(Color(nsColor: .separatorColor).opacity(0.6), lineWidth: 1)
        )
    }

    private func padded(_ cells: [String]) -> [String] {
        var result = cells
        while result.count < columnCount {
            result.append("")
        }
        return result
    }

    private func tableRow(cells: [String], isHeader: Bool) -> some View {
        HStack(alignment: .top, spacing: 0) {
            ForEach(Array(cells.enumerated()), id: \.offset) { _, cell in
                HelpInlineMarkdownText(cell)
                    .font(isHeader ? .callout.weight(.semibold) : .callout)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.vertical, 6)
                    .padding(.horizontal, 8)
            }
        }
    }
}

#Preview {
    DocumentationView()
        .frame(width: 900, height: 640)
}
