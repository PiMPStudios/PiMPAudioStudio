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
                        description: Text("User documentation was not found in the app bundle. Run Scripts/sync-user-help.sh and rebuild.")
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
                        MarkdownDocumentView(markdown: markdown)
                    }
                    .frame(maxWidth: 820, alignment: .leading)
                    .padding(24)
                    .frame(maxWidth: .infinity, alignment: .leading)
                }
                .background(Color(nsColor: .textBackgroundColor))
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
}

/// Renders Markdown using SwiftUI's built-in Markdown support.
struct MarkdownDocumentView: View {
    let markdown: String

    var body: some View {
        // Prefer full document; fall back to plain text if parsing fails.
        if let attributed = try? AttributedString(
            markdown: markdown,
            options: AttributedString.MarkdownParsingOptions(
                interpretedSyntax: .full
            )
        ) {
            Text(attributed)
                .font(.body)
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
        } else {
            Text(markdown)
                .font(.body.monospaced())
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
    }
}

#Preview {
    DocumentationView()
        .frame(width: 900, height: 640)
}
