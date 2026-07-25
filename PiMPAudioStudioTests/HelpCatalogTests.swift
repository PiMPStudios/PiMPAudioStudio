import Testing
@testable import PiMPAudioStudio

@Test func helpCatalogParsesUserReadmeIndexLinks() {
    let sample = """
    # User Docs
    | Topic | Guide |
    | --- | --- |
    | First launch | [Getting Started](getting-started.md) |
    | Models | [Model Setup](model-setup.md) |
    """
    let links = HelpCatalog.parseIndexLinks(from: sample)
    #expect(links.count == 2)
    #expect(links[0].title == "Getting Started")
    #expect(links[0].file == "getting-started.md")
    #expect(links[1].file == "model-setup.md")
}

@Test func helpCatalogLoadsUserDocsFromSourceTreeInDebug() {
    // DEBUG builds fall back to docs/user when the Help bundle folder is missing.
    let pages = HelpCatalog.pages()
    #expect(!pages.isEmpty)
    #expect(pages.contains(where: { $0.fileName == "getting-started.md" }))
    #expect(pages.contains(where: { $0.fileName == "editor.md" }))

    let body = HelpCatalog.loadMarkdown(named: "getting-started.md")
    #expect(body != nil)
    #expect(body?.contains("Getting Started") == true || body?.contains("getting started") == true
            || (body?.isEmpty == false))
}
