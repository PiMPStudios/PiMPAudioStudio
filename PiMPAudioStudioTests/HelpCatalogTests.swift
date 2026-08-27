@testable import PiMPAudioStudio
import Testing

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

@Test func helpMarkdownParsesHeadingsListsCodeAndTables() {
    let sample = """
    # Title Page

    Intro paragraph that wraps
    across soft line breaks.

    ## Section

    - First item
    - Second **item**

    1. One
    2. Two

    ```text
    Settings -> Models
    ```

    | Col A | Col B |
    | --- | --- |
    | a1 | b1 |
    | a2 | b2 |

    See [Editor](editor.md).
    """

    let blocks = HelpMarkdown.parse(sample)
    #expect(blocks.contains(where: {
        if case .heading(1, "Title Page") = $0 {
            return true
        }
        return false
    }))
    #expect(blocks.contains(where: {
        if case .heading(2, "Section") = $0 {
            return true
        }
        return false
    }))
    #expect(blocks.contains(where: {
        if case let .paragraph(text) = $0 {
            return text.contains("Intro paragraph") && text.contains("soft line breaks")
        }
        return false
    }))
    #expect(blocks.contains(where: {
        if case let .unorderedList(items) = $0 {
            return items.count == 2 && items[0] == "First item"
        }
        return false
    }))
    #expect(blocks.contains(where: {
        if case let .orderedList(items) = $0 {
            return items == ["One", "Two"]
        }
        return false
    }))
    #expect(blocks.contains(where: {
        if case let .code(language, code) = $0 {
            return language == "text" && code.contains("Settings -> Models")
        }
        return false
    }))
    #expect(blocks.contains(where: {
        if case let .table(headers, rows) = $0 {
            return headers == ["Col A", "Col B"] && rows.count == 2 && rows[0] == ["a1", "b1"]
        }
        return false
    }))
    #expect(blocks.contains(where: {
        if case let .paragraph(text) = $0 {
            return text.contains("editor.md")
        }
        return false
    }))
}

@Test func helpMarkdownDropsRedundantH1MatchingPageTitle() {
    let blocks: [HelpMarkdownBlock] = [
        .heading(level: 1, text: "Getting Started"),
        .paragraph("Body"),
        .heading(level: 2, text: "Requirements"),
    ]
    let trimmed = HelpMarkdown.droppingRedundantTitle(from: blocks, pageTitle: "Getting Started")
    #expect(trimmed.count == 2)
    #expect({
        if case .paragraph("Body") = trimmed[0] {
            return true
        }
        return false
    }())
}

@Test func helpMarkdownParsesRealGettingStartedDoc() {
    guard let body = HelpCatalog.loadMarkdown(named: "getting-started.md") else {
        Issue.record("Could not load getting-started.md")
        return
    }
    let blocks = HelpMarkdown.parse(body)
    #expect(blocks.count >= 8)
    #expect(blocks.contains(where: {
        if case .heading(2, "Requirements") = $0 {
            return true
        }
        return false
    }))
    #expect(blocks.contains(where: {
        if case .unorderedList = $0 {
            return true
        }
        return false
    }))
    #expect(blocks.contains(where: {
        if case .orderedList = $0 {
            return true
        }
        return false
    }))
    #expect(blocks.contains(where: {
        if case .code = $0 {
            return true
        }
        return false
    }))
}
