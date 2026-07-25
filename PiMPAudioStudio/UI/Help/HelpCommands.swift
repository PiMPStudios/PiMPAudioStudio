import SwiftUI

/// Help menu commands for the offline documentation window.
struct HelpCommands: Commands {
    @Environment(\.openWindow) private var openWindow

    var body: some Commands {
        CommandGroup(replacing: .help) {
            Button("PiMP Audio Studio Help") {
                openWindow(id: "documentation")
            }
            .keyboardShortcut("?", modifiers: .command)

            Divider()

            Button("Keyboard Shortcuts") {
                NotificationCenter.default.post(
                    name: .openHelpPage,
                    object: nil,
                    userInfo: ["fileName": "keyboard-shortcuts.md"]
                )
                openWindow(id: "documentation")
            }
        }
    }
}

extension Notification.Name {
    /// Optional deep-link into the Help window (`userInfo["fileName"]` = e.g. `editor.md`).
    static let openHelpPage = Notification.Name("openHelpPage")
}
