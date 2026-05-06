import SwiftUI
import AnvilCore

@MainActor
class EditorService: ObservableObject {
    @Published var coordinator: InpaintCoordinator?

    func setup(vaeActor: VAEActor) {
        self.coordinator = InpaintCoordinator(
            vaeActor: vaeActor
        )
    }
}
