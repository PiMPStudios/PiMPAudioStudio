import Foundation

public func listAvailableDemucsModels() -> [String] {
    DemucsModelRegistry.allModelNames
}
