import Testing
@testable import AnvilCore

@Test func anvilCoreHasVersion() {
    #expect(!AnvilCoreInfo.version.isEmpty)
}
