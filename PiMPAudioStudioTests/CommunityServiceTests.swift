import Foundation
import Testing
@testable import PiMPAudioStudio

@Test func communityManifestURLUsesPiMPAdaptersRepo() {
    let url = CommunityService.communityManifestURL.absoluteString
    #expect(url == "https://raw.githubusercontent.com/PiMPStudios/pimp-adapters/main/manifest.json")
}
