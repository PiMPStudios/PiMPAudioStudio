import Foundation
import Testing
@testable import PiMPAudioStudio

@Test func youTubeURLPolicyAllowsHTTPSYouTubeHosts() {
    let allowed = [
        "https://www.youtube.com/@channel",
        "https://youtube.com/playlist?list=PLabc",
        "https://m.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://music.youtube.com/playlist?list=PLabc",
        "https://youtu.be/dQw4w9WgXcQ",
        "https://www.youtu.be/dQw4w9WgXcQ",
    ]
    for raw in allowed {
        let url = URL(string: raw)!
        #expect(YouTubeURLPolicy.isAllowed(url), "expected allowed: \(raw)")
    }
}

@Test func youTubeURLPolicyRejectsNonYouTubeAndNonHTTPS() {
    let rejected = [
        "http://www.youtube.com/@channel",
        "file:///etc/passwd",
        "https://example.com/watch?v=dQw4w9WgXcQ",
        "https://notyoutube.com/@channel",
        "https://youtube.com.evil.example/@channel",
        "https://evil.example/youtube.com",
        "ftp://youtube.com/watch?v=dQw4w9WgXcQ",
    ]
    for raw in rejected {
        let url = URL(string: raw)!
        #expect(YouTubeURLPolicy.isAllowed(url) == false, "expected rejected: \(raw)")
    }
}

@Test func youTubeURLPolicyAcceptsSafeVideoIDsOnly() {
    #expect(YouTubeURLPolicy.isSafeVideoID("dQw4w9WgXcQ"))
    #expect(YouTubeURLPolicy.isSafeVideoID("abc-DEF_123"))
    #expect(YouTubeURLPolicy.isSafeVideoID("") == false)
    #expect(YouTubeURLPolicy.isSafeVideoID("../etc/passwd") == false)
    #expect(YouTubeURLPolicy.isSafeVideoID("id with spaces") == false)
    #expect(YouTubeURLPolicy.isSafeVideoID("id;rm -rf") == false)
}

@Test func huggingFaceURLPolicyAllowsOfficialHosts() {
    let allowed = [
        "https://huggingface.co/user/repo",
        "https://www.huggingface.co/user/repo",
        "https://hf.co/user/repo",
        "https://www.hf.co/user/repo",
    ]
    for raw in allowed {
        let url = URL(string: raw)!
        #expect(HuggingFaceURLPolicy.isAllowed(url), "expected allowed: \(raw)")
    }
}

@Test func huggingFaceURLPolicyRejectsOtherHostsAndSchemes() {
    let rejected = [
        "http://huggingface.co/user/repo",
        "https://example.com/user/repo",
        "https://huggingface.co.evil.example/user/repo",
        "https://evil.huggingface.co.attacker.com/user/repo",
        "file:///tmp/adapter.json",
        "https://cdn-lfs.huggingface.co/file",
    ]
    for raw in rejected {
        let url = URL(string: raw)!
        #expect(HuggingFaceURLPolicy.isAllowed(url) == false, "expected rejected: \(raw)")
    }
}
