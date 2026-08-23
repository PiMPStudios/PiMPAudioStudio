import Darwin
import Foundation
import Testing
@testable import PiMPAudioStudio

@Test func rawHTTPRequestParsesGETHealth() {
    let request = """
    GET /health HTTP/1.1\r
    Host: 127.0.0.1:38765\r
    \r

    """
    let parsed = RawHTTPRequest(data: Data(request.utf8))
    #expect(parsed?.method == "GET")
    #expect(parsed?.path == "/health")
    #expect(parsed?.body.isEmpty == true)
}

@Test func rawHTTPRequestParsesPOSTBodyAndStripsQuery() {
    let body = #"{"jsonrpc":"2.0","method":"ping"}"#
    let request = """
    POST /mcp?session=1 HTTP/1.1\r
    Content-Type: application/json\r
    Content-Length: \(body.utf8.count)\r
    \r
    \(body)
    """
    let parsed = RawHTTPRequest(data: Data(request.utf8))
    #expect(parsed?.method == "POST")
    #expect(parsed?.path == "/mcp")
    #expect(parsed?.headers["content-type"] == "application/json")
    #expect(String(data: parsed?.body ?? Data(), encoding: .utf8) == body)
}

@Test func rawHTTPRequestRejectsMissingHeaderTerminator() {
    let request = "POST /mcp HTTP/1.1\r\nContent-Length: 1\r\n"
    #expect(RawHTTPRequest(data: Data(request.utf8)) == nil)
}

@Test func rawHTTPRequestRejectsNegativeContentLength() {
    let request = """
    POST /mcp HTTP/1.1\r
    Content-Length: -1\r
    \r

    """
    #expect(RawHTTPRequest(data: Data(request.utf8)) == nil)
}

@Test func rawHTTPRequestRejectsContentLengthPastAvailableBytes() {
    let request = """
    POST /mcp HTTP/1.1\r
    Content-Length: 99\r
    \r
    short
    """
    #expect(RawHTTPRequest(data: Data(request.utf8)) == nil)
}

@Test func rawHTTPRequestLastDuplicateHeaderWins() {
    let request = """
    POST /mcp HTTP/1.1\r
    Content-Length: 1\r
    X-Dup: first\r
    X-Dup: second\r
    \r
    x
    """
    let parsed = RawHTTPRequest(data: Data(request.utf8))
    #expect(parsed?.headers["x-dup"] == "second")
}

@Test func rawHTTPRequestExtractsBearerToken() {
    let request = """
    POST /mcp HTTP/1.1\r
    Authorization: Bearer abcdef123456\r
    Content-Length: 0\r
    \r

    """
    let parsed = RawHTTPRequest(data: Data(request.utf8))
    #expect(parsed?.bearerToken == "abcdef123456")
}

@Test func rawHTTPRequestTrustedOriginAllowsMissingAndLocalhost() {
    func request(origin: String?) -> RawHTTPRequest {
        var lines = ["POST /mcp HTTP/1.1"]
        if let origin {
            lines.append("Origin: \(origin)")
        }
        lines.append("Content-Length: 0")
        lines.append("")
        lines.append("")
        let text = lines.joined(separator: "\r\n")
        return RawHTTPRequest(data: Data(text.utf8))!
    }

    #expect(request(origin: nil).hasTrustedOrigin(allowNonLocalOrigins: false))
    #expect(request(origin: "http://127.0.0.1:38765").hasTrustedOrigin(allowNonLocalOrigins: false))
    #expect(request(origin: "http://localhost").hasTrustedOrigin(allowNonLocalOrigins: false))
    #expect(request(origin: "https://evil.example").hasTrustedOrigin(allowNonLocalOrigins: false) == false)
    #expect(request(origin: "https://evil.example").hasTrustedOrigin(allowNonLocalOrigins: true))
}

@Test func mcpTokensMatchIsConstantTimeAndRejectsMismatch() {
    #expect(MCPHTTPServer.tokensMatch("secret-token", "secret-token"))
    #expect(MCPHTTPServer.tokensMatch("secret-token", "secret-tokex") == false)
    #expect(MCPHTTPServer.tokensMatch("short", "much-longer-token") == false)
    #expect(MCPHTTPServer.tokensMatch("", "") )
}

@Test func mcpAcceptRetriesTransientErrnos() {
    #expect(MCPHTTPServer.isRetryableAcceptError(EINTR))
    #expect(MCPHTTPServer.isRetryableAcceptError(EAGAIN))
    #expect(MCPHTTPServer.isRetryableAcceptError(EWOULDBLOCK))
    #expect(MCPHTTPServer.isRetryableAcceptError(EMFILE))
    #expect(MCPHTTPServer.isRetryableAcceptError(ENFILE))
    #expect(MCPHTTPServer.isRetryableAcceptError(ECONNABORTED))
    #expect(MCPHTTPServer.isRetryableAcceptError(EBADF) == false)
    #expect(MCPHTTPServer.isRetryableAcceptError(EINVAL) == false)
}

@Test func mcpConfirmationRequiresConfirmWithoutElicitation() {
    #expect(
        MCPConfirmationDecision.evaluate(
            requiresConfirmation: true,
            confirmArgument: nil,
            isDryRun: false,
            clientSupportsFormElicitation: false
        ) == .declined(
            action: "missingConfirm",
            reason: "This tool requires confirm: true or MCP form elicitation approval."
        )
    )
    #expect(
        MCPConfirmationDecision.evaluate(
            requiresConfirmation: true,
            confirmArgument: true,
            isDryRun: false,
            clientSupportsFormElicitation: false
        ) == .proceed
    )
    #expect(
        MCPConfirmationDecision.evaluate(
            requiresConfirmation: true,
            confirmArgument: nil,
            isDryRun: true,
            clientSupportsFormElicitation: false
        ) == .proceed
    )
    #expect(
        MCPConfirmationDecision.evaluate(
            requiresConfirmation: true,
            confirmArgument: nil,
            isDryRun: false,
            clientSupportsFormElicitation: true
        ) == .elicit
    )
    #expect(
        MCPConfirmationDecision.evaluate(
            requiresConfirmation: false,
            confirmArgument: nil,
            isDryRun: false,
            clientSupportsFormElicitation: false
        ) == .proceed
    )
}

@Test func mcpTokenResolutionPrefersKeychainAndMigratesDefaults() {
    let fromKeychain = MCPTokenResolution.resolve(
        keychainValue: "keychain-token",
        defaultsValue: "defaults-token",
        generated: "generated-token"
    )
    #expect(fromKeychain.token == "keychain-token")
    #expect(fromKeychain.writeToKeychain == nil)
    #expect(fromKeychain.clearDefaults)

    let migrateDefaults = MCPTokenResolution.resolve(
        keychainValue: nil,
        defaultsValue: "defaults-token",
        generated: "generated-token"
    )
    #expect(migrateDefaults.token == "defaults-token")
    #expect(migrateDefaults.writeToKeychain == "defaults-token")
    #expect(migrateDefaults.clearDefaults)

    let generated = MCPTokenResolution.resolve(
        keychainValue: nil,
        defaultsValue: nil,
        generated: "generated-token"
    )
    #expect(generated.token == "generated-token")
    #expect(generated.writeToKeychain == "generated-token")
    #expect(generated.clearDefaults == false)
}
