import Foundation
import Testing
@testable import AnvilApp

@Test func providerCLIParsesClaudeSubscriptionAuth() {
    let result = ProviderCommandResult(
        command: "claude",
        arguments: ["auth", "status"],
        exitStatus: 0,
        stdout: """
        {"loggedIn":true,"authMethod":"claude.ai","apiProvider":"firstParty","email":"dahoodie@pimp.net","subscriptionType":"pro"}
        """,
        stderr: ""
    )

    let probe = ProviderCLIParsers.parseClaudeAuth(result)

    #expect(probe.state == .authenticated)
    #expect(probe.detail.contains("claude.ai"))
    #expect(probe.detail.contains("pro"))
}

@Test func providerCLIParsesClaudeLoggedOutJson() {
    let result = ProviderCommandResult(
        command: "claude",
        arguments: ["auth", "status"],
        exitStatus: 1,
        stdout: #"{"loggedIn":false,"authMethod":"none","apiProvider":"firstParty"}"#,
        stderr: ""
    )

    let probe = ProviderCLIParsers.parseClaudeAuth(result)

    #expect(probe.state == .unauthenticated)
    #expect(probe.detail == "Not signed in")
}

@Test func providerCLIParsesCodexChatGPTAuth() {
    let result = ProviderCommandResult(
        command: "codex",
        arguments: ["login", "status"],
        exitStatus: 0,
        stdout: "Logged in using ChatGPT\n",
        stderr: ""
    )

    let probe = ProviderCLIParsers.parseCodexAuth(result)

    #expect(probe.state == .authenticated)
    #expect(probe.detail == "Logged in using ChatGPT")
}

@Test func providerCLIParsesGrokModelsWithWarnings() {
    let result = ProviderCommandResult(
        command: "grok",
        arguments: ["models"],
        exitStatus: 0,
        stdout: """
        You are logged in with grok.com.

        grok-build (default)
        """,
        stderr: "[WARN] plugin unavailable\n"
    )

    let probe = ProviderCLIParsers.parseGrokAuth(result)

    #expect(probe.state == .authenticated)
    #expect(probe.detail.contains("grok.com"))
    #expect(probe.detail.contains("grok-build"))
}

@Test func providerCLIParsesGrokUnauthenticatedOutput() {
    let result = ProviderCommandResult(
        command: "grok",
        arguments: ["models"],
        exitStatus: 0,
        stdout: "You are not authenticated.\n",
        stderr: ""
    )

    let probe = ProviderCLIParsers.parseGrokAuth(result)
    let outcome = ProviderCLIParsers.testOutcome(from: probe)

    #expect(probe.state == .unauthenticated)
    #expect(probe.detail == "Not signed in")
    #expect(!outcome.success)
    #expect(outcome.detail == "Not signed in")
}

@Test func providerCLIParsesHuggingFaceWhoami() {
    let result = ProviderCommandResult(
        command: "hf",
        arguments: ["auth", "whoami"],
        exitStatus: 0,
        stdout: "Logged in\n  user: DaRealDahoodie\n",
        stderr: ""
    )

    let probe = ProviderCLIParsers.parseHuggingFaceAuth(result)

    #expect(probe.state == .authenticated)
    #expect(probe.detail == "Signed in as DaRealDahoodie")
}

@Test func providerCLIParsesGeminiJsonTestResponse() {
    let result = ProviderCommandResult(
        command: "gemini",
        arguments: ["--prompt", "Reply with OK only.", "--output-format", "json"],
        exitStatus: 0,
        stdout: #"{"response":"OK"}"#,
        stderr: ""
    )

    let outcome = ProviderCLIParsers.parseGeminiTest(result)

    #expect(outcome.success)
    #expect(outcome.detail == "Received response: OK")
}

@Test func providerCLIClassifiesGeminiWorkspaceTrustFailure() {
    let result = ProviderCommandResult(
        command: "gemini",
        arguments: ["--prompt", "Reply with OK only.", "--output-format", "json"],
        exitStatus: 1,
        stdout: "",
        stderr: "Approval mode overridden to \"default\" because the current folder is not trusted.\n"
    )

    let outcome = ProviderCLIParsers.parseGeminiTest(result)

    #expect(!outcome.success)
    #expect(outcome.isWorkspaceTrustNeeded)
    #expect(outcome.detail.contains("current folder is not trusted"))
}

@Test func providerCLIEnvironmentPreservesSandboxHomeWithAccountPathFallback() {
    let environment = ProviderCommandRunner.commandEnvironment(from: [
        "HOME": "/Users/testuser/Library/Containers/net.pimp.anvil.AnvilApp/Data",
        "USER": "testuser",
        "PATH": "/usr/bin:/bin",
        "ANVIL_DISABLE_PROVIDER_CLI_GRANTS": "1",
    ])

    #expect(environment["HOME"] == "/Users/testuser/Library/Containers/net.pimp.anvil.AnvilApp/Data")

    let pathDirectories = environment["PATH"]?.split(separator: ":").map(String.init) ?? []
    #expect(pathDirectories.first == "/Users/testuser/Library/Containers/net.pimp.anvil.AnvilApp/Data/.grok/bin")
    #expect(pathDirectories.contains("/Users/testuser/Library/Containers/net.pimp.anvil.AnvilApp/Data/.grok/bin"))
    #expect(pathDirectories.contains("/Users/testuser/.grok/bin"))

    let homebrewIndex = pathDirectories.firstIndex(of: "/opt/homebrew/bin")
    let accountGrokIndex = pathDirectories.firstIndex(of: "/Users/testuser/.grok/bin")
    guard let homebrewIndex, let accountGrokIndex else {
        #expect(Bool(false))
        return
    }
    #expect(homebrewIndex < accountGrokIndex)
}

@Test func providerCLIResolvesExecutableBeforeFallingBackToEnv() throws {
    let directoryURL = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString, isDirectory: true)
    try FileManager.default.createDirectory(
        at: directoryURL,
        withIntermediateDirectories: true
    )
    defer { try? FileManager.default.removeItem(at: directoryURL) }

    let toolURL = directoryURL.appendingPathComponent("mock-cli")
    FileManager.default.createFile(atPath: toolURL.path, contents: Data())
    try FileManager.default.setAttributes(
        [.posixPermissions: 0o755],
        ofItemAtPath: toolURL.path
    )

    let launch = ProviderCommandLaunch.resolve(
        command: "mock-cli",
        arguments: ["--version"],
        environment: ["PATH": directoryURL.path],
        fallbackURL: URL(fileURLWithPath: "/usr/bin/env")
    )

    #expect(launch.executableURL.path == toolURL.path)
    #expect(launch.processArguments == ["--version"])
    #expect(launch.diagnostic.contains("launch=direct"))
}

@Test func providerCLIResolvesSymlinkedExecutableTarget() throws {
    let directoryURL = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString, isDirectory: true)
    let binURL = directoryURL.appendingPathComponent("bin", isDirectory: true)
    let downloadsURL = directoryURL.appendingPathComponent("downloads", isDirectory: true)
    try FileManager.default.createDirectory(
        at: binURL,
        withIntermediateDirectories: true
    )
    try FileManager.default.createDirectory(
        at: downloadsURL,
        withIntermediateDirectories: true
    )
    defer { try? FileManager.default.removeItem(at: directoryURL) }

    let targetURL = downloadsURL.appendingPathComponent("grok-real")
    FileManager.default.createFile(atPath: targetURL.path, contents: Data())
    try FileManager.default.setAttributes(
        [.posixPermissions: 0o755],
        ofItemAtPath: targetURL.path
    )
    try FileManager.default.createSymbolicLink(
        at: binURL.appendingPathComponent("grok"),
        withDestinationURL: targetURL
    )

    let launch = ProviderCommandLaunch.resolve(
        command: "grok",
        arguments: ["--version"],
        environment: ["PATH": binURL.path],
        fallbackURL: URL(fileURLWithPath: "/usr/bin/env")
    )

    #expect(launch.executableURL.path == targetURL.path)
    #expect(launch.processArguments == ["--version"])
}

@Test func providerCLILaunchFailureIncludesDiagnostic() {
    let error = ProviderCommandError.launchFailed(
        "grok",
        "launch=direct /Users/test/.grok/downloads/grok",
        CocoaError(.fileNoSuchFile)
    )

    #expect(error.localizedDescription.contains("grok launch failed"))
    #expect(error.localizedDescription.contains("launch=direct"))
}

@Test func providerCLIClassifiesSandboxPermissionAsBlocked() {
    let result = ProviderCommandResult(
        command: "grok",
        arguments: ["models"],
        exitStatus: 126,
        stdout: "",
        stderr: "env: grok: Operation not permitted\n"
    )

    let probe = ProviderCLIParsers.parseGrokAuth(result)

    guard case .blocked(let detail) = probe.state else {
        #expect(Bool(false))
        return
    }

    #expect(detail.contains("Sandbox blocked CLI access"))
    #expect(detail.contains("Operation not permitted"))
}
