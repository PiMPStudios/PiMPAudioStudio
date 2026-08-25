import Foundation

enum AppRuntimeIdentity {
    static let displayName = "PiMP Audio Studio"
    static let compactName = "PiMPAudioStudio"
    static let bundleIdentifier = "com.pimpstudios.PiMPAudioStudio"
    static let logSubsystem = bundleIdentifier

    static let appSupportFolderName = "PiMP Audio Studio"

    static let mcpScheme = "pimpaudio"

    static let keychainServiceName = "com.pimpstudios.pimpaudiostudio.apikeys"

    static let userDefaultsPrefix = bundleIdentifier

    static func defaultsKey(_ suffix: String) -> String {
        "\(userDefaultsPrefix).\(suffix)"
    }
}

enum AppUserDefaultsKeys {
    static let outputDirectoryBookmark = AppRuntimeIdentity.defaultsKey("outputDirectoryBookmark")

    static let providerCLIAccessBookmarkPrefix =
        AppRuntimeIdentity.defaultsKey("providerCLIAccessBookmark.")

    static let audioDebugVerbose = AppRuntimeIdentity.defaultsKey("audioDebugVerbose")

    static let selectedAdapterID = AppRuntimeIdentity.defaultsKey("selectedAdapterID")

    static let selectedAdapterStrength = AppRuntimeIdentity.defaultsKey("selectedAdapterStrength")

    /// Generate page Prompt / Lyrics editor heights (points), vertical resize only.
    static let generatePromptEditorHeight = AppRuntimeIdentity.defaultsKey("generate.promptEditorHeight")
    static let generateLyricsEditorHeight = AppRuntimeIdentity.defaultsKey("generate.lyricsEditorHeight")

    /// JSON array of `{ "id": String, "strength": Double }` for multi-adapter stacks.
    static let selectedAdapterStack = AppRuntimeIdentity.defaultsKey("selectedAdapterStack")

    static let runPodTemplateID = AppRuntimeIdentity.defaultsKey("runpod.templateID")
    static let runPodAutomaticTemplate = AppRuntimeIdentity.defaultsKey("runpod.automaticTemplate")
    static let runPodImageName = AppRuntimeIdentity.defaultsKey("runpod.imageName")
    static let runPodGPUType = AppRuntimeIdentity.defaultsKey("runpod.gpuType")
    static let runPodRegion = AppRuntimeIdentity.defaultsKey("runpod.region")
    static let runPodContainerDiskGB = AppRuntimeIdentity.defaultsKey("runpod.containerDiskGB")
    static let runPodVolumeGB = AppRuntimeIdentity.defaultsKey("runpod.volumeGB")
    static let runPodSSHKeyPath = AppRuntimeIdentity.defaultsKey("runpod.sshKeyPath")
    static let runPodLastArchivePath = AppRuntimeIdentity.defaultsKey("runpod.lastArchivePath")
}

enum AppStorageLocations {
    static var applicationSupportBase: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
    }

    static var rootDirectory: URL {
        applicationSupportBase.appendingPathComponent(
            AppRuntimeIdentity.appSupportFolderName,
            isDirectory: true
        )
    }

    static var modelsDirectory: URL {
        rootDirectory.appendingPathComponent("models", isDirectory: true)
    }

    static var adaptersDirectory: URL {
        modelsDirectory.appendingPathComponent("adapters", isDirectory: true)
    }

    static var presetsDirectory: URL {
        rootDirectory.appendingPathComponent("presets", isDirectory: true)
    }

    static var llmModelsDirectory: URL {
        rootDirectory.appendingPathComponent("LLM", isDirectory: true)
    }

    static var sourceAudioImportsDirectory: URL {
        rootDirectory.appendingPathComponent("SourceAudioImports", isDirectory: true)
    }

    static var trainingDatasetsDirectory: URL {
        rootDirectory.appendingPathComponent("TrainingDatasets", isDirectory: true)
    }

    static var diagnosticsDirectory: URL {
        rootDirectory.appendingPathComponent("Diagnostics", isDirectory: true)
    }

    static func realUserHomeDirectory(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> URL {
        let environmentHome = environment["HOME"]?.trimmingCharacters(in: .whitespacesAndNewlines)
        let user = environment["USER"]?.trimmingCharacters(in: .whitespacesAndNewlines)
            ?? NSUserName()

        if let environmentHome, !environmentHome.isEmpty, !isSandboxContainerHome(environmentHome) {
            return URL(fileURLWithPath: environmentHome, isDirectory: true)
        }

        if !user.isEmpty,
           let accountHome = NSHomeDirectoryForUser(user),
           !accountHome.isEmpty,
           !isSandboxContainerHome(accountHome) {
            return URL(fileURLWithPath: accountHome, isDirectory: true)
        }

        if !user.isEmpty {
            return URL(fileURLWithPath: "/Users/\(user)", isDirectory: true)
        }

        if let environmentHome, !environmentHome.isEmpty {
            return URL(fileURLWithPath: environmentHome, isDirectory: true)
        }

        return FileManager.default.homeDirectoryForCurrentUser
    }

    private static func isSandboxContainerHome(_ path: String) -> Bool {
        path.contains("/Library/Containers/") && path.hasSuffix("/Data")
    }
}
