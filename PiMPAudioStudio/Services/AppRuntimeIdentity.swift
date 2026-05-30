import Foundation
import OSLog

enum AppRuntimeIdentity {
    static let displayName = "PiMP Audio Studio"
    static let compactName = "PiMPAudioStudio"
    static let bundleIdentifier = "com.pimpstudios.PiMPAudioStudio"
    static let legacyBundleIdentifier = "net.pimp.anvil.AnvilApp"
    static let legacySubsystem = "com.pimp.AnvilApp"
    static let logSubsystem = bundleIdentifier

    static let appSupportFolderName = "PiMP Audio Studio"
    static let legacyAppSupportFolderName = "Anvil"

    static let mcpScheme = "pimpaudio"
    static let legacyMCPScheme = "anvil"

    static let keychainServiceName = "com.pimpstudios.pimpaudiostudio.apikeys"
    static let legacyKeychainServiceName = "com.pimpstudios.anvil.apikeys"

    static let userDefaultsPrefix = bundleIdentifier
    static let legacyUserDefaultsPrefix = legacySubsystem

    static func defaultsKey(_ suffix: String) -> String {
        "\(userDefaultsPrefix).\(suffix)"
    }

    static func legacyDefaultsKey(_ suffix: String) -> String {
        "\(legacyUserDefaultsPrefix).\(suffix)"
    }
}

enum AppUserDefaultsKeys {
    static let outputDirectoryBookmark = AppRuntimeIdentity.defaultsKey("outputDirectoryBookmark")
    static let legacyOutputDirectoryBookmark = "outputDirectoryBookmark"

    static let providerCLIAccessBookmarkPrefix =
        AppRuntimeIdentity.defaultsKey("providerCLIAccessBookmark.")
    static let legacyProviderCLIAccessBookmarkPrefix = "providerCLIAccessBookmark."

    static let audioDebugVerbose = AppRuntimeIdentity.defaultsKey("audioDebugVerbose")
    static let legacyAudioDebugVerbose = "audioDebugVerbose"

    static let selectedAdapterID = AppRuntimeIdentity.defaultsKey("selectedAdapterID")
    static let legacySelectedAdapterID = "selectedAdapterID"

    static let selectedAdapterStrength = AppRuntimeIdentity.defaultsKey("selectedAdapterStrength")
    static let legacySelectedAdapterStrength = "selectedAdapterStrength"

    static let runtimeIdentityMigration = AppRuntimeIdentity.defaultsKey("migration.runtimeIdentity.v1")
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

    static var legacyRootDirectories: [URL] {
        let home = FileManager.default.homeDirectoryForCurrentUser
        let candidates = [
            applicationSupportBase.appendingPathComponent(
                AppRuntimeIdentity.legacyAppSupportFolderName,
                isDirectory: true
            ),
            home
                .appendingPathComponent("Library/Containers", isDirectory: true)
                .appendingPathComponent(AppRuntimeIdentity.legacyBundleIdentifier, isDirectory: true)
                .appendingPathComponent("Data/Library/Application Support", isDirectory: true)
                .appendingPathComponent(AppRuntimeIdentity.legacyAppSupportFolderName, isDirectory: true),
            home
                .appendingPathComponent("Library/Application Support", isDirectory: true)
                .appendingPathComponent(AppRuntimeIdentity.legacyAppSupportFolderName, isDirectory: true)
        ]

        return unique(candidates).filter {
            $0.standardizedFileURL != rootDirectory.standardizedFileURL
        }
    }

    private static func unique(_ urls: [URL]) -> [URL] {
        var seen = Set<String>()
        var result: [URL] = []
        for url in urls {
            let key = url.standardizedFileURL.path(percentEncoded: false)
            if seen.insert(key).inserted {
                result.append(url)
            }
        }
        return result
    }
}

enum AppRuntimeMigration {
    private static let logger = Logger(
        subsystem: AppRuntimeIdentity.logSubsystem,
        category: "RuntimeMigration"
    )

    static func migrateIfNeeded() {
        let defaults = UserDefaults.standard
        guard !defaults.bool(forKey: AppUserDefaultsKeys.runtimeIdentityMigration) else { return }

        migrateApplicationSupportRoots()
        migrateSimpleUserDefaults()

        defaults.set(true, forKey: AppUserDefaultsKeys.runtimeIdentityMigration)
    }

    private static func migrateApplicationSupportRoots() {
        let fileManager = FileManager.default
        let destinationRoot = AppStorageLocations.rootDirectory

        for legacyRoot in AppStorageLocations.legacyRootDirectories {
            var isDirectory: ObjCBool = false
            guard fileManager.fileExists(
                atPath: legacyRoot.path(percentEncoded: false),
                isDirectory: &isDirectory
            ), isDirectory.boolValue else {
                continue
            }

            do {
                try mergeDirectory(
                    from: legacyRoot,
                    into: destinationRoot,
                    fileManager: fileManager
                )
                logger.info(
                    "Migrated legacy app support root from \(legacyRoot.path(percentEncoded: false), privacy: .public)"
                )
            } catch {
                logger.error(
                    "Failed to migrate legacy app support root \(legacyRoot.path(percentEncoded: false), privacy: .public): \(error.localizedDescription, privacy: .public)"
                )
            }
        }
    }

    private static func migrateSimpleUserDefaults() {
        copyDefaultIfNeeded(from: AppUserDefaultsKeys.legacyAudioDebugVerbose, to: AppUserDefaultsKeys.audioDebugVerbose)
        copyDefaultIfNeeded(from: AppUserDefaultsKeys.legacySelectedAdapterID, to: AppUserDefaultsKeys.selectedAdapterID)
        copyDefaultIfNeeded(
            from: AppUserDefaultsKeys.legacySelectedAdapterStrength,
            to: AppUserDefaultsKeys.selectedAdapterStrength
        )
    }

    private static func copyDefaultIfNeeded(from legacyKey: String, to newKey: String) {
        let defaults = UserDefaults.standard
        guard defaults.object(forKey: newKey) == nil,
              let legacyValue = defaults.object(forKey: legacyKey) else {
            return
        }
        defaults.set(legacyValue, forKey: newKey)
    }

    private static func mergeDirectory(
        from source: URL,
        into destination: URL,
        fileManager: FileManager
    ) throws {
        try fileManager.createDirectory(at: destination, withIntermediateDirectories: true)

        let children = try fileManager.contentsOfDirectory(
            at: source,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        )

        for child in children {
            let target = destination.appendingPathComponent(child.lastPathComponent)
            var sourceIsDirectory: ObjCBool = false
            let sourceExists = fileManager.fileExists(
                atPath: child.path(percentEncoded: false),
                isDirectory: &sourceIsDirectory
            )
            guard sourceExists else { continue }

            var targetIsDirectory: ObjCBool = false
            let targetExists = fileManager.fileExists(
                atPath: target.path(percentEncoded: false),
                isDirectory: &targetIsDirectory
            )

            if targetExists {
                if sourceIsDirectory.boolValue, targetIsDirectory.boolValue {
                    try mergeDirectory(from: child, into: target, fileManager: fileManager)
                }
                continue
            }

            do {
                try fileManager.moveItem(at: child, to: target)
            } catch {
                try fileManager.copyItem(at: child, to: target)
            }
        }
    }
}
