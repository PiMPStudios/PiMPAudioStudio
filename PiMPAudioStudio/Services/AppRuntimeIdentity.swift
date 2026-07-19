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

    /// JSON array of `{ "id": String, "strength": Double }` for multi-adapter stacks.
    static let selectedAdapterStack = AppRuntimeIdentity.defaultsKey("selectedAdapterStack")

    static let runPodTemplateID = AppRuntimeIdentity.defaultsKey("runpod.templateID")
    static let runPodAutomaticTemplate = AppRuntimeIdentity.defaultsKey("runpod.automaticTemplate")
    static let runPodImageName = AppRuntimeIdentity.defaultsKey("runpod.imageName")
    static let runPodGPUType = AppRuntimeIdentity.defaultsKey("runpod.gpuType")
    static let runPodInterruptible = AppRuntimeIdentity.defaultsKey("runpod.interruptible")
    static let runPodContainerDiskGB = AppRuntimeIdentity.defaultsKey("runpod.containerDiskGB")
    static let runPodVolumeGB = AppRuntimeIdentity.defaultsKey("runpod.volumeGB")
    static let runPodSSHKeyPath = AppRuntimeIdentity.defaultsKey("runpod.sshKeyPath")
    static let runPodLastArchivePath = AppRuntimeIdentity.defaultsKey("runpod.lastArchivePath")

    static let runtimeIdentityMigration = AppRuntimeIdentity.defaultsKey("migration.runtimeIdentity.v1")
    static let dataDirectoryMigration = AppRuntimeIdentity.defaultsKey("migration.dataDirectories.v2")
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

    static var migrationReportURL: URL {
        rootDirectory.appendingPathComponent("runtime_migration.json")
    }

    static var legacyRootDirectories: [URL] {
        let home = realUserHomeDirectory()
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

    private static func isSandboxContainerHome(_ path: String) -> Bool {
        path.contains("/Library/Containers/") && path.hasSuffix("/Data")
    }
}

struct AppDataDirectoryMigrationReport: Codable, Equatable {
    struct Entry: Codable, Equatable {
        var relativePath: String
        var sourcePath: String
        var destinationPath: String
        var action: String
        var message: String?
    }

    var schemaVersion: String
    var migratedAt: Date
    var sourceRoots: [String]
    var destinationRoot: String
    var entries: [Entry]

    var failedEntries: [Entry] {
        entries.filter { $0.action == "failed" }
    }

    var hasFailures: Bool {
        !failedEntries.isEmpty
    }
}

enum AppRuntimeMigration {
    private static let logger = Logger(
        subsystem: AppRuntimeIdentity.logSubsystem,
        category: "RuntimeMigration"
    )
    private static let dataDirectoryRelativePaths = [
        "models",
        "presets",
        "LLM",
        "SourceAudioImports",
        "TrainingDatasets",
        "MCPImports",
    ]

    static func migrateIfNeeded() {
        guard !isRunningUnitTests else { return }
        let defaults = UserDefaults.standard
        migrateRuntimeIdentityIfNeeded(defaults: defaults)
        migrateDataDirectoriesIfNeeded(defaults: defaults)
    }

    static func migrateRuntimeIdentityIfNeeded(defaults: UserDefaults = .standard) {
        guard !defaults.bool(forKey: AppUserDefaultsKeys.runtimeIdentityMigration) else { return }
        migrateSimpleUserDefaults(defaults: defaults)
        defaults.set(true, forKey: AppUserDefaultsKeys.runtimeIdentityMigration)
    }

    static func migrateDataDirectoriesIfNeeded(defaults: UserDefaults = .standard) {
        guard !defaults.bool(forKey: AppUserDefaultsKeys.dataDirectoryMigration) else { return }
        let report = migrateDataDirectories(
            from: AppStorageLocations.legacyRootDirectories,
            into: AppStorageLocations.rootDirectory,
            reportURL: AppStorageLocations.migrationReportURL
        )
        guard shouldMarkDataDirectoryMigrationComplete(report) else {
            logger.warning(
                "Data directory migration recorded \(report.failedEntries.count, privacy: .public) failures; leaving migration retryable"
            )
            return
        }
        defaults.set(true, forKey: AppUserDefaultsKeys.dataDirectoryMigration)
        logger.info(
            "Data directory migration finished with \(report.entries.count, privacy: .public) recorded entries"
        )
    }

    static func shouldMarkDataDirectoryMigrationComplete(_ report: AppDataDirectoryMigrationReport) -> Bool {
        !report.hasFailures
    }

    @discardableResult
    static func migrateDataDirectories(
        from legacyRoots: [URL],
        into destinationRoot: URL,
        reportURL: URL? = nil,
        fileManager: FileManager = .default,
        migratedAt: Date = Date()
    ) -> AppDataDirectoryMigrationReport {
        var report = AppDataDirectoryMigrationReport(
            schemaVersion: "1.0",
            migratedAt: migratedAt,
            sourceRoots: legacyRoots.map { $0.path(percentEncoded: false) },
            destinationRoot: destinationRoot.path(percentEncoded: false),
            entries: []
        )

        migrateApplicationSupportRoots(
            legacyRoots: legacyRoots,
            destinationRoot: destinationRoot,
            fileManager: fileManager,
            report: &report
        )

        if let reportURL {
            do {
                try writeReport(report, to: reportURL, fileManager: fileManager)
            } catch {
                logger.error(
                    "Failed to write data migration report \(reportURL.path(percentEncoded: false), privacy: .public): \(error.localizedDescription, privacy: .public)"
                )
            }
        }

        return report
    }

    private static func migrateApplicationSupportRoots(
        legacyRoots: [URL],
        destinationRoot: URL,
        fileManager: FileManager,
        report: inout AppDataDirectoryMigrationReport
    ) {
        for legacyRoot in legacyRoots {
            var isDirectory: ObjCBool = false
            guard fileManager.fileExists(
                atPath: legacyRoot.path(percentEncoded: false),
                isDirectory: &isDirectory
            ), isDirectory.boolValue else {
                continue
            }

            for relativePath in dataDirectoryRelativePaths {
                let source = legacyRoot.appendingPathComponent(relativePath, isDirectory: true)
                let destination = destinationRoot.appendingPathComponent(relativePath, isDirectory: true)
                var sourceIsDirectory: ObjCBool = false
                guard fileManager.fileExists(
                    atPath: source.path(percentEncoded: false),
                    isDirectory: &sourceIsDirectory
                ), sourceIsDirectory.boolValue else {
                    continue
                }

                do {
                    let action = try mergeItem(
                        from: source,
                        into: destination,
                        fileManager: fileManager
                    )
                    report.entries.append(
                        .init(
                            relativePath: relativePath,
                            sourcePath: source.path(percentEncoded: false),
                            destinationPath: destination.path(percentEncoded: false),
                            action: action,
                            message: nil
                        )
                    )
                    logger.info(
                        "Migrated legacy \(relativePath, privacy: .public) from \(source.path(percentEncoded: false), privacy: .public)"
                    )
                } catch {
                    report.entries.append(
                        .init(
                            relativePath: relativePath,
                            sourcePath: source.path(percentEncoded: false),
                            destinationPath: destination.path(percentEncoded: false),
                            action: "failed",
                            message: error.localizedDescription
                        )
                    )
                    logger.error(
                        "Failed to migrate legacy \(relativePath, privacy: .public) from \(source.path(percentEncoded: false), privacy: .public): \(error.localizedDescription, privacy: .public)"
                    )
                }
            }
        }
    }

    private static func migrateSimpleUserDefaults(defaults: UserDefaults) {
        copyDefaultIfNeeded(
            from: AppUserDefaultsKeys.legacyAudioDebugVerbose,
            to: AppUserDefaultsKeys.audioDebugVerbose,
            defaults: defaults
        )
        copyDefaultIfNeeded(
            from: AppUserDefaultsKeys.legacySelectedAdapterID,
            to: AppUserDefaultsKeys.selectedAdapterID,
            defaults: defaults
        )
        copyDefaultIfNeeded(
            from: AppUserDefaultsKeys.legacySelectedAdapterStrength,
            to: AppUserDefaultsKeys.selectedAdapterStrength,
            defaults: defaults
        )
    }

    private static func copyDefaultIfNeeded(from legacyKey: String, to newKey: String, defaults: UserDefaults) {
        guard defaults.object(forKey: newKey) == nil,
              let legacyValue = defaults.object(forKey: legacyKey) else {
            return
        }
        defaults.set(legacyValue, forKey: newKey)
    }

    private static func mergeItem(
        from source: URL,
        into destination: URL,
        fileManager: FileManager
    ) throws -> String {
        var destinationIsDirectory: ObjCBool = false
        let destinationExists = fileManager.fileExists(
            atPath: destination.path(percentEncoded: false),
            isDirectory: &destinationIsDirectory
        )

        if destinationExists {
            guard destinationIsDirectory.boolValue else { return "skipped_existing_file" }
            try mergeDirectory(from: source, into: destination, fileManager: fileManager)
            return "merged"
        }

        try fileManager.createDirectory(
            at: destination.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )

        do {
            try fileManager.moveItem(at: source, to: destination)
            return "moved"
        } catch {
            try fileManager.copyItem(at: source, to: destination)
            return "copied"
        }
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

    private static func writeReport(
        _ report: AppDataDirectoryMigrationReport,
        to url: URL,
        fileManager: FileManager
    ) throws {
        try fileManager.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(report)
        try data.write(to: url, options: .atomic)
    }

    private static var isRunningUnitTests: Bool {
        let environment = ProcessInfo.processInfo.environment
        if environment["XCTestConfigurationFilePath"] != nil {
            return true
        }
        return ProcessInfo.processInfo.arguments.contains { argument in
            argument.contains(".xctest") || argument.contains("xctest")
        }
    }
}
