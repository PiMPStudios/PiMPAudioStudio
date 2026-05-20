import Foundation
import AnvilCore

@MainActor
final class AnvilCommandFacade {
    private let registry: ModelRegistry
    private let promptEnhancer: PromptEnhancer
    private let lyricWriterManager: LyricWriterManager
    private var historyStore: HistoryStore
    private var generationService: GenerationService?
    private let dateFormatter = ISO8601DateFormatter()

    init(
        registry: ModelRegistry,
        historyStore: HistoryStore,
        promptEnhancer: PromptEnhancer,
        lyricWriterManager: LyricWriterManager
    ) {
        self.registry = registry
        self.promptEnhancer = promptEnhancer
        self.lyricWriterManager = lyricWriterManager
        self.historyStore = historyStore
    }

    func configure(historyStore: HistoryStore) {
        self.historyStore = historyStore
    }

    func configure(generationService: GenerationService) {
        self.generationService = generationService
    }

    func listModels() async -> [String: Any] {
        let specs = await registry.allSpecs
        var activeByCategory: [String: String] = [:]
        let categories = Set(specs.map(\.category))
        for category in categories {
            if let activeID = await registry.activeModelID(for: category) {
                activeByCategory[category.rawValue] = activeID
            }
        }

        var models: [[String: Any]] = []
        for spec in specs {
            models.append([
                "id": spec.id,
                "displayName": spec.displayName,
                "category": spec.category.rawValue,
                "family": spec.variantFamily ?? NSNull(),
                "huggingFaceRepo": spec.huggingFaceRepo,
                "totalSizeBytes": spec.totalSizeBytes,
                "totalSizeFormatted": spec.totalSizeFormatted,
                "requiresAuth": spec.requiresAuth,
                "isRunnable": spec.isRunnable,
                "runtimeNote": spec.runtimeNote ?? NSNull(),
                "status": await registry.status(for: spec.id).mcpDescription,
                "isActive": activeByCategory[spec.category.rawValue] == spec.id
            ])
        }
        models.sort {
            String(describing: $0["id"] ?? "") < String(describing: $1["id"] ?? "")
        }

        return [
            "activeModels": activeByCategory,
            "models": models
        ]
    }

    func listAdapters() async -> [String: Any] {
        let adapters = await Task.detached(priority: .userInitiated) {
            AdapterRegistry.listAdapters()
        }.value

        return [
            "adapters": adapters.map { adapter in
                [
                    "id": adapter.id,
                    "name": adapter.name,
                    "baseModelID": adapter.baseModelID,
                    "baseModelCategory": adapter.baseModelCategory.rawValue,
                    "rank": adapter.config.rank,
                    "alpha": adapter.config.alpha,
                    "injectionPatterns": adapter.config.injectionPatterns,
                    "sampleCount": adapter.sampleCount,
                    "finalLoss": adapter.finalLoss.map { Double($0) } ?? NSNull(),
                    "createdAt": dateFormatter.string(from: adapter.createdAt),
                    "hasCheckpoint": AdapterRegistry.hasCheckpoint(adapterID: adapter.id)
                ] as [String: Any]
            }
        ]
    }

    func listHistory(limit: Int) -> [String: Any] {
        let cappedLimit = max(1, min(limit, 200))
        let rows = historyStore.entries.prefix(cappedLimit).map(historySummary)
        return [
            "count": rows.count,
            "history": Array(rows)
        ]
    }

    func listResources() async -> [[String: Any]] {
        var resources: [[String: Any]] = [
            resource(
                uri: "anvil://models",
                name: "models",
                title: "Anvil Models",
                description: "Current model registry state and active model selections."
            ),
            resource(
                uri: "anvil://adapters",
                name: "adapters",
                title: "Anvil LoRA Adapters",
                description: "Installed LoRA adapters and checkpoint metadata."
            ),
            resource(
                uri: "anvil://history",
                name: "history",
                title: "Recent Generation History",
                description: "Recent generation rows with model, seed, prompt, audio path, and trace availability."
            )
        ]

        let specs = await registry.allSpecs
        for spec in specs.sorted(by: { $0.id < $1.id }) {
            resources.append(resource(
                uri: resourceURI(kind: "model", identifier: spec.id),
                name: spec.id,
                title: spec.displayName,
                description: "Model registry entry for \(spec.displayName)."
            ))
        }

        let adapters = await loadAdapters()
        for adapter in adapters.sorted(by: { $0.name < $1.name }) {
            resources.append(resource(
                uri: resourceURI(kind: "adapter", identifier: adapter.id),
                name: adapter.id,
                title: adapter.name,
                description: "LoRA adapter metadata for \(adapter.name)."
            ))
        }

        for entry in historyStore.entries.prefix(100) {
            resources.append(resource(
                uri: resourceURI(kind: "history", identifier: entry.id.uuidString),
                name: entry.id.uuidString,
                title: compact(entry.prompt, limit: 80),
                description: "Generation history entry for \(entry.fileName)."
            ))

            if historyStore.traceURL(for: entry) != nil {
                resources.append(resource(
                    uri: resourceURI(kind: "trace", identifier: entry.id.uuidString),
                    name: entry.traceFileName ?? HistoryEntry.traceFileName(forAudioFileName: entry.fileName),
                    title: "Trace: \(compact(entry.prompt, limit: 70))",
                    description: "Generation trace sidecar for \(entry.fileName)."
                ))
            }

            resources.append(resource(
                uri: resourceURI(kind: "audio", identifier: entry.id.uuidString),
                name: entry.fileName,
                title: "Audio: \(compact(entry.prompt, limit: 70))",
                description: "Local audio file metadata and path. Audio bytes are not streamed."
            ))
        }

        return resources
    }

    func readResource(uri: String) async throws -> [String: Any] {
        guard let parsed = AnvilResourceURI(uri) else {
            throw AnvilCommandFacadeError.invalidResourceURI(uri)
        }

        switch parsed.kind {
        case "models":
            return await listModels()
        case "model":
            guard let identifier = parsed.identifier else {
                throw AnvilCommandFacadeError.invalidResourceURI(uri)
            }
            return try await modelResource(id: identifier)
        case "adapters":
            return await listAdapters()
        case "adapter":
            guard let identifier = parsed.identifier else {
                throw AnvilCommandFacadeError.invalidResourceURI(uri)
            }
            return try await adapterResource(id: identifier)
        case "history":
            guard let identifier = parsed.identifier else {
                return listHistory(limit: 200)
            }
            let entry = try resolveHistoryEntry(identifier)
            return historySummary(entry)
        case "trace":
            guard let identifier = parsed.identifier else {
                throw AnvilCommandFacadeError.invalidResourceURI(uri)
            }
            return try getTrace(arguments: ["id": identifier])
        case "audio":
            guard let identifier = parsed.identifier else {
                throw AnvilCommandFacadeError.invalidResourceURI(uri)
            }
            return try audioResource(identifier: identifier)
        default:
            throw AnvilCommandFacadeError.resourceNotFound(uri)
        }
    }

    func getTrace(arguments: [String: Any]) throws -> [String: Any] {
        let trace = try resolveTrace(arguments: arguments)
        return try tracePayload(trace)
    }

    func compareTraces(arguments: [String: Any]) throws -> [String: Any] {
        let left = try resolveTrace(arguments: [
            "id": arguments["left"] ?? arguments["leftID"] ?? arguments["leftTrace"] ?? ""
        ])
        let right = try resolveTrace(arguments: [
            "id": arguments["right"] ?? arguments["rightID"] ?? arguments["rightTrace"] ?? ""
        ])
        let groups = GenerationTraceComparator.groups(left: left, right: right)
        let changedGroups = groups.compactMap { group -> [String: Any]? in
            let rows = group.rows.filter(\.isDifferent)
            guard !rows.isEmpty else { return nil }
            return [
                "title": group.title,
                "systemImage": group.systemImage,
                "rows": rows.map(rowPayload)
            ]
        }

        return [
            "left": traceSummary(left),
            "right": traceSummary(right),
            "changedFieldCount": changedGroups.reduce(0) { total, group in
                total + ((group["rows"] as? [[String: Any]])?.count ?? 0)
            },
            "groups": changedGroups
        ]
    }

    func markPreferredTake(arguments: [String: Any]) throws -> [String: Any] {
        let trace = try resolveTrace(arguments: arguments)
        let hasPreferredArgument = hasArgument("isPreferred", in: arguments)
            || hasArgument("preferred", in: arguments)
        let preferredValue = arguments["isPreferred"] ?? arguments["preferred"]
        let isPreferred: Bool
        if hasPreferredArgument {
            guard let parsed = boolValue(preferredValue) else {
                throw AnvilCommandFacadeError.invalidBoolean("isPreferred")
            }
            isPreferred = parsed
        } else {
            isPreferred = true
        }

        let note: String?
        if hasArgument("listeningNote", in: arguments) || hasArgument("note", in: arguments) {
            note = stringValue(arguments["listeningNote"] ?? arguments["note"])
        } else {
            note = trace.listeningNote
        }

        let updated = try historyStore.saveTraceReview(
            traceFileName: trace.traceFileName,
            listeningNote: note,
            isPreferred: isPreferred
        )

        return reviewPayload(updated)
    }

    func enhancePrompt(arguments: [String: Any]) async throws -> [String: Any] {
        let prompt = try requiredString("prompt", in: arguments)
        let mode = generationMode(from: arguments["mode"] ?? arguments["generationMode"]) ?? .music
        let enhanced = await promptEnhancer.enhance(prompt: prompt, mode: mode)

        return [
            "prompt": prompt,
            "enhancedPrompt": enhanced,
            "mode": mode.rawValue,
            "modeDisplayName": mode.displayName,
            "isAvailable": promptEnhancer.isAvailable,
            "changed": normalizedText(enhanced) != normalizedText(prompt)
        ]
    }

    func writeLyrics(arguments: [String: Any]) async throws -> [String: Any] {
        let prompt = try requiredString("prompt", in: arguments)
        let style = stringValue(arguments["style"]) ?? ""
        let duration = try requiredDuration(in: arguments)
        let lyrics = try await lyricWriterManager.writeLyricsText(
            prompt: prompt,
            style: style,
            duration: duration
        )

        return [
            "prompt": prompt,
            "style": style,
            "duration": duration,
            "backend": lyricWriterManager.activeBackend,
            "lyrics": lyrics,
            "lineCount": lyricLineCount(lyrics)
        ]
    }

    func applyAdapter(arguments: [String: Any]) async throws -> [String: Any] {
        guard let generationService else {
            throw AnvilCommandFacadeError.generationServiceUnavailable
        }

        if boolValue(arguments["clear"] ?? arguments["none"]) == true {
            generationService.selectedAdapterID = nil
            generationService.adapterStack = []
            persistAdapterSelection([])
            notifyAdapterSelectionChanged([])
            return adapterSelectionPayload([])
        }

        let installed = await loadAdapters()
        let selection = try parseAdapterSelection(arguments, installed: installed)
        generationService.selectedAdapterID = selection.first?.adapter.id
        generationService.adapterStack = selection.map { (id: $0.adapter.id, weight: $0.strength) }
        persistAdapterSelection(selection)
        notifyAdapterSelectionChanged(selection)
        return adapterSelectionPayload(selection)
    }

    func releaseMemory(arguments: [String: Any]) async throws -> [String: Any] {
        guard boolValue(arguments["confirm"]) == true else {
            throw AnvilCommandFacadeError.confirmationRequired("release_memory")
        }
        guard let generationService else {
            throw AnvilCommandFacadeError.generationServiceUnavailable
        }

        let summary = await generationService.unloadAllWeights()
        return memoryReleasePayload(summary)
    }

    private func loadAdapters() async -> [LoRAAdapter] {
        await Task.detached(priority: .userInitiated) {
            AdapterRegistry.listAdapters()
        }.value
    }

    private func parseAdapterSelection(
        _ arguments: [String: Any],
        installed: [LoRAAdapter]
    ) throws -> [AdapterSelection] {
        var selection: [AdapterSelection] = []

        if let rawStack = arguments["adapters"] ?? arguments["adapterStack"] ?? arguments["stack"] {
            let items: [[String: Any]]
            if let typed = rawStack as? [[String: Any]] {
                items = typed
            } else if let rawItems = rawStack as? [Any] {
                items = rawItems.compactMap { $0 as? [String: Any] }
            } else {
                throw AnvilCommandFacadeError.invalidAdapterStack
            }

            for item in items {
                let identifier = stringValue(item["id"] ?? item["adapterID"] ?? item["name"])?
                    .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
                guard !identifier.isEmpty else {
                    throw AnvilCommandFacadeError.missingRequiredArgument("id")
                }
                let adapter = try resolveAdapter(identifier, installed: installed)
                let strength = try adapterStrength(item["strength"] ?? item["weight"] ?? item["scale"])
                selection.append(AdapterSelection(adapter: adapter, strength: strength))
            }
        } else {
            let identifier = stringValue(arguments["id"] ?? arguments["adapterID"] ?? arguments["name"])?
                .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
            guard !identifier.isEmpty else {
                throw AnvilCommandFacadeError.missingRequiredArgument("id")
            }
            let adapter = try resolveAdapter(identifier, installed: installed)
            let strength = try adapterStrength(arguments["strength"] ?? arguments["weight"] ?? arguments["scale"])
            selection.append(AdapterSelection(adapter: adapter, strength: strength))
        }

        var unique: [AdapterSelection] = []
        var seen = Set<String>()
        for entry in selection where seen.insert(entry.adapter.id).inserted {
            unique.append(entry)
        }

        guard !unique.isEmpty else {
            throw AnvilCommandFacadeError.missingRequiredArgument("id")
        }
        guard unique.count <= AdapterStackManager.maxStackSize else {
            throw AnvilCommandFacadeError.tooManyAdapters(AdapterStackManager.maxStackSize)
        }
        for entry in unique where entry.adapter.baseModelCategory != .aceStepDiT {
            throw AnvilCommandFacadeError.incompatibleAdapter(
                entry.adapter.name,
                entry.adapter.baseModelCategory.rawValue
            )
        }

        return unique
    }

    private func resolveAdapter(_ identifier: String, installed: [LoRAAdapter]) throws -> LoRAAdapter {
        let needle = identifier.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        if let exactID = installed.first(where: { $0.id.lowercased() == needle }) {
            return exactID
        }
        if let exactName = installed.first(where: { $0.name.lowercased() == needle }) {
            return exactName
        }
        if let partial = installed.first(where: {
            $0.id.lowercased().contains(needle) || $0.name.lowercased().contains(needle)
        }) {
            return partial
        }
        throw AnvilCommandFacadeError.adapterNotFound(identifier)
    }

    private func adapterStrength(_ value: Any?) throws -> Float {
        guard let value else { return 1.0 }
        guard let parsed = doubleValue(value), parsed >= 0.0, parsed <= 1.0 else {
            throw AnvilCommandFacadeError.invalidStrength
        }
        return Float(parsed)
    }

    private func modelResource(id: String) async throws -> [String: Any] {
        let specs = await registry.allSpecs
        guard let spec = specs.first(where: { $0.id == id }) else {
            throw AnvilCommandFacadeError.resourceNotFound(resourceURI(kind: "model", identifier: id))
        }

        let activeID = await registry.activeModelID(for: spec.category)
        return [
            "id": spec.id,
            "displayName": spec.displayName,
            "category": spec.category.rawValue,
            "family": spec.variantFamily ?? NSNull(),
            "huggingFaceRepo": spec.huggingFaceRepo,
            "files": spec.files.map { file in
                [
                    "remotePath": file.remotePath,
                    "localName": file.localName,
                    "sizeBytes": file.sizeBytes,
                    "sha256": file.sha256 ?? NSNull()
                ] as [String: Any]
            },
            "totalSizeBytes": spec.totalSizeBytes,
            "totalSizeFormatted": spec.totalSizeFormatted,
            "requiresAuth": spec.requiresAuth,
            "isRunnable": spec.isRunnable,
            "runtimeNote": spec.runtimeNote ?? NSNull(),
            "status": await registry.status(for: spec.id).mcpDescription,
            "isActive": activeID == spec.id
        ]
    }

    private func adapterResource(id: String) async throws -> [String: Any] {
        guard let adapter = await loadAdapters().first(where: { $0.id == id }) else {
            throw AnvilCommandFacadeError.resourceNotFound(resourceURI(kind: "adapter", identifier: id))
        }

        return [
            "id": adapter.id,
            "name": adapter.name,
            "baseModelID": adapter.baseModelID,
            "baseModelCategory": adapter.baseModelCategory.rawValue,
            "rank": adapter.config.rank,
            "alpha": adapter.config.alpha,
            "dropout": adapter.config.dropout,
            "targetLayers": adapter.config.targetLayers.map(\.rawValue).sorted(),
            "injectionPatterns": adapter.config.injectionPatterns,
            "learningRate": adapter.config.learningRate,
            "maxEpochs": adapter.config.maxEpochs,
            "sampleCount": adapter.sampleCount,
            "finalLoss": adapter.finalLoss.map { Double($0) } ?? NSNull(),
            "createdAt": dateFormatter.string(from: adapter.createdAt),
            "source": adapter.source.rawValue,
            "remoteRepoURL": adapter.remoteRepoURL?.absoluteString ?? NSNull(),
            "hasCheckpoint": AdapterRegistry.hasCheckpoint(adapterID: adapter.id)
        ]
    }

    private func audioResource(identifier: String) throws -> [String: Any] {
        let entry = try resolveHistoryEntry(identifier)
        let audioURL = historyStore.fileURL(for: entry)
        return [
            "id": entry.id.uuidString,
            "fileName": entry.fileName,
            "audioFilePath": audioURL.path(percentEncoded: false),
            "exists": FileManager.default.fileExists(atPath: audioURL.path(percentEncoded: false)),
            "duration": entry.duration,
            "createdAt": dateFormatter.string(from: entry.createdAt),
            "pipeline": entry.pipeline ?? NSNull(),
            "modelID": entry.modelID ?? NSNull(),
            "modelName": entry.modelName ?? NSNull(),
            "traceFileName": entry.traceFileName ?? NSNull(),
            "hasTrace": historyStore.traceURL(for: entry) != nil,
            "isPreferred": entry.isPreferred ?? false,
            "listeningNote": entry.listeningNote ?? NSNull()
        ]
    }

    private func resolveHistoryEntry(_ identifier: String) throws -> HistoryEntry {
        if let entryID = UUID(uuidString: identifier),
           let entry = historyStore.entries.first(where: { $0.id == entryID }) {
            return entry
        }

        if let entry = historyStore.entries.first(where: { entry in
            entry.traceFileName == identifier
                || HistoryEntry.traceFileName(forAudioFileName: entry.fileName) == identifier
                || entry.fileName == identifier
        }) {
            return entry
        }

        throw AnvilCommandFacadeError.historyEntryNotFound(identifier)
    }

    private func resolveTrace(arguments: [String: Any]) throws -> GenerationTrace {
        let id = stringValue(arguments["id"])
            ?? stringValue(arguments["historyEntryID"])
            ?? stringValue(arguments["traceFileName"])
            ?? stringValue(arguments["audioFileName"])

        guard let id, !id.isEmpty else {
            throw AnvilCommandFacadeError.missingTraceIdentifier
        }

        if let entryID = UUID(uuidString: id),
           let entry = historyStore.entries.first(where: { $0.id == entryID }) {
            return try historyStore.loadTrace(for: entry)
        }

        if let entry = historyStore.entries.first(where: { entry in
            entry.traceFileName == id
                || HistoryEntry.traceFileName(forAudioFileName: entry.fileName) == id
                || entry.fileName == id
        }) {
            return try historyStore.loadTrace(for: entry)
        }

        if id.hasSuffix(".trace.json") || id.hasSuffix(".failed.trace.json") {
            return try historyStore.loadTrace(traceFileName: id)
        }

        throw AnvilCommandFacadeError.traceNotFound(id)
    }

    private func historySummary(_ entry: HistoryEntry) -> [String: Any] {
        [
            "id": entry.id.uuidString,
            "createdAt": dateFormatter.string(from: entry.createdAt),
            "prompt": entry.prompt,
            "promptSummary": compact(entry.prompt, limit: 160),
            "duration": entry.duration,
            "seed": entry.seed,
            "method": entry.method,
            "guidanceScale": entry.guidanceScale,
            "precision": entry.precision,
            "fileName": entry.fileName,
            "audioFilePath": historyStore.fileURL(for: entry).path(percentEncoded: false),
            "traceFileName": entry.traceFileName ?? NSNull(),
            "hasTrace": historyStore.traceURL(for: entry) != nil,
            "pipeline": entry.pipeline ?? NSNull(),
            "modelID": entry.modelID ?? NSNull(),
            "modelName": entry.modelName ?? NSNull(),
            "adapterCount": entry.adapters?.count ?? 0,
            "isPreferred": entry.isPreferred ?? false,
            "listeningNote": entry.listeningNote ?? NSNull()
        ]
    }

    private func tracePayload(_ trace: GenerationTrace) throws -> [String: Any] {
        var payload = try encodedJSONObject(trace)
        payload["audioFilePath"] = historyStore.fileURL(forAudioFileName: trace.audioFileName).path(percentEncoded: false)
        return payload
    }

    private func traceSummary(_ trace: GenerationTrace) -> [String: Any] {
        [
            "historyEntryID": trace.historyEntryID.uuidString,
            "createdAt": dateFormatter.string(from: trace.createdAt),
            "pipeline": trace.pipeline,
            "model": trace.primaryModel?.displayName ?? trace.primaryModel?.id ?? "None",
            "audioFileName": trace.audioFileName,
            "audioFilePath": historyStore.fileURL(forAudioFileName: trace.audioFileName).path(percentEncoded: false),
            "traceFileName": trace.traceFileName,
            "seed": trace.request.seed.map(String.init) ?? "Random",
            "guidanceScale": trace.request.guidanceScale,
            "guidancePreset": trace.request.guidancePresetName,
            "duration": trace.request.duration,
            "absMean": trace.audio.absMean,
            "absMax": trace.audio.absMax,
            "isPreferred": trace.isPreferred ?? false,
            "listeningNote": trace.listeningNote ?? NSNull()
        ]
    }

    private func reviewPayload(_ trace: GenerationTrace) -> [String: Any] {
        [
            "historyEntryID": trace.historyEntryID.uuidString,
            "traceFileName": trace.traceFileName,
            "audioFileName": trace.audioFileName,
            "audioFilePath": historyStore.fileURL(forAudioFileName: trace.audioFileName).path(percentEncoded: false),
            "isPreferred": trace.isPreferred ?? false,
            "listeningNote": trace.listeningNote ?? NSNull()
        ]
    }

    private func memoryReleasePayload(_ summary: MemoryReleaseSummary) -> [String: Any] {
        [
            "released": true,
            "before": [
                "activeMemoryBytes": summary.beforeActiveMemory,
                "cacheMemoryBytes": summary.beforeCacheMemory,
                "peakMemoryBytes": summary.beforePeakMemory
            ],
            "after": [
                "activeMemoryBytes": summary.afterActiveMemory,
                "cacheMemoryBytes": summary.afterCacheMemory,
                "peakMemoryBytes": summary.afterPeakMemory
            ],
            "delta": [
                "releasedActiveMemoryBytes": summary.releasedActiveMemory,
                "releasedCacheMemoryBytes": summary.releasedCacheMemory
            ]
        ]
    }

    private func adapterSelectionPayload(_ selection: [AdapterSelection]) -> [String: Any] {
        [
            "applied": true,
            "cleared": selection.isEmpty,
            "count": selection.count,
            "primary": selection.first.map(adapterSelectionEntryPayload) ?? NSNull(),
            "adapters": selection.map(adapterSelectionEntryPayload)
        ]
    }

    private func adapterSelectionEntryPayload(_ entry: AdapterSelection) -> [String: Any] {
        [
            "id": entry.adapter.id,
            "name": entry.adapter.name,
            "strength": entry.strength,
            "baseModelID": entry.adapter.baseModelID,
            "baseModelCategory": entry.adapter.baseModelCategory.rawValue,
            "rank": entry.adapter.config.rank,
            "alpha": entry.adapter.config.alpha
        ]
    }

    private func persistAdapterSelection(_ selection: [AdapterSelection]) {
        let defaults = UserDefaults.standard
        if let primary = selection.first {
            defaults.set(primary.adapter.id, forKey: "selectedAdapterID")
            defaults.set(Double(primary.strength), forKey: "selectedAdapterStrength")
        } else {
            defaults.removeObject(forKey: "selectedAdapterID")
            defaults.set(1.0, forKey: "selectedAdapterStrength")
        }
    }

    private func notifyAdapterSelectionChanged(_ selection: [AdapterSelection]) {
        NotificationCenter.default.post(
            name: .mcpAdapterSelectionDidChange,
            object: nil,
            userInfo: [
                "adapters": selection.map(adapterSelectionEntryPayload)
            ]
        )
    }

    private func rowPayload(_ row: TraceComparisonRow) -> [String: Any] {
        [
            "label": row.label,
            "left": row.left,
            "right": row.right,
            "isDifferent": row.isDifferent
        ]
    }

    private func encodedJSONObject<T: Encodable>(_ value: T) throws -> [String: Any] {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        let data = try encoder.encode(value)
        guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw AnvilCommandFacadeError.encodingFailed
        }
        return object
    }

    private func stringValue(_ value: Any?) -> String? {
        switch value {
        case is NSNull:
            return nil
        case let value as String:
            return value
        case let value as UUID:
            return value.uuidString
        case let value as CustomStringConvertible:
            return value.description
        default:
            return nil
        }
    }

    private func boolValue(_ value: Any?) -> Bool? {
        switch value {
        case let value as Bool:
            return value
        case let value as NSNumber:
            return value.boolValue
        case let value as String:
            switch value.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() {
            case "true", "yes", "1":
                return true
            case "false", "no", "0":
                return false
            default:
                return nil
            }
        default:
            return nil
        }
    }

    private func hasArgument(_ key: String, in arguments: [String: Any]) -> Bool {
        arguments.keys.contains(key)
    }

    private func doubleValue(_ value: Any?) -> Double? {
        switch value {
        case let value as Double:
            return value
        case let value as Float:
            return Double(value)
        case let value as Int:
            return Double(value)
        case let value as NSNumber:
            return value.doubleValue
        case let value as String:
            return Double(value.trimmingCharacters(in: .whitespacesAndNewlines))
        default:
            return nil
        }
    }

    private func requiredString(_ key: String, in arguments: [String: Any]) throws -> String {
        let value = stringValue(arguments[key])?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        guard !value.isEmpty else {
            throw AnvilCommandFacadeError.missingRequiredArgument(key)
        }
        return value
    }

    private func requiredDuration(in arguments: [String: Any]) throws -> Double {
        guard let duration = doubleValue(arguments["duration"] ?? arguments["durationSeconds"]),
              duration > 0 else {
            throw AnvilCommandFacadeError.invalidDuration
        }
        return duration
    }

    private func generationMode(from value: Any?) -> GenerationMode? {
        guard let rawValue = stringValue(value) else { return nil }
        if let mode = GenerationMode(rawValue: rawValue) {
            return mode
        }
        return GenerationMode(fromClassification: rawValue)
    }

    private func normalizedText(_ text: String) -> String {
        text.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func lyricLineCount(_ lyrics: String) -> Int {
        lyrics
            .components(separatedBy: .newlines)
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty && !$0.hasPrefix("[") }
            .count
    }

    private func compact(_ text: String, limit: Int) -> String {
        let oneLine = text
            .replacingOccurrences(of: "\n", with: " ")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard oneLine.count > limit else { return oneLine }
        return String(oneLine.prefix(max(0, limit - 3))) + "..."
    }

    private func resource(
        uri: String,
        name: String,
        title: String,
        description: String
    ) -> [String: Any] {
        [
            "uri": uri,
            "name": name,
            "title": title,
            "description": description,
            "mimeType": "application/json"
        ]
    }

    private func resourceURI(kind: String, identifier: String? = nil) -> String {
        guard let identifier else { return "anvil://\(kind)" }
        let encoded = identifier.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? identifier
        return "anvil://\(kind)/\(encoded)"
    }
}

private enum AnvilCommandFacadeError: LocalizedError {
    case missingTraceIdentifier
    case traceNotFound(String)
    case historyEntryNotFound(String)
    case invalidResourceURI(String)
    case resourceNotFound(String)
    case adapterNotFound(String)
    case incompatibleAdapter(String, String)
    case invalidAdapterStack
    case invalidBoolean(String)
    case invalidStrength
    case tooManyAdapters(Int)
    case missingRequiredArgument(String)
    case invalidDuration
    case confirmationRequired(String)
    case generationServiceUnavailable
    case encodingFailed

    var errorDescription: String? {
        switch self {
        case .missingTraceIdentifier:
            return "Provide a history id, trace filename, or audio filename."
        case .traceNotFound(let identifier):
            return "No generation trace matched '\(identifier)'."
        case .historyEntryNotFound(let identifier):
            return "No history entry matched '\(identifier)'."
        case .invalidResourceURI(let uri):
            return "Invalid Anvil resource URI: \(uri)."
        case .resourceNotFound(let uri):
            return "No Anvil resource matched '\(uri)'."
        case .adapterNotFound(let identifier):
            return "No LoRA adapter matched '\(identifier)'."
        case .incompatibleAdapter(let name, let category):
            return "'\(name)' is a \(category) adapter and cannot be applied to ACE-Step generation."
        case .invalidAdapterStack:
            return "'adapters' must be an array of adapter objects."
        case .invalidBoolean(let key):
            return "'\(key)' must be true or false."
        case .invalidStrength:
            return "Adapter strength must be a number from 0.0 to 1.0."
        case .tooManyAdapters(let maxCount):
            return "Adapter stack can contain at most \(maxCount) adapters."
        case .missingRequiredArgument(let key):
            return "Missing required argument '\(key)'."
        case .invalidDuration:
            return "Provide a positive duration in seconds."
        case .confirmationRequired(let action):
            return "Refusing to run '\(action)' without confirm: true."
        case .generationServiceUnavailable:
            return "Generation service is not ready yet."
        case .encodingFailed:
            return "Could not encode the trace as JSON."
        }
    }
}

private struct AdapterSelection {
    let adapter: LoRAAdapter
    let strength: Float
}

private struct AnvilResourceURI {
    let kind: String
    let identifier: String?

    init?(_ rawValue: String) {
        guard let components = URLComponents(string: rawValue),
              components.scheme == "anvil",
              let host = components.host,
              !host.isEmpty else {
            return nil
        }

        let path = components.percentEncodedPath
            .split(separator: "/", omittingEmptySubsequences: true)
            .map(String.init)

        self.kind = host
        if let first = path.first {
            self.identifier = first.removingPercentEncoding ?? first
        } else {
            self.identifier = nil
        }
    }
}

private extension ModelStatus {
    var mcpDescription: [String: Any] {
        switch self {
        case .notDownloaded:
            return ["state": "notDownloaded"]
        case .downloading(let progress):
            return ["state": "downloading", "progress": progress]
        case .verifying:
            return ["state": "verifying"]
        case .ready:
            return ["state": "ready"]
        case .failed(let message):
            return ["state": "failed", "message": message]
        }
    }
}
