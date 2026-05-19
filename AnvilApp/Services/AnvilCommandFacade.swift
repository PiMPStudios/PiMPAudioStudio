import Foundation
import AnvilCore

@MainActor
final class AnvilCommandFacade {
    private let registry: ModelRegistry
    private var historyStore: HistoryStore
    private let dateFormatter = ISO8601DateFormatter()

    init(registry: ModelRegistry, historyStore: HistoryStore) {
        self.registry = registry
        self.historyStore = historyStore
    }

    func configure(historyStore: HistoryStore) {
        self.historyStore = historyStore
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

    private func compact(_ text: String, limit: Int) -> String {
        let oneLine = text
            .replacingOccurrences(of: "\n", with: " ")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard oneLine.count > limit else { return oneLine }
        return String(oneLine.prefix(max(0, limit - 3))) + "..."
    }
}

private enum AnvilCommandFacadeError: LocalizedError {
    case missingTraceIdentifier
    case traceNotFound(String)
    case encodingFailed

    var errorDescription: String? {
        switch self {
        case .missingTraceIdentifier:
            return "Provide a history id, trace filename, or audio filename."
        case .traceNotFound(let identifier):
            return "No generation trace matched '\(identifier)'."
        case .encodingFailed:
            return "Could not encode the trace as JSON."
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
