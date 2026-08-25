import Foundation
import Security
import SwiftUI

@MainActor
final class APIKeyStore: ObservableObject {
    @Published var availableProviders: Set<String> = []

    static let providers: [(id: String, displayName: String, envVar: String)] = [
        ("anthropic", "Claude (Anthropic)", "ANTHROPIC_API_KEY"),
        ("openai", "OpenAI", "OPENAI_API_KEY"),
        ("google", "Gemini (Google)", "GOOGLE_API_KEY"),
        ("xai", "Grok (xAI)", "XAI_API_KEY"),
        ("huggingface", "HuggingFace", "HUGGINGFACE_API_KEY"),
    ]

    static let huggingFaceProviderID = "huggingface"
    static let runPodProviderID = "runpod"
    static let runPodProvider: (id: String, displayName: String, envVar: String) = (
        runPodProviderID,
        "RunPod",
        "RUNPOD_API_KEY"
    )
    static let apiKeyProviders: [(id: String, displayName: String, envVar: String)] = providers + [
        runPodProvider,
    ]

    private static let serviceName = AppRuntimeIdentity.keychainServiceName
    private static var shouldReadKeychain: Bool {
        ProcessInfo.processInfo.environment["XCTestConfigurationFilePath"] == nil
    }

    init() {
        refreshAvailableProviders()
    }

    func apiKey(for provider: String) -> String? {
        if Self.shouldReadKeychain,
           let keychainKey = readFromKeychain(account: provider) {
            return keychainKey
        }
        if let envVar = Self.apiKeyProviders.first(where: { $0.id == provider })?.envVar {
            return ProcessInfo.processInfo.environment[envVar]
        }
        return nil
    }

    func setAPIKey(_ key: String, for provider: String) throws {
        try KeychainStringStore.write(account: provider, value: key, serviceName: Self.serviceName)
        refreshAvailableProviders()
    }

    func removeAPIKey(for provider: String) {
        KeychainStringStore.delete(account: provider, serviceName: Self.serviceName)
        refreshAvailableProviders()
    }

    func hasKey(for provider: String) -> Bool {
        apiKey(for: provider) != nil
    }

    func huggingFaceToken() -> String? {
        apiKey(for: Self.huggingFaceProviderID)
    }

    func runPodToken() -> String? {
        apiKey(for: Self.runPodProviderID)
    }

    func keySource(for provider: String) -> KeySource {
        if Self.shouldReadKeychain,
           readFromKeychain(account: provider) != nil {
            return .keychain
        }
        if let envVar = Self.apiKeyProviders.first(where: { $0.id == provider })?.envVar,
           ProcessInfo.processInfo.environment[envVar] != nil {
            return .environment
        }
        return .none
    }

    enum KeySource {
        case keychain
        case environment
        case none
    }

    private func refreshAvailableProviders() {
        var available = Set<String>()
        for provider in Self.apiKeyProviders {
            if apiKey(for: provider.id) != nil {
                available.insert(provider.id)
            }
        }
        availableProviders = available
    }

    private func readFromKeychain(account: String) -> String? {
        KeychainStringStore.read(account: account, serviceName: Self.serviceName)
    }
}

enum KeychainStoreError: Error, LocalizedError {
    case encodingFailed
    case unhandled(OSStatus)

    var errorDescription: String? {
        switch self {
        case .encodingFailed:
            return "Could not encode the secret for Keychain storage."
        case .unhandled(let status):
            let reason = SecCopyErrorMessageString(status, nil) as String? ?? "unknown error"
            return "Keychain save failed (\(status)): \(reason)"
        }
    }
}

enum KeychainItemQuery {
    static let accessibility = kSecAttrAccessibleWhenUnlockedThisDeviceOnly

    static func identity(account: String, serviceName: String) -> [String: Any] {
        [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: serviceName,
            kSecAttrAccount as String: account,
        ]
    }

    static func add(account: String, serviceName: String, data: Data) -> [String: Any] {
        var query = identity(account: account, serviceName: serviceName)
        query[kSecValueData as String] = data
        query[kSecAttrAccessible as String] = accessibility
        return query
    }

    static func updateAttributes(data: Data) -> [String: Any] {
        [
            kSecValueData as String: data,
            kSecAttrAccessible as String: accessibility,
        ]
    }
}

enum KeychainStringStore {
    static var isAvailable: Bool {
        ProcessInfo.processInfo.environment["XCTestConfigurationFilePath"] == nil
    }

    static func read(
        account: String,
        serviceName: String = AppRuntimeIdentity.keychainServiceName
    ) -> String? {
        var query = KeychainItemQuery.identity(account: account, serviceName: serviceName)
        query[kSecReturnData as String] = true
        query[kSecMatchLimit as String] = kSecMatchLimitOne
        var result: AnyObject?
        let status = SecItemCopyMatching(query as CFDictionary, &result)
        guard status == errSecSuccess, let data = result as? Data else { return nil }
        return String(data: data, encoding: .utf8)
    }

    static func write(
        account: String,
        value: String,
        serviceName: String = AppRuntimeIdentity.keychainServiceName
    ) throws {
        guard let data = value.data(using: .utf8) else {
            throw KeychainStoreError.encodingFailed
        }
        let identity = KeychainItemQuery.identity(account: account, serviceName: serviceName)
        let updateStatus = SecItemUpdate(
            identity as CFDictionary,
            KeychainItemQuery.updateAttributes(data: data) as CFDictionary
        )
        if updateStatus == errSecSuccess {
            return
        }
        guard updateStatus == errSecItemNotFound else {
            throw KeychainStoreError.unhandled(updateStatus)
        }
        let addStatus = SecItemAdd(
            KeychainItemQuery.add(account: account, serviceName: serviceName, data: data) as CFDictionary,
            nil
        )
        guard addStatus == errSecSuccess else {
            throw KeychainStoreError.unhandled(addStatus)
        }
    }

    static func delete(
        account: String,
        serviceName: String = AppRuntimeIdentity.keychainServiceName
    ) {
        let query = KeychainItemQuery.identity(account: account, serviceName: serviceName)
        SecItemDelete(query as CFDictionary)
    }
}

struct MCPTokenResolution: Equatable {
    let token: String
    let writeToKeychain: String?
    let clearDefaults: Bool

    static func resolve(
        keychainValue: String?,
        defaultsValue: String?,
        generated: String
    ) -> MCPTokenResolution {
        if let keychainValue, !keychainValue.isEmpty {
            return MCPTokenResolution(
                token: keychainValue,
                writeToKeychain: nil,
                clearDefaults: !(defaultsValue ?? "").isEmpty
            )
        }
        if let defaultsValue, !defaultsValue.isEmpty {
            return MCPTokenResolution(
                token: defaultsValue,
                writeToKeychain: defaultsValue,
                clearDefaults: true
            )
        }
        return MCPTokenResolution(
            token: generated,
            writeToKeychain: generated,
            clearDefaults: false
        )
    }
}
