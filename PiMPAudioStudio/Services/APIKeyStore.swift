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

    func setAPIKey(_ key: String, for provider: String) {
        writeToKeychain(account: provider, value: key)
        refreshAvailableProviders()
    }

    func removeAPIKey(for provider: String) {
        deleteFromKeychain(account: provider)
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
        readFromKeychain(account: account, serviceName: Self.serviceName)
    }

    private func readFromKeychain(account: String, serviceName: String) -> String? {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: serviceName,
            kSecAttrAccount as String: account,
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne,
        ]
        var result: AnyObject?
        let status = SecItemCopyMatching(query as CFDictionary, &result)
        guard status == errSecSuccess, let data = result as? Data else { return nil }
        return String(data: data, encoding: .utf8)
    }

    private func writeToKeychain(account: String, value: String) {
        deleteFromKeychain(account: account, serviceName: Self.serviceName)
        guard let data = value.data(using: .utf8) else { return }
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: Self.serviceName,
            kSecAttrAccount as String: account,
            kSecValueData as String: data,
        ]
        SecItemAdd(query as CFDictionary, nil)
    }

    private func deleteFromKeychain(account: String) {
        deleteFromKeychain(account: account, serviceName: Self.serviceName)
    }

    private func deleteFromKeychain(account: String, serviceName: String) {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: serviceName,
            kSecAttrAccount as String: account,
        ]
        SecItemDelete(query as CFDictionary)
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
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: serviceName,
            kSecAttrAccount as String: account,
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne,
        ]
        var result: AnyObject?
        let status = SecItemCopyMatching(query as CFDictionary, &result)
        guard status == errSecSuccess, let data = result as? Data else { return nil }
        return String(data: data, encoding: .utf8)
    }

    static func write(
        account: String,
        value: String,
        serviceName: String = AppRuntimeIdentity.keychainServiceName
    ) {
        delete(account: account, serviceName: serviceName)
        guard let data = value.data(using: .utf8) else { return }
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: serviceName,
            kSecAttrAccount as String: account,
            kSecValueData as String: data,
        ]
        SecItemAdd(query as CFDictionary, nil)
    }

    static func delete(
        account: String,
        serviceName: String = AppRuntimeIdentity.keychainServiceName
    ) {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: serviceName,
            kSecAttrAccount as String: account,
        ]
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
