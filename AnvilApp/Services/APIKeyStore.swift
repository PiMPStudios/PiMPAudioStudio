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
    ]

    private static let serviceName = "com.pimpstudios.anvil.apikeys"

    init() {
        refreshAvailableProviders()
    }

    func apiKey(for provider: String) -> String? {
        if let keychainKey = readFromKeychain(account: provider) {
            return keychainKey
        }
        if let envVar = Self.providers.first(where: { $0.id == provider })?.envVar {
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

    func keySource(for provider: String) -> KeySource {
        if readFromKeychain(account: provider) != nil {
            return .keychain
        }
        if let envVar = Self.providers.first(where: { $0.id == provider })?.envVar,
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
        for provider in Self.providers {
            if apiKey(for: provider.id) != nil {
                available.insert(provider.id)
            }
        }
        availableProviders = available
    }

    private func readFromKeychain(account: String) -> String? {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: Self.serviceName,
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
        deleteFromKeychain(account: account)
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
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: Self.serviceName,
            kSecAttrAccount as String: account,
        ]
        SecItemDelete(query as CFDictionary)
    }
}
