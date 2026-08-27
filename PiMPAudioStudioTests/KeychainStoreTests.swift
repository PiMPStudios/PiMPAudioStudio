import Foundation
@testable import PiMPAudioStudio
import Security
import Testing

@Test func keychainAddQueryPinsThisDeviceOnlyAccessibility() {
    let data = Data("secret".utf8)
    let query = KeychainItemQuery.add(
        account: "huggingface",
        serviceName: AppRuntimeIdentity.keychainServiceName,
        data: data
    )

    #expect(cfString(query[kSecClass as String]) == (kSecClassGenericPassword as String))
    #expect(cfString(query[kSecAttrService as String]) == AppRuntimeIdentity.keychainServiceName)
    #expect(cfString(query[kSecAttrAccount as String]) == "huggingface")
    #expect(query[kSecValueData as String] as? Data == data)
    #expect(cfString(query[kSecAttrAccessible as String]) == (kSecAttrAccessibleWhenUnlockedThisDeviceOnly as String))
}

@Test func keychainUpdateAttributesPinThisDeviceOnlyAccessibility() {
    let data = Data("rotated".utf8)
    let attributes = KeychainItemQuery.updateAttributes(data: data)

    #expect(attributes[kSecValueData as String] as? Data == data)
    #expect(cfString(attributes[kSecAttrAccessible as String]) ==
        (kSecAttrAccessibleWhenUnlockedThisDeviceOnly as String))
}

private func cfString(_ value: Any?) -> String? {
    value as? String ?? (value as? NSString).map { $0 as String }
}
