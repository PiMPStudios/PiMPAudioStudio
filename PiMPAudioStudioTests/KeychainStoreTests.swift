import Foundation
import Security
import Testing
@testable import PiMPAudioStudio

@Test func keychainAddQueryPinsThisDeviceOnlyAccessibility() {
    let data = Data("secret".utf8)
    let query = KeychainItemQuery.add(
        account: "huggingface",
        serviceName: AppRuntimeIdentity.keychainServiceName,
        data: data
    )

    #expect(query[kSecClass as String] as? String == kSecClassGenericPassword as String)
    #expect(query[kSecAttrService as String] as? String == AppRuntimeIdentity.keychainServiceName)
    #expect(query[kSecAttrAccount as String] as? String == "huggingface")
    #expect(query[kSecValueData as String] as? Data == data)
    #expect(
        query[kSecAttrAccessible as String] as? String
            == kSecAttrAccessibleWhenUnlockedThisDeviceOnly as String
    )
}

@Test func keychainUpdateAttributesPinThisDeviceOnlyAccessibility() {
    let data = Data("rotated".utf8)
    let attributes = KeychainItemQuery.updateAttributes(data: data)

    #expect(attributes[kSecValueData as String] as? Data == data)
    #expect(
        attributes[kSecAttrAccessible as String] as? String
            == kSecAttrAccessibleWhenUnlockedThisDeviceOnly as String
    )
}
