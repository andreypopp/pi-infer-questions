import Foundation

struct Settings: Decodable {
    struct Trim: Decodable {
        let closing: [String: String]
        let opening: [String: String]
        let optionTrailing: String
    }

    struct Pattern: Decodable {
        let pattern: String
        let flags: String
    }

    struct Limits: Decodable {
        let rows: Int
        let tokens: Int
    }

    let padTokenId: Int
    let tags: [String: [String]]
    let maxTokens: Int
    let trim: Trim
    let decision: Pattern
    let batch: Limits?

    static func load(_ directory: URL) throws -> Settings {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return try decoder.decode(Settings.self, from: Data(contentsOf: directory.appendingPathComponent("settings.json")))
    }
}
