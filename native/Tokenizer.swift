import Foundation
struct Tokenizer: Sendable {
    struct Encoding: Equatable, Sendable {
        let ids: [Int]
        let offsets: [Range<Int>]
    }

    private struct Pair: Hashable {
        let left: String
        let right: String
    }

    private let vocabulary: [String: Int]
    private let ranks: [Pair: Int]
    private let added: [Unicode.Scalar: [(content: Scalars, id: Int, stripsLeft: Bool)]]
    private let byteCharacters: [String]
    private let unknown: Int
    private let classifier: Int
    private let separator: Int
    let maxTokens: Int

    init(json: Data, maxTokens: Int) throws {
        guard let root = try JSONSerialization.jsonObject(with: json) as? [String: Any],
              let model = root["model"] as? [String: Any],
              let vocabulary = model["vocab"] as? [String: Int],
              let merges = model["merges"] as? [[String]],
              let tokens = root["added_tokens"] as? [[String: Any]] else {
            throw CocoaError(.coderReadCorrupt)
        }
        self.vocabulary = vocabulary
        var ranks: [Pair: Int] = [:]
        for (rank, merge) in merges.enumerated() { ranks[Pair(left: merge[0], right: merge[1])] = rank }
        self.ranks = ranks
        var added: [Unicode.Scalar: [(content: Scalars, id: Int, stripsLeft: Bool)]] = [:]
        var special: [String: Int] = [:]
        for token in tokens {
            guard let content = token["content"] as? String, let id = token["id"] as? Int,
                  let first = content.unicodeScalars.first else { continue }
            added[first, default: []].append((Array(content.unicodeScalars), id, token["lstrip"] as? Bool ?? false))
            special[content] = id
        }
        self.added = added.mapValues { $0.sorted { $0.content.count > $1.content.count } }
        let printable = Array(33...126) + Array(161...172) + Array(174...255)
        var characters = [String](repeating: "", count: 256)
        var extra = 0
        for byte in 0..<256 {
            if printable.contains(byte) {
                characters[byte] = String(Unicode.Scalar(UInt8(byte)))
            } else {
                characters[byte] = String(Unicode.Scalar(UInt32(256 + extra))!)
                extra += 1
            }
        }
        byteCharacters = characters
        guard let unknown = special["[UNK]"], let classifier = special["[CLS]"], let separator = special["[SEP]"] else {
            throw CocoaError(.coderReadCorrupt)
        }
        self.unknown = unknown
        self.classifier = classifier
        self.separator = separator
        self.maxTokens = maxTokens
    }

    func encode(_ text: Scalars) -> Encoding {
        let (normalised, alignment) = nfc(text)
        var tokens: [(id: Int, range: Range<Int>)] = []
        var start = 0
        var index = 0
        func flush(_ end: Int, stripping: Bool = false) {
            var end = end
            while stripping, end > start, normalised[end - 1].properties.isWhitespace { end -= 1 }
            guard start < end else { return }
            for piece in pieces(normalised, start..<end) {
                tokens += merge(normalised[piece], alignment: Array(alignment[piece]))
            }
        }
        while index < normalised.count {
            if let match = added[normalised[index]]?.first(where: { normalised[index...].starts(with: $0.content) }) {
                flush(index, stripping: match.stripsLeft)
                tokens.append((match.id, alignment[index].lowerBound..<alignment[index + match.content.count - 1].upperBound))
                index += match.content.count
                start = index
            } else {
                index += 1
            }
        }
        flush(normalised.count)
        var ids = [classifier], offsets = [0..<0]
        for token in tokens.prefix(maxTokens - 2) {
            var range = token.range
            while range.lowerBound < range.upperBound, isSpace(text[range.lowerBound]) {
                range = (range.lowerBound + 1)..<range.upperBound
            }
            ids.append(token.id)
            offsets.append(range.isEmpty ? 0..<0 : range)
        }
        return Encoding(ids: ids + [separator], offsets: offsets + [0..<0])
    }

    private func nfc(_ text: Scalars) -> (Scalars, [Range<Int>]) {
        let string = String(String.UnicodeScalarView(text))
        if string.precomposedStringWithCanonicalMapping.unicodeScalars.elementsEqual(text) {
            return (text, text.indices.map { $0..<($0 + 1) })
        }
        var result: Scalars = []
        var alignment: [Range<Int>] = []
        var position = 0
        for cluster in string {
            let source = Array(cluster.unicodeScalars)
            var consumed = 0
            for composed in String(cluster).precomposedStringWithCanonicalMapping.unicodeScalars {
                var length = 1
                while consumed + length < source.count,
                      Array(String(String.UnicodeScalarView(source[consumed..<(consumed + length)]))
                          .precomposedStringWithCanonicalMapping.unicodeScalars) != [composed] {
                    length += 1
                }
                result.append(composed)
                alignment.append((position + min(consumed, source.count - 1))..<(position + min(consumed, source.count - 1) + 1))
                consumed += length
            }
            position += source.count
        }
        return (result, alignment)
    }

    private func pieces(_ text: Scalars, _ range: Range<Int>) -> [Range<Int>] {
        func kind(_ scalar: Unicode.Scalar) -> Int {
            if scalar.properties.isWhitespace { return 0 }
            switch scalar.properties.generalCategory {
            case .uppercaseLetter, .lowercaseLetter, .titlecaseLetter, .modifierLetter, .otherLetter: return 1
            case .decimalNumber, .letterNumber, .otherNumber: return 2
            default: return 3
            }
        }
        var found: [Range<Int>] = []
        var index = range.lowerBound
        let end = range.upperBound
        while index < end {
            if text[index] == "'" {
                let rest = text[(index + 1)..<end]
                if let length = ["s", "t", "re", "ve", "m", "ll", "d"].first(where: { rest.starts(with: $0.unicodeScalars) })?.count {
                    found.append(index..<(index + 1 + length))
                    index += 1 + length
                    continue
                }
            }
            let first = text[index] == " " ? index + 1 : index
            if first < end, kind(text[first]) != 0 {
                let type = kind(text[first])
                var stop = first
                while stop < end, kind(text[stop]) == type { stop += 1 }
                found.append(index..<stop)
                index = stop
                continue
            }
            var stop = index
            while stop < end, kind(text[stop]) == 0 { stop += 1 }
            if stop < end, stop - index > 1 { stop -= 1 }
            found.append(index..<stop)
            index = stop
        }
        return found
    }

    private func merge(_ piece: ArraySlice<Unicode.Scalar>, alignment: [Range<Int>]) -> [(id: Int, range: Range<Int>)] {
        var symbols: [(text: String, range: Range<Int>)] = []
        for (scalar, range) in zip(piece, alignment) {
            for byte in String(scalar).utf8 { symbols.append((byteCharacters[Int(byte)], range)) }
        }
        while symbols.count > 1 {
            var best: (rank: Int, index: Int)?
            for index in 0..<(symbols.count - 1) {
                if let rank = ranks[Pair(left: symbols[index].text, right: symbols[index + 1].text)],
                   best == nil || rank < best!.rank {
                    best = (rank, index)
                }
            }
            guard let (_, index) = best else { break }
            symbols[index] = (symbols[index].text + symbols[index + 1].text,
                              symbols[index].range.lowerBound..<symbols[index + 1].range.upperBound)
            symbols.remove(at: index + 1)
        }
        return symbols.map { (vocabulary[$0.text] ?? unknown, $0.range) }
    }
}
