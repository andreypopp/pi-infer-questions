import Foundation

struct Trace {
    let tokens: [Tokenizer.Encoding]
    let packs: [[Int]]
}

struct Found {
    let unit: Int
    let span: Range<Int>
    let range: Range<Int>
    let kind: String
    var options: [Range<Int>] = []
    var optionSpans: [Range<Int>] = []
}

struct Analysis {
    let units: [Unit]
    let gated: [Int]
    let questionTrace: Trace
    let found: [Found]
    let choices: [Int]
    let optionTrace: Trace
}

private let kinds = ["yes_no", "open", "choice"]
private let tagNames = [
    "question": ["O"] + kinds.flatMap { ["B-\($0)", "I-\($0)"] },
    "option": ["O", "B", "I"],
]

final class Engine {
    private let tokenizer: Tokenizer
    private let decoding: Decoding
    private let trailing: Set<Unicode.Scalar>
    private let decision: NSRegularExpression
    private let models: Models

    init(directory: URL) throws {
        let settings = try Settings.load(directory)
        guard settings.tags == tagNames else { throw CocoaError(.coderReadCorrupt) }
        tokenizer = try Tokenizer(
            json: Data(contentsOf: directory.appendingPathComponent("tokenizer.json")), maxTokens: settings.maxTokens)
        func scalars(_ map: [String: String]) -> [Unicode.Scalar: Unicode.Scalar] {
            Dictionary(uniqueKeysWithValues: map.compactMap { key, value in
                guard let key = key.unicodeScalars.first, let value = value.unicodeScalars.first else { return nil }
                return (key, value)
            })
        }
        decoding = Decoding(
            closing: scalars(settings.trim.closing), opening: scalars(settings.trim.opening))
        trailing = Set(settings.trim.optionTrailing.unicodeScalars)
        decision = try NSRegularExpression(pattern: settings.decision.pattern, options: settings.decision.flags.contains("i") ? [.caseInsensitive] : [])
        models = try Models(directory: directory, settings: settings)
    }

    func analyse(_ markdown: Scalars) throws -> Analysis {
        let (normalised, headings) = normalise(markdown)
        let all = units(normalised, headings)
        let gated = all.indices.filter { number in
            let text = all[number].text
            if text.contains("?") { return true }
            return Line(Array(text.split(whereSeparator: isSpace).joined(separator: [" "]))).first(decision) != nil
        }
        let (tagged, questionTrace) = try tag("question", gated.map { all[$0].text }, trailing: [])
        let widen = widening(markdown)
        var found = zip(gated, tagged).flatMap { unit, spans in
            afterContext(spans, all[unit].context).filter { all[unit].text[$0.span].contains(where: isLetter) }.map {
                Found(unit: unit, span: $0.span, range: widen(all[unit].origin, $0.span), kind: kinds[$0.kind])
            }
        }
        let choices = found.indices.filter { found[$0].kind == "choice" }
        let (inner, optionTrace) = try tag("option", choices.map { Array(all[found[$0].unit].text[found[$0].span]) }, trailing: trailing)
        for (number, spans) in zip(choices, inner) {
            let question = found[number]
            found[number].optionSpans = spans.map(\.span)
            found[number].options = spans.map {
                widen(all[question.unit].origin, (question.span.lowerBound + $0.span.lowerBound)..<(question.span.lowerBound + $0.span.upperBound))
            }
        }
        return Analysis(units: all, gated: gated, questionTrace: questionTrace, found: found, choices: choices, optionTrace: optionTrace)
    }

    private func tag(_ task: String, _ texts: [Scalars], trailing: Set<Unicode.Scalar>) throws -> ([[(span: Range<Int>, kind: Int)]], Trace) {
        let encoded = texts.map { tokenizer.encode($0) }
        let (tags, packs) = try models.tags(task, encoded.map(\.ids))
        let out = texts.indices.map { spans(in: texts[$0], offsets: encoded[$0].offsets, tags: tags[$0], decoding, trailing: trailing) }
        return (out, Trace(tokens: encoded, packs: packs))
    }
}

private let letters: Set<Unicode.GeneralCategory> = [.uppercaseLetter, .lowercaseLetter, .titlecaseLetter, .modifierLetter, .otherLetter]

private func isLetter(_ scalar: Unicode.Scalar) -> Bool {
    letters.contains(scalar.properties.generalCategory)
}
