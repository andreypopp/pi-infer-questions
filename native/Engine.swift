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
    let options: [Range<Int>]
    let optionSpans: [Range<Int>]
}

struct Analysis {
    let units: [Unit]
    let gated: [Int]
    let questionTrace: Trace
    let found: [Found]
    let choices: [Int]
    let optionTrace: Trace
}

private let codeSpan = try! NSRegularExpression(pattern: #"(`+)[^`\n]+?\1"#)
private let kinds = ["yes_no", "open", "choice"]
private let tagNames = [
    "question": ["O"] + kinds.flatMap { ["B-\($0)", "I-\($0)"] },
    "option": ["O", "B", "I"],
]

final class Engine {
    private let tokenizer: Tokenizer
    private let settings: Settings
    private let decoding: Decoding
    private let trailing: Set<Unicode.Scalar>
    private let decision: NSRegularExpression
    private let models: Models

    init(directory: URL) throws {
        settings = try Settings.load(directory)
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
        decision = try NSRegularExpression(pattern: settings.decision.pattern, options: settings.decision.options)
        models = try Models(directory: directory, settings: settings)
    }

    func analyse(_ markdown: Scalars) throws -> Analysis {
        let (normalised, headings) = normalise(markdown)
        let all = units(normalised, headings)
        let gated = all.indices.filter { number in
            let text = all[number].text
            if text.contains("?") { return true }
            let line = String(String.UnicodeScalarView(text.split(whereSeparator: isSpace).joined(separator: [" "])))
            return decision.firstMatch(in: line, range: NSRange(location: 0, length: line.utf16.count)) != nil
        }
        let (tagged, questionTrace) = try tag("question", gated.map { all[$0].text }, trailing: [])
        let asked = zip(gated, tagged).flatMap { unit, spans in
            afterContext(spans, all[unit].context).map { (unit, $0.span, kinds[$0.kind]) }
        }
        let choices = asked.indices.filter { asked[$0].2 == "choice" }
        let (inner, optionTrace) = try tag("option", choices.map { Array(all[asked[$0].0].text[asked[$0].1]) }, trailing: trailing)
        let optionsOf = Dictionary(uniqueKeysWithValues: zip(choices, inner.map { $0.map(\.span) }))
        let text = String(String.UnicodeScalarView(markdown))
        var scalarAt: [Int] = []
        for (index, scalar) in markdown.enumerated() { scalarAt += Array(repeating: index, count: scalar.utf16.count) }
        scalarAt.append(markdown.count)
        let codeSpans = codeSpan.matches(in: text, range: NSRange(location: 0, length: text.utf16.count)).map {
            scalarAt[$0.range.location]..<scalarAt[$0.range.location + $0.range.length]
        }
        func input(_ origin: [Int], _ range: Range<Int>) -> Range<Int> {
            var start = origin[range.lowerBound]
            var end = origin[range.upperBound - 1] + 1
            for code in codeSpans where code.lowerBound < end && start < code.upperBound {
                start = min(start, code.lowerBound)
                end = max(end, code.upperBound)
            }
            return start..<end
        }
        let found = asked.enumerated().map { number, question -> Found in
            let (unit, span, kind) = question
            let origin = all[unit].origin
            let inners = optionsOf[number] ?? []
            let options = inners.map { input(origin, (span.lowerBound + $0.lowerBound)..<(span.lowerBound + $0.upperBound)) }
            return Found(unit: unit, span: span, range: input(origin, span), kind: kind, options: options, optionSpans: inners)
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
