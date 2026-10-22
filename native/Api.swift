import Foundation

private let state = NSCondition()
private var loading = true
private var engine: Result<Engine, Error> = .failure(CocoaError(.fileNoSuchFile))

@_cdecl("pq_start")
public func start(_ directory: UnsafePointer<CChar>) {
    let url = URL(fileURLWithPath: String(cString: directory), isDirectory: true)
    Thread.detachNewThread {
        let loaded = Result { try Engine(directory: url) }
        state.lock()
        engine = loaded
        loading = false
        state.broadcast()
        state.unlock()
    }
}

private func pair(_ range: Range<Int>) -> [Int] { [range.lowerBound, range.upperBound] }

private func json(_ trace: Trace, _ key: String, _ indexes: [Int]) -> [String: Any] {
    [
        key: indexes,
        "packs": trace.packs,
        "tokens": trace.tokens.map { ["ids": $0.ids, "offsets": $0.offsets.map(pair)] as [String: Any] },
    ]
}

/// Blocks until the models are loaded. Returns a malloc'd JSON string, or nil with a malloc'd
/// message in `failure`. With `trace`, the JSON holds every step of the analysis.
@_cdecl("pq_questions")
public func questions(_ markdown: UnsafePointer<CChar>, _ trace: Bool, _ failure: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>) -> UnsafeMutablePointer<CChar>? {
    state.lock()
    while loading { state.wait() }
    let current = engine
    state.unlock()
    do {
        let analysis = try current.get().analyse(Array(String(cString: markdown).unicodeScalars))
        let questions = analysis.found.map { question -> [String: Any] in
            var object: [String: Any] = ["range": pair(question.range), "kind": question.kind, "options": question.options.map(pair)]
            if trace {
                object["unit"] = question.unit
                object["span"] = pair(question.span)
                object["option_spans"] = question.optionSpans.map(pair)
            }
            return object
        }
        var result: Any = questions
        if trace {
            result = [
                "questions": questions,
                "units": analysis.units.enumerated().map { number, unit in
                    ["text": unit.text.string, "origin": unit.origin, "context": unit.context, "gated": analysis.gated.contains(number)] as [String: Any]
                },
                "question_tokens": json(analysis.questionTrace, "units", analysis.gated),
                "option_tokens": json(analysis.optionTrace, "questions", analysis.choices),
            ] as [String: Any]
        }
        let data = try JSONSerialization.data(withJSONObject: result, options: [.sortedKeys])
        return strdup(String(decoding: data, as: UTF8.self))
    } catch {
        failure.pointee = strdup(String(describing: error))
        return nil
    }
}
