import Foundation

private let footnote = regex(#"\s*\[\^[^\]]+\]:"#)
private let atxOpen = regex(#" {0,3}#{1,6}(?:\s+|$)"#)
private let atxClose = regex(#"\s+#+\s*$"#)
private let frontMatterEnd = regex(#"(?:---|\.\.\.)\s*"#)
private let codeSpan = regex(#"(`+)(.+?)\1"#)
private let image = regex(#"!\[[^\]]*\]\([^)]*\)"#)
private let link = regex(#"\[([^\]]*)\]\([^)]*\)"#)
private let autolink = regex(#"<[A-Za-z][A-Za-z0-9+.-]*:[^\s>]*>"#)
private let emphasis = [
    regex(#"</?[A-Za-z][^>]*>"#),
    regex(#"\*\*|~~|(?<!\w)__(?=\S)|(?<=\S)__(?!\w)"#),
    regex(#"(?<![\w*])\*(?=[^\s*])|(?<=[^\s*])\*(?![\w*])"#),
    regex(#"(?<!\w)_(?=[^\s_])|(?<=[^\s_])_(?!\w)"#),
]

/// The indexes of the characters of the inline markup in a line. Code spans and autolink URLs keep
/// their text.
private func inlineDrops(_ line: Line) -> Set<Int> {
    var protected = Set<Int>()
    var drops = Set<Int>()
    func drop(_ range: Range<Int>) {
        if protected.isDisjoint(with: range) { drops.formUnion(range) }
    }
    for span in line.matches(codeSpan) {
        let content = line.scalars(span.range(at: 2))
        protected.formUnion(content)
        drops.formUnion(line.scalars(span.range(at: 1)))
        drops.formUnion(content.upperBound..<line.scalars(span.range).upperBound)
    }
    for found in line.matches(autolink) {
        let range = line.scalars(found.range)
        drop(range.lowerBound..<(range.lowerBound + 1))
        drop((range.upperBound - 1)..<range.upperBound)
        protected.formUnion(range)
    }
    for found in line.matches(image) { drop(line.scalars(found.range)) }
    for found in line.matches(link) {
        let all = line.scalars(found.range)
        let label = line.scalars(found.range(at: 1))
        drop(all.lowerBound..<label.lowerBound)
        drop(label.upperBound..<all.upperBound)
    }
    for pattern in emphasis {
        for found in line.matches(pattern) { drop(line.scalars(found.range)) }
    }
    return drops
}

private func frontMatterLines(_ lines: [Line]) -> Int {
    guard let first = lines.first,
          first.string.trimmingCharacters(in: .whitespacesAndNewlines) == "---" else { return 0 }
    for (number, line) in lines.enumerated().dropFirst() {
        if let found = line.starting(frontMatterEnd), found.range == line.full {
            return number + 1
        }
    }
    return 0
}

/// The paragraphs and list items of a Markdown message, with the lines of the others empty; and the
/// lines of the headings and bold lines that introduce a list, without their # markers.
func normalise(_ text: Scalars) -> (mapped: Mapped, headings: Set<Int>) {
    let mapped = Mapped(text: text, origin: Array(text.indices))
    let bounds = mapped.lines
    var lines = bounds.map { range in Line(mapped.text[range].map { $0 == "\r" ? " " : $0 }) }
    for index in 0..<frontMatterLines(lines) { lines[index] = Line([]) }
    let prose = proseParagraphs(lines.map(\.string))
    let kept = Set(prose.kept.filter { lines[$0.lowerBound].starting(footnote) == nil }.joined())
    let headings = Set(prose.headings.joined())
    var mask = [Bool](repeating: true, count: mapped.text.count)
    for (number, range) in bounds.enumerated() {
        guard kept.contains(number) else {
            for index in range { mask[index] = false }
            continue
        }
        let line = lines[number]
        var drops = inlineDrops(line)
        if headings.contains(number) {
            for marker in [line.starting(atxOpen), line.first(atxClose)] {
                if let marker { drops.formUnion(line.scalars(marker.range)) }
            }
        }
        for index in drops { mask[range.lowerBound + index] = false }
    }
    return (mapped.keep(mask), headings)
}

private let singleLineCodeSpan = regex(#"(`+)[^`\n]+?\1"#)

/// Widens a range of the model input to the original Markdown, over every code span it overlaps.
func widening(_ markdown: Scalars) -> (_ origin: [Int], _ range: Range<Int>) -> Range<Int> {
    let line = Line(markdown)
    let codes = line.matches(singleLineCodeSpan).map { line.scalars($0.range) }
    return { origin, range in
        var start = origin[range.lowerBound]
        var end = origin[range.upperBound - 1] + 1
        for code in codes where code.lowerBound < end && start < code.upperBound {
            start = min(start, code.lowerBound)
            end = max(end, code.upperBound)
        }
        return start..<end
    }
}
