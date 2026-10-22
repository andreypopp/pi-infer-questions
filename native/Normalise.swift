import Foundation

private func regex(_ pattern: String) -> NSRegularExpression {
    try! NSRegularExpression(pattern: pattern, options: [.dotMatchesLineSeparators])
}

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

private struct Line {
    let string: String
    let scalarAt: [Int]

    init(_ scalars: ArraySlice<Unicode.Scalar>) {
        string = String(String.UnicodeScalarView(scalars))
        var table: [Int] = []
        for (index, scalar) in scalars.enumerated() {
            table += Array(repeating: index, count: scalar.utf16.count)
        }
        scalarAt = table + [scalars.count]
    }

    func scalars(_ range: NSRange) -> Range<Int> {
        scalarAt[range.location]..<scalarAt[range.location + range.length]
    }

    func matches(_ expression: NSRegularExpression) -> [NSTextCheckingResult] {
        expression.matches(in: string, range: NSRange(location: 0, length: string.utf16.count))
    }

    func matchesStart(_ expression: NSRegularExpression) -> Bool {
        firstAtStart(expression) != nil
    }

    func firstAtStart(_ expression: NSRegularExpression) -> NSTextCheckingResult? {
        expression.firstMatch(in: string, options: [.anchored], range: NSRange(location: 0, length: string.utf16.count))
    }
}

/// The indexes of the characters of the inline markup in a line. Code spans and autolink URLs keep
/// their text.
private func inlineDrops(_ scalars: ArraySlice<Unicode.Scalar>) -> Set<Int> {
    let line = Line(scalars)
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

private func frontMatterLines(_ lines: [String]) -> Int {
    guard let first = lines.first,
          first.trimmingCharacters(in: .whitespacesAndNewlines) == "---" else { return 0 }
    for (number, line) in lines.enumerated().dropFirst() {
        let range = NSRange(location: 0, length: line.utf16.count)
        if let found = frontMatterEnd.firstMatch(in: line, options: [.anchored], range: range), found.range == range {
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
    var lines = bounds.map { range in mapped.text[range].map { $0 == "\r" ? " " : $0 } }
    let strings = lines.map { String(String.UnicodeScalarView($0)) }
    for index in 0..<frontMatterLines(strings) { lines[index] = [] }
    var kept = Set<Int>()
    let prose = proseParagraphs(lines.map { String(String.UnicodeScalarView($0)) })
    for range in prose.kept {
        if !Line(lines[range.lowerBound][...]).matchesStart(footnote) { kept.formUnion(range) }
    }
    var headings = Set<Int>()
    for range in prose.headings { headings.formUnion(range) }
    var mask = [Bool](repeating: true, count: mapped.text.count)
    for (number, range) in bounds.enumerated() {
        if !kept.contains(number) {
            for index in range { mask[index] = false }
        } else {
            for index in inlineDrops(lines[number][...]) { mask[range.lowerBound + index] = false }
            if headings.contains(number) {
                let line = Line(lines[number][...])
                for marker in [line.firstAtStart(atxOpen), line.matches(atxClose).first] {
                    if let marker { for index in line.scalars(marker.range) { mask[range.lowerBound + index] = false } }
                }
            }
        }
    }
    return (mapped.keep(mask), headings)
}
