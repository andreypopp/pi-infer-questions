import Cmark
import Foundation

private func hasStrong(_ node: OpaquePointer) -> Bool {
    var child = cmark_node_first_child(node)
    while let current = child {
        if cmark_node_get_type(current) == CMARK_NODE_STRONG || hasStrong(current) { return true }
        child = cmark_node_next(current)
    }
    return false
}

/// A paragraph that is one strong-emphasis span, with an optional colon after it.
private func isPseudoHeading(_ node: OpaquePointer) -> Bool {
    var spans: [OpaquePointer] = []
    var child = cmark_node_first_child(node)
    while let current = child {
        let trivial: Bool
        if cmark_node_get_type(current) == CMARK_NODE_TEXT, let literal = cmark_node_get_literal(current) {
            let content = String(cString: literal).trimmingCharacters(in: .whitespacesAndNewlines)
            trivial = content.isEmpty || content == ":"
        } else {
            trivial = false
        }
        if !trivial { spans.append(current) }
        child = cmark_node_next(current)
    }
    return spans.count == 1 && cmark_node_get_type(spans[0]) == CMARK_NODE_STRONG && !hasStrong(spans[0])
}

private let delimiterRow = try! NSRegularExpression(pattern: #"^\s*\|?(?:\s*:?-+:?\s*\|)*\s*:?-+:?\s*\|?\s*$"#)

/// A paragraph whose last line became a table header has no line numbers. It starts where the table
/// starts and ends above the header row, which is the line above the delimiter row.
private func paragraphLines(_ node: OpaquePointer, _ lines: [String]) -> Range<Int> {
    if cmark_node_get_start_line(node) > 0 { return (Int(cmark_node_get_start_line(node)) - 1)..<Int(cmark_node_get_end_line(node)) }
    let start = cmark_node_next(node).map { Int(cmark_node_get_start_line($0)) - 1 } ?? 0
    let delimiter = lines[(start + 1)...].firstIndex { line in
        line.contains("-") && delimiterRow.firstMatch(in: line, range: NSRange(location: 0, length: line.utf16.count)) != nil
    }
    return start..<((delimiter ?? start + 1) - 1)
}

private let setextUnderline = try! NSRegularExpression(pattern: #"^ {0,3}(?:=+|-+)\s*$"#)
private let notKept: Set<String> = ["block_quote", "code_block", "html_block", "table", "thematic_break"]

/// The next top-level block that stays is a list.
private func introducesList(_ node: OpaquePointer) -> Bool {
    var next = cmark_node_next(node)
    while let current = next, notKept.contains(String(cString: cmark_node_get_type_string(current))) {
        next = cmark_node_next(current)
    }
    return next.map { cmark_node_get_type($0) == CMARK_NODE_LIST } ?? false
}

/// The 0-based line ranges of the paragraphs of a CommonMark document that are not in block
/// quotes and are not pseudo-headings (a list item can hold them), and the ranges of the headings
/// and pseudo-headings that stay because the next kept block is a list.
func proseParagraphs(_ lines: [String]) -> (kept: [Range<Int>], headings: [Range<Int>]) {
    cmark_gfm_core_extensions_ensure_registered()
    let parser = cmark_parser_new(0)!
    if let table = cmark_find_syntax_extension("table") { _ = cmark_parser_attach_syntax_extension(parser, table) }
    lines.joined(separator: "\n").withCString { cmark_parser_feed(parser, $0, strlen($0)) }
    let root = cmark_parser_finish(parser)!
    defer {
        cmark_node_free(root)
        cmark_parser_free(parser)
    }
    var found: [Range<Int>] = []
    var headings: [Range<Int>] = []
    func walk(_ node: OpaquePointer, top: Bool, listed: Bool) {
        var child = cmark_node_first_child(node)
        while let current = child {
            switch cmark_node_get_type(current) {
            case CMARK_NODE_BLOCK_QUOTE:
                break
            case CMARK_NODE_HEADING where top:
                let first = Int(cmark_node_get_start_line(current)) - 1
                let atx = lines[first].range(of: #"^ {0,3}#"#, options: .regularExpression) != nil
                let underline = lines.indices.dropFirst(first + 1).first { line in
                    setextUnderline.firstMatch(in: lines[line], range: NSRange(location: 0, length: lines[line].utf16.count)) != nil
                }
                let range = first..<(atx ? first + 1 : underline ?? first + 1)
                if introducesList(current) { headings.append(range) }
            case CMARK_NODE_PARAGRAPH:
                let range = paragraphLines(current, lines)
                let asks = lines[range].contains { $0.contains("?") }
                if listed || asks || !isPseudoHeading(current) {
                    found.append(range)
                } else if top, introducesList(current) {
                    headings.append(range)
                }
            default:
                walk(current, top: false, listed: listed || cmark_node_get_type(current) == CMARK_NODE_LIST)
            }
            child = cmark_node_next(current)
        }
    }
    walk(root, top: true, listed: false)
    return (found + headings, headings)
}
