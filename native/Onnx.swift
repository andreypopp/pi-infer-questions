#if !canImport(CoreML)
import Foundation
import OnnxRuntime

private struct Failure: Error, CustomStringConvertible {
    let description: String
}

private struct Batch: Decodable {
    struct Limits: Decodable {
        let rows: Int
        let tokens: Int
    }

    let batch: Limits
}

private func batches(_ lengths: [Int], _ limits: Batch.Limits) -> [[Int]] {
    var found: [[Int]] = []
    for index in lengths.indices.sorted(by: { (lengths[$0], $0) < (lengths[$1], $1) }) {
        if let last = found.last, last.count < limits.rows, (last.count + 1) * lengths[index] <= limits.tokens {
            found[found.count - 1].append(index)
        } else {
            found.append([index])
        }
    }
    return found
}

struct Models {
    private let api: UnsafePointer<OrtApi>
    private let memory: OpaquePointer
    private let sessions: [String: OpaquePointer]
    private let classes: [String: Int]
    private let inputNames: [UnsafePointer<CChar>?]
    private let outputNames: [UnsafePointer<CChar>?]
    private let padTokenId: Int
    private let limits: Batch.Limits

    init(directory: URL, settings: Settings) throws {
        guard let api = OrtGetApiBase()?.pointee.GetApi(UInt32(ORT_API_VERSION)) else { throw CocoaError(.coderReadCorrupt) }
        self.api = api
        var environment: OpaquePointer?
        try check(api, api.pointee.CreateEnv(ORT_LOGGING_LEVEL_ERROR, "questions", &environment))
        // The official ONNX Runtime build sends usage telemetry to Microsoft unless it is disabled.
        try check(api, api.pointee.DisableTelemetryEvents(environment))
        var options: OpaquePointer?
        try check(api, api.pointee.CreateSessionOptions(&options))
        var created: [String: OpaquePointer] = [:]
        for task in ["question", "option"] {
            var session: OpaquePointer?
            try check(api, api.pointee.CreateSession(environment, directory.appendingPathComponent("\(task).onnx").path, options, &session))
            created[task] = session
        }
        sessions = created
        var info: OpaquePointer?
        try check(api, api.pointee.CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &info))
        memory = info!
        classes = settings.tags.mapValues(\.count)
        func names(_ strings: [String]) -> [UnsafePointer<CChar>?] { strings.map { strdup($0).map { UnsafePointer($0) } } }
        inputNames = names(["input_ids", "attention_mask"])
        outputNames = names(["logits"])
        padTokenId = settings.padTokenId
        limits = try JSONDecoder().decode(Batch.self, from: Data(contentsOf: directory.appendingPathComponent("settings.json"))).batch
    }

    func tags(_ task: String, _ ids: [[Int]]) throws -> (tags: [[Int]], packs: [[Int]]) {
        var out = [[Int]](repeating: [], count: ids.count)
        let packs = batches(ids.map(\.count), limits)
        for row in packs {
            let width = row.map { ids[$0].count }.max() ?? 0
            let tags = try run(task, row.map { ids[$0] }, width: width)
            for (number, unit) in row.enumerated() {
                out[unit] = Array(tags[number * width..<(number * width + ids[unit].count)])
            }
        }
        return (out, packs)
    }

    private func run(_ task: String, _ rows: [[Int]], width: Int) throws -> [Int] {
        let count = rows.count * width
        let tokens = UnsafeMutablePointer<Int64>.allocate(capacity: 2 * count)
        defer { tokens.deallocate() }
        for (number, row) in rows.enumerated() {
            for index in 0..<width {
                tokens[number * width + index] = index < row.count ? Int64(row[index]) : Int64(padTokenId)
                tokens[count + number * width + index] = index < row.count ? 1 : 0
            }
        }
        let shape = [Int64(rows.count), Int64(width)]
        var inputs = [OpaquePointer?](repeating: nil, count: 2)
        defer { inputs.forEach { api.pointee.ReleaseValue($0) } }
        for slot in 0..<2 {
            try check(api, api.pointee.CreateTensorWithDataAsOrtValue(
                memory, tokens + slot * count, count * MemoryLayout<Int64>.size, shape, 2,
                ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &inputs[slot]))
        }
        var output: OpaquePointer?
        defer { api.pointee.ReleaseValue(output) }
        try check(api, api.pointee.Run(sessions[task]!, nil, inputNames, inputs, 2, outputNames, 1, &output))
        var info: OpaquePointer?
        try check(api, api.pointee.GetTensorTypeAndShape(output, &info))
        defer { api.pointee.ReleaseTensorTypeAndShapeInfo(info) }
        var elements = 0
        try check(api, api.pointee.GetTensorShapeElementCount(info, &elements))
        let classes = classes[task]!
        guard elements == count * classes else { throw CocoaError(.coderReadCorrupt) }
        var raw: UnsafeMutableRawPointer?
        try check(api, api.pointee.GetTensorMutableData(output, &raw))
        let logits = raw!.assumingMemoryBound(to: Float.self)
        return (0..<count).map { position in
            var best = 0
            for tag in 1..<classes where logits[position * classes + tag] > logits[position * classes + best] { best = tag }
            return best
        }
    }
}

private func check(_ api: UnsafePointer<OrtApi>, _ status: OpaquePointer?) throws {
    guard let status else { return }
    defer { api.pointee.ReleaseStatus(status) }
    throw Failure(description: String(cString: api.pointee.GetErrorMessage(status)!))
}
#endif
