import Metal
import MetalPerformanceShaders
import Accelerate

 let metallib =  "/Users/noppoman/WorkingPlace/program/math/deep-learning-swift/python-nn-with-swift-and-metal/PyMetalBridge/Sources/PyMetalBridge/Shaders.metallib"

 @available(macOS 10.13, *)
 let device = MTLCreateSystemDefaultDevice()!,
     commandQueue = device.makeCommandQueue()!,
     defaultLibrary = try! device.makeLibrary(filepath: metallib)

@available(macOS 10.13, *)
@_cdecl("swift_sigmoid_on_gpu")
public func swift_sigmoid_on_gpu(input: UnsafePointer<Float>, output: UnsafeMutablePointer<Float>, count: Int) -> Int {
    return computeOnGPU1D(functionName: "sigmoid", input: input, output: output, count: count)
}

@available(macOS 10.13, *)
@_cdecl("swift_differential_on_gpu")
public func swift_differential_on_gpu(input: UnsafePointer<Float>, output: UnsafeMutablePointer<Float>, count: Int) -> Int {
    return computeOnGPU1D(functionName: "differential", input: input, output: output, count: count)
}

@available(macOS 10.13, *)
func computeOnGPU1D(functionName: String,  input: UnsafePointer<Float>, output: UnsafeMutablePointer<Float>, count: Int) -> Int {
    do {
        let inputBuffer = UnsafeRawPointer(input)
        let commandBuffer = commandQueue.makeCommandBuffer()!
        let computeCommandEncoder = commandBuffer.makeComputeCommandEncoder()!

        let sigmoidFunction = defaultLibrary.makeFunction(name: functionName)!
        let computePipelineState = try device.makeComputePipelineState(function: sigmoidFunction)
        computeCommandEncoder.setComputePipelineState(computePipelineState)

        let inputByteLength = count*MemoryLayout<Float>.size

        let inVectorBuffer = device.makeBuffer(bytes: inputBuffer, length: inputByteLength, options: [])

        computeCommandEncoder.setBuffer(inVectorBuffer, offset: 0, index: 0)

        let resultRef = UnsafeMutablePointer<Float>.allocate(capacity: count)
        let outVectorBuffer = device.makeBuffer(bytes: resultRef, length: inputByteLength, options: [])

        computeCommandEncoder.setBuffer(outVectorBuffer, offset: 0, index: 1)
        
        let maxTotalThreadsPerThreadgroup = computePipelineState.maxTotalThreadsPerThreadgroup
        let threadExecutionWidth = computePipelineState.threadExecutionWidth
        let width  = maxTotalThreadsPerThreadgroup / threadExecutionWidth * threadExecutionWidth
        let height = 1
        let depth  = 1
        
        // 1D
        let threadsPerGroup = MTLSize(width:width, height: height, depth: depth)
        let numThreadgroups = MTLSize(width: (count + width - 1) / width, height: 1, depth: 1)
        
        computeCommandEncoder.dispatchThreadgroups(numThreadgroups, threadsPerThreadgroup: threadsPerGroup)
        computeCommandEncoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        // unsafe bitcast and assigin result pointer to output
        output.initialize(from: outVectorBuffer!.contents().assumingMemoryBound(to: Float.self), count: count)
        
        free(resultRef)

        return 0
    } catch {
        print("\(error)")
        return 1
    }
}
