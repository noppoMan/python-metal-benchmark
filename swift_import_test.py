import numpy as np
import ctypes
swift_fun = ctypes.CDLL("./PyMetalBridge/.build/release/libPyMetalBridge.dylib")

input_array = np.arange(-50, 50, 0.001).astype("float32") # have to be float32 for GPU
input_array.shape

swift_fun.swift_sigmoid_on_gpu.argtypes = [
    ctypes.POINTER(ctypes.c_float), 
    ctypes.POINTER(ctypes.c_float), 
    ctypes.c_int,
]

swift_fun.swift_differential_on_gpu.argtypes = [
    ctypes.POINTER(ctypes.c_float), 
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int, # cols
]

input_ptr = input_array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
output_mutable_ptr = (ctypes.c_float * len(input_array))()

swift_fun.swift_differential_on_gpu(input_ptr, output_mutable_ptr, len(input_array))
print(np.array(output_mutable_ptr))