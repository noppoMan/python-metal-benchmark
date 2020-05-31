xcrun -sdk macosx metal -c Sources/PyMetalBridge/Shaders.metal -o Sources/PyMetalBridge/Shaders.air
xcrun -sdk macosx metallib Sources/PyMetalBridge/Shaders.air -o Sources/PyMetalBridge/Shaders.metallib
