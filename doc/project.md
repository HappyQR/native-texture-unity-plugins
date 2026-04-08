# NativeTexture Plugins

## Project Overview

### Background and Summary

This project provides native Unity plugins for decoding image files such as PNG and JPG and uploading them to the GPU. It supports separate implementations for Metal, Vulkan, and Direct3D12, and is designed to work with asynchronous usage from Unity.

`Texture2D.LoadImage` in Unity is synchronous and can cause noticeable stalls. To reduce that cost, this project performs image decoding and GPU texture creation in the native layer, returns a native texture pointer to C#, and then lets Unity create a usable texture through `Texture2D.CreateExternalTexture`. The plugin also exposes explicit texture release APIs.

### Directory Structure

- `doc`: project documentation
- `include`: shared header files and third-party headers
- `metal`: Metal implementation sources for NativeTexture
- `vulkan`: Vulkan implementation sources for NativeTexture
- `direct3d`: Direct3D12 implementation sources for NativeTexture
- `build/metal`, `build/vulkan`, `build/direct3d`: output directories for each platform
- `unity-scripts`: Unity-side C# integration scripts

## Solution

### Interface

At minimum, NativeTexture must expose the following APIs to the Unity C# layer:

- `uint8_t* Decode(uint8_t* raw, int length, int* width, int* height)`
- `void Free(uint8_t* rgba)`
- `void* Create(unsigned char* rgba, int width, int height)`
- `void Release(void* texPtr)`
- `bool SaveToFile(const char* fileName, uint8_t* rgba, int width, int height)`

### Build

Build scripts are provided in the project root so each platform can be built independently.

The project currently keeps these standalone scripts:

- `./build_metal.sh`
- `./build_vulkan.sh`
- `build_direct3d.bat`

It also provides a unified CMake entry:

- `cmake -S . -B .cmake/metal -DNATIVE_TEXTURE_BACKEND=metal`
- `cmake --build .cmake/metal --config Release`

- `cmake -S . -B .cmake/vulkan-arm64 -DNATIVE_TEXTURE_BACKEND=vulkan -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/android-unity.cmake -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=24`
- `cmake --build .cmake/vulkan-arm64 --config Release`

- `cmake -S . -B .cmake/d3d12 -DNATIVE_TEXTURE_BACKEND=d3d12`
- `cmake --build .cmake/d3d12 --config Release`

Notes:

- Metal is supported on Apple platforms only.
- Vulkan is currently configured for Android builds. `cmake/toolchains/android-unity.cmake` will try to find the Android NDK from environment variables or the Unity installation directory.
- Direct3D12 is supported on Windows only.
