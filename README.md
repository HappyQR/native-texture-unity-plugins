# NativeTexture

NativeTexture is a cross-backend native plugin for Unity that decodes image data and creates GPU textures in native code, then exposes them to C# through `Texture2D.CreateExternalTexture`.

Currently implemented backends:

- Metal
- Vulkan
- Direct3D12

## Features

- Decode and save `png`, `jpg`, `bmp`, and `tga`
- Async `Decode`, `Create`, `LoadTexture`, and `SaveToFile` APIs on the Unity C# side
- `CancellationToken` support
- Optional `flipY`
- Automatic sRGB native texture selection for color textures in Unity Linear color space
- `CreateFromEncoded` and `CreateFromFile` fast paths to reduce peak memory usage
- A unified facade that routes to Metal, Vulkan, or D3D12 by platform/backend

## Directory Layout

- [doc/project.md](doc/project.md)
- [include](include)
- [metal](metal)
- [vulkan](vulkan)
- [direct3d](direct3d)
- [unity-scripts](unity-scripts)
- [build](build)

## Native ABI

Base exported APIs:

- `uint8_t* Decode(uint8_t* raw, int length, int* width, int* height)`
- `void Free(uint8_t* rgba)`
- `void* Create(unsigned char* rgba, int width, int height)`
- `void Release(void* texPtr)`
- `bool SaveToFile(const char* fileName, uint8_t* rgba, int width, int height)`

Extended APIs:

- `DecodeWithOptions(..., bool flipY)`
- `CreateWithOptions(..., bool useSrgb)`
- `CreateFromEncoded(...)`
- `CreateFromEncodedWithOptions(..., bool useSrgb, bool flipY)`
- `CreateFromFile(...)`
- `CreateFromFileWithOptions(..., bool useSrgb, bool flipY)`

## Unity C# API

The unified entry point is [unity-scripts/NativeTextureEntry.cs](unity-scripts/NativeTextureEntry.cs). Main related types:

- [unity-scripts/NativeTextureEntry.cs](unity-scripts/NativeTextureEntry.cs)
- [unity-scripts/INativeTextureHandle.cs](unity-scripts/INativeTextureHandle.cs)
- [unity-scripts/NativeTextureDecodedImage.cs](unity-scripts/NativeTextureDecodedImage.cs)
- [unity-scripts/NativeTextureBackend.cs](unity-scripts/NativeTextureBackend.cs)

Recommended usage:

1. Call `NativeTextureEntry.Initialize()` as early as possible on the Unity main thread.
2. Use the default `linear: false` for regular color textures.
3. Pass `linear: true` explicitly for linear-data textures such as normal maps or LUTs.
4. Always call `Dispose()` on the returned handle when the texture is no longer needed.

Example:

```csharp
using System.Threading;
using System.Threading.Tasks;
using NativeTexture;

private INativeTextureHandle _loadedTexture;

public async Task LoadAsync(string filePath, CancellationToken cancellationToken)
{
    NativeTextureEntry.Initialize();

    _loadedTexture?.Dispose();
    _loadedTexture = await NativeTextureEntry.LoadTextureFromFileAsync(
        filePath,
        linear: false,
        flipY: true,
        cancellationToken);

    targetRenderer.material.mainTexture = _loadedTexture.Texture;
}
```

If you need CPU-side RGBA first and want to upload later:

```csharp
NativeTextureDecodedImage decoded = await NativeTextureEntry.DecodeFileAsync(path, flipY: true, cancellationToken);
_loadedTexture?.Dispose();
_loadedTexture = await NativeTextureEntry.CreateAsync(decoded, linear: false, cancellationToken);
```

## Build

### Standalone Scripts

- Metal: `./build_metal.sh`
- Vulkan Android: `./build_vulkan.sh`
- Direct3D12: `build_direct3d.bat`

Notes:

- `build_direct3d.bat` is intended to be run directly from `x64 Native Tools Command Prompt for VS 2022` or another Visual Studio Developer Command Prompt on Windows.
- `build_direct3d.sh` is still available for Bash-based Windows environments such as Git Bash, MSYS2, or Cygwin.
- D3D12 builds still require Visual Studio Build Tools or `clang-cl` plus the Windows SDK.

### CMake

Metal:

```bash
cmake -S . -B .cmake/metal -DNATIVE_TEXTURE_BACKEND=metal
cmake --build .cmake/metal
```

Vulkan Android:

```bash
cmake -S . -B .cmake/vulkan-arm64 \
  -DNATIVE_TEXTURE_BACKEND=vulkan \
  -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/android-unity.cmake \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=24
cmake --build .cmake/vulkan-arm64
```

Direct3D12 Windows:

```bash
cmake -S . -B .cmake/d3d12 -DNATIVE_TEXTURE_BACKEND=d3d12
cmake --build .cmake/d3d12 --config Release
```

## Output Locations

- Metal: `build/metal/libNativeTexture.dylib`
- Vulkan Android: `build/vulkan/android/<abi>/libNativeTexture.so`
- D3D12 Windows: `build/direct3d/<arch>/NativeTexture.dll`

## Unity Integration

Recommended plugin placement:

- macOS Metal: `Assets/Plugins/macOS/libNativeTexture.dylib`
- Android Vulkan: `Assets/Plugins/Android/<abi>/libNativeTexture.so`
- Windows D3D12: `Assets/Plugins/x86_64/NativeTexture.dll`

Add the C# files from `unity-scripts` to your Unity project, then call the plugin through `NativeTextureEntry`.

## Notes

- Cancellation is cooperative. It cannot forcibly interrupt a native decode or create call once that synchronous native call has already started.
- On Android, `LoadTextureFromFileAsync` requires a real filesystem path. APK-internal `jar:` paths cannot be passed directly to native `stb_image`.
- The returned handle types implement `IDisposable`. Do not keep only the `Texture2D` and discard the handle, or the native resource may not be released in time.
- The current Vulkan path is implemented for Android. Metal and D3D12 target Apple and Windows platforms respectively.
