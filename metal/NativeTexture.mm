#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include <cctype>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <mutex>
#include <unordered_set>
#include <vector>

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmissing-field-initializers"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#import "NativeTexture.h"
#import "stb/stb_image.h"
#import "stb/stb_image_write.h"
#pragma clang diagnostic pop
#import "unity/IUnityGraphics.h"
#import "unity/IUnityGraphicsMetal.h"

#define NATIVE_TEXTURE_API extern "C" UNITY_INTERFACE_EXPORT

namespace
{
constexpr int kBytesPerPixel = 4;
constexpr int kJpegQuality = 90;

std::mutex gUnityStateMutex;
std::mutex gTextureMutex;
#ifndef STBI_THREAD_LOCAL
std::mutex gStbiLoadMutex;
#endif
IUnityInterfaces* gUnityInterfaces = nullptr;
IUnityGraphics* gUnityGraphics = nullptr;
IUnityGraphicsMetalV2* gUnityGraphicsMetalV2 = nullptr;
IUnityGraphicsMetalV1* gUnityGraphicsMetalV1 = nullptr;
IUnityGraphicsMetal* gUnityGraphicsMetal = nullptr;
id<MTLDevice> gFallbackDevice = nil;
std::unordered_set<void*> gLiveTextures;

struct ScopedStbiLoadFlip
{
#ifndef STBI_THREAD_LOCAL
    std::unique_lock<std::mutex> loadLock;

    explicit ScopedStbiLoadFlip(bool flipY)
        : loadLock(gStbiLoadMutex)
    {
        stbi_set_flip_vertically_on_load(flipY ? 1 : 0);
    }
#else
    explicit ScopedStbiLoadFlip(bool flipY)
    {
        stbi_set_flip_vertically_on_load_thread(flipY ? 1 : 0);
    }
#endif

    ~ScopedStbiLoadFlip()
    {
#ifndef STBI_THREAD_LOCAL
        stbi_set_flip_vertically_on_load(0);
#else
        stbi_set_flip_vertically_on_load_thread(0);
#endif
    }
};

void LogMessage(const char* level, const char* format, ...)
{
    va_list args;
    va_start(args, format);
    fprintf(stderr, "[NativeTexture][%s] ", level);
    vfprintf(stderr, format, args);
    fprintf(stderr, "\n");
    va_end(args);
}

const char* RendererName(UnityGfxRenderer renderer)
{
    switch (renderer)
    {
        case kUnityGfxRendererMetal:
            return "Metal";
        case kUnityGfxRendererVulkan:
            return "Vulkan";
        case kUnityGfxRendererD3D12:
            return "D3D12";
        case kUnityGfxRendererNull:
            return "Null";
        default:
            return "Other";
    }
}

void ClearMetalInterfacesLocked()
{
    gUnityGraphicsMetalV2 = nullptr;
    gUnityGraphicsMetalV1 = nullptr;
    gUnityGraphicsMetal = nullptr;
}

void RefreshMetalInterfacesLocked()
{
    ClearMetalInterfacesLocked();
    if (!gUnityInterfaces || !gUnityGraphics)
    {
        return;
    }

    if (gUnityGraphics->GetRenderer() != kUnityGfxRendererMetal)
    {
        return;
    }

    gUnityGraphicsMetalV2 = gUnityInterfaces->Get<IUnityGraphicsMetalV2>();
    if (!gUnityGraphicsMetalV2)
    {
        gUnityGraphicsMetalV1 = gUnityInterfaces->Get<IUnityGraphicsMetalV1>();
    }
    if (!gUnityGraphicsMetalV2 && !gUnityGraphicsMetalV1)
    {
        gUnityGraphicsMetal = gUnityInterfaces->Get<IUnityGraphicsMetal>();
    }
}

void UNITY_INTERFACE_API OnGraphicsDeviceEvent(UnityGfxDeviceEventType eventType)
{
    std::lock_guard<std::mutex> lock(gUnityStateMutex);
    switch (eventType)
    {
        case kUnityGfxDeviceEventInitialize:
        case kUnityGfxDeviceEventAfterReset:
            RefreshMetalInterfacesLocked();
            break;
        case kUnityGfxDeviceEventBeforeReset:
        case kUnityGfxDeviceEventShutdown:
            ClearMetalInterfacesLocked();
            break;
        default:
            break;
    }
}

bool MatchesExtension(const char* extension, const char* expected)
{
    if (!extension || !expected)
    {
        return false;
    }

    while (*extension != '\0' && *expected != '\0')
    {
        const unsigned char lhs = static_cast<unsigned char>(*extension++);
        const unsigned char rhs = static_cast<unsigned char>(*expected++);
        if (std::tolower(lhs) != std::tolower(rhs))
        {
            return false;
        }
    }
    return *extension == '\0' && *expected == '\0';
}

bool ValidateImageArgs(const void* rgba, int width, int height, const char* operation)
{
    if (!rgba)
    {
        LogMessage("Error", "%s failed: rgba buffer is null", operation);
        return false;
    }

    if (width <= 0 || height <= 0)
    {
        LogMessage("Error", "%s failed: invalid image size %dx%d", operation, width, height);
        return false;
    }

    if (width > std::numeric_limits<int>::max() / kBytesPerPixel)
    {
        LogMessage("Error", "%s failed: width %d overflows bytesPerRow", operation, width);
        return false;
    }

    return true;
}

id<MTLDevice> GetMetalDevice()
{
    std::lock_guard<std::mutex> lock(gUnityStateMutex);

    if (gUnityGraphics)
    {
        const UnityGfxRenderer renderer = gUnityGraphics->GetRenderer();
        if (renderer != kUnityGfxRendererMetal)
        {
            LogMessage("Error", "Create failed: Unity renderer is %s (%d), Metal plugin path cannot create textures",
                       RendererName(renderer), renderer);
            return nil;
        }

        RefreshMetalInterfacesLocked();

        if (gUnityGraphicsMetalV2)
        {
            return gUnityGraphicsMetalV2->MetalDevice();
        }
        if (gUnityGraphicsMetalV1)
        {
            return gUnityGraphicsMetalV1->MetalDevice();
        }
        if (gUnityGraphicsMetal)
        {
            return gUnityGraphicsMetal->MetalDevice();
        }

        LogMessage("Error", "Create failed: Unity Metal device is not ready");
        return nil;
    }

    if (!gFallbackDevice)
    {
        gFallbackDevice = MTLCreateSystemDefaultDevice();
        if (!gFallbackDevice)
        {
            LogMessage("Error", "Create failed: system Metal device is unavailable");
        }
        else
        {
            LogMessage("Warn", "Unity graphics interface is unavailable, falling back to system Metal device");
        }
    }

    return gFallbackDevice;
}

void TrackTexture(void* texturePtr)
{
    if (!texturePtr)
    {
        return;
    }

    std::lock_guard<std::mutex> lock(gTextureMutex);
    gLiveTextures.insert(texturePtr);
}

bool UntrackTexture(void* texturePtr)
{
    std::lock_guard<std::mutex> lock(gTextureMutex);
    return gLiveTextures.erase(texturePtr) > 0;
}

std::vector<void*> TakeOutstandingTextures()
{
    std::lock_guard<std::mutex> lock(gTextureMutex);

    std::vector<void*> textures;
    textures.reserve(gLiveTextures.size());
    for (void* texturePtr : gLiveTextures)
    {
        textures.push_back(texturePtr);
    }

    gLiveTextures.clear();
    return textures;
}

uint8_t* DecodeMemoryInternal(uint8_t* raw, int length, int* width, int* height, bool flipY, const char* operation)
{
    if (width)
    {
        *width = 0;
    }
    if (height)
    {
        *height = 0;
    }

    if (!raw || length <= 0 || !width || !height)
    {
        LogMessage("Error", "%s failed: invalid input buffer or output size pointers", operation);
        return nullptr;
    }

    ScopedStbiLoadFlip scopedFlip(flipY);

    int channels = 0;
    int decodedWidth = 0;
    int decodedHeight = 0;
    uint8_t* rgba = stbi_load_from_memory(raw, length, &decodedWidth, &decodedHeight, &channels, kBytesPerPixel);
    if (!rgba)
    {
        const char* reason = stbi_failure_reason();
        LogMessage("Error", "%s failed: %s", operation, reason ? reason : "unknown stb_image error");
        return nullptr;
    }

    *width = decodedWidth;
    *height = decodedHeight;
    return rgba;
}

uint8_t* DecodeFileInternal(const char* fileName, int* width, int* height, bool flipY, const char* operation)
{
    if (width)
    {
        *width = 0;
    }
    if (height)
    {
        *height = 0;
    }

    if (!fileName || fileName[0] == '\0' || !width || !height)
    {
        LogMessage("Error", "%s failed: invalid fileName or output size pointers", operation);
        return nullptr;
    }

    ScopedStbiLoadFlip scopedFlip(flipY);

    int channels = 0;
    int decodedWidth = 0;
    int decodedHeight = 0;
    uint8_t* rgba = stbi_load(fileName, &decodedWidth, &decodedHeight, &channels, kBytesPerPixel);
    if (!rgba)
    {
        const char* reason = stbi_failure_reason();
        LogMessage("Error", "%s failed: %s", operation, reason ? reason : "unknown stb_image error");
        return nullptr;
    }

    *width = decodedWidth;
    *height = decodedHeight;
    return rgba;
}
}

NATIVE_TEXTURE_API uint8_t* UNITY_INTERFACE_API Decode(uint8_t* raw, int length, int* width, int* height)
{
    return DecodeWithOptions(raw, length, width, height, false);
}

NATIVE_TEXTURE_API uint8_t* UNITY_INTERFACE_API DecodeWithOptions(
    uint8_t* raw,
    int length,
    int* width,
    int* height,
    bool flipY)
{
    return DecodeMemoryInternal(raw, length, width, height, flipY, "Decode");
}

NATIVE_TEXTURE_API void UNITY_INTERFACE_API Free(uint8_t* rgba)
{
    if (rgba)
    {
        stbi_image_free(rgba);
    }
}

NATIVE_TEXTURE_API bool UNITY_INTERFACE_API SaveToFile(const char* fileName, uint8_t* rgba, int width, int height)
{
    if (!fileName || fileName[0] == '\0')
    {
        LogMessage("Error", "SaveToFile failed: fileName is empty");
        return false;
    }
    if (!ValidateImageArgs(rgba, width, height, "SaveToFile"))
    {
        return false;
    }

    const char* extension = std::strrchr(fileName, '.');
    if (!extension)
    {
        extension = ".png";
    }

    int result = 0;
    if (MatchesExtension(extension, ".png"))
    {
        result = stbi_write_png(fileName, width, height, kBytesPerPixel, rgba, width * kBytesPerPixel);
    }
    else if (MatchesExtension(extension, ".jpg") || MatchesExtension(extension, ".jpeg"))
    {
        result = stbi_write_jpg(fileName, width, height, kBytesPerPixel, rgba, kJpegQuality);
    }
    else if (MatchesExtension(extension, ".bmp"))
    {
        result = stbi_write_bmp(fileName, width, height, kBytesPerPixel, rgba);
    }
    else if (MatchesExtension(extension, ".tga"))
    {
        result = stbi_write_tga(fileName, width, height, kBytesPerPixel, rgba);
    }
    else
    {
        LogMessage("Error", "SaveToFile failed: unsupported file extension for %s", fileName);
        return false;
    }

    if (!result)
    {
        LogMessage("Error", "SaveToFile failed: stb_image_write could not write %s", fileName);
        return false;
    }

    return true;
}

static void* CreateInternal(unsigned char* rgba, int width, int height, bool useSrgb)
{
    @autoreleasepool
    {
        if (!ValidateImageArgs(rgba, width, height, "Create"))
        {
            return nullptr;
        }

        id<MTLDevice> device = GetMetalDevice();
        if (!device)
        {
            return nullptr;
        }

        const NSUInteger textureWidth = static_cast<NSUInteger>(width);
        const NSUInteger textureHeight = static_cast<NSUInteger>(height);
        const NSUInteger bytesPerRow = textureWidth * kBytesPerPixel;
        const MTLPixelFormat pixelFormat = useSrgb ? MTLPixelFormatRGBA8Unorm_sRGB : MTLPixelFormatRGBA8Unorm;

        MTLTextureDescriptor* descriptor =
            [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:pixelFormat
                                                              width:textureWidth
                                                             height:textureHeight
                                                          mipmapped:NO];
        descriptor.usage = MTLTextureUsageShaderRead;

        id<MTLTexture> texture = [device newTextureWithDescriptor:descriptor];
        if (!texture)
        {
            LogMessage("Error", "Create failed: Metal texture allocation returned null");
            return nullptr;
        }

        const MTLRegion region = MTLRegionMake2D(0, 0, textureWidth, textureHeight);
        [texture replaceRegion:region
                   mipmapLevel:0
                     withBytes:rgba
                   bytesPerRow:bytesPerRow];

        void* texturePtr = (__bridge_retained void*)texture;
        TrackTexture(texturePtr);
        return texturePtr;
    }
}

NATIVE_TEXTURE_API void* UNITY_INTERFACE_API Create(unsigned char* rgba, int width, int height)
{
    return CreateInternal(rgba, width, height, false);
}

NATIVE_TEXTURE_API void* UNITY_INTERFACE_API CreateWithOptions(unsigned char* rgba, int width, int height, bool useSrgb)
{
    return CreateInternal(rgba, width, height, useSrgb);
}

NATIVE_TEXTURE_API void* UNITY_INTERFACE_API CreateFromEncoded(
    uint8_t* raw,
    int length,
    int* width,
    int* height,
    bool useSrgb)
{
    return CreateFromEncodedWithOptions(raw, length, width, height, useSrgb, false);
}

NATIVE_TEXTURE_API void* UNITY_INTERFACE_API CreateFromEncodedWithOptions(
    uint8_t* raw,
    int length,
    int* width,
    int* height,
    bool useSrgb,
    bool flipY)
{
    uint8_t* rgba = DecodeMemoryInternal(raw, length, width, height, flipY, "CreateFromEncoded");
    if (!rgba)
    {
        return nullptr;
    }

    void* texture = CreateInternal(rgba, *width, *height, useSrgb);
    stbi_image_free(rgba);
    return texture;
}

NATIVE_TEXTURE_API void* UNITY_INTERFACE_API CreateFromFile(
    const char* fileName,
    int* width,
    int* height,
    bool useSrgb)
{
    return CreateFromFileWithOptions(fileName, width, height, useSrgb, false);
}

NATIVE_TEXTURE_API void* UNITY_INTERFACE_API CreateFromFileWithOptions(
    const char* fileName,
    int* width,
    int* height,
    bool useSrgb,
    bool flipY)
{
    uint8_t* rgba = DecodeFileInternal(fileName, width, height, flipY, "CreateFromFile");
    if (!rgba)
    {
        return nullptr;
    }

    void* texture = CreateInternal(rgba, *width, *height, useSrgb);
    stbi_image_free(rgba);
    return texture;
}

NATIVE_TEXTURE_API void UNITY_INTERFACE_API Release(void* texPtr)
{
    if (!texPtr)
    {
        return;
    }

    @autoreleasepool
    {
        if (!UntrackTexture(texPtr))
        {
            LogMessage("Warn", "Release ignored: Metal texture %p was not found", texPtr);
            return;
        }

        (void)(__bridge_transfer id<MTLTexture>)texPtr;
    }
}

NATIVE_TEXTURE_API void UNITY_INTERFACE_API UnityPluginLoad(IUnityInterfaces* unityInterfaces)
{
    IUnityGraphics* unityGraphics = nullptr;
    {
        std::lock_guard<std::mutex> lock(gUnityStateMutex);
        gUnityInterfaces = unityInterfaces;
        gUnityGraphics = unityInterfaces ? unityInterfaces->Get<IUnityGraphics>() : nullptr;
        unityGraphics = gUnityGraphics;
    }

    if (unityGraphics)
    {
        unityGraphics->RegisterDeviceEventCallback(OnGraphicsDeviceEvent);
        OnGraphicsDeviceEvent(kUnityGfxDeviceEventInitialize);
    }
}

NATIVE_TEXTURE_API void UNITY_INTERFACE_API UnityPluginUnload()
{
    IUnityGraphics* unityGraphics = nullptr;
    {
        std::lock_guard<std::mutex> lock(gUnityStateMutex);
        unityGraphics = gUnityGraphics;
    }

    if (unityGraphics)
    {
        unityGraphics->UnregisterDeviceEventCallback(OnGraphicsDeviceEvent);
    }

    std::vector<void*> outstandingTextures = TakeOutstandingTextures();
    if (!outstandingTextures.empty())
    {
        LogMessage("Warn", "Plugin unload with %zu unreleased Metal texture(s)", static_cast<size_t>(outstandingTextures.size()));
        @autoreleasepool
        {
            for (void* texturePtr : outstandingTextures)
            {
                (void)(__bridge_transfer id<MTLTexture>)texturePtr;
            }
        }
    }

    std::lock_guard<std::mutex> lock(gUnityStateMutex);
    ClearMetalInterfacesLocked();
    gUnityGraphics = nullptr;
    gUnityInterfaces = nullptr;
    gFallbackDevice = nil;
}
