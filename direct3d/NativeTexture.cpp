#if !defined(_WIN32)
#error NativeTexture D3D12 backend requires Windows.
#endif

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include "NativeTexture.h"

#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <cctype>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <mutex>
#include <unordered_set>
#include <vector>

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmissing-field-initializers"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image.h"
#include "stb/stb_image_write.h"
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
#include "unity/IUnityGraphics.h"
#include "unity/IUnityGraphicsD3D12.h"

using Microsoft::WRL::ComPtr;

namespace
{
constexpr int kBytesPerPixel = 4;
constexpr int kJpegQuality = 90;

struct D3D12Context
{
    bool valid = false;
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12CommandQueue> commandQueue;
};

std::mutex gUnityStateMutex;
std::mutex gTextureMutex;
#ifndef STBI_THREAD_LOCAL
std::mutex gStbiLoadMutex;
#endif
IUnityInterfaces* gUnityInterfaces = nullptr;
IUnityGraphics* gUnityGraphics = nullptr;
IUnityGraphicsD3D12v8* gUnityGraphicsD3D12v8 = nullptr;
IUnityGraphicsD3D12v7* gUnityGraphicsD3D12v7 = nullptr;
IUnityGraphicsD3D12v6* gUnityGraphicsD3D12v6 = nullptr;
IUnityGraphicsD3D12v5* gUnityGraphicsD3D12v5 = nullptr;
IUnityGraphicsD3D12v4* gUnityGraphicsD3D12v4 = nullptr;
IUnityGraphicsD3D12* gUnityGraphicsD3D12 = nullptr;
ComPtr<ID3D12Device> gD3D12Device;
ComPtr<ID3D12CommandQueue> gD3D12CommandQueue;
std::unordered_set<ID3D12Resource*> gLiveTextures;

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
    fprintf(stderr, "[NativeTexture][D3D12][%s] ", level);
    vfprintf(stderr, format, args);
    fprintf(stderr, "\n");
    va_end(args);
}

void LogHRESULT(const char* operation, HRESULT hr)
{
    LogMessage("Error", "%s failed with HRESULT 0x%08lx", operation, static_cast<unsigned long>(hr));
}

const char* RendererName(UnityGfxRenderer renderer)
{
    switch (renderer)
    {
        case kUnityGfxRendererD3D12:
            return "D3D12";
        case kUnityGfxRendererD3D11:
            return "D3D11";
        case kUnityGfxRendererMetal:
            return "Metal";
        case kUnityGfxRendererVulkan:
            return "Vulkan";
        case kUnityGfxRendererNull:
            return "Null";
        default:
            return "Other";
    }
}

void ClearD3D12InterfacesLocked()
{
    gUnityGraphicsD3D12v8 = nullptr;
    gUnityGraphicsD3D12v7 = nullptr;
    gUnityGraphicsD3D12v6 = nullptr;
    gUnityGraphicsD3D12v5 = nullptr;
    gUnityGraphicsD3D12v4 = nullptr;
    gUnityGraphicsD3D12 = nullptr;
    gD3D12CommandQueue.Reset();
    gD3D12Device.Reset();
}

ID3D12Device* GetUnityD3D12DeviceLocked()
{
    if (gUnityGraphicsD3D12v8)
    {
        return gUnityGraphicsD3D12v8->GetDevice();
    }
    if (gUnityGraphicsD3D12v7)
    {
        return gUnityGraphicsD3D12v7->GetDevice();
    }
    if (gUnityGraphicsD3D12v6)
    {
        return gUnityGraphicsD3D12v6->GetDevice();
    }
    if (gUnityGraphicsD3D12v5)
    {
        return gUnityGraphicsD3D12v5->GetDevice();
    }
    if (gUnityGraphicsD3D12v4)
    {
        return gUnityGraphicsD3D12v4->GetDevice();
    }
    if (gUnityGraphicsD3D12)
    {
        return gUnityGraphicsD3D12->GetDevice();
    }

    return nullptr;
}

ID3D12CommandQueue* GetUnityD3D12CommandQueueLocked()
{
    if (gUnityGraphicsD3D12v8)
    {
        return gUnityGraphicsD3D12v8->GetCommandQueue();
    }
    if (gUnityGraphicsD3D12v7)
    {
        return gUnityGraphicsD3D12v7->GetCommandQueue();
    }
    if (gUnityGraphicsD3D12v6)
    {
        return gUnityGraphicsD3D12v6->GetCommandQueue();
    }
    if (gUnityGraphicsD3D12v5)
    {
        return gUnityGraphicsD3D12v5->GetCommandQueue();
    }
    if (gUnityGraphicsD3D12v4)
    {
        return gUnityGraphicsD3D12v4->GetCommandQueue();
    }
    if (gUnityGraphicsD3D12)
    {
        return gUnityGraphicsD3D12->GetCommandQueue();
    }

    return nullptr;
}

void RefreshD3D12InterfacesLocked()
{
    ClearD3D12InterfacesLocked();
    if (!gUnityInterfaces || !gUnityGraphics)
    {
        return;
    }

    if (gUnityGraphics->GetRenderer() != kUnityGfxRendererD3D12)
    {
        return;
    }

    gUnityGraphicsD3D12v8 = gUnityInterfaces->Get<IUnityGraphicsD3D12v8>();
    if (!gUnityGraphicsD3D12v8)
    {
        gUnityGraphicsD3D12v7 = gUnityInterfaces->Get<IUnityGraphicsD3D12v7>();
    }
    if (!gUnityGraphicsD3D12v8 && !gUnityGraphicsD3D12v7)
    {
        gUnityGraphicsD3D12v6 = gUnityInterfaces->Get<IUnityGraphicsD3D12v6>();
    }
    if (!gUnityGraphicsD3D12v8 && !gUnityGraphicsD3D12v7 && !gUnityGraphicsD3D12v6)
    {
        gUnityGraphicsD3D12v5 = gUnityInterfaces->Get<IUnityGraphicsD3D12v5>();
    }
    if (!gUnityGraphicsD3D12v8 && !gUnityGraphicsD3D12v7 && !gUnityGraphicsD3D12v6 && !gUnityGraphicsD3D12v5)
    {
        gUnityGraphicsD3D12v4 = gUnityInterfaces->Get<IUnityGraphicsD3D12v4>();
    }
    if (!gUnityGraphicsD3D12v8 && !gUnityGraphicsD3D12v7 && !gUnityGraphicsD3D12v6 &&
        !gUnityGraphicsD3D12v5 && !gUnityGraphicsD3D12v4)
    {
        gUnityGraphicsD3D12 = gUnityInterfaces->Get<IUnityGraphicsD3D12>();
    }

    ID3D12Device* device = GetUnityD3D12DeviceLocked();
    if (!device)
    {
        LogMessage("Error", "D3D12 device is not ready");
        return;
    }

    ID3D12CommandQueue* commandQueue = GetUnityD3D12CommandQueueLocked();
    if (!commandQueue)
    {
        LogMessage("Error", "D3D12 command queue is unavailable; NativeTexture requires a Unity D3D12 interface that exposes GetCommandQueue");
        return;
    }

    gD3D12Device = device;
    gD3D12CommandQueue = commandQueue;
}

void TrackTexture(ID3D12Resource* texture)
{
    if (!texture)
    {
        return;
    }

    std::lock_guard<std::mutex> lock(gTextureMutex);
    gLiveTextures.insert(texture);
}

bool UntrackTexture(ID3D12Resource* texture)
{
    std::lock_guard<std::mutex> lock(gTextureMutex);
    return gLiveTextures.erase(texture) > 0;
}

std::vector<ID3D12Resource*> TakeOutstandingTextures()
{
    std::lock_guard<std::mutex> lock(gTextureMutex);

    std::vector<ID3D12Resource*> textures;
    textures.reserve(gLiveTextures.size());
    for (ID3D12Resource* texture : gLiveTextures)
    {
        textures.push_back(texture);
    }

    gLiveTextures.clear();
    return textures;
}

void DrainOutstandingTextures(const char* operation)
{
    std::vector<ID3D12Resource*> textures = TakeOutstandingTextures();
    if (textures.empty())
    {
        return;
    }

    LogMessage("Warn", "%s with %zu unreleased D3D12 texture(s)", operation, static_cast<size_t>(textures.size()));
    for (ID3D12Resource* texture : textures)
    {
        texture->Release();
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

bool SaveTextureToFile(const char* fileName, uint8_t* rgba, int width, int height)
{
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

bool SnapshotD3D12Context(D3D12Context* outContext)
{
    if (!outContext)
    {
        return false;
    }

    *outContext = {};

    std::lock_guard<std::mutex> lock(gUnityStateMutex);
    if (!gD3D12Device || !gD3D12CommandQueue)
    {
        LogMessage("Error", "D3D12 context is not ready");
        return false;
    }

    outContext->valid = true;
    outContext->device = gD3D12Device;
    outContext->commandQueue = gD3D12CommandQueue;
    return true;
}

bool WaitForFence(ID3D12Fence* fence, UINT64 fenceValue, const char* operation)
{
    if (!fence)
    {
        LogMessage("Error", "%s failed: fence is null", operation);
        return false;
    }

    if (fence->GetCompletedValue() >= fenceValue)
    {
        return true;
    }

    HANDLE eventHandle = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!eventHandle)
    {
        LogMessage("Error", "%s failed: CreateEventW returned null", operation);
        return false;
    }

    const HRESULT result = fence->SetEventOnCompletion(fenceValue, eventHandle);
    if (FAILED(result))
    {
        CloseHandle(eventHandle);
        LogHRESULT(operation, result);
        return false;
    }

    const DWORD waitResult = WaitForSingleObject(eventHandle, INFINITE);
    CloseHandle(eventHandle);
    if (waitResult != WAIT_OBJECT_0)
    {
        LogMessage("Error", "%s failed: WaitForSingleObject returned %lu", operation, static_cast<unsigned long>(waitResult));
        return false;
    }

    return true;
}

bool CreateTextureInternal(const D3D12Context& context,
                           unsigned char* rgba,
                           int width,
                           int height,
                           bool useSrgb,
                           ID3D12Resource** outTexture)
{
    if (!outTexture)
    {
        LogMessage("Error", "Create failed: outTexture is null");
        return false;
    }

    *outTexture = nullptr;
    if (!ValidateImageArgs(rgba, width, height, "Create"))
    {
        return false;
    }
    if (!context.valid || !context.device || !context.commandQueue)
    {
        LogMessage("Error", "Create failed: D3D12 context is invalid");
        return false;
    }

    const DXGI_FORMAT textureFormat = useSrgb ? DXGI_FORMAT_R8G8B8A8_UNORM_SRGB : DXGI_FORMAT_R8G8B8A8_UNORM;
    const UINT64 sourceRowBytes = static_cast<UINT64>(width) * kBytesPerPixel;
    const D3D12_RESOURCE_STATES shaderReadState =
        D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

    D3D12_HEAP_PROPERTIES defaultHeapProps = {};
    defaultHeapProps.Type = D3D12_HEAP_TYPE_DEFAULT;
    defaultHeapProps.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    defaultHeapProps.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    defaultHeapProps.CreationNodeMask = 1;
    defaultHeapProps.VisibleNodeMask = 1;

    D3D12_RESOURCE_DESC textureDesc = {};
    textureDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    textureDesc.Alignment = 0;
    textureDesc.Width = static_cast<UINT64>(width);
    textureDesc.Height = static_cast<UINT>(height);
    textureDesc.DepthOrArraySize = 1;
    textureDesc.MipLevels = 1;
    textureDesc.Format = textureFormat;
    textureDesc.SampleDesc.Count = 1;
    textureDesc.SampleDesc.Quality = 0;
    textureDesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    textureDesc.Flags = D3D12_RESOURCE_FLAG_NONE;

    ComPtr<ID3D12Resource> texture;
    HRESULT result = context.device->CreateCommittedResource(
        &defaultHeapProps,
        D3D12_HEAP_FLAG_NONE,
        &textureDesc,
        D3D12_RESOURCE_STATE_COPY_DEST,
        nullptr,
        IID_PPV_ARGS(texture.GetAddressOf()));
    if (FAILED(result))
    {
        LogHRESULT("CreateCommittedResource(texture)", result);
        return false;
    }

    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint = {};
    UINT numRows = 0;
    UINT64 rowSizeInBytes = 0;
    UINT64 uploadBufferSize = 0;
    context.device->GetCopyableFootprints(
        &textureDesc,
        0,
        1,
        0,
        &footprint,
        &numRows,
        &rowSizeInBytes,
        &uploadBufferSize);

    D3D12_HEAP_PROPERTIES uploadHeapProps = {};
    uploadHeapProps.Type = D3D12_HEAP_TYPE_UPLOAD;
    uploadHeapProps.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    uploadHeapProps.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    uploadHeapProps.CreationNodeMask = 1;
    uploadHeapProps.VisibleNodeMask = 1;

    D3D12_RESOURCE_DESC uploadDesc = {};
    uploadDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    uploadDesc.Alignment = 0;
    uploadDesc.Width = uploadBufferSize;
    uploadDesc.Height = 1;
    uploadDesc.DepthOrArraySize = 1;
    uploadDesc.MipLevels = 1;
    uploadDesc.Format = DXGI_FORMAT_UNKNOWN;
    uploadDesc.SampleDesc.Count = 1;
    uploadDesc.SampleDesc.Quality = 0;
    uploadDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    uploadDesc.Flags = D3D12_RESOURCE_FLAG_NONE;

    ComPtr<ID3D12Resource> uploadBuffer;
    result = context.device->CreateCommittedResource(
        &uploadHeapProps,
        D3D12_HEAP_FLAG_NONE,
        &uploadDesc,
        D3D12_RESOURCE_STATE_GENERIC_READ,
        nullptr,
        IID_PPV_ARGS(uploadBuffer.GetAddressOf()));
    if (FAILED(result))
    {
        LogHRESULT("CreateCommittedResource(upload)", result);
        return false;
    }

    UINT8* mapped = nullptr;
    D3D12_RANGE readRange = {0, 0};
    result = uploadBuffer->Map(0, &readRange, reinterpret_cast<void**>(&mapped));
    if (FAILED(result))
    {
        LogHRESULT("Map(upload)", result);
        return false;
    }

    for (UINT row = 0; row < numRows; ++row)
    {
        std::memcpy(
            mapped + footprint.Offset + static_cast<SIZE_T>(row) * footprint.Footprint.RowPitch,
            rgba + static_cast<size_t>(row) * static_cast<size_t>(sourceRowBytes),
            static_cast<size_t>(sourceRowBytes));
    }

    D3D12_RANGE writtenRange = {
        static_cast<SIZE_T>(footprint.Offset),
        static_cast<SIZE_T>(footprint.Offset + static_cast<UINT64>(footprint.Footprint.RowPitch) * numRows)};
    uploadBuffer->Unmap(0, &writtenRange);

    ComPtr<ID3D12CommandAllocator> commandAllocator;
    result = context.device->CreateCommandAllocator(
        D3D12_COMMAND_LIST_TYPE_DIRECT,
        IID_PPV_ARGS(commandAllocator.GetAddressOf()));
    if (FAILED(result))
    {
        LogHRESULT("CreateCommandAllocator", result);
        return false;
    }

    ComPtr<ID3D12GraphicsCommandList> commandList;
    result = context.device->CreateCommandList(
        0,
        D3D12_COMMAND_LIST_TYPE_DIRECT,
        commandAllocator.Get(),
        nullptr,
        IID_PPV_ARGS(commandList.GetAddressOf()));
    if (FAILED(result))
    {
        LogHRESULT("CreateCommandList", result);
        return false;
    }

    D3D12_TEXTURE_COPY_LOCATION destination = {};
    destination.pResource = texture.Get();
    destination.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    destination.SubresourceIndex = 0;

    D3D12_TEXTURE_COPY_LOCATION source = {};
    source.pResource = uploadBuffer.Get();
    source.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    source.PlacedFootprint = footprint;

    commandList->CopyTextureRegion(&destination, 0, 0, 0, &source, nullptr);

    D3D12_RESOURCE_BARRIER barrier = {};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    barrier.Transition.pResource = texture.Get();
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
    barrier.Transition.StateAfter = shaderReadState;
    commandList->ResourceBarrier(1, &barrier);

    result = commandList->Close();
    if (FAILED(result))
    {
        LogHRESULT("Close(commandList)", result);
        return false;
    }

    ID3D12CommandList* commandLists[] = {commandList.Get()};
    context.commandQueue->ExecuteCommandLists(1, commandLists);

    ComPtr<ID3D12Fence> fence;
    result = context.device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(fence.GetAddressOf()));
    if (FAILED(result))
    {
        LogHRESULT("CreateFence", result);
        return false;
    }

    const UINT64 fenceValue = 1;
    result = context.commandQueue->Signal(fence.Get(), fenceValue);
    if (FAILED(result))
    {
        LogHRESULT("CommandQueue::Signal", result);
        return false;
    }

    if (!WaitForFence(fence.Get(), fenceValue, "WaitForFence(Create)"))
    {
        return false;
    }

    *outTexture = texture.Detach();
    return true;
}

void UNITY_INTERFACE_API OnGraphicsDeviceEvent(UnityGfxDeviceEventType eventType)
{
    switch (eventType)
    {
        case kUnityGfxDeviceEventInitialize:
        case kUnityGfxDeviceEventAfterReset:
        {
            std::lock_guard<std::mutex> lock(gUnityStateMutex);
            RefreshD3D12InterfacesLocked();
            break;
        }
        case kUnityGfxDeviceEventBeforeReset:
        case kUnityGfxDeviceEventShutdown:
        {
            DrainOutstandingTextures("Graphics device shutdown");

            std::lock_guard<std::mutex> lock(gUnityStateMutex);
            ClearD3D12InterfacesLocked();
            break;
        }
        default:
            break;
    }
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

    return SaveTextureToFile(fileName, rgba, width, height);
}

NATIVE_TEXTURE_API void* UNITY_INTERFACE_API Create(unsigned char* rgba, int width, int height)
{
    return CreateWithOptions(rgba, width, height, false);
}

NATIVE_TEXTURE_API void* UNITY_INTERFACE_API CreateWithOptions(unsigned char* rgba, int width, int height, bool useSrgb)
{
    D3D12Context context = {};
    if (!SnapshotD3D12Context(&context))
    {
        return nullptr;
    }

    ID3D12Resource* texture = nullptr;
    if (!CreateTextureInternal(context, rgba, width, height, useSrgb, &texture))
    {
        return nullptr;
    }

    TrackTexture(texture);
    return texture;
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

    void* texture = CreateWithOptions(rgba, *width, *height, useSrgb);
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

    void* texture = CreateWithOptions(rgba, *width, *height, useSrgb);
    stbi_image_free(rgba);
    return texture;
}

NATIVE_TEXTURE_API void UNITY_INTERFACE_API Release(void* texPtr)
{
    if (!texPtr)
    {
        return;
    }

    ID3D12Resource* texture = static_cast<ID3D12Resource*>(texPtr);
    if (!UntrackTexture(texture))
    {
        LogMessage("Warn", "Release ignored: D3D12 texture %p was not found", texPtr);
        return;
    }

    texture->Release();
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

    DrainOutstandingTextures("Plugin unload");

    std::lock_guard<std::mutex> lock(gUnityStateMutex);
    ClearD3D12InterfacesLocked();
    gUnityGraphics = nullptr;
    gUnityInterfaces = nullptr;
}
