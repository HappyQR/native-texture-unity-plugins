#include "NativeTexture.h"

#include <condition_variable>
#include <cctype>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <mutex>
#include <new>
#include <string>
#include <unordered_map>
#include <vector>

#define VK_NO_PROTOTYPES
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmissing-field-initializers"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image.h"
#include "stb/stb_image_write.h"
#pragma clang diagnostic pop
#include "unity/IUnityGraphics.h"
#include "unity/IUnityGraphicsVulkan.h"

#define NATIVE_TEXTURE_API extern "C" UNITY_INTERFACE_EXPORT

namespace
{
constexpr int kBytesPerPixel = 4;
constexpr int kJpegQuality = 90;
// Event ID for the render-thread "drain pending uploads" callback fired from C#
// via `GL.IssuePluginEvent(GetRenderEventFunc(), kEventIdSubmitUploads)`. The
// event is registered with ConfigureEvent during device init so the callback
// runs outside a render pass with graphics-queue access enabled, which is what
// vkQueueSubmit on a transient transfer command buffer requires.
constexpr int kEventIdSubmitUploads = 1;

struct VulkanFunctions
{
    PFN_vkGetDeviceProcAddr GetDeviceProcAddr = nullptr;
    PFN_vkGetPhysicalDeviceMemoryProperties GetPhysicalDeviceMemoryProperties = nullptr;
    PFN_vkCreateImage CreateImage = nullptr;
    PFN_vkDestroyImage DestroyImage = nullptr;
    PFN_vkGetImageMemoryRequirements GetImageMemoryRequirements = nullptr;
    PFN_vkAllocateMemory AllocateMemory = nullptr;
    PFN_vkFreeMemory FreeMemory = nullptr;
    PFN_vkBindImageMemory BindImageMemory = nullptr;
    PFN_vkMapMemory MapMemory = nullptr;
    PFN_vkUnmapMemory UnmapMemory = nullptr;
    PFN_vkFlushMappedMemoryRanges FlushMappedMemoryRanges = nullptr;
    PFN_vkCreateBuffer CreateBuffer = nullptr;
    PFN_vkDestroyBuffer DestroyBuffer = nullptr;
    PFN_vkGetBufferMemoryRequirements GetBufferMemoryRequirements = nullptr;
    PFN_vkBindBufferMemory BindBufferMemory = nullptr;
    PFN_vkCreateCommandPool CreateCommandPool = nullptr;
    PFN_vkDestroyCommandPool DestroyCommandPool = nullptr;
    PFN_vkAllocateCommandBuffers AllocateCommandBuffers = nullptr;
    PFN_vkFreeCommandBuffers FreeCommandBuffers = nullptr;
    PFN_vkBeginCommandBuffer BeginCommandBuffer = nullptr;
    PFN_vkEndCommandBuffer EndCommandBuffer = nullptr;
    PFN_vkCmdPipelineBarrier CmdPipelineBarrier = nullptr;
    PFN_vkCmdCopyBufferToImage CmdCopyBufferToImage = nullptr;
    PFN_vkCreateFence CreateFence = nullptr;
    PFN_vkDestroyFence DestroyFence = nullptr;
    PFN_vkQueueSubmit QueueSubmit = nullptr;
    PFN_vkWaitForFences WaitForFences = nullptr;
    PFN_vkGetFenceStatus GetFenceStatus = nullptr;
    PFN_vkQueueWaitIdle QueueWaitIdle = nullptr;
};

struct VulkanContext
{
    bool valid = false;
    UnityVulkanInstance instance = {};
    VulkanFunctions functions = {};
};

struct VulkanTextureResource
{
    VkImage* externalImage = nullptr;
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VulkanContext context = {};
    uint32_t width = 0;
    uint32_t height = 0;
};

struct QueueRequestState
{
    std::mutex mutex;
    std::condition_variable condition;
    bool completed = false;
    bool success = false;
    std::string error;
};

// CreateTextureRequest and its render-thread callback (CreateTextureQueueCallback /
// CreateTextureOnQueue) were the sync upload path before the worker/render-thread
// split. Removed when CreateWithOptions switched to PrepareUploadOnWorker +
// SubmitPendingUploadsCallback (Phase A + Phase B/C). Release path still uses
// ReleaseTextureRequest because vkDestroyImage must run with queue access.

struct ReleaseTextureRequest
{
    QueueRequestState state;
    VulkanContext context;
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
};

// Phase A output (worker thread, no Vulkan command submission): VkImage + staging
// buffer/memory created and populated, awaiting render-thread command submission.
// The image is in VK_IMAGE_LAYOUT_UNDEFINED; the staging buffer holds the RGBA bytes
// flushed from host memory.
struct PendingUpload
{
    VulkanContext context;
    VkImage image = VK_NULL_HANDLE;
    VkBuffer stagingBuffer = VK_NULL_HANDLE;
    VkDeviceMemory stagingMemory = VK_NULL_HANDLE;
    uint32_t width = 0;
    uint32_t height = 0;
};

// Phase B output (render thread, after vkQueueSubmit but before fence signal): the
// VkImage now has commands in flight; staging buffer must stay alive until the fence
// signals. Phase C scans this list each render-event tick and frees the staging
// resources for fences that have signaled.
struct InFlightUpload
{
    VulkanContext context;
    VkImage image = VK_NULL_HANDLE;
    VkBuffer stagingBuffer = VK_NULL_HANDLE;
    VkDeviceMemory stagingMemory = VK_NULL_HANDLE;
    VkCommandPool commandPool = VK_NULL_HANDLE;
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
};

// A user-released texture awaiting fence-gated destruction. Release() enqueues these
// (main thread, no GPU stall); the render-thread upload callback arms a graphics-queue
// fence for each, then frees image+memory once that fence signals — proving the GPU has
// retired Unity's last sampling/blit of the image. Replaces the old per-release
// vkQueueWaitIdle(graphicsQueue) full-queue host stall.
struct PendingDestroy
{
    VulkanContext context;
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImage* externalImage = nullptr;
    VkFence fence = VK_NULL_HANDLE;
};

using TextureKey = uint64_t;

// ---- Pano GPU tile assembly (region-copy) -------------------------------
// Replaces the CPU 134MB stitch buffer: each tile is decoded to a small (~1MB)
// host-visible staging slot (row-reversed for the bottom-up flip), then ONE
// batched vkCmdCopyBufferToImage per pano copies every tile region into a
// persistent device image. Peak drops from ~4x134MB to ~2x134MB.

// Reusable host-visible staging buffer. Returned to gStagingFreeList after the
// batch's fence signals (never a full-size 134MB staging; no per-tile
// vkAllocateMemory after warmup — P1-9).
struct StagingSlot
{
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize capacity = 0;   // vkCreateBuffer size — reuse requires need <= capacity
    bool coherent = false;
};

// In-progress assembly: created by BeginPanoAssembly, appended by UploadTileRegion
// (concurrent worker threads, guarded by gAssembliesMutex), finalized by
// FinishPanoAssembly which moves it to gPendingAssemblies. The device image is
// also registered in gTextureResources so Release() finds it.
struct AssemblyImage
{
    VulkanContext context;
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory imageMemory = VK_NULL_HANDLE;
    VkImage* externalImage = nullptr;
    uint32_t width = 0;
    uint32_t height = 0;
    std::vector<VkBufferImageCopy> regions;
    std::vector<StagingSlot> stagings;   // parallel to regions
};

// Committed assembly awaiting the render-thread batched submit (Phase B).
struct PendingAssembly
{
    VulkanContext context;
    VkImage image = VK_NULL_HANDLE;
    std::vector<VkBufferImageCopy> regions;
    std::vector<StagingSlot> stagings;
};

// Submitted assembly awaiting its fence. Its stagings return to the ring (not
// destroyed) once the fence signals.
struct InFlightAssembly
{
    VulkanContext context;
    VkImage image = VK_NULL_HANDLE;
    VkCommandPool commandPool = VK_NULL_HANDLE;
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    std::vector<StagingSlot> stagings;
};

std::mutex gUnityStateMutex;
std::mutex gQueueAccessMutex;
std::mutex gTextureMutex;
// Protects gPendingUploads + gInFlightUploads. Held briefly to push/take entries;
// never held across Vulkan command recording / submission (those can be slow on
// device-busy frames and would block the worker thread that's about to enqueue
// the NEXT pending upload).
std::mutex gUploadsMutex;
std::vector<PendingUpload> gPendingUploads;
std::vector<InFlightUpload> gInFlightUploads;
std::vector<PendingDestroy> gPendingDestroys;
#ifndef STBI_THREAD_LOCAL
std::mutex gStbiLoadMutex;
#endif
IUnityInterfaces* gUnityInterfaces = nullptr;
IUnityGraphics* gUnityGraphics = nullptr;
IUnityGraphicsVulkanV2* gUnityGraphicsVulkanV2 = nullptr;
IUnityGraphicsVulkan* gUnityGraphicsVulkan = nullptr;
VulkanContext gVulkanContext = {};
std::unordered_map<TextureKey, VulkanTextureResource> gTextureResources;

// Pano assembly state. gAssemblies is worker-thread only (Begin/Upload/Finish/
// Abort), guarded by gAssembliesMutex. gPendingAssemblies / gInFlightAssemblies /
// gStagingFreeList are shared with the render thread and guarded by gUploadsMutex
// (same discipline as gPendingUploads). Lock order when both are held:
// gTextureMutex -> gAssembliesMutex -> gUploadsMutex.
std::mutex gAssembliesMutex;
std::unordered_map<TextureKey, AssemblyImage> gAssemblies;
std::vector<PendingAssembly> gPendingAssemblies;
std::vector<InFlightAssembly> gInFlightAssemblies;
std::vector<StagingSlot> gStagingFreeList;

bool ExecuteQueueRequest(QueueRequestState& state,
                         const VulkanContext& context,
                         UnityRenderingEventAndData callback,
                         void* userData);
void UNITY_INTERFACE_API ReleaseTextureQueueCallback(int eventId, void* userData);
void EnqueuePendingDestroy(const VulkanContext& context, const VulkanTextureResource& resource);
void ProcessPendingDestroys();
void FlushPendingDestroysBlocking();
void ReturnStagingSlotLocked(const StagingSlot& slot);
void DestroyStagingSlotDirect(const VulkanContext& context, const StagingSlot& slot);
void DrainCompletedAssembliesLocked();
void SubmitPendingAssemblies();
void DestroyAssemblyResourcesBlocking();

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
    fprintf(stderr, "[NativeTexture][Vulkan][%s] ", level);
    vfprintf(stderr, format, args);
    fprintf(stderr, "\n");
    va_end(args);
}

const char* RendererName(UnityGfxRenderer renderer)
{
    switch (renderer)
    {
        case kUnityGfxRendererVulkan:
            return "Vulkan";
        case kUnityGfxRendererMetal:
            return "Metal";
        case kUnityGfxRendererD3D12:
            return "D3D12";
        case kUnityGfxRendererNull:
            return "Null";
        default:
            return "Other";
    }
}

void CompleteRequest(QueueRequestState& state, bool success, const std::string& error)
{
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        state.completed = true;
        state.success = success;
        state.error = error;
    }
    state.condition.notify_one();
}

TextureKey TextureKeyFromExternalImage(const VkImage* externalImage)
{
    return static_cast<TextureKey>(reinterpret_cast<uintptr_t>(externalImage));
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

bool LoadVulkanFunctions(const UnityVulkanInstance& instance, VulkanFunctions* outFunctions)
{
    if (!outFunctions || !instance.getInstanceProcAddr || instance.instance == VK_NULL_HANDLE || instance.device == VK_NULL_HANDLE)
    {
        return false;
    }

    VulkanFunctions functions = {};
    functions.GetDeviceProcAddr = reinterpret_cast<PFN_vkGetDeviceProcAddr>(
        instance.getInstanceProcAddr(instance.instance, "vkGetDeviceProcAddr"));
    functions.GetPhysicalDeviceMemoryProperties = reinterpret_cast<PFN_vkGetPhysicalDeviceMemoryProperties>(
        instance.getInstanceProcAddr(instance.instance, "vkGetPhysicalDeviceMemoryProperties"));

    if (!functions.GetDeviceProcAddr || !functions.GetPhysicalDeviceMemoryProperties)
    {
        return false;
    }

#define LOAD_DEVICE_FN(field, symbol) \
    functions.field = reinterpret_cast<PFN_##symbol>(functions.GetDeviceProcAddr(instance.device, #symbol)); \
    if (!functions.field) { return false; }

    LOAD_DEVICE_FN(CreateImage, vkCreateImage);
    LOAD_DEVICE_FN(DestroyImage, vkDestroyImage);
    LOAD_DEVICE_FN(GetImageMemoryRequirements, vkGetImageMemoryRequirements);
    LOAD_DEVICE_FN(AllocateMemory, vkAllocateMemory);
    LOAD_DEVICE_FN(FreeMemory, vkFreeMemory);
    LOAD_DEVICE_FN(BindImageMemory, vkBindImageMemory);
    LOAD_DEVICE_FN(MapMemory, vkMapMemory);
    LOAD_DEVICE_FN(UnmapMemory, vkUnmapMemory);
    LOAD_DEVICE_FN(FlushMappedMemoryRanges, vkFlushMappedMemoryRanges);
    LOAD_DEVICE_FN(CreateBuffer, vkCreateBuffer);
    LOAD_DEVICE_FN(DestroyBuffer, vkDestroyBuffer);
    LOAD_DEVICE_FN(GetBufferMemoryRequirements, vkGetBufferMemoryRequirements);
    LOAD_DEVICE_FN(BindBufferMemory, vkBindBufferMemory);
    LOAD_DEVICE_FN(CreateCommandPool, vkCreateCommandPool);
    LOAD_DEVICE_FN(DestroyCommandPool, vkDestroyCommandPool);
    LOAD_DEVICE_FN(AllocateCommandBuffers, vkAllocateCommandBuffers);
    LOAD_DEVICE_FN(FreeCommandBuffers, vkFreeCommandBuffers);
    LOAD_DEVICE_FN(BeginCommandBuffer, vkBeginCommandBuffer);
    LOAD_DEVICE_FN(EndCommandBuffer, vkEndCommandBuffer);
    LOAD_DEVICE_FN(CmdPipelineBarrier, vkCmdPipelineBarrier);
    LOAD_DEVICE_FN(CmdCopyBufferToImage, vkCmdCopyBufferToImage);
    LOAD_DEVICE_FN(CreateFence, vkCreateFence);
    LOAD_DEVICE_FN(DestroyFence, vkDestroyFence);
    LOAD_DEVICE_FN(QueueSubmit, vkQueueSubmit);
    LOAD_DEVICE_FN(WaitForFences, vkWaitForFences);
    LOAD_DEVICE_FN(GetFenceStatus, vkGetFenceStatus);
    LOAD_DEVICE_FN(QueueWaitIdle, vkQueueWaitIdle);

#undef LOAD_DEVICE_FN

    *outFunctions = functions;
    return true;
}

void ClearVulkanInterfacesLocked()
{
    gUnityGraphicsVulkanV2 = nullptr;
    gUnityGraphicsVulkan = nullptr;
    gVulkanContext = {};
}

void RefreshVulkanContextLocked()
{
    ClearVulkanInterfacesLocked();
    if (!gUnityInterfaces || !gUnityGraphics)
    {
        return;
    }

    if (gUnityGraphics->GetRenderer() != kUnityGfxRendererVulkan)
    {
        return;
    }

    gUnityGraphicsVulkanV2 = gUnityInterfaces->Get<IUnityGraphicsVulkanV2>();
    if (!gUnityGraphicsVulkanV2)
    {
        gUnityGraphicsVulkan = gUnityInterfaces->Get<IUnityGraphicsVulkan>();
    }

    UnityVulkanInstance instance = {};
    if (gUnityGraphicsVulkanV2)
    {
        instance = gUnityGraphicsVulkanV2->Instance();
    }
    else if (gUnityGraphicsVulkan)
    {
        instance = gUnityGraphicsVulkan->Instance();
    }
    else
    {
        return;
    }

    VulkanFunctions functions = {};
    if (!LoadVulkanFunctions(instance, &functions))
    {
        LogMessage("Error", "Failed to load Vulkan function table from Unity");
        return;
    }

    gVulkanContext.valid = true;
    gVulkanContext.instance = instance;
    gVulkanContext.functions = functions;

    // Register kEventIdSubmitUploads with Unity so that when C# fires
    // `GL.IssuePluginEvent(GetRenderEventFunc(), kEventIdSubmitUploads)`, the
    // render-thread callback (OnRenderEvent) runs with:
    //  - kUnityVulkanRenderPass_EnsureOutside: render pass closed before the
    //    callback so we can record image layout transitions + copy commands.
    //  - kUnityVulkanGraphicsQueueAccess_Allow: callback can submit to the
    //    graphics queue (vkQueueSubmit). Without this, the queue access is
    //    blocked at the Unity side and our submit silently fails.
    //  - kUnityVulkanEventConfigFlag_EnsurePreviousFrameSubmission: default;
    //    waits for Unity's prior command-buffer submission before running our
    //    callback so the render thread is at a known state.
    // ConfigureEvent only works on V2; V1 will fall back to AccessQueue path
    // for the rare device that doesn't expose V2.
    if (gUnityGraphicsVulkanV2)
    {
        UnityVulkanPluginEventConfig eventConfig = {};
        eventConfig.renderPassPrecondition = kUnityVulkanRenderPass_EnsureOutside;
        eventConfig.graphicsQueueAccess = kUnityVulkanGraphicsQueueAccess_Allow;
        eventConfig.flags = kUnityVulkanEventConfigFlag_EnsurePreviousFrameSubmission;
        gUnityGraphicsVulkanV2->ConfigureEvent(kEventIdSubmitUploads, &eventConfig);
    }
}

bool SnapshotVulkanContext(VulkanContext* outContext)
{
    std::lock_guard<std::mutex> lock(gUnityStateMutex);
    if (gUnityGraphics)
    {
        const UnityGfxRenderer renderer = gUnityGraphics->GetRenderer();
        if (renderer != kUnityGfxRendererVulkan)
        {
            LogMessage("Error", "Create failed: Unity renderer is %s (%d), Vulkan path cannot create textures",
                       RendererName(renderer), renderer);
            return false;
        }
    }

    RefreshVulkanContextLocked();
    if (!gVulkanContext.valid)
    {
        LogMessage("Error", "Vulkan context is not ready");
        return false;
    }

    *outContext = gVulkanContext;
    return true;
}

std::vector<VulkanTextureResource> TakeOutstandingTextureResources()
{
    std::lock_guard<std::mutex> lock(gTextureMutex);

    std::vector<VulkanTextureResource> resources;
    resources.reserve(gTextureResources.size());
    for (const auto& entry : gTextureResources)
    {
        resources.push_back(entry.second);
    }

    gTextureResources.clear();
    return resources;
}

bool DestroyTextureResourceOnQueue(const VulkanContext& context,
                                   const VulkanTextureResource& resource,
                                   const char* operation)
{
    if (!context.valid)
    {
        return false;
    }

    if (resource.image == VK_NULL_HANDLE || resource.memory == VK_NULL_HANDLE)
    {
        return true;
    }

    ReleaseTextureRequest request = {};
    request.context = context;
    request.image = resource.image;
    request.memory = resource.memory;

    if (!ExecuteQueueRequest(request.state, context, ReleaseTextureQueueCallback, &request))
    {
        LogMessage("Error", "%s failed: %s",
                   operation,
                   request.state.error.empty() ? "unknown queue access error" : request.state.error.c_str());
        return false;
    }

    return true;
}

bool DestroyTextureResourceDirect(const VulkanTextureResource& resource, const char* operation)
{
    const VulkanContext& context = resource.context;
    if (!context.valid || context.instance.device == VK_NULL_HANDLE)
    {
        return false;
    }

    if (!context.functions.DestroyImage || !context.functions.FreeMemory)
    {
        return false;
    }

    if (context.instance.graphicsQueue != VK_NULL_HANDLE && context.functions.QueueWaitIdle)
    {
        const VkResult result = context.functions.QueueWaitIdle(context.instance.graphicsQueue);
        if (result != VK_SUCCESS)
        {
            LogMessage("Error", "%s fallback failed: vkQueueWaitIdle returned VkResult %d",
                       operation,
                       static_cast<int>(result));
            return false;
        }
    }

    if (resource.image != VK_NULL_HANDLE)
    {
        context.functions.DestroyImage(context.instance.device, resource.image, nullptr);
    }
    if (resource.memory != VK_NULL_HANDLE)
    {
        context.functions.FreeMemory(context.instance.device, resource.memory, nullptr);
    }

    return true;
}

void DestroyTextureResourceBestEffort(const VulkanTextureResource& resource,
                                      const VulkanContext* preferredContext,
                                      const char* operation)
{
    bool destroyed = false;
    if (preferredContext && preferredContext->valid)
    {
        destroyed = DestroyTextureResourceOnQueue(*preferredContext, resource, operation);
    }

    if (!destroyed)
    {
        destroyed = DestroyTextureResourceDirect(resource, operation);
    }

    if (!destroyed)
    {
        LogMessage("Warn", "%s leaked Vulkan texture %p because no valid destruction path was available",
                   operation,
                   resource.externalImage);
    }

    delete resource.externalImage;
}

void DrainOutstandingTextures(const char* operation, const VulkanContext* preferredContext)
{
    std::vector<VulkanTextureResource> resources = TakeOutstandingTextureResources();
    if (resources.empty())
    {
        return;
    }

    LogMessage("Warn", "%s with %zu unreleased Vulkan texture(s)",
               operation,
               static_cast<size_t>(resources.size()));

    for (const VulkanTextureResource& resource : resources)
    {
        DestroyTextureResourceBestEffort(resource, preferredContext, operation);
    }
}

void UNITY_INTERFACE_API OnGraphicsDeviceEvent(UnityGfxDeviceEventType eventType)
{
    switch (eventType)
    {
        case kUnityGfxDeviceEventInitialize:
        case kUnityGfxDeviceEventAfterReset:
        {
            std::lock_guard<std::mutex> lock(gUnityStateMutex);
            RefreshVulkanContextLocked();
            break;
        }
        case kUnityGfxDeviceEventBeforeReset:
        case kUnityGfxDeviceEventShutdown:
        {
            FlushPendingDestroysBlocking();
            DestroyAssemblyResourcesBlocking();
            DrainOutstandingTextures("Graphics device shutdown", nullptr);

            std::lock_guard<std::mutex> lock(gUnityStateMutex);
            ClearVulkanInterfacesLocked();
            break;
        }
        default:
            break;
    }
}

bool FindMemoryType(const VulkanContext& context,
                    uint32_t typeBits,
                    VkMemoryPropertyFlags requiredFlags,
                    VkMemoryPropertyFlags preferredFlags,
                    uint32_t* outMemoryTypeIndex,
                    VkMemoryPropertyFlags* outFlags)
{
    VkPhysicalDeviceMemoryProperties properties = {};
    context.functions.GetPhysicalDeviceMemoryProperties(context.instance.physicalDevice, &properties);

    int fallbackIndex = -1;
    VkMemoryPropertyFlags fallbackFlags = 0;

    for (uint32_t memoryTypeIndex = 0; memoryTypeIndex < properties.memoryTypeCount; ++memoryTypeIndex)
    {
        if ((typeBits & (1u << memoryTypeIndex)) == 0)
        {
            continue;
        }

        const VkMemoryPropertyFlags flags = properties.memoryTypes[memoryTypeIndex].propertyFlags;
        if ((flags & requiredFlags) != requiredFlags)
        {
            continue;
        }

        if ((flags & preferredFlags) == preferredFlags)
        {
            *outMemoryTypeIndex = memoryTypeIndex;
            if (outFlags)
            {
                *outFlags = flags;
            }
            return true;
        }

        if (fallbackIndex < 0)
        {
            fallbackIndex = static_cast<int>(memoryTypeIndex);
            fallbackFlags = flags;
        }
    }

    if (fallbackIndex < 0)
    {
        return false;
    }

    *outMemoryTypeIndex = static_cast<uint32_t>(fallbackIndex);
    if (outFlags)
    {
        *outFlags = fallbackFlags;
    }
    return true;
}

// Phase A: runs on a Task.Run worker thread. Performs ONLY device-level Vulkan
// operations that are thread-safe per spec (vkCreateImage / vkCreateBuffer /
// vkAllocateMemory / vkBindXxxMemory / vkMapMemory / memcpy / vkUnmapMemory /
// vkFlushMappedMemoryRanges). Returns a PendingUpload whose VkImage is in
// VK_IMAGE_LAYOUT_UNDEFINED — the actual GPU upload command (barriers + copy +
// barriers + submit) is queued by Phase B on the render thread.
//
// On failure, all partially-created Vulkan resources are destroyed in this same
// function so the caller never sees a leak from a half-built upload.
//
// IMPORTANT: this function MUST NOT call vkQueueSubmit or any vkCmd* function.
// Those operate on a command buffer / queue and must run on the render thread
// (Vulkan spec + Unity's IUnityGraphicsVulkan constraint). Doing them here is
// what caused the earlier SIGSEGV on Thread-6 (libunity.so) — two worker threads
// racing on Unity's internal command buffer slots.
//
// outImageMemory is returned out-of-band so the caller (CreateWithOptions) can
// register it on VulkanTextureResource for the eventual Release path. The staging
// buffer/memory live on PendingUpload only — Phase C destroys them once the
// fence signals.
bool PrepareUploadOnWorker(const VulkanContext& context,
                           const unsigned char* rgba,
                           int width,
                           int height,
                           bool useSrgb,
                           PendingUpload* outPending,
                           VkDeviceMemory* outImageMemory,
                           std::string* outError)
{
    if (!outPending || !outImageMemory || !context.valid)
    {
        if (outError) *outError = "PrepareUploadOnWorker: invalid args / context";
        return false;
    }

    const VulkanFunctions& vk = context.functions;
    const VkDevice device = context.instance.device;
    const VkDeviceSize imageSize = static_cast<VkDeviceSize>(width) *
                                   static_cast<VkDeviceSize>(height) *
                                   static_cast<VkDeviceSize>(kBytesPerPixel);

    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory imageMemory = VK_NULL_HANDLE;
    VkBuffer stagingBuffer = VK_NULL_HANDLE;
    VkDeviceMemory stagingMemory = VK_NULL_HANDLE;
    VkMemoryPropertyFlags stagingFlags = 0;

    auto fail = [&](const char* action, VkResult result) -> bool
    {
        if (outError)
        {
            *outError = std::string(action) + " failed with VkResult " +
                        std::to_string(static_cast<int>(result));
        }
        if (stagingBuffer != VK_NULL_HANDLE) vk.DestroyBuffer(device, stagingBuffer, nullptr);
        if (stagingMemory != VK_NULL_HANDLE) vk.FreeMemory(device, stagingMemory, nullptr);
        if (image != VK_NULL_HANDLE) vk.DestroyImage(device, image, nullptr);
        if (imageMemory != VK_NULL_HANDLE) vk.FreeMemory(device, imageMemory, nullptr);
        return false;
    };

    VkImageCreateInfo imageInfo = {};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.format = useSrgb ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM;
    imageInfo.extent.width = static_cast<uint32_t>(width);
    imageInfo.extent.height = static_cast<uint32_t>(height);
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    // TRANSFER_SRC_BIT is required so Unity OpenXR Composition Layers (LocalTexture
    // source mode) can copy the texture into the OpenXR-allocated swapchain image
    // each frame via vkCmdCopyImage / vkCmdBlitImage. Without it, the composition
    // layer's swapchain stays empty and the HD pano renders pure black on Quest;
    // the failure is silent in release builds since validation layers are disabled.
    imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT
                    | VK_IMAGE_USAGE_TRANSFER_DST_BIT
                    | VK_IMAGE_USAGE_SAMPLED_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VkResult result = vk.CreateImage(device, &imageInfo, nullptr, &image);
    if (result != VK_SUCCESS) return fail("vkCreateImage", result);

    VkMemoryRequirements imageRequirements = {};
    vk.GetImageMemoryRequirements(device, image, &imageRequirements);

    uint32_t imageMemoryTypeIndex = 0;
    if (!FindMemoryType(context, imageRequirements.memoryTypeBits,
                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                        &imageMemoryTypeIndex, nullptr))
    {
        if (outError) *outError = "No compatible device-local Vulkan memory type for image";
        return fail("FindMemoryType(image)", VK_ERROR_OUT_OF_DEVICE_MEMORY);
    }

    VkMemoryAllocateInfo imageMemoryInfo = {};
    imageMemoryInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    imageMemoryInfo.allocationSize = imageRequirements.size;
    imageMemoryInfo.memoryTypeIndex = imageMemoryTypeIndex;
    result = vk.AllocateMemory(device, &imageMemoryInfo, nullptr, &imageMemory);
    if (result != VK_SUCCESS) return fail("vkAllocateMemory(image)", result);

    result = vk.BindImageMemory(device, image, imageMemory, 0);
    if (result != VK_SUCCESS) return fail("vkBindImageMemory", result);

    VkBufferCreateInfo stagingBufferInfo = {};
    stagingBufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    stagingBufferInfo.size = imageSize;
    stagingBufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    stagingBufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    result = vk.CreateBuffer(device, &stagingBufferInfo, nullptr, &stagingBuffer);
    if (result != VK_SUCCESS) return fail("vkCreateBuffer", result);

    VkMemoryRequirements stagingRequirements = {};
    vk.GetBufferMemoryRequirements(device, stagingBuffer, &stagingRequirements);

    uint32_t stagingMemoryTypeIndex = 0;
    if (!FindMemoryType(context, stagingRequirements.memoryTypeBits,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                        &stagingMemoryTypeIndex, &stagingFlags))
    {
        if (outError) *outError = "No compatible host-visible Vulkan memory type for staging buffer";
        return fail("FindMemoryType(staging)", VK_ERROR_OUT_OF_HOST_MEMORY);
    }

    VkMemoryAllocateInfo stagingMemoryInfo = {};
    stagingMemoryInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    stagingMemoryInfo.allocationSize = stagingRequirements.size;
    stagingMemoryInfo.memoryTypeIndex = stagingMemoryTypeIndex;
    result = vk.AllocateMemory(device, &stagingMemoryInfo, nullptr, &stagingMemory);
    if (result != VK_SUCCESS) return fail("vkAllocateMemory(staging)", result);

    result = vk.BindBufferMemory(device, stagingBuffer, stagingMemory, 0);
    if (result != VK_SUCCESS) return fail("vkBindBufferMemory", result);

    // The big memcpy — this is the 3-9ms cost for a 4K pano. Doing it here on the
    // worker thread (instead of inside the render-thread callback as the old code
    // did) is the entire point of this refactor: render thread stays under 1ms
    // for the actual command recording + submit.
    void* mappedMemory = nullptr;
    result = vk.MapMemory(device, stagingMemory, 0, imageSize, 0, &mappedMemory);
    if (result != VK_SUCCESS) return fail("vkMapMemory", result);

    std::memcpy(mappedMemory, rgba, static_cast<size_t>(imageSize));

    if ((stagingFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) == 0)
    {
        VkMappedMemoryRange memoryRange = {};
        memoryRange.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
        memoryRange.memory = stagingMemory;
        memoryRange.offset = 0;
        memoryRange.size = VK_WHOLE_SIZE;
        result = vk.FlushMappedMemoryRanges(device, 1, &memoryRange);
        if (result != VK_SUCCESS)
        {
            vk.UnmapMemory(device, stagingMemory);
            return fail("vkFlushMappedMemoryRanges", result);
        }
    }
    vk.UnmapMemory(device, stagingMemory);

    outPending->context = context;
    outPending->image = image;
    outPending->stagingBuffer = stagingBuffer;
    outPending->stagingMemory = stagingMemory;
    outPending->width = static_cast<uint32_t>(width);
    outPending->height = static_cast<uint32_t>(height);
    *outImageMemory = imageMemory;
    return true;
}

// Phase C: called at the start of each Phase B run. Scans gInFlightUploads,
// reaps entries whose fence has signaled, destroys their staging buffers / cmd
// pools / fences. Non-blocking — uses vkGetFenceStatus, never vkWaitForFences.
//
// Caller MUST hold gUploadsMutex.
void DrainCompletedInFlightsLocked()
{
    for (auto it = gInFlightUploads.begin(); it != gInFlightUploads.end(); )
    {
        const VulkanFunctions& vk = it->context.functions;
        const VkDevice device = it->context.instance.device;
        VkResult fenceState = vk.GetFenceStatus(device, it->fence);
        // VK_SUCCESS = signaled (GPU done). VK_NOT_READY = still running. Other =
        // error (device lost etc) — treat as "give up, free anyway" since the
        // device is already in a bad state and leaking would be worse.
        if (fenceState == VK_NOT_READY)
        {
            ++it;
            continue;
        }
        if (it->commandBuffer != VK_NULL_HANDLE && it->commandPool != VK_NULL_HANDLE)
        {
            vk.FreeCommandBuffers(device, it->commandPool, 1, &it->commandBuffer);
        }
        if (it->commandPool != VK_NULL_HANDLE) vk.DestroyCommandPool(device, it->commandPool, nullptr);
        if (it->fence != VK_NULL_HANDLE)       vk.DestroyFence(device, it->fence, nullptr);
        if (it->stagingBuffer != VK_NULL_HANDLE) vk.DestroyBuffer(device, it->stagingBuffer, nullptr);
        if (it->stagingMemory != VK_NULL_HANDLE) vk.FreeMemory(device, it->stagingMemory, nullptr);
        it = gInFlightUploads.erase(it);
    }
}

// Enqueue a released texture for fence-gated destruction. Called from Release() on the
// main thread — does NOT touch the GPU (no stall); the render-thread callback finishes
// the job. The externalImage handle is freed only when the image is actually destroyed.
void EnqueuePendingDestroy(const VulkanContext& context, const VulkanTextureResource& resource)
{
    PendingDestroy destroy = {};
    destroy.context = context;
    destroy.image = resource.image;
    destroy.memory = resource.memory;
    destroy.externalImage = resource.externalImage;
    std::lock_guard<std::mutex> lock(gUploadsMutex);
    gPendingDestroys.push_back(destroy);
}

// Render-thread only (called from SubmitPendingUploadsCallback, which runs with graphics-
// queue access). Two-phase, NON-BLOCKING destruction:
//   - Arm: a freshly-enqueued image (fence == NULL) gets a new fence submitted to the
//     graphics queue with ZERO command buffers. Queue submissions execute in order, so
//     this fence signals only after every prior submission — including Unity's last
//     sampling / blit of this image — has retired on the GPU.
//   - Reap: once vkGetFenceStatus reports the fence signaled, vkDestroyImage / vkFreeMemory
//     are provably safe. No vkQueueWaitIdle, no host-side full-queue stall.
// gUploadsMutex is held only to move entries in/out of the shared vector, never across the
// vkQueueSubmit (mirrors the existing uploads-queue discipline).
void ProcessPendingDestroys()
{
    std::vector<PendingDestroy> work;
    {
        std::lock_guard<std::mutex> lock(gUploadsMutex);
        work.swap(gPendingDestroys);
    }
    if (work.empty())
    {
        return;
    }

    std::vector<PendingDestroy> stillPending;
    for (PendingDestroy& destroy : work)
    {
        if (!destroy.context.valid)
        {
            // Device gone — leak the VkImage rather than risk UB; drop the wrapper.
            delete destroy.externalImage;
            continue;
        }
        const VulkanFunctions& vk = destroy.context.functions;
        const VkDevice device = destroy.context.instance.device;
        const VkQueue queue = destroy.context.instance.graphicsQueue;

        auto destroyNow = [&]()
        {
            if (destroy.image != VK_NULL_HANDLE) vk.DestroyImage(device, destroy.image, nullptr);
            if (destroy.memory != VK_NULL_HANDLE) vk.FreeMemory(device, destroy.memory, nullptr);
            if (destroy.fence != VK_NULL_HANDLE) vk.DestroyFence(device, destroy.fence, nullptr);
            delete destroy.externalImage;
        };

        if (destroy.fence == VK_NULL_HANDLE)
        {
            VkFenceCreateInfo fenceInfo = {};
            fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
            if (vk.CreateFence(device, &fenceInfo, nullptr, &destroy.fence) != VK_SUCCESS)
            {
                // Cannot arm — hard-wait once then destroy so we never free an in-use image.
                if (queue != VK_NULL_HANDLE) vk.QueueWaitIdle(queue);
                destroyNow();
                continue;
            }
            VkSubmitInfo submitInfo = {};
            submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submitInfo.commandBufferCount = 0;
            if (vk.QueueSubmit(queue, 1, &submitInfo, destroy.fence) != VK_SUCCESS)
            {
                if (queue != VK_NULL_HANDLE) vk.QueueWaitIdle(queue);
                destroyNow();
                continue;
            }
            stillPending.push_back(destroy);
            continue;
        }

        VkResult fenceState = vk.GetFenceStatus(device, destroy.fence);
        if (fenceState == VK_NOT_READY)
        {
            stillPending.push_back(destroy);
            continue;
        }
        // Signaled (GPU done) or an error (device lost — free anyway): safe to destroy.
        destroyNow();
    }

    if (!stillPending.empty())
    {
        std::lock_guard<std::mutex> lock(gUploadsMutex);
        for (PendingDestroy& destroy : stillPending)
        {
            gPendingDestroys.push_back(destroy);
        }
    }
}

// Shutdown / device-loss flush: blocking destruction of every still-pending entry. A host
// stall is acceptable here (the session is tearing down). Must run with a live context.
// Bounded fence wait for teardown / release paths. An infinite (UINT64_MAX) wait hangs the
// calling thread forever if the fence never signals — which is exactly what happens when the
// Android Surface is destroyed (app backgrounded / reset / quit) and the queued GPU work is
// abandoned. That hang on the render/main thread during onSurfaceDestroyed trips Android's ANR
// watchdog (oculus:android_anr ... onSurfaceDestroyedNative). Cap the wait; on timeout / device
// loss we proceed to destroy anyway — during teardown the device is going away regardless, and a
// benign early free is far better than an ANR.
static constexpr uint64_t kFenceWaitTimeoutNs = 500000000ull; // 0.5s — >>10x any real GPU copy

void WaitFenceBounded(const VulkanFunctions& vk, VkDevice device, VkFence fence, const char* where)
{
    const VkResult result = vk.WaitForFences(device, 1, &fence, VK_TRUE, kFenceWaitTimeoutNs);
    if (result != VK_SUCCESS)
    {
        LogMessage("Warn", "%s: bounded fence wait returned VkResult %d (proceeding with teardown)",
            where, static_cast<int>(result));
    }
}

void FlushPendingDestroysBlocking()
{
    std::vector<PendingDestroy> work;
    {
        std::lock_guard<std::mutex> lock(gUploadsMutex);
        work.swap(gPendingDestroys);
    }
    for (PendingDestroy& destroy : work)
    {
        if (destroy.context.valid)
        {
            const VulkanFunctions& vk = destroy.context.functions;
            const VkDevice device = destroy.context.instance.device;
            if (destroy.fence != VK_NULL_HANDLE)
            {
                WaitFenceBounded(vk, device, destroy.fence, "FlushPendingDestroysBlocking");
                vk.DestroyFence(device, destroy.fence, nullptr);
            }
            else if (destroy.context.instance.graphicsQueue != VK_NULL_HANDLE)
            {
                vk.QueueWaitIdle(destroy.context.instance.graphicsQueue);
            }
            if (destroy.image != VK_NULL_HANDLE) vk.DestroyImage(device, destroy.image, nullptr);
            if (destroy.memory != VK_NULL_HANDLE) vk.FreeMemory(device, destroy.memory, nullptr);
        }
        delete destroy.externalImage;
    }
}

// ---- Pano assembly helpers ----------------------------------------------

// Round a tile staging need up to a 1 MiB granularity so all slots for 512x512
// tiles are uniform (1 MiB) and fully interchangeable across full + edge tiles.
VkDeviceSize RoundUpStaging(VkDeviceSize need)
{
    const VkDeviceSize granularity = static_cast<VkDeviceSize>(1) << 20;
    return ((need + granularity - 1) / granularity) * granularity;
}

// Rent a host-visible staging buffer of at least `need` bytes. Reuses a free-list
// slot when possible (capacity >= need); otherwise allocates a new one. Called on
// worker threads — device-level ops only (thread-safe), no command work.
bool AcquireStagingSlot(const VulkanContext& context, VkDeviceSize need, StagingSlot* out, std::string* outError)
{
    {
        std::lock_guard<std::mutex> lock(gUploadsMutex);
        for (auto it = gStagingFreeList.begin(); it != gStagingFreeList.end(); ++it)
        {
            if (it->capacity >= need)
            {
                *out = *it;
                gStagingFreeList.erase(it);
                return true;
            }
        }
    }

    const VulkanFunctions& vk = context.functions;
    const VkDevice device = context.instance.device;
    const VkDeviceSize size = RoundUpStaging(need);

    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkMemoryPropertyFlags flags = 0;
    auto fail = [&](const char* action, VkResult result) -> bool
    {
        if (outError)
        {
            *outError = std::string(action) + " failed with VkResult " + std::to_string(static_cast<int>(result));
        }
        if (buffer != VK_NULL_HANDLE) vk.DestroyBuffer(device, buffer, nullptr);
        if (memory != VK_NULL_HANDLE) vk.FreeMemory(device, memory, nullptr);
        return false;
    };

    VkBufferCreateInfo bufferInfo = {};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkResult result = vk.CreateBuffer(device, &bufferInfo, nullptr, &buffer);
    if (result != VK_SUCCESS) return fail("vkCreateBuffer(staging ring)", result);

    VkMemoryRequirements requirements = {};
    vk.GetBufferMemoryRequirements(device, buffer, &requirements);

    uint32_t memoryTypeIndex = 0;
    if (!FindMemoryType(context, requirements.memoryTypeBits,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                        &memoryTypeIndex, &flags))
    {
        if (outError) *outError = "No host-visible Vulkan memory type for staging ring";
        return fail("FindMemoryType(staging ring)", VK_ERROR_OUT_OF_HOST_MEMORY);
    }

    VkMemoryAllocateInfo memoryInfo = {};
    memoryInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    memoryInfo.allocationSize = requirements.size;
    memoryInfo.memoryTypeIndex = memoryTypeIndex;
    result = vk.AllocateMemory(device, &memoryInfo, nullptr, &memory);
    if (result != VK_SUCCESS) return fail("vkAllocateMemory(staging ring)", result);

    result = vk.BindBufferMemory(device, buffer, memory, 0);
    if (result != VK_SUCCESS) return fail("vkBindBufferMemory(staging ring)", result);

    out->buffer = buffer;
    out->memory = memory;
    out->capacity = size;
    out->coherent = (flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
    return true;
}

// Caller MUST hold gUploadsMutex.
void ReturnStagingSlotLocked(const StagingSlot& slot)
{
    gStagingFreeList.push_back(slot);
}

void DestroyStagingSlotDirect(const VulkanContext& context, const StagingSlot& slot)
{
    if (!context.valid || context.instance.device == VK_NULL_HANDLE)
    {
        return;
    }
    const VulkanFunctions& vk = context.functions;
    const VkDevice device = context.instance.device;
    if (slot.buffer != VK_NULL_HANDLE) vk.DestroyBuffer(device, slot.buffer, nullptr);
    if (slot.memory != VK_NULL_HANDLE) vk.FreeMemory(device, slot.memory, nullptr);
}

// Image-only half of PrepareUploadOnWorker (no staging, no memcpy). Worker thread.
bool CreateAssemblyImageOnWorker(const VulkanContext& context, int width, int height, bool useSrgb,
                                 VkImage* outImage, VkDeviceMemory* outMemory, std::string* outError)
{
    const VulkanFunctions& vk = context.functions;
    const VkDevice device = context.instance.device;

    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory imageMemory = VK_NULL_HANDLE;
    auto fail = [&](const char* action, VkResult result) -> bool
    {
        if (outError)
        {
            *outError = std::string(action) + " failed with VkResult " + std::to_string(static_cast<int>(result));
        }
        if (image != VK_NULL_HANDLE) vk.DestroyImage(device, image, nullptr);
        if (imageMemory != VK_NULL_HANDLE) vk.FreeMemory(device, imageMemory, nullptr);
        return false;
    };

    VkImageCreateInfo imageInfo = {};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.format = useSrgb ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM;
    imageInfo.extent.width = static_cast<uint32_t>(width);
    imageInfo.extent.height = static_cast<uint32_t>(height);
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    // Same usage as PrepareUploadOnWorker — TRANSFER_SRC for the OpenXR composition
    // layer blit-out, TRANSFER_DST for the tile copies, SAMPLED for the sphere blit.
    imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT
                    | VK_IMAGE_USAGE_TRANSFER_DST_BIT
                    | VK_IMAGE_USAGE_SAMPLED_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VkResult result = vk.CreateImage(device, &imageInfo, nullptr, &image);
    if (result != VK_SUCCESS) return fail("vkCreateImage(assembly)", result);

    VkMemoryRequirements requirements = {};
    vk.GetImageMemoryRequirements(device, image, &requirements);

    uint32_t memoryTypeIndex = 0;
    if (!FindMemoryType(context, requirements.memoryTypeBits,
                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                        &memoryTypeIndex, nullptr))
    {
        if (outError) *outError = "No device-local Vulkan memory type for assembly image";
        return fail("FindMemoryType(assembly image)", VK_ERROR_OUT_OF_DEVICE_MEMORY);
    }

    VkMemoryAllocateInfo memoryInfo = {};
    memoryInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    memoryInfo.allocationSize = requirements.size;
    memoryInfo.memoryTypeIndex = memoryTypeIndex;
    result = vk.AllocateMemory(device, &memoryInfo, nullptr, &imageMemory);
    if (result != VK_SUCCESS) return fail("vkAllocateMemory(assembly image)", result);

    result = vk.BindImageMemory(device, image, imageMemory, 0);
    if (result != VK_SUCCESS) return fail("vkBindImageMemory(assembly)", result);

    *outImage = image;
    *outMemory = imageMemory;
    return true;
}

// Phase C for assemblies. Caller MUST hold gUploadsMutex. Reaps signaled fences,
// destroys the transient command pool/buffer/fence, RETURNS stagings to the ring.
void DrainCompletedAssembliesLocked()
{
    for (auto it = gInFlightAssemblies.begin(); it != gInFlightAssemblies.end(); )
    {
        const VulkanFunctions& vk = it->context.functions;
        const VkDevice device = it->context.instance.device;
        VkResult fenceState = vk.GetFenceStatus(device, it->fence);
        if (fenceState == VK_NOT_READY)
        {
            ++it;
            continue;
        }
        if (it->commandBuffer != VK_NULL_HANDLE && it->commandPool != VK_NULL_HANDLE)
        {
            vk.FreeCommandBuffers(device, it->commandPool, 1, &it->commandBuffer);
        }
        if (it->commandPool != VK_NULL_HANDLE) vk.DestroyCommandPool(device, it->commandPool, nullptr);
        if (it->fence != VK_NULL_HANDLE)       vk.DestroyFence(device, it->fence, nullptr);
        for (const StagingSlot& slot : it->stagings)
        {
            ReturnStagingSlotLocked(slot);
        }
        it = gInFlightAssemblies.erase(it);
    }
}

// Phase B for assemblies (render thread, graphics-queue access). For each committed
// assembly records ONE command buffer: UNDEFINED->TRANSFER_DST, N region copies
// (disjoint tiles, no inter-copy barrier), TRANSFER_DST->SHADER_READ; one submit,
// one fence. Fire-and-forget; stagings returned to the ring by Phase C.
void SubmitPendingAssemblies()
{
    std::vector<PendingAssembly> drained;
    {
        std::lock_guard<std::mutex> lock(gUploadsMutex);
        if (gPendingAssemblies.empty())
        {
            return;
        }
        drained.swap(gPendingAssemblies);
    }

    for (PendingAssembly& assembly : drained)
    {
        const VulkanContext& context = assembly.context;
        auto returnStagings = [&]()
        {
            std::lock_guard<std::mutex> lock(gUploadsMutex);
            for (const StagingSlot& slot : assembly.stagings) ReturnStagingSlotLocked(slot);
        };

        if (!context.valid)
        {
            LogMessage("Warn", "SubmitPendingAssemblies skipped: invalid Vulkan context");
            returnStagings();
            continue;
        }
        const VulkanFunctions& vk = context.functions;
        const VkDevice device = context.instance.device;

        VkCommandPool commandPool = VK_NULL_HANDLE;
        VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
        VkFence fence = VK_NULL_HANDLE;
        auto cleanupTransient = [&]()
        {
            if (commandBuffer != VK_NULL_HANDLE && commandPool != VK_NULL_HANDLE)
                vk.FreeCommandBuffers(device, commandPool, 1, &commandBuffer);
            if (commandPool != VK_NULL_HANDLE) vk.DestroyCommandPool(device, commandPool, nullptr);
            if (fence != VK_NULL_HANDLE)       vk.DestroyFence(device, fence, nullptr);
        };
        auto abort = [&](const char* where, VkResult result)
        {
            LogMessage("Error", "SubmitPendingAssemblies: %s=%d", where, static_cast<int>(result));
            cleanupTransient();
            returnStagings();
        };

        VkCommandPoolCreateInfo commandPoolInfo = {};
        commandPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        commandPoolInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
        commandPoolInfo.queueFamilyIndex = context.instance.queueFamilyIndex;
        VkResult result = vk.CreateCommandPool(device, &commandPoolInfo, nullptr, &commandPool);
        if (result != VK_SUCCESS) { abort("vkCreateCommandPool", result); continue; }

        VkCommandBufferAllocateInfo commandBufferInfo = {};
        commandBufferInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        commandBufferInfo.commandPool = commandPool;
        commandBufferInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        commandBufferInfo.commandBufferCount = 1;
        result = vk.AllocateCommandBuffers(device, &commandBufferInfo, &commandBuffer);
        if (result != VK_SUCCESS) { abort("vkAllocateCommandBuffers", result); continue; }

        VkCommandBufferBeginInfo beginInfo = {};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        result = vk.BeginCommandBuffer(commandBuffer, &beginInfo);
        if (result != VK_SUCCESS) { abort("vkBeginCommandBuffer", result); continue; }

        VkImageSubresourceRange range = {};
        range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        range.baseMipLevel = 0;
        range.levelCount = 1;
        range.baseArrayLayer = 0;
        range.layerCount = 1;

        VkImageMemoryBarrier toDst = {};
        toDst.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        toDst.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        toDst.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        toDst.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toDst.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toDst.image = assembly.image;
        toDst.subresourceRange = range;
        toDst.srcAccessMask = 0;
        toDst.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        vk.CmdPipelineBarrier(commandBuffer,
                              VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                              VK_PIPELINE_STAGE_TRANSFER_BIT,
                              0, 0, nullptr, 0, nullptr, 1, &toDst);

        for (size_t i = 0; i < assembly.regions.size(); ++i)
        {
            vk.CmdCopyBufferToImage(commandBuffer, assembly.stagings[i].buffer, assembly.image,
                                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &assembly.regions[i]);
        }

        VkImageMemoryBarrier toRead = {};
        toRead.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        toRead.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        toRead.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        toRead.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toRead.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toRead.image = assembly.image;
        toRead.subresourceRange = range;
        toRead.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        toRead.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vk.CmdPipelineBarrier(commandBuffer,
                              VK_PIPELINE_STAGE_TRANSFER_BIT,
                              VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                              0, 0, nullptr, 0, nullptr, 1, &toRead);

        result = vk.EndCommandBuffer(commandBuffer);
        if (result != VK_SUCCESS) { abort("vkEndCommandBuffer", result); continue; }

        VkFenceCreateInfo fenceInfo = {};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        result = vk.CreateFence(device, &fenceInfo, nullptr, &fence);
        if (result != VK_SUCCESS) { abort("vkCreateFence", result); continue; }

        VkSubmitInfo submitInfo = {};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffer;
        result = vk.QueueSubmit(context.instance.graphicsQueue, 1, &submitInfo, fence);
        if (result != VK_SUCCESS) { abort("vkQueueSubmit", result); continue; }

        InFlightAssembly inflight = {};
        inflight.context = context;
        inflight.image = assembly.image;
        inflight.commandPool = commandPool;
        inflight.commandBuffer = commandBuffer;
        inflight.fence = fence;
        inflight.stagings = std::move(assembly.stagings);
        {
            std::lock_guard<std::mutex> lock(gUploadsMutex);
            gInFlightAssemblies.push_back(std::move(inflight));
        }
        commandPool = VK_NULL_HANDLE;
        commandBuffer = VK_NULL_HANDLE;
        fence = VK_NULL_HANDLE;
    }
}

// Shutdown / device-loss flush for assembly staging/command resources. The assembly
// device images live in gTextureResources and are freed by DrainOutstandingTextures;
// here we free the host stagings + any transient command pools/fences.
void DestroyAssemblyResourcesBlocking()
{
    std::vector<PendingAssembly> pending;
    std::vector<InFlightAssembly> inflight;
    std::vector<StagingSlot> ring;
    {
        std::lock_guard<std::mutex> lock(gUploadsMutex);
        pending.swap(gPendingAssemblies);
        inflight.swap(gInFlightAssemblies);
        ring.swap(gStagingFreeList);
    }
    std::vector<AssemblyImage> inProgress;
    {
        std::lock_guard<std::mutex> lock(gAssembliesMutex);
        inProgress.reserve(gAssemblies.size());
        for (auto& entry : gAssemblies) inProgress.push_back(std::move(entry.second));
        gAssemblies.clear();
    }

    for (InFlightAssembly& assembly : inflight)
    {
        if (!assembly.context.valid) continue;
        const VulkanFunctions& vk = assembly.context.functions;
        const VkDevice device = assembly.context.instance.device;
        if (assembly.fence != VK_NULL_HANDLE)
        {
            WaitFenceBounded(vk, device, assembly.fence, "DestroyAssemblyResourcesBlocking");
        }
        if (assembly.commandBuffer != VK_NULL_HANDLE && assembly.commandPool != VK_NULL_HANDLE)
            vk.FreeCommandBuffers(device, assembly.commandPool, 1, &assembly.commandBuffer);
        if (assembly.commandPool != VK_NULL_HANDLE) vk.DestroyCommandPool(device, assembly.commandPool, nullptr);
        if (assembly.fence != VK_NULL_HANDLE)       vk.DestroyFence(device, assembly.fence, nullptr);
        for (const StagingSlot& slot : assembly.stagings) DestroyStagingSlotDirect(assembly.context, slot);
    }
    for (PendingAssembly& assembly : pending)
        for (const StagingSlot& slot : assembly.stagings) DestroyStagingSlotDirect(assembly.context, slot);
    for (AssemblyImage& assembly : inProgress)
        for (const StagingSlot& slot : assembly.stagings) DestroyStagingSlotDirect(assembly.context, slot);

    if (!ring.empty())
    {
        VulkanContext context = {};
        {
            std::lock_guard<std::mutex> lock(gUnityStateMutex);
            context = gVulkanContext;
        }
        for (const StagingSlot& slot : ring) DestroyStagingSlotDirect(context, slot);
    }
}

// Phase B: render-thread callback fired via vulkanV2->AccessQueue(...). Drains
// gPendingUploads, for each records vkCmdPipelineBarrier × 2 + vkCmdCopyBufferToImage
// onto a transient command buffer and submits. The submit is FIRE-AND-FORGET:
// the staging buffer + command pool + fence stay alive in gInFlightUploads until
// DrainCompletedInFlightsLocked reaps them once the fence has signaled.
//
// Total render-thread cost per upload: <1ms (a few command records + one submit).
// The slow staging memcpy ran on the worker; the GPU does the actual copy
// asynchronously after submit returns.
void UNITY_INTERFACE_API SubmitPendingUploadsCallback(int /*eventId*/, void* /*userData*/)
{
    // Advance fence-gated destruction of user-released textures first — this is the only
    // place with graphics-queue access, and it must run even on ticks with no new uploads.
    ProcessPendingDestroys();

    std::vector<PendingUpload> drained;
    {
        std::lock_guard<std::mutex> lock(gUploadsMutex);
        DrainCompletedInFlightsLocked();
        DrainCompletedAssembliesLocked();
        drained.swap(gPendingUploads);
    }

    for (PendingUpload& pending : drained)
    {
        const VulkanContext& context = pending.context;
        if (!context.valid)
        {
            LogMessage("Warn", "SubmitPendingUploadsCallback skipped: invalid Vulkan context");
            continue;
        }
        const VulkanFunctions& vk = context.functions;
        const VkDevice device = context.instance.device;

        VkCommandPool commandPool = VK_NULL_HANDLE;
        VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
        VkFence fence = VK_NULL_HANDLE;

        auto cleanupTransient = [&]()
        {
            if (commandBuffer != VK_NULL_HANDLE && commandPool != VK_NULL_HANDLE)
                vk.FreeCommandBuffers(device, commandPool, 1, &commandBuffer);
            if (commandPool != VK_NULL_HANDLE) vk.DestroyCommandPool(device, commandPool, nullptr);
            if (fence != VK_NULL_HANDLE)       vk.DestroyFence(device, fence, nullptr);
        };
        auto cleanupAll = [&]()
        {
            cleanupTransient();
            if (pending.stagingBuffer != VK_NULL_HANDLE) vk.DestroyBuffer(device, pending.stagingBuffer, nullptr);
            if (pending.stagingMemory != VK_NULL_HANDLE) vk.FreeMemory(device, pending.stagingMemory, nullptr);
        };

        VkCommandPoolCreateInfo commandPoolInfo = {};
        commandPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        commandPoolInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
        commandPoolInfo.queueFamilyIndex = context.instance.queueFamilyIndex;
        VkResult result = vk.CreateCommandPool(device, &commandPoolInfo, nullptr, &commandPool);
        if (result != VK_SUCCESS)
        {
            LogMessage("Error", "SubmitPendingUploadsCallback: vkCreateCommandPool=%d", static_cast<int>(result));
            cleanupAll();
            continue;
        }

        VkCommandBufferAllocateInfo commandBufferInfo = {};
        commandBufferInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        commandBufferInfo.commandPool = commandPool;
        commandBufferInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        commandBufferInfo.commandBufferCount = 1;
        result = vk.AllocateCommandBuffers(device, &commandBufferInfo, &commandBuffer);
        if (result != VK_SUCCESS)
        {
            LogMessage("Error", "SubmitPendingUploadsCallback: vkAllocateCommandBuffers=%d", static_cast<int>(result));
            cleanupAll();
            continue;
        }

        VkCommandBufferBeginInfo beginInfo = {};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        result = vk.BeginCommandBuffer(commandBuffer, &beginInfo);
        if (result != VK_SUCCESS)
        {
            LogMessage("Error", "SubmitPendingUploadsCallback: vkBeginCommandBuffer=%d", static_cast<int>(result));
            cleanupAll();
            continue;
        }

        VkImageMemoryBarrier uploadBarrier = {};
        uploadBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        uploadBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        uploadBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        uploadBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        uploadBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        uploadBarrier.image = pending.image;
        uploadBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        uploadBarrier.subresourceRange.baseMipLevel = 0;
        uploadBarrier.subresourceRange.levelCount = 1;
        uploadBarrier.subresourceRange.baseArrayLayer = 0;
        uploadBarrier.subresourceRange.layerCount = 1;
        uploadBarrier.srcAccessMask = 0;
        uploadBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        vk.CmdPipelineBarrier(commandBuffer,
                              VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                              VK_PIPELINE_STAGE_TRANSFER_BIT,
                              0, 0, nullptr, 0, nullptr, 1, &uploadBarrier);

        VkBufferImageCopy copyRegion = {};
        copyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        copyRegion.imageSubresource.layerCount = 1;
        copyRegion.imageExtent.width = pending.width;
        copyRegion.imageExtent.height = pending.height;
        copyRegion.imageExtent.depth = 1;
        vk.CmdCopyBufferToImage(commandBuffer, pending.stagingBuffer, pending.image,
                                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copyRegion);

        VkImageMemoryBarrier shaderReadBarrier = {};
        shaderReadBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        shaderReadBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        shaderReadBarrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        shaderReadBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        shaderReadBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        shaderReadBarrier.image = pending.image;
        shaderReadBarrier.subresourceRange = uploadBarrier.subresourceRange;
        shaderReadBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        shaderReadBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vk.CmdPipelineBarrier(commandBuffer,
                              VK_PIPELINE_STAGE_TRANSFER_BIT,
                              VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                              0, 0, nullptr, 0, nullptr, 1, &shaderReadBarrier);

        result = vk.EndCommandBuffer(commandBuffer);
        if (result != VK_SUCCESS)
        {
            LogMessage("Error", "SubmitPendingUploadsCallback: vkEndCommandBuffer=%d", static_cast<int>(result));
            cleanupAll();
            continue;
        }

        VkFenceCreateInfo fenceInfo = {};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        result = vk.CreateFence(device, &fenceInfo, nullptr, &fence);
        if (result != VK_SUCCESS)
        {
            LogMessage("Error", "SubmitPendingUploadsCallback: vkCreateFence=%d", static_cast<int>(result));
            cleanupAll();
            continue;
        }

        VkSubmitInfo submitInfo = {};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffer;
        result = vk.QueueSubmit(context.instance.graphicsQueue, 1, &submitInfo, fence);
        if (result != VK_SUCCESS)
        {
            LogMessage("Error", "SubmitPendingUploadsCallback: vkQueueSubmit=%d", static_cast<int>(result));
            cleanupAll();
            continue;
        }

        // Move to in-flight queue — the fence will signal when GPU finishes the
        // copy + barriers. Phase C reaps these on subsequent callbacks.
        InFlightUpload inflight = {};
        inflight.context = context;
        inflight.image = pending.image;
        inflight.stagingBuffer = pending.stagingBuffer;
        inflight.stagingMemory = pending.stagingMemory;
        inflight.commandPool = commandPool;
        inflight.commandBuffer = commandBuffer;
        inflight.fence = fence;
        {
            std::lock_guard<std::mutex> lock(gUploadsMutex);
            gInFlightUploads.push_back(inflight);
        }
        // Clear locals so cleanupTransient/cleanupAll wouldn't double-free if
        // someone resurrects the lambda later. (Defensive — the cleanups aren't
        // called after this point.)
        commandPool = VK_NULL_HANDLE;
        commandBuffer = VK_NULL_HANDLE;
        fence = VK_NULL_HANDLE;
    }

    // Batched pano tile assemblies share this one render event / queue-access tick.
    SubmitPendingAssemblies();
}


void UNITY_INTERFACE_API ReleaseTextureQueueCallback(int /*eventId*/, void* userData)
{
    ReleaseTextureRequest* request = static_cast<ReleaseTextureRequest*>(userData);
    if (!request)
    {
        return;
    }

    if (request->image == VK_NULL_HANDLE || request->memory == VK_NULL_HANDLE)
    {
        CompleteRequest(request->state, true, {});
        return;
    }

    const VulkanContext& context = request->context;
    const VulkanFunctions& vk = context.functions;
    std::string error;

    VkResult result = vk.QueueWaitIdle(context.instance.graphicsQueue);
    if (result != VK_SUCCESS)
    {
        error = "vkQueueWaitIdle failed with VkResult " + std::to_string(static_cast<int>(result));
        CompleteRequest(request->state, false, error);
        return;
    }

    vk.DestroyImage(context.instance.device, request->image, nullptr);
    vk.FreeMemory(context.instance.device, request->memory, nullptr);
    CompleteRequest(request->state, true, {});
}

bool ExecuteQueueRequest(QueueRequestState& state,
                         const VulkanContext& context,
                         UnityRenderingEventAndData callback,
                         void* userData)
{
    std::unique_lock<std::mutex> queueLock(gQueueAccessMutex);

    IUnityGraphicsVulkanV2* vulkanV2 = nullptr;
    IUnityGraphicsVulkan* vulkanV1 = nullptr;
    {
        std::lock_guard<std::mutex> lock(gUnityStateMutex);
        vulkanV2 = gUnityGraphicsVulkanV2;
        vulkanV1 = gUnityGraphicsVulkan;
    }

    if (!context.valid || (!vulkanV2 && !vulkanV1))
    {
        return false;
    }

    if (vulkanV2)
    {
        vulkanV2->AccessQueue(callback, 0, userData, true);
    }
    else
    {
        vulkanV1->AccessQueue(callback, 0, userData, true);
    }

    std::unique_lock<std::mutex> lock(state.mutex);
    state.condition.wait(lock, [&state]() { return state.completed; });
    return state.success;
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
    // CALLED FROM: a Task.Run worker thread (NativeTextureVulkan.CreateAsync's
    // inner Task.Run on the C# side). Does heavy Vulkan device-level work here
    // (vkCreateImage / alloc / staging memcpy) on the worker, then queues a
    // render-thread command-submission callback and returns the IntPtr handle
    // synchronously. The actual GPU upload (vkCmdCopyBufferToImage + barriers)
    // runs asynchronously on Unity's render thread; the staging buffer/cmd pool
    // are reaped by DrainCompletedInFlightsLocked once the fence signals.
    //
    // Ordering guarantee: by the time the C# caller binds the returned
    // Texture2D into a material / OpenXR composition layer (which queues a
    // render-thread command), the AccessQueue submission we fire below is
    // already at the head of Unity's render-thread event queue, so the upload
    // commands serialize before any draw that samples the image. The user
    // never sees an undefined-content frame.
    if (!ValidateImageArgs(rgba, width, height, "Create"))
    {
        return nullptr;
    }

    VulkanContext context = {};
    if (!SnapshotVulkanContext(&context))
    {
        return nullptr;
    }

    // Phase A — Vulkan device-level ops on the worker thread (thread-safe per
    // Vulkan spec). NO command buffer recording, NO queue submission here.
    PendingUpload pending = {};
    VkDeviceMemory imageMemory = VK_NULL_HANDLE;
    std::string error;
    if (!PrepareUploadOnWorker(context, rgba, width, height, useSrgb, &pending, &imageMemory, &error))
    {
        LogMessage("Error", "Create failed (PrepareUploadOnWorker): %s",
                   error.empty() ? "unknown" : error.c_str());
        return nullptr;
    }

    VkImage* externalImage = new (std::nothrow) VkImage(pending.image);
    if (!externalImage)
    {
        // Phase A succeeded but we couldn't allocate the handle slot. Free
        // everything Phase A created — staging + image + memory — then bail.
        const VulkanFunctions& vk = context.functions;
        const VkDevice device = context.instance.device;
        if (pending.stagingBuffer != VK_NULL_HANDLE) vk.DestroyBuffer(device, pending.stagingBuffer, nullptr);
        if (pending.stagingMemory != VK_NULL_HANDLE) vk.FreeMemory(device, pending.stagingMemory, nullptr);
        if (pending.image != VK_NULL_HANDLE) vk.DestroyImage(device, pending.image, nullptr);
        if (imageMemory != VK_NULL_HANDLE) vk.FreeMemory(device, imageMemory, nullptr);
        LogMessage("Error", "Create failed: unable to allocate external VkImage handle");
        return nullptr;
    }

    // Register for Release() — `memory` is the image's device-local memory,
    // NOT the staging memory (staging is reaped separately by Phase C).
    VulkanTextureResource resource = {};
    resource.externalImage = externalImage;
    resource.image = pending.image;
    resource.memory = imageMemory;
    resource.context = context;
    resource.width = static_cast<uint32_t>(width);
    resource.height = static_cast<uint32_t>(height);

    const TextureKey key = TextureKeyFromExternalImage(externalImage);
    {
        std::lock_guard<std::mutex> lock(gTextureMutex);
        gTextureResources[key] = resource;
    }

    // Push Phase A output to the render-thread work queue. The C# layer (on
    // Unity's main thread) will fire `GL.IssuePluginEvent(GetRenderEventFunc(),
    // kEventIdSubmitUploads)` immediately after wrapping the returned
    // `externalImage` IntPtr in a Texture2D; that IssuePluginEvent queues the
    // render-thread callback in order with subsequent main-thread render
    // commands, so any draw that samples the texture will be processed by the
    // render thread AFTER our upload barriers + copy + submit.
    //
    // We intentionally do NOT call vulkanV2->AccessQueue here. AccessQueue is
    // not documented as safe to invoke from a Task.Run worker thread — only
    // GL.IssuePluginEvent's main-thread API has the right semantics for this.
    // Older revisions of this file did call AccessQueue from worker; removed.
    std::lock_guard<std::mutex> lock(gUploadsMutex);
    gPendingUploads.push_back(pending);
    return externalImage;
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

// ---- Pano GPU tile assembly exports -------------------------------------
// Begin -> N x UploadTileRegion -> Finish, then C# wraps the handle with
// Texture2D.CreateExternalTexture and fires GL.IssuePluginEvent(GetRenderEventFunc(),
// kEventIdSubmitUploads) ONCE. On a pre-Finish fault/cancel, C# calls AbortImage.
// Threading: BeginPanoAssembly/UploadTileRegion/FinishPanoAssembly run on Task.Run
// workers (device-level ops + host memcpy only, NO vkCmd*/vkQueueSubmit — the
// I2 SIGSEGV invariant). The batched copy + barriers + submit happen on the render
// thread in SubmitPendingAssemblies.

NATIVE_TEXTURE_API void* UNITY_INTERFACE_API BeginPanoAssembly(int width, int height, bool useSrgb)
{
    if (width <= 0 || height <= 0)
    {
        LogMessage("Error", "BeginPanoAssembly failed: invalid size %dx%d", width, height);
        return nullptr;
    }
    if (width > std::numeric_limits<int>::max() / kBytesPerPixel)
    {
        LogMessage("Error", "BeginPanoAssembly failed: width %d overflows bytesPerRow", width);
        return nullptr;
    }

    VulkanContext context = {};
    if (!SnapshotVulkanContext(&context))
    {
        return nullptr;
    }

    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory imageMemory = VK_NULL_HANDLE;
    std::string error;
    if (!CreateAssemblyImageOnWorker(context, width, height, useSrgb, &image, &imageMemory, &error))
    {
        LogMessage("Error", "BeginPanoAssembly failed: %s", error.empty() ? "unknown" : error.c_str());
        return nullptr;
    }

    VkImage* externalImage = new (std::nothrow) VkImage(image);
    if (!externalImage)
    {
        const VulkanFunctions& vk = context.functions;
        const VkDevice device = context.instance.device;
        vk.DestroyImage(device, image, nullptr);
        vk.FreeMemory(device, imageMemory, nullptr);
        LogMessage("Error", "BeginPanoAssembly failed: unable to allocate external VkImage handle");
        return nullptr;
    }

    const TextureKey key = TextureKeyFromExternalImage(externalImage);

    // Register for Release() (same handle-as-key discipline as CreateWithOptions).
    VulkanTextureResource resource = {};
    resource.externalImage = externalImage;
    resource.image = image;
    resource.memory = imageMemory;
    resource.context = context;
    resource.width = static_cast<uint32_t>(width);
    resource.height = static_cast<uint32_t>(height);
    {
        std::lock_guard<std::mutex> lock(gTextureMutex);
        gTextureResources[key] = resource;
    }

    AssemblyImage assembly = {};
    assembly.context = context;
    assembly.image = image;
    assembly.imageMemory = imageMemory;
    assembly.externalImage = externalImage;
    assembly.width = static_cast<uint32_t>(width);
    assembly.height = static_cast<uint32_t>(height);
    {
        std::lock_guard<std::mutex> lock(gAssembliesMutex);
        gAssemblies[key] = std::move(assembly);
    }

    return externalImage;
}

NATIVE_TEXTURE_API bool UNITY_INTERFACE_API UploadTileRegion(
    void* assemblyImage,
    unsigned char* tileRgba,
    int tileStride,
    int validW,
    int validH,
    int dstX,
    int dstY)
{
    if (!assemblyImage || !tileRgba)
    {
        LogMessage("Error", "UploadTileRegion failed: null image or tile");
        return false;
    }
    if (validW <= 0 || validH <= 0 || tileStride <= 0)
    {
        LogMessage("Error", "UploadTileRegion failed: invalid dims %dx%d stride %d", validW, validH, tileStride);
        return false;
    }
    if (dstX < 0 || dstY < 0)
    {
        LogMessage("Error", "UploadTileRegion failed: negative dst %d,%d", dstX, dstY);
        return false;
    }
    if (validW > tileStride / kBytesPerPixel)
    {
        LogMessage("Error", "UploadTileRegion failed: validW %d exceeds stride %d", validW, tileStride);
        return false;
    }

    const TextureKey key = static_cast<TextureKey>(reinterpret_cast<uintptr_t>(assemblyImage));

    VulkanContext context = {};
    {
        std::lock_guard<std::mutex> lock(gAssembliesMutex);
        auto it = gAssemblies.find(key);
        if (it == gAssemblies.end())
        {
            LogMessage("Warn", "UploadTileRegion ignored: unknown assembly %p", assemblyImage);
            return false;
        }
        context = it->second.context;
    }
    if (!context.valid)
    {
        return false;
    }

    const VkDeviceSize need = static_cast<VkDeviceSize>(validW) *
                              static_cast<VkDeviceSize>(validH) *
                              static_cast<VkDeviceSize>(kBytesPerPixel);
    StagingSlot slot = {};
    std::string error;
    if (!AcquireStagingSlot(context, need, &slot, &error))
    {
        LogMessage("Error", "UploadTileRegion failed to acquire staging: %s", error.empty() ? "unknown" : error.c_str());
        return false;
    }

    const VulkanFunctions& vk = context.functions;
    const VkDevice device = context.instance.device;
    void* mapped = nullptr;
    // Map the WHOLE slot (not just `need`) so the VK_WHOLE_SIZE flush below stays within the
    // mapped range — a reused ring slot has capacity >= need, so mapping `need` would leave
    // VK_WHOLE_SIZE flushing past the mapped range (VUID-VkMappedMemoryRange-size-01389).
    VkResult result = vk.MapMemory(device, slot.memory, 0, slot.capacity, 0, &mapped);
    if (result != VK_SUCCESS)
    {
        LogMessage("Error", "UploadTileRegion vkMapMemory=%d", static_cast<int>(result));
        std::lock_guard<std::mutex> lock(gUploadsMutex);
        ReturnStagingSlotLocked(slot);
        return false;
    }

    // Row-reversed crop reproduces the CPU bottom-up flip: staging row t receives
    // source row (validH-1-t). Combined with the C#-computed dstY = imageH -
    // panoPixelY - validH this is pixel-identical to BlitTileIntoBuffer.
    const size_t rowBytes = static_cast<size_t>(validW) * kBytesPerPixel;
    uint8_t* destination = static_cast<uint8_t*>(mapped);
    for (int t = 0; t < validH; ++t)
    {
        const uint8_t* source = tileRgba + static_cast<size_t>(validH - 1 - t) * static_cast<size_t>(tileStride);
        std::memcpy(destination + static_cast<size_t>(t) * rowBytes, source, rowBytes);
    }

    if (!slot.coherent)
    {
        VkMappedMemoryRange memoryRange = {};
        memoryRange.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
        memoryRange.memory = slot.memory;
        memoryRange.offset = 0;
        memoryRange.size = VK_WHOLE_SIZE;
        VkResult flushResult = vk.FlushMappedMemoryRanges(device, 1, &memoryRange);
        if (flushResult != VK_SUCCESS)
        {
            LogMessage("Error", "UploadTileRegion vkFlushMappedMemoryRanges=%d", static_cast<int>(flushResult));
            vk.UnmapMemory(device, slot.memory);
            std::lock_guard<std::mutex> lock(gUploadsMutex);
            ReturnStagingSlotLocked(slot);
            return false;
        }
    }
    vk.UnmapMemory(device, slot.memory);

    VkBufferImageCopy region = {};
    region.bufferOffset = 0;
    region.bufferRowLength = static_cast<uint32_t>(validW);
    region.bufferImageHeight = static_cast<uint32_t>(validH);
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    region.imageOffset.x = dstX;
    region.imageOffset.y = dstY;
    region.imageOffset.z = 0;
    region.imageExtent.width = static_cast<uint32_t>(validW);
    region.imageExtent.height = static_cast<uint32_t>(validH);
    region.imageExtent.depth = 1;

    bool appended = false;
    {
        std::lock_guard<std::mutex> lock(gAssembliesMutex);
        auto it = gAssemblies.find(key);
        if (it != gAssemblies.end())
        {
            it->second.regions.push_back(region);
            it->second.stagings.push_back(slot);
            appended = true;
        }
    }
    if (!appended)
    {
        // Aborted/released between the snapshot and here — return the slot.
        std::lock_guard<std::mutex> lock(gUploadsMutex);
        ReturnStagingSlotLocked(slot);
        return false;
    }
    return true;
}

NATIVE_TEXTURE_API void UNITY_INTERFACE_API FinishPanoAssembly(void* assemblyImage)
{
    if (!assemblyImage)
    {
        return;
    }
    const TextureKey key = static_cast<TextureKey>(reinterpret_cast<uintptr_t>(assemblyImage));

    PendingAssembly pending = {};
    bool found = false;
    {
        std::lock_guard<std::mutex> lock(gAssembliesMutex);
        auto it = gAssemblies.find(key);
        if (it != gAssemblies.end())
        {
            pending.context = it->second.context;
            pending.image = it->second.image;
            pending.regions = std::move(it->second.regions);
            pending.stagings = std::move(it->second.stagings);
            gAssemblies.erase(it);
            found = true;
        }
    }
    if (!found)
    {
        LogMessage("Warn", "FinishPanoAssembly ignored: unknown assembly %p", assemblyImage);
        return;
    }

    std::lock_guard<std::mutex> lock(gUploadsMutex);
    gPendingAssemblies.push_back(std::move(pending));
}

NATIVE_TEXTURE_API void UNITY_INTERFACE_API AbortImage(void* assemblyImage)
{
    if (!assemblyImage)
    {
        return;
    }
    const TextureKey key = static_cast<TextureKey>(reinterpret_cast<uintptr_t>(assemblyImage));

    // Drop the in-progress assembly (if still open) and return its stagings.
    AssemblyImage assembly = {};
    bool hadAssembly = false;
    {
        std::lock_guard<std::mutex> lock(gAssembliesMutex);
        auto it = gAssemblies.find(key);
        if (it != gAssemblies.end())
        {
            assembly = std::move(it->second);
            gAssemblies.erase(it);
            hadAssembly = true;
        }
    }
    if (hadAssembly && !assembly.stagings.empty())
    {
        std::lock_guard<std::mutex> lock(gUploadsMutex);
        for (const StagingSlot& slot : assembly.stagings) ReturnStagingSlotLocked(slot);
    }

    // Destroy the device image via the normal fence-gated Release path (also scans
    // gPendingAssemblies/gInFlightAssemblies for a committed-then-aborted race).
    Release(assemblyImage);
}

NATIVE_TEXTURE_API void UNITY_INTERFACE_API Release(void* texPtr)
{
    if (!texPtr)
    {
        return;
    }

    const TextureKey key = static_cast<TextureKey>(reinterpret_cast<uintptr_t>(texPtr));

    VulkanTextureResource resource = {};
    {
        std::lock_guard<std::mutex> lock(gTextureMutex);
        const auto iterator = gTextureResources.find(key);
        if (iterator == gTextureResources.end())
        {
            LogMessage("Warn", "Release ignored: Vulkan texture %p was not found", texPtr);
            return;
        }

        resource = iterator->second;
        gTextureResources.erase(iterator);
    }

    // Defensive: if this handle was still mid-assembly (disposed without Abort/Finish),
    // drop the in-progress record and return its stagings below under gUploadsMutex.
    std::vector<StagingSlot> orphanStagings;
    {
        std::lock_guard<std::mutex> lock(gAssembliesMutex);
        auto it = gAssemblies.find(key);
        if (it != gAssemblies.end())
        {
            orphanStagings = std::move(it->second.stagings);
            gAssemblies.erase(it);
        }
    }

    // If an upload for this image is still pending (Phase A done but Phase B
    // not yet run) OR in-flight (Phase B submitted but fence not signaled),
    // destroying the VkImage now would crash on Adreno (GPU still using it).
    //
    // - Pending: just remove from queue (no GPU work started) + free its
    //   staging immediately, then proceed to destroy the image.
    // - In-flight: MOVE the entries out under the lock, then wait the fence
    //   OUTSIDE the lock. Holding gUploadsMutex across a blocking vkWaitForFences
    //   would stall the render thread (which takes the same mutex at the top of
    //   every SubmitPendingUploadsCallback tick) for the whole GPU copy — a frame
    //   hitch on pano supersede. The fences signal from GPU completion independent
    //   of the render thread, so waiting outside the lock is safe (mirrors the
    //   swap-then-work discipline in ProcessPendingDestroys).
    std::vector<InFlightUpload> inflightUploads;
    std::vector<InFlightAssembly> inflightAssemblies;
    {
        std::lock_guard<std::mutex> lock(gUploadsMutex);
        for (auto it = gPendingUploads.begin(); it != gPendingUploads.end(); )
        {
            if (it->image == resource.image)
            {
                const VulkanFunctions& vk = it->context.functions;
                const VkDevice device = it->context.instance.device;
                if (it->stagingBuffer != VK_NULL_HANDLE) vk.DestroyBuffer(device, it->stagingBuffer, nullptr);
                if (it->stagingMemory != VK_NULL_HANDLE) vk.FreeMemory(device, it->stagingMemory, nullptr);
                it = gPendingUploads.erase(it);
            }
            else
            {
                ++it;
            }
        }
        for (auto it = gInFlightUploads.begin(); it != gInFlightUploads.end(); )
        {
            if (it->image == resource.image)
            {
                inflightUploads.push_back(std::move(*it));
                it = gInFlightUploads.erase(it);
            }
            else
            {
                ++it;
            }
        }

        for (const StagingSlot& slot : orphanStagings) ReturnStagingSlotLocked(slot);

        // Committed-but-not-submitted assemblies for this image: drop + return stagings.
        for (auto it = gPendingAssemblies.begin(); it != gPendingAssemblies.end(); )
        {
            if (it->image == resource.image)
            {
                for (const StagingSlot& slot : it->stagings) ReturnStagingSlotLocked(slot);
                it = gPendingAssemblies.erase(it);
            }
            else
            {
                ++it;
            }
        }
        for (auto it = gInFlightAssemblies.begin(); it != gInFlightAssemblies.end(); )
        {
            if (it->image == resource.image)
            {
                inflightAssemblies.push_back(std::move(*it));
                it = gInFlightAssemblies.erase(it);
            }
            else
            {
                ++it;
            }
        }
    }

    // Outside gUploadsMutex: wait the GPU fences (Adreno would crash if the image were
    // destroyed / the staging reused mid-copy) and destroy the transient command resources.
    for (InFlightUpload& upload : inflightUploads)
    {
        const VulkanFunctions& vk = upload.context.functions;
        const VkDevice device = upload.context.instance.device;
        if (upload.fence != VK_NULL_HANDLE)
        {
            WaitFenceBounded(vk, device, upload.fence, "Release/in-flight upload");
        }
        if (upload.commandBuffer != VK_NULL_HANDLE && upload.commandPool != VK_NULL_HANDLE)
            vk.FreeCommandBuffers(device, upload.commandPool, 1, &upload.commandBuffer);
        if (upload.commandPool != VK_NULL_HANDLE) vk.DestroyCommandPool(device, upload.commandPool, nullptr);
        if (upload.fence != VK_NULL_HANDLE)       vk.DestroyFence(device, upload.fence, nullptr);
        if (upload.stagingBuffer != VK_NULL_HANDLE) vk.DestroyBuffer(device, upload.stagingBuffer, nullptr);
        if (upload.stagingMemory != VK_NULL_HANDLE) vk.FreeMemory(device, upload.stagingMemory, nullptr);
    }
    for (InFlightAssembly& assembly : inflightAssemblies)
    {
        const VulkanFunctions& vk = assembly.context.functions;
        const VkDevice device = assembly.context.instance.device;
        if (assembly.fence != VK_NULL_HANDLE)
        {
            WaitFenceBounded(vk, device, assembly.fence, "Release/in-flight assembly");
        }
        if (assembly.commandBuffer != VK_NULL_HANDLE && assembly.commandPool != VK_NULL_HANDLE)
            vk.FreeCommandBuffers(device, assembly.commandPool, 1, &assembly.commandBuffer);
        if (assembly.commandPool != VK_NULL_HANDLE) vk.DestroyCommandPool(device, assembly.commandPool, nullptr);
        if (assembly.fence != VK_NULL_HANDLE)       vk.DestroyFence(device, assembly.fence, nullptr);
    }
    if (!inflightAssemblies.empty())
    {
        std::lock_guard<std::mutex> lock(gUploadsMutex);
        for (InFlightAssembly& assembly : inflightAssemblies)
        {
            for (const StagingSlot& slot : assembly.stagings) ReturnStagingSlotLocked(slot);
        }
    }

    VulkanContext context = {};
    if (SnapshotVulkanContext(&context) && context.valid)
    {
        // Deferred, fence-gated destruction: the actual vkDestroyImage runs on the render
        // thread (SubmitPendingUploadsCallback → ProcessPendingDestroys) once a graphics-
        // queue fence proves the GPU has retired Unity's last use of this image. No
        // vkQueueWaitIdle here — the previous per-release full-queue host stall on Unity's
        // primary graphics queue was a major frame-pacing and render-thread-stability cost
        // (it fired on every pano release, e.g. twice per timeline switch). The pump runs on
        // the next render event (every texture create fires one; shutdown flushes the rest).
        EnqueuePendingDestroy(context, resource);
    }
    else
    {
        // No live Vulkan context (device tearing down) — best-effort direct destroy.
        DestroyTextureResourceBestEffort(resource, nullptr, "Release");
    }
}

// Render-thread event entry. Called by Unity from the render thread when C#
// invokes `GL.IssuePluginEvent(GetRenderEventFunc(), eventID)`. Event IDs are
// configured during device init (see RefreshVulkanContextLocked) so that
// kEventIdSubmitUploads runs outside any render pass with graphics-queue
// access enabled — both required for our barrier + copy + submit sequence.
NATIVE_TEXTURE_API void UNITY_INTERFACE_API OnRenderEvent(int eventID)
{
    if (eventID == kEventIdSubmitUploads)
    {
        // SubmitPendingUploadsCallback's signature takes (int, void*) because
        // historically it was wired through AccessQueue's UnityRenderingEvent-
        // AndData type. The userData arg is unused; nullptr is safe.
        SubmitPendingUploadsCallback(eventID, nullptr);
    }
}

// Returns the function pointer that C# passes to GL.IssuePluginEvent. The
// exposed function MUST have signature `void OnRenderEvent(int eventID)` —
// no userData, no return value (UnityRenderingEvent typedef). Unity calls
// this from the render thread between draw submissions.
NATIVE_TEXTURE_API UnityRenderingEvent UNITY_INTERFACE_API GetRenderEventFunc()
{
    return OnRenderEvent;
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
    VulkanContext context = {};
    {
        std::lock_guard<std::mutex> lock(gUnityStateMutex);
        unityGraphics = gUnityGraphics;
        context = gVulkanContext;
    }

    FlushPendingDestroysBlocking();
    DestroyAssemblyResourcesBlocking();
    DrainOutstandingTextures("Plugin unload", context.valid ? &context : nullptr);

    if (unityGraphics)
    {
        unityGraphics->UnregisterDeviceEventCallback(OnGraphicsDeviceEvent);
    }

    std::lock_guard<std::mutex> lock(gUnityStateMutex);
    ClearVulkanInterfacesLocked();
    gUnityGraphics = nullptr;
    gUnityInterfaces = nullptr;
}
