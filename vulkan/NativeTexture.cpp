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

struct CreateTextureRequest
{
    QueueRequestState state;
    VulkanContext context;
    const unsigned char* rgba = nullptr;
    int width = 0;
    int height = 0;
    bool useSrgb = false;
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
};

struct ReleaseTextureRequest
{
    QueueRequestState state;
    VulkanContext context;
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
};

using TextureKey = uint64_t;

std::mutex gUnityStateMutex;
std::mutex gTextureMutex;
#ifndef STBI_THREAD_LOCAL
std::mutex gStbiLoadMutex;
#endif
IUnityInterfaces* gUnityInterfaces = nullptr;
IUnityGraphics* gUnityGraphics = nullptr;
IUnityGraphicsVulkanV2* gUnityGraphicsVulkanV2 = nullptr;
IUnityGraphicsVulkan* gUnityGraphicsVulkan = nullptr;
VulkanContext gVulkanContext = {};
std::unordered_map<TextureKey, VulkanTextureResource> gTextureResources;

bool ExecuteQueueRequest(QueueRequestState& state,
                         const VulkanContext& context,
                         UnityRenderingEventAndData callback,
                         void* userData);
void UNITY_INTERFACE_API ReleaseTextureQueueCallback(int eventId, void* userData);

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

bool CreateTextureOnQueue(CreateTextureRequest* request)
{
    const VulkanContext& context = request->context;
    const VulkanFunctions& vk = context.functions;
    const VkDevice device = context.instance.device;

    const VkDeviceSize imageSize = static_cast<VkDeviceSize>(request->width) *
                                   static_cast<VkDeviceSize>(request->height) *
                                   static_cast<VkDeviceSize>(kBytesPerPixel);

    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory imageMemory = VK_NULL_HANDLE;
    VkBuffer stagingBuffer = VK_NULL_HANDLE;
    VkDeviceMemory stagingMemory = VK_NULL_HANDLE;
    VkCommandPool commandPool = VK_NULL_HANDLE;
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    VkMemoryPropertyFlags stagingFlags = 0;

    auto fail = [&](const char* action, VkResult result) -> bool
    {
        request->state.error = std::string(action) + " failed with VkResult " + std::to_string(static_cast<int>(result));
        return false;
    };

    auto cleanup = [&]()
    {
        if (fence != VK_NULL_HANDLE)
        {
            vk.DestroyFence(device, fence, nullptr);
            fence = VK_NULL_HANDLE;
        }
        if (commandBuffer != VK_NULL_HANDLE && commandPool != VK_NULL_HANDLE)
        {
            vk.FreeCommandBuffers(device, commandPool, 1, &commandBuffer);
            commandBuffer = VK_NULL_HANDLE;
        }
        if (commandPool != VK_NULL_HANDLE)
        {
            vk.DestroyCommandPool(device, commandPool, nullptr);
            commandPool = VK_NULL_HANDLE;
        }
        if (stagingBuffer != VK_NULL_HANDLE)
        {
            vk.DestroyBuffer(device, stagingBuffer, nullptr);
            stagingBuffer = VK_NULL_HANDLE;
        }
        if (stagingMemory != VK_NULL_HANDLE)
        {
            vk.FreeMemory(device, stagingMemory, nullptr);
            stagingMemory = VK_NULL_HANDLE;
        }
    };

    VkImageCreateInfo imageInfo = {};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.format = request->useSrgb ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM;
    imageInfo.extent.width = static_cast<uint32_t>(request->width);
    imageInfo.extent.height = static_cast<uint32_t>(request->height);
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VkResult result = vk.CreateImage(device, &imageInfo, nullptr, &image);
    if (result != VK_SUCCESS)
    {
        return fail("vkCreateImage", result);
    }

    VkMemoryRequirements imageRequirements = {};
    vk.GetImageMemoryRequirements(device, image, &imageRequirements);

    uint32_t imageMemoryTypeIndex = 0;
    if (!FindMemoryType(context, imageRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &imageMemoryTypeIndex, nullptr))
    {
        request->state.error = "No compatible device-local Vulkan memory type for image";
        vk.DestroyImage(device, image, nullptr);
        return false;
    }

    VkMemoryAllocateInfo imageMemoryInfo = {};
    imageMemoryInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    imageMemoryInfo.allocationSize = imageRequirements.size;
    imageMemoryInfo.memoryTypeIndex = imageMemoryTypeIndex;

    result = vk.AllocateMemory(device, &imageMemoryInfo, nullptr, &imageMemory);
    if (result != VK_SUCCESS)
    {
        vk.DestroyImage(device, image, nullptr);
        return fail("vkAllocateMemory(image)", result);
    }

    result = vk.BindImageMemory(device, image, imageMemory, 0);
    if (result != VK_SUCCESS)
    {
        vk.FreeMemory(device, imageMemory, nullptr);
        vk.DestroyImage(device, image, nullptr);
        return fail("vkBindImageMemory", result);
    }

    VkBufferCreateInfo stagingBufferInfo = {};
    stagingBufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    stagingBufferInfo.size = imageSize;
    stagingBufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    stagingBufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    result = vk.CreateBuffer(device, &stagingBufferInfo, nullptr, &stagingBuffer);
    if (result != VK_SUCCESS)
    {
        vk.FreeMemory(device, imageMemory, nullptr);
        vk.DestroyImage(device, image, nullptr);
        return fail("vkCreateBuffer", result);
    }

    VkMemoryRequirements stagingRequirements = {};
    vk.GetBufferMemoryRequirements(device, stagingBuffer, &stagingRequirements);

    uint32_t stagingMemoryTypeIndex = 0;
    if (!FindMemoryType(context, stagingRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &stagingMemoryTypeIndex, &stagingFlags))
    {
        cleanup();
        vk.FreeMemory(device, imageMemory, nullptr);
        vk.DestroyImage(device, image, nullptr);
        request->state.error = "No compatible host-visible Vulkan memory type for staging buffer";
        return false;
    }

    VkMemoryAllocateInfo stagingMemoryInfo = {};
    stagingMemoryInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    stagingMemoryInfo.allocationSize = stagingRequirements.size;
    stagingMemoryInfo.memoryTypeIndex = stagingMemoryTypeIndex;

    result = vk.AllocateMemory(device, &stagingMemoryInfo, nullptr, &stagingMemory);
    if (result != VK_SUCCESS)
    {
        cleanup();
        vk.FreeMemory(device, imageMemory, nullptr);
        vk.DestroyImage(device, image, nullptr);
        return fail("vkAllocateMemory(staging)", result);
    }

    result = vk.BindBufferMemory(device, stagingBuffer, stagingMemory, 0);
    if (result != VK_SUCCESS)
    {
        cleanup();
        vk.FreeMemory(device, imageMemory, nullptr);
        vk.DestroyImage(device, image, nullptr);
        return fail("vkBindBufferMemory", result);
    }

    void* mappedMemory = nullptr;
    result = vk.MapMemory(device, stagingMemory, 0, imageSize, 0, &mappedMemory);
    if (result != VK_SUCCESS)
    {
        cleanup();
        vk.FreeMemory(device, imageMemory, nullptr);
        vk.DestroyImage(device, image, nullptr);
        return fail("vkMapMemory", result);
    }

    std::memcpy(mappedMemory, request->rgba, static_cast<size_t>(imageSize));
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
            cleanup();
            vk.FreeMemory(device, imageMemory, nullptr);
            vk.DestroyImage(device, image, nullptr);
            return fail("vkFlushMappedMemoryRanges", result);
        }
    }
    vk.UnmapMemory(device, stagingMemory);

    VkCommandPoolCreateInfo commandPoolInfo = {};
    commandPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    commandPoolInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    commandPoolInfo.queueFamilyIndex = context.instance.queueFamilyIndex;

    result = vk.CreateCommandPool(device, &commandPoolInfo, nullptr, &commandPool);
    if (result != VK_SUCCESS)
    {
        cleanup();
        vk.FreeMemory(device, imageMemory, nullptr);
        vk.DestroyImage(device, image, nullptr);
        return fail("vkCreateCommandPool", result);
    }

    VkCommandBufferAllocateInfo commandBufferInfo = {};
    commandBufferInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    commandBufferInfo.commandPool = commandPool;
    commandBufferInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    commandBufferInfo.commandBufferCount = 1;

    result = vk.AllocateCommandBuffers(device, &commandBufferInfo, &commandBuffer);
    if (result != VK_SUCCESS)
    {
        cleanup();
        vk.FreeMemory(device, imageMemory, nullptr);
        vk.DestroyImage(device, image, nullptr);
        return fail("vkAllocateCommandBuffers", result);
    }

    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    result = vk.BeginCommandBuffer(commandBuffer, &beginInfo);
    if (result != VK_SUCCESS)
    {
        cleanup();
        vk.FreeMemory(device, imageMemory, nullptr);
        vk.DestroyImage(device, image, nullptr);
        return fail("vkBeginCommandBuffer", result);
    }

    VkImageMemoryBarrier uploadBarrier = {};
    uploadBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    uploadBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    uploadBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    uploadBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    uploadBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    uploadBarrier.image = image;
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
                          0,
                          0, nullptr,
                          0, nullptr,
                          1, &uploadBarrier);

    VkBufferImageCopy copyRegion = {};
    copyRegion.bufferOffset = 0;
    copyRegion.bufferRowLength = 0;
    copyRegion.bufferImageHeight = 0;
    copyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copyRegion.imageSubresource.mipLevel = 0;
    copyRegion.imageSubresource.baseArrayLayer = 0;
    copyRegion.imageSubresource.layerCount = 1;
    copyRegion.imageExtent.width = static_cast<uint32_t>(request->width);
    copyRegion.imageExtent.height = static_cast<uint32_t>(request->height);
    copyRegion.imageExtent.depth = 1;

    vk.CmdCopyBufferToImage(commandBuffer,
                            stagingBuffer,
                            image,
                            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                            1,
                            &copyRegion);

    VkImageMemoryBarrier shaderReadBarrier = {};
    shaderReadBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    shaderReadBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    shaderReadBarrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    shaderReadBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    shaderReadBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    shaderReadBarrier.image = image;
    shaderReadBarrier.subresourceRange = uploadBarrier.subresourceRange;
    shaderReadBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    shaderReadBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    vk.CmdPipelineBarrier(commandBuffer,
                          VK_PIPELINE_STAGE_TRANSFER_BIT,
                          VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                          0,
                          0, nullptr,
                          0, nullptr,
                          1, &shaderReadBarrier);

    result = vk.EndCommandBuffer(commandBuffer);
    if (result != VK_SUCCESS)
    {
        cleanup();
        vk.FreeMemory(device, imageMemory, nullptr);
        vk.DestroyImage(device, image, nullptr);
        return fail("vkEndCommandBuffer", result);
    }

    VkFenceCreateInfo fenceInfo = {};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;

    result = vk.CreateFence(device, &fenceInfo, nullptr, &fence);
    if (result != VK_SUCCESS)
    {
        cleanup();
        vk.FreeMemory(device, imageMemory, nullptr);
        vk.DestroyImage(device, image, nullptr);
        return fail("vkCreateFence", result);
    }

    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    result = vk.QueueSubmit(context.instance.graphicsQueue, 1, &submitInfo, fence);
    if (result != VK_SUCCESS)
    {
        cleanup();
        vk.FreeMemory(device, imageMemory, nullptr);
        vk.DestroyImage(device, image, nullptr);
        return fail("vkQueueSubmit", result);
    }

    result = vk.WaitForFences(device, 1, &fence, VK_TRUE, std::numeric_limits<uint64_t>::max());
    if (result != VK_SUCCESS)
    {
        cleanup();
        vk.FreeMemory(device, imageMemory, nullptr);
        vk.DestroyImage(device, image, nullptr);
        return fail("vkWaitForFences", result);
    }

    cleanup();

    request->image = image;
    request->memory = imageMemory;
    return true;
}

void UNITY_INTERFACE_API CreateTextureQueueCallback(int /*eventId*/, void* userData)
{
    CreateTextureRequest* request = static_cast<CreateTextureRequest*>(userData);
    if (!request)
    {
        return;
    }

    const bool success = CreateTextureOnQueue(request);
    CompleteRequest(request->state, success, request->state.error);
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
    if (!ValidateImageArgs(rgba, width, height, "Create"))
    {
        return nullptr;
    }

    VulkanContext context = {};
    if (!SnapshotVulkanContext(&context))
    {
        return nullptr;
    }

    CreateTextureRequest request = {};
    request.context = context;
    request.rgba = rgba;
    request.width = width;
    request.height = height;
    request.useSrgb = useSrgb;

    if (!ExecuteQueueRequest(request.state, context, CreateTextureQueueCallback, &request))
    {
        LogMessage("Error", "Create failed: %s", request.state.error.empty() ? "unknown queue access error" : request.state.error.c_str());
        return nullptr;
    }

    VkImage* externalImage = new (std::nothrow) VkImage(request.image);
    if (!externalImage)
    {
        ReleaseTextureRequest releaseRequest = {};
        releaseRequest.context = context;
        releaseRequest.image = request.image;
        releaseRequest.memory = request.memory;

        if (!ExecuteQueueRequest(releaseRequest.state, context, ReleaseTextureQueueCallback, &releaseRequest))
        {
            LogMessage("Error", "Create cleanup failed after handle allocation failure: %s",
                       releaseRequest.state.error.empty() ? "unknown queue access error" : releaseRequest.state.error.c_str());
        }

        LogMessage("Error", "Create failed: unable to allocate external VkImage handle");
        return nullptr;
    }

    VulkanTextureResource resource = {};
    resource.externalImage = externalImage;
    resource.image = request.image;
    resource.memory = request.memory;
    resource.context = context;
    resource.width = static_cast<uint32_t>(width);
    resource.height = static_cast<uint32_t>(height);

    const TextureKey key = TextureKeyFromExternalImage(externalImage);
    {
        std::lock_guard<std::mutex> lock(gTextureMutex);
        gTextureResources[key] = resource;
    }

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

NATIVE_TEXTURE_API void UNITY_INTERFACE_API Release(void* texPtr)
{
    if (!texPtr)
    {
        return;
    }

    VulkanTextureResource resource = {};
    {
        std::lock_guard<std::mutex> lock(gTextureMutex);
        const TextureKey key = static_cast<TextureKey>(reinterpret_cast<uintptr_t>(texPtr));
        const auto iterator = gTextureResources.find(key);
        if (iterator == gTextureResources.end())
        {
            LogMessage("Warn", "Release ignored: Vulkan texture %p was not found", texPtr);
            return;
        }

        resource = iterator->second;
        gTextureResources.erase(iterator);
    }

    VulkanContext context = {};
    const VulkanContext* preferredContext = nullptr;
    if (SnapshotVulkanContext(&context))
    {
        preferredContext = &context;
    }

    DestroyTextureResourceBestEffort(resource, preferredContext, "Release");
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
