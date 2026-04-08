using System;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.Rendering;

namespace NativeTexture
{
    public static class NativeTextureEntry
    {
        private static GraphicsDeviceType s_cachedGraphicsDeviceType;
        private static bool s_hasCachedGraphicsDeviceType;
        private static int s_mainThreadId;

        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.BeforeSceneLoad)]
        private static void CaptureRuntimeState()
        {
            TryCaptureGraphicsDeviceTypeOnMainThread();
        }

        public static NativeTextureBackend Backend
        {
            get
            {
                TryCaptureGraphicsDeviceTypeOnMainThread();
                if (!s_hasCachedGraphicsDeviceType)
                {
                    return NativeTextureBackend.None;
                }

                return ToBackend(s_cachedGraphicsDeviceType);
            }
        }

        public static bool IsSupported =>
            Backend == NativeTextureBackend.Metal ||
            Backend == NativeTextureBackend.Vulkan ||
            Backend == NativeTextureBackend.D3D12;

        public static void Initialize()
        {
            EnsureGraphicsDeviceTypeOnMainThread();

            switch (Backend)
            {
                case NativeTextureBackend.Metal:
                    NativeTextureMetal.Initialize();
                    break;
                case NativeTextureBackend.Vulkan:
                    NativeTextureVulkan.Initialize();
                    break;
                case NativeTextureBackend.D3D12:
                    NativeTextureD3D12.Initialize();
                    break;
                default:
                    throw new NotSupportedException(
                        $"NativeTexture does not support the current graphics backend: {s_cachedGraphicsDeviceType}.");
            }
        }

        public static Task<NativeTextureDecodedImage> DecodeAsync(
            byte[] encodedData,
            CancellationToken cancellationToken = default)
        {
            return DecodeAsync(encodedData, false, cancellationToken);
        }

        public static Task<NativeTextureDecodedImage> DecodeAsync(
            byte[] encodedData,
            bool flipY)
        {
            return DecodeAsync(encodedData, flipY, CancellationToken.None);
        }

        public static Task<NativeTextureDecodedImage> DecodeAsync(
            byte[] encodedData,
            bool flipY,
            CancellationToken cancellationToken)
        {
            switch (GetValidatedBackend())
            {
                case NativeTextureBackend.Metal:
                    return NativeTextureMetal.DecodeAsync(encodedData, flipY, cancellationToken);
                case NativeTextureBackend.Vulkan:
                    return NativeTextureVulkan.DecodeAsync(encodedData, flipY, cancellationToken);
                case NativeTextureBackend.D3D12:
                    return NativeTextureD3D12.DecodeAsync(encodedData, flipY, cancellationToken);
                default:
                    throw CreateUnsupportedBackendException();
            }
        }

        public static Task<NativeTextureDecodedImage> DecodeFileAsync(
            string filePath,
            CancellationToken cancellationToken = default)
        {
            return DecodeFileAsync(filePath, false, cancellationToken);
        }

        public static Task<NativeTextureDecodedImage> DecodeFileAsync(
            string filePath,
            bool flipY)
        {
            return DecodeFileAsync(filePath, flipY, CancellationToken.None);
        }

        public static Task<NativeTextureDecodedImage> DecodeFileAsync(
            string filePath,
            bool flipY,
            CancellationToken cancellationToken)
        {
            switch (GetValidatedBackend())
            {
                case NativeTextureBackend.Metal:
                    return NativeTextureMetal.DecodeFileAsync(filePath, flipY, cancellationToken);
                case NativeTextureBackend.Vulkan:
                    return NativeTextureVulkan.DecodeFileAsync(filePath, flipY, cancellationToken);
                case NativeTextureBackend.D3D12:
                    return NativeTextureD3D12.DecodeFileAsync(filePath, flipY, cancellationToken);
                default:
                    throw CreateUnsupportedBackendException();
            }
        }

        public static async Task<INativeTextureHandle> CreateAsync(
            NativeTextureDecodedImage decodedImage,
            bool linear = false,
            CancellationToken cancellationToken = default)
        {
            if (decodedImage == null)
            {
                throw new ArgumentNullException(nameof(decodedImage));
            }

            return await CreateAsync(decodedImage.Rgba, decodedImage.Width, decodedImage.Height, linear, cancellationToken)
                .ConfigureAwait(false);
        }

        public static async Task<INativeTextureHandle> CreateAsync(
            byte[] rgba,
            int width,
            int height,
            bool linear = false,
            CancellationToken cancellationToken = default)
        {
            switch (GetValidatedBackend())
            {
                case NativeTextureBackend.Metal:
                    return await NativeTextureMetal.CreateAsync(rgba, width, height, linear, cancellationToken)
                        .ConfigureAwait(false);
                case NativeTextureBackend.Vulkan:
                    return await NativeTextureVulkan.CreateAsync(rgba, width, height, linear, cancellationToken)
                        .ConfigureAwait(false);
                case NativeTextureBackend.D3D12:
                    return await NativeTextureD3D12.CreateAsync(rgba, width, height, linear, cancellationToken)
                        .ConfigureAwait(false);
                default:
                    throw CreateUnsupportedBackendException();
            }
        }

        public static async Task<INativeTextureHandle> LoadTextureAsync(
            byte[] encodedData,
            bool linear = false,
            CancellationToken cancellationToken = default)
        {
            return await LoadTextureAsync(encodedData, linear, false, cancellationToken).ConfigureAwait(false);
        }

        public static Task<INativeTextureHandle> LoadTextureAsync(
            byte[] encodedData,
            bool linear,
            bool flipY)
        {
            return LoadTextureAsync(encodedData, linear, flipY, CancellationToken.None);
        }

        public static async Task<INativeTextureHandle> LoadTextureAsync(
            byte[] encodedData,
            bool linear,
            bool flipY,
            CancellationToken cancellationToken)
        {
            switch (GetValidatedBackend())
            {
                case NativeTextureBackend.Metal:
                    return await NativeTextureMetal.LoadTextureAsync(encodedData, linear, flipY, cancellationToken)
                        .ConfigureAwait(false);
                case NativeTextureBackend.Vulkan:
                    return await NativeTextureVulkan.LoadTextureAsync(encodedData, linear, flipY, cancellationToken)
                        .ConfigureAwait(false);
                case NativeTextureBackend.D3D12:
                    return await NativeTextureD3D12.LoadTextureAsync(encodedData, linear, flipY, cancellationToken)
                        .ConfigureAwait(false);
                default:
                    throw CreateUnsupportedBackendException();
            }
        }

        public static async Task<INativeTextureHandle> LoadTextureFromFileAsync(
            string filePath,
            bool linear = false,
            CancellationToken cancellationToken = default)
        {
            return await LoadTextureFromFileAsync(filePath, linear, false, cancellationToken).ConfigureAwait(false);
        }

        public static Task<INativeTextureHandle> LoadTextureFromFileAsync(
            string filePath,
            bool linear,
            bool flipY)
        {
            return LoadTextureFromFileAsync(filePath, linear, flipY, CancellationToken.None);
        }

        public static async Task<INativeTextureHandle> LoadTextureFromFileAsync(
            string filePath,
            bool linear,
            bool flipY,
            CancellationToken cancellationToken)
        {
            switch (GetValidatedBackend())
            {
                case NativeTextureBackend.Metal:
                    return await NativeTextureMetal.LoadTextureFromFileAsync(filePath, linear, flipY, cancellationToken)
                        .ConfigureAwait(false);
                case NativeTextureBackend.Vulkan:
                    return await NativeTextureVulkan.LoadTextureFromFileAsync(filePath, linear, flipY, cancellationToken)
                        .ConfigureAwait(false);
                case NativeTextureBackend.D3D12:
                    return await NativeTextureD3D12.LoadTextureFromFileAsync(filePath, linear, flipY, cancellationToken)
                        .ConfigureAwait(false);
                default:
                    throw CreateUnsupportedBackendException();
            }
        }

        public static Task SaveToFileAsync(
            string filePath,
            NativeTextureDecodedImage decodedImage,
            CancellationToken cancellationToken = default)
        {
            switch (GetValidatedBackend())
            {
                case NativeTextureBackend.Metal:
                    return NativeTextureMetal.SaveToFileAsync(filePath, decodedImage, cancellationToken);
                case NativeTextureBackend.Vulkan:
                    return NativeTextureVulkan.SaveToFileAsync(filePath, decodedImage, cancellationToken);
                case NativeTextureBackend.D3D12:
                    return NativeTextureD3D12.SaveToFileAsync(filePath, decodedImage, cancellationToken);
                default:
                    throw CreateUnsupportedBackendException();
            }
        }

        public static Task SaveToFileAsync(
            string filePath,
            byte[] rgba,
            int width,
            int height,
            CancellationToken cancellationToken = default)
        {
            switch (GetValidatedBackend())
            {
                case NativeTextureBackend.Metal:
                    return NativeTextureMetal.SaveToFileAsync(filePath, rgba, width, height, cancellationToken);
                case NativeTextureBackend.Vulkan:
                    return NativeTextureVulkan.SaveToFileAsync(filePath, rgba, width, height, cancellationToken);
                case NativeTextureBackend.D3D12:
                    return NativeTextureD3D12.SaveToFileAsync(filePath, rgba, width, height, cancellationToken);
                default:
                    throw CreateUnsupportedBackendException();
            }
        }

        private static NativeTextureBackend GetValidatedBackend()
        {
            NativeTextureBackend backend = Backend;
            if (backend == NativeTextureBackend.None)
            {
                throw CreateUnsupportedBackendException();
            }

            return backend;
        }

        private static NotSupportedException CreateUnsupportedBackendException()
        {
            string detail = s_hasCachedGraphicsDeviceType
                ? s_cachedGraphicsDeviceType.ToString()
                : "unknown (call NativeTexture.Initialize() on the Unity main thread first)";

            return new NotSupportedException(
                $"NativeTexture only supports Metal, Vulkan and Direct3D12. Current graphics backend: {detail}.");
        }

        private static void TryCaptureGraphicsDeviceTypeOnMainThread()
        {
            if (s_hasCachedGraphicsDeviceType || !IsMainThread())
            {
                return;
            }

            s_cachedGraphicsDeviceType = SystemInfo.graphicsDeviceType;
            s_hasCachedGraphicsDeviceType = true;
            s_mainThreadId = Thread.CurrentThread.ManagedThreadId;
        }

        private static void EnsureGraphicsDeviceTypeOnMainThread()
        {
            if (!IsMainThread())
            {
                throw new InvalidOperationException("NativeTexture.Initialize must be called on the Unity main thread.");
            }

            TryCaptureGraphicsDeviceTypeOnMainThread();
            if (!s_hasCachedGraphicsDeviceType)
            {
                throw new InvalidOperationException("NativeTexture failed to capture Unity graphics backend.");
            }
        }

        private static bool IsMainThread()
        {
            if (s_mainThreadId != 0)
            {
                return Thread.CurrentThread.ManagedThreadId == s_mainThreadId;
            }

            return SynchronizationContext.Current != null;
        }

        private static NativeTextureBackend ToBackend(GraphicsDeviceType graphicsDeviceType)
        {
            switch (graphicsDeviceType)
            {
                case GraphicsDeviceType.Metal:
                    return NativeTextureBackend.Metal;
                case GraphicsDeviceType.Vulkan:
                    return NativeTextureBackend.Vulkan;
                case GraphicsDeviceType.Direct3D12:
                    return NativeTextureBackend.D3D12;
                default:
                    return NativeTextureBackend.None;
            }
        }
    }
}
