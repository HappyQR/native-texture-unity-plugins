using System;
using System.IO;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.Rendering;

namespace NativeTexture
{
    public static class NativeTextureVulkan
    {
        private const string PluginName = "NativeTexture";

        private const int BytesPerPixel = 4;
        private static SynchronizationContext s_mainThreadContext;
        private static int s_mainThreadId;
        private static GraphicsDeviceType s_cachedGraphicsDeviceType;
        private static bool s_hasCachedGraphicsDeviceType;
        private static bool s_pluginLoaded;

        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.BeforeSceneLoad)]
        private static void CaptureMainThreadContext()
        {
            TryCaptureMainThreadContext();
        }

        public static bool IsSupported
        {
            get
            {
                TryCaptureMainThreadContext();
                if (!s_hasCachedGraphicsDeviceType && IsMainThread())
                {
                    RefreshGraphicsDeviceTypeOnMainThread();
                }

                return s_hasCachedGraphicsDeviceType && s_cachedGraphicsDeviceType == GraphicsDeviceType.Vulkan;
            }
        }

        public static void Initialize()
        {
            EnsureMainThreadContext();
            EnsurePluginLoadedOnMainThread();
            EnsureVulkanBackendOnMainThread();
        }

        public static async Task<NativeTextureDecodedImage> DecodeAsync(
            byte[] encodedData,
            CancellationToken cancellationToken = default)
        {
            return await DecodeAsync(encodedData, false, cancellationToken).ConfigureAwait(false);
        }

        public static Task<NativeTextureDecodedImage> DecodeAsync(
            byte[] encodedData,
            bool flipY)
        {
            return DecodeAsync(encodedData, flipY, CancellationToken.None);
        }

        public static async Task<NativeTextureDecodedImage> DecodeAsync(
            byte[] encodedData,
            bool flipY,
            CancellationToken cancellationToken)
        {
            TryCaptureMainThreadContext();
            ValidateEncodedData(encodedData, nameof(encodedData));
            await EnsurePluginLoadedAsync(cancellationToken).ConfigureAwait(false);

            return await Task.Run(() =>
            {
                cancellationToken.ThrowIfCancellationRequested();

                GCHandle encodedHandle = default;
                IntPtr rgbaPtr = IntPtr.Zero;

                try
                {
                    encodedHandle = GCHandle.Alloc(encodedData, GCHandleType.Pinned);
                    rgbaPtr = DecodeWithOptions(
                        encodedHandle.AddrOfPinnedObject(),
                        encodedData.Length,
                        out int width,
                        out int height,
                        flipY);
                    if (rgbaPtr == IntPtr.Zero)
                    {
                        throw new InvalidOperationException("Native Decode returned a null RGBA buffer.");
                    }

                    if (cancellationToken.IsCancellationRequested)
                    {
                        Free(rgbaPtr);
                        rgbaPtr = IntPtr.Zero;
                        cancellationToken.ThrowIfCancellationRequested();
                    }

                    int byteCount = checked(width * height * BytesPerPixel);
                    byte[] rgba = new byte[byteCount];
                    Marshal.Copy(rgbaPtr, rgba, 0, byteCount);
                    cancellationToken.ThrowIfCancellationRequested();

                    return new NativeTextureDecodedImage(rgba, width, height);
                }
                finally
                {
                    if (rgbaPtr != IntPtr.Zero)
                    {
                        Free(rgbaPtr);
                    }

                    if (encodedHandle.IsAllocated)
                    {
                        encodedHandle.Free();
                    }
                }
            }, cancellationToken).ConfigureAwait(false);
        }

        public static async Task<NativeTextureDecodedImage> DecodeFileAsync(
            string filePath,
            CancellationToken cancellationToken = default)
        {
            return await DecodeFileAsync(filePath, false, cancellationToken).ConfigureAwait(false);
        }

        public static Task<NativeTextureDecodedImage> DecodeFileAsync(
            string filePath,
            bool flipY)
        {
            return DecodeFileAsync(filePath, flipY, CancellationToken.None);
        }

        public static async Task<NativeTextureDecodedImage> DecodeFileAsync(
            string filePath,
            bool flipY,
            CancellationToken cancellationToken)
        {
            TryCaptureMainThreadContext();

            if (string.IsNullOrWhiteSpace(filePath))
            {
                throw new ArgumentException("File path cannot be empty.", nameof(filePath));
            }

            byte[] encodedData = await Task.Run(() =>
            {
                cancellationToken.ThrowIfCancellationRequested();
                return File.ReadAllBytes(filePath);
            }, cancellationToken).ConfigureAwait(false);

            cancellationToken.ThrowIfCancellationRequested();
            return await DecodeAsync(encodedData, flipY, cancellationToken).ConfigureAwait(false);
        }

        public static Task<NativeTextureVulkanTexture> CreateAsync(
            NativeTextureDecodedImage decodedImage,
            bool linear = false,
            CancellationToken cancellationToken = default)
        {
            if (decodedImage == null)
            {
                throw new ArgumentNullException(nameof(decodedImage));
            }

            return CreateAsync(decodedImage.Rgba, decodedImage.Width, decodedImage.Height, linear, cancellationToken);
        }

        public static async Task<NativeTextureVulkanTexture> CreateAsync(
            byte[] rgba,
            int width,
            int height,
            bool linear = false,
            CancellationToken cancellationToken = default)
        {
            TryCaptureMainThreadContext();
            ValidateRgbaBuffer(rgba, width, height, nameof(rgba));
            await EnsurePluginLoadedAsync(cancellationToken).ConfigureAwait(false);
            bool useSrgb = await RunOnMainThreadAsync(() =>
            {
                EnsureVulkanBackendOnMainThread();
                return ShouldUseSrgbNativeFormat(linear);
            }, cancellationToken).ConfigureAwait(false);

            IntPtr nativeTexture = await Task.Run(() =>
            {
                cancellationToken.ThrowIfCancellationRequested();

                GCHandle rgbaHandle = default;
                try
                {
                    rgbaHandle = GCHandle.Alloc(rgba, GCHandleType.Pinned);
                    IntPtr texture = CreateWithOptions(rgbaHandle.AddrOfPinnedObject(), width, height, useSrgb);
                    if (texture == IntPtr.Zero)
                    {
                        throw new InvalidOperationException("Native Create returned a null Vulkan texture.");
                    }

                    if (cancellationToken.IsCancellationRequested)
                    {
                        Release(texture);
                        texture = IntPtr.Zero;
                        cancellationToken.ThrowIfCancellationRequested();
                    }

                    return texture;
                }
                finally
                {
                    if (rgbaHandle.IsAllocated)
                    {
                        rgbaHandle.Free();
                    }
                }
            }, cancellationToken).ConfigureAwait(false);

            return await WrapNativeTextureAsync(nativeTexture, width, height, linear, cancellationToken).ConfigureAwait(false);
        }

        public static async Task<NativeTextureVulkanTexture> LoadTextureAsync(
            byte[] encodedData,
            bool linear = false,
            CancellationToken cancellationToken = default)
        {
            return await LoadTextureAsync(encodedData, linear, false, cancellationToken).ConfigureAwait(false);
        }

        public static Task<NativeTextureVulkanTexture> LoadTextureAsync(
            byte[] encodedData,
            bool linear,
            bool flipY)
        {
            return LoadTextureAsync(encodedData, linear, flipY, CancellationToken.None);
        }

        public static async Task<NativeTextureVulkanTexture> LoadTextureAsync(
            byte[] encodedData,
            bool linear,
            bool flipY,
            CancellationToken cancellationToken)
        {
            TryCaptureMainThreadContext();
            ValidateEncodedData(encodedData, nameof(encodedData));
            await EnsurePluginLoadedAsync(cancellationToken).ConfigureAwait(false);
            bool useSrgb = await RunOnMainThreadAsync(() =>
            {
                EnsureVulkanBackendOnMainThread();
                return ShouldUseSrgbNativeFormat(linear);
            }, cancellationToken).ConfigureAwait(false);

            NativeCreateResult nativeResult = await Task.Run(() =>
            {
                cancellationToken.ThrowIfCancellationRequested();

                GCHandle encodedHandle = default;
                try
                {
                    encodedHandle = GCHandle.Alloc(encodedData, GCHandleType.Pinned);
                    IntPtr texture = CreateFromEncodedWithOptions(
                        encodedHandle.AddrOfPinnedObject(),
                        encodedData.Length,
                        out int decodedWidth,
                        out int decodedHeight,
                        useSrgb,
                        flipY);
                    if (texture == IntPtr.Zero)
                    {
                        throw new InvalidOperationException("Native CreateFromEncoded returned a null Vulkan texture.");
                    }

                    if (cancellationToken.IsCancellationRequested)
                    {
                        Release(texture);
                        texture = IntPtr.Zero;
                        cancellationToken.ThrowIfCancellationRequested();
                    }

                    return new NativeCreateResult(texture, decodedWidth, decodedHeight);
                }
                finally
                {
                    if (encodedHandle.IsAllocated)
                    {
                        encodedHandle.Free();
                    }
                }
            }, cancellationToken).ConfigureAwait(false);

            return await WrapNativeTextureAsync(
                nativeResult.NativeTexture,
                nativeResult.Width,
                nativeResult.Height,
                linear,
                cancellationToken).ConfigureAwait(false);
        }

        public static async Task<NativeTextureVulkanTexture> LoadTextureFromFileAsync(
            string filePath,
            bool linear = false,
            CancellationToken cancellationToken = default)
        {
            return await LoadTextureFromFileAsync(filePath, linear, false, cancellationToken).ConfigureAwait(false);
        }

        public static Task<NativeTextureVulkanTexture> LoadTextureFromFileAsync(
            string filePath,
            bool linear,
            bool flipY)
        {
            return LoadTextureFromFileAsync(filePath, linear, flipY, CancellationToken.None);
        }

        public static async Task<NativeTextureVulkanTexture> LoadTextureFromFileAsync(
            string filePath,
            bool linear,
            bool flipY,
            CancellationToken cancellationToken)
        {
            TryCaptureMainThreadContext();
            if (string.IsNullOrWhiteSpace(filePath))
            {
                throw new ArgumentException("File path cannot be empty.", nameof(filePath));
            }

            await EnsurePluginLoadedAsync(cancellationToken).ConfigureAwait(false);
            bool useSrgb = await RunOnMainThreadAsync(() =>
            {
                EnsureVulkanBackendOnMainThread();
                return ShouldUseSrgbNativeFormat(linear);
            }, cancellationToken).ConfigureAwait(false);

            NativeCreateResult nativeResult = await Task.Run(() =>
            {
                cancellationToken.ThrowIfCancellationRequested();

                IntPtr texture = CreateFromFileWithOptions(filePath, out int decodedWidth, out int decodedHeight, useSrgb, flipY);
                if (texture == IntPtr.Zero)
                {
                    throw new InvalidOperationException("Native CreateFromFile returned a null Vulkan texture.");
                }

                if (cancellationToken.IsCancellationRequested)
                {
                    Release(texture);
                    texture = IntPtr.Zero;
                    cancellationToken.ThrowIfCancellationRequested();
                }

                return new NativeCreateResult(texture, decodedWidth, decodedHeight);
            }, cancellationToken).ConfigureAwait(false);

            return await WrapNativeTextureAsync(
                nativeResult.NativeTexture,
                nativeResult.Width,
                nativeResult.Height,
                linear,
                cancellationToken).ConfigureAwait(false);
        }

        public static async Task SaveToFileAsync(
            string filePath,
            NativeTextureDecodedImage decodedImage,
            CancellationToken cancellationToken = default)
        {
            if (decodedImage == null)
            {
                throw new ArgumentNullException(nameof(decodedImage));
            }

            await SaveToFileAsync(filePath, decodedImage.Rgba, decodedImage.Width, decodedImage.Height, cancellationToken)
                .ConfigureAwait(false);
        }

        public static async Task SaveToFileAsync(
            string filePath,
            byte[] rgba,
            int width,
            int height,
            CancellationToken cancellationToken = default)
        {
            TryCaptureMainThreadContext();
            await EnsurePluginLoadedAsync(cancellationToken).ConfigureAwait(false);

            if (string.IsNullOrWhiteSpace(filePath))
            {
                throw new ArgumentException("File path cannot be empty.", nameof(filePath));
            }

            ValidateRgbaBuffer(rgba, width, height, nameof(rgba));

            bool saved = await Task.Run(() =>
            {
                cancellationToken.ThrowIfCancellationRequested();

                GCHandle rgbaHandle = default;
                try
                {
                    rgbaHandle = GCHandle.Alloc(rgba, GCHandleType.Pinned);
                    return SaveToFile(filePath, rgbaHandle.AddrOfPinnedObject(), width, height);
                }
                finally
                {
                    if (rgbaHandle.IsAllocated)
                    {
                        rgbaHandle.Free();
                    }
                }
            }, cancellationToken).ConfigureAwait(false);

            if (!saved)
            {
                throw new IOException($"Native SaveToFile failed for '{filePath}'.");
            }

            cancellationToken.ThrowIfCancellationRequested();
        }

        internal static void ReleaseExternalTexture(Texture2D texture, IntPtr nativeTexture)
        {
            NativeTextureReleaseScheduler.ReleaseExternalTexture("Vulkan", texture, nativeTexture, Release);
        }

        private static async Task<NativeTextureVulkanTexture> WrapNativeTextureAsync(
            IntPtr nativeTexture,
            int width,
            int height,
            bool linear,
            CancellationToken cancellationToken)
        {
            try
            {
                Texture2D texture = await RunOnMainThreadAsync(() =>
                {
                    EnsureVulkanBackendOnMainThread();
                    return Texture2D.CreateExternalTexture(width, height, TextureFormat.RGBA32, false, linear, nativeTexture);
                }, cancellationToken).ConfigureAwait(false);

                if (cancellationToken.IsCancellationRequested)
                {
                    ReleaseExternalTexture(texture, nativeTexture);
                    nativeTexture = IntPtr.Zero;
                    cancellationToken.ThrowIfCancellationRequested();
                }

                return new NativeTextureVulkanTexture(texture, nativeTexture);
            }
            catch
            {
                if (nativeTexture != IntPtr.Zero)
                {
                    Release(nativeTexture);
                }

                throw;
            }
        }

        private static async Task EnsurePluginLoadedAsync(CancellationToken cancellationToken)
        {
            if (s_pluginLoaded)
            {
                return;
            }

            await RunOnMainThreadAsync(() =>
            {
                EnsurePluginLoadedOnMainThread();
            }, cancellationToken).ConfigureAwait(false);
        }

        private static void EnsurePluginLoadedOnMainThread()
        {
            TryCaptureMainThreadContext();
            if (!IsMainThread())
            {
                throw new InvalidOperationException("NativeTextureVulkan plugin warmup must run on the Unity main thread.");
            }

            if (s_pluginLoaded)
            {
                return;
            }

            Free(IntPtr.Zero);
            s_pluginLoaded = true;
        }

        private static void EnsureVulkanBackendOnMainThread()
        {
            TryCaptureMainThreadContext();
            if (!IsMainThread())
            {
                throw new InvalidOperationException("NativeTextureVulkan backend validation must run on the Unity main thread.");
            }

            RefreshGraphicsDeviceTypeOnMainThread();
            if (s_cachedGraphicsDeviceType != GraphicsDeviceType.Vulkan)
            {
                throw new NotSupportedException(
                    $"NativeTextureVulkan only supports Vulkan. Current backend is {s_cachedGraphicsDeviceType}.");
            }
        }

        private static void RefreshGraphicsDeviceTypeOnMainThread()
        {
            s_cachedGraphicsDeviceType = SystemInfo.graphicsDeviceType;
            s_hasCachedGraphicsDeviceType = true;
        }

        private static bool ShouldUseSrgbNativeFormat(bool linear)
        {
            return !linear && QualitySettings.activeColorSpace == ColorSpace.Linear;
        }

        private static void EnsureMainThreadContext()
        {
            if (s_mainThreadContext != null)
            {
                return;
            }

            if (SynchronizationContext.Current == null)
            {
                throw new InvalidOperationException(
                    "NativeTextureVulkan must be initialized from the Unity main thread before CreateAsync or disposal.");
            }

            s_mainThreadContext = SynchronizationContext.Current;
            s_mainThreadId = Thread.CurrentThread.ManagedThreadId;
            NativeTextureReleaseScheduler.RegisterMainThreadContext(s_mainThreadContext, s_mainThreadId);
        }

        private static void TryCaptureMainThreadContext()
        {
            if (s_mainThreadContext != null)
            {
                return;
            }

            if (SynchronizationContext.Current == null)
            {
                return;
            }

            s_mainThreadContext = SynchronizationContext.Current;
            s_mainThreadId = Thread.CurrentThread.ManagedThreadId;
            NativeTextureReleaseScheduler.RegisterMainThreadContext(s_mainThreadContext, s_mainThreadId);
        }

        private static bool IsMainThread()
        {
            return s_mainThreadId != 0 && Thread.CurrentThread.ManagedThreadId == s_mainThreadId;
        }

        private static async Task RunOnMainThreadAsync(Action action, CancellationToken cancellationToken)
        {
            await RunOnMainThreadAsync(() =>
            {
                action();
                return true;
            }, cancellationToken).ConfigureAwait(false);
        }

        private static Task<T> RunOnMainThreadAsync<T>(Func<T> action, CancellationToken cancellationToken)
        {
            if (action == null)
            {
                throw new ArgumentNullException(nameof(action));
            }

            EnsureMainThreadContext();
            cancellationToken.ThrowIfCancellationRequested();

            if (IsMainThread())
            {
                return Task.FromResult(action());
            }

            var taskSource = new TaskCompletionSource<T>(TaskCreationOptions.RunContinuationsAsynchronously);
            CancellationTokenRegistration cancellationRegistration = default;

            if (cancellationToken.CanBeCanceled)
            {
                cancellationRegistration = cancellationToken.Register(() =>
                {
                    taskSource.TrySetCanceled(cancellationToken);
                });
            }

            s_mainThreadContext.Post(_ =>
            {
                if (taskSource.Task.IsCompleted)
                {
                    cancellationRegistration.Dispose();
                    return;
                }

                try
                {
                    taskSource.TrySetResult(action());
                }
                catch (Exception ex)
                {
                    taskSource.TrySetException(ex);
                }
                finally
                {
                    cancellationRegistration.Dispose();
                }
            }, null);

            return taskSource.Task;
        }

        private static void ValidateEncodedData(byte[] encodedData, string paramName)
        {
            if (encodedData == null)
            {
                throw new ArgumentNullException(paramName);
            }

            if (encodedData.Length == 0)
            {
                throw new ArgumentException("Encoded image data cannot be empty.", paramName);
            }
        }

        private static void ValidateRgbaBuffer(byte[] rgba, int width, int height, string paramName)
        {
            if (rgba == null)
            {
                throw new ArgumentNullException(paramName);
            }

            if (width <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(width));
            }

            if (height <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(height));
            }

            int expectedLength = checked(width * height * BytesPerPixel);
            if (rgba.Length != expectedLength)
            {
                throw new ArgumentException(
                    $"RGBA buffer length must be exactly {expectedLength} bytes for {width}x{height}.",
                    paramName);
            }
        }

        private struct NativeCreateResult
        {
            internal NativeCreateResult(IntPtr nativeTexture, int width, int height)
            {
                NativeTexture = nativeTexture;
                Width = width;
                Height = height;
            }

            internal IntPtr NativeTexture { get; }

            internal int Width { get; }

            internal int Height { get; }
        }

        [DllImport(PluginName, EntryPoint = "Decode")]
        private static extern IntPtr Decode(IntPtr raw, int length, out int width, out int height);

        [DllImport(PluginName, EntryPoint = "DecodeWithOptions")]
        private static extern IntPtr DecodeWithOptions(
            IntPtr raw,
            int length,
            out int width,
            out int height,
            [MarshalAs(UnmanagedType.I1)] bool flipY);

        [DllImport(PluginName, EntryPoint = "Free")]
        private static extern void Free(IntPtr rgba);

        [DllImport(PluginName, EntryPoint = "Create")]
        private static extern IntPtr Create(IntPtr rgba, int width, int height);

        [DllImport(PluginName, EntryPoint = "CreateWithOptions")]
        private static extern IntPtr CreateWithOptions(
            IntPtr rgba,
            int width,
            int height,
            [MarshalAs(UnmanagedType.I1)] bool useSrgb);

        [DllImport(PluginName, EntryPoint = "CreateFromEncoded")]
        private static extern IntPtr CreateFromEncoded(
            IntPtr raw,
            int length,
            out int width,
            out int height,
            [MarshalAs(UnmanagedType.I1)] bool useSrgb);

        [DllImport(PluginName, EntryPoint = "CreateFromEncodedWithOptions")]
        private static extern IntPtr CreateFromEncodedWithOptions(
            IntPtr raw,
            int length,
            out int width,
            out int height,
            [MarshalAs(UnmanagedType.I1)] bool useSrgb,
            [MarshalAs(UnmanagedType.I1)] bool flipY);

        [DllImport(PluginName, EntryPoint = "CreateFromFile")]
        private static extern IntPtr CreateFromFile(
            string fileName,
            out int width,
            out int height,
            [MarshalAs(UnmanagedType.I1)] bool useSrgb);

        [DllImport(PluginName, EntryPoint = "CreateFromFileWithOptions")]
        private static extern IntPtr CreateFromFileWithOptions(
            string fileName,
            out int width,
            out int height,
            [MarshalAs(UnmanagedType.I1)] bool useSrgb,
            [MarshalAs(UnmanagedType.I1)] bool flipY);

        [DllImport(PluginName, EntryPoint = "Release")]
        private static extern void Release(IntPtr texPtr);

        [DllImport(PluginName, EntryPoint = "SaveToFile")]
        [return: MarshalAs(UnmanagedType.I1)]
        private static extern bool SaveToFile(string fileName, IntPtr rgba, int width, int height);
    }
}
