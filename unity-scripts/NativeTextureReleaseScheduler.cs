using System;
using System.Collections.Generic;
using System.Runtime.ExceptionServices;
using System.Threading;
using UnityEngine;

namespace NativeTexture
{
    internal static class NativeTextureReleaseScheduler
    {
        private const int DeferredReleaseFrameDelay = 1;

        // After a native release is scheduled, keep pumping the backend's render-thread
        // queues for this many frames so a fence-gated destroy (which needs one event to arm
        // and another to reap) actually completes even when no new texture is being created.
        // Without it, the last released device image lingers until the next Create/Finish —
        // the "memory not released after the final pano load" symptom. 6 frames comfortably
        // covers the 1-frame release defer + arm + reap; idle frames don't pump.
        private const int RenderPumpFrameWindow = 6;

        private static readonly object s_sync = new object();
        private static readonly List<ScheduledRelease> s_pendingReleases = new List<ScheduledRelease>();

        private static SynchronizationContext s_mainThreadContext;
        private static int s_mainThreadId;
        private static ReleasePump s_releasePump;
        private static bool s_applicationQuitting;
        // Backend-specific render-thread pump (Vulkan only; Metal/D3D12 destroy synchronously
        // and never register one). Touched only on the Unity main thread.
        private static Action s_renderPump;
        private static int s_renderPumpFramesRemaining;

        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.BeforeSceneLoad)]
        private static void CaptureRuntimeState()
        {
            lock (s_sync)
            {
                s_applicationQuitting = false;
                if (s_releasePump == null)
                {
                    s_pendingReleases.Clear();
                }
            }

            TryCaptureMainThreadContext();
        }

        internal static void RegisterMainThreadContext(SynchronizationContext context, int mainThreadId)
        {
            if (context == null)
            {
                throw new ArgumentNullException(nameof(context));
            }

            if (mainThreadId == 0)
            {
                throw new ArgumentOutOfRangeException(nameof(mainThreadId));
            }

            lock (s_sync)
            {
                if (s_mainThreadContext == null)
                {
                    s_mainThreadContext = context;
                    s_mainThreadId = mainThreadId;
                }
            }

            if (Thread.CurrentThread.ManagedThreadId == mainThreadId)
            {
                EnsureReleasePumpOnMainThread();
            }
        }

        // Registered by the Vulkan backend (main thread) so the ReleasePump can advance the
        // native render-thread destroy / in-flight reap queues after a release. No-op for
        // backends that destroy synchronously.
        internal static void RegisterRenderPump(Action pump)
        {
            if (pump == null)
            {
                throw new ArgumentNullException(nameof(pump));
            }

            lock (s_sync)
            {
                if (s_renderPump == null)
                {
                    s_renderPump = pump;
                }
            }
        }

        internal static void ReleaseExternalTexture(
            string backendName,
            Texture2D texture,
            IntPtr nativeTexture,
            Action<IntPtr> nativeRelease)
        {
            if (texture == null && nativeTexture == IntPtr.Zero)
            {
                return;
            }

            if (string.IsNullOrWhiteSpace(backendName))
            {
                throw new ArgumentException("Backend name cannot be empty.", nameof(backendName));
            }

            if (nativeRelease == null)
            {
                throw new ArgumentNullException(nameof(nativeRelease));
            }

            ExecuteOnMainThread(() =>
            {
                EnsureReleasePumpOnMainThread();

                if (texture != null)
                {
                    UnityEngine.Object.Destroy(texture);
                }

                if (nativeTexture == IntPtr.Zero)
                {
                    return;
                }

                if (s_applicationQuitting)
                {
                    SafeRelease(backendName, nativeTexture, nativeRelease);
                    return;
                }

                lock (s_sync)
                {
                    s_pendingReleases.Add(new ScheduledRelease(
                        backendName,
                        nativeTexture,
                        Time.frameCount + DeferredReleaseFrameDelay,
                        nativeRelease));
                }

                // Keep the render-thread pump running for a few frames so the fence-gated
                // native destroy this release enqueues actually drains (main-thread only).
                s_renderPumpFramesRemaining = RenderPumpFrameWindow;
            });
        }

        private static void TryCaptureMainThreadContext()
        {
            SynchronizationContext currentContext = SynchronizationContext.Current;
            if (currentContext == null)
            {
                return;
            }

            int currentThreadId = Thread.CurrentThread.ManagedThreadId;
            if (currentThreadId == 0)
            {
                return;
            }

            lock (s_sync)
            {
                if (s_mainThreadContext == null)
                {
                    s_mainThreadContext = currentContext;
                    s_mainThreadId = currentThreadId;
                }
            }
        }

        private static void EnsureReleasePumpOnMainThread()
        {
            if (!IsMainThread())
            {
                throw new InvalidOperationException("NativeTextureReleaseScheduler must be initialized on the Unity main thread.");
            }

            if (s_releasePump != null)
            {
                return;
            }

            var host = new GameObject("NativeTexture.ReleasePump");
            host.hideFlags = HideFlags.HideAndDontSave;
            UnityEngine.Object.DontDestroyOnLoad(host);
            s_releasePump = host.AddComponent<ReleasePump>();
        }

        private static void ExecuteOnMainThread(Action action)
        {
            if (action == null)
            {
                throw new ArgumentNullException(nameof(action));
            }

            TryCaptureMainThreadContext();

            if (IsMainThread())
            {
                action();
                return;
            }

            SynchronizationContext mainThreadContext;
            lock (s_sync)
            {
                mainThreadContext = s_mainThreadContext;
            }

            if (mainThreadContext == null)
            {
                throw new InvalidOperationException(
                    "NativeTexture release requires a captured Unity main-thread SynchronizationContext. Call NativeTexture.Initialize() from the main thread first.");
            }

            Exception dispatchException = null;
            using (var waitHandle = new ManualResetEventSlim(false))
            {
                mainThreadContext.Post(_ =>
                {
                    try
                    {
                        action();
                    }
                    catch (Exception ex)
                    {
                        dispatchException = ex;
                    }
                    finally
                    {
                        waitHandle.Set();
                    }
                }, null);

                waitHandle.Wait();
            }

            if (dispatchException != null)
            {
                ExceptionDispatchInfo.Capture(dispatchException).Throw();
            }
        }

        private static bool IsMainThread()
        {
            return s_mainThreadId != 0 && Thread.CurrentThread.ManagedThreadId == s_mainThreadId;
        }

        private static void FlushReadyReleases(bool flushAll)
        {
            List<ScheduledRelease> readyReleases = null;
            int currentFrame = Time.frameCount;

            lock (s_sync)
            {
                if (s_pendingReleases.Count == 0)
                {
                    return;
                }

                for (int index = s_pendingReleases.Count - 1; index >= 0; --index)
                {
                    ScheduledRelease pending = s_pendingReleases[index];
                    if (!flushAll && pending.ReleaseFrame > currentFrame)
                    {
                        continue;
                    }

                    if (readyReleases == null)
                    {
                        readyReleases = new List<ScheduledRelease>();
                    }

                    readyReleases.Add(pending);
                    s_pendingReleases.RemoveAt(index);
                }
            }

            if (readyReleases == null)
            {
                return;
            }

            foreach (ScheduledRelease pending in readyReleases)
            {
                SafeRelease(pending.BackendName, pending.NativeTexture, pending.Release);
            }

            // The native Release just enqueued fence-gated destroy(s); refresh the pump window
            // anchored to this actual enqueue so the destroy drains within the next few frames.
            s_renderPumpFramesRemaining = RenderPumpFrameWindow;
        }

        private static void SafeRelease(string backendName, IntPtr nativeTexture, Action<IntPtr> nativeRelease)
        {
            try
            {
                nativeRelease(nativeTexture);
            }
            catch (Exception ex)
            {
                Debug.LogException(new InvalidOperationException(
                    $"NativeTexture{backendName} failed to release native texture {nativeTexture}.", ex));
            }
        }

        private struct ScheduledRelease
        {
            internal ScheduledRelease(
                string backendName,
                IntPtr nativeTexture,
                int releaseFrame,
                Action<IntPtr> release)
            {
                BackendName = backendName;
                NativeTexture = nativeTexture;
                ReleaseFrame = releaseFrame;
                Release = release;
            }

            internal string BackendName { get; }

            internal IntPtr NativeTexture { get; }

            internal int ReleaseFrame { get; }

            internal Action<IntPtr> Release { get; }
        }

        private sealed class ReleasePump : MonoBehaviour
        {
            private void LateUpdate()
            {
                FlushReadyReleases(flushAll: false);

                // Advance the backend render-thread destroy / reap queues for a bounded window
                // after each release, so a freed device image is actually reaped instead of
                // lingering until the next texture create/finish event.
                if (s_renderPumpFramesRemaining > 0)
                {
                    Action pump = s_renderPump;
                    if (pump != null)
                    {
                        pump();
                    }
                    s_renderPumpFramesRemaining--;
                }
            }

            private void OnApplicationQuit()
            {
                s_applicationQuitting = true;
                FlushReadyReleases(flushAll: true);
            }

            private void OnDestroy()
            {
                s_applicationQuitting = true;
                FlushReadyReleases(flushAll: true);
                s_releasePump = null;
            }
        }
    }
}
