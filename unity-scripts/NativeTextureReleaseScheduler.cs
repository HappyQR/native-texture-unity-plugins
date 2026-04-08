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

        private static readonly object s_sync = new object();
        private static readonly List<ScheduledRelease> s_pendingReleases = new List<ScheduledRelease>();

        private static SynchronizationContext s_mainThreadContext;
        private static int s_mainThreadId;
        private static ReleasePump s_releasePump;
        private static bool s_applicationQuitting;

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
