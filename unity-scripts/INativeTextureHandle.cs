using System;
using UnityEngine;

namespace NativeTexture
{
    public interface INativeTextureHandle : IDisposable
    {
        Texture2D Texture { get; }

        IntPtr NativeTexture { get; }

        int Width { get; }

        int Height { get; }

        bool IsDisposed { get; }

        NativeTextureBackend Backend { get; }
    }
}
