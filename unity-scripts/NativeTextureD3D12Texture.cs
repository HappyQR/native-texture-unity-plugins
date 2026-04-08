using System;
using UnityEngine;

namespace NativeTexture
{
    public sealed class NativeTextureD3D12Texture : INativeTextureHandle
    {
        internal NativeTextureD3D12Texture(Texture2D texture, IntPtr nativeTexture)
        {
            Texture = texture ?? throw new ArgumentNullException(nameof(texture));
            NativeTexture = nativeTexture;
        }

        public Texture2D Texture { get; private set; }

        public IntPtr NativeTexture { get; private set; }

        public int Width => Texture != null ? Texture.width : 0;

        public int Height => Texture != null ? Texture.height : 0;

        public bool IsDisposed => Texture == null && NativeTexture == IntPtr.Zero;

        public NativeTextureBackend Backend => NativeTextureBackend.D3D12;

        public void Dispose()
        {
            if (IsDisposed)
            {
                return;
            }

            Texture2D texture = Texture;
            IntPtr nativeTexture = NativeTexture;

            Texture = null;
            NativeTexture = IntPtr.Zero;

            NativeTextureD3D12.ReleaseExternalTexture(texture, nativeTexture);
            GC.SuppressFinalize(this);
        }
    }
}
