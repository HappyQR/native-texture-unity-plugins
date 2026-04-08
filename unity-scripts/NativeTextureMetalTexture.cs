using System;
using UnityEngine;

namespace NativeTexture
{
    public sealed class NativeTextureMetalTexture : INativeTextureHandle
    {
        internal NativeTextureMetalTexture(Texture2D texture, IntPtr nativeTexture)
        {
            Texture = texture ?? throw new ArgumentNullException(nameof(texture));
            NativeTexture = nativeTexture;
        }

        public Texture2D Texture { get; private set; }

        public IntPtr NativeTexture { get; private set; }

        public int Width => Texture != null ? Texture.width : 0;

        public int Height => Texture != null ? Texture.height : 0;

        public bool IsDisposed => Texture == null && NativeTexture == IntPtr.Zero;

        public NativeTextureBackend Backend => NativeTextureBackend.Metal;

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

            NativeTextureMetal.ReleaseExternalTexture(texture, nativeTexture);
            GC.SuppressFinalize(this);
        }
    }
}
