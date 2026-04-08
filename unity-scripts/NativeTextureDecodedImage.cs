using System;

namespace NativeTexture
{
    public sealed class NativeTextureDecodedImage
    {
        private const int BytesPerPixel = 4;

        public NativeTextureDecodedImage(byte[] rgba, int width, int height)
        {
            if (rgba == null)
            {
                throw new ArgumentNullException(nameof(rgba));
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
                    nameof(rgba));
            }

            Rgba = rgba;
            Width = width;
            Height = height;
        }

        public byte[] Rgba { get; }

        public int Width { get; }

        public int Height { get; }

        public int ByteCount => Rgba.Length;
    }
}
