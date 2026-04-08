#pragma once

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Decodes image bytes into an RGBA8 buffer. The returned buffer must be freed with Free.
uint8_t* Decode(uint8_t* raw, int length, int* width, int* height);

// Decodes image bytes into an RGBA8 buffer with per-call decode options.
uint8_t* DecodeWithOptions(uint8_t* raw, int length, int* width, int* height, bool flipY);

// Frees an RGBA buffer returned by Decode.
void Free(uint8_t* rgba);

// Creates a Metal texture from an RGBA8 buffer. The returned pointer must be released with Release.
void* Create(unsigned char* rgba, int width, int height);

// Creates a Metal texture and optionally marks it as sRGB-backed for color textures in Linear projects.
void* CreateWithOptions(unsigned char* rgba, int width, int height, bool useSrgb);

// Decodes image bytes and creates a Metal texture in one pass. Outputs the decoded size on success.
void* CreateFromEncoded(uint8_t* raw, int length, int* width, int* height, bool useSrgb);

// Decodes image bytes and creates a Metal texture in one pass with per-call decode options.
void* CreateFromEncodedWithOptions(uint8_t* raw, int length, int* width, int* height, bool useSrgb, bool flipY);

// Loads an image file, decodes it, and creates a Metal texture in one pass. Outputs the decoded size on success.
void* CreateFromFile(const char* fileName, int* width, int* height, bool useSrgb);

// Loads an image file, decodes it, and creates a Metal texture in one pass with per-call decode options.
void* CreateFromFileWithOptions(const char* fileName, int* width, int* height, bool useSrgb, bool flipY);

// Releases a texture pointer returned by Create.
void Release(void* texPtr);

// Writes an RGBA8 buffer to disk. Supported extensions: png, jpg/jpeg, bmp, tga.
bool SaveToFile(const char* fileName, uint8_t* rgba, int width, int height);

#ifdef __cplusplus
}
#endif
