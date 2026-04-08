#pragma once

#include <stdbool.h>
#include <stdint.h>
#include "unity/IUnityInterface.h"

#ifdef __cplusplus
#define NATIVE_TEXTURE_API extern "C" UNITY_INTERFACE_EXPORT
#else
#define NATIVE_TEXTURE_API UNITY_INTERFACE_EXPORT
#endif

NATIVE_TEXTURE_API uint8_t* UNITY_INTERFACE_API Decode(uint8_t* raw, int length, int* width, int* height);
NATIVE_TEXTURE_API uint8_t* UNITY_INTERFACE_API DecodeWithOptions(uint8_t* raw, int length, int* width, int* height, bool flipY);
NATIVE_TEXTURE_API void UNITY_INTERFACE_API Free(uint8_t* rgba);
NATIVE_TEXTURE_API void* UNITY_INTERFACE_API Create(unsigned char* rgba, int width, int height);
NATIVE_TEXTURE_API void* UNITY_INTERFACE_API CreateWithOptions(unsigned char* rgba, int width, int height, bool useSrgb);
NATIVE_TEXTURE_API void* UNITY_INTERFACE_API CreateFromEncoded(uint8_t* raw, int length, int* width, int* height, bool useSrgb);
NATIVE_TEXTURE_API void* UNITY_INTERFACE_API CreateFromEncodedWithOptions(uint8_t* raw, int length, int* width, int* height, bool useSrgb, bool flipY);
NATIVE_TEXTURE_API void* UNITY_INTERFACE_API CreateFromFile(const char* fileName, int* width, int* height, bool useSrgb);
NATIVE_TEXTURE_API void* UNITY_INTERFACE_API CreateFromFileWithOptions(const char* fileName, int* width, int* height, bool useSrgb, bool flipY);
NATIVE_TEXTURE_API void UNITY_INTERFACE_API Release(void* texPtr);
NATIVE_TEXTURE_API bool UNITY_INTERFACE_API SaveToFile(const char* fileName, uint8_t* rgba, int width, int height);
