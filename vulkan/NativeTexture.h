#pragma once

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

uint8_t* Decode(uint8_t* raw, int length, int* width, int* height);
uint8_t* DecodeWithOptions(uint8_t* raw, int length, int* width, int* height, bool flipY);
void Free(uint8_t* rgba);
void* Create(unsigned char* rgba, int width, int height);
void* CreateWithOptions(unsigned char* rgba, int width, int height, bool useSrgb);
void* CreateFromEncoded(uint8_t* raw, int length, int* width, int* height, bool useSrgb);
void* CreateFromEncodedWithOptions(uint8_t* raw, int length, int* width, int* height, bool useSrgb, bool flipY);
void* CreateFromFile(const char* fileName, int* width, int* height, bool useSrgb);
void* CreateFromFileWithOptions(const char* fileName, int* width, int* height, bool useSrgb, bool flipY);
void Release(void* texPtr);
bool SaveToFile(const char* fileName, uint8_t* rgba, int width, int height);

#ifdef __cplusplus
}
#endif
