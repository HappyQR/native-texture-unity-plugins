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

// Pano GPU tile assembly: Begin -> N x UploadTileRegion -> Finish (all on worker
// threads), then wrap the returned handle with Texture2D.CreateExternalTexture and
// fire GL.IssuePluginEvent(GetRenderEventFunc(), 1) once. AbortImage tears down a
// pre-Finish assembly. See NativeTexture.cpp for the threading contract.
void* BeginPanoAssembly(int width, int height, bool useSrgb);
bool UploadTileRegion(void* assemblyImage, unsigned char* tileRgba, int tileStride, int validW, int validH, int dstX, int dstY);
void FinishPanoAssembly(void* assemblyImage);
void AbortImage(void* assemblyImage);

#ifdef __cplusplus
}
#endif
