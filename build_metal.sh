#!/bin/bash

set -euo pipefail

mkdir -p build/metal

clang++ \
  -std=c++17 \
  -fobjc-arc \
  -Wall \
  -Wextra \
  -dynamiclib \
  -o build/metal/libNativeTexture.dylib \
  metal/NativeTexture.mm \
  -I metal \
  -framework Metal \
  -framework Foundation \
  -I include
