#!/bin/bash

set -euo pipefail

case "$(uname -s 2>/dev/null || echo unknown)" in
  CYGWIN*|MINGW*|MSYS*)
    ;;
  *)
    if [[ "${OS:-}" != "Windows_NT" ]]; then
      echo "build_direct3d.sh must be run on Windows with MSVC/clang-cl and the Windows SDK available." >&2
      exit 1
    fi
    ;;
esac

to_native_path() {
  if command -v cygpath >/dev/null 2>&1; then
    cygpath -aw "$1"
  else
    printf '%s' "$1"
  fi
}

resolve_compiler() {
  if [[ -n "${CXX:-}" ]]; then
    echo "${CXX}"
    return 0
  fi

  if command -v clang-cl >/dev/null 2>&1; then
    echo "clang-cl"
    return 0
  fi

  if command -v cl >/dev/null 2>&1; then
    echo "cl"
    return 0
  fi

  if command -v cl.exe >/dev/null 2>&1; then
    echo "cl.exe"
    return 0
  fi

  return 1
}

COMPILER="$(resolve_compiler || true)"
if [[ -z "${COMPILER}" ]]; then
  echo "Unable to find a Windows C++ compiler. Run this script from a Visual Studio developer shell or set CXX=clang-cl/cl." >&2
  exit 1
fi

OUTPUT_DIR="build/direct3d/x86_64"
mkdir -p "${OUTPUT_DIR}"

SRC_FILE="$(to_native_path "direct3d/NativeTexture.cpp")"
OUTPUT_FILE="$(to_native_path "${OUTPUT_DIR}/NativeTexture.dll")"
INCLUDE_DIRECT3D="$(to_native_path "direct3d")"
INCLUDE_COMMON="$(to_native_path "include")"

"${COMPILER}" \
  /nologo \
  /std:c++17 \
  /EHsc \
  /MD \
  /W4 \
  /utf-8 \
  /LD \
  /DWIN32 \
  /D_WINDOWS \
  /DNOMINMAX \
  /D_CRT_SECURE_NO_WARNINGS \
  /I"${INCLUDE_DIRECT3D}" \
  /I"${INCLUDE_COMMON}" \
  "${SRC_FILE}" \
  /link \
  /OUT:"${OUTPUT_FILE}" \
  d3d12.lib \
  dxgi.lib \
  dxguid.lib \
  user32.lib \
  ole32.lib

echo "Built ${OUTPUT_DIR}/NativeTexture.dll"
