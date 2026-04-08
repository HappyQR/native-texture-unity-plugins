#!/bin/bash

set -euo pipefail

OUTPUT_ROOT="build/vulkan/android"
ANDROID_API_LEVEL="${ANDROID_API_LEVEL:-24}"
ANDROID_ABIS="${ANDROID_ABIS:-arm64-v8a}"

resolve_ndk_root() {
  if [[ -n "${ANDROID_NDK_ROOT:-}" && -d "${ANDROID_NDK_ROOT}" ]]; then
    echo "${ANDROID_NDK_ROOT}"
    return 0
  fi

  if [[ -n "${ANDROID_NDK_HOME:-}" && -d "${ANDROID_NDK_HOME}" ]]; then
    echo "${ANDROID_NDK_HOME}"
    return 0
  fi

  if [[ -n "${UNITY_ANDROID_NDK:-}" && -d "${UNITY_ANDROID_NDK}" ]]; then
    echo "${UNITY_ANDROID_NDK}"
    return 0
  fi

  local candidate
  candidate="$(find /Applications/Unity/Hub/Editor -path '*AndroidPlayer/NDK' 2>/dev/null | sort | tail -n 1)"
  if [[ -n "${candidate}" && -d "${candidate}" ]]; then
    echo "${candidate}"
    return 0
  fi

  return 1
}

compiler_for_abi() {
  case "$1" in
    arm64-v8a)
      echo "aarch64-linux-android${ANDROID_API_LEVEL}-clang++"
      ;;
    armeabi-v7a)
      echo "armv7a-linux-androideabi${ANDROID_API_LEVEL}-clang++"
      ;;
    x86_64)
      echo "x86_64-linux-android${ANDROID_API_LEVEL}-clang++"
      ;;
    *)
      echo "Unsupported ABI: $1" >&2
      return 1
      ;;
  esac
}

NDK_ROOT="$(resolve_ndk_root || true)"
if [[ -z "${NDK_ROOT}" ]]; then
  echo "Unable to find Android NDK. Set ANDROID_NDK_ROOT, ANDROID_NDK_HOME or UNITY_ANDROID_NDK." >&2
  exit 1
fi

TOOLCHAIN_BIN="${NDK_ROOT}/toolchains/llvm/prebuilt/darwin-x86_64/bin"
if [[ ! -d "${TOOLCHAIN_BIN}" ]]; then
  echo "Android NDK clang toolchain was not found under ${TOOLCHAIN_BIN}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}"

for abi in ${ANDROID_ABIS}; do
  compiler_name="$(compiler_for_abi "${abi}")"
  compiler_path="${TOOLCHAIN_BIN}/${compiler_name}"
  if [[ ! -x "${compiler_path}" ]]; then
    echo "Missing Android compiler for ${abi}: ${compiler_path}" >&2
    exit 1
  fi

  abi_output_dir="${OUTPUT_ROOT}/${abi}"
  mkdir -p "${abi_output_dir}"

  "${compiler_path}" \
    -std=c++17 \
    -fPIC \
    -shared \
    -Wall \
    -Wextra \
    -pthread \
    -static-libstdc++ \
    -o "${abi_output_dir}/libNativeTexture.so" \
    vulkan/NativeTexture.cpp \
    -I vulkan \
    -I include \
    -latomic

  echo "Built ${abi_output_dir}/libNativeTexture.so"
done
