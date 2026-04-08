cmake_minimum_required(VERSION 3.21)

project(NativeTexture LANGUAGES CXX)

set(NATIVE_TEXTURE_BACKEND "" CACHE STRING "Backend to build: metal, vulkan, or d3d12")
set_property(CACHE NATIVE_TEXTURE_BACKEND PROPERTY STRINGS metal vulkan d3d12)

if(NOT NATIVE_TEXTURE_BACKEND)
    if(APPLE)
        set(_native_texture_default_backend "metal")
    elseif(ANDROID)
        set(_native_texture_default_backend "vulkan")
    elseif(WIN32)
        set(_native_texture_default_backend "d3d12")
    else()
        set(_native_texture_default_backend "")
    endif()

    if(_native_texture_default_backend STREQUAL "")
        message(FATAL_ERROR
            "Unable to infer a NativeTexture backend for this platform. "
            "Set -DNATIVE_TEXTURE_BACKEND=metal|vulkan|d3d12 explicitly.")
    endif()

    set(NATIVE_TEXTURE_BACKEND "${_native_texture_default_backend}" CACHE STRING
        "Backend to build: metal, vulkan, or d3d12" FORCE)
endif()

string(TOLOWER "${NATIVE_TEXTURE_BACKEND}" NATIVE_TEXTURE_BACKEND)

function(native_texture_enable_warnings target_name)
    if(MSVC)
        target_compile_options("${target_name}" PRIVATE /W4)
    else()
        target_compile_options("${target_name}" PRIVATE -Wall -Wextra)
    endif()
endfunction()

function(native_texture_set_output_dir target_name output_dir)
    set_target_properties("${target_name}" PROPERTIES
        ARCHIVE_OUTPUT_DIRECTORY "${output_dir}"
        LIBRARY_OUTPUT_DIRECTORY "${output_dir}"
        RUNTIME_OUTPUT_DIRECTORY "${output_dir}")

    if(CMAKE_CONFIGURATION_TYPES)
        foreach(config_name IN LISTS CMAKE_CONFIGURATION_TYPES)
            string(TOUPPER "${config_name}" config_upper)
            set_target_properties("${target_name}" PROPERTIES
                "ARCHIVE_OUTPUT_DIRECTORY_${config_upper}" "${output_dir}"
                "LIBRARY_OUTPUT_DIRECTORY_${config_upper}" "${output_dir}"
                "RUNTIME_OUTPUT_DIRECTORY_${config_upper}" "${output_dir}")
        endforeach()
    endif()
endfunction()

function(native_texture_normalize_arch out_var raw_arch)
    string(TOLOWER "${raw_arch}" normalized_arch)

    if(normalized_arch STREQUAL "x64" OR
       normalized_arch STREQUAL "amd64" OR
       normalized_arch STREQUAL "x86_64")
        set(normalized_arch "x86_64")
    elseif(normalized_arch STREQUAL "win32" OR
           normalized_arch STREQUAL "x86" OR
           normalized_arch STREQUAL "i686")
        set(normalized_arch "x86")
    elseif(normalized_arch STREQUAL "arm64" OR
           normalized_arch STREQUAL "aarch64")
        set(normalized_arch "arm64")
    elseif(normalized_arch STREQUAL "")
        set(normalized_arch "unknown")
    endif()

    set("${out_var}" "${normalized_arch}" PARENT_SCOPE)
endfunction()

set(_native_texture_target "NativeTexture")
set(_native_texture_include_dir "${CMAKE_CURRENT_SOURCE_DIR}/include")

if(NATIVE_TEXTURE_BACKEND STREQUAL "metal")
    if(NOT APPLE)
        message(FATAL_ERROR "The Metal backend can only be built on Apple platforms.")
    endif()

    enable_language(OBJCXX)

    add_library("${_native_texture_target}" SHARED
        metal/NativeTexture.mm)

    set_target_properties("${_native_texture_target}" PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
        OBJCXX_STANDARD 17
        OBJCXX_STANDARD_REQUIRED YES
        OUTPUT_NAME "NativeTexture")

    target_include_directories("${_native_texture_target}" PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/metal"
        "${_native_texture_include_dir}")

    target_compile_options("${_native_texture_target}" PRIVATE
        "$<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>")
    native_texture_enable_warnings("${_native_texture_target}")

    find_library(NATIVE_TEXTURE_METAL_FRAMEWORK Metal REQUIRED)
    find_library(NATIVE_TEXTURE_FOUNDATION_FRAMEWORK Foundation REQUIRED)
    target_link_libraries("${_native_texture_target}" PRIVATE
        "${NATIVE_TEXTURE_METAL_FRAMEWORK}"
        "${NATIVE_TEXTURE_FOUNDATION_FRAMEWORK}")

    native_texture_set_output_dir("${_native_texture_target}"
        "${CMAKE_CURRENT_SOURCE_DIR}/build/metal")

elseif(NATIVE_TEXTURE_BACKEND STREQUAL "vulkan")
    if(NOT ANDROID)
        message(FATAL_ERROR
            "The Vulkan backend is configured for Android builds only. "
            "Pass -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/android-unity.cmake "
            "or the Android NDK toolchain file when configuring CMake.")
    endif()

    add_library("${_native_texture_target}" SHARED
        vulkan/NativeTexture.cpp)

    set_target_properties("${_native_texture_target}" PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
        OUTPUT_NAME "NativeTexture")

    target_include_directories("${_native_texture_target}" PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/vulkan"
        "${_native_texture_include_dir}")

    native_texture_enable_warnings("${_native_texture_target}")

    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        target_link_options("${_native_texture_target}" PRIVATE -static-libstdc++)
    endif()

    find_library(NATIVE_TEXTURE_ATOMIC_LIBRARY atomic)
    if(NATIVE_TEXTURE_ATOMIC_LIBRARY)
        target_link_libraries("${_native_texture_target}" PRIVATE "${NATIVE_TEXTURE_ATOMIC_LIBRARY}")
    endif()

    set(_native_texture_android_abi "${ANDROID_ABI}")
    if(NOT _native_texture_android_abi AND DEFINED CMAKE_ANDROID_ARCH_ABI)
        set(_native_texture_android_abi "${CMAKE_ANDROID_ARCH_ABI}")
    endif()
    if(NOT _native_texture_android_abi)
        set(_native_texture_android_abi "unknown-abi")
    endif()

    native_texture_set_output_dir("${_native_texture_target}"
        "${CMAKE_CURRENT_SOURCE_DIR}/build/vulkan/android/${_native_texture_android_abi}")

elseif(NATIVE_TEXTURE_BACKEND STREQUAL "d3d12")
    if(NOT WIN32)
        message(FATAL_ERROR "The D3D12 backend can only be built on Windows.")
    endif()

    add_library("${_native_texture_target}" SHARED
        direct3d/NativeTexture.cpp)

    set_target_properties("${_native_texture_target}" PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
        OUTPUT_NAME "NativeTexture")

    target_include_directories("${_native_texture_target}" PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/direct3d"
        "${_native_texture_include_dir}")

    target_compile_definitions("${_native_texture_target}" PRIVATE
        WIN32
        _WINDOWS
        NOMINMAX
        _CRT_SECURE_NO_WARNINGS)

    if(MSVC)
        target_compile_options("${_native_texture_target}" PRIVATE /EHsc /utf-8)
    endif()
    native_texture_enable_warnings("${_native_texture_target}")

    target_link_libraries("${_native_texture_target}" PRIVATE
        d3d12
        dxgi
        dxguid
        user32
        ole32)

    set(_native_texture_windows_arch "${CMAKE_VS_PLATFORM_NAME}")
    if(NOT _native_texture_windows_arch)
        set(_native_texture_windows_arch "${CMAKE_SYSTEM_PROCESSOR}")
    endif()
    native_texture_normalize_arch(_native_texture_windows_arch "${_native_texture_windows_arch}")

    native_texture_set_output_dir("${_native_texture_target}"
        "${CMAKE_CURRENT_SOURCE_DIR}/build/direct3d/${_native_texture_windows_arch}")

else()
    message(FATAL_ERROR
        "Unsupported NATIVE_TEXTURE_BACKEND='${NATIVE_TEXTURE_BACKEND}'. "
        "Expected metal, vulkan, or d3d12.")
endif()
