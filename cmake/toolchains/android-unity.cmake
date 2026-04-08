if(DEFINED NATIVE_TEXTURE_ANDROID_UNITY_TOOLCHAIN_INCLUDED)
    return()
endif()
set(NATIVE_TEXTURE_ANDROID_UNITY_TOOLCHAIN_INCLUDED TRUE)

function(_native_texture_try_set_ndk candidate_path)
    if(candidate_path AND IS_DIRECTORY "${candidate_path}")
        set(_native_texture_ndk_root "${candidate_path}" PARENT_SCOPE)
    endif()
endfunction()

set(_native_texture_ndk_root "")

if(DEFINED ENV{ANDROID_NDK_ROOT})
    _native_texture_try_set_ndk("$ENV{ANDROID_NDK_ROOT}")
endif()

if(NOT _native_texture_ndk_root AND DEFINED ENV{ANDROID_NDK_HOME})
    _native_texture_try_set_ndk("$ENV{ANDROID_NDK_HOME}")
endif()

if(NOT _native_texture_ndk_root AND DEFINED ENV{UNITY_ANDROID_NDK})
    _native_texture_try_set_ndk("$ENV{UNITY_ANDROID_NDK}")
endif()

if(NOT _native_texture_ndk_root)
    file(GLOB _native_texture_unity_ndks_mac
        LIST_DIRECTORIES TRUE
        "/Applications/Unity/Hub/Editor/*/PlaybackEngines/AndroidPlayer/NDK")
    list(SORT _native_texture_unity_ndks_mac)
    list(REVERSE _native_texture_unity_ndks_mac)
    foreach(candidate_path IN LISTS _native_texture_unity_ndks_mac)
        if(IS_DIRECTORY "${candidate_path}")
            set(_native_texture_ndk_root "${candidate_path}")
            break()
        endif()
    endforeach()
endif()

if(NOT _native_texture_ndk_root)
    file(GLOB _native_texture_unity_ndks_win
        LIST_DIRECTORIES TRUE
        "C:/Program Files/Unity/Hub/Editor/*/Editor/Data/PlaybackEngines/AndroidPlayer/NDK")
    list(SORT _native_texture_unity_ndks_win)
    list(REVERSE _native_texture_unity_ndks_win)
    foreach(candidate_path IN LISTS _native_texture_unity_ndks_win)
        if(IS_DIRECTORY "${candidate_path}")
            set(_native_texture_ndk_root "${candidate_path}")
            break()
        endif()
    endforeach()
endif()

if(NOT _native_texture_ndk_root)
    message(FATAL_ERROR
        "Unable to locate the Android NDK. Set ANDROID_NDK_ROOT, ANDROID_NDK_HOME, "
        "or UNITY_ANDROID_NDK before configuring CMake.")
endif()

if(NOT DEFINED ANDROID_ABI)
    set(ANDROID_ABI "arm64-v8a" CACHE STRING "Android ABI" FORCE)
endif()

if(NOT DEFINED ANDROID_PLATFORM)
    set(ANDROID_PLATFORM "24" CACHE STRING "Android API level" FORCE)
endif()

set(CMAKE_SYSTEM_NAME Android)
set(CMAKE_ANDROID_NDK "${_native_texture_ndk_root}" CACHE PATH "Android NDK root" FORCE)
set(ANDROID_NDK "${_native_texture_ndk_root}" CACHE PATH "Android NDK root" FORCE)

include("${_native_texture_ndk_root}/build/cmake/android.toolchain.cmake")
