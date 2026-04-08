@echo off
setlocal EnableExtensions

set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%" >nul
if errorlevel 1 (
    echo Failed to enter script directory: %SCRIPT_DIR%
    exit /b 1
)
set "ROOT_DIR=%CD%"

set "COMPILER="
if defined CXX (
    set "COMPILER=%CXX%"
)

if not defined COMPILER (
    where /Q clang-cl.exe
    if not errorlevel 1 (
        set "COMPILER=clang-cl.exe"
    )
)

if not defined COMPILER (
    where /Q cl.exe
    if not errorlevel 1 (
        set "COMPILER=cl.exe"
    )
)

if not defined COMPILER (
    echo Unable to find a Windows C++ compiler.
    echo Run this script from a Visual Studio Developer Command Prompt, or set CXX=clang-cl.exe / cl.exe.
    popd >nul
    exit /b 1
)

set "OUTPUT_DIR=%ROOT_DIR%\build\direct3d\x86_64"
if not exist "%OUTPUT_DIR%" (
    mkdir "%OUTPUT_DIR%"
    if errorlevel 1 (
        echo Failed to create output directory: %OUTPUT_DIR%
        popd >nul
        exit /b 1
    )
)

set "SRC_FILE=%ROOT_DIR%\direct3d\NativeTexture.cpp"
set "OUTPUT_FILE=%OUTPUT_DIR%\NativeTexture.dll"
set "IMPLIB_FILE=%OUTPUT_DIR%\NativeTexture.lib"
set "PDB_FILE=%OUTPUT_DIR%\NativeTexture.pdb"
set "OBJECT_FILE=%OUTPUT_DIR%\NativeTexture.obj"
set "INCLUDE_DIRECT3D=%ROOT_DIR%\direct3d"
set "INCLUDE_COMMON=%ROOT_DIR%\include"

echo Using compiler: %COMPILER%

pushd "%OUTPUT_DIR%" >nul
if errorlevel 1 (
    echo Failed to enter output directory: %OUTPUT_DIR%
    popd >nul
    exit /b 1
)

call "%COMPILER%" ^
  /nologo ^
  /std:c++17 ^
  /EHsc ^
  /MD ^
  /W4 ^
  /utf-8 ^
  /LD ^
  /DWIN32 ^
  /D_WINDOWS ^
  /DNOMINMAX ^
  /D_CRT_SECURE_NO_WARNINGS ^
  /Fo"%OBJECT_FILE%" ^
  /I"%INCLUDE_DIRECT3D%" ^
  /I"%INCLUDE_COMMON%" ^
  "%SRC_FILE%" ^
  /link ^
  /OUT:"%OUTPUT_FILE%" ^
  /IMPLIB:"%IMPLIB_FILE%" ^
  /PDB:"%PDB_FILE%" ^
  d3d12.lib ^
  dxgi.lib ^
  dxguid.lib ^
  user32.lib ^
  ole32.lib

if errorlevel 1 (
    echo Build failed.
    popd >nul
    popd >nul
    exit /b 1
)

popd >nul
echo Built %OUTPUT_FILE%
popd >nul
exit /b 0
