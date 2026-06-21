@echo off
REM Build the standalone CUDA river-subgame engine.
REM Uses MSVC 14.38 (CUDA 11.8-compatible; 14.42 is too new) + Ninja + nvcc.
setlocal

set "VCVARS=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat"
set "CMAKE=C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
set "NINJA=C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe"

call "%VCVARS%" x64 -vcvars_ver=14.38 || exit /b 1

set "SRC=%~dp0"
set "BUILD=%SRC%build"

"%CMAKE%" -S "%SRC%." -B "%BUILD%" -G Ninja -DCMAKE_MAKE_PROGRAM="%NINJA%" -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl -DCMAKE_CUDA_HOST_COMPILER=cl || exit /b 1

"%CMAKE%" --build "%BUILD%" || exit /b 1

echo === BUILD OK ===
endlocal
