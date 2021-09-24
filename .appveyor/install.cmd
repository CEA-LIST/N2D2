@echo on

echo Installing dirent
git clone -q --branch=master https://github.com/tronkko/dirent.git C:\projects\dirent

echo Installing gnuplot
choco install gnuplot

echo Installing OpenCV 2.4.13
choco install opencv -version 2.4.13 --force

echo Installing OpenCV 2.4.13.2-vc14
appveyor DownloadFile ^
  https://github.com/opencv/opencv/releases/download/2.4.13.2/opencv-2.4.13.2-vc14.exe ^
  -FileName opencv-2.4.13.2-vc14.exe
opencv-2.4.13.2-vc14.exe -o"C:\tools_vc14" -y

echo Installing Protobuf
git clone -q --branch=master https://github.com/google/protobuf.git C:\projects\protobuf
cd C:\projects\protobuf
mkdir build_cmake
cd build_cmake
cmake ..\cmake -A x64 -Dprotobuf_BUILD_TESTS=OFF -Dprotobuf_MSVC_STATIC_RUNTIME=OFF -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDLL
cmake --build . --config Release
cmake --build . --config Release --target install
set CMAKE_INCLUDE_PATH=%CMAKE_INCLUDE_PATH%;C:/Program Files (x86)/protobuf/include
set CMAKE_LIBRARY_PATH=%CMAKE_LIBRARY_PATH%;C:/Program Files (x86)/protobuf/lib
set PATH=C:/Program Files (x86)/protobuf/bin;%PATH%
cd C:\projects\n2d2

echo Installing graphviz (optional)
choco install graphviz

if DEFINED USE_CUDA goto :use_cuda
goto :endif

:use_cuda
echo Installing CUDA toolkit 8.0
appveyor DownloadFile ^
  https://developer.nvidia.com/compute/cuda/8.0/prod/local_installers/cuda_8.0.44_windows-exe ^
  -FileName cuda_8.0.44_windows.exe
cuda_8.0.44_windows.exe -s compiler_8.0 ^
                           cublas_8.0 ^
                           cublas_dev_8.0 ^
                           cudart_8.0 ^
                           curand_8.0 ^
                           curand_dev_8.0 ^
                           nvml_dev_8.0

set PATH=%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v8.0\bin;%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v8.0\libnvvp;%PATH%

echo Installing cuDNN 8.0
appveyor DownloadFile ^
  http://developer.download.nvidia.com/compute/redist/cudnn/v5.1/cudnn-8.0-windows7-x64-v5.1.zip ^
  -FileName cudnn-8.0-windows7-x64-v5.1.zip
7z x cudnn-8.0-windows7-x64-v5.1.zip

copy cuda\include\*.* ^
  "%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v8.0\include\"
copy cuda\lib\x64\*.* ^
  "%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v8.0\lib\x64\"
copy cuda\bin\*.* ^
  "%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v8.0\bin\"

nvcc -V || exit /b

:endif
