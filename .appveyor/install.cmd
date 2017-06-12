@echo on

echo Installing dirent
git clone -q --branch=master https://github.com/tronkko/dirent.git C:\projects\dirent

echo Installing gnuplot
choco install gnuplot

echo Installing OpenCV 2.4.13
choco install opencv -version 2.4.13

echo Installing OpenCV 2.4.13.2-vc14
appveyor DownloadFile ^
  https://github.com/opencv/opencv/releases/download/2.4.13.2/opencv-2.4.13.2-vc14.exe ^
  -FileName opencv-2.4.13.2-vc14.exe
opencv-2.4.13.2-vc14.exe -o"C:\tools_vc14" -y

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
                           curand_dev_8.0

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
