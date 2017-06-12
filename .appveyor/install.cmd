@echo off
echo Installing dirent
git clone -q --branch=master https://github.com/tronkko/dirent.git C:\projects\dirent

echo Installing gnuplot
choco install gnuplot

echo Installing OpenCV 2.4.13
choco install opencv -version 2.4.13

echo Installing OpenCV 2.4.13.2-vc14
appveyor DownloadFile https://github.com/opencv/opencv/releases/download/2.4.13.2/opencv-2.4.13.2-vc14.exe -FileName opencv-2.4.13.2-vc14.exe
opencv-2.4.13.2-vc14.exe -o"C:\tools_vc14" -y

if DEFINED USE_CUDA (
    echo Downloading CUDA toolkit 8.0
    appveyor DownloadFile https://developer.nvidia.com/compute/cuda/8.0/prod/local_installers/cuda_8.0.44_windows-exe -FileName cuda_8.0.44_windows.exe

    echo Installing CUDA toolkit 8.0
    cuda_8.0.44_windows.exe -s compiler_8.0 ^
                               cublas_8.0 ^
                               cublas_dev_8.0 ^
                               cudart_8.0 ^
                               curand_8.0 ^
                               curand_dev_8.0

    echo Downloading cuDNN 8.0
    appveyor DownloadFile https://developer.nvidia.com/compute/machine-learning/cudnn/secure/v5.1/prod/8.0/cudnn-8.0-windows7-x64-v5.1-zip -FileName cudnn-8.0-windows7-x64-v5.1.zip
    7z x cudnn-8.0-windows7-x64-v5.1.zip -ocudnn

    if NOT EXIST "%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v8.0\bin\cudart64_80.dll" (
        echo Failed to install CUDA
        exit /B 1
    )

    set PATH=%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v8.0\bin;%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v8.0\libnvvp;%PATH%

    nvcc -V
)
