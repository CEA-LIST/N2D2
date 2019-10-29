Performing simulations
======================


.. role:: raw-html(raw)
   :format: html

.. |check|  unicode:: U+02713 .. CHECK MARK
.. |cross|  unicode:: U+02717 .. BALLOT X

.. |ccheck| replace:: :raw-html:`<font color="green">` |check| :raw-html:`</font>`
.. |ccross| replace:: :raw-html:`<font color="red">` |cross| :raw-html:`</font>`



Obtaining the latest version of this manual
-------------------------------------------

Before going further, please make sure you are reading the latest
version of this manual. It is located in the manual sub-directory. To
compile the manual in PDF, just run the following command:

::

    cd manual && make

In order to compile the manual, you must have ``pdflatex`` and
``bibtex`` installed, as well as some common LaTeX packages.

- On Ubuntu, this can be done by installing the ``texlive`` and
  ``texlive-latex-extra`` software packages.

- On Windows, you can install the ``MiKTeX`` software, which includes
  everything needed and will install the required LaTeX packages on the
  fly.

Minimum system requirements
---------------------------

- Supported processors:

  - ARM Cortex A15 (tested on Tegra K1)
  - ARM Cortex A53/A57 (tested on Tegra X1)
  - Pentium-compatible PC (Pentium III, Athlon or more-recent system 
    recommended)

- Supported operating systems:

  - Windows :math:`\geq` 7 or Windows Server
    :math:`\geq` 2012, 64 bits with Visual Studio :math:`\geq` 2015.2 (2015
    Update 2)
  - GNU/Linux with GCC :math:`\geq` 4.4 (tested on RHEL
    :math:`\geq` 6, Debian :math:`\geq` 6, Ubuntu :math:`\geq` 14.04)

- At least 256 MB of RAM (1 GB with GPU/CUDA) for MNIST dataset processing

- At least 150 MB available hard disk space + 350 MB for MNIST dataset
  processing

For CUDA acceleration:

- CUDA :math:`\geq` 6.5 and CuDNN :math:`\geq` 1.0

- NVIDIA GPU with CUDA compute capability :math:`\geq` 3 (starting from
  *Kepler* micro-architecture)

- At least 512 MB GPU RAM for MNIST dataset processing

Obtaining N2D2
--------------

Prerequisites
~~~~~~~~~~~~~

Red Hat Enterprise Linux (RHEL) 6
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Make sure you have the following packages installed:

- ``cmake``

- ``gnuplot``

- ``opencv``

- ``opencv-devel`` (may require the ``rhel-x86_64-workstation-optional-6``
  repository channel)

Plus, to be able to use GPU acceleration:

- Install the CUDA repository package:

::

    rpm -Uhv http://developer.download.nvidia.com/compute/cuda/repos/rhel6/x86_64/cuda-repo-rhel6-7.5-18.x86_64.rpm
    yum clean expire-cache
    yum install cuda

- Install cuDNN from the NVIDIA website: register to `NVIDIA
  Developer <https://developer.nvidia.com/cudnn>`__ and download the
  latest version of cuDNN. Simply copy the header and library files from
  the cuDNN archive to the corresponding directories in the CUDA
  installation path (by default: /usr/local/cuda/include and
  /usr/local/cuda/lib64, respectively).

- Make sure the CUDA library path (e.g. /usr/local/cuda/lib64) is added to
  the LD\_LIBRARY\_PATH environment variable.

Ubuntu
^^^^^^

Make sure you have the following packages installed, if they are
available on your Ubuntu version:

- ``cmake``

- ``gnuplot``

- ``libopencv-dev``

- ``libcv-dev``

- ``libhighgui-dev``

Plus, to be able to use GPU acceleration:

- Install the CUDA repository package matching your distribution. For
  example, for Ubuntu 14.04 64 bits:

::

    wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu!\color{gray}{1404}!/!\color{gray}{x86\_64}!/cuda-repo-ubuntu!\color{gray}{1404}!_7.5-18_!\color{gray}{amd64}!.deb
    dpkg -i cuda-repo-ubuntu!\color{gray}{1404}!_7.5-18_!\color{gray}{amd64}!.deb

- Install the cuDNN repository package matching your distribution. For
  example, for Ubuntu 14.04 64 bits:

::

    wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu!\color{gray}{1404}!/!\color{gray}{x86\_64}!/nvidia-machine-learning-repo-ubuntu!\color{gray}{1404}!_4.0-2_!\color{gray}{amd64}!.deb
    dpkg -i nvidia-machine-learning-repo-ubuntu!\color{gray}{1404}!_4.0-2_!\color{gray}{amd64}!.deb

  Note that the cuDNN repository package is provided by NVIDIA for Ubuntu
  starting from version 14.04.

- Update the package lists: ``apt-get update``

- Install the CUDA and cuDNN required packages:

::

    apt-get install cuda-core-7-5 cuda-cudart-dev-7-5 cuda-cublas-dev-7-5 cuda-curand-dev-7-5 libcudnn5-dev

- Make sure there is a symlink to ``/usr/local/cuda``:

::

    ln -s /usr/local/cuda-7.5 /usr/local/cuda

- Make sure the CUDA library path (e.g. /usr/local/cuda/lib64) is added to
  the LD\_LIBRARY\_PATH environment variable.

Windows
^^^^^^^

On Windows 64 bits, Visual Studio :math:`\geq` 2015.2 (2015 Update 2) is
required.

Make sure you have the following software installed:

- CMake (http://www.cmake.org/): download and run the Windows installer.

- ``dirent.h`` C++ header (https://github.com/tronkko/dirent): to be put
  in the Visual Studio include path.

- Gnuplot (http://www.gnuplot.info/): the bin sub-directory in the install
  path needs to be added to the Windows ``PATH`` environment variable.

- OpenCV (http://opencv.org/): download the latest 2.x version for Windows
  and extract it to, for example, ``C:\OpenCV\``. Make sure to define the
  environment variable ``OpenCV_DIR`` to point to
  ``C:\OpenCV\opencv\build``. Make sure to add the bin sub-directory
  (``C:\OpenCV\opencv\build\x64\vc12\bin``) to the Windows ``PATH``
  environment variable.

Plus, to be able to use GPU acceleration:

- Download and install CUDA toolkit 8.0 located at
  https://developer.nvidia.com/compute/cuda/8.0/prod/local_installers/cuda_8.0.44_windows-exe:

::

    rename cuda_8.0.44_windows-exe cuda_8.0.44_windows.exe
    cuda_8.0.44_windows.exe -s compiler_8.0 cublas_8.0 cublas_dev_8.0 cudart_8.0 curand_8.0 curand_dev_8.0

- Update the ``PATH`` environment variable:

::

    set PATH=%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v8.0\bin;%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v8.0\libnvvp;%PATH%

- Download and install cuDNN 8.0 located at
  http://developer.download.nvidia.com/compute/redist/cudnn/v5.1/cudnn-8.0-windows7-x64-v5.1.zip
  (the following command assumes that you have 7-Zip installed):

::

    7z x cudnn-8.0-windows7-x64-v5.1.zip
    copy cuda\include\*.* ^
      "%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v8.0\include\"
    copy cuda\lib\x64\*.* ^
      "%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v8.0\lib\x64\"
    copy cuda\bin\*.* ^
      "%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v8.0\bin\"

Getting the sources
~~~~~~~~~~~~~~~~~~~

Use the following command:

::

    git clone git@github.com:CEA-LIST/N2D2.git

Compilation
~~~~~~~~~~~

To compile the program:

::

    mkdir build
    cd build
    cmake .. && make

On Windows, you may have to specify the generator, for example:

::

    cmake .. -G"Visual Studio 14"

Then open the newly created N2D2 project in Visual Studio 2015. Select
“Release” for the build target. Right click on ``ALL_BUILD`` item and
select “Build”.

Downloading training datasets
-----------------------------

A python script located in the repository root directory allows you to
select and automatically download some well-known datasets, like MNIST
and GTSRB (the script requires Python 2.x with bindings for GTK 2
package):

::

    ./tools/install_stimuli_gui.py

By default, the datasets are downloaded in the path specified in the
``N2D2_DATA`` environment variable, which is the root path used by the
N2D2 tool to locate the databases. If the ``N2D2_DATA`` variable is not
set, the default value used is /local/$USER/n2d2\_data/ (or
/local/n2d2\_data/ if the ``USER`` environment variable is not set) on
Linux and C:\\n2d2\_data\\ on Windows.

Please make sure you have write access to the ``N2D2_DATA`` path, or if
not set, in the default /local/$USER/n2d2\_data/ path.

Run the learning
----------------

The following command will run the learning for 600,000 image
presentations/steps and log the performances of the network every 10,000
steps:

::

    ./n2d2 "mnist24_16c4s2_24c5s2_150_10.ini" -learn 600000 -log 10000

Note: you may want to check the gradient computation using the
``-check`` option. Note that it can be extremely long and can
occasionally fail if the required precision is too high.

Test a learned network
----------------------

After the learning is completed, this command evaluate the network
performances on the test data set:

::

    ./n2d2 "mnist24_16c4s2_24c5s2_150_10.ini" -test

Import an ONNX model
~~~~~~~~~~~~~~~~~~~~

Instead of loading a N2D2 INI file, you can directly import an ONNX model with
the same command.

Additionally, it is possible to include an ONNX model inside a N2D2 INI file.
This is particularly useful to add pre-processing and post-processing to an
existing ONNX model. Below is an example with the MobileNet ONNX model provided
by Google:

.. code-block:: ini

    $BATCH_SIZE=256

    DefaultModel=Frame_CUDA

    ; Database
    [database]
    Type=ILSVRC2012_Database
    RandomPartitioning=0
    Learn=1.0
    BackgroundClass=1  ; Necessary for Google MobileNet pre-trained models

    ; Environment
    [sp]
    SizeX=224
    SizeY=224
    NbChannels=3
    BatchSize=${BATCH_SIZE}

    [sp.Transformation-1]
    Type=RescaleTransformation
    Width=256
    Height=256

    [sp.Transformation-2]
    Type=PadCropTransformation
    Width=224
    Height=224

    [sp.Transformation-3]
    Type=ColorSpaceTransformation
    ColorSpace=RGB

    [sp.Transformation-4]
    Type=RangeAffineTransformation
    FirstOperator=Minus
    FirstValue=127.5
    SecondOperator=Divides
    SecondValue=127.5

    ; Here, we insert an ONNX graph in the N2D2 flow the same way as a regular Cell
    [onnx]
    Input=sp
    Type=ONNX
    File=mobilenet_v1_1.0_224.onnx

    ; We can add targets to ONNX cells
    [MobilenetV1/Predictions/Softmax:0.Target-Top5]
    TopN=5


The table below summarizes the currently implemented ONNX operators:

+-----------------------+-----------+---------------------------------------------+
| Operator              | Support   | Remarks                                     |
+=======================+===========+=============================================+
| Add                   | |ccheck|  |                                             |
+-----------------------+-----------+---------------------------------------------+
| AveragePool           | |check|   | Exc. `ceil_mode` and `count_include_pad`    |
|                       |           | attributes                                  |
+-----------------------+-----------+---------------------------------------------+
| BatchNormalization    | |ccheck|  |                                             |
+-----------------------+-----------+---------------------------------------------+
| Cast                  | |cross|   | Ignored                                     |
+-----------------------+-----------+---------------------------------------------+
| Clip                  | |check|   | Only for `min` = 0 and `max` > 0            |
+-----------------------+-----------+---------------------------------------------+
| Concat                | |check|   | Only for layers that support it             |
+-----------------------+-----------+---------------------------------------------+
| Constant              | |ccheck|  | In some contexts only                       |
+-----------------------+-----------+---------------------------------------------+
| Conv                  | |ccheck|  |                                             |
+-----------------------+-----------+---------------------------------------------+
| Dropout               | |ccheck|  | Exc. `mask` output                          |
+-----------------------+-----------+---------------------------------------------+
| Div                   | |check|   | With constant second operand only           |
+-----------------------+-----------+---------------------------------------------+
| Flatten               | |check|   | Ignored (not necessary)                     |
+-----------------------+-----------+---------------------------------------------+
| Gemm                  | |ccross|  | Planned (only for fully-connected layers)   |
+-----------------------+-----------+---------------------------------------------+
| GlobalAveragePool     | |ccheck|  |                                             |
+-----------------------+-----------+---------------------------------------------+
| GlobalMaxPool         | |ccheck|  |                                             |
+-----------------------+-----------+---------------------------------------------+
| LRN                   | |ccheck|  |                                             |
+-----------------------+-----------+---------------------------------------------+
| LeakyRelu             | |ccheck|  |                                             |
+-----------------------+-----------+---------------------------------------------+
| MatMul                | |check|   | Only for fully-connected layers             |
+-----------------------+-----------+---------------------------------------------+
| Max                   | |ccheck|  |                                             |
+-----------------------+-----------+---------------------------------------------+
| MaxPool               | |ccheck|  | Exc. `Indices` output                       |
+-----------------------+-----------+---------------------------------------------+
| Mul                   | |ccheck|  |                                             |
+-----------------------+-----------+---------------------------------------------+
| Pad                   | |ccross|  | Planned                                     |
+-----------------------+-----------+---------------------------------------------+
| Relu                  | |ccheck|  |                                             |
+-----------------------+-----------+---------------------------------------------+
| Reshape               | |cross|   | Ignored                                     |
+-----------------------+-----------+---------------------------------------------+
| Resize                | |ccross|  | Planned (partially)                         |
+-----------------------+-----------+---------------------------------------------+
| Shape                 | |cross|   | Ignored                                     |
+-----------------------+-----------+---------------------------------------------+
| Sigmoid               | |ccheck|  |                                             |
+-----------------------+-----------+---------------------------------------------+
| Softmax               | |ccheck|  |                                             |
+-----------------------+-----------+---------------------------------------------+
| Softplus              | |ccheck|  |                                             |
+-----------------------+-----------+---------------------------------------------+
| Squeeze               | |cross|   | Ignored                                     |
+-----------------------+-----------+---------------------------------------------+
| Sub                   | |ccheck|  |                                             |
+-----------------------+-----------+---------------------------------------------+
| Sum                   | |ccheck|  |                                             |
+-----------------------+-----------+---------------------------------------------+
| Tanh                  | |ccheck|  |                                             |
+-----------------------+-----------+---------------------------------------------+
| Upsample              | |ccross|  | Planned                                     |
+-----------------------+-----------+---------------------------------------------+


Interpreting the results
~~~~~~~~~~~~~~~~~~~~~~~~

Recognition rate
^^^^^^^^^^^^^^^^

The recognition rate and the validation score are reported during the
learning in the *TargetScore\_/Success\_validation.png* file, as shown
in figure [fig:validationScore].

.. figure:: _static/validation_score.png
   :alt: Recognition rate and validation score during learning.

   Recognition rate and validation score during learning.

Confusion matrix
^^^^^^^^^^^^^^^^

The software automatically outputs the confusion matrix during learning,
validation and test, with an example shown in figure
[fig:ConfusionMatrix]. Each row of the matrix contains the number of
occurrences estimated by the network for each label, for all the data
corresponding to a single actual, target label. Or equivalently, each
column of the matrix contains the number of actual, target label
occurrences, corresponding to the same estimated label. Idealy, the
matrix should be diagonal, with no occurrence of an estimated label for
a different actual label (network mistake).

.. figure:: _static/confusion_matrix.png
   :alt: Example of confusion matrix obtained after the learning.

   Example of confusion matrix obtained after the learning.

The confusion matrix reports can be found in the simulation directory:

- *TargetScore\_/ConfusionMatrix\_learning.png*;

- *TargetScore\_/ConfusionMatrix\_validation.png*;

- *TargetScore\_/ConfusionMatrix\_test.png*.

Memory and computation requirements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The software also report the memory and computation requirements of the
network, as shown in figure [fig:stats]. The corresponding report can be
found in the *stats* sub-directory of the simulation.

.. figure:: _static/stats.png
   :alt: Example of memory and computation requirements of the network.

   Example of memory and computation requirements of the network.

Kernels and weights distribution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The synaptic weights obtained during and after the learning can be
analyzed, in terms of distribution (*weights* sub-directory of the
simulation) or in terms of kernels (*kernels* sub-directory of the
simulation), as shown in [fig:weights].


Output maps activity
^^^^^^^^^^^^^^^^^^^^

The initial output maps activity for each layer can be visualized in the
*outputs\_init* sub-directory of the simulation, as shown in figure
[fig:outputs].

.. figure:: _static/conv1-dat.png
   :alt: Output maps activity example of the first convolutional layer
         of the network.

   Output maps activity example of the first convolutional layer of the
   network.

Export a learned network
------------------------

::

    ./n2d2 "mnist24_16c4s2_24c5s2_150_10.ini" -export CPP_OpenCL

Export types:

- ``C`` C export using OpenMP;

- ``C_HLS`` C export tailored for HLS with Vivado HLS;

- ``CPP_OpenCL`` C++ export using OpenCL;

- ``CPP_Cuda`` C++ export using Cuda;

- ``CPP_cuDNN`` C++ export using cuDNN;

- ``CPP_TensorRT`` C++ export using tensorRT 2.1 API;

- ``SC_Spike`` SystemC spike export.

Other program options related to the exports:

+--------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Option [default value]   | Description                                                                                                                                                                                                                                                                                                   |
+==========================+===============================================================================================================================================================================================================================================================================================================+
| ``-nbbits`` [8]          | Number of bits for the weights and signals. Must be 8, 16, 32 or 64 for integer export, or -32, -64 for floating point export. The number of bits can be arbitrary for the ``C_HLS`` export (for example, 6 bits). It must be -32 for the ``CPP_TensorRT`` export, the precision is directly set at runtime   |
+--------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``-calib`` [0]           | Number of stimuli used for the calibration. 0 = no calibration (default), -1 = use the full test dataset for calibration                                                                                                                                                                                      |
+--------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``-calib-passes`` [2]    | Number of KL passes for determining the layer output values distribution truncation threshold (0 = use the max. value, no truncation)                                                                                                                                                                         |
+--------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``-no-unsigned``         | If present, disable the use of unsigned data type in integer exports                                                                                                                                                                                                                                          |
+--------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``-db-export`` [-1]      | Max. number of stimuli to export (0 = no dataset export, -1 = unlimited)                                                                                                                                                                                                                                      |
+--------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

C export
~~~~~~~~

Test the exported network:

::

    cd export_C_int8
    make
    ./bin/n2d2_test

The result should look like:

::

    ...
    1652.00/1762    (avg = 93.757094%)
    1653.00/1763    (avg = 93.760635%)
    1654.00/1764    (avg = 93.764172%)
    Tested 1764 stimuli
    Success rate = 93.764172%
    Process time per stimulus = 187.548186 us (12 threads)

    Confusion matrix:
    -------------------------------------------------
    | T \ E |       0 |       1 |       2 |       3 |
    -------------------------------------------------
    |     0 |     329 |       1 |       5 |       2 |
    |       |  97.63% |   0.30% |   1.48% |   0.59% |
    |     1 |       0 |     692 |       2 |       6 |
    |       |   0.00% |  98.86% |   0.29% |   0.86% |
    |     2 |      11 |      27 |     609 |      55 |
    |       |   1.57% |   3.85% |  86.75% |   7.83% |
    |     3 |       0 |       0 |       1 |      24 |
    |       |   0.00% |   0.00% |   4.00% |  96.00% |
    -------------------------------------------------
    T: Target    E: Estimated

CPP\_OpenCL export
~~~~~~~~~~~~~~~~~~

The OpenCL export can run the generated program in GPU or CPU
architectures. Compilation features:

+----------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Preprocessor command [default value]   | Description                                                                                                                                                           |
+========================================+=======================================================================================================================================================================+
| ``PROFILING`` [0]                      | Compile the binary with a synchronization between each layers and return the mean execution time of each layer. This preprocessor option can decrease performances.   |
+----------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``GENERATE_KBIN`` [0]                  | Generate the binary output of the OpenCL kernel .cl file use. The binary is store in the /bin folder.                                                                 |
+----------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``LOAD_KBIN`` [0]                      | Indicate to the program to load an OpenCL kernel as a binary from the /bin folder instead of a .cl file.                                                              |
+----------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``CUDA`` [0]                           | Use the CUDA OpenCL SDK locate at :math:`{/usr/local/cuda}`                                                                                                           |
+----------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``MALI`` [0]                           | Use the MALI OpenCL SDK locate at :math:`{/usr/Mali_OpenCL_SDK_vXXX}`                                                                                                 |
+----------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``INTEL`` [0]                          | Use the INTEL OpenCL SDK locate at :math:`{/opt/intel/opencl}`                                                                                                        |
+----------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``AMD`` [1]                            | Use the AMD OpenCL SDK locate at :math:`{/opt/AMDAPPSDK-XXX}`                                                                                                         |
+----------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------+

Program options related to the OpenCL export:

+--------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Option [default value]   | Description                                                                                                                                                        |
+==========================+====================================================================================================================================================================+
| ``-cpu``                 | If present, force to use a CPU architecture to run the program                                                                                                     |
+--------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``-gpu``                 | If present, force to use a GPU architecture to run the program                                                                                                     |
+--------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``-batch`` [1]           | Size of the batch to use                                                                                                                                           |
+--------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``-stimulus`` [NULL]     | Path to a specific input stimulus to test. For example: -stimulus :math:`{/stimulus/env0000.pgm}` command will test the file env0000.pgm of the stimulus folder.   |
+--------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------+

Test the exported network:

::

    cd export_CPP_OpenCL_float32
    make
    ./bin/n2d2_opencl_test -gpu

CPP\_TensorRT export
~~~~~~~~~~~~~~~~~~~~

The TensorRT API export can run the generated program in NVIDIA GPU
architecture. It use CUDA, cuDNN and TensorRT API library. All the
native TensorRT layers are supported. The export support from TensorRT
2.1 to TensorRT 5.0 versions.

Program options related to the TensorRT API export:

+--------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Option [default value]   | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
+==========================+===============================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================+
| ``-batch`` [1]           | Size of the batch to use                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
+--------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``-dev`` [0]             | CUDA Device ID selection                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
+--------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``-stimulus`` [NULL]     | Path to a specific input stimulus to test. For example: -stimulus :math:`{/stimulus/env0000.pgm}` command will test the file env0000.pgm of the stimulus folder.                                                                                                                                                                                                                                                                                                                                              |
+--------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``-prof``                | Activates the layer wise profiling mechanism. This option can decrease execution time performance.                                                                                                                                                                                                                                                                                                                                                                                                            |
+--------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``-iter-build`` [1]      | Sets the number of minimization build iterations done by the tensorRT builder to find the best layer tactics.                                                                                                                                                                                                                                                                                                                                                                                                 |
+--------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``-nbbits`` [-32]        | Number of bits used for computation. Value -32 for Full FP32 bits configuration, -16 for Half FP16 bits configuration and 8 for INT8 bits configuration. When running INT8 mode for the first time, the TensorRT calibration process can be very long. Once generated the generated calibration table will be automatically reused. Supported compute mode in function of the compute capability are provided here: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities .   |
+--------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

Test the exported network with layer wise profiling:

::

    cd export_CPP_TensorRT_float32
    make
    ./bin/n2d2_tensorRT_test -prof

The results of the layer wise profiling should look like:

::

    (19%)  **************************************** CONV1 + CONV1_ACTIVATION: 0.0219467 ms
    (05%)  ************ POOL1: 0.00675573 ms
    (13%)  **************************** CONV2 + CONV2_ACTIVATION: 0.0159089 ms
    (05%)  ************ POOL2: 0.00616047 ms
    (14%)  ****************************** CONV3 + CONV3_ACTIVATION: 0.0159713 ms
    (19%)  **************************************** FC1 + FC1_ACTIVATION: 0.0222242 ms
    (13%)  **************************** FC2: 0.0149013 ms
    (08%)  ****************** SOFTMAX: 0.0100633 ms
    Average profiled tensorRT process time per stimulus = 0.113932 ms

CPP\_cuDNN export
~~~~~~~~~~~~~~~~~

The cuDNN export can run the generated program in NVIDIA GPU
architecture. It use CUDA and cuDNN library. Compilation features:

+----------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Preprocessor command [default value]   | Description                                                                                                                                                           |
+========================================+=======================================================================================================================================================================+
| ``PROFILING`` [0]                      | Compile the binary with a synchronization between each layers and return the mean execution time of each layer. This preprocessor option can decrease performances.   |
+----------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``ARCH32`` [0]                         | Compile the binary with the 32-bits architecture compatibility.                                                                                                       |
+----------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------+

Program options related to the cuDNN export:

+--------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Option [default value]   | Description                                                                                                                                                        |
+==========================+====================================================================================================================================================================+
| ``-batch`` [1]           | Size of the batch to use                                                                                                                                           |
+--------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``-dev`` [0]             | CUDA Device ID selection                                                                                                                                           |
+--------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``-stimulus`` [NULL]     | Path to a specific input stimulus to test. For example: -stimulus :math:`{/stimulus/env0000.pgm}` command will test the file env0000.pgm of the stimulus folder.   |
+--------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------+

Test the exported network:

::

    cd export_CPP_cuDNN_float32
    make
    ./bin/n2d2_cudnn_test

C\_HLS export
~~~~~~~~~~~~~

Test the exported network:

::

    cd export_C_HLS_int8
    make
    ./bin/n2d2_test

Run the High-Level Synthesis (HLS) with Xilinx Vivado HLS:

::

    vivado_hls -f run_hls.tcl

Layer compatibility table
~~~~~~~~~~~~~~~~~~~~~~~~~

Layer compatibility table in function of the export type:

+---------------+------------------------------------------------------+
| Layer         | Export type                                          |
| compatibility +----------+-------------+-------------+---------------+
| table         | C        | C\_HLS      | CPP\_OpenCL | CPP\_TensorRT |
+===============+==========+=============+=============+===============+
|Conv           | |ccheck| | |ccheck|    | |ccheck|    | |ccheck|      |
+---------------+----------+-------------+-------------+---------------+
|Pool           | |ccheck| | |ccheck|    | |ccheck|    | |ccheck|      |
+---------------+----------+-------------+-------------+---------------+
|Fc             | |ccheck| | |ccheck|    | |ccheck|    | |ccheck|      |
+---------------+----------+-------------+-------------+---------------+
|Softmax        | |ccheck| | |ccross|    | |ccheck|    | |ccheck|      |
+---------------+----------+-------------+-------------+---------------+
|FMP            | |ccheck| | |ccross|    | |ccheck|    | |ccross|      |
+---------------+----------+-------------+-------------+---------------+
|Deconv         | |ccross| | |ccross|    | |ccross|    | |ccheck|      |
+---------------+----------+-------------+-------------+---------------+
|ElemWise       | |ccross| | |ccross|    | |ccross|    | |ccheck|      |
+---------------+----------+-------------+-------------+---------------+
|Resize         | |ccheck| | |ccross|    | |ccross|    | |ccheck|      |
+---------------+----------+-------------+-------------+---------------+
|Padding        | |ccross| | |ccross|    | |ccross|    | |ccheck|      |
+---------------+----------+-------------+-------------+---------------+
|LRN            | |ccross| | |ccross|    | |ccross|    | |ccheck|      |
+---------------+----------+-------------+-------------+---------------+
|Anchor         | |ccross| | |ccross|    | |ccross|    | |ccheck|      |
+---------------+----------+-------------+-------------+---------------+
|ObjectDet      | |ccross| | |ccross|    | |ccross|    | |ccheck|      |
+---------------+----------+-------------+-------------+---------------+
|ROIPooling     | |ccross| | |ccross|    | |ccross|    | |ccheck|      |
+---------------+----------+-------------+-------------+---------------+
|RP             | |ccross| | |ccross|    | |ccross|    | |ccheck|      |
+---------------+----------+-------------+-------------+---------------+


BatchNorm is not mentionned because batch normalization parameters are
automatically fused with convolutions parameters with the command
“-fuse”.
