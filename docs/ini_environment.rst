Environment
===========

The environment simply specify the input data format of the network
(width, height and batch size). Example:

::

    [env]
    SizeX=24
    SizeY=24
    BatchSize=12 ; [default: 1]

+--------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Option [default value]               | Description                                                                                                                                                                                                                                                                                                  |
+======================================+==============================================================================================================================================================================================================================================================================================================+
| ``SizeX``                            | Environment width                                                                                                                                                                                                                                                                                            |
+--------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``SizeY``                            | Environment height                                                                                                                                                                                                                                                                                           |
+--------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``NbChannels`` [1]                   | Number of channels (applicable only if there is no ``env.ChannelTransformation[...]``)                                                                                                                                                                                                                       |
+--------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``BatchSize`` [1]                    | Batch size                                                                                                                                                                                                                                                                                                   |
+--------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``CompositeStimuli`` [0]             | If true, use pixel-wise stimuli labels                                                                                                                                                                                                                                                                       |
+--------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``CachePath`` []                     | Stimuli cache path (no cache if left empty)                                                                                                                                                                                                                                                                  |
+--------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``StimulusType`` [``SingleBurst``]   | Method for converting stimuli into spike trains. Can be any of ``SingleBurst``, ``Periodic``, ``JitteredPeriodic`` or ``Poissonian``                                                                                                                                                                         |
+--------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``DiscardedLateStimuli`` [1.0]       | The pixels in the pre-processed stimuli with a value above this limit never generate spiking events                                                                                                                                                                                                          |
+--------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``PeriodMeanMin`` [50 ``TimeMs``]    | Mean minimum period :math:`\overline{T_{min}}`, used for periodic temporal codings, corresponding to pixels in the pre-processed stimuli with a value of 0 (which are supposed to be the most significant pixels)                                                                                            |
+--------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``PeriodMeanMax`` [12 ``TimeS``]     | Mean maximum period :math:`\overline{T_{max}}`, used for periodic temporal codings, corresponding to pixels in the pre-processed stimuli with a value of 1 (which are supposed to be the least significant pixels). This maximum period may be never reached if ``DiscardedLateStimuli`` is lower than 1.0   |
+--------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``PeriodRelStdDev`` [0.1]            | Relative standard deviation, used for periodic temporal codings, applied to the spiking period of a pixel                                                                                                                                                                                                    |
+--------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``PeriodMin`` [11 ``TimeMs``]        | Absolute minimum period, or spiking interval, used for periodic temporal codings, for any pixel                                                                                                                                                                                                              |
+--------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

Built-in transformations
------------------------

There are 6 possible categories of transformations:

``env.Transformation[...]`` Transformations applied to the input images
before channels creation;

``env.OnTheFlyTransformation[...]`` On-the-fly transformations applied
to the input images before channels creation;

``env.ChannelTransformation[...]`` Create or add transformation for a
specific channel;

``env.ChannelOnTheFlyTransformation[...]`` Create or add on-the-fly
transformation for a specific channel;

``env.ChannelsTransformation[...]`` Transformations applied to all the
channels of the input images;

``env.ChannelsOnTheFlyTransformation[...]`` On-the-fly transformations
applied to all the channels of the input images.

Example:

::

    [env.Transformation]
    Type=PadCropTransformation
    Width=24
    Height=24

Several transformations can applied successively. In this case, to be
able to apply multiple transformations of the same category, a different
suffix (``[...]``) must be added to each transformation.

**The transformations will be processed in the order of appearance in
the INI file regardless of their suffix.**

Common set of parameters for any kind of transformation:

+--------------------------+------------------------------------------------------------------------+
| Option [default value]   | Description                                                            |
+==========================+========================================================================+
| ``ApplyTo`` [``All``]    | Apply the transformation only to the specified stimuli sets. Can be:   |
+--------------------------+------------------------------------------------------------------------+
|                          | ``LearnOnly``: learning set only                                       |
+--------------------------+------------------------------------------------------------------------+
|                          | ``ValidationOnly``: validation set only                                |
+--------------------------+------------------------------------------------------------------------+
|                          | ``TestOnly``: testing set only                                         |
+--------------------------+------------------------------------------------------------------------+
|                          | ``NoLearn``: validation and testing sets only                          |
+--------------------------+------------------------------------------------------------------------+
|                          | ``NoValidation``: learning and testing sets only                       |
+--------------------------+------------------------------------------------------------------------+
|                          | ``NoTest``: learning and validation sets only                          |
+--------------------------+------------------------------------------------------------------------+
|                          | ``All``: all sets (default)                                            |
+--------------------------+------------------------------------------------------------------------+

Example:

::

    [env.Transformation-1]
    Type=ChannelExtractionTransformation
    CSChannel=Gray

    [env.Transformation-2]
    Type=RescaleTransformation
    Width=29
    Height=29

    [env.Transformation-3]
    Type=EqualizeTransformation

    [env.OnTheFlyTransformation]
    Type=DistortionTransformation
    ApplyTo=LearnOnly ; Apply this transformation for the Learning set only
    ElasticGaussianSize=21
    ElasticSigma=6.0
    ElasticScaling=20.0
    Scaling=15.0
    Rotation=15.0

List of available transformations:

AffineTransformation
~~~~~~~~~~~~~~~~~~~~

Apply an element-wise affine transformation to the image with matrixes
of the same size.

+---------------------------------+-----------------------------------------------------------------------------------------+
| Option [default value]          | Description                                                                             |
+=================================+=========================================================================================+
| ``FirstOperator``               | First element-wise operator, can be ``Plus``, ``Minus``, ``Multiplies``, ``Divides``    |
+---------------------------------+-----------------------------------------------------------------------------------------+
| ``FirstValue``                  | First matrix file name                                                                  |
+---------------------------------+-----------------------------------------------------------------------------------------+
| ``SecondOperator`` [``Plus``]   | Second element-wise operator, can be ``Plus``, ``Minus``, ``Multiplies``, ``Divides``   |
+---------------------------------+-----------------------------------------------------------------------------------------+
| ``SecondValue`` []              | Second matrix file name                                                                 |
+---------------------------------+-----------------------------------------------------------------------------------------+

The final operation is the following, with :math:`A` the image matrix,
:math:`B_{1st}`, :math:`B_{2nd}` the matrixes to
add/substract/multiply/divide and :math:`\odot` the element-wise
operator :

.. math::

   f(A) = \left(A\;\substack{\odot\\op_{1st}}\;B_{1st}\right)\;
   \substack{\odot\\op_{2nd}}\;B_{2nd}

ApodizationTransformation
~~~~~~~~~~~~~~~~~~~~~~~~~

Apply an apodization window to each data row.

+------------------------------------+-------------------------------------------------------------+
| Option [default value]             | Description                                                 |
+====================================+=============================================================+
| ``Size``                           | Window total size (must match the number of data columns)   |
+------------------------------------+-------------------------------------------------------------+
| ``WindowName`` [``Rectangular``]   | Window name. Possible values are:                           |
+------------------------------------+-------------------------------------------------------------+
|                                    | ``Rectangular``: Rectangular                                |
+------------------------------------+-------------------------------------------------------------+
|                                    | ``Hann``: Hann                                              |
+------------------------------------+-------------------------------------------------------------+
|                                    | ``Hamming``: Hamming                                        |
+------------------------------------+-------------------------------------------------------------+
|                                    | ``Cosine``: Cosine                                          |
+------------------------------------+-------------------------------------------------------------+
|                                    | ``Gaussian``: Gaussian                                      |
+------------------------------------+-------------------------------------------------------------+
|                                    | ``Blackman``: Blackman                                      |
+------------------------------------+-------------------------------------------------------------+
|                                    | ``Kaiser``: Kaiser                                          |
+------------------------------------+-------------------------------------------------------------+

Gaussian window
^^^^^^^^^^^^^^^

Gaussian window.

| \| m4cm \| m7cm \| Option [default value] & Description
| *WindowName*\ ``.Sigma`` [0.4] & Sigma

Blackman window
^^^^^^^^^^^^^^^

Blackman window.

| \| m4cm \| m7cm \| Option [default value] & Description
| *WindowName*\ ``.Alpha`` [0.16] & Alpha

Kaiser window
^^^^^^^^^^^^^

Kaiser window.

| \| m4cm \| m7cm \| Option [default value] & Description
| *WindowName*\ ``.Beta`` [5.0] & Beta

ChannelExtractionTransformation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Extract an image channel.

+-----------------+---------------------------------------------------------------------------------------+
| Option          | Description                                                                           |
+=================+=======================================================================================+
| ``CSChannel``   | ``Blue``: blue channel in the BGR colorspace, or first channel of any colorspace      |
+-----------------+---------------------------------------------------------------------------------------+
|                 | ``Green``: green channel in the BGR colorspace, or second channel of any colorspace   |
+-----------------+---------------------------------------------------------------------------------------+
|                 | ``Red``: red channel in the BGR colorspace, or third channel of any colorspace        |
+-----------------+---------------------------------------------------------------------------------------+
|                 | ``Hue``: hue channel in the HSV colorspace                                            |
+-----------------+---------------------------------------------------------------------------------------+
|                 | ``Saturation``: saturation channel in the HSV colorspace                              |
+-----------------+---------------------------------------------------------------------------------------+
|                 | ``Value``: value channel in the HSV colorspace                                        |
+-----------------+---------------------------------------------------------------------------------------+
|                 | ``Gray``: gray conversion                                                             |
+-----------------+---------------------------------------------------------------------------------------+
|                 | ``Y``: Y channel in the YCbCr colorspace                                              |
+-----------------+---------------------------------------------------------------------------------------+
|                 | ``Cb``: Cb channel in the YCbCr colorspace                                            |
+-----------------+---------------------------------------------------------------------------------------+
|                 | ``Cr``: Cr channel in the YCbCr colorspace                                            |
+-----------------+---------------------------------------------------------------------------------------+

ColorSpaceTransformation
~~~~~~~~~~~~~~~~~~~~~~~~

Change the current image colorspace.

+------------------+-------------------------------------------------------+
| Option           | Description                                           |
+==================+=======================================================+
| ``ColorSpace``   | ``BGR``: convert any gray, BGR or BGRA image to BGR   |
+------------------+-------------------------------------------------------+
|                  | ``RGB``: convert any gray, BGR or BGRA image to RGB   |
+------------------+-------------------------------------------------------+
|                  | ``HSV``: convert BGR image to HSV                     |
+------------------+-------------------------------------------------------+
|                  | ``HLS``: convert BGR image to HLS                     |
+------------------+-------------------------------------------------------+
|                  | ``YCrCb``: convert BGR image to YCrCb                 |
+------------------+-------------------------------------------------------+
|                  | ``CIELab``: convert BGR image to CIELab               |
+------------------+-------------------------------------------------------+
|                  | ``CIELuv``: convert BGR image to CIELuv               |
+------------------+-------------------------------------------------------+
|                  | ``RGB_to_BGR``: convert RGB image to BGR              |
+------------------+-------------------------------------------------------+
|                  | ``RGB_to_HSV``: convert RGB image to HSV              |
+------------------+-------------------------------------------------------+
|                  | ``RGB_to_HLS``: convert RGB image to HLS              |
+------------------+-------------------------------------------------------+
|                  | ``RGB_to_YCrCb``: convert RGB image to YCrCb          |
+------------------+-------------------------------------------------------+
|                  | ``RGB_to_CIELab``: convert RGB image to CIELab        |
+------------------+-------------------------------------------------------+
|                  | ``RGB_to_CIELuv``: convert RGB image to CIELuv        |
+------------------+-------------------------------------------------------+
|                  | ``HSV_to_BGR``: convert HSV image to BGR              |
+------------------+-------------------------------------------------------+
|                  | ``HSV_to_RGB``: convert HSV image to RGB              |
+------------------+-------------------------------------------------------+
|                  | ``HLS_to_BGR``: convert HLS image to BGR              |
+------------------+-------------------------------------------------------+
|                  | ``HLS_to_RGB``: convert HLS image to RGB              |
+------------------+-------------------------------------------------------+
|                  | ``YCrCb_to_BGR``: convert YCrCb image to BGR          |
+------------------+-------------------------------------------------------+
|                  | ``YCrCb_to_RGB``: convert YCrCb image to RGB          |
+------------------+-------------------------------------------------------+
|                  | ``CIELab_to_BGR``: convert CIELab image to BGR        |
+------------------+-------------------------------------------------------+
|                  | ``CIELab_to_RGB``: convert CIELab image to RGB        |
+------------------+-------------------------------------------------------+
|                  | ``CIELuv_to_BGR``: convert CIELuv image to BGR        |
+------------------+-------------------------------------------------------+
|                  | ``CIELuv_to_RGB``: convert CIELuv image to RGB        |
+------------------+-------------------------------------------------------+

Note that the default colorspace in N2D2 is BGR, the same as in OpenCV.

DFTTransformation
~~~~~~~~~~~~~~~~~

Apply a DFT to the data. The input data must be single channel, the
resulting data is two channels, the first for the real part and the
second for the imaginary part.

+--------------------------+-----------------------------------------------------------------------------------+
| Option [default value]   | Description                                                                       |
+==========================+===================================================================================+
| ``TwoDimensional`` [1]   | If true, compute a 2D image DFT. Otherwise, compute the 1D DFT of each data row   |
+--------------------------+-----------------------------------------------------------------------------------+

Note that this transformation can add zero-padding if required by the
underlying FFT implementation.

[par:DistortionTransformation]DistortionTransformation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Apply elastic distortion to the image. This transformation is generally
used on-the-fly (so that a different distortion is performed for each
image), and for the learning only.

+--------------------------------+-----------------------------------------------------------+
| Option [default value]         | Description                                               |
+================================+===========================================================+
| ``ElasticGaussianSize`` [15]   | Size of the gaussian for elastic distortion (in pixels)   |
+--------------------------------+-----------------------------------------------------------+
| ``ElasticSigma`` [6.0]         | Sigma of the gaussian for elastic distortion              |
+--------------------------------+-----------------------------------------------------------+
| ``ElasticScaling`` [0.0]       | Scaling of the gaussian for elastic distortion            |
+--------------------------------+-----------------------------------------------------------+
| ``Scaling`` [0.0]              | Maximum random scaling amplitude (+/-, in percentage)     |
+--------------------------------+-----------------------------------------------------------+
| ``Rotation`` [0.0]             | Maximum random rotation amplitude (+/-, in °)             |
+--------------------------------+-----------------------------------------------------------+

EqualizeTransformation
~~~~~~~~~~~~~~~~~~~~~~

Image histogram equalization.

+------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Option [default value]       | Description                                                                                                                                                                                     |
+==============================+=================================================================================================================================================================================================+
| ``Method`` [``Standard``]    | ``Standard``: standard histogram equalization                                                                                                                                                   |
+------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|                              | ``CLAHE``: contrast limited adaptive histogram equalization                                                                                                                                     |
+------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``CLAHE_ClipLimit`` [40.0]   | Threshold for contrast limiting (for ``CLAHE`` only)                                                                                                                                            |
+------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``CLAHE_GridSize`` [8]       | Size of grid for histogram equalization (for ``CLAHE`` only). Input image will be divided into equally sized rectangular tiles. This parameter defines the number of tiles in row and column.   |
+------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

ExpandLabelTransformation
~~~~~~~~~~~~~~~~~~~~~~~~~

Expand single image label (1x1 pixel) to full frame label.

FilterTransformation
~~~~~~~~~~~~~~~~~~~~

Apply a convolution filter to the image.

+--------------------------+--------------------------------------------+
| Option [default value]   | Description                                |
+==========================+============================================+
| ``Kernel``               | Convolution kernel. Possible values are:   |
+--------------------------+--------------------------------------------+
|                          | ``*``: custom kernel                       |
+--------------------------+--------------------------------------------+
|                          | ``Gaussian``: Gaussian kernel              |
+--------------------------+--------------------------------------------+
|                          | ``LoG``: Laplacian Of Gaussian kernel      |
+--------------------------+--------------------------------------------+
|                          | ``DoG``: Difference Of Gaussian kernel     |
+--------------------------+--------------------------------------------+
|                          | ``Gabor``: Gabor kernel                    |
+--------------------------+--------------------------------------------+

\* kernel
^^^^^^^^^

Custom kernel.

| \| m4cm \| m7cm \| Option & Description
| ``Kernel.SizeX`` [0] & Width of the kernel (numer of columns)
| ``Kernel.SizeY`` [0] & Height of the kernel (number of rows)
| ``Kernel.Mat`` & List of row-major ordered coefficients of the kernel

If both ``Kernel.SizeX`` and ``Kernel.SizeY`` are 0, the kernel is
assumed to be square.

Gaussian kernel
^^^^^^^^^^^^^^^

Gaussian kernel.

| \| m4cm \| m7cm \| Option [default value] & Description
| ``Kernel.SizeX`` & Width of the kernel (numer of columns)
| ``Kernel.SizeY`` & Height of the kernel (number of rows)
| ``Kernel.Positive`` [1] & If true, the center of the kernel is
  positive
| ``Kernel.Sigma`` [:math:`\sqrt{2.0}`] & Sigma of the kernel

LoG kernel
^^^^^^^^^^

Laplacian Of Gaussian kernel.

| \| m4cm \| m7cm \| Option [default value] & Description
| ``Kernel.SizeX`` & Width of the kernel (numer of columns)
| ``Kernel.SizeY`` & Height of the kernel (number of rows)
| ``Kernel.Positive`` [1] & If true, the center of the kernel is
  positive
| ``Kernel.Sigma`` [:math:`\sqrt{2.0}`] & Sigma of the kernel

DoG kernel
^^^^^^^^^^

Difference Of Gaussian kernel kernel.

| \| m4cm \| m7cm \| Option [default value] & Description
| ``Kernel.SizeX`` & Width of the kernel (numer of columns)
| ``Kernel.SizeY`` & Height of the kernel (number of rows)
| ``Kernel.Positive`` [1] & If true, the center of the kernel is
  positive
| ``Kernel.Sigma1`` [2.0] & Sigma1 of the kernel
| ``Kernel.Sigma2`` [1.0] & Sigma2 of the kernel

Gabor kernel
^^^^^^^^^^^^

Gabor kernel.

| \| m4cm \| m7cm \| Option [default value] & Description
| ``Kernel.SizeX`` & Width of the kernel (numer of columns)
| ``Kernel.SizeY`` & Height of the kernel (number of rows)
| ``Kernel.Theta`` & Theta of the kernel
| ``Kernel.Sigma`` [:math:`\sqrt{2.0}`] & Sigma of the kernel
| ``Kernel.Lambda`` [10.0] & Lambda of the kernel
| ``Kernel.Psi`` [:math:`\pi/2.0`] & Psi of the kernel
| ``Kernel.Gamma`` [0.5] & Gamma of the kernel

FlipTransformation
~~~~~~~~~~~~~~~~~~

Image flip transformation.

+--------------------------------+-------------------------------------------------+
| Option [default value]         | Description                                     |
+================================+=================================================+
| ``HorizontalFlip`` [0]         | If true, flip the image horizontally            |
+--------------------------------+-------------------------------------------------+
| ``VerticalFlip`` [0]           | If true, flip the image vertically              |
+--------------------------------+-------------------------------------------------+
| ``RandomHorizontalFlip`` [0]   | If true, randomly flip the image horizontally   |
+--------------------------------+-------------------------------------------------+
| ``RandomVerticalFlip`` [0]     | If true, randomly flip the image vertically     |
+--------------------------------+-------------------------------------------------+

GradientFilterTransformation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Compute image gradient.

+----------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Option [default value]           | Description                                                                                                                                                                                    |
+==================================+================================================================================================================================================================================================+
| ``Scale`` [1.0]                  | Scale to apply to the computed gradient                                                                                                                                                        |
+----------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``Delta`` [0.0]                  | Bias to add to the computed gradient                                                                                                                                                           |
+----------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``GradientFilter`` [``Sobel``]   | Filter type to use for computing the gradient. Possible options are: ``Sobel``, ``Scharr`` and ``Laplacian``                                                                                   |
+----------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``KernelSize`` [3]               | Size of the filter kernel (has no effect when using the ``Scharr`` filter, which kernel size is always 3x3)                                                                                    |
+----------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``ApplyToLabels`` [0]            | If true, use the computed gradient to filter the image label and ignore pixel areas where the gradient is below the ``Threshold``. In this case, only the labels are modified, not the image   |
+----------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``InvThreshold`` [0]             | If true, ignored label pixels will be the ones with a low gradient (low contrasted areas)                                                                                                      |
+----------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``Threshold`` [0.5]              | Threshold applied on the image gradient                                                                                                                                                        |
+----------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``Label`` []                     | List of labels to filter (space-separated)                                                                                                                                                     |
+----------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``GradientScale`` [1.0]          | Rescale the image by this factor before applying the gradient and the threshold, then scale it back to filter the labels                                                                       |
+----------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

LabelSliceExtractionTransformation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Extract a slice from an image belonging to a given label.

+---------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Option [default value]                | Description                                                                                                                                                                                                                                                                                                              |
+=======================================+==========================================================================================================================================================================================================================================================================================================================+
| ``Width``                             | Width of the slice to extract                                                                                                                                                                                                                                                                                            |
+---------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``Height``                            | Height of the slice to extract                                                                                                                                                                                                                                                                                           |
+---------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``Label`` [-1]                        | Slice should belong to this label ID. If -1, the label ID is random                                                                                                                                                                                                                                                      |
+---------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``RandomRotation`` [0]                | If true, extract randomly rotated slices                                                                                                                                                                                                                                                                                 |
+---------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``RandomRotationRange`` [0.0 360.0]   | Range of the random rotations, in degrees, counterclockwise (if ``RandomRotation`` is enabled)                                                                                                                                                                                                                           |
+---------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``SlicesMargin`` [0]                  | Positive or negative, indicates the margin around objects that can be extracted in the slice                                                                                                                                                                                                                             |
+---------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``KeepComposite`` [0]                 | If false, the 2D label image is reduced to a single value corresponding to the extracted object label (useful for patches classification tasks). Note that if ``SlicesMargin`` is > 0, the 2D label image may contain other labels before reduction. For pixel-wise segmentation tasks, set ``KeepComposite`` to true.   |
+---------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

This transformation is useful to learn sparse object occurrences in a
lot of background. If the dataset is very unbalanced towards background,
this transformation will ensure that the learning is done on a more
balanced set of every labels, regardless of their actual pixel-wise
ratio.

When ``SlicesMargin`` is 0, only slices that fully include a given label
are extracted, as shown in figure
[fig:LabelSliceExtractionTransformation0]. The behavior with
``SlicesMargin`` < 0 is illustrated in figure
[fig:LabelSliceExtractionTransformation1]. Note that setting a negative
``SlicesMargin`` larger in absolute value than ``Width``/2 or
``Height``/2 will lead in some (random) cases in incorrect slice labels
in respect to the majority pixel label in the slice.

MagnitudePhaseTransformation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Compute the magnitude and phase of a complex two channels input data,
with the first channel :math:`x` being the real part and the second
channel :math:`y` the imaginary part. The resulting data is two
channels, the first one with the magnitude and the second one with the
phase.

+--------------------------+-----------------------------------------------+
| Option [default value]   | Description                                   |
+==========================+===============================================+
| ``LogScale`` [0]         | If true, compute the magnitude in log scale   |
+--------------------------+-----------------------------------------------+

The magnitude is:

.. math:: M_{i,j} = \sqrt{x_{i,j}^2 + x_{i,j}^2}

If ``LogScale`` = 1, compute :math:`M'_{i,j} = log(1 + M_{i,j})`.

The phase is:

.. math:: \theta_{i,j} = atan2(y_{i,j}, x_{i,j})

MorphologicalReconstructionTransformation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Apply a morphological reconstruction transformation to the image. This
transformation is also useful for post-processing.

+-------------------------------+-----------------------------------------------------------------------------------------------------------------------+
| Option [default value]        | Description                                                                                                           |
+===============================+=======================================================================================================================+
| ``Operation``                 | Morphological operation to apply. Can be:                                                                             |
+-------------------------------+-----------------------------------------------------------------------------------------------------------------------+
|                               | ``ReconstructionByErosion``: reconstruction by erosion operation                                                      |
+-------------------------------+-----------------------------------------------------------------------------------------------------------------------+
|                               | ``ReconstructionByDilation``: reconstruction by dilation operation                                                    |
+-------------------------------+-----------------------------------------------------------------------------------------------------------------------+
|                               | ``OpeningByReconstruction``: opening by reconstruction operation                                                      |
+-------------------------------+-----------------------------------------------------------------------------------------------------------------------+
|                               | ``ClosingByReconstruction``: closing by reconstruction operation                                                      |
+-------------------------------+-----------------------------------------------------------------------------------------------------------------------+
| ``Size``                      | Size of the structuring element                                                                                       |
+-------------------------------+-----------------------------------------------------------------------------------------------------------------------+
| ``ApplyToLabels`` [0]         | If true, apply the transformation to the labels instead of the image                                                  |
+-------------------------------+-----------------------------------------------------------------------------------------------------------------------+
| ``Shape`` [``Rectangular``]   | Shape of the structuring element used for morphology operations. Can be ``Rectangular``, ``Elliptic`` or ``Cross``.   |
+-------------------------------+-----------------------------------------------------------------------------------------------------------------------+
| ``NbIterations`` [1]          | Number of times erosion and dilation are applied for opening and closing reconstructions                              |
+-------------------------------+-----------------------------------------------------------------------------------------------------------------------+

MorphologyTransformation
~~~~~~~~~~~~~~~~~~~~~~~~

Apply a morphology transformation to the image. This transformation is
also useful for post-processing.

+-------------------------------+-----------------------------------------------------------------------------------------------------------------------+
| Option [default value]        | Description                                                                                                           |
+===============================+=======================================================================================================================+
| ``Operation``                 | Morphological operation to apply. Can be:                                                                             |
+-------------------------------+-----------------------------------------------------------------------------------------------------------------------+
|                               | ``Erode``: erode operation (:math:`=erode(src)`)                                                                      |
+-------------------------------+-----------------------------------------------------------------------------------------------------------------------+
|                               | ``Dilate``: dilate operation (:math:`=dilate(src)`)                                                                   |
+-------------------------------+-----------------------------------------------------------------------------------------------------------------------+
|                               | ``Opening``: opening operation (:math:`open(src)=dilate(erode(src))`)                                                 |
+-------------------------------+-----------------------------------------------------------------------------------------------------------------------+
|                               | ``Closing``: closing operation (:math:`close(src)=erode(dilate(src))`)                                                |
+-------------------------------+-----------------------------------------------------------------------------------------------------------------------+
|                               | ``Gradient``: morphological gradient (:math:`=dilate(src)-erode(src)`)                                                |
+-------------------------------+-----------------------------------------------------------------------------------------------------------------------+
|                               | ``TopHat``: top hat (:math:`=src-open(src)`)                                                                          |
+-------------------------------+-----------------------------------------------------------------------------------------------------------------------+
|                               | ``BlackHat``: black hat (:math:`=close(src)-src`)                                                                     |
+-------------------------------+-----------------------------------------------------------------------------------------------------------------------+
| ``Size``                      | Size of the structuring element                                                                                       |
+-------------------------------+-----------------------------------------------------------------------------------------------------------------------+
| ``ApplyToLabels`` [0]         | If true, apply the transformation to the labels instead of the image                                                  |
+-------------------------------+-----------------------------------------------------------------------------------------------------------------------+
| ``Shape`` [``Rectangular``]   | Shape of the structuring element used for morphology operations. Can be ``Rectangular``, ``Elliptic`` or ``Cross``.   |
+-------------------------------+-----------------------------------------------------------------------------------------------------------------------+
| ``NbIterations`` [1]          | Number of times erosion and dilation are applied                                                                      |
+-------------------------------+-----------------------------------------------------------------------------------------------------------------------+

NormalizeTransformation
~~~~~~~~~~~~~~~~~~~~~~~

Normalize the image.

+--------------------------+--------------------------------------------------+
| Option [default value]   | Description                                      |
+==========================+==================================================+
| ``Norm`` [``MinMax``]    | Norm type, can be:                               |
+--------------------------+--------------------------------------------------+
|                          | ``L1``: L1 normalization                         |
+--------------------------+--------------------------------------------------+
|                          | ``L2``: L2 normalization                         |
+--------------------------+--------------------------------------------------+
|                          | ``Linf``: Linf normalization                     |
+--------------------------+--------------------------------------------------+
|                          | ``MinMax``: min-max normalization                |
+--------------------------+--------------------------------------------------+
| ``NormValue`` [1.0]      | Norm value (for ``L1``, ``L2`` and ``Linf``)     |
+--------------------------+--------------------------------------------------+
|                          | Such that :math:`||data||_{L_{p}} = NormValue`   |
+--------------------------+--------------------------------------------------+
| ``NormMin`` [0.0]        | Min value (for ``MinMax`` only)                  |
+--------------------------+--------------------------------------------------+
|                          | Such that :math:`min(data) = NormMin`            |
+--------------------------+--------------------------------------------------+
| ``NormMax`` [1.0]        | Max value (for ``MinMax`` only)                  |
+--------------------------+--------------------------------------------------+
|                          | Such that :math:`max(data) = NormMax`            |
+--------------------------+--------------------------------------------------+
| ``PerChannel`` [0]       | If true, normalize each channel individually     |
+--------------------------+--------------------------------------------------+

PadCropTransformation
~~~~~~~~~~~~~~~~~~~~~

Pad/crop the image to a specified size.

+----------------------------------------------+-----------------------------------------------------------------------------------------------------------------+
| Option [default value]                       | Description                                                                                                     |
+==============================================+=================================================================================================================+
| ``Width``                                    | Width of the padded/cropped image                                                                               |
+----------------------------------------------+-----------------------------------------------------------------------------------------------------------------+
| ``Height``                                   | Height of the padded/cropped image                                                                              |
+----------------------------------------------+-----------------------------------------------------------------------------------------------------------------+
| ``BorderType`` [``MinusOneReflectBorder``]   | Border type used when padding. Possible values:                                                                 |
+----------------------------------------------+-----------------------------------------------------------------------------------------------------------------+
|                                              | ``ConstantBorder``: pad with ``BorderValue``                                                                    |
+----------------------------------------------+-----------------------------------------------------------------------------------------------------------------+
|                                              | ``ReplicateBorder``: last element is replicated throughout, like aaaaaa\|abcdefgh\|hhhhhhh                      |
+----------------------------------------------+-----------------------------------------------------------------------------------------------------------------+
|                                              | ``ReflectBorder``: border will be mirror reflection of the border elements, like fedcba\|abcdefgh\|hgfedcb      |
+----------------------------------------------+-----------------------------------------------------------------------------------------------------------------+
|                                              | ``WrapBorder``: it will look like cdefgh\|abcdefgh\|abcdefg                                                     |
+----------------------------------------------+-----------------------------------------------------------------------------------------------------------------+
|                                              | ``MinusOneReflectBorder``: same as ``ReflectBorder`` but with a slight change, like gfedcb\|abcdefgh\|gfedcba   |
+----------------------------------------------+-----------------------------------------------------------------------------------------------------------------+
|                                              | ``MeanBorder``: pad with the mean color of the image                                                            |
+----------------------------------------------+-----------------------------------------------------------------------------------------------------------------+
| ``BorderValue`` [0.0 0.0 0.0]                | Background color used when padding with ``BorderType`` is ``ConstantBorder``                                    |
+----------------------------------------------+-----------------------------------------------------------------------------------------------------------------+

RandomAffineTransformation
~~~~~~~~~~~~~~~~~~~~~~~~~~

Apply a global random affine transformation to the values of the image.

+-------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Option [default value]        | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
+===============================+==================================================================================================================================================================================================================================================================================================================================================================================================================================================================+
| ``GainRange`` [1.0 1.0]       | Random gain (:math:`\alpha`) range (identical for all channels)                                                                                                                                                                                                                                                                                                                                                                                                  |
+-------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``GainRange[*]`` [1.0 1.0]    | Random gain (:math:`\alpha`) range for channel ``*``. Mutually exclusive with ``GainRange``. If any specified, a different random gain will always be sampled for each channel. Default gain is 1.0 (no gain) for missing channels                                                                                                                                                                                                                               |
+-------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|                               | The gain control the *contrast* of the image                                                                                                                                                                                                                                                                                                                                                                                                                     |
+-------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``BiasRange`` [0.0 0.0]       | Random bias (:math:`\beta`) range (identical for all channels)                                                                                                                                                                                                                                                                                                                                                                                                   |
+-------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``BiasRange[*]`` [0.0 0.0]    | Random bias (:math:`\beta`) range for channel ``*``. Mutually exclusive with ``BiasRange``. If any specified, a different random bias will always be sampled for each channel. Default bias is 0.0 (no bias) for missing channels                                                                                                                                                                                                                                |
+-------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|                               | The bias control the *brightness* of the image                                                                                                                                                                                                                                                                                                                                                                                                                   |
+-------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``GammaRange`` [1.0 1.0]      | Random gamma (:math:`\gamma`) range (identical for all channels)                                                                                                                                                                                                                                                                                                                                                                                                 |
+-------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``GammaRange[*]`` [1.0 1.0]   | Random gamma (:math:`\gamma`) range for channel ``*``. Mutually exclusive with ``GammaRange``. If any specified, a different random gamma will always be sampled for each channel. Default gamma is 1.0 (no change) for missing channels                                                                                                                                                                                                                         |
+-------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|                               | The gamma control more or less the *exposure* of the image                                                                                                                                                                                                                                                                                                                                                                                                       |
+-------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``GainVarProb`` [1.0]         | Probability to have a gain variation for each channel. If only one value is specified, the same probability applies to all the channels. In this case, the same gain variation will be sampled for all the channels only if a single range if specified for all the channels using ``GainRange``. If more than one value is specified, a different random gain will always be sampled for each channel, even if the probabilities and ranges are identical       |
+-------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``BiasVarProb`` [1.0]         | Probability to have a bias variation for each channel. If only one value is specified, the same probability applies to all the channels. In this case, the same bias variation will be sampled for all the channels only if a single range if specified for all the channels using ``BiasRange``. If more than one value is specified, a different random bias will always be sampled for each channel, even if the probabilities and ranges are identical       |
+-------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``GammaVarProb`` [1.0]        | Probability to have a gamma variation for each channel. If only one value is specified, the same probability applies to all the channels. In this case, the same gamma variation will be sampled for all the channels only if a single range if specified for all the channels using ``GammaRange``. If more than one value is specified, a different random gamma will always be sampled for each channel, even if the probabilities and ranges are identical   |
+-------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``DisjointGamma`` [0]         | If true, gamma variation and gain/bias variation are mutually exclusive. The probability to have a random gamma variation is therefore ``GammaVarProb`` and the probability to have a gain/bias variation is 1-\ ``GammaVarProb``.                                                                                                                                                                                                                               |
+-------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``ChannelsMask`` []           | If not empty, specifies on which channels the transformation is applied. For example, to apply the transformation only to the first and third channel, set ``ChannelsMask`` to ``1 0 1``                                                                                                                                                                                                                                                                         |
+-------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

The equation of the transformation is:

.. math::

   S =
      \begin{cases}
         \text{numeric\_limits<T>::max()}  &  \text{if } \text{is\_integer<T>} \\
         1.0   & \text{otherwise}
      \end{cases}

.. math:: v(i,j) = \text{cv::saturate\_cast<T>}\left(\alpha \left(\frac{v(i,j)}{S}\right)^{\gamma} S + \beta.S\right)

RangeAffineTransformation
~~~~~~~~~~~~~~~~~~~~~~~~~

Apply an affine transformation to the values of the image.

+---------------------------------+----------------------------------------------------------------------------+
| Option [default value]          | Description                                                                |
+=================================+============================================================================+
| ``FirstOperator``               | First operator, can be ``Plus``, ``Minus``, ``Multiplies``, ``Divides``    |
+---------------------------------+----------------------------------------------------------------------------+
| ``FirstValue``                  | First value                                                                |
+---------------------------------+----------------------------------------------------------------------------+
| ``SecondOperator`` [``Plus``]   | Second operator, can be ``Plus``, ``Minus``, ``Multiplies``, ``Divides``   |
+---------------------------------+----------------------------------------------------------------------------+
| ``SecondValue`` [0.0]           | Second value                                                               |
+---------------------------------+----------------------------------------------------------------------------+

The final operation is the following:

.. math::

   f(x) = \left(x\;\substack{o\\op_{1st}}\;val_{1st}\right)\;
   \substack{o\\op_{2nd}}\;val_{2nd}

RangeClippingTransformation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Clip the value range of the image.

+------------------------------------+----------------------------------------------------------------------------------------------------+
| Option [default value]             | Description                                                                                        |
+====================================+====================================================================================================+
| ``RangeMin`` [:math:`min(data)`]   | Image values below ``RangeMin`` are clipped to 0                                                   |
+------------------------------------+----------------------------------------------------------------------------------------------------+
| ``RangeMax`` [:math:`max(data)`]   | Image values above ``RangeMax`` are clipped to 1 (or the maximum integer value of the data type)   |
+------------------------------------+----------------------------------------------------------------------------------------------------+

RescaleTransformation
~~~~~~~~~~~~~~~~~~~~~

Rescale the image to a specified size.

+---------------------------+--------------------------------------------------------------------------------+
| Option [default value]    | Description                                                                    |
+===========================+================================================================================+
| ``Width``                 | Width of the rescaled image                                                    |
+---------------------------+--------------------------------------------------------------------------------+
| ``Height``                | Height of the rescaled image                                                   |
+---------------------------+--------------------------------------------------------------------------------+
| ``KeepAspectRatio`` [0]   | If true, keeps the aspect ratio of the image                                   |
+---------------------------+--------------------------------------------------------------------------------+
| ``ResizeToFit`` [1]       | If true, resize along the longest dimension when ``KeepAspectRatio`` is true   |
+---------------------------+--------------------------------------------------------------------------------+

ReshapeTransformation
~~~~~~~~~~~~~~~~~~~~~

Reshape the data to a specified size.

+--------------------------+------------------------------------------+
| Option [default value]   | Description                              |
+==========================+==========================================+
| ``NbRows``               | New number of rows                       |
+--------------------------+------------------------------------------+
| ``NbCols`` [0]           | New number of cols (0 = no check)        |
+--------------------------+------------------------------------------+
| ``NbChannels`` [0]       | New number of channels (0 = no change)   |
+--------------------------+------------------------------------------+

SliceExtractionTransformation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Extract a slice from an image.

+----------------------------------------------+-----------------------------------------------------------------------------------------------------------------+
| Option [default value]                       | Description                                                                                                     |
+==============================================+=================================================================================================================+
| ``Width``                                    | Width of the slice to extract                                                                                   |
+----------------------------------------------+-----------------------------------------------------------------------------------------------------------------+
| ``Height``                                   | Height of the slice to extract                                                                                  |
+----------------------------------------------+-----------------------------------------------------------------------------------------------------------------+
| ``OffsetX`` [0]                              | X offset of the slice to extract                                                                                |
+----------------------------------------------+-----------------------------------------------------------------------------------------------------------------+
| ``OffsetY`` [0]                              | Y offset of the slice to extract                                                                                |
+----------------------------------------------+-----------------------------------------------------------------------------------------------------------------+
| ``RandomOffsetX`` [0]                        | If true, the X offset is chosen randomly                                                                        |
+----------------------------------------------+-----------------------------------------------------------------------------------------------------------------+
| ``RandomOffsetY`` [0]                        | If true, the Y offset is chosen randomly                                                                        |
+----------------------------------------------+-----------------------------------------------------------------------------------------------------------------+
| ``RandomRotation`` [0]                       | If true, extract randomly rotated slices                                                                        |
+----------------------------------------------+-----------------------------------------------------------------------------------------------------------------+
| ``RandomRotationRange`` [0.0 360.0]          | Range of the random rotations, in degrees, counterclockwise (if ``RandomRotation`` is enabled)                  |
+----------------------------------------------+-----------------------------------------------------------------------------------------------------------------+
| ``RandomScaling`` [0]                        | If true, extract randomly scaled slices                                                                         |
+----------------------------------------------+-----------------------------------------------------------------------------------------------------------------+
| ``RandomScalingRange`` [0.8 1.2]             | Range of the random scaling (if ``RandomRotation`` is enabled)                                                  |
+----------------------------------------------+-----------------------------------------------------------------------------------------------------------------+
| ``AllowPadding`` [0]                         | If true, zero-padding is allowed if the image is smaller than the slice to extract                              |
+----------------------------------------------+-----------------------------------------------------------------------------------------------------------------+
| ``BorderType`` [``MinusOneReflectBorder``]   | Border type used when padding. Possible values:                                                                 |
+----------------------------------------------+-----------------------------------------------------------------------------------------------------------------+
|                                              | ``ConstantBorder``: pad with ``BorderValue``                                                                    |
+----------------------------------------------+-----------------------------------------------------------------------------------------------------------------+
|                                              | ``ReplicateBorder``: last element is replicated throughout, like aaaaaa\|abcdefgh\|hhhhhhh                      |
+----------------------------------------------+-----------------------------------------------------------------------------------------------------------------+
|                                              | ``ReflectBorder``: border will be mirror reflection of the border elements, like fedcba\|abcdefgh\|hgfedcb      |
+----------------------------------------------+-----------------------------------------------------------------------------------------------------------------+
|                                              | ``WrapBorder``: it will look like cdefgh\|abcdefgh\|abcdefg                                                     |
+----------------------------------------------+-----------------------------------------------------------------------------------------------------------------+
|                                              | ``MinusOneReflectBorder``: same as ``ReflectBorder`` but with a slight change, like gfedcb\|abcdefgh\|gfedcba   |
+----------------------------------------------+-----------------------------------------------------------------------------------------------------------------+
|                                              | ``MeanBorder``: pad with the mean color of the image                                                            |
+----------------------------------------------+-----------------------------------------------------------------------------------------------------------------+
| ``BorderValue`` [0.0 0.0 0.0]                | Background color used when padding with ``BorderType`` is ``ConstantBorder``                                    |
+----------------------------------------------+-----------------------------------------------------------------------------------------------------------------+

ThresholdTransformation
~~~~~~~~~~~~~~~~~~~~~~~

Apply a thresholding transformation to the image. This transformation is
also useful for post-processing.

+--------------------------+------------------------------------------------------------------------------------------------------+
| Option [default value]   | Description                                                                                          |
+==========================+======================================================================================================+
| ``Threshold``            | Threshold value                                                                                      |
+--------------------------+------------------------------------------------------------------------------------------------------+
| ``OtsuMethod`` [0]       | Use Otsu’s method to determine the optimal threshold (if true, the ``Threshold`` value is ignored)   |
+--------------------------+------------------------------------------------------------------------------------------------------+
| ``Operation`` [Binary]   | Thresholding operation to apply. Can be:                                                             |
+--------------------------+------------------------------------------------------------------------------------------------------+
|                          | ``Binary``                                                                                           |
+--------------------------+------------------------------------------------------------------------------------------------------+
|                          | ``BinaryInverted``                                                                                   |
+--------------------------+------------------------------------------------------------------------------------------------------+
|                          | ``Truncate``                                                                                         |
+--------------------------+------------------------------------------------------------------------------------------------------+
|                          | ``ToZero``                                                                                           |
+--------------------------+------------------------------------------------------------------------------------------------------+
|                          | ``ToZeroInverted``                                                                                   |
+--------------------------+------------------------------------------------------------------------------------------------------+
| ``MaxValue`` [1.0]       | Max. value to use with ``Binary`` and ``BinaryInverted`` operations                                  |
+--------------------------+------------------------------------------------------------------------------------------------------+

TrimTransformation
~~~~~~~~~~~~~~~~~~

Trim the image.

+-------------------------------+--------------------------------------------------------------+
| Option [default value]        | Description                                                  |
+===============================+==============================================================+
| ``NbLevels``                  | Number of levels for the color discretization of the image   |
+-------------------------------+--------------------------------------------------------------+
| ``Method`` [``Discretize``]   | Possible values are:                                         |
+-------------------------------+--------------------------------------------------------------+
|                               | ``Reduce``: discretization using K-means                     |
+-------------------------------+--------------------------------------------------------------+
|                               | ``Discretize``: simple discretization                        |
+-------------------------------+--------------------------------------------------------------+

WallisFilterTransformation
~~~~~~~~~~~~~~~~~~~~~~~~~~

Apply Wallis filter to the image.

+--------------------------+------------------------------------------------------------------------------------------------------------------+
| Option [default value]   | Description                                                                                                      |
+==========================+==================================================================================================================+
| ``Size``                 | Size of the filter                                                                                               |
+--------------------------+------------------------------------------------------------------------------------------------------------------+
| ``Mean`` [0.0]           | Target mean value                                                                                                |
+--------------------------+------------------------------------------------------------------------------------------------------------------+
| ``StdDev`` [1.0]         | Target standard deviation                                                                                        |
+--------------------------+------------------------------------------------------------------------------------------------------------------+
| ``PerChannel`` [0]       | If true, apply Wallis filter to each channel individually (this parameter is meaningful only if ``Size`` is 0)   |
+--------------------------+------------------------------------------------------------------------------------------------------------------+
