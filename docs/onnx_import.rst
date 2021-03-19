Import ONNX models
==================


.. role:: raw-html(raw)
   :format: html

.. |check|  unicode:: U+02713 .. CHECK MARK
.. |cross|  unicode:: U+02717 .. BALLOT X

.. |ccheck| replace:: :raw-html:`<font color="green">` |check| :raw-html:`</font>`
.. |ccross| replace:: :raw-html:`<font color="red">` |cross| :raw-html:`</font>`



With an INI file
----------------

It is possible to include an ONNX model inside a N2D2 INI file, as part of a
graph. This is particularly useful to add pre-processing and post-processing to 
an existing ONNX model. Below is an example with the MobileNet ONNX model 
provided by Google:

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


A N2D2 target must be associated to the output layer of the ONNX model in order
to compute the score in N2D2.

.. Note::

    The imported ONNX layer names in N2D2 is the name of their first output (
    the operator "name" field is indeed optional in the ONNX standard).
    You can easily find the ONNX cell names after running N2D2 or by opening 
    the ONNX graph in a graph viewer like NETRON 
    (https://lutzroeder.github.io/netron/).


Once the INI file including the ONNX model is ready, the following command must
be used to run N2D2 in test (inference) mode:

::

    n2d2 MobileNet_ONNX.ini -seed 1 -w /dev/null -test

There required command line arguments for running INI files including ONNX model
are described above:

+--------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Command line argument                | Description                                                                                                                                                                                                                                                                                                  |
+======================================+==============================================================================================================================================================================================================================================================================================================+
| ``-seed 1``                          | Initial seed, necessary for test without learning before                                                                                                                                                                                                                                                     |
+--------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``-w /dev/null``                     | No external weight loading: trained weight values are contained in the ONNX model                                                                                                                                                                                                                            |
+--------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+



ONNX INI section type
~~~~~~~~~~~~~~~~~~~~~

The table below summarizes the parameters of an ONNX INI section. To declare an
ONNX section, the ``Type`` parameter must be equal to ``ONNX``. The name of the
section can be arbitrary.

+--------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Option [default value]               | Description                                                                                                                                                                                                                                                                                                  |
+======================================+==============================================================================================================================================================================================================================================================================================================+
| ``Type=ONNX``                        | ONNX section type                                                                                                                                                                                                                                                                                            |
+--------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``File``                             | Path to the ONNX file                                                                                                                                                                                                                                                                                        |
+--------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``Ignore`` []                        | Space-separated list of ONNX operators to ignore during import                                                                                                                                                                                                                                               |
+--------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``IgnoreInputSize`` [0]              | If true (1), the input size specified in the ONNX model is ignored and the N2D2 ``StimuliProvider`` size is used                                                                                                                                                                                             |
+--------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+



With the Python API
-------------------

The ``DeepNetGenerator`` can be used to load ONNX file as well as INI file.

.. code-block:: python

        net = N2D2.Network(1)
        deepNet = N2D2.DeepNetGenerator.generate(net, "mobilenet_v1_1.0_224.onnx")
        deepNet.initialize()



Supported operators
-------------------


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
| Gemm                  | |check|   | Only for fully-connected layers             |
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
| Pad                   | |check|   |                                             |
+-----------------------+-----------+---------------------------------------------+
| Relu                  | |ccheck|  |                                             |
+-----------------------+-----------+---------------------------------------------+
| Reshape               | |check|   | Only for fixed dimensions                   |
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
| Transpose             | |ccheck|  |                                             |
+-----------------------+-----------+---------------------------------------------+
| Upsample              | |ccross|  | Planned                                     |
+-----------------------+-----------+---------------------------------------------+



