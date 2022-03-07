Quantization-Aware Training
==================================
.. role:: raw-html(raw)
   :format: html

.. |check|  unicode:: U+02713 .. CHECK MARK
.. |cross|  unicode:: U+02717 .. BALLOT X

.. |ccheck| replace:: :raw-html:`<font color="green">` |check| :raw-html:`</font>`
.. |ccross| replace:: :raw-html:`<font color="red">` |cross| :raw-html:`</font>`

**N2D2-IP only: available upon request.**


Getting Started
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
N2D2 provides a complete design environement for a super wide range of quantization modes. Theses modes are implemented as a set of integrated highly modular blocks. N2D2 implements a per layer quantization scheme that can be different at
each level of the neural network. This high granularity enables to search for the best implementation depending on the
hardware constraints. Moreover to achieve the best performances, N2D2 implements the latest quantization methods currently at the best of the state-of-the-art, summarized in the figure below. Each dot represents one DNN (from the MobileNet or ResNet family), quantized with the number of bits indicated beside.

.. figure:: /_static/qat_sota.png
   :alt: QAT state-of-the-art.

The user can leverage the high modularity of our super set of quantizer blocks and simply choose the  method that best fits with the initial requirements, computation resources and time to market strategy.
For example to implement the ``LSQ`` method, one just need a limited number of training epochs to quantize a model
while implementing the ``SAT`` method requires a higher number of training epochs but gives today the best quantization performance.
In addition, the final objectives can be expressed in terms of different user requirements, depending on the compression capability of the targeted hardware. 
Depending on these different objectives we can consider different quantization schemes:

Weights-Only Quantization
 In this quantization scheme only weights are discretized to fit in a limited set of possible states. Activations
 are not impacted.
 Let's say we want to evaluate the performances of our model with 3 bits weights for convolutions layers. N2D2 natively provides 
 the possibility to add a quantizer module, no need to import a new package or to modify any source code. We then
 just need to specify ``QWeight`` type and ``QWeight.Range`` for step level discretization.

.. code-block:: ini

  ...
  QWeight=SAT ; Quantization Method can be ``LSQ`` or ``SAT``
  QWeight.Range=15 ; Range is set to ``15`` step level, can be represented as a 4-bits word
  ...

Example of fake-quantized weights on 4-bits / 15 levels:

.. figure:: /_static/qat_weights_fakeQ.png
   :alt: Weights Quantization in fake quantization on 15 levels.

Mixed Weights-Activations Quantization
 In this quantization scheme both activations and weights are quantized at different possible step levels. For layers that have a non-linear activation function and learnable parameters, such as ``Fc`` and ``Conv``, we first specify ``QWeight`` in the same way as Weights-Only quantization mode.

 Let's say now that we want to evaluate the performances of our model with activations quantized to 3-bits.
 In a similar manner, as for ``QWeight`` quantizer we specify the activation quantizer ``QAct`` for all layers that have a non-linear activation function. Where the method itself, here ``QAct=SAT`` ensures the non-linearity of the activation function.

.. code-block:: ini

  ...
  ActivationFunction=Linear
  QAct=SAT ; Quantization Method can be ``LSQ`` or ``SAT``
  QAct.Range=7 ; Range is set to ``7`` step level, can be represented as a 3-bits word
  ...

Example of an activation feature map quantized in 4-bits / 15 levels:

.. figure:: /_static/qat_fm_4b.png
   :alt: 4-bits Quantized Activation Feature Map .

Integer-Only Quantization
 Activations and weights are only represented as Integer during the learning phase, it's one step beyond classical fake quantization !! In practice,
 taking advantage of weight-only quantization scheme or fake quantization is clearly not obvious on hardware components. The Integer-Only
 quantization mode is made to fill this void and enable to exploit QAT independently of the targeted hardware architecture. Most
 common programmable architectures like CPU, GPU, DSP can implement it without additional burden. 
 In addition, hardware implementation like HLS or RTL description natively support low-precision integer operators. 
 In this mode, we replace the default quantization mode of the weights as follows :

.. code-block:: ini

  ...
  QWeight.Mode=Integer ; Can be ``Default`` (fake-quantization) mode or ``Integer``(true integer) mode
  ...

Example of full integer weights on 4-bits / 15 levels:

.. figure:: /_static/qat_weights_integer.png
   :alt: Weights Quantization in integer mode on 15 levels.

      
Cell Quantizer Definition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
N2D2 implements a cell quantizer block for discretizing weights and biases at training time. This cell quantizer block
is totally transparent for the user. The quantization phase of the learnable parameters requires intensive operation
to adapt the distribution of the full-precision weights and to adapt the gradient. In addition the implementation
can become highly memory greedy which can be a problem to train a complex model on a single GPU without specific treatment (gradient accumulation, etc..).
That is why N2D2 merged different operations under dedicated CUDA kernels or CPU kernels allowing efficient utilization
of available computation resources.

Overview of the cell quantizer implementation :


.. figure:: /_static/qat_cell_flow.png
   :alt: Cell Quantizer Functional Block.

The common set of parameters for any kind of Cell Quantizer.

+--------------------------------------+-----------------------------------------------------------------------------------------+
| Option [default value]               | Description                                                                             |
+======================================+=========================================================================================+
| ``QWeight``                          | Quantization method can be ``SAT`` or ``LSQ``.                                          |
+--------------------------------------+-----------------------------------------------------------------------------------------+
| ``QWeight.Range`` [``255``]          | Range of Quantization, can be ``1`` for binary, ``255`` for 8-bits etc..                |
+--------------------------------------+-----------------------------------------------------------------------------------------+
| ``QWeight.Solver`` [``SGD``]         | Type of the Solver for learnable quantization parameters, can be ``SGD`` or ``ADAM``    |
+--------------------------------------+-----------------------------------------------------------------------------------------+
| ``QWeight.Mode`` [``Default``]       | Type of quantization Mode, can be ``Default`` or  ``Integer``                           |
+--------------------------------------+-----------------------------------------------------------------------------------------+

LSQ
################################

The Learned Step size Quantization method is tailored to learn the optimal quantization step size parameters in parallel with the network weights.
As described in  :cite:`bhalgat2020lsq`, LSQ tries to estimate and scale the task loss gradient at each weight and activations layer's quantizer step size,
such that it can be learned in conjunction with other network parameters. This method can be initialized using weights from a pre-trained 
full precision model.

+--------------------------------------------+---------------------------------------------------------------+
| Option [default value]                     | Description                                                   |
+============================================+===============================================================+
| ``QWeight.StepSize`` [``100``]             | Initial value of the learnable StepSize parameter             |
+--------------------------------------------+---------------------------------------------------------------+
|``QWeight.StepOptInitStepSize`` [``true``]  | If ``true`` initialize StepSize along first batch variance    |
+--------------------------------------------+---------------------------------------------------------------+

SAT
################################
Scale-Adjusted Training : :cite:`jin2019efficient` method is one of the most promising solutions. The authors proposed SAT as a simple yet effective technique with which the rules of 
efficient training are maintained so that performance can be boosted and low-precision models can even surpass their
full-precision counterparts in some cases. This method exploits DoReFa scheme for the weights quantization.

+--------------------------------------------+-------------------------------------------------------------------------------------------------+
| Option [default value]                     | Description                                                                                     |
+============================================+=================================================================================================+
| ``QWeight.ApplyQuantization`` [``true``]   | Use ``true`` to enable quantization, if ``false`` parameters will be clamped between [-1.0,1.0] |
+--------------------------------------------+-------------------------------------------------------------------------------------------------+
| ``QWeight.ApplyScaling`` [``false``]       | Use ``true`` to scale the parameters as described in the SAT paper                              |
+--------------------------------------------+-------------------------------------------------------------------------------------------------+

Example of clamped weights when ``QWeight.ApplyQuantization=false``:

.. figure:: /_static/qat_weights_Clamp.png
   :alt: Weights Full-Precision clamped.


Activation Quantizer Definition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
N2D2 implements an activation quantizer block to discretize activation at training time. Activation quantizer block
is totally transparent for the user. Quantization phase of the activation requires intensive operation
to learn parameters that will rescale the histogram of full-precision activation at training time. In addition the implementation can become highly memory greedy which can be a problem to train a complex model on a single GPU without specific treatment (gradient accumulation etc..).
That why N2D2 merged different operations under dedicated CUDA kernels or CPU kernels allowing efficient utilization
of available computing resources.

Overview of the activation quantizer implementation:

.. figure:: /_static/qat_act_flow.png
   :alt: Activation Quantizer Functionnal Block.

The common set of parameters for any kind of Activation Quantizer.

+--------------------------------------+-----------------------------------------------------------------------------------------+
| Option [default value]               | Description                                                                             |
+======================================+=========================================================================================+
| ``QAct``                             | Quantization method can be ``SAT`` or ``LSQ``.                                          |
+--------------------------------------+-----------------------------------------------------------------------------------------+
| ``QAct.Range`` [``255``]             | Range of Quantization, can be ``1`` for binary, ``255`` for 8-bits etc..                |
+--------------------------------------+-----------------------------------------------------------------------------------------+
| ``QAct.Solver`` [``SGD``]            | Type of the Solver for learnable quantization parameters, can be ``SGD`` or ``ADAM``    |
+--------------------------------------+-----------------------------------------------------------------------------------------+

LSQ
################################

The Learned Step size Quantization method is tailored to learn the optimum quantization stepsize parameters in parallel to the network's weights.
As described in  :cite:`bhalgat2020lsq`, LSQ tries to estimate and scale the task loss gradient at each weight and activations layer's quantizer step size,
such that it can be learned in conjunction with other network parameters. This method can be initialized using weights from a pre-trained full precision model.

+--------------------------------------------+---------------------------------------------------------------+
| Option [default value]                     | Description                                                   |
+============================================+===============================================================+
| ``QAct.StepSize`` [``100``]                | Initial value of the learnable StepSize parameter             |
+--------------------------------------------+---------------------------------------------------------------+
|``QAct.StepOptInitStepSize`` [``true``]     | If ``true`` initialize StepSize following first batch variance|
+--------------------------------------------+---------------------------------------------------------------+

SAT
################################

Scale-Adjusted Training : :cite:`jin2019efficient` is one of the most promising solutions. The authors proposed SAT as a simple yet effective technique for which the rules of 
efficient training are maintained so that performance can be boosted and low-precision models can even surpass their
full-precision counterparts in some cases. 
This method exploits a CG-PACT scheme for the activations quantization which is a boosted version of PACT for low precision quantization.

+--------------------------------------------+---------------------------------------------------------------+
| Option [default value]                     | Description                                                   |
+============================================+===============================================================+
| ``QAct.Alpha`` [``8.0``]                   | Initial value of the learnable alpha parameter                |
+--------------------------------------------+---------------------------------------------------------------+

Layer compatibility table
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here we describe the compatibility table as a function of the quantization mode. The column ``Cell`` indicates layers that have a full support
to quantize their learnable parameters during the training phase. The column ``Activation`` indicates layers that can support an activation quantizer to their
output feature map. An additional column ``Integer Core`` indicates layers that can be represented without any full-precision
operators at inference time. Of course it is necessary that their input comes from quantized activations.


+---------------+-----------------------------------------------------+
| Layer         | Quantization Mode                                   |
| compatibility +-------------------+-----------------+---------------+
| table         | Cell (parameters) |   Activation    |  Integer Core |
+===============+===================+=================+===============+
|Activation     |                   | |ccheck|        | |ccheck|      |
+---------------+-------------------+-----------------+---------------+
|Anchor         |                   | |ccheck|        | |ccross|      |
+---------------+-------------------+-----------------+---------------+
|BatchNorm*     | |ccheck|          | |ccheck|        | |ccheck|      |
+---------------+-------------------+-----------------+---------------+
|Conv           | |ccheck|          | |ccheck|        | |ccheck|      |
+---------------+-------------------+-----------------+---------------+
|Deconv         | |ccheck|          | |ccheck|        | |ccheck|      |
+---------------+-------------------+-----------------+---------------+
|ElemWise       |                   | |ccheck|        | |ccheck|      |
+---------------+-------------------+-----------------+---------------+
|Fc             | |ccheck|          | |ccheck|        | |ccheck|      |
+---------------+-------------------+-----------------+---------------+
|FMP            |                   | |ccheck|        | |ccross|      |
+---------------+-------------------+-----------------+---------------+
|LRN            | |ccross|          | |ccross|        | |ccross|      |
+---------------+-------------------+-----------------+---------------+
|LSTM           | |ccross|          | |ccross|        | |ccross|      |
+---------------+-------------------+-----------------+---------------+
|ObjectDet      |                   | |ccheck|        | |ccross|      |
+---------------+-------------------+-----------------+---------------+
|Padding        |                   | |ccheck|        | |ccheck|      |
+---------------+-------------------+-----------------+---------------+
|Pool           |                   | |ccheck|        | |ccheck|      |
+---------------+-------------------+-----------------+---------------+
|Proposal       |                   | |ccheck|        | |ccross|      |
+---------------+-------------------+-----------------+---------------+
|Reshape        |                   | |ccheck|        | |ccheck|      |
+---------------+-------------------+-----------------+---------------+
|Resize         |                   | |ccheck|        | |ccheck|      |
+---------------+-------------------+-----------------+---------------+
|ROIPooling     |                   | |ccheck|        | |ccross|      |
+---------------+-------------------+-----------------+---------------+
|RP             |                   | |ccheck|        | |ccross|      |
+---------------+-------------------+-----------------+---------------+
|Scaling        |                   | |ccheck|        | |ccheck|      |
+---------------+-------------------+-----------------+---------------+
|Softmax        |                   | |ccheck|        | |ccross|      |
+---------------+-------------------+-----------------+---------------+
|Threshold      |                   | |ccheck|        | |ccheck|      |
+---------------+-------------------+-----------------+---------------+
|Transformation |                   | |ccheck|        | |ccross|      |
+---------------+-------------------+-----------------+---------------+
|Transpose      |                   | |ccheck|        | |ccheck|      |
+---------------+-------------------+-----------------+---------------+
|Unpool         |                   | |ccheck|        | |ccross|      |
+---------------+-------------------+-----------------+---------------+

*BatchNorm Cell parameters are not directly quantized during the training phase. N2D2 provides a unique approach
to absorb its trained parameters as an integer within the only-integer representation of
the network during a fusion phase. This method is guaranteed without any loss of applicative 
performances.*

Tutorial 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ONNX model : ResNet-18 Example - INI File
#############################################

In this example we show how to quantize the ``resnet-18-v1`` ONNX model with 4-bits weights and 4-bits activations using the ``SAT`` quantization method.
We start from the ``resnet18v1.onnx`` file that you can pick-up at https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet18v1/resnet18v1.onnx .
You can also download it from the  N2D2 script ``N2D2/tools/install_onnx_models.py`` that will automatically install a set of pre-trained
ONNX models under your ``N2D2_MODELS`` system path. 

Moreover you can start from ``.ini`` located at ``N2D2/models/ONNX/resnet-18-v1-onnx.ini`` and directly modify it or you can create an empty 
``resnet18-v1.ini`` file in your simulation folder and to copy/paste all the following ``ini`` inistruction in it. 

Also in this example you will need to know the ONNX cell names of your graph. We recommend you to opening the ONNX graph in a graph viewer 
like NETRON (https://lutzroeder.github.io/netron/).

In this example we focus to demonstrate how to apply ``SAT`` quantization procedure in the ``resnet-18-v1`` ONNX model. The first step of the procedure consists
to learn ``resnet-18-v1`` on ``ImageNet`` database with clamped weights.

First of all we instantiate driver dataset and pre-processing / data augmentation function:

.. code-block:: ini

  DefaultModel=Frame_CUDA
  ;ImageNet dataset
  [database]
  Type=ILSVRC2012_Database
  RandomPartitioning=1
  Learn=1.0
  
  ;Standard image resolution for ImageNet, batchsize=128
  [sp]
  SizeX=224
  SizeY=224
  NbChannels=3
  BatchSize=128
  
  [sp.Transformation-1]
  Type=ColorSpaceTransformation
  ColorSpace=RGB
  
  [sp.Transformation-2]
  Type=RangeAffineTransformation
  FirstOperator=Divides
  FirstValue=255.0 
  
  [sp.Transformation-3]
  Type=RandomResizeCropTransformation
  Width=224
  Height=224
  ScaleMin=0.2
  ScaleMax=1.0
  RatioMin=0.75
  RatioMax=1.33
  ApplyTo=LearnOnly
  
  [sp.Transformation-4]
  Type=RescaleTransformation
  Width=256
  Height=256
  KeepAspectRatio=1
  ResizeToFit=0
  ApplyTo=NoLearn
  
  [sp.Transformation-5]
  Type=PadCropTransformation
  Width=[sp.Transformation-4]Width
  Height=[sp.Transformation-4]Height
  ApplyTo=NoLearn
  
  [sp.Transformation-6]
  Type=SliceExtractionTransformation
  Width=[sp]SizeX
  Height=[sp]SizeY
  OffsetX=16
  OffsetY=16
  ApplyTo=NoLearn
  
  [sp.OnTheFlyTransformation-7]
  Type=FlipTransformation
  ApplyTo=LearnOnly
  RandomHorizontalFlip=1


Now that dataset driver and pre-processing are well defined we can now focus on the neural network configuration.
In our example we decide to quantize all convolutions and fully-connected layers. 
A base block common to all convolution layers can be defined in the *.ini* file. This specific base-block uses ``onnx:Conv_def`` that will
overwrite the native definition of all convolution layers defined in the ONNX file. 
This base block is used to set quantization parameters, like weights bits range, the scaling mode and the quantization mode, and also solver configuration.

.. code-block:: ini

  [onnx:Conv_def]
  QWeight=SAT 
  QWeight.ApplyScaling=0  ; No scaling needed because each conv is followed by batch-normalization layers
  QWeight.ApplyQuantization=0 ; Only clamp mode for the 1st step 
  WeightsFiller=XavierFiller ; Specific filler for SAT method
  WeightsFiller.VarianceNorm=FanOut ; Specific filler for SAT method
  WeightsFiller.Scaling=1.0 ; Specific filler for SAT method
  ConfigSection=conv.config ; Config for conv parameters

  [conv.config]
  NoBias=1 ; No bias needed because each conv is followed by batch-normalization layers
  Solvers.LearningRatePolicy=CosineDecay ; Can be different Policy following your problem, recommended with SAT method
  Solvers.LearningRate=0.05 ; Typical value for batchsize=256 with SAT method
  Solvers.Momentum=0.9 ; Typical value for batchsize=256 with SAT method
  Solvers.Decay=0.00004 ; Typical value for batchsize=256 with SAT method
  Solvers.MaxIterations=192175050; For 150-epoch on ImageNet 1 epoch = 1281167 samples, 150 epoch = 1281167*150 samples
  Solvers.IterationSize=2 ;Our physical batch size is set to 128, iteration size is set to 2 because we want a batchsize of 256

A base block common to all Fully-Connected layers can be defined in the *.ini* file. This specific base-block uses ``onnx:Fc_def`` that will
overwrite the native definition of all fully-connected layers defined in the ONNX file. 
This base block is used to set quantization parameters, like weights bits range, the scaling mode and the quantization mode, and also solver configuration.

.. code-block:: ini

  [onnx:Fc_def]
  QWeight=SAT 
  QWeight.ApplyScaling=1  ; Scaling needed for Full-Connected
  QWeight.ApplyQuantization=0 ; Only clamp mode for the 1st step 
  WeightsFiller=XavierFiller ; Specific filler for SAT method
  WeightsFiller.VarianceNorm=FanOut ; Specific filler for SAT method
  WeightsFiller.Scaling=1.0 ; Specific filler for SAT method
  ConfigSection=fc.config ; Config for conv parameters

  [fc.config]
  NoBias=0 ; Bias needed for fully-connected
  Solvers.LearningRatePolicy=CosineDecay ; Can be different Policy following your problem, recommended with SAT method
  Solvers.LearningRate=0.05 ; Typical value for batchsize=256 with SAT method
  Solvers.Momentum=0.9 ; Typical value for batchsize=256 with SAT method
  Solvers.Decay=0.00004 ; Typical value for batchsize=256 with SAT method
  Solvers.MaxIterations=192175050; For 150-epoch on ImageNet 1 epoch = 1281167 samples, 150 epoch = 1281167*150 samples
  Solvers.IterationSize=2 ;Our physical batch size is set to 128, iteration size is set to 2 because we want a batch size of 256

A base block common to all Batch-Normalization layers can be defined in the *.ini* file. This specific base-block uses ``onnx:Batchnorm_def`` that will
overwrites the native definition of all the batch-normalization defined in the ONNX file. 
We simply defined here hyper-parameters of batch-normalization layers.

.. code-block:: ini

  [onnx:BatchNorm_def]
  ConfigSection=bn_train.config

  [bn_train.config]
  Solvers.LearningRatePolicy=CosineDecay ; Can be different Policy following your problem, recommended with SAT method
  Solvers.LearningRate=0.05 ; Typical value for batchsize=256 with SAT method
  Solvers.Momentum=0.9 ; Typical value for batchsize=256 with SAT method
  Solvers.Decay=0.00004 ; Typical value for batchsize=256 with SAT method
  Solvers.MaxIterations=192175050; For 150-epoch on ImageNet 1 epoch = 1281167 samples, 150 epoch = 1281167*150 samples
  Solvers.IterationSize=2 ;Our physical batchsize is set to 128, iterationsize is set to 2 because we want a batchsize of 256

Then we described the ``resnet-18-v1`` topology directly from the ONNX file that you previously installed in your simulation folder :

.. code-block:: ini

  [onnx]
  Input=sp
  Type=ONNX
  File=resnet18v1.onnx
  ONNX_init=0 ; For SAT method we need to initialize from clamped weights or dedicated filler 

  [soft1]
  Input=resnetv15_dense0_fwd
  Type=Softmax
  NbOutputs=1000
  WithLoss=1

  [soft1.Target]

Now that you set your ``resnet18-v1.ini`` file in your simulation folder you juste have to run the learning phase to clamp the weights
with the command: 

::

./n2d2 resnet18-v1.ini -learn-epoch 150 -valid-metric Precision

This command will run the learning phase over 150 epochs with the ``Imagenet`` dataset. 
The final test accuracy must reach at least 70%.

Next, you have to save parameters of the weights folder to the other location,
for example *weights_clamped* folder.

Congratulations! Your ``resnet-18-v1`` model have clamped weights now ! You can check the results 
in your *weights_clamped* folder.
Now that your ``resnet-18-v1`` model provides clamped weights you can play with it and try different quantization mode.

In addition, if you want to quantized also the ``resnet-18-v1`` activations you need to create a specific base-block in your
``resnet-18-v1.ini`` file in that way :

.. code-block:: ini

  [ReluQ_def]
  ActivationFunction=Linear ; No more need Relu because SAT quantizer integrates it's own non-linear activation
  QAct=SAT ; SAT quantization method
  QAct.Range=15 ; Range=15 for 4-bits quantization model
  QActSolver=SGD ; Specify SGD solver for learned alpha parameter
  QActSolver.LearningRatePolicy=CosineDecay ; Can be different Policy following your problem, recommended with SAT method
  QActSolver.LearningRate=0.05 ; Typical value for batchsize=256 with SAT method
  QActSolver.Momentum=0.9 ; Typical value for batchsize=256 with SAT method
  QActSolver.Decay=0.00004 ; Typical value for batchsize=256 with SAT method
  QActSolver.MaxIterations=192175050; For 150-epoch on ImageNet 1 epoch = 1281167 samples, 150 epoch = 1281167*150 samples
  QActSolver.IterationSize=2 ;Our physical batch size is set to 128, iteration size is set to 2 because we want a batchsize of 256

This base-block will be used to overwrites all the ``rectifier`` activation function of the ONNX model.
To identify the name of the different activation function you can use the netron tool: 

.. figure:: /_static/qat_netron_r.png
   :alt: Relu Name.

We then overrides all the activation function of the model by our previously described activation quantizer:

.. code-block:: ini

  [resnetv15_relu0_fwd]ReluQ_def
  [resnetv15_stage1_relu0_fwd]ReluQ_def
  [resnetv15_stage1_activation0]ReluQ_def
  [resnetv15_stage1_relu1_fwd]ReluQ_def
  [resnetv15_stage1_activation1]ReluQ_def
  [resnetv15_stage2_relu0_fwd]ReluQ_def
  [resnetv15_stage2_activation0]ReluQ_def
  [resnetv15_stage2_relu1_fwd]ReluQ_def
  [resnetv15_stage2_activation1]ReluQ_def
  [resnetv15_stage3_relu0_fwd]ReluQ_def
  [resnetv15_stage3_activation0]ReluQ_def
  [resnetv15_stage3_relu1_fwd]ReluQ_def
  [resnetv15_stage3_activation1]ReluQ_def
  [resnetv15_stage4_relu0_fwd]ReluQ_def
  [resnetv15_stage4_activation0]ReluQ_def
  [resnetv15_stage4_relu1_fwd]ReluQ_def
  [resnetv15_stage4_activation1]ReluQ_def

Now that activations quantization mode is set we focuses on the weights parameters quantization.
For example to quantize weights also in a 4 bits range, you should set the parameters convolution base-block
in that way:

.. code-block:: ini

  [onnx:Conv_def]
  ... 
  QWeight.ApplyQuantization=1 ; Set to 1 for quantization mode
  QWeight.Range=15 ;  Conv is now quantized in 4-bits range (2^4 - 1)
  ...

In a same manner, you can modify the fully-connected base-block in that way :

.. code-block:: ini

  [onnx:Fc_def]
  ... 
  QWeight.ApplyQuantization=1 ; Set to 1 for quantization mode
  QWeight.Range=15 ;  Fc is now quantized in 4-bits range (2^4 - 1)
  ...


As a common practice in quantization aware training the first and last layers are quantized in 8-bits. 
In ResNet-18 the first layer is a convolution layer, we have to specify that to the first layer. 

We first start to identify the name of the first layer under the netron environement: 

.. figure:: /_static/qat_netron_conv_name.png
   :alt: First Conv Cell Name.

We then overrides the range of the first convolution layer of the ``resnet18v1.onnx`` model:

.. code-block:: ini

  [resnetv15_conv0_fwd]onnx:Conv_def
  QWeight.Range=255 ;resnetv15_conv0_fwd is now quantized in 8-bits range (2^8 - 1)


In a same way we overrides the range of the last fully-connected layer in 8-bits :

.. code-block:: ini

  [resnetv15_dense0_fwd]onnx:Fc_def
  QWeight.Range=255 ;resnetv15_dense0_fwd is now quantized in 8-bits range (2^8 - 1)

Now that your modified ``resnet-18-v1.ini`` file is ready just have to run a learning phase with the same hyperparameters by 
using transfer learning method from the previously clamped weights
with this command:

::

./n2d2 resnet-18-v1.ini -learn-epoch 150 -w weights_clamped -valid-metric Precision

This command will run the learning phase over 150 epochs with the ``Imagenet`` dataset. 
The final test accuracy must reach at least 70%.

Congratulations! Your ``resnet-18-v1`` model have now it's weights parameters and activations quantized in a 4-bits way ! 


ONNX model : ResNet-18 Example - Python
#############################################

Coming soon.

Hand-Made model : LeNet Example - INI File
#############################################
One can apply the ``SAT`` quantization methodology on the chosen deep neural network by adding the right parameters to the 
``.ini`` file. Here we show how to configure the ``.ini`` file to correctly apply the SAT quantization.
In this example we decide to apply the SAT quantization procedure in a hand-made LeNet model. The first step of the procedure consists
to learn ``LeNet`` on ``MNIST`` database with clamped weights.

We recommend you to create an empty ``LeNet.ini`` file in your simulation folder and to copy/paste all following ``ini`` block
inside.

First of all we start to described ``MNIST`` driver dataset and pre-processing use for data augmentation at training and test phase:

.. code-block:: ini

  ; Frame_CUDA for GPU and Frame for CPU
  DefaultModel=Frame_CUDA

  ; MNIST Driver Database Instantiation
  [database]
  Type=MNIST_IDX_Database
  RandomPartitioning=1

  ; Environment Description , batch=256
  [env]
  SizeX=32
  SizeY=32
  BatchSize=256

  [env.Transformation_0]
  Type=RescaleTransformation
  Width=32
  Height=32


In our example we decide to quantize all convolutions and fully-connected layers. 
A base block common to all convolution layers can be defined in the *.ini* file. This base block is used to set quantization parameters, like weights bits range, the scaling mode and the quantization mode, and also solver configuration.

.. code-block:: ini

  [Conv_def]
  Type=Conv
  ActivationFunction=Linear
  QWeight=SAT
  QWeight.ApplyScaling=0 ; No scaling needed because each conv is followed by batch-normalization layers
  QWeight.ApplyQuantization=0 ; Only clamp mode for the 1st step
  ConfigSection=common.config

  [common.config]
  NoBias=1
  Solvers.LearningRate=0.05
  Solvers.LearningRatePolicy=None
  Solvers.Momentum=0.0
  Solvers.Decay=0.0


A base block common to all Full-Connected layers can be defined in the *.ini* file. 
This base block is used to set quantization parameters, like weights bits range, the scaling mode and the quantization mode, and also solver configuration.

.. code-block:: ini

  [Fc_def]
  Type=Fc
  ActivationFunction=Linear
  QWeight=SAT
  QWeight.ApplyScaling=1 ; Scaling needed because for Full-Conncted
  QWeight.ApplyQuantization=0 ; Only clamp mode for the 1st step
  ConfigSection=common.config


A base block common to all Batch-Normalization layers can be defined in the *.ini* file. 
This base block is used to set quantization activations, like activations bits range, the quantization mode, and also solver configuration.
In this first step batch-normalization activation are not quantized yet. We simply defined a typical batch-normalization layer with ``Rectifier`` as
non-linear activation function.

.. code-block:: ini

  [Bn_def]
  Type=BatchNorm
  ActivationFunction=Rectifier 
  ConfigSection=bn.config

  [bn.config]
  Solvers.LearningRate=0.05
  Solvers.LearningRatePolicy=None
  Solvers.Momentum=0.0
  Solvers.Decay=0.0

Finally we described the full backbone of ``LeNet`` topology:

.. code-block:: ini

  [conv1] Conv_def
  Input=env
  KernelWidth=5
  KernelHeight=5
  NbOutputs=6
  
  [bn1] Bn_def
  Input=conv1
  NbOutputs=[conv1]NbOutputs
  
  ; Non-overlapping max pooling P2
  [pool1]
  Input=bn1
  Type=Pool
  PoolWidth=2
  PoolHeight=2
  NbOutputs=6
  Stride=2
  Pooling=Max
  Mapping.Size=1
  
  [conv2] Conv_def
  Input=pool1
  KernelWidth=5
  KernelHeight=5
  NbOutputs=16
  [bn2] Bn_def
  Input=conv2
  NbOutputs=[conv2]NbOutputs
  
  [pool2]
  Input=bn2
  Type=Pool
  PoolWidth=2
  PoolHeight=2
  NbOutputs=16
  Stride=2
  Pooling=Max
  Mapping.Size=1
  
  [conv3] Conv_def
  Input=pool2
  KernelWidth=5
  KernelHeight=5
  NbOutputs=120
  
  [bn3]Bn_def
  Input=conv3
  NbOutputs=[conv3]NbOutputs
  
  [conv3.drop]
  Input=bn3
  Type=Dropout
  NbOutputs=[conv3]NbOutputs
  
  [fc1] Fc_def
  Input=conv3.drop
  NbOutputs=84
  
  [fc1.drop]
  Input=fc1
  Type=Dropout
  NbOutputs=[fc1]NbOutputs
  
  [fc2] Fc_def
  Input=fc1.drop
  ActivationFunction=Linear
  NbOutputs=10
  
  [softmax]
  Input=fc2
  Type=Softmax
  NbOutputs=10
  WithLoss=1
  
  [softmax.Target]

Now that you have your ready ``LeNet.ini`` file in your simulation folder you juste have to run the learning phase to clamp the weights
with the command: 

::

./n2d2 LeNet.ini -learn-epoch 100

This command will run the learning phase over 100 epochs with the MNIST dataset. 
The final test accuracy must reach at least 98.9\%:

::

    Final recognition rate: 98.95%    (error rate: 1.05%)
    Sensitivity: 98.94% / Specificity: 99.88% / Precision: 98.94%
    Accuracy: 99.79% / F1-score: 98.94% / Informedness: 98.82%


Next, you have to save parameters of the weights folder to the other location,
for example *weights_clamped* folder.

Congratulations! Your ``LeNet`` model have clamped weights now ! You can check the results 
in your *weights_clamped* folder, for example check your *conv3_weights_quant.distrib.png* file :

.. figure:: /_static/qat_lenet_clamp.png
   :alt: Clamp weights.

Now that your ``LeNet`` model provides clamped weights you can play with it and try different quantization mode.
Moreover, if you want to quantized also the ``LeNet`` activations you have to modify the batch-normalization base-block from your
``LeNet.ini`` file in that way :

.. code-block:: ini

  [Bn_def]
  Type=BatchNorm
  ActivationFunction=Linear ; Replace by linear: SAT quantizer directly apply non-linear activation
  QAct=SAT
  QAct.Alpha=6.0
  QAct.Range=15 ; ->15 for 4-bits range (2^4 - 1)
  QActSolver=SGD
  QActSolver.LearningRate=0.05
  QActSolver.LearningRatePolicy=None
  QActSolver.Momentum=0.0
  QActSolver.Decay=0.0
  ConfigSection=bn.config

For example to quantize weights also in a 4 bits range, these parameters from the convolution base-block
must be modified in that way:

.. code-block:: ini

  [Conv_def]
  Type=Conv
  ActivationFunction=Linear
  QWeight=SAT
  QWeight.ApplyScaling=0
  QWeight.ApplyQuantization=1 ; ApplyQuantization is now set to 1
  QWeight.Range=15 ; Conv is now quantized in 4-bits range (2^4 - 1)
  ConfigSection=common.config

In the same way, you have to modify the fully-connected base-block:

.. code-block:: ini

  [Fc_def]
  Type=Fc
  ActivationFunction=Linear
  QWeight=SAT
  QWeight.ApplyScaling=1 
  QWeight.ApplyQuantization=1 ; ApplyQuantization is now set to 1
  QWeight.Range=15 ; FC is now quantized in 4-bits range (2^4 - 1)
  ConfigSection=common.config

As a common practice, the first and last layer are kept with 8-bits range weights parameters.
To do that, the first *conv1* layer of the ``LeNet`` backbone must be modified in that way:

.. code-block:: ini

  [conv1] Conv_def
  Input=env
  KernelWidth=5
  KernelHeight=5
  NbOutputs=6
  QWeight.Range=255 ; conv1 is now quantized in 8-bits range (2^8 - 1)

And the last layer *fc2* of the ``LeNet`` must be modified in that way:

.. code-block:: ini

  [fc2] Fc_def
  Input=fc1.drop
  ActivationFunction=Linear
  NbOutputs=10
  QWeight.Range=255 ; FC is now quantized in 8-bits range (2^8 - 1)


Now that your modified ``LeNet.ini`` file is ready just have to run a learning phase with the same hyperparameters by 
using transfer learning method from the previously clamped weights
with this command:

::

./n2d2 LeNet.ini -learn-epoch 100 -w weights_clamped


The final test accuracy should be close to 99%:

::

  Final recognition rate: 99.18%    (error rate: 0.82%)
    Sensitivity: 99.173293% / Specificity: 99.90895% / Precision: 99.172422%
    Accuracy: 99.836% / F1-score: 99.172195% / Informedness: 99.082242%


Congratulations! Your ``LeNet`` model is now fully-quantized ! You can check the results 
in your *weights* folder, for example check your *conv3_weights_quant.distrib.png* file :

.. figure:: /_static/qat_lenet_conv_q.png
   :alt: Quantized LeNet weights.

In addition you can have your model graph view that integrates the quantization information. This graph is automatically generated 
at the learning phase or at the test phase. In this example this graph is generated under the name ``LeNet.ini.png``.

As you can see in the following figure, the batch-normalization layers are present (and essential) in your quantized model:

.. figure:: /_static/qat_conv_bn.png
   :alt: batchnorm.

Obviously, no one wants batch-normalization layers in it's quantized model. We answer this problem with our internal tool
named *DeepNetQAT*. This tool allowed us to fused batch normalization parameters within the scaling, clipping and biases parameters
of our quantized models under the ``SAT`` method.

You can fuse the batch normalization parameters of your model with this command :

::

./n2d2 LeNet.ini -test -qat-sat -w weights

Results must be exactly the same than with batch-normalization. Moreover quantizer modules have been entirely removed from your
model !
You can check the results in the newly generated ``LeNet.ini.png`` graph :

.. figure:: /_static/qat_conv_nobn.png
   :alt: no batchnorm.

Moreover you can find your quantized weights and biases under the folder ``weights_quantized``.

Hand-Made model : LeNet Example - Python
#############################################

Coming soon.


Results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Training Time Performances
################################

Quantization-aware training induces intensive operations at training phase. Forward and backward phases
require a lot of additional arithmetic operations compared to the standard floating-point training. The cost of operations
involved in quantization-aware training method directly impacts the training time of a model.

To mitigate this loss at training time, that can be a huge handicap to quantize your own model, N2D2 implements
CUDA kernels to efficiently perform these additional operations.

Here we estimate the training time per epoch for several well-known models on ``ImageNet`` and ``CIFAR-100`` datasets.
These data are shared for information purpose, to give you a realistic idea of the necessary time required to quantize your model. It relies on a lot of parameters like
the dimension of your input data, the size of your dataset, pre-processing, your server/computer set-up installation, etc...

+----------------------------------------------------------+
| ResNet-18   Per Epoch Training Time                      |
+-----------------------+----------------------------------+
| Quantization          | GPU Configuration                |
| Method -              +-------------+--------------------+
| Database              | ``A100`` x1 | ``2080 RTX Ti`` x1 |
+=======================+=============+====================+
|``SAT`` - ``ImageNet`` | 15 min      | 40 min             |
+-----------------------+-------------+--------------------+
|``SAT`` - ``CIFAR100`` | 20 sec      | 1:15 min           |
+-----------------------+-------------+--------------------+
|``LSQ`` - ``ImageNet`` | 15 min      | 55 min             |
+-----------------------+-------------+--------------------+

+----------------------------------------------------------+
| MobileNet-v1   Per Epoch Training Time                   |
+-----------------------+----------------------------------+
| Quantization          | GPU Configuration                |
| Method -              +-------------+--------------------+
| Database              | ``A100`` x1 | ``2080 RTX Ti`` x1 |
+=======================+=============+====================+
|``SAT`` - ``ImageNet`` | 25 min      | 45 min             |
+-----------------------+-------------+--------------------+
|``SAT`` - ``CIFAR100`` | 30 sec      | 1:30 min           |
+-----------------------+-------------+--------------------+

+----------------------------------------------------------+
| MobileNet-v2   Per Epoch Training Time                   |
+-----------------------+----------------------------------+
| Quantization          | GPU Configuration                |
| Method -              +-------------+--------------------+
| Database              | ``A100`` x1 | ``2080 RTX Ti`` x1 |
+=======================+=============+====================+
|``SAT`` - ``ImageNet`` | 30 min      | 62 min             |
+-----------------------+-------------+--------------------+
|``SAT`` - ``CIFAR100`` | 1:15 min    | 2:10 min           |
+-----------------------+-------------+--------------------+
|``LSQ`` - ``ImageNet`` | 33 min      | xx min             |
+-----------------------+-------------+--------------------+

+----------------------------------------------------------+
| Inception-v1   Per Epoch Training Time                   |
+-----------------------+----------------------------------+
| Quantization          | GPU Configuration                |
| Method -              +-------------+--------------------+
| Database              | ``A100`` x1 | ``2080 RTX Ti`` x1 |
+=======================+=============+====================+
|``SAT`` - ``ImageNet`` | 40 min      | 80 min             |
+-----------------------+-------------+--------------------+
|``SAT`` - ``CIFAR100`` | 35 sec      | 2:20 min           |
+-----------------------+-------------+--------------------+
|``LSQ`` - ``ImageNet`` | 25 min      | xx min             |
+-----------------------+-------------+--------------------+

These performances indicators have been realized with typical ``Float32`` datatype. Even if most of the operations used in the 
quantizations methods provides support for ``Float16`` (half-precision) datatypes we recommend to not use it. In our experiments we
observes performances differences compared to the ``Float32`` datatype mode. These differences comes from gradient instability when
datatype is reduced to ``Float16``.


MobileNet-v1
##############

Results obtained with the ``SAT`` method (~150 epochs) under the integer only mode :

+-------------------------------------------------------------------------+
| MobileNet-v1 - ``SAT`` ImageNet Performances - Integer ONLY             |
+-------------+---------------------------+-------------+--------+--------+
| Top-1       | Quantization Range (bits) | Parameters  | Memory | Alpha  |
| Precision   +---------+-----------------+             |        |        |
|             | Weights | Activations     |             |        |        |
+=============+=========+=================+=============+========+========+
| ``72.60 %`` | 8       | 8               | 4 209 088   | 4.2 MB | 1.0    |
+-------------+---------+-----------------+-------------+--------+--------+
| ``71.50 %`` | 4       | 8               | 4 209 088   | 2.6 MB | 1.0    |
+-------------+---------+-----------------+-------------+--------+--------+
| ``65.00 %`` | 2       | 8               | 4 209 088   | 1.8 MB | 1.0    |
+-------------+---------+-----------------+-------------+--------+--------+
| ``60.15 %`` | 1       | 8               | 4 209 088   | 1.4 MB | 1.0    |
+-------------+---------+-----------------+-------------+--------+--------+
| ``70.90 %`` | 4       | 4               | 4 209 088   | 2.6 MB | 1.0    |
+-------------+---------+-----------------+-------------+--------+--------+
| ``64.60 %`` | 3       | 3               | 4 209 088   | 2.2 MB | 1.0    |
+-------------+---------+-----------------+-------------+--------+--------+
| ``57.00 %`` | 2       | 2               | 4 209 088   | 1.8 MB | 1.0    |
+-------------+---------+-----------------+-------------+--------+--------+
| ``69.00 %`` | 8       | 8               | 3 156 816   | 2.6 MB | 0.75   |
+-------------+---------+-----------------+-------------+--------+--------+
| ``69.00 %`` | 4       | 8               | 3 156 816   | 1.6 MB | 0.75   |
+-------------+---------+-----------------+-------------+--------+--------+
| ``65.60 %`` | 3       | 8               | 3 156 816   | 1.4 MB | 0.75   |
+-------------+---------+-----------------+-------------+--------+--------+
| ``58.70 %`` | 2       | 8               | 3 156 816   | 1.2 MB | 0.75   |
+-------------+---------+-----------------+-------------+--------+--------+
| ``53.80 %`` | 1       | 8               | 3 156 816   | 0.9 MB | 0.75   |
+-------------+---------+-----------------+-------------+--------+--------+
| ``64.70 %`` | 8       | 8               | 1 319 648   | 1.3 MB | 0.5    |
+-------------+---------+-----------------+-------------+--------+--------+
| ``63.40 %`` | 4       | 8               | 1 319 648   | 0.9 MB | 0.5    |
+-------------+---------+-----------------+-------------+--------+--------+
| ``51.70 %`` | 2       | 8               | 1 319 648   | 0.7 MB | 0.5    |
+-------------+---------+-----------------+-------------+--------+--------+
| ``44.00 %`` | 1       | 8               | 1 319 648   | 0.6 MB | 0.5    |
+-------------+---------+-----------------+-------------+--------+--------+
| ``63.70 %`` | 4       | 4               | 1 319 648   | 0.9 MB | 0.5    |
+-------------+---------+-----------------+-------------+--------+--------+
| ``54.80 %`` | 3       | 3               | 1 319 648   | 0.8 MB | 0.5    |
+-------------+---------+-----------------+-------------+--------+--------+
| ``42.80 %`` | 2       | 2               | 1 319 648   | 0.7 MB | 0.5    |
+-------------+---------+-----------------+-------------+--------+--------+
| ``55.01 %`` | 8       | 8               |   463 600   | 0.4 MB | 0.25   |
+-------------+---------+-----------------+-------------+--------+--------+
| ``50.02 %`` | 4       | 8               |   463 600   | 0.3 MB | 0.25   |
+-------------+---------+-----------------+-------------+--------+--------+
| ``46.80 %`` | 3       | 8               |   463 600   | 0.3 MB | 0.25   |
+-------------+---------+-----------------+-------------+--------+--------+
| ``48.80 %`` | 4       | 4               |   463 600   | 0.3 MB | 0.25   |
+-------------+---------+-----------------+-------------+--------+--------+





MobileNet-v2
##############

Results obtained with the ``SAT`` method (~150 epochs) under the integer only mode :

+-------------------------------------------------------------------------+
| MobileNet-v2 - ``SAT`` ImageNet Performances - Integer ONLY             |
+-------------+---------------------------+-------------+--------+--------+
| Top-1       | Quantization Range (bits) | Parameters  | Memory | Alpha  |
| Precision   +---------+-----------------+             |        |        |
|             | Weights | Activations     |             |        |        |
+=============+=========+=================+=============+========+========+
| ``72.5 %``  | 8       | 8               | 3 214 048   | 3.2 MB | 1.0    |
+-------------+---------+-----------------+-------------+--------+--------+
| ``58.59 %`` | 1       | 8               | 3 214 048   | 1.3 MB | 1.0    |
+-------------+---------+-----------------+-------------+--------+--------+
| ``70.93 %`` | 4       | 4               | 3 214 048   | 2.1 MB | 1.0    |
+-------------+---------+-----------------+-------------+--------+--------+


Results obtained with the ``LSQ`` method on 1 epoch :

+-------------------------------------------------------------------------+
| MobileNet-v2 - ``LSQ`` ImageNet Performances - 1-Epoch                  |
+-------------+---------------------------+-------------+--------+--------+
| Top-1       | Quantization Range (bits) | Parameters  | Memory | Alpha  |
| Precision   +---------+-----------------+             |        |        |
|             | Weights | Activations     |             |        |        |
+=============+=========+=================+=============+========+========+
| ``70.1 %``  | 8       | 8               | 3 214 048   | 3.2 MB | 1.0    |
+-------------+---------+-----------------+-------------+--------+--------+



ResNet
##############

Results obtained with the ``SAT`` method (~150 epochs) under the integer only mode :

+-------------------------------------------------------------------------+
| ResNet - ``SAT`` ImageNet Performances - Integer ONLY                   |
+-------------+---------------------------+-------------+--------+--------+
| Top-1       | Quantization Range (bits) | Parameters  | Memory | Depth  |
| Precision   +---------+-----------------+             |        |        |
|             | Weights | Activations     |             |        |        |
+=============+=========+=================+=============+========+========+
| ``70.80 %`` | 8       | 8               | 11 506 880  | 11.5 MB| 18     |
+-------------+---------+-----------------+-------------+--------+--------+
| ``67.6 %``  | 1       | 8               | 11 506 880  | 1.9 MB | 18     |
+-------------+---------+-----------------+-------------+--------+--------+
| ``70.00 %`` | 4       | 4               | 11 506 880  | 6.0 MB | 18     |
+-------------+---------+-----------------+-------------+--------+--------+


Results obtained with the ``LSQ`` method on 1 epoch :

+-------------------------------------------------------------------------+
| ResNet - ``LSQ`` ImageNet Performances - 1-Epoch                        |
+-------------+---------------------------+-------------+--------+--------+
| Top-1       | Quantization Range (bits) | Parameters  | Memory | Depth  |
| Precision   +---------+-----------------+             |        |        |
|             | Weights | Activations     |             |        |        |
+=============+=========+=================+=============+========+========+
| ``70.20 %`` | 8       | 8               | 11 506 880  | 11.5 MB| 18     |
+-------------+---------+-----------------+-------------+--------+--------+



Inception-v1
##############

Results obtained with the ``SAT`` method (~150 epochs) under the integer only mode :

+----------------------------------------------------------------+
| Inception-v1 - ``SAT`` ImageNet Performances - Integer ONLY    |
+-------------+---------------------------+-------------+--------+
| Top-1       | Quantization Range (bits) | Parameters  | Memory |
| Precision   +---------+-----------------+             |        |
|             | Weights | Activations     |             |        |
+=============+=========+=================+=============+========+
| ``73.60 %`` | 8       | 8               | 6 600 006   | 6.6 MB |
+-------------+---------+-----------------+-------------+--------+
| ``68.60 %`` | 1       | 8               | 6 600 006   | 1.7 MB |
+-------------+---------+-----------------+-------------+--------+
| ``72.30 %`` | 4       | 4               | 6 600 006   | 3.8 MB |
+-------------+---------+-----------------+-------------+--------+
| ``68.50 %`` | 1       | 4               | 6 600 006   | 1.7 MB |
+-------------+---------+-----------------+-------------+--------+
| ``67.50 %`` | 1       | 3               | 6 600 006   | 1.7 MB |
+-------------+---------+-----------------+-------------+--------+
| ``63.30 %`` | 1       | 2               | 6 600 006   | 1.7 MB |
+-------------+---------+-----------------+-------------+--------+
| ``47.36 %`` | 1       | 1               | 6 600 006   | 1.7 MB |
+-------------+---------+-----------------+-------------+--------+

Results obtained with the ``LSQ`` method on 1 epoch :

+----------------------------------------------------------------+
| Inception-v1 - ``LSQ`` ImageNet Performances - 1-Epoch         |
+-------------+---------------------------+-------------+--------+
| Top-1       | Quantization Range (bits) | Parameters  | Memory |
| Precision   +---------+-----------------+             |        |
|             | Weights | Activations     |             |        |
+=============+=========+=================+=============+========+
| ``72.60 %`` | 8       | 8               | 6 600 006   | 6.6 MB |
+-------------+---------+-----------------+-------------+--------+
