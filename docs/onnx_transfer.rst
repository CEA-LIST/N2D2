Train from ONNX models
======================

The ONNX specification does not include any training parameter. To perform a
training on an imported ONNX model, it is possible to add the training elements
(solvers, learning rate scheduler...) on top of an ONNX model in N2D2, in the
INI file directly or using the Python API.

This is particularly useful to perform transfer learning from an existing ONNX
model trained on ImageNet for example.


With an INI file
----------------

We propose in this section to apply transfer learning to a MobileNet v1 ONNX
model. We assume that this model is obtained by converting the reference
pre-trained model from Google using the ``tools/mobilenet_v1_to_onnx.sh`` tool
provided in N2D2. The resulting model file name is therefore assumed to be
``mobilenet_v1_1.0_224.onnx``.

1) Remove the original classifier
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The first step to perform transfer learning is to remove the existing classifier
from the ONNX model. To do so, one can simply use the ``Ignore`` parameter in
the ONNX INI section.

.. code-block:: ini

    [onnx]
    Input=sp
    Type=ONNX
    File=mobilenet_v1_1.0_224.onnx
    ; Remove the last layer and the softmax for transfer learning
    Ignore=Conv__252:0 MobilenetV1/Predictions/Softmax:0


2) Add a new classifier to the ONNX model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The next step is to add a new classifier (fully connected layer with a softmax)
and connect it to the last layer in the ONNX model.

In order to properly connect a N2D2 layer to a layer embedded in an ONNX model,
one must prefix the ONNX layer name with ``onnx,`` in the ``Input`` parameter
of the ``Fc`` cell, as shown below:

.. code-block:: ini

    ; Here, we add our new layers for transfer learning
    [fc]
    ; first input MUST BE "onnx" 
    ; for proper dependency handling
    Input=onnx,MobilenetV1/Logits/AvgPool_1a/AvgPool:0
    Type=Fc
    NbOutputs=100
    ActivationFunction=Linear
    WeightsFiller=XavierFiller
    ConfigSection=common.config

    [softmax]
    Input=fc
    Type=Softmax
    NbOutputs=[fc]NbOutputs
    WithLoss=1
    [softmax.Target]

    ; Common config for static model
    [common.config]
    WeightsSolver.LearningRate=0.01
    WeightsSolver.Momentum=0.9
    WeightsSolver.Decay=0.0005
    Solvers.LearningRatePolicy=StepDecay
    Solvers.LearningRateStepSize=[sp]_EpochSize
    Solvers.LearningRateDecay=0.993

As this new classifier must be trained, all the training parameter must be
specified as usual for this layer.

3) Fine tuning (optional)
~~~~~~~~~~~~~~~~~~~~~~~~~

If one wants to also fine-tune the existing ONNX layers, one must set the 
solver configuration for the ONNX layers, using default configuration sections.

Default configuration sections applies to all the layers of the same type in the
ONNX model. For example, to add default parameters to all convolution layers,
just add a section named ``[onnx:Conv_def]`` in the INI file. The name of the
default section follows the convention ``[onnx:N2D2CellType_def]``.


.. code-block:: ini

    ; Default section for ONNX Conv from section "onnx"
    ; "ConfigSection", solvers and fillers can be specified here...
    [onnx:Conv_def]
    ConfigSection=common.config

    ; Default section for ONNX Fc from section "onnx"
    [onnx:Fc_def]
    ConfigSection=common.config

    ; For BatchNorm, make sure the stats won't change if there is no fine-tuning
    [onnx:BatchNorm_def]
    ConfigSection=bn_notrain.config
    [bn_notrain.config]
    MovingAverageMomentum=0.0


.. Note::

    Important: make sure that the BatchNorm stats does not change if the 
    BatchNorm layer are not fine-tuned! This can be done by setting the 
    parameter ``MovingAverageMomentum`` to 0.0 for the layer than must not be
    fine-tuned.


It is possible to add parameters for a specific ONNX layer by adding a section
with the ONNX layer named prefixed by ``onnx:``.

You can fine-tune the whole network or only some of its layers, usually the last
ones. To stop the fine-tuning at a specific layer, one can simply prevent the
gradient from back-propagating further. This can be achieved with the 
``BackPropagate=0`` configuration parameter.


.. code-block:: ini

    [onnx:Conv__250]
    ConfigSection=common.config,notrain.config
    [notrain.config]
    BackPropagate=0



For the full configuration related to this example and more information, have a
look in ``models/MobileNet_v1_ONNX_transfer.ini``.



With the Python API
-------------------

Coming soon.


