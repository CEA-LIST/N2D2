Obtain ONNX models
==================

Convert from PyTorch
--------------------
ONNX conversion is natively supported in PyTorch with the ``torch.onnx.export``
function. An example of a pre-trained PyTorch model conversion to ONNX is 
provided in ``tools/pytorch_to_onnx.py``:

.. code-block:: python

    import torch
    from MobileNetV2 import mobilenet_v2

    dummy_input = torch.randn(10, 3, 224, 224)
    model = mobilenet_v2(pretrained=True)

    input_names = [ "input" ]
    output_names = [ "output" ]

    torch.onnx.export(model, dummy_input, "mobilenet_v2_pytorch.onnx", verbose=True, input_names=input_names, output_names=output_names)


Convert from TF/Keras
------------------
ONNX conversion is not natively supported by TF/Keras. Instead, a third-party
tool must be used, like ``keras2onnx`` or ``tf2onnx``. Currently, the ``tf2onnx``
is the most active and most maintained solution.

The ``tf2onnx`` tool can be used in command line, by providing a TensorFlow
frozen graph (.pb).

.. Note::

    Make sure to use the option ``--inputs-as-nchw`` on the model input(s)
    because N2D2 expects NCHW inputs, but the default format in TF/Keras is
    NHWC.

    The format of the exported ONNX graph from TF/Keras will depend on the
    execution platform (CPU or GPU). The default format is NHWC on CPU and
    NCHW on GPU. ONNX mandates the NCHW format for the operators, so exporting
    an ONNX model on CPU can result in the insertion of many ``Transpose`` 
    operations in the graph before and after other operators.


.. code-block:: bash

    tfmodel=mobilenet_v1_1.0_224_frozen.pb
    onnxmodel=mobilenet_v1_1.0_224.onnx
    url=http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224.tgz
    tgz=$(basename $url)

    if [ ! -r $tgz ]; then
        wget  -q  $url
        tar zxvf $tgz
    fi
    python3 -m tf2onnx.convert --input $tfmodel --output $onnxmodel \
        --opset 10 --verbose \
        --inputs-as-nchw input:0 \
        --inputs input:0 \
        --outputs MobilenetV1/Predictions/Reshape_1:0

Example conversion scripts are provided for the Mobilenet families:
``tools/mobilenet_v1_to_onnx.sh``, ``tools/mobilenet_v2_to_onnx.sh`` and
``tools/mobilenet_v3_to_onnx.sh``.


Download pre-trained models
---------------------------

Many already trained ONNX models are freely available and ready to use in the
ONNX Model Zoo: https://github.com/onnx/models/blob/master/README.md

