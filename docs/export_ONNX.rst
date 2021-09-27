.. _export_onnx-label:

Export: ONNX
============

Export type: ``ONNX``
 ONNX export.

::

    n2d2 MobileNet_ONNX.ini -seed 1 -w /dev/null -export ONNX

Principle
---------

The ONNX export allows you to generate an ONNX model from a N2D2 model. The
generated ONNX model is optimized for inference and can be quantized beforehand
with either post-training quantization or Quantization Aware Training (QAT).

Graph optimizations
~~~~~~~~~~~~~~~~~~~

- Weights are equalized between layers when possible;
- ``BatchNorm`` is automatically fused with the preceding ``Conv`` when possible;
- ``Padding`` layers are fused with ``Conv`` when possible;
- ``Dropout`` layers are removed.


Example
-------

::

    n2d2 MobileNet_ONNX.ini -seed 1 -w /dev/null -export ONNX -nbbits 8 -calib -1 -db-export 100 -test

This command generates a quantized ONNX model in the sub-directory 
``export_ONNX_int8``.
