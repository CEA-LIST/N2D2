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
- ``BatchNorm`` is automatically fused with the preceding ``Conv`` or ``Fc`` when possible;
- ``Padding`` layers are fused with ``Conv`` when possible;
- ``Dropout`` layers are removed.

Export parameters
~~~~~~~~~~~~~~~~~

Extra parameters can be passed during export using the 
``-export-parameters params.ini`` command line argument. The parameters must be 
saved in an INI-like file.

List of available parameters:

+-----------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------+
| Argument [default value]                                        | Description                                                                                                              |
+=================================================================+==========================================================================================================================+
| ``ImplicitCasting`` [0]                                         | If true (1), casting in the graph is implicit and ``Cast`` ONNX operators are not inserted                               |
+-----------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------+
| ``FakeQuantization`` [0]                                        | If true (1), the graph is fake quantized, meaning floating-point ONNX operators are used for the computation             |
+-----------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------+

The parameters ``ImplicitCasting`` and ``FakeQuantization`` are useful only for
quantized networks. In this case, a full integer ONNX graph is generated when 
possible, notably using the ONNX *ConvInteger* and *MatMulInteger* when 
``-nbbits`` is ≤ 8 bits. An example of generated graph is shown below, with a
``Single-shift`` activation rescaling mode (``-act-rescaling-mode``, see 
:ref:`post_quant-label`):

.. figure:: /_static/export_ONNX_quant.svg
   :alt: Example of fully integer, quantized, exported ONNX graph.
   :align: center

By default, strict adherence to the ONNX standard is enforced, by adding 
explicit ``Cast`` operators when required. The automatic insertion of ``Cast``
operators can be disabled by setting the ``ImplicitCasting`` export parameter
to true. This results in the simplified graph below:

.. figure:: /_static/export_ONNX_quant_implicit_cast.svg
   :alt: Example of fully integer, quantized, exported ONNX graph without 
         ``Cast`` operators (with ``ImplicitCasting`` set to 1).
   :align: center

The ``FakeQuantization`` parameter allows to export a quantized network using
fake quantization, meaning the parameters of the network are quantized (integer) 
but their representation remains in floating-point and the computation is done
with floating-point operators. However, the output values of the network 
should be almost identical to when the computation is done in integer. The 
differences are due to numerical errors as all integers cannot be represented
exactly with floating-point.


.. figure:: /_static/export_ONNX_quant_fake.svg
   :alt: Example of fully integer, quantized, exported ONNX graph with fake
         quantization (``FakeQuantization`` set to 1).
   :align: center

.. Note::

    The ``FakeQuantization``, when set, implies ``ImplicitCasting``, as no
    casting operator is required in a fully floating-point graph.


Example
-------

::

    n2d2 MobileNet_ONNX.ini -seed 1 -w weights_validation -export ONNX -nbbits 8 -calib -1 -db-export 100 -test

This command generates a 8-bits integer quantized ONNX model in the sub-directory 
``export_ONNX_int8``.
