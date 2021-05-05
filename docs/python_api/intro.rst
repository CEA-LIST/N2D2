Introduction
============

For notation purposes, we will refer to the python library of N2D2 as n2d2. 
This library uses the core function of N2D2 and add an extra layer of abstraction to make the experience more user friendly. 
With the library you can import data, pre-process them, create a deep neural network model, train it and realize inference with it.
You can also import a network using the :doc:`ini file configuration<../ini/intro>` or the ONNX library.


Default values
--------------

The python API used default values that you can modify at any time in your scripts.
You can modify the following values :

Default Model
~~~~~~~~~~~~~

The default model used is ``Frame`` which correspond to a pure CPU based computation.
If you have compiled N2D2 with **CUDA**, you can accelerate your neural networks by using a ``Frame_CUDA`` model. 
To do so, use the following line :

.. testcode::

        n2d2.global_variables.default_model = "Frame_CUDA"

Default Data Type
~~~~~~~~~~~~~~~~~

The default data type used is ``float``, but you can change it at any time. 

The available default data type available are :

- ``int``
- ``float``

To change the default data type use the following line :

.. testcode::

        n2d2.global_variables.default_model = "int"

.. note::

    The :py:class:`n2d2.Tensor` class is not affected by this variable.