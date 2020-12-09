Containers
==========

Introduction
------------

N2D2 have it's own Tensor implementation. 

.. testsetup:: *

   import numpy
   import N2D2

.. testcode::

   a = N2D2.Tensor_float([1, 2, 3])
   print(a)

.. testoutput::

   [0]:
   0
   0
   [1]:
   0
   0
   [2]:
   0
   0


Tensor can be created using numpy.array object.

.. doctest::

   >>> print(N2D2.CudaTensor_float(numpy.array([[1.0, 2.0], [3.0, 4.0]])))
   1 2
   3 4
   

Tensor
------

.. autoclass:: N2D2.BaseTensor
   :members:

.. autoclass:: N2D2.Tensor_float
   :members:

.. autoclass:: N2D2.Tensor_double
   :members:

.. autoclass:: N2D2.Tensor_bool
   :members:

CudaTensor
----------

.. autoclass:: N2D2.CudaBaseDeviceTensor
   :members:

.. autoclass:: N2D2.CudaBaseTensor
   :members:

.. autoclass:: N2D2.CudaTensor_float
   :members:

.. autoclass:: N2D2.CudaTensor_double
   :members:
   
.. autoclass:: N2D2.CudaTensor_bool
   :members:
