Containers
==========

Introduction
------------

N2D2 has his own Tensor implementation. 

.. testsetup:: *

   import N2D2
   import numpy
   

.. testcode::

   N2D2.Tensor_float([1, 2, 3])


Tensor can be also be created using numpy.array object.

.. testcode::

   N2D2.CudaTensor_float(numpy.array([[1.0, 2.0], [3.0, 4.0]]))

   
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
