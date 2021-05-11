Activation
==========

Introduction
------------

Activation functions in N2D2 are passed as arguments to initialize :py:class:`N2D2.Cell`.

.. testsetup:: *

   import N2D2

.. testcode::

   tanh = N2D2.TanhActivation_Frame_float()

Activation
----------

Activation
~~~~~~~~~~

.. autoclass:: N2D2.Activation
        :members:

LinearActivation
~~~~~~~~~~~~~~~~

.. autoclass:: N2D2.LinearActivation
        :members:

RectifierActivation
~~~~~~~~~~~~~~~~~~~

.. autoclass:: N2D2.RectifierActivation
        :members:

TanhActivation
~~~~~~~~~~~~~~

.. autoclass:: N2D2.TanhActivation
        :members:



SwishActivation
~~~~~~~~~~~~~~~

.. autoclass:: N2D2.SwishActivation
        :members:

SaturationActivation
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: N2D2.SaturationActivation
        :members:

LogisticActivation
~~~~~~~~~~~~~~~~~~

.. autoclass:: N2D2.LogisticActivation
        :members:

SoftplusActivation
~~~~~~~~~~~~~~~~~~

.. autoclass:: N2D2.SoftplusActivation
        :members:

Activation_Frame
----------------

LinearActivation_Frame
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: N2D2.LinearActivation_Frame_float
        :members:
.. autoclass:: N2D2.LinearActivation_Frame_double
        :members:
.. autoclass:: N2D2.LinearActivation_Frame_CUDA_float
        :members:
.. autoclass:: N2D2.LinearActivation_Frame_CUDA_double
        :members:

RectifierActivation_Frame
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: N2D2.RectifierActivation_Frame_float
        :members:
.. autoclass:: N2D2.RectifierActivation_Frame_double
        :members:
.. autoclass:: N2D2.RectifierActivation_Frame_CUDA_float
        :members:
.. autoclass:: N2D2.RectifierActivation_Frame_CUDA_double
        :members:

TanhActivation_Frame
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: N2D2.TanhActivation_Frame_float
        :members:
.. autoclass:: N2D2.TanhActivation_Frame_double
        :members:
.. autoclass:: N2D2.TanhActivation_Frame_CUDA_float
        :members:
.. autoclass:: N2D2.TanhActivation_Frame_CUDA_double
        :members:

SwishActivation_Frame
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: N2D2.SwishActivation_Frame_float
        :members:
.. autoclass:: N2D2.SwishActivation_Frame_double
        :members:
.. autoclass:: N2D2.SwishActivation_Frame_CUDA_float
        :members:
.. autoclass:: N2D2.SwishActivation_Frame_CUDA_double
        :members:

.. testcode::
   :hide:

   N2D2.LinearActivation_Frame_CUDA_float()
   N2D2.LinearActivation_Frame_float()
   N2D2.RectifierActivation_Frame_CUDA_float()
   N2D2.RectifierActivation_Frame_float()
   N2D2.TanhActivation_Frame_CUDA_float()
   N2D2.TanhActivation_Frame_float()
   N2D2.SwishActivation_Frame_float()
   N2D2.SwishActivation_Frame_CUDA_float()
   N2D2.SoftplusActivation_Frame_float()
   N2D2.SoftplusActivation_Frame_CUDA_float()
   N2D2.SaturationActivation_Frame_float()
   N2D2.SaturationActivation_Frame_CUDA_float()
   N2D2.LogisticActivation_Frame_float()
   N2D2.LogisticActivation_Frame_CUDA_float()
