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

.. autoclass:: N2D2.Activation
   :members:

TanhActivation
~~~~~~~~~~~~~~

.. autoclass:: N2D2.TanhActivation
   :members:

LinearActivation
~~~~~~~~~~~~~~~~

.. autoclass:: N2D2.LinearActivation
   :members:

RectifierActivation
~~~~~~~~~~~~~~~~~~~

.. autoclass:: N2D2.RectifierActivation
   :members:


Activation_Frame
----------------

TanhActivation_Frame
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: N2D2.TanhActivation_Frame_float
   :members:
   
.. autoclass:: N2D2.TanhActivation_Frame_double
   :members:

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