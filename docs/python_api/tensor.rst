Tensor
======

Introduction :
--------------
The n2d2 library propose a tensor implementation with the :py:class:`n2d2.Tensor` class.

:py:class:`n2d2.Tensor` is a wrapper of the ``Tensor`` object available in N2D2.





Setting and getting values :
----------------------------

For setting and getting value we will be using the following tensor as an example :

.. testcode::

    tensor = n2d2.Tensor([2, 3])

.. testoutput::

    0 0 0
    0 0 0


You can set and get values using :


Coordinates :
~~~~~~~~~~~~~

.. testcode::

    tensor[1,0] = 1 # Using coordinates
    value = tensor[1,0]

If you print the tensor you will see :

.. testoutput::

    0 0 0
    1 0 0

Index :
~~~~~~~

You can use an index to get or set elements of a tensor. 
The index correspond to the flatten representation of your tensor.

.. testcode::

    tensor[0] = 2
    value = tensor[0]

If you print the tensor you will see :

.. testoutput::

    2 0 0
    0 0 0


Slice :
~~~~~~~

.. note::

    Slice are supported only for assignment !

.. testcode::

    tensor[1:3] = 3 

If you print the tensor you will see :

.. testoutput::

    0 3 3
    0 0 0


Set values method :
~~~~~~~~~~~~~~~~~~~

If you want to set multiple values easily, you can use the method :py:meth:`n2d2.Tensor.set_values` 

.. testcode::

    tensor.set_values([[1,2,3], [4,5,6]])
    
If you print the tensor you will see :

.. testoutput::

    1 2 3
    4 5 6


Fom Numpy :
~~~~~~~~~~~

You can create a tensor using a ``numpy.array`` with the class method : :py:meth:`n2d2.Tensor.from_numpy` 

.. testcode::

    np_array = numpy.array([[1,2,3], [4,5,6]])
    tensor = n2d2.Tensor.from_numpy(np_array)

This will create the following tensor :

.. testoutput::

    1 2 3
    4 5 6

CUDA Tensor 
-----------

You can store your tensor with CPU or GPU (using CUDA). By default, n2d2 creates a CPU tensor.

If you want to create a CUDA Tensor you can do so by setting the parameter ``cuda`` to True in the constructor

.. testcode::

    tensor = n2d2.Tensor([2,3], cuda=True)

You can switch from CPU to GPU at anytime : 

.. testcode::

    tensor.cpu()  # Converting to a CPU tensor
    tensor.cuda() # Converting to a CUDA tensor


Tensor 
------

.. autoclass:: n2d2.Tensor
   :members:


