Tensor
------

Introduction :
~~~~~~~~~~~~~~

The n2d2 library propose a tensor implementation with the :py:class:`n2d2.Tensor` class.

.. testcode::

    tensor = n2d2.Tensor([2, 2, 2])

You can store your tensor with CPU or GPU (using CUDA). By default, n2d2 creates a CPU tensor.

.. testcode::

    tensor.cuda() # Converting to a CUDA tensor
    tensor.cpu()  # Converting back the tensor to a regular one.

.. testsetup:: 

   import n2d2


Setting and getting values :
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When setting values, n2d2 tries to perform an auto cast to fit the datatype if there is a discrepancy. 
You can set and get values using :

**Coordinates :**

.. testcode::

    tensor[1,0,1] = 1 # Using coordinates
    print(tensor[1,0,1])

**Index :**

.. testcode::

    tensor[0] = 1
    print(tensor[0])

**Slice :**

(Slice are supported only for assignment.)

.. testcode::

    tensor[1:3] = 1 
    print(tensor[1], tensor[2])

**Set values method :** 

.. testcode::

    tensor.set_values([[[[1, 2],
                        [3, 4]],
                        [[5, 6],
                        [7, 8]]]])



.. autoclass:: n2d2.Tensor
   :members:


