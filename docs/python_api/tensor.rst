Tensor
------
The n2d2 library comes with a wrapper for the :py:class:`n2d2.Tensor` class.

.. testsetup:: 

   import n2d2

You can create a tensor using the following code :

.. testcode::

    tensor = n2d2.Tensor([2, 2, 2], defaultDataType=int)


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

(Slice are supported only for assignement.)

.. testcode::

    tensor[1:3] = 1 
    print(tensor[1], tensor[2])


.. autoclass:: n2d2.Tensor
   :members:


