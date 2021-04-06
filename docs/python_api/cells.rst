Cells
=====
 
Introduction
------------

Cell objects are the atomics elements that compose a deep neural network.

Solver
------

You can associate to some cell a specific weight solver.


.. autoclass:: n2d2.solver.SGD
        :members:
        :inherited-members:

Filler
------

You can associate to some cell a specific weight filler.

.. autoclass:: n2d2.filler.He
        :members:
        :inherited-members:

.. autoclass:: n2d2.filler.Normal
        :members:
        :inherited-members:

.. autoclass:: n2d2.filler.Constant
        :members:
        :inherited-members:



Activations
-----------
 
 You can associate to some cell an activation function.

.. autoclass:: n2d2.activation.Linear
        :members:
        :inherited-members:
        
.. autoclass:: n2d2.activation.Rectifier
        :members:
        :inherited-members:
        
.. autoclass:: n2d2.activation.Tanh
        :members:
        :inherited-members:
        

Cell
----

.. autoclass:: n2d2.cell.Cell
        :members:
        :inherited-members:

.. autoclass:: n2d2.cell.Conv
        :members:
        :inherited-members:
        
.. autoclass:: n2d2.cell.Fc
        :members:
        :inherited-members:
        
.. autoclass:: n2d2.cell.Dropout
        :members:
        :inherited-members:
        
.. autoclass:: n2d2.cell.Padding
        :members:
        :inherited-members:
        
.. autoclass:: n2d2.cell.Softmax
        :members:
        :inherited-members:

.. autoclass:: n2d2.cell.BatchNorm2d
        :members:
        :inherited-members:
        
.. autoclass:: n2d2.cell.LRN
        :members:
        :inherited-members:
        
.. autoclass:: n2d2.cell.Pool
        :members:
        :inherited-members: