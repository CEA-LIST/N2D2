Cells
=====
 
Introduction
------------

Cell objects are the atomics elements that compose a deep neural network.


Cell
----

If you wan to add the same parameters to multiple cells, you can use a ``ConfigSection`` 

.. testcode::

        from n2d2.cell import Conv, Pool 

        conv_config = n2d2.ConfigSection(activation_function=Rectifier(), 
                                weights_filler=He(), 
                                weights_solver=SGD(**solver_config), 
                                no_bias=True)
        model = n2d2.deepnet.Sequence([
                Conv(3, 32, [4, 4], **conv_config),
                Pool([2, 2], stride_dims=[2, 2], pooling='Max'),
                Conv(20, 48, [5, 5], **conv_config),
                Pool([3, 3], stride_dims=[2, 2], pooling='Max'),
        ])

If you have compiled N2D2 with *CUDA*, you can also accelerate your network by using CUDA model. For this you just have to add this line at the beginning of the script :

..testcode::

        n2d2.global_variables.default_model = "Frame_CUDA"


.. autoclass:: n2d2.cell.Cell
        :members:
        :inherited-members:

Conv
~~~~

.. autoclass:: n2d2.cell.Conv
        :members:
        :inherited-members:

Deconv
~~~~~~

.. autoclass:: n2d2.cell.Deconv
        :members:
        :inherited-members:

Fc
~~

.. autoclass:: n2d2.cell.Fc
        :members:
        :inherited-members:

Dropout
~~~~~~~

.. autoclass:: n2d2.cell.Dropout
        :members:
        :inherited-members:

ElemWise
~~~~~~~~

.. autoclass:: n2d2.cell.ElemWise
        :members:
        :inherited-members:

Padding
~~~~~~~

.. autoclass:: n2d2.cell.Padding
        :members:
        :inherited-members:
        
Softmax
~~~~~~~

.. autoclass:: n2d2.cell.Softmax
        :members:
        :inherited-members:

BatchNorm2d
~~~~~~~~~~~

.. autoclass:: n2d2.cell.BatchNorm2d
        :members:
        :inherited-members:

LRN
~~~

.. autoclass:: n2d2.cell.LRN
        :members:
        :inherited-members:

Pool
~~~~

.. autoclass:: n2d2.cell.Pool
        :members:
        :inherited-members:


Solver
------

You can associate to some cell a specific weight solver.

..autoclass:: n2d2.solver.Solver
        :members:
        :inherited-members:

SGD
~~~

.. autoclass:: n2d2.solver.SGD
        :members:
        :inherited-members:

Filler
------

You can associate to some cell a specific weight filler.


..autoclass:: n2d2.filler.Filler
        :members:
        :inherited-members:


He
~~

.. autoclass:: n2d2.filler.He
        :members:
        :inherited-members:

Normal
~~~~~~

.. autoclass:: n2d2.filler.Normal
        :members:
        :inherited-members:

Constant
~~~~~~~~

.. autoclass:: n2d2.filler.Constant
        :members:
        :inherited-members:


Activations
-----------
 
 You can associate to some cell an activation function.

..autoclass:: n2d2.activation.ActivationFunction
        :members:
        :inherited-members:

Linear
~~~~~~

.. autoclass:: n2d2.activation.Linear
        :members:
        :inherited-members:

Rectifier
~~~~~~~~~

.. autoclass:: n2d2.activation.Rectifier
        :members:
        :inherited-members:
        
Tanh
~~~~

.. autoclass:: n2d2.activation.Tanh
        :members:
        :inherited-members:
        