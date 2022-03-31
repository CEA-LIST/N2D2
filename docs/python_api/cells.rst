Cells
=====
 
Introduction
------------

Cell objects are the atomics elements that compose a deep neural network.

They are the node of the computation graph. :py:class:`n2d2.cells.NeuralNetworkCell` are not dependant of a DeepNet this allow a dynamic management of the computation. 

Cells are organize with the following logic : 
 - :py:class:`n2d2.cells.NeuralNetworkCell` : Atomic cell of a neural network;
 - :py:class:`n2d2.cells.Block` : Store a collection of :py:class:`n2d2.cells.NeuralNetworkCell`, the storage order does **not** determine the graph computation;
 - :py:class:`n2d2.cells.DeepNetCell` : This cell allow you to use an :py:class:`N2D2.DeepNet`, it can be used for `ONNX` and `INI` import or to run optimize learning;
 - :py:class:`n2d2.cells.Iterable` : Similar to :py:class:`n2d2.cells.Block` but the order of storage determine the computation graph;
 - :py:class:`n2d2.cells.Sequence` : A vertical structure to create neural network;
 - :py:class:`n2d2.cells.Layer` : An horizontal structure to create neural network.

.. figure:: ../_static/n2d2_cell_diagram.png
   :alt: Cell class diagram

Sequence
~~~~~~~~

.. autoclass:: n2d2.cells.Sequence
        :members:
        :inherited-members:

Layer
~~~~~

.. autoclass:: n2d2.cells.Layer
        :members:
        :inherited-members:

DeepNetCell
~~~~~~~~~~~

The :py:class:`n2d2.cells.DeepNetCell` constructor require an :py:class:`N2D2.DeepNet`. In practice, you will not use the constructor directly.

There are three methods to generate a :py:class:`n2d2.cells.DeepNetCell` : :py:meth:`n2d2.cells.DeepNetCell.load_from_ONNX`, :py:meth:`n2d2.cells.DeepNetCell.load_from_INI`, :py:meth:`n2d2.cells.Sequence.to_deepnet_cell` 

The DeepNetCell can be used to train the neural network in an efficient way thanks to :py:meth:`n2d2.cells.DeepNetCell.fit`.


.. autoclass:: n2d2.cells.DeepNetCell
        :members:
        :inherited-members:

Example :
^^^^^^^^^

You can create a DeepNet cell with :py:meth:`n2d2.cells.DeepNetCell.load_from_ONNX` :

.. code-block::

        database = n2d2.database.MNIST(data_path=DATA_PATH, validation=0.1)
        provider = n2d2.provider.DataProvider(database, [28, 28, 1], batch_size=BATCH_SIZE)
        model = n2d2.cells.DeepNetCell.load_from_ONNX(provider, ONNX_PATH)
        model.fit(nb_epochs)
        model.run_test()

Using :py:meth:`n2d2.cells.DeepNetCell.fit` method will reduce the learning time as it will parallelize the loading of the batch of data and the propagation. 

If you want to use the dynamic computation graph  provided by the API, you can use the :py:class:`n2d2.cells.DeepNetCell` as a simple cell.


.. code-block::

        database = n2d2.database.MNIST(data_path=DATA_PATH, validation=0.1)
        provider = n2d2.provider.DataProvider(database, [28, 28, 1], batch_size=BATCH_SIZE)
        model = n2d2.cells.DeepNetCell.load_from_ONNX(provider, ONNX_PATH)
        sequence = n2d2.cells.Sequence([model, n2d2.cells.Softmax(with_loss=True)])
        input_tensor = n2d2.Tensor(DIMS)
        output_tensor = sequence(input_tensor)


Cells
-----

Conv
~~~~

.. autoclass:: n2d2.cells.Conv
        :members:
        :inherited-members:

Deconv
~~~~~~

.. autoclass:: n2d2.cells.Deconv
        :members:
        :inherited-members:

Fc
~~

.. autoclass:: n2d2.cells.Fc
        :members:
        :inherited-members:

Dropout
~~~~~~~

.. autoclass:: n2d2.cells.Dropout
        :members:
        :inherited-members:

ElemWise
~~~~~~~~

.. autoclass:: n2d2.cells.ElemWise
        :members:
        :inherited-members:

Padding
~~~~~~~

.. autoclass:: n2d2.cells.Padding
        :members:
        :inherited-members:
        
Softmax
~~~~~~~

.. autoclass:: n2d2.cells.Softmax
        :members:
        :inherited-members:

BatchNorm2d
~~~~~~~~~~~

.. autoclass:: n2d2.cells.BatchNorm2d
        :members:
        :inherited-members:


Pool
~~~~

.. autoclass:: n2d2.cells.Pool
        :members:
        :inherited-members:


Configuration section
---------------------

If you want to add the same parameters to multiple cells, you can use a :py:class:`n2d2.ConfigSection`.

.. autoclass:: n2d2.ConfigSection
        :members:
        :inherited-members:

:py:class:`n2d2.ConfigSection` are used like dictionaries and passes to the constructor of classes like ``kwargs``. 

Example
~~~~~~~

.. code-block:: python

        conv_config = n2d2.ConfigSection(no_bias=True)
        n2d2.cells.Conv(3, 32, [4, 4], **conv_config)

This creates a :py:class:`n2d2.cells.Conv` with the parameter `no_bias=True`.
This functionality allow you to write more concise code, when multiple cells share the same parameters.

.. warning::
        If you want to pass an object as a parameter for multiple n2d2 object. You need to create a wrapping function to create your object.
        Example :

        .. code-block:: python

                def conv_def():
                        return n2d2.ConfigSection(weights_solver=n2d2.solver.SGD())
                n2d2.cells.Conv(3, 32, [4, 4], **conv_def())


Mapping
-------

You can change the mapping of the input for some cells (see if they have ``mapping`` parameter available).

You can create a mapping manually with a :py:class:`n2d2.Tensor` object :

.. testcode::

        mapping=n2d2.Tensor([15, 24], datatype="bool")
        mapping.set_values([
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1],
        [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1],
        [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1]])

Or use the Mapping object :

.. testcode::

        mapping=n2d2.mapping.Mapping(nb_channels_per_group=2).create_mapping(15, 24)

Which create the following mapping :

.. testoutput::

        1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
        0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
        0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0
        0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0
        0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1
        0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1



.. autoclass:: n2d2.mapping.Mapping
        :members:
        :inherited-members:


Solver
------

You can associate to some cell a specific weight solver.

.. autoclass:: n2d2.solver.Solver
        :members:
        :inherited-members:

SGD
~~~

.. autoclass:: n2d2.solver.SGD
        :members:
        :inherited-members:

Adam
~~~~

.. autoclass:: n2d2.solver.Adam
        :members:
        :inherited-members:


Filler
------

You can associate to some cell a specific weights and/or biases filler.

.. autoclass:: n2d2.filler.Filler
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

.. autoclass:: n2d2.activation.ActivationFunction
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

Target
------

Last cell of the network this object computes the loss.

To understand what the Target does, please refer to this part of the documentation : :doc:`Target INI </ini/target>`.



.. autoclass:: n2d2.target.Score
        :members:
        :inherited-members:

Example
~~~~~~~

How to use a `Target` to train your model :

.. code-block::

        # Propagation & BackPropagation example
        output = model(stimuli)
        loss = target(output)
        loss.back_propagate()
        loss.update()

Log performance analysis of your training :

.. code-block::

        ### After validation ###
        # save computational stats of the network 
        target.log_stats("name")
        # save a confusion matrix
        target.log_confusion_matrix("name")
        # save a graph of the loss and the validation score as a function of the number of steps
        target.log_success("name")