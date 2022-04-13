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

Example
^^^^^^^

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

NeuralNetworkCell
~~~~~~~~~~~~~~~~~

.. autoclass:: n2d2.cells.NeuralNetworkCell
        :members:
        :inherited-members:

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

Saving parameters
-----------------

You can save the parameters (weights, biases ...) of your network with the method `export_free_parameters`.
To load those parameters you can use the method `import_free_parameters`.

With n2d2 you can choose wether you want to save the parameters of a part of your network or of all your graph.

+------------------------------------------+----------------------------------------------------------------+----------------------------------------------------------------+
|            Object                        |                      Save parameters                           |                        Load parameters                         |
+==========================================+================================================================+================================================================+
| :py:class:`n2d2.cells.NeuralNetworkCell` | :py:meth:`n2d2.cells.NeuralNetworkCell.export_free_parameters` | :py:meth:`n2d2.cells.NeuralNetworkCell.import_free_parameters` |
+------------------------------------------+----------------------------------------------------------------+----------------------------------------------------------------+
| :py:class:`n2d2.cells.Block`             | :py:meth:`n2d2.cells.Block.import_free_parameters`             |  :py:meth:`n2d2.cells.Block.import_free_parameters`            |
+------------------------------------------+----------------------------------------------------------------+----------------------------------------------------------------+


Configuration section
---------------------

If you want to add the same parameters to multiple cells, you can use a :py:class:`n2d2.ConfigSection`.

.. autoclass:: n2d2.ConfigSection


:py:class:`n2d2.ConfigSection` are used like dictionaries and passes to the constructor of classes like ``kwargs``. 

Usage example
~~~~~~~~~~~~~

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

You can associate at construction and run time a :py:class:`n2d2.solver.Solver` object to a cell. This solver object will optimize the parameters of your cell using a specific algorithm. 

Usage example
~~~~~~~~~~~~~

In this short example we will see how to associate a solver to a model and to a cell object at construction and at runtime.

Set solver at construction time
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Let's create a couple of :py:class:`n2d2.cells.Fc` cell and add them to a :py:class:`n2d2.cells.Sequence`.
At construction time we will set the solver of one of them to a :py:class:`n2d2.solver.SGD` with a ``learning_rate=0.1``.

.. code-block::

        import n2d2

        cell1 = n2d2.cells.Fc(2,2, solver=n2d2.solver.SGD(learning_rate=0.1))
        cell2 = n2d2.cells.Fc(2,2)

        model = n2d2.cells.Sequence([cell1, cell2])

        print(model)

**Output :**

.. testoutput::

        'Sequence_0' Sequence(
                (0): 'Fc_0' Fc(Frame<float>)(nb_inputs=2, nb_outputs=2 | back_propagate=True, drop_connect=1.0, no_bias=False, normalize=False, outputs_remap=, weights_export_format=OC, activation=None, weights_solver=SGD(clamping=, decay=0.0, iteration_size=1, learning_rate=0.1, learning_rate_decay=0.1, learning_rate_policy=None, learning_rate_step_size=1, max_iterations=0, min_decay=0.0, momentum=0.0, polyak_momentum=True, power=0.0, warm_up_duration=0, warm_up_lr_frac=0.25), bias_solver=SGD(clamping=, decay=0.0, iteration_size=1, learning_rate=0.1, learning_rate_decay=0.1, learning_rate_policy=None, learning_rate_step_size=1, max_iterations=0, min_decay=0.0, momentum=0.0, polyak_momentum=True, power=0.0, warm_up_duration=0, warm_up_lr_frac=0.25), weights_filler=Normal(mean=0.0, std_dev=0.05), bias_filler=Normal(mean=0.0, std_dev=0.05), quantizer=None)
                (1): 'Fc_1' Fc(Frame<float>)(nb_inputs=2, nb_outputs=2 | back_propagate=True, drop_connect=1.0, no_bias=False, normalize=False, outputs_remap=, weights_export_format=OC, activation=None, weights_solver=SGD(clamping=, decay=0.0, iteration_size=1, learning_rate=0.01, learning_rate_decay=0.1, learning_rate_policy=None, learning_rate_step_size=1, max_iterations=0, min_decay=0.0, momentum=0.0, polyak_momentum=True, power=0.0, warm_up_duration=0, warm_up_lr_frac=0.25), bias_solver=SGD(clamping=, decay=0.0, iteration_size=1, learning_rate=0.01, learning_rate_decay=0.1, learning_rate_policy=None, learning_rate_step_size=1, max_iterations=0, min_decay=0.0, momentum=0.0, polyak_momentum=True, power=0.0, warm_up_duration=0, warm_up_lr_frac=0.25), weights_filler=Normal(mean=0.0, std_dev=0.05), bias_filler=Normal(mean=0.0, std_dev=0.05), quantizer=None)
        )

Set a solver for a specific parameter
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We can set a new solver for the bias of the second cell fully connected cell. This solver will be different than the weight parameter one.

.. Note::

        Here we access the cell via its instanciate object but we could have used its name : ``model["Fc_1"].bias_solver=n2d2.solver.Adam()``.

.. code-block::

        cell2.bias_solver=n2d2.solver.Adam()

        print(model)

**Output :**

.. testoutput::

        'Sequence_0' Sequence(
                (0): 'Fc_0' Fc(Frame<float>)(nb_inputs=2, nb_outputs=2 | back_propagate=True, drop_connect=1.0, no_bias=False, normalize=False, outputs_remap=, weights_export_format=OC, activation=None, weights_solver=SGD(clamping=, decay=0.0, iteration_size=1, learning_rate=0.1, learning_rate_decay=0.1, learning_rate_policy=None, learning_rate_step_size=1, max_iterations=0, min_decay=0.0, momentum=0.0, polyak_momentum=True, power=0.0, warm_up_duration=0, warm_up_lr_frac=0.25), bias_solver=SGD(clamping=, decay=0.0, iteration_size=1, learning_rate=0.1, learning_rate_decay=0.1, learning_rate_policy=None, learning_rate_step_size=1, max_iterations=0, min_decay=0.0, momentum=0.0, polyak_momentum=True, power=0.0, warm_up_duration=0, warm_up_lr_frac=0.25), weights_filler=Normal(mean=0.0, std_dev=0.05), bias_filler=Normal(mean=0.0, std_dev=0.05), quantizer=None)
                (1): 'Fc_1' Fc(Frame<float>)(nb_inputs=2, nb_outputs=2 | back_propagate=True, drop_connect=1.0, no_bias=False, normalize=False, outputs_remap=, weights_export_format=OC, activation=None, weights_solver=SGD(clamping=, decay=0.0, iteration_size=1, learning_rate=0.01, learning_rate_decay=0.1, learning_rate_policy=None, learning_rate_step_size=1, max_iterations=0, min_decay=0.0, momentum=0.0, polyak_momentum=True, power=0.0, warm_up_duration=0, warm_up_lr_frac=0.25), bias_solver=Adam(beta1=0.9, beta2=0.999, clamping=, epsilon=1e-08, learning_rate=0.001), weights_filler=Normal(mean=0.0, std_dev=0.05), bias_filler=Normal(mean=0.0, std_dev=0.05), quantizer=None)
        )


Set a solver for a model
^^^^^^^^^^^^^^^^^^^^^^^^

We can set a solver to the whole :py:class:`n2d2.cells.Sequence` with the method :py:meth:`n2d2.cells.Sequence.set_solver`.

.. code-block::

        model.set_solver(n2d2.solver.Adam(learning_rate=0.1))

        print(model)

**Output :**

.. testoutput::

        'Sequence_0' Sequence(
                (0): 'Fc_0' Fc(Frame<float>)(nb_inputs=2, nb_outputs=2 | back_propagate=True, drop_connect=1.0, no_bias=False, normalize=False, outputs_remap=, weights_export_format=OC, activation=None, weights_solver=Adam(beta1=0.9, beta2=0.999, clamping=, epsilon=1e-08, learning_rate=0.1), bias_solver=Adam(beta1=0.9, beta2=0.999, clamping=, epsilon=1e-08, learning_rate=0.1), weights_filler=Normal(mean=0.0, std_dev=0.05), bias_filler=Normal(mean=0.0, std_dev=0.05), quantizer=None)
                (1): 'Fc_1' Fc(Frame<float>)(nb_inputs=2, nb_outputs=2 | back_propagate=True, drop_connect=1.0, no_bias=False, normalize=False, outputs_remap=, weights_export_format=OC, activation=None, weights_solver=Adam(beta1=0.9, beta2=0.999, clamping=, epsilon=1e-08, learning_rate=0.1), bias_solver=Adam(beta1=0.9, beta2=0.999, clamping=, epsilon=1e-08, learning_rate=0.1), weights_filler=Normal(mean=0.0, std_dev=0.05), bias_filler=Normal(mean=0.0, std_dev=0.05), quantizer=None)
        )




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

You can associate to a cell at construction time a :py:class:`n2d2.filler.Filler` object. This object will fill weights and biases using a specific method.


Usage example
~~~~~~~~~~~~~

In this short example we will see how to associate a filler to a cell object, how to get the weights and biases and how to set a new filler and refill the weights.


Setting a filler at construction time
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We begin by importing ``n2d2`` and creating a :py:class:`n2d2.cells.Fc` object. We will associate a :py:class:`n2d2.filler.Constant` filler.

.. Note::

        If you want to set a filler only for weights (or biases) you could have used the parameter ``weight_filler`` (or ``bias_filler``).

.. code-block::

        import n2d2
        cell = n2d2.cells.Fc(2,2, filler=n2d2.filler.Constant(value=1.0))

If you print the weights, you will see that they are all set to one.

.. code-block::

        print("--- Weights ---")
        for channel in cell.get_weights():
        for value in channel:
                print(value)

**Output :**

.. testoutput::

        --- Weights ---
        n2d2.Tensor([
        1
        ], device=cpu, datatype=f)
        n2d2.Tensor([
        1
        ], device=cpu, datatype=f)
        n2d2.Tensor([
        1
        ], device=cpu, datatype=f)
        n2d2.Tensor([
        1
        ], device=cpu, datatype=f)

Same with the biases

.. code-block::

        print("--- Biases ---")
        for channel in cell.get_biases():
        print(channel)

**Output :**

.. testoutput::

        --- Biases ---
        n2d2.Tensor([
        1
        ], device=cpu, datatype=f)
        n2d2.Tensor([
        1
        ], device=cpu, datatype=f)



Changing the filler of an instanciated object 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can set a new filler for bias by changing the ``bias_filler`` attribute (or ``weight_filler`` for only weights or ``filer`` for both).

However changing the filler doesn't change the parameter values, you need to use the method :py:meth:`n2d2.cells.Fc.refill_bias` (see also :py:meth:`n2d2.cells.Fc.refill_weights`)

.. Note::
 
        You can also use the method  :py:meth:`n2d2.cells.Fc.set_filler`, :py:meth:`n2d2.cells.Fc.set_weights_filler` and :py:meth:`n2d2.cells.Fc.set_biases_filler`. Which have a refill option.

.. code-block::

        cell.bias_filler=n2d2.filler.Normal()
        cell.refill_bias()

You can then observe the new biases :

.. code-block::

        print("--- New Biases ---")
        for channel in cell.get_biases():
        print(channel)


**Output :**

.. testoutput::

        --- New Biases ---
        n2d2.Tensor([
        1.32238
        ], device=cpu, datatype=f)
        n2d2.Tensor([
        -0.0233932
        ], device=cpu, datatype=f)


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

Usage example
~~~~~~~~~~~~~

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