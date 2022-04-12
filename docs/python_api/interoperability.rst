Interoperability
================

In this section, we will present how you can use n2d2 with other python framework. 

Keras *[experimental feature]*
------------------------------

Presentation
~~~~~~~~~~~~

The Keras interoperability allow you to train a model using the N2D2 backend with the TensorFlow/Keras frontend.

The interoperability consist of a wrapper around the N2D2 Network.

In order to integrate N2D2 into the Keras environment, we run TensorFlow in eager mode. 



Documentation
~~~~~~~~~~~~~

.. autofunction:: keras_interoperability.wrap

.. autoclass:: keras_interoperability.CustomSequential
        :members:

Changing the optimizer
^^^^^^^^^^^^^^^^^^^^^^

.. warning::
        Due to the implementation, n2d2 parameters are not visible to ``Keras`` and thus cannot be optimized by a ``Keras`` optimizer.

When compiling the :py:class:`keras_interoperability.CustomSequential`, you can pass an :py:class:`n2d2.solver.Solver` object to the parameter `optimizer`.
This will change the method used to optimize the parameters.

.. code-block:: python

        model.summary() # Use the default SGD solver. 
        model.compile(loss="categorical_crossentropy", optimizer=n2d2.solver.Adam(), metrics=["accuracy"])
        model.summary() # Use the newly defined Adam solver.


Example
~~~~~~~

See the :doc:`keras example</python_api/example/keras>` section.

PyTorch *[experimental feature]*
--------------------------------

Presentation
~~~~~~~~~~~~

The PyTorch interoperability allow you to run an n2d2 model by using the Torch functions.

The interoperability consist of a wrapper around the N2D2 Network.
We created an autograd function which on ``Forward`` call the n2d2 ``Propagate`` method and on ``Backward`` call the n2d2 ``Back Propagate`` and ``Update`` methods.

.. figure:: ../_static/torch_interop.png
   :alt: schematic of the interoperability

.. warning::
        Due to the implementation n2d2 parameters are not visible to ``Torch`` and thus cannot be trained with a torch ``Optimizer``.

Tensor conversion
~~~~~~~~~~~~~~~~~ 

In order to achieve this interoperability, we need to convert Tensor from ``Torch`` to ``n2d2`` and vice versa.

:py:class:`n2d2.Tensor` require a contiguous memory space which is not the case for ``Torch``. Thus the conversion ``Torch`` to ``n2d2`` require a memory copy.
The opposite conversion is done with no memory copy.

If you work with ``CUDA`` tensor, the conversion ``Torch`` to ``n2d2`` is also done with no copy on the GPU (a copy on the host is however required).


Documentation
~~~~~~~~~~~~~

.. autofunction:: pytorch_interoperability.wrap

.. autoclass:: pytorch_interoperability.Block
        :members:

Example
~~~~~~~

See the :doc:`torch example</python_api/example/torch>` section.