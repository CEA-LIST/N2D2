Target
======

Introduction
------------

A :py:class:`N2D2.Target` is associated to a :py:class:`N2D2.Cell`, it define the output of the network.
The computation of the loss and other tools to compute score such as the confusion matrix are also computed with this class. 

To train a neural network you need to use :py:meth:`N2D2.Target.provideTargets` then to :py:meth:`N2D2.cell.propagate` then :py:meth:`N2D2.Target.process` and finally :py:meth:`N2D2.Cell.backpropagate`.
(See :doc:`the MNIST example<../python_binding/example>`.)

.. autoclass:: N2D2.Target
   :members: