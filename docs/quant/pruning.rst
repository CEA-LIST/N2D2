Pruning
=======

Getting Started
~~~~~~~~~~~~~~~

N2D2 provides a pruning module to perform pruning operations on your model in order to reduce its memory footprint.
The module works like the QAT module i.e. it is possible to carry out trainings with pruned weights in order to improve the performance of the network.
Only weights can be pruned so far.

Example with Python
~~~~~~~~~~~~~~~~~~~

.. autoclass:: n2d2.quantizer.PruneCell
        :members:
        :inherited-members:

Example of code to use the *PruneCell* in your scripts:

.. code-block:: python

    for cell in model:
    ### Add Pruning ###
    if isinstance(cell, n2d2.cells.Conv) or isinstance(cell, n2d2.cells.Fc):
        cell.quantizer = n2d2.quantizer.PruneCell(prune_mode="Static", threshold=0.3, prune_filler="IterNonStruct")

Some explanations with the differents options of the *PruneCell*:

Pruning mode
^^^^^^^^^^^^

3 modes are possible:

- Identity: no pruning is applied to the cell
- Static: all weights of the cell are pruned to the requested ``threshold`` at initialization
- Gradual: the weights are pruned to the ``start`` threshold at initialization and at each update of the current threshold, it is increased by ``gamma`` until it reaches ``threshold``. By default, the update is performed at the end of each epoch (possible to change it with ``stepsize``)

**Warning**: if you use ``stepsize``, please indicate the number of steps and not the number of epochs.
For example, to update each two epochs, write:

.. code-block:: python

    n2d2.quantizer.PruneCell(prune_mode="Gradual", threshold=0.3, stepsize=2*DATASET_SIZE)

Where *DATASET_SIZE* is the size of the dataset you are using.

Pruning filler
^^^^^^^^^^^^^^

2 fillers are available to fill the masks:

- Random: The masks are filled randomly
- IterNonStruct: all weights below than the ``delta`` factor are pruned. If this is not enough to reach ``threshold``, all the weights below 2 "delta" are pruned and so on...


**Important**: With *PruneCell*, ``quant_mode`` and ``range`` are not used.


Example with INI file
~~~~~~~~~~~~~~~~~~~~~

The common set of parameters for any kind of Prune Quantizer.

+----------------------------------------+-------------------------------------------------------------------------------------------+
| Option [default value]                 | Description                                                                               |
+========================================+===========================================================================================+
| ``QWeight``                            | Quantization / Pruning method, choose ``Prune`` to activate the Pruning mode.             |
+----------------------------------------+-------------------------------------------------------------------------------------------+
| ``QWeight.PruningMode`` [``Identity``] | Pruning mode, can be ``Identity``, ``Static`` or ``Gradual``                              |
+----------------------------------------+-------------------------------------------------------------------------------------------+
| ``QWeight.PruningFiller`` [``Random``] | Pruning filler for the weights, can be ``Random``, ``IterNonStruct`` or ``None``          |
+----------------------------------------+-------------------------------------------------------------------------------------------+
| ``QWeight.Threshold`` [``0.2``]        | Weight threshold to be pruned, 0.2 means 20% for example                                  |
+----------------------------------------+-------------------------------------------------------------------------------------------+
| ``QWeight.Delta`` [``0.001``]          | Factor for iterative pruning, use it with ``IterNonStruct`` pruning filler                |
+----------------------------------------+-------------------------------------------------------------------------------------------+
| ``QWeight.StartThreshold`` [``0.1``]   | Starting threshold, use it with ``Gradual`` pruning mode                                  |
+----------------------------------------+-------------------------------------------------------------------------------------------+
| ``QWeight.StepSizeThreshold`` [``0``]  | Step size for the threshold update, use it with ``Gradual`` pruning mode                  |
+----------------------------------------+-------------------------------------------------------------------------------------------+
| ``QWeight.GammaThreshold`` [``0.05``]  | Value to add to current threshold during its update, use it with ``Gradual`` pruning mode |
+----------------------------------------+-------------------------------------------------------------------------------------------+

Example of code to use the *Prune Quantizer* in your scripts:

.. code-block:: ini

    [conv1]
    Input=sp
    Type=Conv
    KernelDims=5 5
    NbOutputs=6
    ActivationFunction=Rectifier
    WeightsFiller=HeFiller
    ConfigSection=common.config
    QWeight=Prune
    QWeight.PruningMode=Static
    QWeight.PruningFiller=IterNonStruct
    QWeight.Threshold=0.3
    QWeight.StartThreshold=0.1
    QWeight.GammaThreshold=0.1


All explanations in relation to the parameters of Prune Quantizer are provided in the python section of this page.
