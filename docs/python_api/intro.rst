Introduction
============

For notation purposes, we will refer to the python library of N2D2 as n2d2. 
This library uses the core function of N2D2 and add an extra layer of abstraction to make the experience more user friendly. 
With the library you can import data, pre-process them, create a deep neural network model, train it and realize inference with it.
You can also import a network using the :doc:`ini file configuration<../ini/intro>` or the ONNX library.


Installation
------------

To run the python API, you need to use ``python 3.7``.

We highly recommend that you use a virtual environment, to set one up, you can follow these steps :

.. code-block:: bash

        # Creating python virtual environment
        virtualenv -p python3.7 env
        # Activating the virtual environment
        source env/bin/activate
        # Checking versions
        python --version
        pip --version
        # Leaving the virtual environment
        deactivate

If everything went well, you should have the version ``3.7`` of python. 

With setup.py
^^^^^^^^^^^^^

To install n2d2, you can go to the root of the project and use the ``setup.py`` script (with you **virtual environment activated**).

.. code-block:: bash

        python setup.py install

This should compile the n2d2 libraries and add it to your virtual environnement.

You can test it by trying to import n2d2 in your python interpreter :

.. code-block:: bash

        python
        >>> import n2d2
        >>> exit()

Manually
^^^^^^^^
If the ``setup.py`` script doesn't work, you can try to install manually the librarie.
When you compile ``N2D2``, the compiler creates a folder ``lib`` which contains the shared library of the binding between C++ and python (the file should be named ``N2D2-*.so``).
You need to move/copy this file at the root of the python folder ``N2D2-IP/N2D2/python``.

You can check that the binding is working by moving to the python folder and typing :

.. code-block:: bash

        python
        >>> import N2D2
        >>> exit()

If you have no error while importing ``N2D2``, the binding is working.

If you don't want to always move/copy the library, you can add the path where the library is located to your ``pythonpath``.
For this, you need to edit your ``.bashrc`` file. You can use any editor, for example : 

.. code-block:: bash

        nano ~/.bashrc

then add the line :

.. code-block:: bash

        export PYTHONPATH=$PYTHONPATH:path_to_build_lib

where ``path_to_build_lib`` is the path to the lib folder. Once this is done, use this command to apply the changes :

.. code-block:: bash

        source ~/.bashrc

You can also add the library n2d2 to you python path, if you don't plan to work on the python directory.

Once this is done, you can use the python library or the binding by importing respectively n2d2 or N2D2 in your python script. 


Default values
--------------

The python API used default values that you can modify at any time in your scripts.

List of modifiable parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here we will list parameters which can be directly modified in your script.

+--------------------------+-------------------------------------------------------------------+
| Default parameters       | Description                                                       |
+==========================+===================================================================+
| ``default_model``        | If you have compiled N2D2 with **CUDA**, you                      |
|                          | can use ``Frame_CUDA``, default= ``Frame``                        |
+--------------------------+-------------------------------------------------------------------+
| ``default_datatype``     | Datatype of the layer of the neural network. Can be ``int``or     |
|                          | ``float``, default= ``float``                                     |
|                          |                                                                   |
|                          | **Important :** This variable doesn't affect the data type of     |
|                          | :py:class:`n2d2.Tensor` objects.                                  |
+--------------------------+-------------------------------------------------------------------+
| ``verbosity``            | Level of verbosity, can be                                        |
|                          | ``n2d2.global_variables.Verbosity.graph_only``,                   |
|                          | ``n2d2.global_variables.Verbosity.short`` or                      |
|                          | ``n2d2.global_variables.Verbosity.detailed``,                     |
|                          | default= ``n2d2.global_variables.Verbosity.detailed``             |
+--------------------------+-------------------------------------------------------------------+

Example
^^^^^^^

.. code-block:: python

        n2d2.global_variables.default_model = "Frame_CUDA"

        n2d2.global_variables.default_datatype = "int"

        n2d2.global_variables.verbosity = n2d2.global_variables.Verbosity.graph_only


Method to set default values
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Some modifiable parameters require a method to be set.

+--------------------------+-------------------------------------------------------------------+
| Default values           | Description                                                       |
+==========================+===================================================================+
| ``set_random_seed``      | Seed used to generate random numbers, default = ``0``             |
+--------------------------+-------------------------------------------------------------------+
| ``set_cuda_device``      | Device to use for GPU computation with CUDA, default = ``0``      |
+--------------------------+-------------------------------------------------------------------+

Example
^^^^^^^

.. code-block:: python

        n2d2.global_variables.set_random_seed(1)
        n2d2.global_variables.set_cuda_device(1)