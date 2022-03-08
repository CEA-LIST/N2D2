Introduction
============

For notation purposes, we will refer to the python library of N2D2 as n2d2. 
This library uses the core function of N2D2 and add an extra layer of abstraction to make the experience more user friendly. 
With the library you can import data, pre-process them, create a deep neural network model, train it and realize inference with it.
You can also import a network using the :doc:`ini file configuration<../ini/intro>` or the ONNX library.


Here are the functionalities available with the Python API :

+------------------------+------------+------------------+
|        Feature         |  Available | Python API Only  |
+========================+============+==================+
| Import a network from  | ✔️         |                  |
| an INI file            |            |                  |
+------------------------+------------+------------------+
| Import a network from  | ✔️         |                  |
| an ONNX file           |            |                  |
+------------------------+------------+------------------+
| Build a network with   | ✔️         |                  |
| the API                |            |                  |
+------------------------+------------+------------------+
| Load and apply         | ✔️         |                  |
| transformation to a    |            |                  |
| dataset                |            |                  |
+------------------------+------------+------------------+
| Train a network        | ✔️         |                  |
+------------------------+------------+------------------+
| Flexible definition of | ✔️         | ✔️               |
| the computation graph  |            |                  |
+------------------------+------------+------------------+
| Test a network with    | ✔️         |                  |
| the N2D2 analysis tools|            |                  |
+------------------------+------------+------------------+
| Torch interoperability | ✔️         | ✔️               |
+------------------------+------------+------------------+
| Keras interoperability | ❌         | ✔️               |
+------------------------+------------+------------------+
| Multi GPU support      | ✔️         |                  |
+------------------------+------------+------------------+
| Exporting network      | ❌         |                  |
+------------------------+------------+------------------+


Installation of the virtual environment
---------------------------------------

| To run the python API, it’s good practice to use ``python 3.7`` or a newer version in a virtual environment.
| To set up your environment, please follow these steps:

.. code-block:: bash

        # Create your python virtual environment
        virtualenv -p python3.7 env

        # Activate the virtual environment
        source env/bin/activate

        # Check versions
        python --version
        pip --version

        # To leave the virtual environment
        deactivate

If everything went well, you should have the version ``3.7`` of python. 


Installation of the Python API
------------------------------

| There are multiple methods to install the python API on your device.
| Feel free to use the method of your choice.


With the Python Package Index (Py Pi)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can have access to the last stable version of the python API by using
``pip`` and importing the package ``n2d2``.

.. code-block:: bash

        pip install n2d2



From the N2D2 Github repository
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can have access to the developer version by importing the API from
the N2D2 Github repository via ``pip``.

.. code-block:: bash

        pip install git+https://github.com/CEA-LIST/N2D2  



If you have already cloned the Github repository  
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  

You can still build the python API with a cloned N2D2 repository.
Go at the root of the N2D2 projet and follow the following steps 
(don't forget to activate your virtual environment before).

.. code-block:: bash

        # Build the N2D2 library
        python setup.py bdist_wheel

        # Install the n2d2 python packages in your virtual environment
        pip install dist/*


Test of the Python API
----------------------

Whatever the method you chose, it should compile the n2d2 libraries and add them to your virtual environnement.

You can test it by trying to import n2d2 in your python interpreter :

.. code-block:: bash

        python
        >>> import n2d2
        >>> print(n2d2.Tensor([2,3]))
        n2d2.Tensor([
        0 0 0
        0 0 0
        ], device=cpu, datatype=float)
        >>> exit()

You can find more examples in the Python API section if you want to test every feature.

| It might be possible you could find some issues by using the API.
| So please notify us at https://github.com/CEA-LIST/N2D2/issues if you find any problem or any possible improvement.


Default values
--------------

The python API used default values that you can modify at any time in your scripts.

List of modifiable parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Here we will list parameters which can be directly modified in your script.

+--------------------------+-------------------------------------------------------------------+
| Default parameters       | Description                                                       |
+==========================+===================================================================+
| ``default_model``        | If you have compiled N2D2 with **CUDA**, you                      |
|                          | can use ``Frame_CUDA``, default= ``Frame``                        |
+--------------------------+-------------------------------------------------------------------+
| ``default_datatype``     | Datatype of the layer of the neural network. Can be ``double`` or |
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
|``seed``                  | Seed used to generate random numbers(0 = time based),             |
|                          | default = ``0``                                                   |
+--------------------------+-------------------------------------------------------------------+
|``cuda_device``           | Device to use for GPU computation with CUDA, you can enable multi | 
|                          | GPU by giving a tuple of device, default = ``0``                  |
+--------------------------+-------------------------------------------------------------------+



Example
^^^^^^^

.. code-block:: python

        n2d2.global_variables.default_model = "Frame_CUDA"

        n2d2.global_variables.default_datatype = "double"

        n2d2.global_variables.verbosity = n2d2.global_variables.Verbosity.graph_only
        
        n2d2.global_variables.seed = 1

        n2d2.global_variables.cuda_device = 1
        # Multi GPU example :
        n2d2.global_variables.cuda_device = 0, 1 
