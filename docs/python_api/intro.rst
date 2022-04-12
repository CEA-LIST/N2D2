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
| Keras interoperability | ✔️         | ✔️               |
+------------------------+------------+------------------+
| Multi GPU support      | ✔️         |                  |
+------------------------+------------+------------------+
| Exporting network      | ✔️         |                  |
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

.. Warning::
        
        This method is not supported anymore, we are working on it !

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
        pip install .

Installation for developer
^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you want to install n2d2 as seomeone who wants to contribute to n2d2, we recommand the following setup :

Inside your n2d2 project, create a build folder and compile N2D2 inside it :

.. code-block:: bash

        mkdir build && cd build
        cmake .. && make -j 8

Once this is done, you have generated the shared object : ``lib/n2d2.*.so``.

You can add the generated `lib` folder and the python source in your ``PYTHONPATH`` with the command :

.. code-block:: bash

        export PYTHONPATH=$PYTHONPATH:<N2D2_BUILD_PATH>/lib:<N2D2_PATH>/python

.. Note::

        Add this line in your bashrc to always have a good ``PYTHONPATH`` setup !

To check if your PYTHONPATH works properly you can try to import ``N2D2`` (verify that the compilation went well) 
and then ``n2d2`` (verify that your ``PYTHONPATH`` point the n2d2 python API).

Frequent issues
^^^^^^^^^^^^^^^

Module not found N2D2
~~~~~~~~~~~~~~~~~~~~~

If when you import ``n2d2`` you get this error :

.. code-block::
        
        ModuleNotFoundError: No module named 'N2D2'

This is likely due to your python version not matching with the one used to compile N2D2.

You can find in your ``site-packages`` (or in your ``build/lib`` if you have compiled N2D2 with CMake) a ``.so`` file named like this : ``N2D2.cpython-37m-x86_64-linux-gnu.so``.

This file name indicates the python version used to compile N2D2, in this example 3.7.

You should either make sure to use a virtualenv with the right python version or check the bellow section.

N2D2 doesn't compile with the right version of Python
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When compiling N2D2 you can use an argument to specify the python version you want to compile N2D2 for.

.. code-block::

        cmake -DPYTHON_EXECUTABLE=<path_to_python_binary> <path_to_n2d2_cmakefile>

.. note::

        On linux you can use ``$(which python)`` to  use your default python binary.

You can then check the version of python on the shared object in ``build/lib``. 

For example, this shared object ``N2D2.cpython-37m-x86_64-linux-gnu.so`` have been compiled for python3.7.


Lib not found when compiling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If CMake fails to find lib files when compiling, this may be due to the absence of the dependency ``python3-dev``.

When generating a new virtualenv after installing the dependency, you should see ``include/python3.7m`` inside the generated folder.

If not, you may need to reboot in order to update system variables.


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
