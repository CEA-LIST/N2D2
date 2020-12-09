Introduction
============

In this section we will present the C++ core function that are binded to Python with the framework pybind.
The binding of the C++ core is straightforward, thus this section can also be seen as a documentation of the C++ core implementation of N2D2. 

If you want to use the raw python binding, you will need to compile N2D2 using the command :

.. code-block:: bash

    make N2D2_BINDIR=./build CUDA=1 MONGODB=1 ONX=1 PYBIND=python3.7 -j12 NOWERROR=1 pybind

This command will create a '.so' file in the folder *./python*. 
If you want to use the raw binding, you will need to have this file at the root of your project.
It is however not recommended to use the raw binding, you should instead use the :doc:`n2d2 python library<../python_api/intro>`. 

