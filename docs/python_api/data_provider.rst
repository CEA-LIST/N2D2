Data provider
=============
 
Introduction
------------

Data are loaded with the :doc:`Database<python_api/database>` objects. 
Once loaded, n2d2 use :py:class:`n2d2.provider.DataProvider` to feed the neural network with data.
:py:class:`n2d2.provider.DataProvider`, is the object that apply the :doc:`transformations<python_api/transformations>`.

.. autoclass:: n2d2.provider.DataProvider
        :members:
        :inherited-members:
