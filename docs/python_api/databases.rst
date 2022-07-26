Databases
=========
 
Introduction
------------

The python library integrates pre-defined modules for several well-known database used in the deep learning community, such as MNIST, GTSRB, CIFAR10 and so on. 
That way, no extra step is necessary to be able to directly build a network and learn it on these database.
The library allow you to add pre-process data with built in Transformation.



Database
--------

The python library provide you with multiple object to manipulate common database.

Loading hand made database can be done using :py:class:`n2d2.database.DIR`.

Like in the following example :

.. testcode::

        # Creating the database object 
        db = n2d2.database.DIR()

        provider = n2d2.provider.DataProvider(db, data_dims)

        # The zeroes represent the depth to seek the data.
        db.load(data_path, 0, label_path, 0)

        # With this line we put all the data in the learn partition:
        db.partition_stimuli(learn=1, validation=0, test=0)
        provider.set_partition("Learn")

        inputs_tensor = provider.read_random_batch() 


DIR
~~~


Loading a custom database
^^^^^^^^^^^^^^^^^^^^^^^^^

Hand made database stored in files directories are directly supported
with the ``DIR_Database`` module. For example, suppose your database is
organized as following :

- ``GST/airplanes``: 800 images

- ``GST/car_side``: 123 images

- ``GST/Faces``: 435 images

- ``GST/Motorbikes``: 798 images



You can then instanciate this database as input of your neural network
using the following line:

.. code-block:: python

        database = n2d2.database.DIR("./GST", learn=0.4, validation=0.2)

Each subdirectory will be treated as a different label, so there will be
4 different labels, named after the directory name.

The stimuli are equi-partitioned for the learning set and the validation
set, meaning that the same number of stimuli for each category is used.
If the learn fraction is 0.4 and the validation fraction is 0.2, as in
the example above, the partitioning will be the following:

+-------------+------------------+-------------+------------------+------------+
| Label ID    | Label name       | Learn set   | Validation set   | Test set   |
+-------------+------------------+-------------+------------------+------------+
| [0.5ex] 0   | ``airplanes``    | 49          | 25               | 726        |
+-------------+------------------+-------------+------------------+------------+
| 1           | ``car_side``     | 49          | 25               | 49         |
+-------------+------------------+-------------+------------------+------------+
| 2           | ``Faces``        | 49          | 25               | 361        |
+-------------+------------------+-------------+------------------+------------+
| 3           | ``Motorbikes``   | 49          | 25               | 724        |
+-------------+------------------+-------------+------------------+------------+
|             | Total:           | 196         | 100              | 1860       |
+-------------+------------------+-------------+------------------+------------+


.. Note::

    If ``equiv_label_partitioning`` is 1 (default setting), the number of stimuli
    per label that will be partitioned in the learn and validation sets will 
    correspond to the number of stimuli from the label with the fewest stimuli.


To load and partition more than one ``DataPath``, one can use the 
:py:meth:`n2d2.database.Database.load` method.

This method will load data in the partition ``Unpartitionned``, you can move the stimuli
in the ``Learn``, ``Validation`` or ``Test`` partition using the 
:py:meth:`n2d2.database.Database.partition_stimuli` method.

Handling labelization
^^^^^^^^^^^^^^^^^^^^^

By default, your labels will be ordered by alphabetical order.
If you need your label to be in a specific order, you can specify it using 
an exterior file we will name it ``label.dat`` for this example :

.. code-block::

        airplanes 0
        car_side 1
        Motorbikes 3
        Faces 2


Then to load the database we will use :

.. code-block:: python

        database = n2d2.database.DIR("./GST", learn=0.4, validation=0.2, label_path="./label.dat", label_depth=0)

.. warning::

        It is important to specify ``label_depth=0`` if you are specifying ``label_path`` !

.. autoclass:: n2d2.database.DIR
        :members:
        :inherited-members:

MNIST
~~~~~

.. autoclass:: n2d2.database.MNIST
        :members:
        :inherited-members:


ILSVRC2012
~~~~~~~~~~

.. autoclass:: n2d2.database.ILSVRC2012
        :members:
        :inherited-members:


CIFAR10
~~~~~~~

.. autoclass:: n2d2.database.CIFAR10
        :members:
        :inherited-members:


CIFAR100
~~~~~~~~

.. autoclass:: n2d2.database.CIFAR100
        :members:
        :inherited-members:


Cityscapes
~~~~~~~~~~

.. autoclass:: n2d2.database.Cityscapes
        :members:
        :inherited-members:



GTSRB
~~~~~

.. autoclass:: n2d2.database.GTSRB
        :members:
        :inherited-members:

Transformations
---------------

.. autoclass:: n2d2.transform.Transformation
        :members:
        :inherited-members:

Composite
~~~~~~~~~

.. autoclass:: n2d2.transform.Composite
        :members:
        :inherited-members:
        
PadCrop
~~~~~~~

.. autoclass:: n2d2.transform.PadCrop
        :members:
        :inherited-members:
        
Distortion
~~~~~~~~~~

.. autoclass:: n2d2.transform.Distortion
        :members:
        :inherited-members:

Rescale
~~~~~~~

.. autoclass:: n2d2.transform.Rescale
        :members:
        :inherited-members:

ColorSpace
~~~~~~~~~~

.. autoclass:: n2d2.transform.ColorSpace
        :members:
        :inherited-members:


Flip
~~~~~~~~~~

.. autoclass:: n2d2.transform.Flip
        :members:
        :inherited-members:

RangeAffine
~~~~~~~~~~~

.. autoclass:: n2d2.transform.RangeAffine
        :members:
        :inherited-members:

SliceExtraction
~~~~~~~~~~~~~~~

.. autoclass:: n2d2.transform.SliceExtraction
        :members:
        :inherited-members:

RandomResizeCrop
~~~~~~~~~~~~~~~~

.. autoclass:: n2d2.transform.RandomResizeCrop
        :members:
        :inherited-members:

ChannelExtraction
~~~~~~~~~~~~~~~~~

.. autoclass:: n2d2.transform.ChannelExtraction
        :members:
        :inherited-members:

Sending data to the Neural Network
----------------------------------

Once a database is loaded, n2d2 use :py:class:`n2d2.provider.DataProvider` to provide data to the neural network.

The :py:class:`n2d2.provider.DataProvider` will automatically apply the :py:class:`n2d2.transform.Transformation` to the dataset. 
To add a transformation to the provider, you should use the method :py:meth:`n2d2.transform.Transformation.add_transformation`.

.. autoclass:: n2d2.provider.DataProvider
        :members:
        :inherited-members:

Example
-------

In this example, we will show you how to create a :py:class:`n2d2.database.Database`, :py:class:`n2d2.provider.Provider` and apply :py:class:`n2d2.transformation.Transformation` to the data.

We will use the :py:class:`n2d2.database.MNIST` database driver, rescale the images to a 32x32 pixels size and then print the data used for the learning.

.. testcode::

        # Loading data
        database = n2d2.database.MNIST(data_path=path, validation=0.1)

        # Initializing DataProvider
        provider = n2d2.provider.DataProvider(database, [32, 32, 1], batch_size=batch_size)

        # Applying Transformation
        provider.add_transformation(n2d2.transform.Rescale(width=32, height=32))

        # Setting the partition of data we will use
        provider.set_partition("Learn")

        # Iterating other the inputs
        for inputs in provider:
                print(inputs)