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



.. autoclass:: n2d2.database.Database
        :members:
        :inherited-members:

DIR
~~~

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
---------------------------------

With a DataProvider
~~~~~~~~~~~~~~~~~~~

Once a database loaded, n2d2 use :py:class:`n2d2.provider.DataProvider` to provide data to the neural network.

.. autoclass:: n2d2.provider.DataProvider
        :members:
        :inherited-members:

Without a DataProvider
~~~~~~~~~~~~~~~~~~~~~~

You can send a :py:class:`n2d2.Tensor` that doesn't come from a :py:class:`n2d2.provider.DataProvider` to the network. 

By doing so, you create a :py:class:`n2d2.provider.TensorPlaceholder` that will stream your tensor directly to the network.

If you want to do a back propagation, you need to use a :py:class:`n2d2.application.LossFunction` that require a :py:class:`n2d2.provider.DataProvider`.

You can create a :py:class:`n2d2.provider.TensorPlaceholder` and specify the labels associated with the data, this will act as a :py:class:`n2d2.provider.DataProvider`.

.. autoclass:: n2d2.provider.TensorPlaceholder
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