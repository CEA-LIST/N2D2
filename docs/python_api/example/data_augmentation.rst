Data augmentation
=================

In this example, we will see how to use :py:class:`n2d2.provider.DataProvider` and :py:class:`n2d2.transform.Transformation` to load data and do some data augmentation.

You can find the full python script here :download:`data_augmentation.py</../python/examples/data_augmentation.py>`.

Preliminary
-----------


For this tutorial, we will use n2d2 for data augmentation, and numpy and matplotlib for the visualization.

We will create a method plot_tensor to save the generated images from an :py:class:`n2d2.Tensor`


.. code-block::

    import n2d2
    import matplotlib.pyplot as plt

    def plot_tensor(tensor, path):
        plt.imshow(tensor[0,0,:], cmap='gray', vmin=0, vmax=255)
        plt.savefig(path)

Loading data
------------

We will begin by creating a :py:class:`n2d2.database.MNIST` driver to load the MNIST dataset.
We will then create a provider to get the images, we use a batch size of 1 to get only one image.


.. code-block::

    database = n2d2.database.MNIST(data_path="/local/DATABASE/mnist", validation=0.1)
    provider = n2d2.provider.DataProvider(database, [28, 28, 1], batch_size=1)


You can get the number of data per partition by using the method :py:meth:`n2d2.database.Database.get_partition_summary` which will print the paritionement of data.

.. code-block::

    database.get_partition_summary()


**Output :**

.. testoutput::

    Number of stimuli : 70000
    Learn         : 54000 stimuli (77.14%)
    Test          : 10000 stimuli (14.29%)
    Validation    : 6000 stimuli (8.57%)
    Unpartitioned : 0 stimuli (0.0%)


To select which partition you want to read from you need to use the method :py:meth:`n2d2.provider.DataProvider.set_partition`

To read data from a :py:class:`n2d2.provider.DataProvider` you can use multiple methods.

You can use the methods :py:meth:`n2d2.provider.DataProvider.read_batch` or :py:meth:`n2d2.provider.DataProvider.read_random_batch`. 


.. note::

    Since :py:class:`n2d2.provider.DataProvider` is an `iterable`, so you can use the ``next()`` function or a for loop !

    .. code-block::

        # for loop example
        for data in provider:
            pass
        # next example
        data = next(provider)

For this tutorial we will use :py:meth:`n2d2.provider.DataProvider.read_batch` !

With this code we will get the first image and plot it :

.. code-block::

    image = provider.read_batch(idx=0).to_numpy() * 255
    plot_tensor(image, "first_stimuli.png")


.. figure:: /_static/first_stimuli.png
   :alt: First stimuli of the MNIST dataset.

Data augmentation
-----------------

To do data augmentation with N2D2 we use :py:class:`n2d2.transform.Transformation`.
You can add transformation to provider with the method :py:meth:`n2d2.provider.DataProvider.add_on_the_fly_transformation` and :py:meth:`n2d2.provider.DataProvider.add_transformation`.

.. warning::

    Since we already loaded the first image the method :py:meth:`n2d2.provider.DataProvider.add_transformation` would not apply the transformation to the image.

By using the transformation :py:class:`n2d2.transform.Flip` we will flip vertically our image.

.. code-block::

    provider.add_on_the_fly_transformation(n2d2.transform.Flip(vertical_flip=True))

    image = provider.read_batch(idx=0).to_numpy() * 255
    plot_tensor(image, "first_stimuli_fliped.png")


.. figure:: /_static/first_stimuli_fliped.png
   :alt: First stimuli of the MNIST dataset but flipped.

We will negate the first transformation with another :py:class:`n2d2.transform.Flip` which we will add with the method :py:meth:`n2d2.provider.DataProvider.add_transformation`.

.. code-block::

    # negating the first transformation with another one
    provider.add_transformation(n2d2.transform.Flip(vertical_flip=True))
    image = provider.read_batch(idx=1).to_numpy() * 255
    plot_tensor(image, "second_stimuli.png")

.. figure:: /_static/second_stimuli.png
   :alt: Second stimuli of the MNIST dataset.


Getting labels
--------------

To get the labels 

.. code-block::


    print("Second stimuli label : ", provider.get_labels()[0])

**Output :**

.. testoutput::

    Second stimuli label : 5
