Load Numpy Data
===============

In this example, we will see how to load data from a Numpy array using :py:class:`n2d2.database.numpy`.

You can find the full python script here :download:`load_numpy_data.py</../python/examples/load_numpy_data.py>`.

Preliminary
-----------

For this tutorial, we will create a database using the following keras data loader : https://keras.io/api/datasets/fashion_mnist/.

Available by importing :

.. code-block::

    from tensorflow.keras.datasets.fashion_mnist import load_data

    (x_train, y_train), (x_test, y_test) = load_data()


Load data into N2D2
-------------------

Now that we have our data in the form of Numpy array we can create and populate the :py:class:`n2d2.database.numpy`.

.. code-block::

    import n2d2

    # Instanciate Numpy database object
    db = n2d2.database.Numpy()
    # Load train set
    db.load([a for a in x_train], [(int)(i.item()) for i in y_train])
    # Add the loaded data to the Learn partition
    db.partition_stimuli(1., 0., 0.) # Learn Validation Test

    # Load test set in the validation partition
    db.load([a for a in x_test], [(int)(i.item()) for i in y_test], partition="Validation")

    # Print a summary
    db.get_partition_summary()

.. testoutput::

    Number of stimuli : 70000
    Learn         : 60000 stimuli (85.71%)
    Test          : 0 stimuli (0.0%)
    Validation    : 10000 stimuli (14.29%)
    Unpartitioned : 0 stimuli (0.0%)


Training a model using the numpy database
-----------------------------------------

Before anything, we will import the following modules :

.. code-block::

    import n2d2
    from n2d2.cells.nn import Fc, Softmax
    from n2d2.cells import Sequence
    from n2d2.solver import SGD
    from n2d2.activation import Rectifier, Linear
    from math import ceil


For this example we will create a very simple model :

.. code-block::

    model = Sequence([
            Fc(28*28, 128, activation=Rectifier()),
            Fc(128, 10, activation=Linear()),
        ])
    softmax = Softmax(with_loss=True)
    model.set_solver(SGD(learning_rate=0.001))

    print("Model :")
    print(model)

In order to provide data to the model for the training, we will create a :py:class:`n2d2.provider.DataProvider`.

.. code-block::

    provider = n2d2.provider.DataProvider(db, [28, 28, 1], batch_size=BATCH_SIZE)

    provider.set_partition("Learn")

    target = n2d2.target.Score(provider)

Then we can write a classic training loop to learn using the :py:class:`n2d2.provider.DataProvider` : 

.. code-block::

    print("\n### Training ###")
    for epoch in range(EPOCH):

        provider.set_partition("Learn")
        model.learn()

        print("\n# Train Epoch: " + str(epoch) + " #")

        for i in range(ceil(db.get_nb_stimuli('Learn')/BATCH_SIZE)):

            x = provider.read_random_batch()
            x = model(x)
            x = softmax(x)
            x = target(x)
            x.back_propagate()
            x.update()

            print("Example: " + str(i * BATCH_SIZE) + ", loss: "
                + "{0:.3f}".format(target.loss()), end='\r')

        print("\n### Validation ###")

        target.clear_success()
        
        provider.set_partition('Validation')
        model.test()

        for i in range(ceil(db.get_nb_stimuli('Validation')/BATCH_SIZE)):
            batch_idx = i * BATCH_SIZE

            x = provider.read_batch(batch_idx)
            x = model(x)
            x = softmax(x)
            x = target(x)

            print("Validate example: " + str(i * BATCH_SIZE) + ", val success: "
                + "{0:.2f}".format(100 * target.get_average_success()) + "%", end='\r')
    print("\nEND")
