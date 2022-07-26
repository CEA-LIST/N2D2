Keras interoperability
======================

For this example, we will use an example provided in the Keras documentation : https://keras.io/examples/vision/mnist_convnet/

You can find the full python script here :download:`keras_example.py</../python/examples/keras_example.py>`.


Example
-------

We begin by importing the same library as in the example plus our interoperability library.

.. code-block:: python

        import numpy as np
        from tensorflow import keras
        from tensorflow.keras import layers
        # Importing the interoperability library
        import keras_to_n2d2

We then import the data by following the tutorial.

.. code-block:: python

        # training parameters
        batch_size = 128
        epochs = 10
        # Model / data parameters
        num_classes = 10
        input_shape = (28, 28, 1)

        # the data, split between train and test sets
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

        # Scale images to the [0, 1] range
        x_train = x_train.astype("float32") / 255
        x_test = x_test.astype("float32") / 255

        # Make sure images have shape (28, 28, 1)
        x_train = np.expand_dims(x_train, -1)
        x_test = np.expand_dims(x_test, -1)
        # convert class vectors to binary class matrices
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)

When declaring the model, we will use the :py:func:`keras_to_n2d2.wrap` function to generate an :py:class:`keras_to_n2d2.CustomSequential` which embedded N2D2.

.. code-block:: python

        tf_model = keras.Sequential([
                keras.Input(shape=input_shape),
                layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Flatten(),
                layers.Dense(num_classes, activation="softmax"),
        ])
        model = keras_to_n2d2.wrap(tf_model, batch_size=batch_size, for_export=True)

Once this is done, we can follow again the tutorial and run the training and the evaluation.

.. code-block:: python

        model.compile(loss="categorical_crossentropy", metrics=["accuracy"])

        model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
        score = model.evaluate(x_test, y_test, verbose=0)
        print("Test loss:", score[0])
        print("Test accuracy:", score[1])

And that is it ! You have successfully trained your model with N2D2 using the keras interface.

You can then retrieve the N2D2 model by using the method :py:meth:`keras_to_n2d2.CustomSequential.get_deepnet_cell` if you want to perform operation on it.

.. code-block:: python

        n2d2_model = model.get_deepnet_cell()