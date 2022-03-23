Interoperability
================

In this section, we will present how you can use n2d2 with other python framework. 

Keras *[experimental feature]*
------------------------------

Presentation
~~~~~~~~~~~~

The Keras interoperability allow you to train a model using the N2D2 backend with the TensorFlow/Keras frontend.

The interoperability consist of a wrapper around the N2D2 Network.

In order to integrate N2D2 into the Keras environment, we run TensorFlow in eager mode. 



Documentation
~~~~~~~~~~~~~

.. autofunction:: keras_interoperability.wrap

.. autoclass:: keras_interoperability.CustomSequential
        :members:

Changing the optimizer
^^^^^^^^^^^^^^^^^^^^^^

.. warning::
        Due to the implementation, n2d2 parameters are not visible to ``Keras`` and thus cannot be optimized by a ``Keras`` optimizer.

When compiling the :py:class:`keras_interoperability.CustomSequential`, you can pass an :py:class:`n2d2.solver.Solver` object to the parameter `optimizer`.
This will change the method used to optimize the parameters.

.. code-block:: python

        model.summary() # Use the default SGD solver. 
        model.compile(loss="categorical_crossentropy", optimizer=n2d2.solver.Adam(), metrics=["accuracy"])
        model.summary() # Use the newly defined Adam solver.


Example
~~~~~~~

For this example, we will use an example provided in the Keras documentation : https://keras.io/examples/vision/mnist_convnet/

We begin by importing the same library as in the example plus our interoperability library.

.. code-block:: python

        import numpy as np
        from tensorflow import keras
        from tensorflow.keras import layers
        # Importing the interoperability library
        import keras_interoperability

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

When declaring the model, we will use the :py:func:`keras_interoperability.wrap` function to generate an :py:class:`keras_interoperability.CustomSequential` which embedded N2D2.

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
        model = keras_interoperability.wrap(tf_model, batch_size=batch_size, for_export=True)

Once this is done, we can follow again the tutorial and run the training and the evaluation.

.. code-block:: python

        model.compile(loss="categorical_crossentropy", metrics=["accuracy"])

        model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
        score = model.evaluate(x_test, y_test, verbose=0)
        print("Test loss:", score[0])
        print("Test accuracy:", score[1])

And that is it ! You have successfully trained your model with N2D2 using the keras interface.

You can then retrieve the N2D2 model by using the method :py:meth:`keras_interoperability.CustomSequential.get_deepnet_cell` if you want to perform operation on it.

.. code-block:: python

        n2d2_model = model.get_deepnet_cell()

PyTorch *[experimental feature]*
--------------------------------

Presentation
~~~~~~~~~~~~

The PyTorch interoperability allow you to run an n2d2 model by using the Torch functions.

The interoperability consist of a wrapper around the N2D2 Network.
We created an autograd function which on ``Forward`` call the n2d2 ``Propagate`` method and on ``Backward`` call the n2d2 ``Back Propagate`` and ``Update`` methods.

.. figure:: ../_static/torch_interop.png
   :alt: schematic of the interoperability

.. warning::
        Due to the implementation n2d2 parameters are not visible to ``Torch`` and thus cannot be trained with a torch ``Optimizer``.

Tensor conversion
~~~~~~~~~~~~~~~~~ 

In order to achieve this interoperability, we need to convert Tensor from ``Torch`` to ``n2d2`` and vice versa.

:py:class:`n2d2.Tensor` require a contiguous memory space which is not the case for ``Torch``. Thus the conversion ``Torch`` to ``n2d2`` require a memory copy.
The opposite conversion is done with no memory copy.

If you work with ``CUDA`` tensor, the conversion ``Torch`` to ``n2d2`` is also done with no copy on the GPU (a copy on the host is however required).


Documentation
~~~~~~~~~~~~~

.. autofunction:: pytorch_interoperability.wrap

.. autoclass:: pytorch_interoperability.Block
        :members:

Example
~~~~~~~

In this example, we will follow the Torch tutorial : https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html.
And run the network with N2D2 instead of Torch.

Firstly, we import the same libraries as in the tutorial plus our ``pytorch_interoperability`` and ``n2d2`` libraries.

.. code-block:: python

        import torch
        import torchvision
        import torchvision.transforms as transforms
        import matplotlib.pyplot as plt
        import numpy as np
        import torch.nn as nn
        import torch.nn.functional as F
        import torch.optim as optim
        import n2d2
        import pytorch_interoperability


We then still follow the tutorial and add the code to load the data and we define the Network.

.. code-block:: python

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


        transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        batch_size = 4

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                shuffle=False, num_workers=2)

        classes = ('plane', 'car', 'bird', 'cat',
                'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



        # functions to show an image


        def imshow(img, img_path):
                img = img / 2 + 0.5     # unnormalize
                cpu_img = img.cpu()
                npimg = cpu_img.numpy()
                plt.imshow(np.transpose(npimg, (1, 2, 0)))
                plt.savefig(img_path)


        class Net(nn.Module):
        def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 6, 5)
                self.pool = nn.MaxPool2d(2, 2)
                self.conv2 = nn.Conv2d(6, 16, 5)
                self.fc1 = nn.Linear(16 * 5 * 5, 120)
                self.fc2 = nn.Linear(120, 84)
                self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
                x = self.pool(F.relu(self.conv1(x)))
                x = self.pool(F.relu(self.conv2(x)))
                x = torch.flatten(x, 1) # flatten all dimensions except batch
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                x = self.fc3(x)
                return x

Here we begin to add our code, we intialize the Torch Network and we pass it to the :py:func:`pytorch_interoperability.wrap` method.
This will give us a ``torch.nn.Module`` which run N2D2 and that we will use instead of the Torch Network.

.. code-block:: python

        torch_net = Net()
        # specify that we want to use CUDA.
        n2d2.global_variables.default_model = "Frame_CUDA" 
        # creating a model which run with N2D2 backend.
        net = pytorch_interoperability.wrap(torch_net, (batch_size, 3, 32, 32)) 

        criterion = nn.CrossEntropyLoss()
        # Reminder : We define an optimizer, but it will not be used to optimized N2D2 parameters.
        # If you want to change the optimizer of N2D2 refer to the N2D2 solver.
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

And that is it ! From this point, we can follow again the tutorial provided by PyTorch and we have a script ready to run.
You can compare the N2D2 and the torch version by commenting the code we added and renaming ``torch_net`` into ``net``.

.. code-block:: python

        for epoch in range(2):  # loop over the dataset multiple times
        e_t = time()
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data[0].to(device), data[1].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                        (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
        print(f"Expoch {epoch} : {time()-e_t}")
        print('Finished Training')

        dataiter = iter(testloader)
        images, labels = dataiter.next()
        images = images.to(device)
        labels = labels.to(device)
        # print images
        imshow(torchvision.utils.make_grid(images), "torch_inference.png")
        print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
        outputs = net(images)

        _, predicted = torch.max(outputs, 1)

        print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                                for j in range(4)))
