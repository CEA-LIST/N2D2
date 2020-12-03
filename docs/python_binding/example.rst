Example
=======

.. testsetup:: *

   import N2D2
   import numpy
   path = "/nvme0/DATABASE/MNIST/raw/"
   batchSize = 1
   nb_epochs = 1
   epoch_size = 1


In this section we will create a simple convolutional neural network for the dataset `MNIST <https://en.wikipedia.org/wiki/MNIST_database>`_ using the python binding of N2D2.

Creation of the network
-----------------------

We first have to create an object Network. This object will be the backbone of the model, linking the different cells.

We have to begin with the initialisation of this object since it creates a seed that generates randomness.

.. testcode::
    
    net = N2D2.Network()
    deepNet = N2D2.DeepNet(net)

Importation of the dataset
--------------------------

To import the MNIST dataset we will use a custom class :py:class:`N2D2.MNIST_IDX_Database`.

.. testcode::
    
    database = N2D2.MNIST_IDX_Database()
    database.load(path)

In the following code, the *path* variable represent the path to the dataset MNIST.

Applying transformation to the dataset
--------------------------------------

We can create transformation using the class :py:class:`N2D2.Transformation`.

.. testcode::

    trans = N2D2.DistortionTransformation()
    trans.setParameter("ElasticGaussianSize", "21")
    trans.setParameter("ElasticSigma", "6.0")
    trans.setParameter("ElasticScaling", "36.0")
    trans.setParameter("Scaling", "10.0")
    trans.setParameter("Rotation", "10.0")
    padcrop = N2D2.PadCropTransformation(24, 24)

But to apply them to the data, we need :py:class:`N2D2.StimuliProvider`. 

:py:class:`N2D2.StimuliProvider` is a class that acts as a data loader for the neural network.

.. testcode::

    stimuli = N2D2.StimuliProvider(database, [24, 24, 1], batchSize, False)
    stimuli.addTransformation(N2D2.CompositeTransformation(N2D2.PadCropTransformation(24, 24)), database.StimuliSetMask(0))
    stimuli.addOnTheFlyTransformation(N2D2.CompositeTransformation(trans), database.StimuliSetMask(0))

We can apply transformation in two ways. 
The first one is the standard one, we apply the transformation once to the whole dataset.
This is useful for transformation like normalization or :py:class:`N2D2.PadCropTransformation`.
The other way is to add the transformation "on the fly", this mean that each time we load a data, we apply the transformation.
This is especially adapted to random transformation like :py:class:`N2D2.DistortionTransformation` since you add more diversity to the data.

You can note that we need to use :py:class:`N2D2.CompositeTransformation` to apply transformation with the :py:class:`N2D2.StimuliProvider`.

Defining network topology
-------------------------

To define our network topology, we use :py:class:`N2D2.Cell` objects. 

.. testcode::

    conv1 = N2D2.ConvCell_Frame_float(deepNet, "conv1", [4, 4], 16, [1, 1], [2, 2], [5, 5], [1, 1], N2D2.TanhActivation_Frame_float())
    conv2 = N2D2.ConvCell_Frame_float(deepNet, "conv2", [5, 5], 24, [1, 1], [2, 2], [5, 5], [1, 1], N2D2.TanhActivation_Frame_float())
    fc1 = N2D2.FcCell_Frame_float(deepNet, "fc1", 150, N2D2.TanhActivation_Frame_float())
    fc2 = N2D2.FcCell_Frame_float(deepNet, "fc2", 10, N2D2.TanhActivation_Frame_float())

Once the cells are created, we need to connect them.

.. testcode::

    conv2mapping = [
        True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, True, True,
        True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, True, True,
        False, True, True, False, False, False, False, False, False, False, False, False, False, False, False, True, True, False, False, False, False, False, True, True,
        False, False, True, True, False, False, False, False, False, False, False, False, False, False, False, True, True, False, False, False, False, False, True, True,
        False, False, False, True, True, False, False, False, False, False, False, False, False, False, False, False, True, True, False, False, False, False, True, True,
        False, False, False, False, True, True, False, False, False, False, False, False, False, False, False, False, True, True, False, False, False, False, True, True,
        False, False, False, False, False, True, True, False, False, False, False, False, False, False, False, False, False, True, True, False, False, False, True, True,
        False, False, False, False, False, False, True, True, False, False, False, False, False, False, False, False, False, True, True, False, False, False, True, True,
        False, False, False, False, False, False, False, True, True, False, False, False, False, False, False, False, False, False, True, True, False, False, True, True,
        False, False, False, False, False, False, False, False, True, True, False, False, False, False, False, False, False, False, True, True, False, False, True, True,
        False, False, False, False, False, False, False, False, False, True, True, False, False, False, False, False, False, False, False, True, True, False, True, True,
        False, False, False, False, False, False, False, False, False, False, True, True, False, False, False, False, False, False, False, True, True, False, True, True,
        False, False, False, False, False, False, False, False, False, False, False, True, True, False, False, False, False, False, False, False, True, True, True, True,
        False, False, False, False, False, False, False, False, False, False, False, False, True, True, False, False, False, False, False, False, True, True, True, True,
        False, False, False, False, False, False, False, False, False, False, False, False, False, True, True, False, False, False, False, False, False, True, True, True,
        False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, True, True, True]

    t_conv2mapping = N2D2.Tensor_bool(numpy.array(conv2mapping))

    conv1.addInput(stimuli)
    conv2.addInput(conv1, t_conv2mapping)
    fc1.addInput(conv2)
    fc2.addInput(fc1)

The first layer receive the :py:class:`N2D2.StimuliProvider` class as an input. The other layers have their input set with the previous cell.
In this example, we also create a different mapping for the :py:class:`N2D2.ConvCell_Frame_float` *conv2*.

Learning phase
--------------

Once the network is created, we can begin the learning phase. First, we need to create a :py:class:`N2D2.Target` object. This object defines the output of the network.

.. testcode::

    tar = N2D2.TargetScore('target', fc2, stimuli)

    conv1.initialize()
    conv2.initialize()
    fc1.initialize()
    fc2.initialize()

Finally, we can initiate the learning loop.

.. testcode::

    for epoch in range(nb_epochs):
        for i in range(epoch_size):
            stimuli.readRandomBatch(set=N2D2.Database.Learn)
            tar.provideTargets(N2D2.Database.Learn)
            conv1.propagate()
            conv2.propagate()
            fc1.propagate()
            fc2.propagate()
            tar.process(N2D2.Database.Learn)
            fc2.backPropagate()
            fc1.backPropagate()
            conv2.backPropagate()
            conv1.backPropagate()
            conv1.update()
            conv2.update()
            fc1.update()
            fc2.update()

