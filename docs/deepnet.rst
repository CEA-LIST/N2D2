DeepNet
=======

In order to create a neural network in N2D2 using an INI file, you can use the
DeepNetGenerator:

.. testsetup:: *

   import numpy
   import N2D2

   N2D2.CudaContext.setDevice(1)

.. testcode::

   net = N2D2.Network(seed=1)
   deepNet = N2D2.DeepNetGenerator.generate(net, "../models/mnist24_16c4s2_24c5s2_150_10.ini")

Before executing the model, the network must first be initialized:

.. testcode::

   deepNet.initialize()

In order to test the first batch sample from the dataset, we retrieve the 
StimuliProvider and read the first batch from the test set:

.. testcode::

   sp = deepNet.getStimuliProvider()
   sp.readBatch(N2D2.Database.Test, 0)

We can now run the network on this data:

.. testcode::

   deepNet.test(N2D2.Database.Test, [])

Finally, in order to retrieve the estimated outputs, one has to retrieve the
first and unique target of the model and get the estimated labels and values:

.. testcode::

   target = deepNet.getTargets()[0]
   labels = numpy.array(target.getEstimatedLabels()).flatten()
   values = numpy.array(target.getEstimatedLabelsValue()).flatten()
   results = list(zip(labels, values))

   print(results)

.. testoutput::

   [(1, 0.15989691), (1, 0.1617092), (9, 0.14962792), (9, 0.16899541), (1, 0.16261548), (1, 0.17289816), (1, 0.13728766), (1, 0.15315214), (1, 0.14424478), (9, 0.17937173), (9, 0.1518211), (1, 0.12860793), (9, 0.17310674), (9, 0.14563303), (1, 0.1782302), (9, 0.14206158), (1, 0.18292117), (9, 0.14831853), (1, 0.2224524), (9, 0.1745578), (1, 0.20414244), (1, 0.26987872), (1, 0.16570412), (9, 0.17435187)]

.. autoclass:: N2D2.DeepNet
   :members: