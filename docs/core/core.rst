Core N2D2
=========

Introduction
------------

In this section we will present the C++ core function that are binded to Python with the framework pybind.
The binding of the C++ core is straightforward, thus this section can also be seen as a documentation of the C++ core implementation of N2D2. 

If you want to use the raw python binding, you need to compile N2D2. This  will create a '.so' file in the `lib` folder. 
If you want to use the raw binding, you will need to have this file at the root of your project or in your `PYTHONPATH`.

You can then access the raw binding by importing N2D2 in your python script with the line `import N2D2`.
It is however not recommended to use the raw binding, you should instead use the :doc:`n2d2 python library<../python_api/intro>`. 


DeepNet
-------

Introduction
~~~~~~~~~~~~

In order to create a neural network in N2D2 using an INI file, you can use the
DeepNetGenerator:

.. testsetup::

   import numpy
   import pyn2d2 as N2D2

   N2D2.CudaContext.setDevice(1)

.. testcode::

   net - N2D2.Network(seed-1)
   deepNet - N2D2.DeepNetGenerator.generate(net, "../models/mnist24_16c4s2_24c5s2_150_10.ini")

Before executing the model, the network must first be initialized:

.. testcode::

   deepNet.initialize()

In order to test the first batch sample from the dataset, we retrieve the 
StimuliProvider and read the first batch from the test set:

.. testcode::

   sp - deepNet.getStimuliProvider()
   sp.readBatch(N2D2.Database.Test, 0)

We can now run the network on this data:

.. testcode::

   deepNet.test(N2D2.Database.Test, [])

Finally, in order to retrieve the estimated outputs, one has to retrieve the
first and unique target of the model and get the estimated labels and values:

.. testcode::

   target - deepNet.getTargets()[0]
   labels - numpy.array(target.getEstimatedLabels()).flatten()
   values - numpy.array(target.getEstimatedLabelsValue()).flatten()
   results - list(zip(labels, values))

   print(results)

.. testoutput::

   [(1, 0.15989691), (1, 0.1617092), (9, 0.14962792), (9, 0.16899541), (1, 0.16261548), (1, 0.17289816), (1, 0.13728766), (1, 0.15315214), (1, 0.14424478), (9, 0.17937174), (9, 0.1518211), (1, 0.12860791), (9, 0.17310674), (9, 0.14563303), (1, 0.17823018), (9, 0.14206158), (1, 0.18292117), (9, 0.14831856), (1, 0.22245243), (9, 0.1745578), (1, 0.20414244), (1, 0.26987872), (1, 0.16570412), (9, 0.17435187)]


API Reference
~~~~~~~~~~~~~

.. autoclass:: N2D2.DeepNet
   :members:
 

Cells
-----

Cell
~~~~

AnchorCell
^^^^^^^^^^

.. autoclass:: N2D2.AnchorCell
	:members:
	:inherited-members:

BatchNormCell
^^^^^^^^^^^^^

.. autoclass:: N2D2.BatchNormCell
	:members:
	:inherited-members:

Cell
^^^^

.. autoclass:: N2D2.Cell
	:members:
	:inherited-members:

ConvCell
^^^^^^^^

.. autoclass:: N2D2.ConvCell
	:members:
	:inherited-members:

DeconvCell
^^^^^^^^^^

.. autoclass:: N2D2.DeconvCell
	:members:
	:inherited-members:

DropoutCell
^^^^^^^^^^^

.. autoclass:: N2D2.DropoutCell
	:members:
	:inherited-members:

ElemWiseCell
^^^^^^^^^^^^

.. autoclass:: N2D2.ElemWiseCell
	:members:
	:inherited-members:

FMPCell
^^^^^^^

.. autoclass:: N2D2.FMPCell
	:members:
	:inherited-members:

FcCell
^^^^^^

.. autoclass:: N2D2.FcCell
	:members:
	:inherited-members:

LRNCell
^^^^^^^

.. autoclass:: N2D2.LRNCell
	:members:
	:inherited-members:

LSTMCell
^^^^^^^^

.. autoclass:: N2D2.LSTMCell
	:members:
	:inherited-members:

NormalizeCell
^^^^^^^^^^^^^

.. autoclass:: N2D2.NormalizeCell
	:members:
	:inherited-members:

ObjectDetCell
^^^^^^^^^^^^^

.. autoclass:: N2D2.ObjectDetCell
	:members:
	:inherited-members:

PaddingCell
^^^^^^^^^^^

.. autoclass:: N2D2.PaddingCell
	:members:
	:inherited-members:

PoolCell
^^^^^^^^

.. autoclass:: N2D2.PoolCell
	:members:
	:inherited-members:

ProposalCell
^^^^^^^^^^^^

.. autoclass:: N2D2.ProposalCell
	:members:
	:inherited-members:

ROIPoolingCell
^^^^^^^^^^^^^^

.. autoclass:: N2D2.ROIPoolingCell
	:members:
	:inherited-members:

RPCell
^^^^^^

.. autoclass:: N2D2.RPCell
	:members:
	:inherited-members:

ResizeCell
^^^^^^^^^^

.. autoclass:: N2D2.ResizeCell
	:members:
	:inherited-members:

ScalingCell
^^^^^^^^^^^

.. autoclass:: N2D2.ScalingCell
	:members:
	:inherited-members:

SoftmaxCell
^^^^^^^^^^^

.. autoclass:: N2D2.SoftmaxCell
	:members:
	:inherited-members:

TargetBiasCell
^^^^^^^^^^^^^^

.. autoclass:: N2D2.TargetBiasCell
	:members:
	:inherited-members:

ThresholdCell
^^^^^^^^^^^^^

.. autoclass:: N2D2.ThresholdCell
	:members:
	:inherited-members:

TransformationCell
^^^^^^^^^^^^^^^^^^

.. autoclass:: N2D2.TransformationCell
	:members:
	:inherited-members:

UnpoolCell
^^^^^^^^^^

.. autoclass:: N2D2.UnpoolCell
	:members:
	:inherited-members:


Frame
~~~~~

AnchorCell_Frame
^^^^^^^^^^^^^^^^

.. autoclass:: N2D2.AnchorCell_Frame
	:members:
	:inherited-members:

AnchorCell_Frame_CUDA
^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: N2D2.AnchorCell_Frame_CUDA
	:members:
	:inherited-members:

BatchNormCell_Frame_float
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: N2D2.BatchNormCell_Frame_float
	:members:
	:inherited-members:

BatchNormCell_Frame_double
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: N2D2.BatchNormCell_Frame_double
	:members:
	:inherited-members:

BatchNormCell_Frame_CUDA_float
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: N2D2.BatchNormCell_Frame_CUDA_float
	:members:
	:inherited-members:

BatchNormCell_Frame_CUDA_double
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: N2D2.BatchNormCell_Frame_CUDA_double
	:members:
	:inherited-members:

Cell_Frame_float
^^^^^^^^^^^^^^^^

.. autoclass:: N2D2.Cell_Frame_float
	:members:
	:inherited-members:

Cell_Frame_double
^^^^^^^^^^^^^^^^^

.. autoclass:: N2D2.Cell_Frame_double
	:members:
	:inherited-members:

Cell_Frame_CUDA_float
^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: N2D2.Cell_Frame_CUDA_float
	:members:
	:inherited-members:

Cell_Frame_CUDA_double
^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: N2D2.Cell_Frame_CUDA_double
	:members:
	:inherited-members:

Cell_Frame_Top
^^^^^^^^^^^^^^

.. autoclass:: N2D2.Cell_Frame_Top
	:members:
	:inherited-members:

ConvCell_Frame_float
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: N2D2.ConvCell_Frame_float
	:members:
	:inherited-members:

ConvCell_Frame_double
^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: N2D2.ConvCell_Frame_double
	:members:
	:inherited-members:

ConvCell_Frame_CUDA_float
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: N2D2.ConvCell_Frame_CUDA_float
	:members:
	:inherited-members:

ConvCell_Frame_CUDA_double
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: N2D2.ConvCell_Frame_CUDA_double
	:members:
	:inherited-members:

DeconvCell_Frame_float
^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: N2D2.DeconvCell_Frame_float
	:members:
	:inherited-members:

DeconvCell_Frame_double
^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: N2D2.DeconvCell_Frame_double
	:members:
	:inherited-members:

DeconvCell_Frame_CUDA_float
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: N2D2.DeconvCell_Frame_CUDA_float
	:members:
	:inherited-members:

DeconvCell_Frame_CUDA_double
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: N2D2.DeconvCell_Frame_CUDA_double
	:members:
	:inherited-members:

DropoutCell_Frame_float
^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: N2D2.DropoutCell_Frame_float
	:members:
	:inherited-members:

DropoutCell_Frame_double
^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: N2D2.DropoutCell_Frame_double
	:members:
	:inherited-members:

DropoutCell_Frame_CUDA_float
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: N2D2.DropoutCell_Frame_CUDA_float
	:members:
	:inherited-members:

DropoutCell_Frame_CUDA_double
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: N2D2.DropoutCell_Frame_CUDA_double
	:members:
	:inherited-members:

ElemWiseCell_Frame
^^^^^^^^^^^^^^^^^^

.. autoclass:: N2D2.ElemWiseCell_Frame
	:members:
	:inherited-members:

ElemWiseCell_Frame_CUDA
^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: N2D2.ElemWiseCell_Frame_CUDA
	:members:
	:inherited-members:

FMPCell_Frame
^^^^^^^^^^^^^

.. autoclass:: N2D2.FMPCell_Frame
	:members:
	:inherited-members:

FMPCell_Frame_CUDA
^^^^^^^^^^^^^^^^^^

.. autoclass:: N2D2.FMPCell_Frame_CUDA
	:members:
	:inherited-members:

FcCell_Frame_float
^^^^^^^^^^^^^^^^^^

.. autoclass:: N2D2.FcCell_Frame_float
	:members:
	:inherited-members:

FcCell_Frame_double
^^^^^^^^^^^^^^^^^^^

.. autoclass:: N2D2.FcCell_Frame_double
	:members:
	:inherited-members:

FcCell_Frame_CUDA_float
^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: N2D2.FcCell_Frame_CUDA_float
	:members:
	:inherited-members:

FcCell_Frame_CUDA_double
^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: N2D2.FcCell_Frame_CUDA_double
	:members:
	:inherited-members:

LRNCell_Frame_float
^^^^^^^^^^^^^^^^^^^

.. autoclass:: N2D2.LRNCell_Frame_float
	:members:
	:inherited-members:

LRNCell_Frame_double
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: N2D2.LRNCell_Frame_double
	:members:
	:inherited-members:

LRNCell_Frame_CUDA_float
^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: N2D2.LRNCell_Frame_CUDA_float
	:members:
	:inherited-members:

LRNCell_Frame_CUDA_double
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: N2D2.LRNCell_Frame_CUDA_double
	:members:
	:inherited-members:

LSTMCell_Frame_CUDA_float
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: N2D2.LSTMCell_Frame_CUDA_float
	:members:
	:inherited-members:

LSTMCell_Frame_CUDA_double
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: N2D2.LSTMCell_Frame_CUDA_double
	:members:
	:inherited-members:

NormalizeCell_Frame_float
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: N2D2.NormalizeCell_Frame_float
	:members:
	:inherited-members:

NormalizeCell_Frame_double
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: N2D2.NormalizeCell_Frame_double
	:members:
	:inherited-members:

NormalizeCell_Frame_CUDA_float
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: N2D2.NormalizeCell_Frame_CUDA_float
	:members:
	:inherited-members:

NormalizeCell_Frame_CUDA_double
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: N2D2.NormalizeCell_Frame_CUDA_double
	:members:
	:inherited-members:

ObjectDetCell_Frame
^^^^^^^^^^^^^^^^^^^

.. autoclass:: N2D2.ObjectDetCell_Frame
	:members:
	:inherited-members:

ObjectDetCell_Frame_CUDA
^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: N2D2.ObjectDetCell_Frame_CUDA
	:members:
	:inherited-members:

PaddingCell_Frame
^^^^^^^^^^^^^^^^^

.. autoclass:: N2D2.PaddingCell_Frame
	:members:
	:inherited-members:

PaddingCell_Frame_CUDA
^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: N2D2.PaddingCell_Frame_CUDA
	:members:
	:inherited-members:

PoolCell_Frame_float
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: N2D2.PoolCell_Frame_float
	:members:
	:inherited-members:

PoolCell_Frame_double
^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: N2D2.PoolCell_Frame_double
	:members:
	:inherited-members:

PoolCell_Frame_CUDA_float
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: N2D2.PoolCell_Frame_CUDA_float
	:members:
	:inherited-members:

PoolCell_Frame_CUDA_double
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: N2D2.PoolCell_Frame_CUDA_double
	:members:
	:inherited-members:

PoolCell_Frame_EXT_CUDA_float
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: N2D2.PoolCell_Frame_EXT_CUDA_float
	:members:
	:inherited-members:

PoolCell_Frame_EXT_CUDA_double
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: N2D2.PoolCell_Frame_EXT_CUDA_double
	:members:
	:inherited-members:

ProposalCell_Frame
^^^^^^^^^^^^^^^^^^

.. autoclass:: N2D2.ProposalCell_Frame
	:members:
	:inherited-members:

ProposalCell_Frame_CUDA
^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: N2D2.ProposalCell_Frame_CUDA
	:members:
	:inherited-members:

ROIPoolingCell_Frame
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: N2D2.ROIPoolingCell_Frame
	:members:
	:inherited-members:

ROIPoolingCell_Frame_CUDA
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: N2D2.ROIPoolingCell_Frame_CUDA
	:members:
	:inherited-members:

RPCell_Frame
^^^^^^^^^^^^

.. autoclass:: N2D2.RPCell_Frame
	:members:
	:inherited-members:

RPCell_Frame_CUDA
^^^^^^^^^^^^^^^^^

.. autoclass:: N2D2.RPCell_Frame_CUDA
	:members:
	:inherited-members:

ResizeCell_Frame
^^^^^^^^^^^^^^^^

.. autoclass:: N2D2.ResizeCell_Frame
	:members:
	:inherited-members:

ResizeCell_Frame_CUDA
^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: N2D2.ResizeCell_Frame_CUDA
	:members:
	:inherited-members:

ScalingCell_Frame_float
^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: N2D2.ScalingCell_Frame_float
	:members:
	:inherited-members:

ScalingCell_Frame_double
^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: N2D2.ScalingCell_Frame_double
	:members:
	:inherited-members:

ScalingCell_Frame_CUDA_float
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: N2D2.ScalingCell_Frame_CUDA_float
	:members:
	:inherited-members:

ScalingCell_Frame_CUDA_double
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: N2D2.ScalingCell_Frame_CUDA_double
	:members:
	:inherited-members:

SoftmaxCell_Frame_float
^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: N2D2.SoftmaxCell_Frame_float
	:members:
	:inherited-members:

SoftmaxCell_Frame_double
^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: N2D2.SoftmaxCell_Frame_double
	:members:
	:inherited-members:

SoftmaxCell_Frame_CUDA_float
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: N2D2.SoftmaxCell_Frame_CUDA_float
	:members:
	:inherited-members:

SoftmaxCell_Frame_CUDA_double
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: N2D2.SoftmaxCell_Frame_CUDA_double
	:members:
	:inherited-members:

TargetBiasCell_Frame_float
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: N2D2.TargetBiasCell_Frame_float
	:members:
	:inherited-members:

TargetBiasCell_Frame_double
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: N2D2.TargetBiasCell_Frame_double
	:members:
	:inherited-members:

TargetBiasCell_Frame_CUDA_float
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: N2D2.TargetBiasCell_Frame_CUDA_float
	:members:
	:inherited-members:

TargetBiasCell_Frame_CUDA_double
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: N2D2.TargetBiasCell_Frame_CUDA_double
	:members:
	:inherited-members:

ThresholdCell_Frame
^^^^^^^^^^^^^^^^^^^

.. autoclass:: N2D2.ThresholdCell_Frame
	:members:
	:inherited-members:

ThresholdCell_Frame_CUDA
^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: N2D2.ThresholdCell_Frame_CUDA
	:members:
	:inherited-members:

TransformationCell_Frame
^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: N2D2.TransformationCell_Frame
	:members:
	:inherited-members:

TransformationCell_Frame_CUDA
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: N2D2.TransformationCell_Frame_CUDA
	:members:
	:inherited-members:

UnpoolCell_Frame
^^^^^^^^^^^^^^^^

.. autoclass:: N2D2.UnpoolCell_Frame
	:members:
	:inherited-members:

UnpoolCell_Frame_CUDA
^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: N2D2.UnpoolCell_Frame_CUDA
	:members:
	:inherited-members:


Filler
------

.. autoclass:: N2D2.Filler
   :members:

Activation
----------

Introduction
~~~~~~~~~~~~

Activation functions in N2D2 are passed as arguments to initialize :py:class:`N2D2.Cell`.

.. testsetup:: *

   import N2D2

.. testcode::

   tanh - N2D2.TanhActivation_Frame_float()

Activation
~~~~~~~~~~

Activation
^^^^^^^^^^

.. autoclass:: N2D2.Activation
        :members:

LinearActivation
^^^^^^^^^^^^^^^^

.. autoclass:: N2D2.LinearActivation
        :members:

RectifierActivation
^^^^^^^^^^^^^^^^^^^

.. autoclass:: N2D2.RectifierActivation
        :members:

TanhActivation
^^^^^^^^^^^^^^

.. autoclass:: N2D2.TanhActivation
        :members:



SwishActivation
^^^^^^^^^^^^^^^

.. autoclass:: N2D2.SwishActivation
        :members:

SaturationActivation
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: N2D2.SaturationActivation
        :members:

LogisticActivation
^^^^^^^^^^^^^^^^^^

.. autoclass:: N2D2.LogisticActivation
        :members:

SoftplusActivation
^^^^^^^^^^^^^^^^^^

.. autoclass:: N2D2.SoftplusActivation
        :members:

Activation_Frame
~~~~~~~~~~~~~~~~

LinearActivation_Frame
^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: N2D2.LinearActivation_Frame_float
        :members:
.. autoclass:: N2D2.LinearActivation_Frame_double
        :members:
.. autoclass:: N2D2.LinearActivation_Frame_CUDA_float
        :members:
.. autoclass:: N2D2.LinearActivation_Frame_CUDA_double
        :members:

RectifierActivation_Frame
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: N2D2.RectifierActivation_Frame_float
        :members:
.. autoclass:: N2D2.RectifierActivation_Frame_double
        :members:
.. autoclass:: N2D2.RectifierActivation_Frame_CUDA_float
        :members:
.. autoclass:: N2D2.RectifierActivation_Frame_CUDA_double
        :members:

TanhActivation_Frame
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: N2D2.TanhActivation_Frame_float
        :members:
.. autoclass:: N2D2.TanhActivation_Frame_double
        :members:
.. autoclass:: N2D2.TanhActivation_Frame_CUDA_float
        :members:
.. autoclass:: N2D2.TanhActivation_Frame_CUDA_double
        :members:

SwishActivation_Frame
^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: N2D2.SwishActivation_Frame_float
        :members:
.. autoclass:: N2D2.SwishActivation_Frame_double
        :members:
.. autoclass:: N2D2.SwishActivation_Frame_CUDA_float
        :members:
.. autoclass:: N2D2.SwishActivation_Frame_CUDA_double
        :members:

Solver
------

.. autoclass:: N2D2.Solver
   :members:

Target
------

Introduction
~~~~~~~~~~~~

A :py:class:`N2D2.Target` is associated to a :py:class:`N2D2.Cell`, it define the output of the network.
The computation of the loss and other tools to compute score such as the confusion matrix are also computed with this class. 

To train a neural network you need to use :py:meth:`N2D2.Target.provideTargets` then to :py:meth:`N2D2.cell.propagate` then :py:meth:`N2D2.Target.process` and finally :py:meth:`N2D2.Cell.backpropagate`.
(See :doc:`the MNIST example<./example>`.)

.. autoclass:: N2D2.Target
   :members:

Databases
---------

.. testsetup:: *

   import N2D2
   path - "/nvme0/DATABASE/MNIST/raw/"


Introduction: 
~~~~~~~~~~~~~

N2D2 allow you to import default dataset or to load your own dataset. 
This can be done suing Database objects.

Download datasets:
~~~~~~~~~~~~~~~~~~

To import Data you can use a python Script situated in ``./tools/install_stimuli_gui.py``.

This script will download the data in ``/local/$USER/n2d2_data/``. 
You can change this path with the environment variable ``N2D2_data``.

Once the dataset downloaded, you can load it with the appropriate class. 
Here is an example of the loading of the MNIST dataset :

.. testcode::
    
    database - N2D2.MNIST_IDX_Database()
    database.load(path)

In this example, the data are located in the folder **path**.


Database:
~~~~~~~~~

Database
^^^^^^^^

.. autoclass:: N2D2.Database
        :members:
        :inherited-members:

MNIST_IDX_Database
^^^^^^^^^^^^^^^^^^

.. autoclass:: N2D2.MNIST_IDX_Database
        :members:
        :inherited-members:

Actitracker_Database
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: N2D2.Actitracker_Database
        :members:
        :inherited-members:

AER_Database
^^^^^^^^^^^^

.. autoclass:: N2D2.AER_Database
        :members:
        :inherited-members:

Caltech101_DIR_Database
^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: N2D2.Caltech101_DIR_Database
        :members:
        :inherited-members:

Caltech256_DIR_Database
^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: N2D2.Caltech256_DIR_Database
        :members:
        :inherited-members:

CaltechPedestrian_Database
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: N2D2.CaltechPedestrian_Database
        :members:
        :inherited-members:

CelebA_Database
^^^^^^^^^^^^^^^

.. autoclass:: N2D2.CelebA_Database
        :members:
        :inherited-members:

CIFAR_Database
^^^^^^^^^^^^^^

.. autoclass:: N2D2.CIFAR_Database
        :members:
        :inherited-members:

CKP_Database
^^^^^^^^^^^^

.. autoclass:: N2D2.CKP_Database
        :members:
        :inherited-members:

DIR_Database
^^^^^^^^^^^^

.. autoclass:: N2D2.DIR_Database
        :members:
        :inherited-members:

GTSRB_DIR_Database
^^^^^^^^^^^^^^^^^^

.. autoclass:: N2D2.GTSRB_DIR_Database
        :members:
        :inherited-members:

GTSDB_DIR_Database
^^^^^^^^^^^^^^^^^^

.. autoclass:: N2D2.GTSDB_DIR_Database
        :members:
        :inherited-members:

ILSVRC2012_Database
^^^^^^^^^^^^^^^^^^^

.. autoclass:: N2D2.ILSVRC2012_Database
        :members:
        :inherited-members:

IDX_Database
^^^^^^^^^^^^

.. autoclass:: N2D2.IDX_Database
        :members:
        :inherited-members:

IMDBWIKI_Database
^^^^^^^^^^^^^^^^^

.. autoclass:: N2D2.IMDBWIKI_Database
        :members:
        :inherited-members:

KITTI_Database
^^^^^^^^^^^^^^

.. autoclass:: N2D2.KITTI_Database
        :members:
        :inherited-members:

KITTI_Object_Database
^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: N2D2.KITTI_Object_Database
        :members:
        :inherited-members:

KITTI_Road_Database
^^^^^^^^^^^^^^^^^^^

.. autoclass:: N2D2.KITTI_Road_Database
        :members:
        :inherited-members:

LITISRouen_Database
^^^^^^^^^^^^^^^^^^^

.. autoclass:: N2D2.LITISRouen_Database
        :members:
        :inherited-members:

N_MNIST_Database
^^^^^^^^^^^^^^^^

.. autoclass:: N2D2.N_MNIST_Database
        :members:
        :inherited-members:

DOTA_Database
^^^^^^^^^^^^^

.. autoclass:: N2D2.DOTA_Database
        :members:
        :inherited-members:

Fashion_MNIST_IDX_Database
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: N2D2.Fashion_MNIST_IDX_Database
        :members:
        :inherited-members:

FDDB_Database
^^^^^^^^^^^^^

.. autoclass:: N2D2.FDDB_Database
        :members:
        :inherited-members:

Daimler_Database
^^^^^^^^^^^^^^^^

.. autoclass:: N2D2.Daimler_Database
        :members:
        :inherited-members:

StimuliProvider
---------------

.. autoclass:: N2D2.StimuliProvider
   :members:



Transformation
--------------

Introduction
~~~~~~~~~~~~

In order to apply transformation to a dataset, we use the transformation object.

.. testsetup:: *

   import N2D2

Creation of different Transformation object.

.. testcode::

    dist - N2D2.DistortionTransformation()
    dist.setParameter("ElasticGaussianSize", "21")
    dist.setParameter("ElasticSigma", "6.0")
    dist.setParameter("ElasticScaling", "36.0")
    dist.setParameter("Scaling", "10.0")
    dist.setParameter("Rotation", "10.0")

    padcrop - N2D2.PadCropTransformation(24, 24)

    ct - N2D2.CompositeTransformation(padcrop)
    ct.push_back(dist)

To apply Transformation to a dataset, we use an object :py:class:`N2D2.StimuliProvider` which acts as a data loader.

Transformations
~~~~~~~~~~~~~~~

Transformation
^^^^^^^^^^^^^^^

.. autoclass:: N2D2.Transformation
        :members:

DistortionTransformation
^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: N2D2.DistortionTransformation
        :members:

PadCropTransformation
^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: N2D2.PadCropTransformation
        :members:

CompositeTransformation
^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: N2D2.CompositeTransformation
        :members:

AffineTransformation
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: N2D2.AffineTransformation
        :members:

ChannelExtractionTransformation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: N2D2.ChannelExtractionTransformation
        :members:

ColorSpaceTransformation
^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: N2D2.ColorSpaceTransformation
        :members:

CompressionNoiseTransformation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: N2D2.CompressionNoiseTransformation
        :members:

DCTTransformation
^^^^^^^^^^^^^^^^^

.. autoclass:: N2D2.DCTTransformation
        :members:

DFTTransformation
^^^^^^^^^^^^^^^^^

.. autoclass:: N2D2.DFTTransformation
        :members:

EqualizeTransformation
^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: N2D2.EqualizeTransformation
        :members:

ExpandLabelTransformation
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: N2D2.ExpandLabelTransformation
        :members:

WallisFilterTransformation
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: N2D2.WallisFilterTransformation
        :members:

ThresholdTransformation
^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: N2D2.ThresholdTransformation
        :members:

SliceExtractionTransformation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: N2D2.SliceExtractionTransformation
        :members:

ReshapeTransformation
^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: N2D2.ReshapeTransformation
        :members:

RescaleTransformation
^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: N2D2.RescaleTransformation
        :members:

RangeClippingTransformation
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: N2D2.RangeClippingTransformation
        :members:

RangeAffineTransformation
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: N2D2.RangeAffineTransformation
        :members:

RandomAffineTransformation
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: N2D2.RandomAffineTransformation
        :members:

NormalizeTransformation
^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: N2D2.NormalizeTransformation
        :members:

MorphologyTransformation
^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: N2D2.MorphologyTransformation
        :members:

MorphologicalReconstructionTransformation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: N2D2.MorphologicalReconstructionTransformation
        :members:

MagnitudePhaseTransformation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: N2D2.MagnitudePhaseTransformation
        :members:

LabelSliceExtractionTransformation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: N2D2.LabelSliceExtractionTransformation
        :members:

LabelExtractionTransformation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: N2D2.LabelExtractionTransformation
        :members:

GradientFilterTransformation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: N2D2.GradientFilterTransformation
        :members:

ApodizationTransformation
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: N2D2.ApodizationTransformation
        :members:

FilterTransformation
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: N2D2.FilterTransformation
        :members:

FlipTransformation
^^^^^^^^^^^^^^^^^^

.. autoclass:: N2D2.FlipTransformation
        :members:

Containers
----------

Introduction
~~~~~~~~~~~~

N2D2 has his own Tensor implementation. 

.. testsetup:: *

   import N2D2
   import numpy
   

.. testcode::

   N2D2.Tensor_float([1, 2, 3])


Tensor can be also be created using numpy.array object.

.. testcode::

   N2D2.CudaTensor_float(numpy.array([[1.0, 2.0], [3.0, 4.0]]))

   
Tensor
~~~~~~

.. autoclass:: N2D2.BaseTensor
   :members:

.. autoclass:: N2D2.Tensor_float
   :members:

.. autoclass:: N2D2.Tensor_double
   :members:

.. autoclass:: N2D2.Tensor_bool
   :members:

CudaTensor
~~~~~~~~~~

.. autoclass:: N2D2.CudaBaseDeviceTensor
   :members:

.. autoclass:: N2D2.CudaBaseTensor
   :members:

.. autoclass:: N2D2.CudaTensor_float
   :members:

.. autoclass:: N2D2.CudaTensor_double
   :members:
s