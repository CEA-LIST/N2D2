Transformation
==============

Introduction
------------

In order to apply transformation to a dataset, we use the transformation object.

.. testsetup:: *

   import N2D2

Creation of different Transformation object.

.. testcode::

    dist = N2D2.DistortionTransformation()
    dist.setParameter("ElasticGaussianSize", "21")
    dist.setParameter("ElasticSigma", "6.0")
    dist.setParameter("ElasticScaling", "36.0")
    dist.setParameter("Scaling", "10.0")
    dist.setParameter("Rotation", "10.0")

    padcrop = N2D2.PadCropTransformation(24, 24)

    ct = N2D2.CompositeTransformation(padcrop)
    ct.push_back(dist)

To apply Transformation to a dataset, we used an object :py:class:`N2D2.StimuliProvider` which acts as a data loader.

.. testcode::

   N2D2.mtSeed(0)
   database = N2D2.MNIST_IDX_Database()
   database.load("/nvme0/DATABASE/MNIST/raw/")
   stimuli = N2D2.StimuliProvider(database, [24, 24, 1], 10, False)
   stimuli.addTransformation(ct, database.StimuliSetMask(0))

Transformation:
---------------

.. autoclass:: N2D2.Transformation
        :members:

DistortionTransformation:
-------------------------

.. autoclass:: N2D2.DistortionTransformation
        :members:


PadCropTransformation:
----------------------

.. autoclass:: N2D2.PadCropTransformation
        :members:

CompositeTransformation:
------------------------

.. autoclass:: N2D2.CompositeTransformation
        :members:

AffineTransformation:
---------------------

.. autoclass:: N2D2.AffineTransformation
        :members:

ChannelExtractionTransformation:
--------------------------------

.. autoclass:: N2D2.ChannelExtractionTransformation
        :members:

ColorSpaceTransformation:
-------------------------

.. autoclass:: N2D2.ColorSpaceTransformation
        :members:

CompressionNoiseTransformation:
-------------------------------

.. autoclass:: N2D2.CompressionNoiseTransformation
        :members:

DCTTransformation:
------------------

.. autoclass:: N2D2.DCTTransformation
        :members:

DFTTransformation:
------------------

.. autoclass:: N2D2.DFTTransformation
        :members:

EqualizeTransformation:
-----------------------

.. autoclass:: N2D2.EqualizeTransformation
        :members:

ExpandLabelTransformation:
--------------------------

.. autoclass:: N2D2.ExpandLabelTransformation
        :members:

WallisFilterTransformation:
---------------------------

.. autoclass:: N2D2.WallisFilterTransformation
        :members:

ThresholdTransformation:
------------------------

.. autoclass:: N2D2.ThresholdTransformation
        :members:

SliceExtractionTransformation:
------------------------------

.. autoclass:: N2D2.SliceExtractionTransformation
        :members:

ReshapeTransformation:
----------------------

.. autoclass:: N2D2.ReshapeTransformation
        :members:

RescaleTransformation:
----------------------

.. autoclass:: N2D2.RescaleTransformation
        :members:

RangeClippingTransformation:
----------------------------

.. autoclass:: N2D2.RangeClippingTransformation
        :members:

RangeAffineTransformation:
--------------------------

.. autoclass:: N2D2.RangeAffineTransformation
        :members:

RandomAffineTransformation:
---------------------------

.. autoclass:: N2D2.RandomAffineTransformation
        :members:

NormalizeTransformation:
------------------------

.. autoclass:: N2D2.NormalizeTransformation
        :members:

MorphologyTransformation:
-------------------------

.. autoclass:: N2D2.MorphologyTransformation
        :members:

MorphologicalReconstructionTransformation:
------------------------------------------

.. autoclass:: N2D2.MorphologicalReconstructionTransformation
        :members:

MagnitudePhaseTransformation:
-----------------------------

.. autoclass:: N2D2.MagnitudePhaseTransformation
        :members:

LabelSliceExtractionTransformation:
-----------------------------------

.. autoclass:: N2D2.LabelSliceExtractionTransformation
        :members:

LabelExtractionTransformation:
------------------------------

.. autoclass:: N2D2.LabelExtractionTransformation
        :members:

GradientFilterTransformation:
-----------------------------

.. autoclass:: N2D2.GradientFilterTransformation
        :members:

ApodizationTransformation:
--------------------------

.. autoclass:: N2D2.ApodizationTransformation
        :members:

FilterTransformation:
---------------------

.. autoclass:: N2D2.FilterTransformation
        :members:

FlipTransformation:
-------------------

.. autoclass:: N2D2.FlipTransformation
        :members:



.. testcode::
   :hide:

   N2D2.ApodizationTransformation(N2D2.Rectangular_double(), 1)
   # N2D2.AffineTransformation(N2D2.AffineTransformation.Operator.Plus, "") # Empty string doesn't work ...
   N2D2.ChannelExtractionTransformation(N2D2.ChannelExtractionTransformation.Channel.Red)
   N2D2.ColorSpaceTransformation(N2D2.ColorSpaceTransformation.ColorSpace.BGR)
   N2D2.CompressionNoiseTransformation()
   N2D2.DCTTransformation()
   N2D2.DFTTransformation()
   N2D2.EqualizeTransformation()
   N2D2.ExpandLabelTransformation()
   N2D2.FlipTransformation()
   # N2D2.FilterTransformation(N2D2.Kernel_double("")) # kernel is empty ...
   N2D2.GradientFilterTransformation()
   N2D2.LabelExtractionTransformation("2", "2")
   N2D2.LabelSliceExtractionTransformation(2, 2)
   N2D2.MagnitudePhaseTransformation()
   N2D2.MorphologicalReconstructionTransformation(N2D2.MorphologicalReconstructionTransformation.Operation.ReconstructionByErosion, 2)
   N2D2.MorphologyTransformation(N2D2.MorphologyTransformation.Operation.Erode, 2)
   N2D2.NormalizeTransformation()
   N2D2.RandomAffineTransformation([[0, 0]])
   N2D2.RangeAffineTransformation(N2D2.RangeAffineTransformation.Operator.Plus, 5)
   N2D2.RangeClippingTransformation()
   N2D2.RescaleTransformation(2, 2)
   N2D2.ReshapeTransformation(2)
   N2D2.SliceExtractionTransformation(1,1)
   N2D2.ThresholdTransformation(5)
   # N2D2.TrimTransformation(1)
   N2D2.WallisFilterTransformation(10, 0.0, 1.0)