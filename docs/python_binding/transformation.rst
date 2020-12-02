Transformation
====

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
    ct.push_back(trans)

To apply Transformation to a dataset, we need an object :py:class:`N2D2.StimuliProvider` which act as a data loader.

.. testcode::

    stimuli = N2D2.StimuliProvider(database, [24, 24, 1], batchSize, False)
    stimuli.addTransformation(ct, database.StimuliSetMask(0))

Transformation:
---------------

.. autoclass:: N2D2.Transformation
   :members:

CompositeTransformation:
------------------------

.. autoclass:: N2D2.CompositeTransformation
   :members:

DistortionTransformation:
-------------------------

.. autoclass:: N2D2.DistortionTransformation
   :members:

PadCropTransformation:
----------------------

.. autoclass:: N2D2.PadCropTransformation
   :members: