"""
    (C) Copyright 2020 CEA LIST. All Rights Reserved.
    Contributor(s): Cyril MOINEAU (cyril.moineau@cea.fr) 
                    Johannes THIELE (johannes.thiele@cea.fr)

    This software is governed by the CeCILL-C license under French law and
    abiding by the rules of distribution of free software.  You can  use,
    modify and/ or redistribute the software under the terms of the CeCILL-C
    license as circulated by CEA, CNRS and INRIA at the following URL
    "http://www.cecill.info".

    As a counterpart to the access to the source code and  rights to copy,
    modify and redistribute granted by the license, users are provided only
    with a limited warranty  and the software's author,  the holder of the
    economic rights,  and the successive licensors  have only  limited
    liability.

    The fact that you are presently reading this means that you have had
    knowledge of the CeCILL-C license and that you accept its terms.
"""

import N2D2
import n2d2
from n2d2.parameterizable import Parameterizable

"""
For the tranformations, it is relatively important to be able to write custom 
transformations in Python. This is because these transformations are anyways not
executed on GPU at the moment, and additionaly they depend strongly on the data
that is used.
However, at the moment this remains tricky due to how data is loaded in an optimized
way in N2D2
"""

"""
Example for custom transformation
"""
"""
class CustomRescaleTransform():
    def __init__(self, factor):
        self.factor = factor
        
    def __call__(self, x):
        return self.factor*x
"""
"""
Example for a Python wrapper around the binding
"""
"""
class PadCropTransformation():
    def __init__(self, dimX, dimY):
        self.trans = N2D2.PadCropTransformation(dimX, dimY)
        
    def __call__(self, x):
        return self.trans.apply(x)
"""


class Transformation(Parameterizable):

    # TODO: Is there any way to check that no abstract Transformation object is generated?
    def __init__(self):
        Parameterizable.__init__(self)

    def __str__(self):
        output = self._Type + "Transformation"
        output += Parameterizable.__str__(self)
        return output


class Composite(Transformation):

    _Type = "Composite"

    def __init__(self, transformations):
        Transformation.__init__(self)

        if not isinstance(transformations, list):
            raise TypeError("Wanted ", type(list), " got ", type(transformations))
        if not transformations:
            raise ValueError("Got empty list as input. List must contain at least one element")
        self._N2D2_object = N2D2.CompositeTransformation(transformations[0].N2D2())
        for transformation in transformations[1:]:
            self._N2D2_object.push_back(transformation.N2D2())


class PadCrop(Transformation):

    _Type = "PadCrop"

    """
    # Currently not necessary since configured with N2D2.set_Parameter(string)
    _border_type = {
        "ConstantBorder", N2D2.PadCropTransformation.BorderType.ConstantBorder,
        "ReplicateBorder", N2D2.PadCropTransformation.BorderType.ReplicateBorder,
        "ReflectBorder", N2D2.PadCropTransformation.BorderType.ReflectBorder,
        "WrapBorder", N2D2.PadCropTransformation.BorderType.WrapBorder,
        "MinusOneReflectBorder", N2D2.PadCropTransformation.BorderType.MinusOneReflectBorder,
        "MeanBorder", N2D2.PadCropTransformation.BorderType.MeanBorder
    }
    """

    # INI file parameters have same upper case name convention
    def __init__(self, Width, Height, **config_parameters):
        Transformation.__init__(self)

        self._constructor_arguments.update({
            'Width': Width,
            'Height': Height,
        })

        self._config_parameters.update({
            'AdditiveWH': False,
            'BorderType': 'MinusOneReflectBorder',
            'BorderValue': []
        })

        self._set_config_parameters(config_parameters)
        self._N2D2_object = N2D2.PadCropTransformation(self._constructor_arguments['Width'],
                                                       self._constructor_arguments['Height'])
        self._set_N2D2_parameters(self._config_parameters)


class Distortion(Transformation):

    _Type = "Distortion"

    def __init__(self, **config_parameters):
        Transformation.__init__(self)

        self._config_parameters.update({
            'ElasticGaussianSize': 15,
            'ElasticSigma': 6.0,
            'ElasticScaling': 0.0,
            'Scaling': 0.0,
            'Rotation': 0.0,
            'IgnoreMissingData': False
        })

        self._set_config_parameters(config_parameters)
        self._N2D2_object = N2D2.DistortionTransformation()
        self._set_N2D2_parameters(self._config_parameters)
