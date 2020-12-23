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
class Transformation:

    def __init__(self):
        self._constructor_parameters = {}
        self._trans_parameters = {}
        self._modified_keys = []
        self._transformation = None

    def _set_parameters(self, parameters):
        for key, value in parameters.items():
            if key in self._trans_parameters:
                self._trans_parameters[key] = value
                self._modified_keys.append(key)
            else:
                raise n2d2.UndefinedParameterError(key, self)

    def _set_N2D2_parameters(self, parameters):
        for key, value in parameters.items():
            if isinstance(value, bool):
                self._transformation.setParameter(key, str(int(value)))
            elif isinstance(value, list):
                list_string = ""
                for elem in value:
                    list_string += str(elem) + " "
                self._transformation.setParameter(key, list_string)
            else:
                self._transformation.setParameter(key, str(value))

    def N2D2(self):
        if self._transformation is None:
            raise n2d2.UndefinedModelError("N2D2 transformation member has not been created")
        return self._transformation

    def __str__(self):
        output = self._Type + "Transformation("
        for key, value in self._constructor_parameters.items():
            output += key + "=" + str(value) + ", "
        for key, value in self._trans_parameters.items():
            if key in self._modified_keys:
                output += key + "=" + str(value) + ", "
        output = output[:len(output)-2]
        output += ")"
        return output


class Composite(Transformation):

    _Type = "Composite"

    def __init__(self, transformations):
        # NOTE: At the moment superfluous in this class
        super().__init__()

        if not isinstance(transformations, list):
            raise TypeError("Wanted ", type(list), " got ", type(transformations))
        # TODO: This cannot be empty anyways since compulsory argument?
        if not transformations:
            raise ValueError("Got empty list as input. List must contain at least one element")
        self._transformation = N2D2.CompositeTransformation(transformations[0].N2D2())
        for transformation in transformations[1:]:
            self._transformation.push_back(transformation.N2D2())


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
    def __init__(self, Width, Height, **trans_parameters):
        super().__init__()

        self._constructor_parameters = {
            'Width': Width,
            'Height': Height,
        }

        self._trans_parameters = {
            'AdditiveWH': False,
            'BorderType': 'MinusOneReflectBorder',
            'BorderValue': []
        }

        self._set_parameters(trans_parameters)
        self._transformation = N2D2.PadCropTransformation(self._constructor_parameters['Width'],
                                                          self._constructor_parameters['Height'])
        self._set_N2D2_parameters(self._trans_parameters)


class Distortion(Transformation):

    _Type = "Distortion"

    def __init__(self, **trans_parameters):
        super().__init__()

        self._trans_parameters = {
            'ElasticGaussianSize': 15,
            'ElasticSigma': 6.0,
            'ElasticScaling': 0.0,
            'Scaling': 0.0,
            'Rotation': 0.0,
            'IgnoreMissingData': False
        }

        self._set_parameters(trans_parameters)
        self._transformation = N2D2.DistortionTransformation()
        self._set_N2D2_parameters(self._trans_parameters)
