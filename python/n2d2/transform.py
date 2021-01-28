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
from n2d2.n2d2_interface import N2D2_Interface

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


class Transformation(N2D2_Interface):

    # TODO: Is there any way to check that no abstract Transformation object is generated?
    def __init__(self, **config_parameters):

        self._ApplyTo = N2D2.Database.StimuliSetMask.All
        if 'ApplyTo' in config_parameters:
            self._ApplyTo = N2D2.Database.StimuliSetMask.__members__[config_parameters.pop('ApplyTo')]

        N2D2_Interface.__init__(self, **config_parameters)

    def __str__(self):
        output = self._Type #+ "Transformation"
        output += N2D2_Interface.__str__(self)
        if self._ApplyTo is not N2D2.Database.StimuliSetMask.All:
            output += "[ApplyTo=" + str(self._ApplyTo) + "]"
        return output

    def get_apply_set(self):
        return self._ApplyTo


class Composite(Transformation):

    _Type = "Composite"

    def __init__(self, transformations, **config_parameters):
        Transformation.__init__(self, **config_parameters)

        if not isinstance(transformations, list):
            raise TypeError("Wanted ", type(list), " got ", type(transformations))
        if not transformations:
            raise ValueError("Got empty list as input. List must contain at least one element")

        self._transformations = transformations


        self._N2D2_object = N2D2.CompositeTransformation(self._transformations[0].N2D2())
        for transformation in self._transformations[1:]:
            self._N2D2_object.push_back(transformation.N2D2())

    def __str__(self):
        output = Transformation.__str__(self)
        output += "(\n"
        for trans in self._transformations:
            output += "\t" + trans.__str__()
            output += "\n"
        output += ")"
        return output


class PadCrop(Transformation):

    _Type = "PadCrop"

    # INI file parameters have same upper case name convention
    def __init__(self, Width, Height, **config_parameters):
        Transformation.__init__(self, **config_parameters)

        self._constructor_arguments.update({
            'Width': Width,
            'Height': Height,
        })

        self._N2D2_object = N2D2.PadCropTransformation(self._constructor_arguments['Width'],
                                                       self._constructor_arguments['Height'])
        self._set_N2D2_parameters(self._config_parameters)


class Distortion(Transformation):

    _Type = "Distortion"

    def __init__(self, **config_parameters):
        Transformation.__init__(self, **config_parameters)

        self._N2D2_object = N2D2.DistortionTransformation()
        self._set_N2D2_parameters(self._config_parameters)



class Rescale(Transformation):

    _Type = "Rescale"

    def __init__(self,  Width, Height, **config_parameters):
        Transformation.__init__(self, **config_parameters)

        self._constructor_arguments.update({
            'Width': Width,
            'Height': Height,
        })

        self._N2D2_object = N2D2.RescaleTransformation(self._constructor_arguments['Width'], self._constructor_arguments['Height'])
        self._set_N2D2_parameters(self._config_parameters)




class ColorSpace(Transformation):

    _Type = "ColorSpace"

    def __init__(self, ColorSpace, **config_parameters):
        Transformation.__init__(self, **config_parameters)

        self._constructor_arguments.update({
            'ColorSpace': N2D2.ColorSpaceTransformation.ColorSpace.__members__[ColorSpace],
        })

        self._N2D2_object = N2D2.ColorSpaceTransformation(self._constructor_arguments['ColorSpace'])
        self._set_N2D2_parameters(self._config_parameters)





class RangeAffine(Transformation):

    _Type = "RangeAffine"

    def __init__(self, FirstOperator, FirstValue, **config_parameters):
        Transformation.__init__(self, **config_parameters)

        self._constructor_arguments.update({
            'FirstOperator': N2D2.RangeAffineTransformation.Operator.__members__[FirstOperator],
            'FirstValue': FirstValue,
        })

        self._parse_optional_arguments(['SecondOperator', 'SecondValue'])

        self._optional_constructor_arguments['secondOperator'] = \
            N2D2.RangeAffineTransformation.Operator.__members__[self._optional_constructor_arguments['secondOperator']]

        self._N2D2_object = N2D2.RangeAffineTransformation(self._constructor_arguments['FirstOperator'],
                                                           self._constructor_arguments['FirstValue'],
                                                           **self._optional_constructor_arguments)
        self._set_N2D2_parameters(self._config_parameters)



class SliceExtraction(Transformation):

    _Type = "SliceExtraction"

    def __init__(self, Width, Height, **config_parameters):
        Transformation.__init__(self, **config_parameters)

        self._constructor_arguments.update({
            'Width': Width,
            'Height': Height,
        })

        self._parse_optional_arguments(['OffsetX', 'OffsetY'])

        self._N2D2_object = N2D2.SliceExtractionTransformation(self._constructor_arguments['Width'],
                                                           self._constructor_arguments['Height'],
                                                           **self._optional_constructor_arguments)
        self._set_N2D2_parameters(self._config_parameters)



class Flip(Transformation):

    _Type = "Flip"

    def __init__(self, **config_parameters):
        Transformation.__init__(self, **config_parameters)

        self._parse_optional_arguments(['HorizontalFlip', 'VerticalFlip'])

        self._N2D2_object = N2D2.FlipTransformation(**self._optional_constructor_arguments)
        self._set_N2D2_parameters(self._config_parameters)


# TODO : Change binding to expose apply method 
# class CustomTransformation(Transformation):
#     def __init__(self, custom_transformation):
#         super().__init__()
#         self._transformation = custom_transformation
