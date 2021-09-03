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
from abc import ABC, abstractmethod

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
# class Test(N2D2.CustomTransformation):
#     """
#     Override this class to create your own Transformation.
#     You need to override the following method : 
#         - apply_unsigned_char(self, Tensor<unsigned char>& frame, Tensor<int>& /*labels*/, std::vector<std::shared_ptr<ROI> >& /*labelsROI*/, int /*id*/ = -1)
#         - apply_int(self, Tensor<int>& frame, Tensor<int>& /*labels*/, std::vector<std::shared_ptr<ROI> >& /*labelsROI*/, int /*id*/ = -1)
#         - etc ..
#     """
#     def __init__(self):
#         print("init TEST")
#         N2D2.CustomTransformation.__init__(self)
#         print("DONE")

#     def apply_unsigned_char(self, frame, labels, labelsROI, id):
#          print("using python function !")

# class CustomTransformation(Transformation):
#     def __init__(self, **config_parameters):
#         Transformation.__init__(self, **config_parameters)
#         self._N2D2_object = Test()

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

# TODO : Change binding to expose apply method 
# class CustomTransformation(Transformation):
#     def __init__(self, custom_transformation):
#         super().__init__()
#         self._transformation = custom_transformation

# https://pybind11.readthedocs.io/en/stable/reference.html#c.PYBIND11_OVERRIDE
# https://pybind11.readthedocs.io/en/stable/advanced/classes.html#overriding-virtuals

class Transformation(N2D2_Interface, ABC):

    @abstractmethod
    def __init__(self, **config_parameters):

        self._apply_to = N2D2.Database.StimuliSetMask.All
        if 'apply_to' in config_parameters:
            self._apply_to = N2D2.Database.StimuliSetMask.__members__[config_parameters.pop('apply_to')]

        N2D2_Interface.__init__(self, **config_parameters)

    def __str__(self):
        output = self._Type #+ "Transformation"
        output += N2D2_Interface.__str__(self)
        if self._apply_to is not N2D2.Database.StimuliSetMask.All:
            output += "[apply_to=" + str(self._apply_to) + "]"
        return output

    def get_apply_set(self):
        return self._apply_to


class Composite(Transformation):
    """
    A composition of transformations
    """

    _Type = "Composite"

    def __init__(self, transformations, **config_parameters):
        """
        :param transformations: List of the transformations to use.
        :type transformations: list
        """
        Transformation.__init__(self, **config_parameters)

        if not isinstance(transformations, list):
            raise n2d2.error_handler.WrongInputType("transformations", type(transformations), [str(type(list))])
        if not transformations:
            raise n2d2.error_handler.IsEmptyError(transformations)

        self._transformations = transformations

        #self._N2D2_object = N2D2.CompositeTransformation(self._transformations[0].N2D2())
        #for transformation in self._transformations[1:]:
        #    self._N2D2_object.push_back(transformation.N2D2())

    def get_transformations(self):
        """
        Return the list of transformations applied by the composite transformation
        """
        return self._transformations

    def __str__(self):
        output = Transformation.__str__(self)
        output += "(\n"
        for trans in self._transformations:
            output += "\t" + trans.__str__()
            output += "\n"
        output += ")"
        return output


class PadCrop(Transformation):
    """
    Pad/crop the image to a specified size.
    """

    _Type = "PadCrop"

    _convention_converter= n2d2.ConventionConverter({
        "width": "Width",
        "height": "Height",
        "additive_wh": "AdditiveWH",
        "border_type": "BorderType",
        "border_value": "BorderValue",

    })

    # INI file parameters have same upper case name convention
    def __init__(self, width, height, **config_parameters):
        """Possible values for border_type parameter : 
        ``ConstantBorder``: pad with ``BorderValue``, 
        ``ReplicateBorder``: last element is replicated throughout, like aaaaaa\|abcdefgh\|hhhhhhh, 
        ``ReflectBorder``: border will be mirror reflection of the border elements, like fedcba\|abcdefgh\|hgfedcb,
        ``WrapBorder``: it will look like cdefgh\|abcdefgh\|abcdefg, 
        ``MinusOneReflectBorder``: same as ``ReflectBorder`` but with a slight change, like gfedcb\|abcdefgh\|gfedcba, 
        ``MeanBorder``: pad with the mean color of the image

        :param width: Width of the padded/cropped image
        :type width: int
        :param height: height of the padded/cropped image
        :type height: int
        :param border_type:  Border type used when padding, default="MinusOneReflectBorder"
        :type border_type: str, optional
        :param border_value: Background color used when padding with ``BorderType`` is ``ConstantBorder``,default=[0.0, 0.0, 0.0]
        :type border_value: list, optional
        """
        Transformation.__init__(self, **config_parameters)

        self._constructor_arguments.update({
            'width': width,
            'height': height,
        })

        self._N2D2_object = N2D2.PadCropTransformation(self._constructor_arguments['width'],
                                                       self._constructor_arguments['height'])
        self._set_N2D2_parameters(self._config_parameters)
        self.load_N2D2_parameters(self.N2D2())

class Distortion(Transformation):
    """
    Apply elastic distortion to the image. 
    This transformation is generally used on-the-fly 
    (so that a different distortion is performed for each image), and for the learning only.
    """

    _Type = "Distortion"

    _convention_converter= n2d2.ConventionConverter({
        "elastic_gaussian_size": "ElasticGaussianSize",
        "elastic_sigma": "ElasticSigma",
        "elastic_scaling": "ElasticScaling",
        "scaling": "Scaling",
        "rotation": "Rotation",
        "ignore_missing_data": "IgnoreMissingData",

    })

    def __init__(self, **config_parameters):
        """
        :param elastic_gaussian_size: Size of the gaussian for elastic distortion (in pixels), default=15
        :type elastic_gaussian_size: int, optional
        :param elastic_sigma: Sigma of the gaussian for elastic distortion, default=6.0
        :type elastic_sigma: float, optional
        :param elastic_scaling: Scaling of the gaussian for elastic distortion, default=0.0
        :type elastic_scaling: float, optional
        :param scaling: Maximum random scaling amplitude (+/-, in percentage), default=0.0
        :type scaling: float, optional
        :param rotation: Maximum random rotation amplitude (+/-, in °), default=0.0
        :type rotation: float, optional
        """
        Transformation.__init__(self, **config_parameters)

        self._N2D2_object = N2D2.DistortionTransformation()

        # Scaling parameter is written with an upper case here but with a lower case in other classes
        # Treating this as an exception
        # if "scaling" in self._config_parameters: 
        #     self._N2D2_object.setParameter("Scaling", str(self._config_parameters.pop('scaling')))

        self._set_N2D2_parameters(self._config_parameters)
        self.load_N2D2_parameters(self.N2D2())

class Rescale(Transformation):
    """
    Rescale the image to a specified size.
    """
    _Type = "Rescale"
    _convention_converter= n2d2.ConventionConverter({
        "width": "Width",
        "height": "Height",
        "keep_aspect_ratio": "KeepAspectRatio",
        "resize_to_fit": "ResizeToFit",
    })

    def __init__(self,  width, height, **config_parameters):
        """
        :param width: Width of the rescaled image
        :type width: int
        :param height: Height of the rescaled image
        :type height: int
        :param keep_aspect_ratio: If true, keeps the aspect ratio of the image, default=False
        :type keep_aspect_ratio: bool, optional
        :param resize_to_fit: If true, resize along the longest dimension when ``KeepAspectRatio`` is true, default=True
        :type resize_to_fit: bool, optional
        """
        Transformation.__init__(self, **config_parameters)

        self._constructor_arguments.update({
            'width': width,
            'height': height,
        })

        self._N2D2_object = N2D2.RescaleTransformation(self._constructor_arguments['width'],
                                                       self._constructor_arguments['height'])
        self._set_N2D2_parameters(self._config_parameters)
        self.load_N2D2_parameters(self.N2D2())


class ColorSpace(Transformation):

    _Type = "ColorSpace"
    _convention_converter= n2d2.ConventionConverter({
        "color_space": "ColorSpace",
    })

    def __init__(self, color_space, **config_parameters):
        """
        Possible values for color_space parameter :
        ``BGR``: convert any gray, BGR or BGRA image to BGR,
        ``RGB``: convert any gray, BGR or BGRA image to RGB,
        ``HSV``: convert BGR image to HSV,
        ``HLS``: convert BGR image to HLS,
        ``YCrCb``: convert BGR image to YCrCb,
        ``CIELab``: convert BGR image to CIELab,
        ``CIELuv``: convert BGR image to CIELuv,
        ``RGB_to_BGR``: convert RGB image to BGR,
        ``RGB_to_HSV``: convert RGB image to HSV,
        ``RGB_to_HLS``: convert RGB image to HLS,
        ``RGB_to_YCrCb``: convert RGB image to YCrCb,
        ``RGB_to_CIELab``: convert RGB image to CIELab,
        ``RGB_to_CIELuv``: convert RGB image to CIELuv,
        ``HSV_to_BGR``: convert HSV image to BGR,
        ``HSV_to_RGB``: convert HSV image to RGB,
        ``HLS_to_BGR``: convert HLS image to BGR,
        ``HLS_to_RGB``: convert HLS image to RGB,
        ``YCrCb_to_BGR``: convert YCrCb image to BGR,
        ``YCrCb_to_RGB``: convert YCrCb image to RGB,
        ``CIELab_to_BGR``: convert CIELab image to BGR,
        ``CIELab_to_RGB``: convert CIELab image to RGB,
        ``CIELuv_to_BGR``: convert CIELuv image to BGR,
        ``CIELuv_to_RGB``: convert CIELuv image to RGB.

        :param color_space: Convert image color.
        :type color_space: str
        """
        Transformation.__init__(self, **config_parameters)

        self._constructor_arguments.update({
            'color_space': N2D2.ColorSpaceTransformation.ColorSpace.__members__[color_space],
        })

        self._N2D2_object = N2D2.ColorSpaceTransformation(self._constructor_arguments['color_space'])
        self._set_N2D2_parameters(self._config_parameters)
        self.load_N2D2_parameters(self.N2D2())



class RangeAffine(Transformation):
    """
    Apply an affine transformation to the values of the image.
    """

    _Type = "RangeAffine"
    _convention_converter= n2d2.ConventionConverter({
        "first_operator": "FirstOperator",
        "first_value": "FirstValue",
        "second_operator": "secondOperator",
        "second_value": "secondValue",
        "truncate": "Truncate"
    })

    def __init__(self, first_operator, first_value, **config_parameters):
        """
        :param first_operator: First operator, can be ``Plus``, ``Minus``, ``Multiplies``, ``Divides``
        :type first_operator: str
        :param first_value: First value
        :type first_value: float 
        :param second_operator: Second operator, can be ``Plus``, ``Minus``, ``Multiplies``, ``Divides``, default="Plus"
        :type second_operator: str, optional
        :param second_value: Second value, default=0.0
        :type second_value: float, optional
        """
        Transformation.__init__(self, **config_parameters)

        self._constructor_arguments.update({
            'first_operator': N2D2.RangeAffineTransformation.Operator.__members__[first_operator],
            'first_value': first_value,
        })

        self._parse_optional_arguments(['second_operator', 'second_value'])

        if 'second_operator' in self._optional_constructor_arguments:
            self._optional_constructor_arguments['second_operator'] = \
                N2D2.RangeAffineTransformation.Operator.__members__[self._optional_constructor_arguments['second_operator']]

        self._N2D2_object = N2D2.RangeAffineTransformation(self._constructor_arguments['first_operator'],
                                                           self._constructor_arguments['first_value'],
                                                           **self.n2d2_function_argument_parser(self._optional_constructor_arguments))
        self._set_N2D2_parameters(self._config_parameters)
        self.load_N2D2_parameters(self.N2D2())

class SliceExtraction(Transformation):
    """
    Extract a slice from an image.
    """
    _Type = "SliceExtraction"
    _convention_converter= n2d2.ConventionConverter({
        "width": "width",
        "height": "height",
        "offset_x": "offsetX",
        "offset_y": "offsetY",
        "random_offset_x": "RandomOffsetX",
        "random_offset_y": "RandomOffsetY",
        "random_rotation": "RandomRotation",
        "random_rotation_range": "RandomRotationRange",
        "random_scaling": "RandomScaling",
        "random_scaling_range": "RandomScalingRange",
        "allow_padding": "AllowPadding",
        "border_type": "BorderType",
        "border_value": "BorderValue",
    })

    def __init__(self, width, height, **config_parameters):
        """Possible values for border_type parameter
        ``ConstantBorder``: pad with ``BorderValue``, 
        ``ReplicateBorder``: last element is replicated throughout, like aaaaaa\|abcdefgh\|hhhhhhh, 
        ``ReflectBorder``: border will be mirror reflection of the border elements, like fedcba\|abcdefgh\|hgfedcb,
        ``WrapBorder``: it will look like cdefgh\|abcdefgh\|abcdefg,
        ``MinusOneReflectBorder``: same as ``ReflectBorder`` but with a slight change, like gfedcb\|abcdefgh\|gfedcba, 
        ``MeanBorder``: pad with the mean color of the image

        :param width: Width of the slice to extract
        :type width: int
        :param height: Height of the slice to extract
        :type height: int
        :param offset_x: X offset of the slice to extract, default=0
        :type offset_x: int, optional
        :param offset_y: Y offset of the slice to extract, default=0
        :type offset_y: int, optional
        :param random_offset_x: If true, the X offset is chosen randomly, default=False
        :type random_offset_x: bool, optional
        :param random_offset_y: If true, the Y offset is chosen randomly, default=False
        :type random_offset_y: bool, optional
        :param random_rotation: If true, extract randomly rotated slices, default=False
        :type random_rotation: bool, optional
        :param random_rotation_range: Range of the random rotations, in degrees, counterclockwise (if ``RandomRotation`` is enabled), default=[0.0, 360.0]
        :type random_rotation_range: list, optional
        :param random_scaling: If true, extract randomly scaled slices, default=False
        :type random_scaling: bool, optional
        :param random_scaling_range: Range of the random scaling (if ``RandomRotation`` is enabled), default=[0.8, 1.2]
        :type random_scaling_range: list, optional
        :param allow_padding: If true, zero-padding is allowed if the image is smaller than the slice to extract, default=False
        :type allow_padding: bool, optional
        :param border_type: Border type used when padding, default="MinusOneReflectBorder"
        :type border_type: str, optional
        """
        Transformation.__init__(self, **config_parameters)

        self._constructor_arguments.update({
            'width': width,
            'height': height,
        })

        self._parse_optional_arguments(['offset_x', 'offset_y'])

        self._N2D2_object = N2D2.SliceExtractionTransformation(self._constructor_arguments['width'],
                                                           self._constructor_arguments['height'],
                                                           **self.n2d2_function_argument_parser(self._optional_constructor_arguments))
        self._set_N2D2_parameters(self._config_parameters)
        self.load_N2D2_parameters(self.N2D2())



class Flip(Transformation):
    """
    Image flip transformation.
    """

    _Type = "Flip"
    _convention_converter= n2d2.ConventionConverter({
        "horizontal_flip": "horizontalFlip",
        "vertical_flip": "verticalFlip",
        "random_horizontal_flip": "RandomHorizontalFlip",
        "random_vertical_flip": "RandomVerticalFlip",
    })

    def __init__(self, **config_parameters):
        """
        :param horizontal_flip: If true, flip the image horizontally, default=False
        :type horizontal_flip: bool, optional
        :param vertical_flip: If true, flip the image vertically, default=False
        :type vertical_flip: bool, optional
        :param random_horizontal_flip: If true, randomly flip the image horizontally, default=False
        :type random_horizontal_flip: bool, optional
        :param random_vertical_flip: If true, randomly flip the image vertically, default=False
        :type random_vertical_flip: bool, optional
        """
        Transformation.__init__(self, **config_parameters)

        self._parse_optional_arguments(['horizontal_flip', 'vertical_flip'])

        self._N2D2_object = N2D2.FlipTransformation(**self.n2d2_function_argument_parser(self._optional_constructor_arguments))
        self._set_N2D2_parameters(self._config_parameters)
        self.load_N2D2_parameters(self.N2D2())




class RandomResizeCrop(Transformation):

    _Type = "RandomResizeCrop"

    _convention_converter= n2d2.ConventionConverter({
        "width": "Width",
        "height": "Height",
        "offset_x": "offsetX",
        "offset_y": "offsetY",
        "scale_min": "ScaleMin",
        "scale_max": "ScaleMax",
        "ratio_min": "RatioMin",
        "ratio_max": "RatioMax",
    })

    # INI file parameters have same upper case name convention
    def __init__(self, width, height, **config_parameters):
        """
        :param width: Width of the image to Crop.
        :type width: int
        :param height: Height of the image to Crop.
        :type height: int
        :param offset_x: X offset, default=0
        :type offset_x: int, optional
        :param offset_y: Y offset, default=0
        :type offset_y: int, optional
        """
        Transformation.__init__(self, **config_parameters)

        self._constructor_arguments.update({
            'width': width,
            'height': height,
        })

        self._parse_optional_arguments(['offset_x', 'offset_y'])

        self._N2D2_object = N2D2.RandomResizeCropTransformation(self._constructor_arguments['width'],
                                                self._constructor_arguments['height'],
                                                **self.n2d2_function_argument_parser(self._optional_constructor_arguments))
        self._set_N2D2_parameters(self._config_parameters)
        self.load_N2D2_parameters(self.N2D2())

class ChannelExtraction(Transformation):
    """
    Extract an image channel.
    """
    _convention_converter= n2d2.ConventionConverter({
        "channel": "Channel",
    })
    def __init__(self, channel, **config_parameters):
        """
            The ``channel`` parameter can take the following values :
            ``Blue``: blue channel in the BGR colorspace, or first channel of any colorspace, 
            ``Green``: green channel in the BGR colorspace, or second channel of any colorspace,
            ``Red``: red channel in the BGR colorspace, or third channel of any colorspace,
            ``Hue``: hue channel in the HSV colorspace,
            ``Saturation``: saturation channel in the HSV colorspace,
            ``Value``: value channel in the HSV colorspace,
            ``Gray``: gray conversion,
            ``Y``: Y channel in the YCbCr colorspace,
            ``Cb``: Cb channel in the YCbCr colorspace,
            ``Cr``: Cr channel in the YCbCr colorspace

            :param channel: channel to extract
            :type channel: str
        """
        Transformation.__init__(self, **config_parameters)
        
        if channel not in N2D2.ChannelExtractionTransformation.Channel.__members__.keys():
            raise n2d2.error_handler.WrongValue("channel", channel,
                                                ", ".join(N2D2.ChannelExtractionTransformation.Channel.__members__.keys()))

        self._constructor_arguments.update({
            'channel': N2D2.ChannelExtractionTransformation.Channel.__members__[channel],
        })

        self._parse_optional_arguments([])

        self._N2D2_object = N2D2.ChannelExtractionTransformation(self._constructor_arguments['channel'],
                                                **self.n2d2_function_argument_parser(self._optional_constructor_arguments))
        self._set_N2D2_parameters(self._config_parameters)
        self.load_N2D2_parameters(self.N2D2())
