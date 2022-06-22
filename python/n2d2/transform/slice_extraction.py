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

from n2d2.n2d2_interface import ConventionConverter
from n2d2.transform.transformation import Transformation
from n2d2.utils import inherit_init_docstring

import N2D2

@inherit_init_docstring()
class SliceExtraction(Transformation):
    """
    Extract a slice from an image.
    """
    _Type = "SliceExtraction"
    _parameters={
        "width": "width",
        "height": "height",
        "offset_x": "OffsetX",
        "offset_y": "OffsetY",
        "random_offset_x": "RandomOffsetX",
        "random_offset_y": "RandomOffsetY",
        "random_rotation": "RandomRotation",
        "random_rotation_range": "RandomRotationRange",
        "random_scaling": "RandomScaling",
        "random_scaling_range": "RandomScalingRange",
        "allow_padding": "AllowPadding",
        "border_type": "BorderType",
        "border_value": "BorderValue",
    }
    _convention_converter= ConventionConverter(_parameters)

    def __init__(self, width, height, **config_parameters):
        """Possible values for border_type  \
        ``ConstantBorder``: pad with ``BorderValue``, \
        ``ReplicateBorder``: last element is replicated throughout, like aaaaaa|abcdefgh|hhhhhhh, \
        ``ReflectBorder``: border will be mirror reflection of the border elements, like fedcba|abcdefgh|hgfedcb, \
        ``WrapBorder``: it will look like cdefgh|abcdefgh|abcdefg, \
        ``MinusOneReflectBorder``: same as ``ReflectBorder`` but with a slight change, like gfedcb|abcdefgh|gfedcba, \
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
    def _load_N2D2_constructor_parameters(self, N2D2_object):
        self._constructor_arguments.update({
            'width': N2D2_object.getWidth(),
            'height': N2D2_object.getHeight(),
        })
    def _load_N2D2_optional_parameters(self, N2D2_object):
        self._optional_constructor_arguments.update({
            'offset_x':  N2D2_object.getOffsetX(),
            'offset_y':  N2D2_object.getOffsetY(),
        })
