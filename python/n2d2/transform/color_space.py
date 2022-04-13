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
class ColorSpace(Transformation):

    _Type = "ColorSpace"
    _parameters={
        "color_space": "ColorSpace",
    }
    _convention_converter= ConventionConverter(_parameters)

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

    def _load_N2D2_constructor_parameters(self, N2D2_object):
        self._constructor_arguments.update({
            'color_space': N2D2_object.getColorSpace(),
        })
