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
class Rescale(Transformation):
    """
    Rescale the image to a specified size.
    """
    _Type = "Rescale"
    _parameters={
        "width": "Width",
        "height": "Height",
        "keep_aspect_ratio": "KeepAspectRatio",
        "resize_to_fit": "ResizeToFit",
    }
    _convention_converter= ConventionConverter(_parameters)

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

    def _load_N2D2_constructor_parameters(self, N2D2_object):
        self._constructor_arguments.update({
            'width': N2D2_object.getWidth(),
            'height': N2D2_object.getHeight(),
        })
