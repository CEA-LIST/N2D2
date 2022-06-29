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
class Flip(Transformation):
    """
    Image flip transformation.
    """

    _Type = "Flip"
    _parameters={
        "horizontal_flip": "horizontalFlip",
        "vertical_flip": "verticalFlip",
        "random_horizontal_flip": "RandomHorizontalFlip",
        "random_vertical_flip": "RandomVerticalFlip",
    }
    _convention_converter= ConventionConverter(_parameters)

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
