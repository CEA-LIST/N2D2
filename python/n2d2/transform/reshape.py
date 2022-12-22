"""
    (C) Copyright 2022 CEA LIST. All Rights Reserved.
    Contributor(s): Cyril MOINEAU (cyril.moineau@cea.fr)

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
class Reshape(Transformation):
    """
    Reshape the data to a specified size.
    """
    _Type = "Reshape"
    _parameters={
        "nb_rows": "nbRows",
        "nb_cols": "nbCols",
        "nb_channels": "nbChannels",
    }
    _convention_converter= ConventionConverter(_parameters)

    def __init__(self,  nb_rows, **config_parameters):
        """
        :param nb_rows: New number of rows
        :type nb_rows: int
        :param nb_cols: New number of cols (0 = no check), default=0
        :type nb_cols: int, optional
        :param nb_channels: New number of channels (0 = no change), default=0
        :type nb_channels: int, optional
        """
        Transformation.__init__(self, **config_parameters)

        self._constructor_arguments.update({
            'nb_rows': nb_rows
        })
        self._parse_optional_arguments(['nb_cols', 'nb_channels'])

        self._N2D2_object = N2D2.ReshapeTransformation(self._constructor_arguments['nb_rows'],
                                                        **self.n2d2_function_argument_parser(self._optional_constructor_arguments)
                                                       )
        self._set_N2D2_parameters(self._config_parameters)
        self.load_N2D2_parameters(self.N2D2())

    def _load_N2D2_constructor_parameters(self, N2D2_object):
        self._constructor_arguments.update({
            'nb_rows': N2D2_object.getNbRows(), 
        })
    def _load_N2D2_optional_parameters(self, N2D2_object):
        self._optional_constructor_arguments.update({
            'nb_cols': N2D2_object.getNbCols(),
            'nb_channels': N2D2_object.getNbChannels(),
        })