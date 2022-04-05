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
class RangeAffine(Transformation):
    """
    Apply an affine transformation to the values of the image.
    """

    _Type = "RangeAffine"
    _parameters={
        "first_operator": "FirstOperator",
        "first_value": "FirstValue",
        "second_operator": "secondOperator",
        "second_value": "secondValue",
        "truncate": "Truncate"
    }
    _convention_converter= ConventionConverter(_parameters)

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

    def _load_N2D2_constructor_parameters(self, N2D2_object):
        self._constructor_arguments.update({
            'first_operator': N2D2_object.getFirstOperator(),
            'first_value': N2D2_object.getFirstValue(),
        })
    
    def _load_N2D2_optional_parameters(self, N2D2_object):
        self._optional_constructor_arguments.update({
            'second_operator': N2D2_object.getSecondOperator(),
            'second_value': N2D2_object.getSecondValue(),
        })
