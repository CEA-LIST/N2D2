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

from n2d2.utils import inherit_init_docstring
from n2d2.error_handler import WrongInputType
from n2d2.filler.filler import Filler
from n2d2 import ConventionConverter

@inherit_init_docstring()
class Constant(Filler):
    """
    Fill with a constant value.
    """

    _N2D2_constructors = {
        '<float>': N2D2.ConstantFiller_float,        
    }
    _parameters={
        "value": "value",
    }
    _convention_converter= ConventionConverter(_parameters)

    def __init__(self, **config_parameters):
        """
        :param value: Value for the filling, default=0.0
        :type value: float, optional
        """
        Filler.__init__(self, **config_parameters)

        self._parse_optional_arguments(['value'])
        for k, v in self._optional_constructor_arguments.items():
            if k is 'value' and not isinstance(v, float):
                raise WrongInputType("value", str(type(v)), ["float"])
        self._set_N2D2_object(self._N2D2_constructors[self._model_key](**self.n2d2_function_argument_parser(self._optional_constructor_arguments)))
        self._set_N2D2_parameters(self._config_parameters)
        self.load_N2D2_parameters(self.N2D2())

    def _load_N2D2_optional_parameters(self, N2D2_object):
        self._optional_constructor_arguments['value'] = N2D2_object.getValue()
