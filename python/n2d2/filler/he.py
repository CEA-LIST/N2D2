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
from n2d2.error_handler import WrongValue, WrongInputType
from n2d2.filler.filler import Filler
from n2d2 import ConventionConverter

@inherit_init_docstring()
class He(Filler):
    """
    Fill with an normal distribution with normalized variance taking into account the rectifier nonlinearity :cite:`He2015`. This filler is sometimes referred as MSRA filler or Kaiming initialization.
    """
    _N2D2_constructors = {
        '<float>': N2D2.HeFiller_float,
    }
    _parameters = {
        "variance_norm": "varianceNorm",
        "mean_norm": "meanNorm",
        "scaling": "scaling",
    }
    _convention_converter= ConventionConverter(_parameters)

    def __init__(self, **config_parameters):
        """
        :param variance_norm: Normalization, can be ``FanIn``, ``Average`` or ``FanOut``, default='FanIn'
        :type variance_norm: str, optional
        :param scaling: Scaling factor, default=1.0
        :type scaling: float, optional
        :param mean_norm: 
        :type mean_norm: float, optional
        """
        Filler.__init__(self, **config_parameters)

        if "variance_norm" in self._config_parameters:
            variance_norm = self._config_parameters["variance_norm"]
            if variance_norm not in self._N2D2_constructors[self._model_key].VarianceNorm.__members__.keys():
                raise WrongValue("variance_norm", variance_norm, self._N2D2_constructors[self._model_key].VarianceNorm.__members__.keys())
            self._config_parameters["variance_norm"] = self._N2D2_constructors[self._model_key].VarianceNorm.__members__[variance_norm]

        self._parse_optional_arguments(['variance_norm', 'mean_norm', 'scaling'])
        for k, v in self._optional_constructor_arguments.items():
            if k is 'scaling' and not isinstance(v, float):
                raise WrongInputType("scaling", str(type(v)), ["float"])
            if k is 'mean_norm' and not isinstance(v, float):
                raise WrongInputType("mean_norm", str(type(v)), ["float"])

        self._set_N2D2_object(self._N2D2_constructors[self._model_key](**self.n2d2_function_argument_parser(self._optional_constructor_arguments)))
        self._set_N2D2_parameters(self._config_parameters)
        self.load_N2D2_parameters(self.N2D2())


    def _load_N2D2_optional_parameters(self, N2D2_object):
        self._optional_constructor_arguments['variance_norm'] = N2D2_object.getVarianceNorm()
        self._optional_constructor_arguments['mean_norm'] = N2D2_object.getMeanNorm()
        self._optional_constructor_arguments['scaling'] = N2D2_object.getScaling()
