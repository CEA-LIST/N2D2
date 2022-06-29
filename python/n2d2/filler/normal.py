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
class Normal(Filler):
    """
    Fill with a normal distribution.
    """

    _N2D2_constructors = {
        '<float>': N2D2.NormalFiller_float,
    }
    _parameters={
        "mean": "mean",
        "std_dev": "stdDev",
    }
    _convention_converter= ConventionConverter(_parameters)

    def __init__(self, **config_parameters):
        """
        :param mean: Mean value of the distribution, default=0.0
        :type mean: float, optional
        :param std_dev: Standard deviation of the distribution, default=1.0
        :type std_dev: float, optional
        """
        Filler.__init__(self, **config_parameters)

        self._parse_optional_arguments(['mean', 'std_dev'])
        for k, v in self._optional_constructor_arguments.items():
            if k == 'mean' and not isinstance(v, float):
                raise WrongInputType("mean", str(type(v)), ["float"])
            if k == 'std_dev' and not isinstance(v, float):
                raise WrongInputType("std_dev", str(type(v)), ["float"])
        self._set_N2D2_object(self._N2D2_constructors[self._model_key](**self.n2d2_function_argument_parser(self._optional_constructor_arguments)))
        self._set_N2D2_parameters(self._config_parameters)
        self.load_N2D2_parameters(self.N2D2())

    def _load_N2D2_optional_parameters(self, N2D2_object):
        self._optional_constructor_arguments['mean'] = N2D2_object.getMean()
        self._optional_constructor_arguments['std_dev'] = N2D2_object.getStdDev()
