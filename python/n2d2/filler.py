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

class Filler(N2D2_Interface):

    def __init__(self, **config_parameters):

        if 'datatype' in config_parameters:
            self._datatype = config_parameters.pop('datatype')
        else:
            self._datatype = n2d2.global_variables.default_datatype

        self._model_key = '<' + self._datatype + '>'

        N2D2_Interface.__init__(self, **config_parameters)

    def __str__(self):
        output = self._type
        output += N2D2_Interface.__str__(self)
        return output


class He(Filler):

    _type = "HeFiller"

    _filler_generators = {
        '<float>': N2D2.HeFiller_float,
    }

    def __init__(self, **config_parameters):
        Filler.__init__(self, **config_parameters)

        self._parse_optional_arguments(['variance_norm', 'mean_norm', 'scaling'])
        self._N2D2_object = self._filler_generators[self._model_key](**self.n2d2_function_argument_parser(self._optional_constructor_arguments))
        self._set_N2D2_parameters(self._config_parameters)


class Normal(Filler):

    _type = "NormalFiller"

    """Static members"""
    _filler_generators = {
        '<float>': N2D2.NormalFiller_float,
    }

    def __init__(self, **config_parameters):
        Filler.__init__(self, **config_parameters)

        self._parse_optional_arguments(['mean', 'std_dev'])
        self._N2D2_object = self._filler_generators[self._model_key](**self.n2d2_function_argument_parser(self._optional_constructor_arguments))
        self._set_N2D2_parameters(self._config_parameters)


class Xavier(Filler):

    _type = "XavierFiller"

    """Static members"""
    _filler_generators = {
        '<float>': N2D2.XavierFiller_float
    }

    def __init__(self, **config_parameters):
        Filler.__init__(self, **config_parameters)

        self._parse_optional_arguments(['variance_norm', 'distribution', 'scaling'])
        if 'variance_norm' in self._optional_constructor_arguments:
            self._optional_constructor_arguments['variance_norm'] = \
                N2D2.XavierFiller_float.VarianceNorm.__members__[self._optional_constructor_arguments['variance_norm']]
        if 'distribution' in self._optional_constructor_arguments:
            self._optional_constructor_arguments['distribution'] = \
                self.N2D2.XavierFiller_float.Distribution.__members__[self._optional_constructor_arguments['distribution']]
        self._N2D2_object = self._filler_generators[self._model_key](**self.n2d2_function_argument_parser(self._optional_constructor_arguments))
        self._set_N2D2_parameters(self._config_parameters)


class Constant(Filler):

    _type = "ConstantFiller"

    """Static members"""
    _filler_generators = {
        '<float>': N2D2.ConstantFiller_float
    }

    def __init__(self, **config_parameters):
        Filler.__init__(self, **config_parameters)

        self._parse_optional_arguments(['value'])
        self._N2D2_object = self._filler_generators[self._model_key](**self.n2d2_function_argument_parser(self._optional_constructor_arguments))
        self._set_N2D2_parameters(self._config_parameters)
