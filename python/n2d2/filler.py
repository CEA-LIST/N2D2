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
from abc import ABC, abstractmethod

class Filler(N2D2_Interface, ABC):
    @abstractmethod
    def __init__(self, **config_parameters):

        if 'datatype' in config_parameters:
            self._datatype = config_parameters.pop('datatype')
        else:
            self._datatype = n2d2.global_variables.default_datatype

        self._model_key = '<' + self._datatype + '>'

        
        N2D2_Interface.__init__(self, **config_parameters)

    def get_type(self):
        return type(self).__name__

    def __str__(self):
        output = self.get_type()
        output += N2D2_Interface.__str__(self)
        return output


class He(Filler):
    """
    Fill with an normal distribution with normalized variance taking into account the rectifier nonlinearity :cite:`He2015`. This filler is sometimes referred as MSRA filler or Kaiming initialization.
    """
    _filler_generators = {
        '<float>': N2D2.HeFiller_float,
    }

    _convention_converter= n2d2.ConventionConverter({
        "variance_norm": "varianceNorm",
        "mean_norm": "meanNorm",
        "scaling": "scaling",
        "datatype": "Datatype",
    })

    def __init__(self, **config_parameters):
        """
        :param datatype: datatype, default='float'
        :type datatype: str, optional
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
            if variance_norm not in self._filler_generators[self._model_key].VarianceNorm.__members__.keys():
                raise n2d2.error_handler.WrongValue("variance_norm", variance_norm,
                                                    ", ".join(self._filler_generators[self._model_key].VarianceNorm.__members__.keys()))
            self._config_parameters["variance_norm"] = self._filler_generators[self._model_key].VarianceNorm.__members__[variance_norm]

        self._parse_optional_arguments(['variance_norm', 'mean_norm', 'scaling'])
        for k, v in self._optional_constructor_arguments.items():
            if k is 'scaling' and not isinstance(v, float):
                raise n2d2.error_handler.WrongInputType("scaling", str(type(v)), ["float"])
            if k is 'mean_norm' and not isinstance(v, float):
                raise n2d2.error_handler.WrongInputType("mean_norm", str(type(v)), ["float"])
        
        self._set_N2D2_object(self._filler_generators[self._model_key](**self.n2d2_function_argument_parser(self._optional_constructor_arguments)))
        self._set_N2D2_parameters(self._config_parameters)
        self.load_N2D2_parameters(self.N2D2())


    def _load_N2D2_optional_parameters(self, N2D2_object):
        self._optional_constructor_arguments['variance_norm'] = N2D2_object.getVarianceNorm()
        self._optional_constructor_arguments['mean_norm'] = N2D2_object.getMeanNorm()
        self._optional_constructor_arguments['scaling'] = N2D2_object.getScaling()


class Normal(Filler):
    """
    Fill with a normal distribution.
    """

    """Static members"""
    _filler_generators = {
        '<float>': N2D2.NormalFiller_float,
    }
    _convention_converter= n2d2.ConventionConverter({
        "mean": "mean",
        "std_dev": "stdDev",
        "datatype": "Datatype",
    })

    def __init__(self, **config_parameters):
        """
        :param datatype: datatype, default='float'
        :type datatype: str, optional
        :param mean: Mean value of the distribution, default=0.0
        :type mean: float, optional
        :param std_dev: Standard deviation of the distribution, default=1.0
        :type std_dev: float, optional
        """
        Filler.__init__(self, **config_parameters)

        self._parse_optional_arguments(['mean', 'std_dev'])
        for k, v in self._optional_constructor_arguments.items():
            if k is 'mean' and not isinstance(v, float):
                raise n2d2.error_handler.WrongInputType("mean", str(type(v)), ["float"])
            if k is 'std_dev' and not isinstance(v, float):
                raise n2d2.error_handler.WrongInputType("std_dev", str(type(v)), ["float"])
        self._set_N2D2_object(self._filler_generators[self._model_key](**self.n2d2_function_argument_parser(self._optional_constructor_arguments)))
        self._set_N2D2_parameters(self._config_parameters)
        self.load_N2D2_parameters(self.N2D2())

    def _load_N2D2_optional_parameters(self, N2D2_object):
        self._optional_constructor_arguments['mean'] = N2D2_object.getMean()
        self._optional_constructor_arguments['std_dev'] = N2D2_object.getStdDev()


class Xavier(Filler):
    """
    Fill with an uniform distribution with normalized variance :cite:`Glorot2010`.
    """

    """Static members"""
    _filler_generators = {
        '<float>': N2D2.XavierFiller_float
    }

    _convention_converter= n2d2.ConventionConverter({
        "variance_norm": "varianceNorm",
        "distribution": "distribution",
        "scaling": "scaling",
        "datatype": "Datatype",
    })

    def __init__(self,  **config_parameters):
        """
        :param datatype: datatype, default='float'
        :type datatype: str, optional
        :param variance_norm: Normalization, can be ``FanIn``, ``Average`` or ``FanOut``, default='FanIn'
        :type variance_norm: str, optional
        :param distribution: Distribution, can be ``Uniform`` or ``Normal``, default='Uniform'
        :type distribution: str, optional
        :param scaling: Scaling factor, default=1.0
        :type scaling: float, optional
        """
        Filler.__init__(self, **config_parameters)

        self._parse_optional_arguments(['variance_norm', 'distribution', 'scaling'])
        for k, v in self._optional_constructor_arguments.items():
            if k is 'scaling' and not isinstance(v, float):
                raise n2d2.error_handler.WrongInputType("scaling", str(type(v)), ["float"])
            if k is 'std_dev' and not isinstance(v, float):
                raise n2d2.error_handler.WrongInputType("std_dev", str(type(v)), ["float"])
            if k is "variance_norm" and v not in self._filler_generators[self._model_key].VarianceNorm.__members__.keys():
                raise n2d2.error_handler.WrongValue("variance_norm", v,
                        ", ".join(self._filler_generators[self._model_key].VarianceNorm.__members__.keys()))
            if k is "distribution" and v not in self._filler_generators[self._model_key].Distribution.__members__.keys():
                raise n2d2.error_handler.WrongValue("distribution", v,
                        ", ".join(self._filler_generators[self._model_key].Distribution.__members__.keys()))

        if 'variance_norm' in self._optional_constructor_arguments:
            self._optional_constructor_arguments['variance_norm'] = \
                self._filler_generators[self._model_key].VarianceNorm.__members__[self._optional_constructor_arguments['variance_norm']]
        if 'distribution' in self._optional_constructor_arguments:
            self._optional_constructor_arguments['distribution'] = \
                self._filler_generators[self._model_key].Distribution.__members__[self._optional_constructor_arguments['distribution']]
        self._set_N2D2_object(self._filler_generators[self._model_key](**self.n2d2_function_argument_parser(self._optional_constructor_arguments)))
        self._set_N2D2_parameters(self._config_parameters)
        self.load_N2D2_parameters(self.N2D2())

    def _load_N2D2_optional_parameters(self, N2D2_object):
        self._optional_constructor_arguments['variance_norm'] = N2D2_object.getVarianceNorm()
        self._optional_constructor_arguments['distribution'] = N2D2_object.getDistribution()
        self._optional_constructor_arguments['scaling'] = N2D2_object.getScaling()


class Constant(Filler):
    """
    Fill with a constant value.
    """

    """Static members"""
    _filler_generators = {
        '<float>': N2D2.ConstantFiller_float
    }
    _convention_converter= n2d2.ConventionConverter({
        "value": "value",
        "datatype": "Datatype",
    })

    def __init__(self, **config_parameters):
        """
        :param datatype: datatype, default='float'
        :type datatype: str, optional
        :param value: Value for the filling, default=0.0
        :type value: float, optional
        """
        Filler.__init__(self, **config_parameters)

        self._parse_optional_arguments(['value'])
        for k, v in self._optional_constructor_arguments.items():
            if k is 'value' and not isinstance(v, float):
                raise n2d2.error_handler.WrongInputType("value", str(type(v)), ["float"])
        self._set_N2D2_object(self._filler_generators[self._model_key](**self.n2d2_function_argument_parser(self._optional_constructor_arguments)))
        self._set_N2D2_parameters(self._config_parameters)
        self.load_N2D2_parameters(self.N2D2())

    def _load_N2D2_optional_parameters(self, N2D2_object):
        self._optional_constructor_arguments['value'] = N2D2_object.getValue()
