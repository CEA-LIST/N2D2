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
from n2d2.parameterizable import Parameterizable

class Activation(Parameterizable):

    def __init__(self):
        Parameterizable.__init__(self)
        self._model_config_parameters = {}
        self._model_parameter_definitions = {}
        self._model_key = ""

    def _set_model_config_parameters(self, Model, model_config_parameters):
        if Model in self._model_parameter_definitions:
            self._model_config_parameters = self._model_parameter_definitions[Model]
        self._set_parameters(self._model_config_parameters, model_config_parameters)
        self._set_N2D2_parameters(self._model_config_parameters)


    def __str__(self):
        output = str(self._activation_parameters)
        # output += "\n"
        return output


class Linear(Activation):
    """Static members"""
    _linear_activation_generators = {
        'Frame<float>': N2D2.LinearActivation_Frame_float,
        'Frame_CUDA<float>': N2D2.LinearActivation_Frame_CUDA_float
    }

    def __init__(self, **config_parameters):
        Activation.__init__(self)

        # No model specific parameters

    def generate_model(self, Model='Frame', DataType='float', **model_config_parameters):
        self._model_key = Model + '<' + DataType + '>'

        self._N2D2_object = self._linear_activation_generators[self._model_key]()

        self._set_model_config_parameters(Model, model_config_parameters)

    def __str__(self):
        output = "LinearActivation(" + self._model_key + "): "
        output += Activation.__str__(self)
        return output


class Rectifier(Activation):
    """Static members"""
    _relu_activation_generators = {
        'Frame<float>': N2D2.RectifierActivation_Frame_float,
        'Frame_CUDA<float>': N2D2.RectifierActivation_Frame_CUDA_float,
    }

    def __init__(self, **config_parameters):
        Activation.__init__(self)

    def generate_model(self, Model='Frame', DataType='float', **model_config_parameters):
        self._model_key = Model + '<' + DataType + '>'

        self._N2D2_object = self._relu_activation_generators[self._model_key]()

        self._set_model_config_parameters(Model, model_config_parameters)

    def __str__(self):
        output = "RectifierActivation(" + self._model_key + "): "
        output += Activation.__str__(self)
        return output


class Tanh(Activation):
    """Static members"""
    _tanh_activation_generators = {
        'Frame<float>': N2D2.TanhActivation_Frame_float,
        'Frame_CUDA<float>': N2D2.TanhActivation_Frame_CUDA_float,
    }

    def __init__(self, **config_parameters):
        Activation.__init__(self)


    def generate_model(self, Model='Frame', DataType='float', **model_config_parameters):
        self._model_key = Model + '<' + DataType + '>'

        self._N2D2_object = self._tanh_activation_generators[self._model_key]()

        self._set_model_config_parameters(Model, model_config_parameters)

    def __str__(self):
        output = "TanhActivation(" + self._model_key + "): "
        output += Activation.__str__(self)
        return output