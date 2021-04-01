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

# TODO: Make obligatory to pass model and datatype of cell in cell constructor
class ActivationFunction(N2D2_Interface):

    def __init__(self, **config_parameters):
        if 'model' in config_parameters:
            self._model = config_parameters.pop('model')
        else:
            self._model = n2d2.global_variables.default_model
        if 'datatype' in config_parameters:
            self._datatype = config_parameters.pop('datatype')
        else:
            self._datatype = n2d2.global_variables.default_datatype

        self._model_key = self._model + '<' + self._datatype + '>'

        N2D2_Interface.__init__(self, **config_parameters)

    def get_quantizer(self):
        return self._config_parameters['quantizer']


    def get_type(self):
        return self._N2D2_object.getType()

    def __str__(self):
        output = self.get_type()
        output += N2D2_Interface.__str__(self)
        return output


class Linear(ActivationFunction):
    """
    Linear activation function for n2d2.
    """
    _linear_activation_generators = {
        'Frame<float>': N2D2.LinearActivation_Frame_float,
        'Frame_CUDA<float>': N2D2.LinearActivation_Frame_CUDA_float
    }

    def __init__(self, **config_parameters):
        ActivationFunction.__init__(self, **config_parameters)
        # No optional constructor arguments
        self._N2D2_object = self._linear_activation_generators[self._model_key]()
        for key, value in self._config_parameters.items():
            if key is 'quantizer':
                self._N2D2_object.setQuantizer(value.N2D2())
            else:
                self._set_N2D2_parameter(self.python_to_n2d2_convention(key), value)


class Rectifier(ActivationFunction):
    """
    Rectifier activation function for n2d2.
    """
    _rectifier_activation_generators = {
        'Frame<float>': N2D2.RectifierActivation_Frame_float,
        'Frame_CUDA<float>': N2D2.RectifierActivation_Frame_CUDA_float,
    }

    def __init__(self, **config_parameters):
        ActivationFunction.__init__(self, **config_parameters)
        # No optional constructor arguments
        self._N2D2_object = self._rectifier_activation_generators[self._model_key]()
        for key, value in self._config_parameters.items():
            if key is 'quantizer':
                self._N2D2_object.setQuantizer(value.N2D2())
            else:
                self._set_N2D2_parameter(self.python_to_n2d2_convention(key), value)


class Tanh(ActivationFunction):
    """
    Tanh activation function for n2d2.
    """
    _tanh_activation_generators = {
        'Frame<float>': N2D2.TanhActivation_Frame_float,
        'Frame_CUDA<float>': N2D2.TanhActivation_Frame_CUDA_float,
    }

    def __init__(self, **config_parameters):
        ActivationFunction.__init__(self, **config_parameters)
        # No optional constructor arguments
        self._N2D2_object = self._tanh_activation_generators[self._model_key]()
        for key, value in self._config_parameters.items():
            if key is 'quantizer':
                self._N2D2_object.setQuantizer(value.N2D2())
            else:
                self._set_N2D2_parameter(self.python_to_n2d2_convention(key), value)

