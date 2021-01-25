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

class Activation(N2D2_Interface):

    def __init__(self, **config_parameters):
        if 'Model' in config_parameters:
            self._Model = config_parameters.pop('Model')
        else:
            self._Model = n2d2.global_variables.default_DeepNet.get_model()
        if 'DataType' in config_parameters:
            self._DataType = config_parameters.pop('DataType')
        else:
            self._DataType = n2d2.global_variables.default_DeepNet.get_datatype()

        self._model_key = self._Model + '<' + self._DataType + '>'

        N2D2_Interface.__init__(self, **config_parameters)


    def get_type(self):
        return self._N2D2_object.getType()

    def __str__(self):
        output = self.get_type()
        output += N2D2_Interface.__str__(self)
        return output


class Linear(Activation):
    """
    Linear activation function for n2d2.
    """
    _linear_activation_generators = {
        'Frame<float>': N2D2.LinearActivation_Frame_float,
        'Frame_CUDA<float>': N2D2.LinearActivation_Frame_CUDA_float
    }

    def __init__(self, **config_parameters):
        Activation.__init__(self, **config_parameters)
        # No optional constructor arguments
        self._N2D2_object = self._linear_activation_generators[self._model_key]()
        self._set_N2D2_parameters(self._config_parameters)


class Rectifier(Activation):
    """
    Rectifier activation function for n2d2.
    """
    _rectifier_activation_generators = {
        'Frame<float>': N2D2.RectifierActivation_Frame_float,
        'Frame_CUDA<float>': N2D2.RectifierActivation_Frame_CUDA_float,
    }

    def __init__(self, **config_parameters):
        Activation.__init__(self, **config_parameters)
        # No optional constructor arguments
        self._N2D2_object = self._rectifier_activation_generators[self._model_key]()
        self._set_N2D2_parameters(self._config_parameters)


class Tanh(Activation):
    """
    Tanh activation function for n2d2.
    """
    _tanh_activation_generators = {
        'Frame<float>': N2D2.TanhActivation_Frame_float,
        'Frame_CUDA<float>': N2D2.TanhActivation_Frame_CUDA_float,
    }

    def __init__(self, **config_parameters):
        Activation.__init__(self, **config_parameters)
        # No optional constructor arguments
        self._N2D2_object = self._tanh_activation_generators[self._model_key]()
        self._set_N2D2_parameters(self._config_parameters)

