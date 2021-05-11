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


_activation_parameters = {
        "quantizer": "Quantizer"
}
class ActivationFunction(N2D2_Interface, ABC):
    
    # Cell_frame_parameter contains the parameters from cell_parameter
    @abstractmethod
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

    @classmethod
    def create_from_N2D2_object(cls, N2D2_object):
        activation = cls(**cls.load_N2D2_parameters(N2D2_object), from_arguments=False)
        activation._N2D2_object = N2D2_object
        quantizer = activation._N2D2_object.getQuantizer()
        if quantizer:
            activation._config_parameters['quantizer'] = \
                n2d2.quantizer.ActivationQuantizer.create_from_N2D2_object(quantizer)
        return activation

    def get_quantizer(self):
        if 'quantizer' in self._config_parameters:
            return self._config_parameters['quantizer']
        else:
            raise RuntimeError("No Quantizer in activation")

    def set_quantizer(self, quantizer):
        if 'quantizer' in self._config_parameters:
            raise RuntimeError("Quantizer already exists in activation")
        else:
            self._config_parameters['quantizer'] = quantizer
            self._N2D2_object.setQuantizer(self._config_parameters['quantizer'].N2D2())

    def get_type(self):
        return type(self).__name__

    def __str__(self):
        output = self.get_type()
        output += N2D2_Interface.__str__(self)
        return output


class Linear(ActivationFunction):
    """
    Linear activation function.
    """
    _linear_activation_generators = {
        'Frame<float>': N2D2.LinearActivation_Frame_float,
        'Frame_CUDA<float>': N2D2.LinearActivation_Frame_CUDA_float
    }
    _parameters = {
        "clipping": "Clipping",
    }
    _parameters.update(_activation_parameters)
    _convention_converter= n2d2.ConventionConverter(_parameters)

    def __init__(self, from_arguments=True, **config_parameters):
        """
        :param quantizer: Quantizer
        :type quantizer: :py:class:`n2d2.quantizer.ActivationQuantizer`, optional
        """
        ActivationFunction.__init__(self, **config_parameters)

        if from_arguments:
            # No optional constructor arguments
            self._set_N2D2_object(self._linear_activation_generators[self._model_key]())
            for key, value in self._config_parameters.items():
                if key is 'quantizer':
                    self._N2D2_object.setQuantizer(value.N2D2())
                else:
                    self._set_N2D2_parameter(self.python_to_n2d2_convention(key), value)


class Rectifier(ActivationFunction):
    """
    Rectifier or ReLU activation function.
    """
    _rectifier_activation_generators = {
        'Frame<float>': N2D2.RectifierActivation_Frame_float,
        'Frame_CUDA<float>': N2D2.RectifierActivation_Frame_CUDA_float,
    }
    _parameters = {
        "leak_slope": "LeakSlope",
        "clipping": "Clipping",
    }
    _parameters.update(_activation_parameters)
    _convention_converter= n2d2.ConventionConverter(_parameters)

    def __init__(self, from_arguments=True, **config_parameters):
        """
        :param leak_slope: Leak slope for negative inputs, default=0.0
        :type leak_slope: float, optional
        :param clipping: Clipping value for positive outputs, default=0.0
        :type clipping: float, optional
        :param quantizer: Quantizer
        :type quantizer: :py:class:`n2d2.quantizer.ActivationQuantizer`, optional
        """
        ActivationFunction.__init__(self, **config_parameters)

        if from_arguments:
            # No optional constructor arguments
            self._set_N2D2_object(self._rectifier_activation_generators[self._model_key]())
            for key, value in self._config_parameters.items():
                if key is 'quantizer':
                    self._N2D2_object.setQuantizer(value.N2D2())
                else:
                    self._set_N2D2_parameter(self.python_to_n2d2_convention(key), value)


class Tanh(ActivationFunction):
    r"""
    Tanh activation function.

    Computes :math:`y = tanh(\alpha.x)`.
    """
    _tanh_activation_generators = {
        'Frame<float>': N2D2.TanhActivation_Frame_float,
        'Frame_CUDA<float>': N2D2.TanhActivation_Frame_CUDA_float,
    }
    _parameters = {
        "alpha": "Alpha",
    }
    _parameters.update(_activation_parameters)
    _convention_converter= n2d2.ConventionConverter(_parameters)

    def __init__(self, from_arguments=True, **config_parameters):
        r"""
        :param alpha: :math:`\alpha` parameter, default=1.0
        :type alpha: float, optional
        :param quantizer: Quantizer
        :type quantizer: :py:class:`n2d2.quantizer.ActivationQuantizer`, optional
        """
        ActivationFunction.__init__(self, **config_parameters)

        if from_arguments:
            # No optional constructor arguments
            self._set_N2D2_object(self._tanh_activation_generators[self._model_key]())
            for key, value in self._config_parameters.items():
                if key is 'quantizer':
                    self._N2D2_object.setQuantizer(value.N2D2())
                else:
                    self._set_N2D2_parameter(self.python_to_n2d2_convention(key), value)


