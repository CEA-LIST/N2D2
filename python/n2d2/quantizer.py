"""
    (C) Copyright 2021 CEA LIST. All Rights Reserved.
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
from n2d2.n2d2_interface import N2D2_Interface
import n2d2.global_variables


class Quantizer(N2D2_Interface):
    _convention_converter= n2d2.ConventionConverter({
        "range": "Range",
        "apply_scaling": "ApplyScaling",
        "apply_quantization": "ApplyQuantization",
        "quant_mode": "QuantMode",
        "alpha": "Alpha", # TODO : alpha is only for SATAct ...
    })
    
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

    def set_range(self, integer_range):
        self._N2D2_object.setRange(integer_range)

    def get_type(self):
        return type(self).__name__

    def __str__(self):
        output = self.get_type()
        output += N2D2_Interface.__str__(self)
        return output


class CellQuantizer(Quantizer):

    def __init__(self, **config_parameters):
        # TODO : add a type check for parameters range, solver and mode. (for mode also add a value check)
        Quantizer.__init__(self, **config_parameters)

    def add_weights(self, weights, diff_weights):
        """
        :arg weights: Biases
        :param weights: :py:class:`n2d2.Tensor`
        :arg diff_weights: Diff Biases
        :param diff_weights: :py:class:`n2d2.Tensor`
        """
        if not isinstance(diff_weights, n2d2.Tensor):
            raise n2d2.error_handler("diff_weights", str(type(diff_weights)), ["n2d2.Tensor"])
        if not isinstance(weights, n2d2.Tensor):
            raise n2d2.error_handler("weights", str(type(weights)), ["n2d2.Tensor"])
        self.N2D2().addWeights(weights.N2D2(), diff_weights.N2D2())

    def add_biases(self, biases, diff_biases):
        """
        :arg biases: Biases
        :param biases: :py:class:`n2d2.Tensor`
        :arg diff_biases: Diff Biases
        :param diff_biases: :py:class:`n2d2.Tensor`
        """
        if not isinstance(diff_biases, n2d2.Tensor):
            raise n2d2.error_handler("diff_biases", type(diff_biases) ["n2d2.Tensor"])
        if not isinstance(biases, n2d2.Tensor):
            raise n2d2.error_handler("biases", type(biases) ["n2d2.Tensor"])
        self.N2D2().addBiases(biases.N2D2(), diff_biases.N2D2())

class ActivationQuantizer(Quantizer):

    def __init__(self, **config_parameters):
        Quantizer.__init__(self, **config_parameters)


class SATCell(CellQuantizer):
    """
    Scale Adjust Training (SAT) weight quantizer.
    """
    _quantizer_generators = {
        'Frame<float>': N2D2.SATQuantizerCell_Frame_float,
        'Frame_CUDA<float>': N2D2.SATQuantizerCell_Frame_CUDA_float
    }

    def __init__(self, from_arguments=True, **config_parameters):
        """
        :param range: Range of Quantization, can be ``1`` for binary, ``255`` for 8-bits etc.., default=255
        :type range: int, optional
        :param solver: Type of the Solver for learnable quantization parameters, default= :py:class:`n2d2.solver.SGD`
        :type solver: :py:class:`n2d2.solver.Solver`, optional
        :param mode: Type of quantization Mode, can be ``Default`` or ``Integer``, default=``Default``
        :type mode: string, optional
        :param apply_quantization: Use ``True`` to enable quantization, if ``False`` parameters will be clamped between [-1.0,1.0], default=``True``
        :type apply_quantization: bool, optional
        :param apply_scaling: Use true to scale the parameters as described in the SAT paper, default=``False``
        :type apply_scaling: bool, optional
        """
        CellQuantizer.__init__(self, **config_parameters)
        if from_arguments:
            # No optional constructor arguments
            self._set_N2D2_object(self._quantizer_generators[self._model_key]())
            self._set_N2D2_parameters(self._config_parameters)

    """
    Access the quantized weights of the cell the quantizer is attached to.
    """
    def get_quantized_weights(self, input_idx):
        return n2d2.Tensor.from_N2D2(self.N2D2().getQuantizedWeights(input_idx))

    """
    Access the quantized weights of the cell the quantizer is attached to.
    """
    def get_quantized_biases(self):
        return n2d2.Tensor.from_N2D2(self.N2D2().getQuantizedBiases())

    def set_scaling(self, status):
        """
        :arg status: Status
        :param status: boolean
        """
        if not isinstance(status, bool):
            raise n2d2.error_handler("status", type(status) ["bool"])
        self.N2D2().setScaling(status)

    def set_quantization(self, status):
        """
        :arg status: Status
        :param status: boolean
        """
        if not isinstance(status, bool):
            raise n2d2.error_handler("status", type(status) ["bool"])
        self.N2D2().setQuantization(status)

class SATAct(ActivationQuantizer):
    """
    Scale Adjust Training (SAT) activation quantizer.
    """
    _quantizer_generators = {
        'Frame<float>': N2D2.SATQuantizerActivation_Frame_float,
        'Frame_CUDA<float>': N2D2.SATQuantizerActivation_Frame_CUDA_float
    }

    def __init__(self, from_arguments=True, **config_parameters):
        """
        :param range: Range of Quantization, can be ``1`` for binary, ``255`` for 8-bits etc.., default=255
        :type range: int, optional
        :param solver: Type of the Solver for learnable quantization parameters, default= :py:class:`n2d2.solver.SGD`
        :type solver: :py:class:`n2d2.solver.Solver`, optional
        :param alpha: Initial value of the learnable alpha parameter, default=8.0
        :type alpha: float, optional
        """
        ActivationQuantizer.__init__(self, **config_parameters)

        if from_arguments:
            # No optional constructor arguments
            self._N2D2_object = self._quantizer_generators[self._model_key]()

            """Set and initialize here all complex cells members"""
            for key, value in self._config_parameters.items():
                if key is 'solver':
                    self._N2D2_object.setSolver(value.N2D2())
                else:
                    self._set_N2D2_parameter(self.python_to_n2d2_convention(key), value)

    """
    def set_solver(self, solver):
        if 'solver' in self._config_parameters:
            print("Note: Replacing existing solver in SATAct quantizer")
        self._config_parameters['solver'] = solver
        self._N2D2_object.setSolver(self._config_parameters['solver'].N2D2())
    """

    """
    Access the full precision activations of the activation function.
    Note: This may be empty for some Quantizers if they are run exclusively in inference mode
    """
    def get_full_precision_activations(self):
        return n2d2.Tensor.from_N2D2(self.N2D2().getFullPrecisionActivations())
