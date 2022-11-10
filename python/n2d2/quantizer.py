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
from abc import ABC, abstractmethod

from n2d2 import ConventionConverter, check_types, Tensor
from n2d2.n2d2_interface import N2D2_Interface, Options
from n2d2.error_handler import WrongValue, WrongInputType
import n2d2.global_variables as gb
from n2d2.solver import Solver
from n2d2.deepnet import associate_provider_to_deepnet
import n2d2


def fuse_qat(deep_net, provider, act_scaling_mode, w_mode="NONE", b_mode="NONE", c_mode="NONE"):
    """This method allow you to fuse BatchNorm parameters into Conv layers once you have trained your model.

    :param deep_net: DeepNet to fuse.
    :type deep_net: :py:class:`n2d2.DeepNet`
    :param provider: Data provider used
    :type provider: :py:class:`n2d2.Provider`
    :param act_scaling_mode: Scaling mode for activation can be ``NONE``, ``FLOAT_MULT``, ``FIXED_MULT16``, ``FIXED_MULT32``, ``SINGLE_SHIFT``, ``DOUBLE_SHIFT``
    :type act_scaling_mode: str
    :param w_mode: Can be ``NONE`` or ``RINTF``, default="NONE"
    :type w_mode: str, optional
    :param b_mode: Can be ``NONE`` or ``RINTF``, default="NONE"
    :type b_mode: str, optional
    :param c_mode: Can be ``NONE`` or ``RINTF``, default="NONE"
    :type c_mode: str, optional
    """
    # Type check
    if not isinstance(deep_net, n2d2.deepnet.DeepNet):
        raise WrongInputType("deep_net", type(deep_net), ["n2d2.deepnet.DeepNet"])
    if not isinstance(provider, n2d2.provider.Provider):
        raise WrongInputType("provider", type(provider), ["n2d2.provider.Provider"])
    if not isinstance(act_scaling_mode, str):
        raise WrongInputType("act_scaling_mode", type(act_scaling_mode), ["str"])
    if not isinstance(w_mode, str):
        raise WrongInputType("w_mode", type(w_mode), ["str"])
    if not isinstance(b_mode, str):
        raise WrongInputType("b_mode", type(b_mode), ["str"])
    if not isinstance(c_mode, str):
        raise WrongInputType("c_mode", type(c_mode), ["str"])

    deep_net_qat = N2D2.DeepNetQAT(deep_net.N2D2())
    deep_net_qat.fuseQATGraph(
        provider.N2D2(),
        N2D2.ScalingMode.__members__[act_scaling_mode],
        N2D2.WeightsApprox.__members__[w_mode],
        N2D2.WeightsApprox.__members__[b_mode],
        N2D2.WeightsApprox.__members__[c_mode]
    )
    return deep_net_qat

@check_types
def PTQ(deepnet_cell,
        nb_bits:int,
        nb_sitmuli:int=-1,
        provider:n2d2.provider.Provider=None,
        no_unsigned:bool=False,
        cross_layer_equalization:bool=True,
        wt_clipping_mode:str="NONE",
        act_clipping_mode:str="MSE",
        act_scaling_mode:str="FLOAT_MULT",
        act_rescale_per_output:bool=False,
        **kwargs):
    """
    :param nb_bits: Number of bits per weight for exports (can be for example `-16` for float 16 bits or `8` int 8 bits)
    :type nb_bits: int
    :param nb_sitmuli: The number of stimuli used for the calibration (``0`` = no calibration, ``-1`` = use the full test dataset), default=-1
    :type nb_sitmuli: int, optional
    :param provider: Data provider to use for calibration, default=None
    :type provider: :py:class:`n2d2.provider.DataProvider`, optional
    :param no_unsigned: If True, disable the use of unsigned data type in integer calibration, default=False
    :type no_unsigned: bool, optional
    :param cross_layer_equalization: If True, disable the use of cross layer equalization in integer calibration, default=False
    :type cross_layer_equalization: bool, optional
    :param wt_clipping_mode: Weights clipping mode on calibration, can be ``NONE``, ``MSE`` or ``KL_DIVERGENCE``, default="NONE"
    :type wt_clipping_mode: str, optional
    :param act_clipping_mode: activation clipping mode on calibration, can be ``NONE``, ``MSE`` or ``KL_DIVERGENCE`` or ``Quantile``, default="MSE"
    :type act_clipping_mode: str, optional
    :param act_rescale_per_output: rescale activation per output on export, default=False
    :type act_rescale_per_output: bool, optional
    """
    no_unsigned = kwargs.get("export_no_unsigned", no_unsigned)
    cross_layer_equalization = not kwargs.get("export_no_cross_layer_equalization", not cross_layer_equalization)
    nb_sitmuli = kwargs.get("calibration", nb_sitmuli)

    if act_clipping_mode not in N2D2.ClippingMode.__members__.keys():
        raise WrongValue("act_clipping_mode", act_clipping_mode, ", ".join(N2D2.ClippingMode.__members__.keys()))
    N2D2_act_clipping_mode = N2D2.ClippingMode.__members__[act_clipping_mode]
    if wt_clipping_mode not in N2D2.ClippingMode.__members__.keys():
        raise WrongValue("wt_clipping_mode", wt_clipping_mode, ", ".join(N2D2.ClippingMode.__members__.keys()))
    N2D2_wt_clipping_mode = N2D2.ClippingMode.__members__[wt_clipping_mode]
    if act_scaling_mode not in N2D2.ScalingMode.__members__.keys():
        raise WrongValue("act_scaling_mode", act_scaling_mode, ", ".join(N2D2.ScalingMode.__members__.keys()))
    N2D2_act_scaling_mode = N2D2.ScalingMode.__members__[act_scaling_mode]
    parameters = Options(
        nb_bits=nb_bits,
        export_no_unsigned=no_unsigned,
        calibration=nb_sitmuli,
        qat_SAT=False,
        export_no_cross_layer_equalization=not cross_layer_equalization,
        wt_clipping_mode=N2D2_wt_clipping_mode,
        act_clipping_mode=N2D2_act_clipping_mode,
        act_scaling_mode=N2D2_act_scaling_mode,
        act_rescale_per_output=act_rescale_per_output,
    ).N2D2()

    N2D2_deepnet = deepnet_cell.get_embedded_deepnet().N2D2()
    N2D2_deepnet.initialize()

    if provider is not None:
        associate_provider_to_deepnet(N2D2_deepnet, provider.N2D2())
    
    if N2D2_deepnet.getStimuliProvider().getDatabase().getNbStimuli() == 0:
        raise RuntimeError("cannot calibrate a network without data, use a provider with data.")

    if N2D2_deepnet.getDatabase().getNbStimuli(N2D2.Database.StimuliSet.__members__["Validation"]) > 0:
        N2D2_deepnet.exportNetworkFreeParameters("weights_validation")
    else:
        N2D2_deepnet.exportNetworkFreeParameters("weights")

    N2D2.calibNetwork(parameters, N2D2_deepnet)


class Quantizer(N2D2_Interface, ABC):

    @abstractmethod
    def __init__(self, **config_parameters):
        if 'model' in config_parameters:
            self._model = config_parameters.pop('model')
        else:
            self._model = gb.default_model
        if 'datatype' in config_parameters:
            self._datatype = config_parameters.pop('datatype')
        else:
            self._datatype = gb.default_datatype

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


class CellQuantizer(Quantizer, ABC):
    @abstractmethod
    def __init__(self, **config_parameters):
        Quantizer.__init__(self, **config_parameters)

    @check_types
    def add_weights(self, weights:Tensor, diff_weights:Tensor):
        """
        :arg weights: Weights
        :param weights: :py:class:`n2d2.Tensor`
        :arg diff_weights: Diff Weights
        :param diff_weights: :py:class:`n2d2.Tensor`
        """
        self.N2D2().addWeights(weights.N2D2(), diff_weights.N2D2())

    @check_types
    def add_biases(self, biases:Tensor, diff_biases:Tensor):
        """
        :arg biases: Biases
        :param biases: :py:class:`n2d2.Tensor`
        :arg diff_biases: Diff Biases
        :param diff_biases: :py:class:`n2d2.Tensor`
        """
        self.N2D2().addBiases(biases.N2D2(), diff_biases.N2D2())

class ActivationQuantizer(Quantizer, ABC):
    @abstractmethod
    def __init__(self, **config_parameters):
        Quantizer.__init__(self, **config_parameters)

class SATCell(CellQuantizer):
    """
    Scale Adjust Training (SAT) weight quantizer.
    """
    _quantizer_generators = {
        'Frame<float>': N2D2.SATQuantizerCell_Frame_float,
    }
    if gb.cuda_available:
        _quantizer_generators.update({
            'Frame_CUDA<float>': N2D2.SATQuantizerCell_Frame_CUDA_float
        })
    _convention_converter = ConventionConverter({
        "apply_scaling": "ApplyScaling",
        "apply_quantization": "ApplyQuantization",
        "quant_mode": "QuantMode",
        "range": "Range",
    })
    def __init__(self, **config_parameters):
        """
        :param range: Range of Quantization, can be ``1`` for binary, ``255`` for 8-bits etc.., default=255
        :type range: int, optional
        :param quant_mode: Type of quantization Mode, can be ``Default`` or ``Integer``, default=``Default``
        :type quant_mode: string, optional
        :param apply_quantization: Use ``True`` to enable quantization, if ``False`` parameters will be clamped between [-1.0,1.0], default=``True``
        :type apply_quantization: bool, optional
        :param apply_scaling: Use true to scale the parameters as described in the SAT paper, default=``False``
        :type apply_scaling: bool, optional
        """
        CellQuantizer.__init__(self, **config_parameters)
        if "quant_mode" in config_parameters:
            print(", ".join(self._quantizer_generators[self._model_key].QuantMode.__members__.keys()))
            quant_mode = config_parameters["quant_mode"]
            if quant_mode not in self._quantizer_generators[self._model_key].QuantMode.__members__.keys():
                raise WrongValue("quant_mode", quant_mode,
                        ", ".join(self._quantizer_generators[self._model_key].QuantMode.__members__.keys()))

        # No optional constructor arguments
        self._set_N2D2_object(self._quantizer_generators[self._model_key]())
        self._set_N2D2_parameters(self._config_parameters)
        self.load_N2D2_parameters(self.N2D2())


    def get_quantized_weights(self, input_idx):
        """
        Access the quantized weights of the cell the quantizer is attached to.
        """
        return n2d2.Tensor.from_N2D2(self.N2D2().getQuantizedWeights(input_idx))


    def get_quantized_biases(self):
        """
        Access the quantized weights of the cell the quantizer is attached to.
        """
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


class LSQCell(CellQuantizer):
    """
    Learned Step size Quantization (LSQ) weight quantizer.
    """
    _quantizer_generators = {
    }
    if gb.cuda_available:
        _quantizer_generators.update({
            'Frame_CUDA<float>': N2D2.LSQQuantizerCell_Frame_CUDA_float
        })
    _convention_converter = ConventionConverter({
        "range": "Range",
        "solver": "Solver",
        "set_opt_init_step_size":"SetOptInitStepSize",
        "step_size":"StepSize",
        "quant_mode": "QuantMode", # To remove later we never use it but we get it from inheritance of QuantizerCell
    })
    def __init__(self, **config_parameters):
        """
        :param range: Range of Quantization, can be ``1`` for binary, ``255`` for 8-bits etc.., default=255
        :type range: int, optional
        :param solver: Type of the Solver for learnable quantization parameters, default= :py:class:`n2d2.solver.SGD`
        :type solver: :py:class:`n2d2.solver.Solver`, optional
        """
        CellQuantizer.__init__(self, **config_parameters)
        self._N2D2_object = self._quantizer_generators[self._model_key]()

        """Set and initialize here all complex cells members"""
        for key, value in self._config_parameters.items():
            if key == 'solver':
                if isinstance(value, Solver):
                    self._N2D2_object.setSolver(value.N2D2())
                else:
                    raise WrongInputType("solver", str(type(value)),
                                                            [str(Solver)])
            else:
                self._set_N2D2_parameter(self._python_to_n2d2_convention(key), value)
        # No optional constructor arguments

        self.load_N2D2_parameters(self.N2D2())

    def set_solver(self, solver):
        self._config_parameters['solver'] = solver
        self._N2D2_object.setSolver(self._config_parameters['solver'].N2D2())

    def get_solver(self):
        return self._config_parameters['solver']

    def __setattr__(self, key: str, value) -> None:
        if key == 'solver':
            self.set_solver(value)
        else:
            super().__setattr__(key, value)


class SATAct(ActivationQuantizer):
    """
    Scale Adjust Training (SAT) activation quantizer.
    """
    _quantizer_generators = {
        'Frame<float>': N2D2.SATQuantizerActivation_Frame_float,
    }
    if gb.cuda_available:
        _quantizer_generators.update({
            'Frame_CUDA<float>': N2D2.SATQuantizerActivation_Frame_CUDA_float
        })
    _convention_converter = ConventionConverter({
        "range": "Range",
        "alpha": "Alpha",
        "desc_rule": "DescRule",
        "end_rand_IT": "EndRandIT",
        "rand_range": "RandRange",
        "start_rand_IT": "StartRandIT"
    })

    def __init__(self, **config_parameters):
        """
        :param range: Range of Quantization, can be ``1`` for binary, ``255`` for 8-bits etc.., default=255
        :type range: int, optional
        :param solver: Type of the Solver for learnable quantization parameters, default= :py:class:`n2d2.solver.SGD`
        :type solver: :py:class:`n2d2.solver.Solver`, optional
        :param alpha: Initial value of the learnable alpha parameter, default=8.0
        :type alpha: float, optional
        """
        ActivationQuantizer.__init__(self, **config_parameters)

        # No optional constructor arguments
        self._N2D2_object = self._quantizer_generators[self._model_key]()

        """Set and initialize here all complex cells members"""
        for key, value in self._config_parameters.items():
            if key is 'solver':
                if isinstance(value, n2d2.solver.Solver):
                    self._N2D2_object.setSolver(value.N2D2())
                else:
                    raise WrongInputType("solver", str(type(value)),
                                                            [str(n2d2.solver.Solver)])
            else:
                self._set_N2D2_parameter(self._python_to_n2d2_convention(key), value)
        self.load_N2D2_parameters(self.N2D2())

    @classmethod
    def _get_N2D2_complex_parameters(cls, N2D2_object):
        parameters = {}
        parameters['solver'] = \
            n2d2.converter.from_N2D2_object(N2D2_object.getSolver())
        return parameters

    def __setattr__(self, key: str, value) -> None:
        if key is 'filler':
            self.set_filler(value)
        elif key is 'solver':
            self.set_solver(value)
        else:
            return super().__setattr__(key, value)

    def set_solver(self, solver):
        self._config_parameters['solver'] = solver
        self._N2D2_object.setSolver(self._config_parameters['solver'].N2D2())

    def get_solver(self):
        return self._config_parameters['solver']

    """
    Access the full precision activations of the activation function.
    Note: This may be empty for some Quantizers if they are run exclusively in inference mode
    """
    def get_full_precision_activations(self):
        return n2d2.Tensor.from_N2D2(self.N2D2().getFullPrecisionActivations())


class LSQAct(ActivationQuantizer):
    """
    Learned Step size Quantization (LSQ) activation quantizer.
    """
    _quantizer_generators = {
    }
    if gb.cuda_available:
        _quantizer_generators.update({
            'Frame_CUDA<float>': N2D2.LSQQuantizerActivation_Frame_CUDA_float
        })
    _convention_converter = ConventionConverter({
        "range": "Range",
        "solver": "Solver",
        "set_opt_init_step_size":"SetOptInitStepSize",
        "step_size":"StepSize",
        "quant_mode": "QuantMode", # To remove later we never use it but we get it from inheritance of QuantizerCell
    })
    def __init__(self, **config_parameters):
        """
        :param range: Range of Quantization, can be ``1`` for binary, ``255`` for 8-bits etc.., default=255
        :type range: int, optional
        :param solver: Type of the Solver for learnable quantization parameters, default= :py:class:`n2d2.solver.SGD`
        :type solver: :py:class:`n2d2.solver.Solver`, optional
        """
        ActivationQuantizer.__init__(self, **config_parameters)
        self._N2D2_object = self._quantizer_generators[self._model_key]()

        """Set and initialize here all complex cells members"""
        for key, value in self._config_parameters.items():
            if key == 'solver':
                if isinstance(value, Solver):
                    self._N2D2_object.setSolver(value.N2D2())
                else:
                    raise WrongInputType("solver", str(type(value)),
                                                            [str(Solver)])
            else:
                self._set_N2D2_parameter(self._python_to_n2d2_convention(key), value)
        # No optional constructor arguments
        self.load_N2D2_parameters(self.N2D2())

    def set_solver(self, solver):
        self._config_parameters['solver'] = solver
        self._N2D2_object.setSolver(self._config_parameters['solver'].N2D2())

    def get_solver(self):
        return self._config_parameters['solver']

    def __setattr__(self, key: str, value) -> None:
        if key == 'solver':
            self.set_solver(value)
        else:
            super().__setattr__(key, value)
