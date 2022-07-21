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

from n2d2 import ConventionConverter, Tensor, global_variables, error_handler
from n2d2.filler import Filler
from n2d2.solver import Solver
from n2d2.cells.cell import Trainable
from n2d2.cells.nn.abstract_cell import (NeuralNetworkCell,
                                         _cell_frame_parameters)

from n2d2.converter import from_N2D2_object
from n2d2.error_handler import deprecated
from n2d2.typed import ModelDatatyped
from n2d2.utils import inherit_init_docstring
from n2d2.activation import Linear

@inherit_init_docstring()
class Deconv(NeuralNetworkCell, ModelDatatyped, Trainable):
    """
    Deconvolution layer.
    """
    mappable = True
    _N2D2_constructors = {
        'Frame<float>': N2D2.DeconvCell_Frame_float,
    }
    if global_variables.cuda_available:
        _N2D2_constructors.update({
            'Frame_CUDA<float>': N2D2.DeconvCell_Frame_CUDA_float,
        })
    _parameters = {
        "no_bias": "NoBias",
        "back_propagate": "BackPropagate",
        "weights_export_format": "WeightsExportFormat",
        "weights_export_flip": "WeightsExportFlip",
        "outputs_remap": "OutputsRemap",
        "kernel_dims": "kernelDims",
        "sub_sample_dims": "subSampleDims",
        "stride_dims": "strideDims",
        "padding_dims": "paddingDims",
        "dilation_dims": "dilationDims",
        "ext_padding_dims":" ExtPaddingDims",
        "weights_filler": "WeightsFiller",
        "bias_filler": "BiasFiller",
        "weights_solver": "WeightsSolver",
        "bias_solver": "BiasSolver",
    }
    _parameters.update(_cell_frame_parameters)

    _convention_converter= ConventionConverter(_parameters)

    def __init__(self,
                 nb_inputs,
                 nb_outputs,
                 kernel_dims,
                 nb_input_cells=1,
                 **config_parameters):
        """
        :param nb_inputs: Number of inputs of the cells.
        :type nb_inputs: int
        :param nb_outputs: Number of output channels
        :type nb_outputs: int
        :param kernel_dims: Kernel dimension.
        :type kernel_dims: list
        :param nb_input_cells: Number of cell who are an input of this cell, default=1
        :type nb_input_cells: int, optional
        :param stride_dims: Dimension of the stride of the kernel, default=[1, 1]
        :type stride_dims: list, optional
        :param padding_dims: Dimensions of the padding, default=[0, 0]
        :type padding_dims: list, optional
        :param dilation_dims: Dimensions of the dilation of the kernels, default=[1, 1]
        :type dilation_dims: list, optional
        :param filler: Set the weights and bias filler, this parameter override parameters ``weights_filler`` and ``bias_filler``, default= :py:class:`n2d2.filler.NormalFiller`
        :type filler: :py:class:`n2d2.filler.Filler`, optional
        :param weights_filler: Weights initial values filler, default= :py:class:`n2d2.filler.NormalFiller`
        :type weights_filler: :py:class:`n2d2.filler.Filler`, optional
        :param bias_filler: Biases initial values filler, default= :py:class:`n2d2.filler.NormalFiller`
        :type bias_filler: :py:class:`n2d2.filler.Filler`, optional
        :param solver: Set the weights and bias solver, this parameter override parameters ``weights_solver`` and ``bias_solver``, default= :py:class:`n2d2.solver.SGD`
        :type solver: :py:class:`n2d2.solver.Solver`, optional
        :param weights_solver: Solver for weights, default= :py:class:`n2d2.solver.SGD`
        :type weights_solver: :py:class:`n2d2.solver.Solver`, optional
        :param bias_solver: Solver for biases, default= :py:class:`n2d2.solver.SGD`
        :type bias_solver: :py:class:`n2d2.solver.Solver`, optional
        :param no_bias: If ``True``, don’t use bias, default=False
        :type no_bias: bool, optional
        :param back_propagate: If ``True``, enable backpropagation, default=True
        :type back_propagate: bool, optional
        :param weights_export_flip: If ``True``, import/export flipped kernels, default=False
        :type weights_export_flip: bool, optional
        :param mapping: Mapping
        :type mapping: :py:class:`Tensor`, optional
        """
        # Need to set no_bias before filler parameter !
        if "no_bias" in config_parameters:
            self.no_bias = config_parameters["no_bias"]
        else:
            self.no_bias = False
        if not isinstance(nb_inputs, int):
            raise error_handler.WrongInputType("nb_inputs", str(type(nb_inputs)), ["int"])
        if not isinstance(nb_outputs, int):
            raise error_handler.WrongInputType("nb_outputs", str(type(nb_outputs)), ["int"])
        if not isinstance(kernel_dims, list):
            raise error_handler.WrongInputType("kernel_dims", str(type(kernel_dims)), ["list"])

        NeuralNetworkCell.__init__(self, **config_parameters)
        ModelDatatyped.__init__(self, **config_parameters)

        # activation is set in NeuralNetworkCell
        if self.activation is None: # pylint : disable=access-member-before-definition
            self.activation = Linear() # pylint : disable=access-member-before-definition

        self._constructor_arguments.update({
            'nb_inputs': nb_inputs,
            'nb_outputs': nb_outputs,
            'kernel_dims': kernel_dims,
        })

        self._parse_optional_arguments(['stride_dims', 'padding_dims', 'dilation_dims'])

        self._set_N2D2_object(self._N2D2_constructors[self._model_key](N2D2.DeepNet(global_variables.default_net),
                                                                     self.get_name(),
                                                                     self._constructor_arguments['kernel_dims'],
                                                                     self._constructor_arguments['nb_outputs'],
                                                                     **self.n2d2_function_argument_parser(self._optional_constructor_arguments)))

        Trainable.__init__(self)

        # Set and initialize here all complex cells members
        for key, value in self._config_parameters.items():
            self.__setattr__(key, value)

        self._N2D2_object.initializeParameters(nb_inputs, nb_input_cells)
        self.load_N2D2_parameters(self.N2D2())

    def __setattr__(self, key: str, value) -> None:
        if key == 'weights_solver':
            if isinstance(value, Solver):
                if self._N2D2_object:
                    self._N2D2_object.resetWeightsSolver(value.N2D2())
                self._config_parameters["weights_solver"] = value
            else:
                raise error_handler.WrongInputType("weights_solver", str(type(value)), [str(Solver)])
        elif key == 'bias_solver':
            if isinstance(value, Solver):
                if self._N2D2_object:
                    self._N2D2_object.setBiasSolver(value.N2D2())
                self._config_parameters["bias_solver"] = value
            else:
                raise error_handler.WrongInputType("bias_solver", str(type(value)), [str(Solver)])
        elif key == 'weights_filler':
            if isinstance(value, Filler):
                if self._N2D2_object:
                    self._N2D2_object.setWeightsFiller(value.N2D2())
                self._config_parameters["weights_filler"] = value
            else:
                raise error_handler.WrongInputType("weights_filler", str(type(value)), [str(Filler)])
        elif key == 'bias_filler':
            if isinstance(value, Filler):
                if self._N2D2_object:
                    self._N2D2_object.setBiasFiller(value.N2D2())
                self._config_parameters["bias_filler"] = value
            else:
                raise error_handler.WrongInputType("bias_filler", str(type(value)), [str(Filler)])
        elif key == 'mapping':
            if isinstance(value, Tensor): # TODO : add mapping in _config_parameters
                self._N2D2_object.setMapping(value.N2D2())
            else:
                raise error_handler.WrongInputType('mapping', type(value), [str(type(Tensor))])
        elif key == 'filler':
            self.set_filler(value)
        elif key == 'solver':
            self.set_solver(value)
        else:
            super().__setattr__(key, value)

    def _load_N2D2_constructor_parameters(self, N2D2_object):
        self._constructor_arguments.update({
            'nb_inputs':  N2D2_object.getNbChannels(),
            'nb_outputs':  N2D2_object.getNbOutputs(),
            'kernel_dims': [N2D2_object.getKernelWidth(), N2D2_object.getKernelHeight()]
        })

    def _load_N2D2_optional_parameters(self, N2D2_object):
        self._optional_constructor_arguments.update({
            'stride_dims':  [N2D2_object.getStrideX(), N2D2_object.getStrideY()],
            'padding_dims': [N2D2_object.getPaddingX(), N2D2_object.getPaddingY()],
            'dilation_dims': [N2D2_object.getDilationX(), N2D2_object.getDilationY()],
        })
    @classmethod
    def _get_N2D2_complex_parameters(cls, N2D2_object):
        parameter =  super()._get_N2D2_complex_parameters(N2D2_object)
        parameter['weights_solver'] = \
            from_N2D2_object(N2D2_object.getWeightsSolver())
        parameter['bias_solver'] = \
            from_N2D2_object(N2D2_object.getBiasSolver())
        parameter['weights_filler'] = \
            from_N2D2_object(N2D2_object.getWeightsFiller())
        parameter['bias_filler'] = \
            from_N2D2_object(N2D2_object.getBiasFiller())

        return parameter

    def __call__(self, inputs):
        super().__call__(inputs)

        self._add_to_graph(inputs)

        self._N2D2_object.propagate(self._inference)

        return self.get_outputs()

    def set_solver(self, solver):
        """"Set the weights and bias solver with the same solver.

        :param solver: Solver object
        :type solver: :py:class:`n2d2.solver.Solver`
        """
        if not isinstance(solver, Solver):
            raise error_handler.WrongInputType("solver", str(type(solver)), ["n2d2.solver.Solver"])
        self.bias_solver = solver.copy()
        self.weights_solver= solver.copy()

    def set_filler(self, filler, refill=False):
        """Set a filler for the weights and bias.

        :param filler: Filler object
        :type filler: :py:class:`n2d2.filler.Filler`
        """
        if not isinstance(filler, Filler):
            raise error_handler.WrongInputType("filler", str(type(filler)), ["n2d2.filler.Filler"])
        self.weights_filler = filler
        self.bias_filler = filler
        if refill:
            self.refill_bias()
            self.refill_weights()
    def set_weights_filler(self, solver, refill=False):
        self._config_parameters['weights_filler'] = solver
        self._N2D2_object.setWeightsFiller(self._config_parameters['weights_filler'].N2D2())
        if refill:
            self.refill_weights()
    def set_bias_filler(self, solver, refill=False):
        self._config_parameters['bias_filler'] = solver
        self._N2D2_object.setBiasFiller(self._config_parameters['bias_filler'].N2D2())
        if refill:
            self.refill_weights()

    @deprecated(reason="You should use weights_filler as an argument")
    def get_weights_filler(self):
        return self._config_parameters['weights_filler']
    @deprecated(reason="You should use bias_filler as an argument")
    def get_bias_filler(self):
        return self._config_parameters['bias_filler']
    @deprecated(reason="You should use weights_solver as an argument")
    def get_weights_solver(self):
        return self._config_parameters['weights_solver']
    @deprecated(reason="You should use bias_solver as an argument")
    def get_bias_solver(self):
        return self._config_parameters['bias_solver']
    @deprecated(reason="You should use weights_solver as an argument")
    def set_weights_solver(self, solver):
        self._config_parameters['weights_solver'] = solver
        self._N2D2_object.resetWeightsSolver(self._config_parameters['weights_solver'].N2D2())
    @deprecated(reason="You should use bias_solver as an argument")
    def set_bias_solver(self, solver):
        self._config_parameters['bias_solver'] = solver
        self._N2D2_object.setBiasSolver(self._config_parameters['bias_solver'].N2D2())

    def refill_bias(self):
        """Re-fill the bias using the associated bias filler
        """
        self._N2D2_object.resetBias()
    def refill_weights(self):
        """Re-fill the weights using the associated weights filler
        """
        self._N2D2_object.resetWeights()

    def set_weight(self, output_index, channel_index, value):
        """
        :param output_index:
        :type output_index:
        :param channel_index:
        :type channel_index:
        :param value:
        :type value: :py:class:`Tensor`
        """
        if channel_index >= self.N2D2().getNbChannels():
            raise ValueError("Channel index : " + str(channel_index) + " must be < " + str(self.N2D2().getNbChannels()) +")")
        if output_index >= self.N2D2().getNbOutputs():
            raise ValueError("Output index : " + str(output_index) + " must be < " + str(self.N2D2().getNbOutputs()) +")")
        self.N2D2().setWeight(output_index, channel_index, value.N2D2())

    def get_weight(self, output_index, channel_index):
        """
        :param output_index:
        :type output_index: int
        :param channel_index:
        :type channel_index: int
        """
        if channel_index >= self.N2D2().getNbChannels():
            raise ValueError("Channel index : " + str(channel_index) + " must be < " + str(self.N2D2().getNbChannels()) +")")
        if output_index >= self.N2D2().getNbOutputs():
            raise ValueError("Output index : " + str(output_index) + " must be < " + str(self.N2D2().getNbOutputs()) +")")
        tensor = N2D2.Tensor_float([])
        self.N2D2().getWeight(output_index, channel_index, tensor)
        return Tensor.from_N2D2(tensor)

    def get_weights(self):
        """
        :return: list of weights
        :rtype: list
        """
        weights = []
        for output_index in range(self.N2D2().getNbOutputs()):
            chan = []
            for channel_index in range(self.N2D2().getNbChannels()):
                tensor = N2D2.Tensor_float([])
                self.N2D2().getWeight(output_index, channel_index, tensor)
                chan.append(Tensor.from_N2D2(tensor))
            weights.append(chan)
        return weights

    def set_bias(self, output_index, value):
        """
        :param output_index:
        :type output_index: int
        :param value:
        :type value: :py:class:`Tensor`
        """
        if output_index >= self.N2D2().getNbOutputs():
            raise ValueError("Output index : " + str(output_index) + " must be < " + str(self.N2D2().getNbOutputs()) +")")
        self.N2D2().setBias(output_index, value.N2D2())

    def has_bias(self):
        return not self.no_bias

    def get_bias(self, output_index):
        """
        :param output_index:
        :type output_index: int
        """
        if output_index >= self.N2D2().getNbOutputs():
            raise ValueError("Output index : " + str(output_index) + " must be < " + str(self.N2D2().getNbOutputs()) +")")
        tensor = N2D2.Tensor_float([])
        self.N2D2().getBias(output_index, tensor)
        return Tensor.from_N2D2(tensor)

    def get_biases(self):
        """
        :return: list of biases
        :rtype: list
        """
        biases = []
        for output_index in range(self.N2D2().getNbOutputs()):
            tensor = N2D2.Tensor_float([])
            self.N2D2().getBias(output_index, tensor)
            biases.append(Tensor.from_N2D2(tensor))
        return biases

    def has_quantizer(self):
        """
        :return: True if the cell use a quantizer
        :rtype: bool
        """
        return False
