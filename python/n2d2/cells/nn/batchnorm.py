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

import n2d2.filler
import n2d2.global_variables as gb
import n2d2.solver
from n2d2 import ConventionConverter, Tensor
from n2d2.cells.cell import Trainable
from n2d2.cells.nn.abstract_cell import (NeuralNetworkCell,
                                         _cell_frame_parameters)
from n2d2.error_handler import deprecated
from n2d2.typed import ModelDatatyped
from n2d2.utils import inherit_init_docstring


@inherit_init_docstring()
class BatchNorm2d(NeuralNetworkCell, ModelDatatyped, Trainable):
    """Batch Normalization layer :cite:`Ioffe2015`.
    """
    _N2D2_constructors = {
        'Frame<float>': N2D2.BatchNormCell_Frame_float,
    }
    if gb.cuda_compiled:
        _N2D2_constructors.update({
            'Frame_CUDA<float>': N2D2.BatchNormCell_Frame_CUDA_float,
        })
    _parameters = {
        "nb_inputs": "NbInputs",
        "scale_solver": "ScaleSolver",
        "bias_solver": "BiasSolver",
        "moving_average_momentum": "MovingAverageMomentum",
        "epsilon": "Epsilon",
        "back_propagate": "BackPropagate"
    }
    _parameters.update(_cell_frame_parameters)

    _convention_converter = ConventionConverter(_parameters)

    def __init__(self, nb_inputs, nb_input_cells=1, **config_parameters):
        """
        :param nb_inputs: Number of intput neurons
        :type nb_inputs: int
        :param nb_input_cells: Number of cell who are an input of this cell, default=1
        :type nb_input_cells: int, optional
        :param solver: Set the scale and bias solver, this parameter override parameters ``scale_solver`` and bias_solver``, default= :py:class:`n2d2.solver.SGD`
        :type solver: :py:class:`n2d2.solver.Solver`, optional
        :param scale_solver: Scale solver parameters, default= :py:class:`n2d2.solver.SGD`
        :type scale_solver: :py:class:`n2d2.solver.Solver`, optional
        :param bias_solver: Bias  solver parameters, default= :py:class:`n2d2.solver.SGD`
        :type bias_solver: :py:class:`n2d2.solver.Solver`, optional
        :param epsilon: Epsilon value used in the batch normalization formula. If ``0.0``, automatically choose the minimum possible value, default=0.0
        :type epsilon: float, optional
        :param moving_average_momentum: Moving average rate: used for the moving average of batch-wise means and standard deviations during training.The closer to ``1.0``, \
        the more it will depend on the last batch.
        :type moving_average_momentum: float, optional
        """
        if not isinstance(nb_inputs, int):
            raise n2d2.error_handler.WrongInputType("nb_inputs", str(type(nb_inputs)), ["int"])

        NeuralNetworkCell.__init__(self, **config_parameters)
        ModelDatatyped.__init__(self, **config_parameters)
        # No optional parameter
        self._parse_optional_arguments([])
        self._set_N2D2_object(self._N2D2_constructors[self._model_key](N2D2.DeepNet(n2d2.global_variables.default_net),
                                                self.get_name(),
                                                nb_inputs,
                                                **self.n2d2_function_argument_parser(self._optional_constructor_arguments)))

        Trainable.__init__(self)

        # Set and initialize here all complex cells members
        for key, value in self._config_parameters.items():
            self.__setattr__(key, value)

        self._N2D2_object.initializeParameters(nb_inputs, nb_input_cells)
        self.load_N2D2_parameters(self.N2D2())

    def __setattr__(self, key: str, value) -> None:
        if key == 'scale_solver':
            if not isinstance(value, n2d2.solver.Solver):
                raise n2d2.error_handler.WrongInputType("scale_solver", str(type(value)), [str(n2d2.solver.Solver)])
            if self._N2D2_object:
                self._N2D2_object.setScaleSolver(value.N2D2())
            self._config_parameters["scale_solver"] = value
        elif key == 'bias_solver':
            if not isinstance(value, n2d2.solver.Solver):
                raise n2d2.error_handler.WrongInputType("bias_solver", str(type(value)), [str(n2d2.solver.Solver)])
            if self._N2D2_object:
                self._N2D2_object.setBiasSolver(value.N2D2())
            self._config_parameters["bias_solver"] = value
        elif key == 'solver':
            self.set_solver(value)
        else:
            super().__setattr__(key, value)

    def _load_N2D2_constructor_parameters(self, N2D2_object):
        self._constructor_arguments.update({
            'nb_inputs':  N2D2_object.getNbChannels(),
        })

    @classmethod
    def _get_N2D2_complex_parameters(cls, N2D2_object):
        parameter =  super()._get_N2D2_complex_parameters(N2D2_object)
        parameter['scale_solver'] = \
            n2d2.converter.from_N2D2_object(N2D2_object.getScaleSolver())
        parameter['bias_solver'] = \
            n2d2.converter.from_N2D2_object(N2D2_object.getBiasSolver())
        return parameter

    def __call__(self, inputs):
        if self._constructor_arguments["nb_inputs"] != inputs.dimZ():
            raise ValueError(self.get_name() + " : expected an input with " + str(self._constructor_arguments["nb_inputs"]) + " channels got a tensor with " + str(inputs.dimZ()) + " instead.")
        super().__call__(inputs)

        self._add_to_graph(inputs)

        self._N2D2_object.propagate(self._inference)

        return self.get_outputs()

    def has_bias(self):
        return True

    @deprecated(reason="You should use scale_solver as an attribute.")
    def set_scale_solver(self, solver):
        self._config_parameters['scale_solver'] = solver
        self._N2D2_object.setScaleSolver(self._config_parameters['scale_solver'].N2D2())

    def get_biases(self) -> Tensor:
        return Tensor.from_N2D2(self.N2D2().getBiases())

    def get_scales(self) -> Tensor:
        return Tensor.from_N2D2(self.N2D2().getScales())

    def get_means(self) -> Tensor:
        return Tensor.from_N2D2(self.N2D2().getMeans())

    def get_variances(self) -> Tensor:
        return Tensor.from_N2D2(self.N2D2().getVariances())

    def set_solver_parameter(self, key, value):
        """Set the parameter ``key`` with the value ``value`` for the attribute ``scale`` and ``bias`` solver.

        :param key: Parameter name
        :type key: str
        :param value: The value of the parameter
        :type value: Any
        """
        self._config_parameters['scale_solver'].set_parameter(key, value)
        self._config_parameters['bias_solver'].set_parameter(key, value)

    def set_filler(self, filler, refill=False):
        raise ValueError("Batchnorm doesn't support Filler")

    def set_solver(self, solver):
        """"Set the ``scale`` and ``bias`` solver with the same solver.

        :param solver: Solver object
        :type solver: :py:class:`n2d2.solver.Solver`
        """
        if not isinstance(solver, n2d2.solver.Solver):
            raise n2d2.error_handler.WrongInputType("solver", str(type(solver)), ["n2d2.solver.Solver"])
        self.bias_solver = solver.copy()
        self.scale_solver = solver.copy()

    def has_quantizer(self):
        # BatchNorm objects don't have a quantizer !
        return False
