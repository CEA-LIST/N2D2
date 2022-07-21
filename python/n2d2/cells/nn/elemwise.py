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

from n2d2 import global_variables
from n2d2 import ConventionConverter, Interface, Tensor
from n2d2.cells.nn.abstract_cell import (NeuralNetworkCell,
                                         _cell_frame_parameters)
from n2d2.typed import Modeltyped
from n2d2.utils import inherit_init_docstring
from n2d2.error_handler import WrongInputType, WrongValue


@inherit_init_docstring()
class ElemWise(NeuralNetworkCell, Modeltyped):
    """Element-wise operation layer.
    """

    _N2D2_constructors = {
        'Frame': N2D2.ElemWiseCell_Frame,
    }

    if global_variables.cuda_available:
        _N2D2_constructors.update({
            'Frame_CUDA': N2D2.ElemWiseCell_Frame_CUDA,
        })
    _parameters = {
        "operation": "operation",
        "mode": "mode",
        "weights": "weights",
        "shifts": "shifts"
    }
    _parameters.update(_cell_frame_parameters)

    _convention_converter = ConventionConverter(_parameters)

    _parameter_loaded = True # boolean to indicate if parameters have been loaded.

    def __init__(self, **config_parameters):
        """
        :param operation: Type of operation (``Sum``, ``AbsSum``, ``EuclideanSum``, ``Prod``, or ``Max``), default="Sum"
        :type operation: str, optional
        :param mode: (``PerLayer``, ``PerInput``, ``PerChannel``), default="PerLayer"
        :type mode: str, optional
        :param weights: Weights for the ``Sum``, ``AbsSum``, and ``EuclideanSum`` operation, in the same order as the inputs, default=[1.0]
        :type weights: list, optional
        :param shifts: Shifts for the ``Sum`` and ``EuclideanSum`` operation, in the same order as the inputs, default=[0.0]
        :type shifts: list, optional
        """

        NeuralNetworkCell.__init__(self, **config_parameters)
        Modeltyped.__init__(self, **config_parameters)

        self._parse_optional_arguments(['operation', 'mode', 'weights', 'shifts'])

        if "operation" in self._optional_constructor_arguments:
            operation = self._optional_constructor_arguments["operation"]
            if not isinstance(operation, str):
                raise WrongInputType("operation", str(type(operation)), ["str"])
            if operation not in N2D2.ElemWiseCell.Operation.__members__.keys():
                raise WrongValue("operation", operation, N2D2.ElemWiseCell.Operation.__members__.keys())
            self._optional_constructor_arguments['operation'] = \
                N2D2.ElemWiseCell.Operation.__members__[self._optional_constructor_arguments['operation']]
        if "mode" in self._optional_constructor_arguments:
            mode = self._optional_constructor_arguments["mode"]
            if not isinstance(mode, str):
                raise WrongInputType("mode", str(type(mode)), ["str"])
            if mode not in N2D2.ElemWiseCell.CoeffMode.__members__.keys():
                raise WrongValue("mode", mode, N2D2.ElemWiseCell.CoeffMode.__members__.keys())
            self._optional_constructor_arguments['mode'] = \
                N2D2.ElemWiseCell.CoeffMode.__members__[self._optional_constructor_arguments['mode']]
        if "weights" in self._optional_constructor_arguments:
            if not isinstance(self._optional_constructor_arguments["weights"], list):
                raise WrongInputType("weights", str(type(self._optional_constructor_arguments["weights"])), ["float"])
        if "shifts" in self._optional_constructor_arguments:
            if not isinstance(self._optional_constructor_arguments["shifts"], list):
                raise WrongInputType("shifts", str(type(self._optional_constructor_arguments["shifts"])), ["float"])

    def _load_N2D2_optional_parameters(self, N2D2_object):
        self._optional_constructor_arguments['operation'] = N2D2_object.getOperation()
        self._optional_constructor_arguments['mode'] = N2D2_object.getCoeffMode()
        self._optional_constructor_arguments['weights'] = N2D2_object.getWeights()
        self._optional_constructor_arguments['shifts'] = N2D2_object.getShifts()

    def __call__(self, inputs):
        super().__call__(inputs)

        if self._N2D2_object is None:

            mapping_row = 0

            if isinstance(inputs, Interface):
                for tensor in inputs.get_tensors():
                    if tensor.nb_dims() != 4:
                        raise ValueError("Input Tensor should have 4 dimensions, " + str(inputs.nb_dims()),
                                         " were given.")
                    mapping_row = tensor.dimZ()
            elif isinstance(inputs, Tensor):
                if inputs.nb_dims() != 4:
                    raise ValueError("Input Tensor should have 4 dimensions, " + str(inputs.nb_dims()), " were given.")
                mapping_row = inputs.dimZ()

            else:
                raise WrongInputType("inputs", inputs, [str(type(list)), str(type(Tensor))])


            self._set_N2D2_object(self._N2D2_constructors[self._model_key](self._deepnet.N2D2(),
                                                                     self.get_name(),
                                                                     mapping_row,
                                                                     **self.n2d2_function_argument_parser(
                                                                         self._optional_constructor_arguments)))
            # Set and initialize here all complex cells members
            for key, value in self._config_parameters.items():
                self.__setattr__(key, value)
            self._parameter_loaded = False

        self._add_to_graph(inputs)

        self._N2D2_object.propagate(self._inference)

        if not self._parameter_loaded:
            # ElemWise initialize weights and shift after propagation
            self.load_N2D2_parameters(self.N2D2())

        return self.get_outputs()
