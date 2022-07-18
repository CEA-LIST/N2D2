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

from n2d2 import ConventionConverter, global_variables
from n2d2.cells.nn.abstract_cell import (NeuralNetworkCell,
                                         _cell_frame_parameters)
from n2d2.typed import ModelDatatyped
from n2d2.utils import inherit_init_docstring


@inherit_init_docstring()
class Activation(NeuralNetworkCell, ModelDatatyped):
    """Activation layer which can apply any activation to a stimuli.
    """
    _N2D2_constructors = {
        'Frame<float>': N2D2.ActivationCell_Frame_float,
    }
    if global_variables.cuda_available:
        _N2D2_constructors.update({
            'Frame_CUDA<float>': N2D2.ActivationCell_Frame_CUDA_float,
        })

    _parameters = {
    }
    _parameters.update(_cell_frame_parameters)

    _convention_converter= ConventionConverter(_parameters)

    def __init__(self, activation, **config_parameters):
        """
        """
        NeuralNetworkCell.__init__(self, **config_parameters)
        ModelDatatyped.__init__(self, **config_parameters)

        # No optional parameter
        self._config_parameters["activation"] = activation
        self._parse_optional_arguments([])

    def __call__(self, inputs):
        super().__call__(inputs)

        if self._N2D2_object is None:
            nb_outputs = inputs.dimZ()

            self._set_N2D2_object(self._N2D2_constructors[self._model_key](self._deepnet.N2D2(),
                                                                         self.get_name(),
                                                                         nb_outputs,
                                                                         **self._optional_constructor_arguments))
            # Set and initialize here all complex cells members
            for key, value in self._config_parameters.items():
                self.__setattr__(key, value)
            self.load_N2D2_parameters(self.N2D2())

        self._add_to_graph(inputs)

        self._N2D2_object.propagate(self._inference)

        return self.get_outputs()
