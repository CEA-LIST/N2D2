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

import n2d2.activation
import n2d2.global_variables as gb
from n2d2 import ConventionConverter
from n2d2.cells.nn.abstract_cell import (NeuralNetworkCell,
                                         _cell_frame_parameters)
from n2d2.typed import Modeltyped
from n2d2.utils import inherit_init_docstring


@inherit_init_docstring()
class Padding(NeuralNetworkCell, Modeltyped):
    """Padding layer allows to insert asymmetric padding for each layer axes.
    """
    _N2D2_constructors = {
        'Frame': N2D2.PaddingCell_Frame,
    }
    if gb.cuda_available:
        _N2D2_constructors.update({
            'Frame_CUDA': N2D2.PaddingCell_Frame_CUDA,
        })
    _parameters = {
        "top_pad":"top_pad",
        "bot_pad":"bot_pad",
        "left_pad":"left_pad",
        "right_pad":"right_pad",
    }
    _parameters.update(_cell_frame_parameters)

    _convention_converter= ConventionConverter(_parameters)

    def __init__(self,
                 top_pad,
                 bot_pad,
                 left_pad,
                 right_pad,
                 **config_parameters):
        """
        :param top_pad: Size of the top padding (positive or negative)
        :type top_pad: int
        :param bot_pad: Size of the bottom padding (positive or negative)
        :type bot_pad: int
        :param left_pad: Size of the left padding (positive or negative)
        :type left_pad: int
        :param right_pad: Size of the right padding (positive or negative)
        :type right_pad: int
        """
        if not isinstance(top_pad, int):
            raise n2d2.error_handler.WrongInputType("top_pad", str(type(top_pad)), ["int"])
        if not isinstance(bot_pad, int):
            raise n2d2.error_handler.WrongInputType("bot_pad", str(type(bot_pad)), ["int"])
        if not isinstance(left_pad, int):
            raise n2d2.error_handler.WrongInputType("left_pad", str(type(left_pad)), ["int"])
        if not isinstance(right_pad, int):
            raise n2d2.error_handler.WrongInputType("right_pad", str(type(right_pad)), ["int"])

        NeuralNetworkCell.__init__(self, **config_parameters)
        Modeltyped.__init__(self, **config_parameters)

        self._constructor_arguments.update({
                 'top_pad': top_pad,
                 'bot_pad': bot_pad,
                 'left_pad': left_pad,
                 'right_pad': right_pad
        })
        # No optional args
        self._parse_optional_arguments([])

    def _load_N2D2_constructor_parameters(self, N2D2_object):
        self._constructor_arguments['top_pad'] = N2D2_object.getTopPad()
        self._constructor_arguments['bot_pad'] = N2D2_object.getBotPad()
        self._constructor_arguments['left_pad'] = N2D2_object.getLeftPad()
        self._constructor_arguments['right_pad'] = N2D2_object.getRightPad()

    def __call__(self, inputs):
        super().__call__(inputs)

        if self._N2D2_object is None:
            nb_outputs = inputs.dims()[2]

            self._set_N2D2_object(self._N2D2_constructors[self._model_key](self._deepnet.N2D2(),
                                                                     self.get_name(),
                                                                     nb_outputs,
                                                                     self._constructor_arguments['top_pad'],
                                                                     self._constructor_arguments['bot_pad'],
                                                                     self._constructor_arguments['left_pad'],
                                                                     self._constructor_arguments['right_pad'],
                                                                     **self.n2d2_function_argument_parser(
                                                                         self._optional_constructor_arguments)))
            # Set and initialize here all complex cells members
            for key, value in self._config_parameters.items():
                self.__setattr__(key, value)

            self.load_N2D2_parameters(self.N2D2())

        self._add_to_graph(inputs)

        self._N2D2_object.propagate(self._inference)

        return self.get_outputs()
