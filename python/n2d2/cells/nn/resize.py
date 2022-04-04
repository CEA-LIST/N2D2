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

import n2d2.global_variables as gb
from n2d2 import ConventionConverter
from n2d2.cells.nn.abstract_cell import (NeuralNetworkCell,
                                         _cell_frame_parameters)
from n2d2.typed import Modeltyped
from n2d2.utils import inherit_init_docstring
from n2d2.error_handler import WrongInputType

@inherit_init_docstring()
class Resize(NeuralNetworkCell, Modeltyped):
    """Resize layer.
    """
    _N2D2_constructors = {
        'Frame': N2D2.ResizeCell_Frame,
    }
    if gb.cuda_compiled:
        _N2D2_constructors.update({
            'Frame_CUDA': N2D2.ResizeCell_Frame_CUDA,
        })
    _parameters = {
        "align_corners": "AlignCorners",
    }
    _parameters.update(_cell_frame_parameters)

    _convention_converter = ConventionConverter(_parameters)

    def __init__(self, outputs_width, outputs_height, resize_mode, **config_parameters):
        """
        :param outputs_width: outputs_width
        :type outputs_width: int
        :param outputs_height: outputs_height
        :type outputs_height: int
        :param resize_mode: Resize interpolation mode. Can be, ``Bilinear`` or ``BilinearTF`` (TensorFlow implementation)
        :type resize_mode: str
        :param align_corners: Corner alignement mode if ``BilinearTF`` is used as interpolation mode, default=True
        :type align_corners: boolean, optional
        """
        if not isinstance(outputs_width, int):
            raise WrongInputType("outputs_width", type(outputs_width), ["int"])
        if not isinstance(outputs_height, int):
            raise WrongInputType("outputs_height", type(outputs_height), ["int"])

        NeuralNetworkCell.__init__(self, **config_parameters)
        Modeltyped.__init__(self, **config_parameters)

        self._constructor_arguments.update({
            'outputs_width': outputs_width,
            'outputs_height': outputs_height,
            'resize_mode': N2D2.ResizeCell.ResizeMode.__members__[resize_mode],
        })

        # No optional parameter
        self._parse_optional_arguments([])

    def _load_N2D2_constructor_parameters(self, N2D2_object):
        self._constructor_arguments['outputs_width'] =  N2D2_object.getResizeOutputWidth()
        self._constructor_arguments['outputs_height'] = N2D2_object.getResizeOutputHeight()
        self._constructor_arguments['resize_mode'] = N2D2_object.getMode()


    def __call__(self, inputs):
        super().__call__(inputs)

        if self._N2D2_object is None:
            nb_outputs = inputs.dims()[2]

            self._set_N2D2_object(self._N2D2_constructors[self._model_key](self._deepnet.N2D2(),
                                                                           self.get_name(),
                                                                           self._constructor_arguments['outputs_width'],
                                                                           self._constructor_arguments['outputs_height'],
                                                                           nb_outputs,
                                                                           self._constructor_arguments['resize_mode'],
                                                                           **self.n2d2_function_argument_parser(
                                                                               self._optional_constructor_arguments)))

            # Set and initialize here all complex cells members
            for key, value in self._config_parameters.items():
                self.__setattr__(key, value)
            self.load_N2D2_parameters(self.N2D2())

        self._add_to_graph(inputs)

        self._N2D2_object.propagate(self._inference)

        return self.get_outputs()
