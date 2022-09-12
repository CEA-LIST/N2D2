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
from n2d2.utils import inherit_init_docstring, check_types
from n2d2.scaling import ScalingObject
from n2d2.error_handler import WrongInputType, WrongValue

@inherit_init_docstring()
class Scaling(NeuralNetworkCell, ModelDatatyped):
    """Scaling layer.
    """
    _N2D2_constructors = {
        'Frame<float>': N2D2.ScalingCell_Frame_float,
    }
    if global_variables.cuda_available:
        _N2D2_constructors.update({
            'Frame_CUDA<float>': N2D2.ScalingCell_Frame_CUDA_float,
        })
    _parameters = {}
    _parameters.update(_cell_frame_parameters)

    _convention_converter = ConventionConverter(_parameters)

    def __init__(self, scaling, **config_parameters):
        """
        :param scaling: Scaling object
        :type scaling: :py:class:`n2d2.scaling.Scaling`
        """
        if not isinstance(scaling, ScalingObject):
            raise WrongInputType("outputs_width", type(scaling), ["list"])

        NeuralNetworkCell.__init__(self, **config_parameters)
        ModelDatatyped.__init__(self, **config_parameters)
        self._constructor_arguments.update({
            'scaling': scaling,
        })

        # No optional parameter
        self._parse_optional_arguments([])

    def _load_N2D2_constructor_parameters(self, N2D2_object):
        self._constructor_arguments['scaling'] = N2D2_object.getScaling() # TODO: add Scaling to converter

    def __call__(self, inputs):
        super().__call__(inputs)

        if self._N2D2_object is None:
            nb_outputs = inputs.dims()[2]

            self._set_N2D2_object(self._N2D2_constructors[self._model_key](self._deepnet.N2D2(),
                                                                           self.get_name(),
                                                                           nb_outputs,
                                                                           self._constructor_arguments['scaling'],
                                                                           **self.n2d2_function_argument_parser(
                                                                               self._optional_constructor_arguments)))

            # Set and initialize here all complex cells members
            for key, value in self._config_parameters.items():
                self.__setattr__(key, value)
            self.load_N2D2_parameters(self.N2D2())

        self._add_to_graph(inputs)

        self._N2D2_object.propagate(self._inference)

        return self.get_outputs()

    @staticmethod
    @check_types
    def is_exportable_to(export_name:str) -> bool:
        """
        :param export_name: Name of the export 
        :type export_name: str
        :return: ``True`` if the cell is exportable to the ``export_name`` export. 
        :rtype: bool
        """
        from n2d2.export import available_export
        if export_name not in available_export:
            raise WrongValue("export_name", export_name, available_export)
        return N2D2.ScalingCellExport.isExportableTo(export_name)