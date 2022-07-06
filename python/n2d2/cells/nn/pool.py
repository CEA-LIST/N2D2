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
from n2d2 import ConventionConverter, Interface, Tensor
from n2d2.cells.nn.abstract_cell import (NeuralNetworkCell,
                                         _cell_frame_parameters)
from n2d2.typed import ModelDatatyped
from n2d2.utils import inherit_init_docstring
from n2d2.mapping import Mapping
from n2d2.error_handler import WrongInputType, WrongValue


@inherit_init_docstring()
class Pool(NeuralNetworkCell, ModelDatatyped):
    '''
    Pooling layer.
    '''

    mappable = True

    _N2D2_constructors = {
        'Frame<float>': N2D2.PoolCell_Frame_float,
    }
    if gb.cuda_available:
        _N2D2_constructors.update({
            'Frame_CUDA<float>': N2D2.PoolCell_Frame_CUDA_float,
        })

    _parameters = {
        "pool_dims": "poolDims",
        "stride_dims": "strideDims",
        "padding_dims": "paddingDims",
        "pooling": "pooling",
        "ext_padding_dims": "ExtPaddingDims",
    }
    _parameters.update(_cell_frame_parameters)

    _convention_converter = ConventionConverter(_parameters)

    def __init__(self,
                 pool_dims,
                 **config_parameters):
        """
        :param pool_dims: Pooling area dimensions
        :type pool_dims: list
        :param pooling: Type of pooling (``Max`` or ``Average``), default="Max"
        :type pooling: str, optional
        :param stride_dims: Dimension of the stride of the kernel, default= [1, 1]
        :type stride_dims: list, optional
        :param padding_dims: Dimensions of the padding, default= [0, 0]
        :type padding_dims: list, optional
        :param mapping: Mapping
        :type mapping: :py:class:`Tensor`, optional
        """
        if not isinstance(pool_dims, list):
            raise WrongInputType("pool_dims", str(type(pool_dims)), ["list"])

        NeuralNetworkCell.__init__(self, **config_parameters)
        ModelDatatyped.__init__(self, **config_parameters)

        self._constructor_arguments.update({
            'pool_dims': pool_dims,
        })

        self._parse_optional_arguments(['stride_dims', 'padding_dims', 'pooling'])
        if "pooling" in self._optional_constructor_arguments:
            pooling = self._optional_constructor_arguments["pooling"]
            if not isinstance(pooling, str):
                raise WrongInputType("pooling", str(type(pooling)), ["str"])
            if pooling not in N2D2.PoolCell.Pooling.__members__.keys():
                raise WrongValue("pooling", pooling, N2D2.PoolCell.Pooling.__members__.keys())
            self._optional_constructor_arguments['pooling'] = \
                N2D2.PoolCell.Pooling.__members__[self._optional_constructor_arguments['pooling']]

    def _load_N2D2_constructor_parameters(self, N2D2_object):
        self._constructor_arguments['pool_dims'] = [N2D2_object.getPoolWidth(),
                                                        N2D2_object.getPoolHeight()]

    def _load_N2D2_optional_parameters(self, N2D2_object):
        self._optional_constructor_arguments.update({
            'stride_dims': [N2D2_object.getStrideX(), N2D2_object.getStrideY()],
            'padding_dims': [N2D2_object.getPaddingX(), N2D2_object.getPaddingY()],
            'pooling': N2D2_object.getPooling(),
        })

    def __call__(self, inputs):
        super().__call__(inputs)

        if self._N2D2_object is None:
            mapping_row = 0
            if isinstance(inputs, Interface): # Here we try to support multi input
                for tensor in inputs.get_tensors():
                    if tensor.nb_dims() != 4:
                        raise ValueError("Input Tensor should have 4 dimensions, " + str(inputs.nb_dims()), " were given.")
                    mapping_row += tensor.dimZ()
            elif isinstance(inputs, Tensor):
                if inputs.nb_dims() != 4:
                    raise ValueError("Input Tensor should have 4 dimensions, " + str(inputs.nb_dims()), " were given.")
                mapping_row += inputs.dimZ()
            else:
                raise WrongInputType("inputs", inputs, [str(type(list)), str(type(Tensor))])

            self._set_N2D2_object(self._N2D2_constructors[self._model_key](self._deepnet.N2D2(),
                                                                         self.get_name(),
                                                                         self._constructor_arguments['pool_dims'],
                                                                         mapping_row,
                                                                         **self.n2d2_function_argument_parser(self._optional_constructor_arguments)))

            # Set and initialize here all complex cells members
            for key, value in self._config_parameters.items():
                self.__setattr__(key, value)

            self.load_N2D2_parameters(self.N2D2())

        self._add_to_graph(inputs)

        self._N2D2_object.propagate(self._inference)

        return self.get_outputs()

    def __setattr__(self, key: str, value) -> None:
        if key == 'mapping':
            if not isinstance(value, Tensor):
                raise WrongInputType('mapping', type(value), [str(type(Tensor))])
            if value.dimX() != value.dimY():
                raise ValueError("Pool Cell supports only unit maps")
            self._N2D2_object.setMapping(value.N2D2())
        else:
            super().__setattr__(key, value)

@inherit_init_docstring()
class Pool2d(Pool):
    """'Standard' pooling where all feature maps are pooled independently.
    """

    _N2D2_constructors = {
        'Frame<float>': N2D2.PoolCell_Frame_float,
    }
    if gb.cuda_available:
        _N2D2_constructors.update({
            'Frame_CUDA<float>': N2D2.PoolCell_Frame_CUDA_float,
        })
    _parameters = {
        "pool_dims": "poolDims",
        "stride_dims": "strideDims",
        "padding_dims": "paddingDims",
        "pooling": "pooling",
        "ext_padding_dims": "ExtPaddingDims",
    }
    _parameters.update(_cell_frame_parameters)

    _convention_converter= ConventionConverter(_parameters)

    def __init__(self,
                 pool_dims,
                 **config_parameters):
        """
        :param pool_dims: Pooling area dimensions with the format [Height, Width]
        :type pool_dims: list
        :param pooling: Type of pooling (``Max`` or ``Average``), default="Max"
        :type pooling: str, optional
        :param stride_dims: Dimension of the stride of the kernel, default= [1, 1]
        :type stride_dims: list, optional
        :param padding_dims: Dimensions of the padding, default= [0, 0]
        :type padding_dims: list, optional
        """
        if 'mapping' in config_parameters:
            raise RuntimeError('Pool2d does not support custom mappings')
        Pool.__init__(self, pool_dims, **config_parameters)

    def __call__(self, inputs):
        NeuralNetworkCell.__call__(self, inputs)

        if self._N2D2_object is None:

            self._set_N2D2_object(self._N2D2_constructors[self._model_key](self._deepnet.N2D2(),
                                                                         self.get_name(),
                                                                         self._constructor_arguments['pool_dims'],
                                                                         inputs.dims()[2],
                                                                         **self.n2d2_function_argument_parser(self._optional_constructor_arguments)))

            # Set and initialize here all complex cells members
            for key, value in self._config_parameters.items():
                self.__setattr__(key, value)

            self._N2D2_object.setMapping(
                Mapping(nb_channels_per_group=1).create_mapping(inputs.dims()[2],
                                                                             inputs.dims()[2]).N2D2())
            self.load_N2D2_parameters(self.N2D2())

        self._add_to_graph(inputs)

        self._N2D2_object.propagate(self._inference)

        return self.get_outputs()

@inherit_init_docstring()
class GlobalPool2d(Pool2d):
    """
    Global 2d pooling on full spatial dimension of input. Before the first call, the pooling
    dimension will be an empty list, which will be filled with the inferred dimensions after
    the first call.
    """
    _N2D2_constructors = {
        'Frame<float>': N2D2.PoolCell_Frame_float,
    }
    if gb.cuda_available:
        _N2D2_constructors.update({
            'Frame_CUDA<float>': N2D2.PoolCell_Frame_CUDA_float,
        })

    _parameters = {
        "pooling": "pooling",
    }
    _parameters.update(_cell_frame_parameters)

    _convention_converter= ConventionConverter(_parameters)

    def __init__(self, **config_parameters):
        """
        """
        if 'pool_dims' in config_parameters:
            raise RuntimeError('GlobalPool2d does not support custom pool dims')
        Pool2d.__init__(self, [], **config_parameters)


    def __call__(self, inputs):
        NeuralNetworkCell.__call__(self, inputs)

        if self._N2D2_object is None:

            self._set_N2D2_object(self._N2D2_constructors[self._model_key](self._deepnet.N2D2(),
                                                                         self.get_name(),
                                                                         [inputs.dims()[0], inputs.dims()[1]],
                                                                         inputs.dims()[2],
                                                                         strideDims=[1, 1],
                                                                         **self.n2d2_function_argument_parser(self._optional_constructor_arguments)))

            # Set and initialize here all complex cells members
            for key, value in self._config_parameters.items():
                self.__setattr__(key, value)

            self._N2D2_object.setMapping(Mapping(nb_channels_per_group=1).create_mapping(inputs.dims()[2], inputs.dims()[2]).N2D2())
            self.load_N2D2_parameters(self.N2D2())

        self._add_to_graph(inputs)

        self._N2D2_object.propagate(self._inference)

        return self.get_outputs()
