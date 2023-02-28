"""
    (C) Copyright 2023 CEA LIST. All Rights Reserved.
    Contributor(s): Cyril MOINEAU (cyril.moineau@cea.fr)

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
from n2d2.database import AbstractDatabase, _database_parameters
from n2d2.utils import inherit_init_docstring
from n2d2.n2d2_interface import ConventionConverter
from n2d2 import Tensor
import N2D2
from numpy import ndarray
from typing import List

@inherit_init_docstring()
class Numpy(AbstractDatabase):
    """
    TODO : write doc string
    """

    _type = "Numpy"
    _parameters = {}
    _parameters.update(_database_parameters)

    _convention_converter= ConventionConverter(_parameters)
    _N2D2_constructors = N2D2.Tensor_Database
    def __init__(self, **config_parameters):
        """
        TODO : write doc string
        """
        AbstractDatabase.__init__(self, **config_parameters)

        self._parse_optional_arguments([])
        self._N2D2_object = self._N2D2_constructors(**self.n2d2_function_argument_parser(self._optional_constructor_arguments))
        self._set_N2D2_parameters(self._config_parameters)
        self.load_N2D2_parameters(self.N2D2())
    
    def load(self, stimuli_list:List[ndarray], labels_list:List[int]):
        """Load numpy array as input and int as labels. 
        The loaded stimuli are stored in the ``Unpartitioned`` partition.

        :param stimuli_list: List of stimulus, they must respect the format [C, H, W].
        :type stimuli_list: List[ndarray]
        :param labels_list: List of label
        :type labels_list: List[int]
        TODO : write doc string
        """
        # TODO : add type check
        if len(stimuli_list) != len(labels_list):
            raise RuntimeError(f"stimuli_list and labels_list have different lengths ({len(stimuli_list)}, {len(labels_list)}), every stimuli need to have a corresponding label")
        self._N2D2_object.load([Tensor.from_numpy(i).N2D2() for i in stimuli_list], labels_list)

    def transpose_load(self, inputs:List[ndarray], labels:List[int], perm:List[int]=None):
        """
        TODO : write doc string
        """
        if len(perm) != 3:
            raise RuntimeError(f"perm list should be of size 3, but is of size {len(perm)} instead.")
        self.load([np.transpose(i, perm) for i in inputs], labels)
