"""
    (C) Copyright 2020 CEA LIST. All Rights Reserved.
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
from n2d2.database.database import Database, _database_parameters
from n2d2.utils import inherit_init_docstring
from n2d2.n2d2_interface import N2D2_Interface, ConventionConverter
import N2D2


@inherit_init_docstring()
class CIFAR100(Database):
    """
    CIFAR100 database :cite:`Krizhevsky2009`.
    """

    _type = "CIFAR100"
    _parameters = {
        "use_coarse": "useCoarse",
        "validation": "validation",
        "use_test_for_validation": "useTestForVal",
    }  
    _parameters.update(_database_parameters)

    _convention_converter= ConventionConverter(_parameters)
    _N2D2_constructors = N2D2.CIFAR100_Database
    def __init__(self, **config_parameters):
        """
        :param data_path: Path to the database, default="``$N2D2_DATA``/cifar-100-binary"
        :type data_path: str, optional
        :param validation: Fraction of the learning set used for validation, default=0.0
        :type validation: float, optional
        :param use_coarse: If ``True``, use the coarse labeling (10 labels instead of 100), default=False
        :type use_coarse: bool, optional
        """
        N2D2_Interface.__init__(self, **config_parameters)

        self._parse_optional_arguments(['validation', 'use_coarse', "use_test_for_validation"])
        self._N2D2_object = self._N2D2_constructors(**self.n2d2_function_argument_parser(self._optional_constructor_arguments))
        self._set_N2D2_parameters(self._config_parameters)
        self.load_N2D2_parameters(self.N2D2())