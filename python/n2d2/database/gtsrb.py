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
from n2d2.database import AbstractDatabase, _database_parameters
from n2d2.utils import inherit_init_docstring
from n2d2.n2d2_interface import ConventionConverter
import N2D2


@inherit_init_docstring()
class GTSRB(AbstractDatabase):
    """
    The German Traffic Sign Benchmark (https://benchmark.ini.rub.de/) is a multi-class, \
    single-image classification challenge held at the International Joint Conference on Neural Networks (IJCNN) 2011.
    """

    _type = "GTSRB"
    _parameters = {
        "validation": "validation",
    }
    _parameters.update(_database_parameters)
    _convention_converter= ConventionConverter(_parameters)
    _N2D2_constructors = N2D2.GTSRB_DIR_Database
    def __init__(self, validation, **config_parameters):
        """
        :param validation: Fraction of the learning set used for validation
        :type validation: float
        """
        AbstractDatabase.__init__(self, **config_parameters)

        # No optional args
        self._parse_optional_arguments([])
        self._N2D2_object = self._N2D2_constructors(validation, **self.n2d2_function_argument_parser(self._optional_constructor_arguments))
        self._set_N2D2_parameters(self._config_parameters)
        self.load_N2D2_parameters(self.N2D2())
