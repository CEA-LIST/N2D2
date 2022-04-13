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
class ILSVRC2012(Database):
    """
    ILSVRC2012 database :cite:`ILSVRC15`.
    """

    _type = "ILSVRC2012"
    _parameters = {
        "use_validation_for_test": "useValidationForTest",
        "learn": "Learn",
        "random_partitioning": "RandomPartitioning",
        "background_class": "backgroundClass"
    }
    _parameters.update(_database_parameters)
    _convention_converter= ConventionConverter(_parameters)
    _N2D2_constructors = N2D2.ILSVRC2012_Database

    def __init__(self, learn, **config_parameters):
        """
        :param learn: Fraction of images used for the learning
        :type learn: float
        :param use_validation_for_test: If ``True``, use the validation partition for test, default=False
        :type use_validation_for_test: bool, optional
        """
        N2D2_Interface.__init__(self, **config_parameters)
        self._constructor_arguments.update({
            'learn': learn,
        })
        self._parse_optional_arguments(['use_validation_for_test', 'background_class'])
        self._N2D2_object = self._N2D2_constructors(self._constructor_arguments['learn'],
                                                    **self.n2d2_function_argument_parser(self._optional_constructor_arguments))
        self._set_N2D2_parameters(self._config_parameters)
        self.load_N2D2_parameters(self.N2D2())
