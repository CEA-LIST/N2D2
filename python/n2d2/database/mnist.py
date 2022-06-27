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
from n2d2.database.database import AbstractDatabase, _database_parameters
from n2d2.utils import inherit_init_docstring
from n2d2.n2d2_interface import ConventionConverter
import N2D2


@inherit_init_docstring()
class MNIST(AbstractDatabase):
    """
    MNIST database :cite:`LeCun1998`.
    Label are hard coded, you don't need to specify a path to the label file.
    """
    _type = "MNIST"
    _parameters = {
        "extract_roi": "extractROIs",
        "validation": "validation",
        "label_path": "labelPath",
        "stimuli_per_label_train": "StimuliPerLabelTrain",
        "stimuli_per_label_test": "StimuliPerLabelTest",
    }
    _parameters.update(_database_parameters)
    _convention_converter= ConventionConverter(_parameters)
    _N2D2_constructors = N2D2.MNIST_IDX_Database

    def __init__(self, data_path, **config_parameters):
        """
        :param data_path: Path to the database
        :type data_path: str
        :param label_path: Path to the label, default=""
        :type label_path: str, optional
        :param extract_roi: Set if we extract region of interest, default=False
        :type extract_roi: boolean, optional
        :param validation: Fraction of the learning set used for validation, default=0.0
        :type validation: float, optional
        """
        AbstractDatabase.__init__(self, **config_parameters)

        self._constructor_arguments.update({
            'data_path': data_path,
        })
        self._parse_optional_arguments(['label_path', 'extract_roi', 'validation'])

        self._N2D2_object = self._N2D2_constructors(self._constructor_arguments['data_path'],
                                                    **self.n2d2_function_argument_parser(self._optional_constructor_arguments))
        self._set_N2D2_parameters(self._config_parameters)
        self.load_N2D2_parameters(self.N2D2())
