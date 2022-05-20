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
import n2d2.global_variables as gb
import N2D2


@inherit_init_docstring()
class Cityscapes(Database):
    """
    Cityscapes database :cite:`Cordts2016Cityscapes`.

    .. warning::
        Don't forget to install the **libjsoncpp-dev** package on your device if you wish to use this database.
    """

    _type = "Cityscapes"
    _parameters = {
        "inc_train_extra": "incTrainExtra",
        "use_coarse": "useCoarse",
        "single_instance_labels": "singleInstanceLabels",
        "labels": "Labels"
    }
    _parameters.update(_database_parameters)
    _convention_converter= ConventionConverter(_parameters)
    if gb.json_compiled:
        _N2D2_constructors = N2D2.Cityscapes_Database

    def __init__(self, **config_parameters):
        """
        :param inc_train_extra: If ``True``, includes the left 8-bit images - ``trainextra`` set (19,998 images), default=False
        :type inc_train_extra: boolean, optional
        :param use_coarse: If ``True``, only use coarse annotations (which are the only annotations available for the ``trainextra`` set), default=False
        :type use_coarse: boolean, optional 
        :param single_instance_labels: If ``True``, convert group labels to single instance labels (for example, ``cargroup`` becomes ``car``), default=True
        :type single_instance_labels: boolean, optional 
        """
        if not gb.json_compiled:
            raise RuntimeError(
                "JSON for C++ library not installed\n\n"
                "\tPlease install the libjsoncpp-dev package and reinstall n2d2\n\n")

        N2D2_Interface.__init__(self, **config_parameters)

        self._parse_optional_arguments(['inc_train_extra', 'use_coarse', 'single_instance_labels'])
        self._N2D2_object = self._N2D2_constructors(**self.n2d2_function_argument_parser(self._optional_constructor_arguments))
        self._set_N2D2_parameters(self._config_parameters)
        self.load_N2D2_parameters(self.N2D2())
