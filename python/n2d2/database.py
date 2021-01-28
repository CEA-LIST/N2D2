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

import N2D2
import n2d2
from n2d2.n2d2_interface import N2D2_Interface

"""
At the moment, this class is rather superfluous, and servers mainly for hiding
the raw N2D2 binding class. However, in the long term it could serve as a 
canvas for defining datasets without the need to write a N2D2 database driver.
Alternatively, this could simply be done by the corresponding Pytorch functions
since there is no GPU model involved.
"""

# TODO: Abstract classes?
class Database(N2D2_Interface):
    """
    Database loader object.
    """

    def __init__(self, **config_parameters):
        N2D2_Interface.__init__(self, **config_parameters)

    def get_nb_stimuli(self, partition):
        return self._N2D2_object.getNbStimuli(N2D2.Database.StimuliSet.__members__[partition])

    def load(self, dataPath, **kwargs):
        self._N2D2_object.load(dataPath=dataPath, **kwargs)

    def __str__(self):
        return self._type + N2D2_Interface.__str__(self)

    def convert_to_INI_section(self):
        output = "[database]\n"
        output += "Type=" + self._INI_type + "\n"
        #N2D2_Interface.create_INI_section()
        return output

class DIR(Database):
    """
    Allow you to load your own database.
    """
    _INI_type = 'DIR_Database'
    _type = "DIR"
    def __init__(self, **config_parameters):
        Database.__init__(self, **config_parameters)
        self._parse_optional_arguments(['LoadDataInMemory'])
        self._N2D2_object = N2D2.DIR_Database(**self._optional_constructor_arguments)
        self._set_N2D2_parameters(self._config_parameters)

    def load(self, dataPath, depth=0, labelPath="", labelDepth=0):
        """
        :param dataPath: Path to the dataset file.
        :type dataPath: str
        :param depth: Number of sub-directory levels to include.
        :type depth: int
        :param labelPath: Path to the label file.
        :type labelPath: str, optional
        :param labelDepth: Number of sub-directory name levels used to form the data labels.
        :type labelDepth: int
        """
        self._N2D2_object.loadDir(dataPath, depth, labelPath, labelDepth)

class MNIST(Database):

    _INI_type = 'MNIST_IDX_Database'
    _type = "MNIST"

    def __init__(self, DataPath, **config_parameters):
        Database.__init__(self, **config_parameters)

        self._constructor_arguments.update({
            'DataPath': DataPath,
        })
        self._parse_optional_arguments(['LabelPath', 'ExtractROIs', 'Validation'])
        self._N2D2_object = N2D2.MNIST_IDX_Database(self._constructor_arguments['DataPath'],
                                                    **self._optional_constructor_arguments)
        self._set_N2D2_parameters(self._config_parameters)

class ILSVRC2012(Database):

    _INI_type = 'ILSVRC2012_Database'
    _type = "ILSVRC2012"

    def __init__(self, Learn, **config_parameters):
        Database.__init__(self, **config_parameters)
        self._constructor_arguments.update({
            'Learn': Learn,
        })
        self._parse_optional_arguments(['useValidationForTest', 'backgroundClass'])
        self._N2D2_object = N2D2.ILSVRC2012_Database(self._constructor_arguments['Learn'],
                                                    **self._optional_constructor_arguments)
        self._set_N2D2_parameters(self._config_parameters)





