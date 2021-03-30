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

class Database(N2D2_Interface):
    """
    Database loader object.
    """

    _INI_type = 'Database'
    _type = ""

    # This constructor is not called by children, because not abstract class
    def __init__(self, **config_parameters):
        N2D2_Interface.__init__(self, **config_parameters)

        self._parse_optional_arguments(['loadDataInMemory'])
        self._N2D2_object = N2D2.Database(**self._optional_constructor_arguments)
        self._set_N2D2_parameters(self._config_parameters)

    def get_nb_stimuli(self, partition):
        return self._N2D2_object.getNbStimuli(N2D2.Database.StimuliSet.__members__[partition])

    def get_label_name(self, label_idx):
        """
        :param label_idx: Index of the label 
        :type label_idx: int
        :returns: Label name
        :rtype: string
        """
        return self._N2D2_object.getLabelName(label_idx)

    def partition_stimuli(self, learn, validation, test):
        """Create partitions of the data with the given ratio.
        :param learn: Ratio for the learning partition.
        :type learn: float
        :param validation: Ratio for the validation partition.
        :type validation: float
        :param test: Ratio for the test partition.
        :type test: float
        """
        if learn + validation + test > 1:
            raise ValueError("The total partition ratio cannot be higher than 1")
        self._N2D2_object.partitionStimuli(learn, validation, test)


    def load(self, dataPath, **kwargs):
        self._N2D2_object.load(dataPath=dataPath, **kwargs)

    def __str__(self):
        return self._type + N2D2_Interface.__str__(self)



class DIR(Database):
    """
    Allow you to load your own database.
    """
    _INI_type = 'DIR_Database'
    _type = "DIR"
    def __init__(self, **config_parameters):
        N2D2_Interface.__init__(self, **config_parameters)
        self._parse_optional_arguments(['loadDataInMemory'])
        self._N2D2_object = N2D2.DIR_Database(**self._optional_constructor_arguments)
        self._set_N2D2_parameters(self._config_parameters)

    def load(self, dataPath, depth=0, labelPath="", labelDepth=0):
        """
        :param dataPath: Path to the dataset file.
        :type dataPath: str
        :param depth: Number of sub-directory levels to include, defaults=0 
        :type depth: int, optional
        :param labelPath: Path to the label file, defaults="" 
        :type labelPath: str, optional
        :param labelDepth: Number of sub-directory name levels used to form the data labels, defaults=0
        :type labelDepth: int, optional
        """
        self._N2D2_object.loadDir(dataPath, depth, labelPath, labelDepth)

class MNIST(Database):
    """
    MNIST database :cite:`LeCun1998`.
    Label are hard coded, you don't need to specify a path to the label file.
    """
    _INI_type = 'MNIST_IDX_Database'
    _type = "MNIST"

    def __init__(self, dataPath, **config_parameters):
        """
        :param dataPath: Path to the database
        :type dataPath: str
        :param labelPath: Path to the label, default=""
        :type labelPath: str, optional
        :param extractROIs: Set if we extract region of interest, default=False
        :type extractROIs: boolean, optional
        :param validation: Fraction of the learning set used for validation, default=0.0
        :type validation: float, optional
        """
        N2D2_Interface.__init__(self, **config_parameters)

        self._constructor_arguments.update({
            'dataPath': dataPath,
        })
        self._parse_optional_arguments(['labelPath', 'extractROIs', 'validation'])
        self._N2D2_object = N2D2.MNIST_IDX_Database(self._constructor_arguments['dataPath'],
                                                    **self._optional_constructor_arguments)
        self._set_N2D2_parameters(self._config_parameters)


class CIFAR100(Database):
    """
    CIFAR100 database :cite:`Krizhevsky2009`.
    """

    _INI_type = 'CIFAR100_Database'
    _type = "CIFAR100"

    def __init__(self, **config_parameters):
        """
        :param validation: Fraction of the learning set used for validation, default=0.0
        :type validation: float, optional
        """
        N2D2_Interface.__init__(self, **config_parameters)

        self._parse_optional_arguments(['validation', 'useCoarse'])
        self._N2D2_object = N2D2.CIFAR100_Database(**self._optional_constructor_arguments)
        self._set_N2D2_parameters(self._config_parameters)


class ILSVRC2012(Database):
    """
    ILSVRC2012 database :cite:`ILSVRC15`.
    """

    _INI_type = 'ILSVRC2012_Database'
    _type = "ILSVRC2012"

    def __init__(self, learn, **config_parameters):
        N2D2_Interface.__init__(self, **config_parameters)
        self._constructor_arguments.update({
            'learn': learn,
        })
        self._parse_optional_arguments(['useValidationForTest', 'backgroundClass'])
        self._N2D2_object = N2D2.ILSVRC2012_Database(self._constructor_arguments['learn'],
                                                    **self._optional_constructor_arguments)
        self._set_N2D2_parameters(self._config_parameters)



class Cityscapes(Database):
    """
    Cityscapes database :cite:`Cordts2016Cityscapes`.
    """

    _INI_type = 'Cityscapes_Database'
    _type = "Cityscapes"

    def __init__(self, **config_parameters):
        """
        :param incTrainExtra: If true, includes the left 8-bit images - trainextra set (19,998 images), default=False
        :type incTrainExtra: boolean, optional
        :param useCoarse: If true, only use coarse annotations (which are the only annotations available for the trainextra set), default=False
        :type useCoarse: boolean, optional 
        :param singleInstanceLabels: If true, convert group labels to single instance labels (for example, cargroup becomes car), default=True
        :type useCoarse: boolean, optional 
        """
        N2D2_Interface.__init__(self, **config_parameters)

        self._parse_optional_arguments(['incTrainExtra', 'useCoarse', 'singleInstanceLabels'])
        self._N2D2_object = N2D2.Cityscapes_Database(**self._optional_constructor_arguments)
        self._set_N2D2_parameters(self._config_parameters)

