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

from n2d2.n2d2_interface import N2D2_Interface
from n2d2 import ConventionConverter
from abc import ABC, abstractmethod

from n2d2.utils import check_types, download as download_file
from os import getenv
from os.path import expanduser

# At the moment, this class is rather superfluous, and servers mainly for hiding
# the raw N2D2 binding class. However, in the long term it could serve as a
# canvas for defining datasets without the need to write a N2D2 database driver.
# Alternatively, this could simply be done by the corresponding Pytorch functions
# since there is no GPU model involved.



_database_parameters = {
    "default_label": "DefaultLabel",
    "rois_margin": "ROIsMargin",
    "random_partitioning": "RandomPartitioning",
    "data_file_label": "DataFileLabel",
    "composite_label": "CompositeLabel",
    "target_data_path": "TargetDataPath",
    "multi_channel_match": "MultiChannelMatch",
    "multi_channel_replace": "MultiChannelReplace"
}

class AbstractDatabase(N2D2_Interface, ABC):

    # Name of the database
    _type = ""

    # List of links ot download database from.
    _download_links = []

    @abstractmethod
    def __init__(self, **config_parameters):
        """
        :param load_data_in_memory: if `True` cache data in memory, default=False
        :type load_data_in_memory: boolean, optional
        """
        N2D2_Interface.__init__(self, **config_parameters)

    @classmethod
    def is_downloadable(cls)->bool:
        """
        :return: ``True`` if the database is downloadable.
        :rtype: bool
        """
        return cls._download_links != []

    
    @classmethod
    def download(cls, path:str=None)->None:
        """Download the dataset at the defined path

        :param path: Path where to download the dataset, default=$(N2D2_DATA)
        :type path: str, optional
        """
        if path is None:
            path = getenv("N2D2_DATA")
            if path is None:
                path=f"{expanduser('~')}+/DATABASE/"
        if not cls.is_downloadable():
            raise NotImplementedError(f"Database {cls.__name__} does not support download !")
        for url in cls._download_links:
            download_file(url, path, cls._type)

    def get_nb_stimuli(self, partition):
        """
        Return the number fo stimuli  available for this partition

        :param partition: The partition can be  ``Learn``, ``Validation``, ``Test``,  ``Unpartitioned``
        :type partition: str
        """
        return self.N2D2().getNbStimuli(N2D2.Database.StimuliSet.__members__[partition])

    def get_partition_summary(self):
        """
        Print the number of stimuli for each partition.
        """
        learn = self.get_nb_stimuli("Learn")
        test = self.get_nb_stimuli("Test")
        validation = self.get_nb_stimuli("Validation")
        unpartitioned = self.get_nb_stimuli("Unpartitioned")
        total = validation + learn + test + unpartitioned
        if total != 0:
            print("Number of stimuli : " + str(total) +"\n"+
            "Learn         : " + str(learn) + " stimuli (" + str(round(((learn/total) * 100), 2)) + "%)\n"+
            "Test          : " + str(test) + " stimuli (" + str(round(((test/total) * 100), 2)) + "%)\n"+
            "Validation    : " + str(validation) + " stimuli (" + str(round(((validation/total) * 100), 2)) + "%)\n"+
            "Unpartitioned : " + str(unpartitioned) + " stimuli (" + str(round(((unpartitioned/total) * 100), 2)) + "%)\n"
            )
        else:
            print("No stimulus in the database !")

    def get_label_name(self, label_idx):
        """
        :param label_idx: Index of the label
        :type label_idx: int
        :returns: Label name
        :rtype: string
        """
        return self._N2D2_object.getLabelName(label_idx)

    @check_types
    def partition_stimuli(self, learn:float, validation:float, test:float):
        """Partition the ``Unpartitioned`` data with the given ratio (the sum of the given ratio must be equal to 1).

        :param learn: Ratio for the ``Learn`` partition.
        :type learn: float
        :param validation: Ratio for the ``Validation`` partition.
        :type validation: float
        :param test: Ratio for the ``Test`` partition.
        :type test: float
        """
        if learn + validation + test > 1:
            raise ValueError("The total partition ratio cannot be higher than 1")
        self._N2D2_object.partitionStimuli(learn, validation, test)


    def load(self, data_path, label_path='', extract_ROIs=False):
        self._N2D2_object.load(dataPath=data_path, labelPath=label_path, extractROIs=extract_ROIs)

    def __str__(self):
        return self._type + N2D2_Interface.__str__(self)

class Database(AbstractDatabase):
    """
    Database loader object.
    """

    _type = ""
    _parameters={
        "load_data_in_memory": "loadDataInMemory",
    }
    _N2D2_constructors = N2D2.Database
    _convention_converter = ConventionConverter(_parameters)

    # This constructor is not called by children, because not abstract class
    def __init__(self, **config_parameters):
        """
        :param load_data_in_memory: if `True` cache data in memory, default=False
        :type load_data_in_memory: boolean, optional
        """
        AbstractDatabase.__init__(self, **config_parameters)

        self._parse_optional_arguments(['load_data_in_memory'])
        self._N2D2_object = self._N2D2_constructors(**self.n2d2_function_argument_parser(self._optional_constructor_arguments))
        self._set_N2D2_parameters(self._config_parameters)
