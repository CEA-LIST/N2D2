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

from abc import ABC, abstractmethod
from n2d2 import ConventionConverter, generate_name, inherit_init_docstring, check_types, Tensor, global_variables
from n2d2.error_handler import WrongValue
from n2d2.deepnet import DeepNet
from n2d2.transform import Transformation, Composite
from n2d2.database import Database
from n2d2.n2d2_interface import N2D2_Interface
from n2d2.converter import from_N2D2_object

class Provider(N2D2_Interface,ABC):
    _parameters={
        "name": "Name",
        "batch_size": "batchSize",
        "composite_stimuli": "compositeStimuli",
        "database": "Database",
        "size": "Size",
        "random_read": "RandomRead",
    }
    _convention_converter= ConventionConverter(_parameters)
    @abstractmethod
    def __init__(self, **config_parameters):
        """
        :param name: Provider name, default = ``Provider_id``
        :type name: str, optional
        """
        N2D2_Interface.__init__(self, **config_parameters)

        if 'name' in config_parameters:
            self._name = config_parameters.pop['name']
        else:
            self._name = generate_name(self)
        self._deepnet = None

    def get_deepnet(self):
        """
        :returns: DeepNet object
        :rtype: :py:class:`n2d2.deepnet.DeepNet`
        """
        return self._deepnet

    @check_types
    def set_deepnet(self, deepnet:DeepNet):
        self._deepnet = deepnet
        deepnet.set_provider(self)

    def get_size(self):
        return self._N2D2_object.getSize()

    def get_batch_size(self):
        """
        :returns: Batch size
        :rtype: int
        """
        return self._N2D2_object.getBatchSize()

    def dims(self):
        return self.get_size() + [self.get_batch_size()]

    def shape(self):
        return list(reversed(self.dims()))

    def get_name(self):
        """
        :returns: Name of the data provider
        :rtype: str
        """
        return self._name

@inherit_init_docstring()
class DataProvider(Provider):
    """
    Provide the data to the network.
    """
    _type = "DataProvider"
    _N2D2_constructors = N2D2.StimuliProvider
    # Be careful to match default parameters in python and N2D2 constructor
    def __init__(self, database, size, random_read=False, **config_parameters):
        """
        :param database: Database used to read data from
        :type database: :py:class:`n2d2.database.Database`
        :param size: Size of the data, in the format [W, H, C].
        :type size: list
        :param batch_size: Batch size, default=1
        :type batch_size: int, optional
        :param composite_stimuli: If ``True``, use pixel-wise stimuli labels, default=False
        :type composite_stimuli: bool, optional
        :param random_read: if ``False`` we use get_batch when iterating other the provider, else we use get ``get_random_batch``, default = False
        :type random_read: boolean, optional
        """
        Provider.__init__(self, **config_parameters)

        self._constructor_arguments.update({
            'database': database,
            'size': size
        })

        self._parse_optional_arguments(['batch_size', 'composite_stimuli'])

        self._N2D2_object = self._N2D2_constructors(database=self._constructor_arguments['database'].N2D2(),
                                                 size=self._constructor_arguments['size'],
                                                 **self.n2d2_function_argument_parser(self._optional_constructor_arguments))
        self._set_N2D2_parameters(self._config_parameters)

        # Dictionary of transformation objects
        self._transformations = []
        self._otf_transformations = []

        self._partition = 'Test'
        # Index for __iter__ method !
        self._index = 0
        self._random_read = random_read

    def set_reading_randomly(self, random_read):
        """
        Set if we use get_batch or get_random_batch when iterating other the provider

        :param random_read: If ``True``, the provider will give stimuli in a random order.
        :type random_read: bool
        """
        self._random_read = random_read

    def set_partition(self, partition):
        """
        :param partition: The partition can be  ``Learn``, ``Validation``, ``Test``,  ``Unpartitioned``
        :type partition: str
        """
        if partition not in N2D2.Database.StimuliSet.__members__.keys():
            raise WrongValue("partition", partition, N2D2.Database.StimuliSet.__members__.keys())
        self._partition = partition

    def get_partition(self):
        """
        :returns: The partition can be  ``Learn``, ``Validation``, ``Test``,  ``Unpartitioned``
        :rtype: str
        """
        return N2D2.Database.StimuliSet.__members__[self._partition]


    def get_data(self):
        """
        :returns: Data.
        :rtype: :py:class:`n2d2.Tensor`
        """
        return Tensor.from_N2D2(self._N2D2_object.getData())


    def get_labels(self):
        """
        :returns: Labels associated with the current batch.
        :rtype: :py:class:`n2d2.Tensor`
        """
        return Tensor.from_N2D2(self._N2D2_object.getLabelsData())


    def get_database(self):
        """
        :returns: Give the database
        :rtype: :py:class:`n2d2.database.Database`
        """
        return self._constructor_arguments['database']

    def read_random_batch(self):
        """
        :return: Return a random batch
        :rtype: :py:class:`n2d2.Tensor`
        """

        self._deepnet = DeepNet()
        self._deepnet.set_provider(self)
        self._deepnet.N2D2().initialize()

        self._N2D2_object.readRandomBatch(set=self.get_partition())
        return Tensor.from_N2D2(self._N2D2_object.getData())._set_cell(self)

    def set_batch(self, shuffle=True):
        """
        :param shuffle: If true the data will be shuffled, default=True
        :type shuffle: bool, optional
        """
        self._N2D2_object.setBatch(set=self.get_partition(), randShuffle=shuffle)

    def all_batchs_provided(self):
        """
        :return: Return True if all batchs have been provided for the current partition.
        :rtype: bool
        """
        return self._N2D2_object.allBatchsProvided(self.get_partition())

    def normalize_stimuli(self):
        """Normalize the integer value range of stimuli between [0,1]
        """
        self._N2D2_object.normalizeIntegersStimuli(self._N2D2_object.getDatabase().getStimuliDepth())

    @check_types
    def read_batch(self, idx:int=None):
        """
        :param idx: Start index to begin reading the stimuli
        :type idx: int
        :return: Return a batch of data
        :rtype: :py:class:`n2d2.Tensor`
        """
        self._deepnet = DeepNet()
        self._deepnet.set_provider(self)
        self._deepnet.N2D2().initialize()
        if idx is None: # if idx is not enough as this will be evaluate to false if idx=0
            self._N2D2_object.readBatch(set=self.get_partition())
        else:
            self._N2D2_object.readBatch(set=self.get_partition(), startIndex=idx)
        return Tensor.from_N2D2(self._N2D2_object.getData())._set_cell(self)

    @check_types
    def add_transformation(self, transformation:Transformation):
        """Apply transformation to the dataset.

        :param transformation: Transformation to apply
        :type transformation: :py:class:`n2d2.transformation.Transformation`
        """
        if isinstance(transformation, Composite):
            for trans in transformation.get_transformations():
                self._N2D2_object.addTransformation(trans.N2D2(), trans.get_apply_set())
                self._transformations.append(trans)
        else:
            self._N2D2_object.addTransformation(transformation.N2D2(), transformation.get_apply_set())
            self._transformations.append(transformation)

    def add_on_the_fly_transformation(self, transformation):
        """Add transformation to apply to the dataset when reading them.

        :param transformation: Transformation to apply
        :type transformation: :py:class:`n2d2.transformation.Transformation`
        """
        if isinstance(transformation, Composite):
            for trans in transformation.get_transformations():
                self._N2D2_object.addOnTheFlyTransformation(trans.N2D2(), trans.get_apply_set())
                self._transformations.append(transformation)
        else:
            self._N2D2_object.addOnTheFlyTransformation(transformation.N2D2(), transformation.get_apply_set())
            self._transformations.append(transformation)
    
    def get_transformations(self):
        """Return the transformation (``OnTheFly`` or not) associated to the provider object.
        """
        transformations = []
        composite_trans = self._N2D2_object.getOnTheFlyTransformation(self.get_partition())
        for i in range(composite_trans.size()):
            transformations.append(from_N2D2_object(composite_trans[i]))
        trans = self._N2D2_object.getTransformation(self.get_partition())
        for i in range(trans.size()):
            transformations.append(from_N2D2_object(trans[i]))
        return transformations

    def batch_number(self):
        return self._index

    def __next__(self):
        """
        Magic method called by __iter__ to access the next element
        """
        if self._index < int(self.get_database().get_nb_stimuli(self._partition) / self.get_batch_size()):
            self._index += 1
            if self._random_read:
                return self.read_random_batch()
            return self.read_batch(self._index-1)
        raise StopIteration


    def __iter__(self):
        self._index = 0
        return self


    def __str__(self):
        output = "'" + self.get_name() + "' " + self._type + N2D2_Interface.__str__(self)
        if len(self._transformations) > 0:
            output += "[Transformations="
            for trans in self._transformations:
                output += trans.__str__() + ", "
            output = output[:-2]
            output += "]"
        return output

class TensorPlaceholder(Provider):
    """
    A provider used to stream a **single** tensor through a neural network.
    This is automatically used when you pass a Tensor that doesn't come from :py:class:`n2d2.provider.DataProvider`.
    """
    @check_types
    def __init__(self, inputs:Tensor, labels:Tensor =None, **config_parameters):
        """
        :param inputs: The data tensor you want to stream, if N2D2 is compiled with CUDA it must be CUDA, the datatype used should be `float`.
        :type inputs: :py:class:`n2d2.Tensor`
        :param labels: Labels associated with the tensor you want to stream, the datatype of labels must be integer, default= None
        :type labels: :py:class:`n2d2.Tensor`, optional
        """
        Provider.__init__(self, **config_parameters)

        self._tensor = inputs
        dims = [self._tensor.N2D2().dimX(), self._tensor.N2D2().dimY(), self._tensor.N2D2().dimZ()]
        self._N2D2_object = N2D2.StimuliProvider(database=Database().N2D2(),
                                                 size=dims,
                                                 batchSize=self._tensor.N2D2().dimB())

        self._set_streamed_tensor()
        if labels:
            self._labels = labels
            self._set_streamed_label()

        self._deepnet = DeepNet()
        self._deepnet.set_provider(self)
        self._deepnet.N2D2().initialize()

        self._tensor.cell = self
        self.set_partition("Learn")

    def _set_streamed_label(self):
        if not (self._labels.data_type() == 'int' or self._labels.data_type() == 'i'):
            raise RuntimeError("Labels datatype must be int, is " + self._labels.data_type() + "instead.")
        self._set_N2D2_parameter('StreamLabel', True)
        self._N2D2_object.setStreamedLabel(self._labels.N2D2())

    def set_partition(self, partition):
        """
        :param partition: The partition can be  ``Learn``, ``Validation``, ``Test``,  ``Unpartitioned``
        :type partition: str
        """
        if partition not in N2D2.Database.StimuliSet.__members__.keys():
            raise WrongValue("partition", partition, N2D2.Database.StimuliSet.__members__.keys())
        self._partition = partition

    def get_partition(self):
        """
        :returns: The partition can be  ``Learn``, ``Validation``, ``Test``,  ``Unpartitioned``
        :rtype: str
        """
        return N2D2.Database.StimuliSet.__members__[self._partition]

    def _set_streamed_tensor(self):
        """
        Streamed a tensor in a data provider to simulate the output of a database.
        The model of the tensor is defined by the compilation of the library.
        """
        if global_variables.cuda_available:
            if not self._tensor.is_cuda:
                self._tensor.cuda()
            self._tensor.htod()
        if not (self._tensor.data_type() == "f" or self._tensor.data_type() == "float"):
            raise TypeError("You try to stream a Tensor with the datatype '" + self._tensor.data_type() + "' you should provide a Tensor with a 'float' datatype.")
        self._set_N2D2_parameter('StreamTensor', True)
        self._N2D2_object.setStreamedTensor(self._tensor.N2D2())

    def __call__(self):
        return self._tensor


    def __str__(self):
        return f"'{self.get_name()}' TensorPlaceholder"

class MultipleOutputsProvider(Provider):
    """
    Provider used to give multiple tensors to the network.
    """
    def __init__(self, size, batch_size=1):
        """
        :param size: List of ``X``, ``Y`` and ``Z`` dimensions
        :type size: list
        :param batch_size: Batch size, default=1
        :type batch_size: int, optional
        """
        self._N2D2_object = N2D2.StimuliProvider(database=N2D2.Database(),
                                                 size=size,
                                                 batchSize=batch_size)
        self._deepnet = DeepNet()
        self._deepnet.set_provider(self)
        self._name = generate_name(self)
