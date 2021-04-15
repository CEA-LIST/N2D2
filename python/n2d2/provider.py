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


class Provider(N2D2_Interface):
    def __init__(self, **config_parameters):
        N2D2_Interface.__init__(self, **config_parameters)

        if 'name' in config_parameters:
            self._name = config_parameters.pop['name']
        else:
            self._name = n2d2.global_variables.generate_name(self)
        self._deepnet = None

    def get_deepnet(self):
        """
        :returns: DeepNet object
        :rtype: :py:class:`n2d2.deepnet.DeepNet` 
        """
        return self._deepnet

    def set_deepnet(self, deepnet):
        self._deepnet = deepnet
        deepnet.set_provider(self)

    def get_size(self):
        return self._N2D2_object.getSize()

    def dims(self):
        return self._N2D2_object.getData().dims()

    def get_name(self):
        """
        :returns: Name of the data provider
        :rtype: str
        """
        return self._name

class DataProvider(Provider):
    """
    Provide the data to the network.
    """
    _type = "DataProvider"

    # Be careful to match default parameters in python and N2D2 constructor
    def __init__(self, database, size, random_read=False, **config_parameters):
        """
        :param database: Database used to read data from
        :type database: :py:class:`n2d2.database.Database` 
        :param batch_size: Batch size, default=1
        :type batch_size: int, optional
        :param composite_stimuli: If true, use pixel-wise stimuli labels, default=False
        :type composite_stimuli: bool, optional
        :param size: Size of the data
        :type size: list
        :param random_read: if False we use get_batch when iterating other the provider, else we use get get_random_batch, default = False
        :type random_read: boolean, optional
        """
        Provider.__init__(self, **config_parameters)

        self._constructor_arguments.update({
            'database': database,
            'size': size
        })


        self._parse_optional_arguments(['batch_size', 'composite_stimuli'])

        self._N2D2_object = N2D2.StimuliProvider(database=self._constructor_arguments['database'].N2D2(),
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

        :param random_read: If True, the provider will give stimuli in a random order.
        :type random_read: bool
        """
        self._random_read = random_read

    def set_partition(self, partition):
        """
        :param partition: The partition can be  ```Test``, ``Validation``, ``Test``,  ``Unpartitioned``
        :type partition: str 
        """
        if partition not in N2D2.Database.StimuliSet.__members__.keys():
            raise n2d2.error_handler.WrongValue("partition", partition,
                                                " ".join(N2D2.Database.StimuliSet.__members__.keys()))
        self._partition = partition

    def get_partition(self):
        """
        :returns: The partition can be  ```Test``, ``Validation``, ``Test``,  ``Unpartitioned``
        :rtype: str
        """
        return N2D2.Database.StimuliSet.__members__[self._partition]


    def get_data(self):
        """
        :returns: Data.
        :rtype: :py:class:`n2d2.Tensor`
        """
        return n2d2.Tensor.from_N2D2(self._N2D2_object.getData()) 

    def get_batch_size(self):
        """
        :returns: Batch size
        :rtype: int
        """
        return self._N2D2_object.getBatchSize()

    def get_database(self):
        """
        :returns: Give the database
        :rtype: :py:class:`n2d2.database.Database`
        """
        return self._constructor_arguments['database']

    def read_random_batch(self):
        """
        :return: Return a random batch 
        :rtype: :py:class:`n2d2.tensor.Tensor`
        """

        self._deepnet = n2d2.deepnet.DeepNet()
        self._deepnet.set_provider(self)
        self._N2D2_object.readRandomBatch(set=self.get_partition())
        return n2d2.Tensor.from_N2D2(self._N2D2_object.getData())._set_cell(self)

    def read_batch(self, idx):
        """
        :param idx: Start index to begin reading the stimuli
        :type idx: int
        :return: Return a batch of data
        :rtype: :py:class:`n2d2.tensor.Tensor`
        """
        self._deepnet = n2d2.deepnet.DeepNet()
        self._deepnet.set_provider(self)
        self._N2D2_object.readBatch(set=self.get_partition(), startIndex=idx)
        return n2d2.Tensor.from_N2D2(self._N2D2_object.getData())._set_cell(self)

    def add_transformation(self, transformation):
        """Apply transformation to the dataset.

        :param transformation: Transformation to apply
        :type transformation: :py:class:`n2d2.transformation.Transformation`
        """ 
        if isinstance(transformation, n2d2.transform.Composite):
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
        if isinstance(transformation, n2d2.transform.Composite):
            for trans in transformation.get_transformations():
                self._N2D2_object.addOnTheFlyTransformation(trans.N2D2(), trans.get_apply_set())
                self._transformations.append(transformation)
        else:
            self._N2D2_object.addOnTheFlyTransformation(transformation.N2D2(), transformation.get_apply_set())
            self._transformations.append(transformation)
            
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
            else:
                return self.read_batch(self._index-1)
        else:
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
    A provider used to stream a single tensor through a neural network.
    This is automatically used when you pass a Tensor that doesn't come from :py:class:`n2d2.provider.DataProvider`.
    """
    def __init__(self, inputs, **config_parameters):
        """
        :param inputs: The tensor you want to stream
        :type inputs: :py:class:`N2D2.tensor.Tensor`
        """
        Provider.__init__(self, **config_parameters)

        if isinstance(inputs, n2d2.tensor.Tensor):
            self._tensor = inputs
        else:
            raise ValueError("Wrong input of type " + str(type(inputs)))
            # n2d2.error_handler.wrong_input_type("inputs", type(inputs), [type(list), 'n2d2.tensor.Tensor', 'N2D2.BaseTensor'])
        dims = [self._tensor.N2D2().dimX(), self._tensor.N2D2().dimY(), self._tensor.N2D2().dimZ()]
        self._N2D2_object = N2D2.StimuliProvider(database=n2d2.database.Database().N2D2(),
                                                 size=dims,
                                                 batchSize=self._tensor.N2D2().dimB())
        self._set_N2D2_parameter('StreamTensor', True)
        self._set_streamed_tensor()

        self._deepnet = n2d2.deepnet.DeepNet()
        self._deepnet.set_provider(self)


    def _set_streamed_tensor(self): # TODO : only used in the constructor at the moment, remove the method ?
        """
        Streamed a tensor in a data provider to simulate the output of a database.
        The model of the tensor is defined by the compilation of the library.
        """
        if N2D2.cuda_compiled and not self._tensor.is_cuda: 
            raise ValueError("You compiled N2D2 with CUDA this doesn't match the tensor model you are providing to the network.")
        self._N2D2_object.setStreamedTensor(self._tensor.N2D2())

    def __call__(self):
        return self._tensor

    

    def __str__(self):
        return "'" + self.get_name() + "' TensorPlaceholder"



"""
TODO : Not up to date to keep ?
"""

# class Input(Provider):
#     
#     def __init__(self, dims, model=None, **config_parameters):
#         Provider.__init__(self, **config_parameters)

#         if model is None:
#             model = n2d2.global_variables.default_model

#         if model == "Frame":
#             self._tensor = n2d2.Tensor(dims)
#         elif model == "Frame_CUDA":
#             self._tensor = n2d2.CudaTensor(dims)
#         else:
#             ValueError("Invalid model '" + model + "'")

#         # n2d2.error_handler.wrong_input_type("inputs", type(inputs), [type(list), 'n2d2.tensor.Tensor', 'N2D2.BaseTensor'])
#         provider_dims = [self._tensor.N2D2().dimX(), self._tensor.N2D2().dimY(), self._tensor.N2D2().dimZ()]
#         self._N2D2_object = N2D2.StimuliProvider(database=n2d2.database.Database().N2D2(),
#                                                  size=provider_dims,
#                                                  batchSize=self._tensor.N2D2().dimB())
#         self._set_N2D2_parameter('StreamTensor', True)
#         self._N2D2_object.setStreamedTensor(self._tensor.N2D2())

#         self._deepnet = n2d2.deepnet.DeepNet()
#         self._deepnet.set_provider(self)


#     def set_streamed_tensor(self, tensor):
#         self._N2D2_object.setStreamedTensor(tensor)

#     def __call__(self, inputs):

#         if not self.dims() == inputs.dims():
#             raise RuntimeError("Received input tensor with dims " + str(inputs.dims()) +
#                                " but object dims are " + str(self.dims()) +
#                                ". Input dimensions cannot change after initialization")

#         if "Cuda" in str(type(self._tensor)):
#             if not "Cuda" in str(type(inputs)):
#                 raise RuntimeError("'inputs' argument is not a cuda tensor, but internal tensor is.")
#             self._tensor.N2D2().synchronizeHToD()
#         else:
#             if "Cuda" in str(type(inputs)):
#                 raise RuntimeError("inputs is a cuda tensor, but internal tensor is not.")

#         self._N2D2_object.setStreamedTensor(inputs.N2D2())


#         return n2d2.tensor.GraphTensor(self._tensor, self)

#     def get_name(self):
#         return self._name

#     def __str__(self):
#         return "'" + self.get_name() + "' Input"

