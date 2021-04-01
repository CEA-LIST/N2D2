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
from n2d2 import error_handler
from n2d2.provider import TensorPlaceholder
from functools import reduce

hard_coded_type = {
    "f": float,
    "float": float,
    "i": int,
    "int": int,
    "b": bool,
    "bool": bool,
}


class Tensor():
    
    _tensor_generators = {
        float: N2D2.Tensor_float,
        int: N2D2.Tensor_int,
        bool: N2D2.Tensor_bool,
    }
    _cuda_tensor_generators = {
        float: N2D2.CudaTensor_float,
        int: N2D2.CudaTensor_int,
        # bool: N2D2.CudaTensor_bool, # Not defined
    }
    
    def __init__(self, dims, value=None, cuda=False, defaultDataType=float):
        """
        :param dims: Dimensions of the :py:class:`n2d2.Tensor` object. (Numpy convention)
        :type dims: list
        :param value: A value to fill the :py:class:`n2d2.Tensor` object.
        :type value: Must be coherent with **defaultDataType**
        :param defaultDataType: Type of the data stocked by the tensor, default=float
        :type defaultDataType: type, optional
        """

        if cuda:
            generators = self._cuda_tensor_generators
        else:
            generators = self._tensor_generators
            
        # Dimensions convention on N2D2 are reversed from python. 
        if isinstance(dims, list):
            dims = [d for d in reversed(dims)]
        else:
            raise error_handler.WrongInputType("dims", type(dims), [str(list)])

        if value and not isinstance(value, defaultDataType): # TODO : We may want to try an auto-cast ! 
            raise TypeError("You want to fill the tensor with " + str(type(value)) + " but " + str(defaultDataType) + " is the defaultDataType.")

        if defaultDataType in generators:
            if not value:
                self._tensor = generators[defaultDataType](dims)
            else:
                self._tensor = generators[defaultDataType](dims, value)
                if cuda:
                    # TODO a bug cause the "value" argument to be ignored for CUDA tensor :
                    # example : N2D2.CudaTensor_int([2, 2], value=int(5.0)
                    self._tensor[0:] = value
        else:
            raise TypeError("Unrecognized Tensor datatype " + str(defaultDataType))

        self._dataType = defaultDataType
        self.is_cuda = cuda

    def N2D2(self):
        """
        :return: The N2D2 tensor object
        :rtype: :py:class:`N2D2.BaseTensor`
        """
        return self._tensor

    def nb_dims(self):
        """
        Return the number of dimensions.
        """
        return len(self._tensor.dims())

    def dims(self):
        """
        Return dimensions with N2D2 convention 
        """
        return self._tensor.dims()

    def dimX(self):
        return self._tensor.dimX()

    def dimY(self):
        return self._tensor.dimY()

    def dimZ(self):
        return self._tensor.dimZ()
        
    def dimB(self):
        return self._tensor.dimB()

    def shape(self):
        """
        Return dimensions with python convention 
        """
        return [d for d in reversed(self._tensor.dims())]
    
    def data_type(self):
        """
        Return the data type of the object stored by the tensor.
        """
        return self._dataType

    def _get_index(self, coord):
        """
        :param coord: Tuple of the coordinate
        :type coord: tuple
        From the coordinate returns the 1D index of an element in the tensor.
        """
        dims = self.dims()
        coord = [i for i in reversed(coord)]
        if len(dims) != len(coord):
            raise ValueError(str(len(coord)) + "D array does not match " + str(len(dims)) + "D tensor.") 
        for c, d in zip(coord, dims):
            if not c < d:
                raise ValueError("Coordinate does not fit the dimensions of the tensor, max: "+str(d)+" got " + str(c)) 
        idx = 0
        for i in range(len(dims)):
            if i == 0:
                idx += coord[i]
            else:
                idx += (coord[i] * reduce((lambda x,y: x*y), dims[:i]))
        return idx
        
    def _get_coord(self, index):
        """
        :param index: index of an element
        :type index: int
        From the the 1D index, return the coordinate of an element in the tensor.
        """ 
        coord = []
        for i in self.shapes():
            coord.append(int(index%i))
            index = index/i
        return [i for i in reversed(coord)]

    def reshape(self, new_dims):
        """
        :param new_dims: New dimensions
        :type new_dims: list
        Reshape the Tensor to the specified dims (defined by the Numpy convention). 
        """
        if reduce((lambda x,y: x*y), new_dims) != len(self):
            new_dims_str = ""
            for dim in new_dims:
                new_dims_str += str(dim) +" "
            old_dims_str = ""
            for dim in self.shape():
                old_dims_str += str(dim) +" "
            raise ValueError("new size ("+new_dims_str+"= " + str(reduce((lambda x,y: x*y), new_dims))+") does not match current size ("+ old_dims_str+"= " +str(self.__len__())+")")
        self._tensor.reshape([int(d) for d in reversed(new_dims)])

    def copy(self):
        """
        Copy in memory the Tensor object.
        """
        copy = Tensor(self.shape(), defaultDataType=self.data_type(), cuda=self.is_cuda)
        for i in range(len(copy)):
            copy[i] = self._tensor[i]
        return copy

    def to_numpy(self):
        """
        Create a numpy array equivalent to the tensor.
        """
        try:
            # TODO : Create a dependance to numpy in the library ?
            # Import like this can cause performance issue
            from numpy import array 
        except ImportError:
            raise ImportError("Numpy is not installed")
        return array(self.N2D2()) 

    @classmethod
    def from_numpy(cls, np_array):
        """Convert a numpy array into a tensor.
        TODO : fix /!\ Known issues /!\  
        - Using a 1D numpy array have unintended behaviour 

        :param np_array: A numpy array to convert to a tensor.
        :type np_array: :py:class:`numpy.array`
        :return: Converted tensor
        :rtype: :py:class:`n2d2.Tensor`
        """
        try: 
            # TODO : Create a dependance to numpy in the library ?
            # Import like this can cause performance issue
            from numpy import ndarray 
        except ImportError:
            raise ImportError("Numpy is not installed !")
        if not isinstance(np_array, ndarray):
            raise error_handler.WrongInputType("np_array", type(np_array), ["numpy.array"])

        np_array = np_array.reshape([d for d in reversed(np_array.shape)]) 
        n2d2_tensor = cls([])

        # Retrieving the first element of the numpy array to get dataType.
        try:
            first_element = np_array[0]
        except IndexError:
            raise ValueError('Numpy array is empty, you need to have at least one element')
        is_first_element = False
        while not is_first_element:
            try:
                first_element = first_element[0]
            except:
                is_first_element = True
        data_type = type(first_element.item())

        if data_type == bool:
            # Numpy -> N2D2 doesn't work for bool because there is no buffer protocol for it.
            n2d2_tensor._dataType = data_type
            tmp_tensor = n2d2_tensor._tensor_generators[int](np_array)
            shape = [d for d in reversed(tmp_tensor.dims())]
            n2d2_tensor._tensor = n2d2_tensor._tensor_generators[data_type](shape)
            for i, value in enumerate(tmp_tensor):
                n2d2_tensor._tensor[i] = value
            del tmp_tensor
        else:
            n2d2_tensor._dataType = data_type
            n2d2_tensor._tensor = n2d2_tensor._tensor_generators[data_type](np_array)

        return n2d2_tensor

    @classmethod
    def from_N2D2(cls, N2D2_Tensor):
        if not isinstance(N2D2_Tensor, N2D2.BaseTensor):
            raise error_handler.WrongInputType("N2D2_Tensor", str(type(N2D2_Tensor), [str(N2D2.BaseTensor)]))
        n2d2_tensor = cls([])
        n2d2_tensor._tensor = N2D2_Tensor
        n2d2_tensor._dataType = hard_coded_type[N2D2_Tensor.getTypeName()]
        n2d2_tensor.is_cuda = "CudaTensor" in str(type(N2D2_Tensor))
        return n2d2_tensor

    def __setitem__(self, index, value):
        """
        Set an element of the tensor.
        To select the element to modify you can use :
            - the coordinate of the element;
            - the index of the flatten tensor;
            - a slice index of the flatten tensor. 
        """
        if isinstance(index, tuple) or isinstance(index, list):
            self._tensor[self._get_index(index)] = value
        elif isinstance(index, int) or isinstance(index, float):
            # Force conversion to int if it's a float
            self._tensor[int(index)] = value
        elif isinstance(index, slice):
            self._tensor[index] = value
        else:
            raise error_handler.WrongInputType("index", type(index), [str(list), str(tuple), str(float), str(int), str(slice)])

    def __getitem__(self, index):
        """
        Get an element of the tensor.
        To select the element to get you can use :
            - the coordinate of the element;
            - the index of the flatten tensor.
        """
        value = None
        if isinstance(index, tuple) or isinstance(index, list):
            value = self._tensor[self._get_index(index)]
        elif isinstance(index, int) or isinstance(index, float):
            value = self._tensor[int(index)]
        else:
            raise error_handler.WrongInputType("index", type(index), [str(list), str(tuple), str(float), str(int)])
        return value
        
    def __len__(self):
        return len(self._tensor)

    def __iter__(self):
        return self._tensor.__iter__()

    def __contains__(self, value):
        return self._tensor.__contains__(value)

    def __eq__(self, other_tensor):
        """
        Magic method called when comparing two tensors with the "==" operator.
        """
        if not isinstance(other_tensor, Tensor):
            raise TypeError("You can only compare tensor with each other.")
        # Quick initialization of is_equal by checking the tensors have the same dimensions
        is_equal = (self.dims() == other_tensor.dims())
        cpt = 0
        while is_equal and cpt < len(other_tensor):
            is_equal = (self._tensor[cpt] == other_tensor[cpt])
            cpt += 1
        return is_equal

    def __str__(self):
        return str(self._tensor)


class GraphTensor(Tensor):
    def __init__(self, tensor, cell=None):
    
        """
        :param tensor: The streamed tensor
        :type tensor: :py:class:`n2d2.tensor.Tensor`
        :param cell: The cell that output the tensor object. If None, the object will create a :py:class:`n2d2.provider.TensorPlaceholder, default= None
        :type cell: :py:class:`n2d2.cell.Cell` or, if input :py:class:`n2d2.provider.Provider`, optional
        """
        self._tensor = tensor.N2D2()
        self.tensor = tensor
        if cell is None:
            self.cell = TensorPlaceholder(tensor)
        else:
            self.cell = cell

    # def dims(self):
    #     return self.tensor.dims()

    def get_deepnet(self):
        return self.cell.get_deepnet()

    def back_propagate(self):
        # TODO: Add leaf node check
        self.cell.get_deepnet().back_propagate()
    
    def update(self):
        # TODO: Add leaf node check
        self.cell.get_deepnet().update()

