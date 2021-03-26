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
from functools import reduce



class Tensor():
    
    _tensor_generators = {
        float: N2D2.Tensor_float,
        int: N2D2.Tensor_int,
        bool: N2D2.Tensor_bool,
    }
    
    def __init__(self, dims, value=None, defaultDataType=float, N2D2_tensor=None):
        """
        :param dims: Dimensions of the :py:class:`n2d2.Tensor` object. (Numpy convention)
        :type dims: list
        :param value: A value to fill the :py:class:`n2d2.Tensor` object.
        :type value: Must be coherent with **defaultDataType**
        :param defaultDataType: Type of the data stocked by the tensor
        :type defaultDataType: type (optional, default=float)
        :param N2D2_tensor: If not none this is the tensor that will be wrapped by :py:class:`n2d2.Tensor`.
        :type N2D2_tensor: :py:class:`N2D2.BaseTensor` (optional, default=None)
        """
        # Dimensions convention on N2D2 are reversed from python. 
        if not N2D2_tensor:
            if isinstance(dims, list):
                dims = [d for d in reversed(dims)]
            else:
                raise TypeError("Dims should be of type list got " + type(dims) + " instead")
            if value and not isinstance(value, defaultDataType):
                raise TypeError("You want to fill the tensor with " + type(value) + " but " + str(defaultDataType) + " is the defaultDataType.")
            if defaultDataType in self._tensor_generators:
                if not value:
                    self._tensor = self._tensor_generators[defaultDataType](dims)
                else:
                    self._tensor = self._tensor_generators[defaultDataType](dims, value)
            else:
                raise TypeError("Unrecognized Tensor datatype " + str(defaultDataType))
        else:
            # TODO : We may want to have a stricter check here, if a tensor is CUDA for example it's not filtered
            if not isinstance(N2D2_tensor, N2D2.BaseTensor):
                # TODO : Change this error message !
                raise TypeError("N2D2_tensor should be of type N2D2.Tensor got " + str(type(N2D2_tensor)) + " instead")
            self._tensor = N2D2_tensor
        self._dataType = defaultDataType
        self.is_cuda = False 

    def from_N2D2(self, N2D2_Tensor):
        self._tensor = N2D2_Tensor

    def N2D2(self):
        return self._tensor
        
    def dims(self):
        """
        Return dimensions with N2D2 convention 
        """
        return self._tensor.dims()

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
        copy = Tensor(self.shape(), defaultDataType=self.data_type())
        for i in range(len(copy)):
            copy[i] = self._tensor[i]
        return copy

    # Those method doesn't really work as intended

    # def dimX(self):
    #     return self._tensor.dimX()

    # def dimY(self):
    #     return self._tensor.dimY()

    # def dimZ(self):
    #     return self._tensor.dimZ()

    # def dimB(self):
    #     return self._tensor.dimB()

    def to_numpy(self):
        """
        Create a numpy array equivalent to the tensor.
        """
        try:
            from numpy import array 
        except ImportError:
            raise ImportError("Numpy is not installed")
        return array(self.N2D2()) 

    def from_numpy(self, np_array):
        """
        :param np_array: A numpy array to convert to a tensor.
        :type np_array: :py:class:`numpy.array`
        Convert a numpy array into a tensor.
        Auto convert data type
        """
        try:
            from numpy import ndarray, dtype 
        except ImportError:
            raise ImportError("Numpy is not installed")
        if not isinstance(np_array, ndarray):
            n2d2.error_handler.wrong_input_type("np_array", type(np_array), ["numpy.array"])


        np_array = np_array.reshape([d for d in reversed(np_array.shape)]) 

        if np_array.dtype is dtype("bool"):
            # N2D2.Tensor doesn't support transformation from numpy.array with a boolean data type
            # So we create a temporary tensor with a int datatype and we manually copy it into a tensor with a boolean data type.  
            tmp_tensor = self._tensor_generators[int](np_array)
            self._tensor = self._tensor_generators[bool](tmp_tensor.dims())
            for cpt in range(len(tmp_tensor)):
                self._tensor[cpt] = tmp_tensor[cpt]
            self._dataType = bool
        # TODO : Convert to better data type, ugly for proof of concept ! 
        # https://numpy.org/doc/stable/user/basics.types.html
        elif np_array.dtype is dtype("int") or np_array.dtype is dtype("int32") or np_array.dtype is dtype("int64") :
            self._dataType = int
            self._tensor = self._tensor_generators[int](np_array)
        elif np_array.dtype is dtype("float") or np_array.dtype is dtype("float32") or np_array.dtype is dtype("float64"):
            self._dataType = float
            self._tensor = self._tensor_generators[float](np_array)
        else:
            raise TypeError("The numpy array data type is unsupported : " + str(np_array.dtype))

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
            n2d2.error_handler.wrong_input_type("index", type(index), [str(list), str(tuple), str(float), str(int), str(slice)])

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
            n2d2.error_handler.wrong_input_type("index", type(index), [str(list), str(tuple), str(float), str(int)])
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
    

class CudaTensor(Tensor):
    _tensor_generators = {
        float: N2D2.CudaTensor_float,
        int: N2D2.CudaTensor_int,
    }
    
    def __init__(self, dims, value=None, defaultDataType=float, N2D2_tensor=None):
        super().__init__(dims, value, defaultDataType, N2D2_tensor)
        if value:
            # TODO : a buyg cause the value argument to be ignored for CUDA tensor :
            # example : N2D2.CudaTensor_int([2, 2], value=int(5.0))

            self._tensor[0:] = value
        self.is_cuda = True 


    def copy(self):
        """
        Copy in memory the CudaTensor object.
        """
        copy = CudaTensor(self.shape(), defaultDataType=self.data_type())
        for i in range(len(copy)):
            copy[i] = self._tensor[i]
        return copy


class GraphTensor:
    def __init__(self, tensor, cell):
        self.tensor = tensor
        self.cell = cell

    def dims(self):
        return self.tensor.dims()

    def get_deepnet(self):
        return self.cell.get_deepnet()

    def back_propagate(self):
        # TODO: Add leaf node check
        self.cell.get_deepnet().back_propagate()



