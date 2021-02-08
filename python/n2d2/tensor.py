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
from functools import reduce

class Tensor():
    
    _tensor_generators = {
        float: N2D2.Tensor_float,
        int: N2D2.Tensor_int,
        bool: N2D2.Tensor_bool,
    }
    
    def __init__(self, dims, value=None, DefaultDataType=float):
        # Dimensions convention on N2D2 are reversed from python. 
        dims = [d for d in reversed(dims)]
        if DefaultDataType in self._tensor_generators:
            if not value:
                self._tensor = self._tensor_generators[DefaultDataType](dims)
            else:
                self._tensor = self._tensor_generators[DefaultDataType](dims, value)
        else:
           raise TypeError("Unrecognized Tensor datatype " + str(DefaultDataType))
        self._dataType = DefaultDataType

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
        From the coordinate returns the 1D index of an element in the tensor.
        :param coord: Tuple of the coordinate
        :type coord: tuple
        """
        dims = self.dims()
        if len(dims) != len(coord):
            raise ValueError(str(len(coord)) + "D array does not match " + str(len(dims)) + "D tensor.") 
        for c, d in zip(coord, self.shape()):
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
        From the the 1D index, return the coordinate of an element in the tensor.
        :param index: index of an element
        :type index: int
        """ 
        coord = []
        for i in self.dims():
            coord.append(int(index%i))
            index = index/i
        return [i for i in reversed(coord)]

    def resize(self, new_dims):
        """
        Resize the Tensor to the specified dims. 
        :param new_dims: New dimensions
        :type new_dims: list
        """
        self._tensor.resize([d for d in reversed(new_dims)])

    def copy(self):
        """
        Copy in memory the Tensor object.
        """
        copy = Tensor(self.shape(), DefaultDataType=self.data_type())
        for i in range(len(copy)):
            copy[i] = self._tensor[i]
        return copy

    def from_Tf(tf_tensor):
        """
        We can't call this method when using a custom layer ...
        """
        # really messy function ...
        try:
            import numpy as np
            from tensorflow.compat.v1 import enable_eager_execution
            import tensorflow.keras.backend as K
        except ImportError:
            raise ImportError("Numpy is not installed")
        dims = tf_tensor.shape.as_list()
        data_type = tf_tensor.dtype.name
        # Ugly ?
        if data_type == "int32" or data_type == "int64":
            data_type = int
        elif data_type == "float32" or data_type == "float64":
            data_type = float
        else:
            raise TypeError("Unknown type :", data_type)
        print("Dims :", dims)
        print("dtype :", data_type)

        def recursive_browse(x): 
            # Would be better if we got a flatten list out of this mess ...
            # Super ugly try except condition to check if object is iterable, hasattr(class, '__getitem__') may be a better option 
            try:
                t_browse = []
                for i in range(len(x)):
                    t_browse.append(recursive_browse(x[i]))
                return t_browse
            except:
                t_browse = []
                for i in range(len(x)):
                    # print(K.get_value(x[i]))
                    t_browse.append(K.get_value(x[i]))
                # Need to use the keras method "get_value" to get the value ... doesn't looks like there is a better option.
                return t_browse
        n2d2_tensor = Tensor(dims, DefaultDataType=data_type)
        n2d2_tensor.from_numpy(np.array(recursive_browse(tf_tensor)))
        return(n2d2_tensor)
    
    def to_list(self):
        """
        Convert the tensor to a list object
        """
        def create_empty_list(dim, other_dims):
            result = []
            if other_dims:
                for _ in range(dim):
                    result.append(create_empty_list(other_dims[-1], other_dims[:-1]))
            else:
                result = [0] * dim
            return result

        dim = self.dims()
        empty_list = create_empty_list(dim[-1], dim[:-1])


        for i in range(len(self._tensor)):
            coord = self._get_coord(i)
            id = empty_list
            for c in coord[:-1]:
                id = id[c]
            id[coord[-1]] = self._tensor[i]
        return empty_list

    def to_numpy(self):
        """
        Create a numpy array equivalent to the tensor.
        """
        try:
            from numpy import array 
        except ImportError:
            raise ImportError("Numpy is not installed")
        return array(self.to_list())

    def from_numpy(self, np_array):
        """
        Convert a numpy array into a tensor.
        Auto convert data type
        :param np_array: A numpy array to convert to a tensor.
        :type np_array: :py:class:`numpy.array`
        """
        try:
            from numpy import ndarray, dtype 
        except ImportError:
            raise ImportError("Numpy is not installed")
        if not isinstance(np_array, ndarray):
            raise TypeError("arg 1 must be a numpy array.")

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
            raise TypeError("Unsupported index type :" + str(type(index)))

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
            raise TypeError("Unsupported index type :" + str(type(index)))
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
    

class CUDA_Tensor(Tensor):
    _tensor_generators = {
        float: N2D2.CudaTensor_float,
        int: N2D2.CudaTensor_int,
    }
    
    def __init__(self, dims, value=None, DefaultDataType=float):
        super().__init__(dims, value, DefaultDataType)

    def copy(self):
        """
        Copy in memory the CUDA_Tensor object.
        """
        copy = CUDA_Tensor(self.shape(), DefaultDataType=self.data_type())
        for i in range(len(copy)):
            copy[i] = self._tensor[i]
        return copy
