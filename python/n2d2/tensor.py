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

class Tensor():
    
    # TODO : deal with CUDA tensor
    _tensor_generators = {
        float: N2D2.Tensor_float,
        int: N2D2.Tensor_int,
        bool: N2D2.Tensor_bool,
        #'float': N2D2.Cuda_Tensor_float,
        #'half': N2D2.Cuda_Tensor_half,
        #'double': N2D2.Cuda_Tensor_double
    }
    
    def __init__(self, dims, DefaultDataType=float):
        if DefaultDataType in self._tensor_generators:
            self._tensor = self._tensor_generators[DefaultDataType](dims)
        else:
           raise TypeError("Unrecognized Tensor datatype " + str(DefaultDataType))
        self._dataType = DefaultDataType
    """
    Add basic methods like size based on N2D2::Tensor class
    """
    
    def N2D2(self):
        return self._tensor
        
    def dims(self):
        return self._tensor.dims()
    
    def dataType(self):
        return self._dataType

    def getindex(self, coord):
        """
        From the coordinate returns the 1D index of an element in the tensor.
        :param coord: Tuple of the coordinate
        :type coord: tuple
        """
        dims = self.dims()
        if len(dims) != len(coord):
            raise ValueError(str(len(coord)) + "D array does not match " + str(len(dims)) + "D tensor.") 
        idx = 0
        for i in reversed(range(len(dims))):
            if i != 0:
                idx = dims[i] * (coord[i] + idx)
            else:
                idx += coord[i]
        return idx
        
    def resize(self, new_dims):
        self._tensor.resize(new_dims)

    def __setitem__(self, index, value):
        """
        Set an element of the tensor.
        To select the element to modify you can use :
            - the coordinate of the element;
            - the index of the flatten tensor;
            - a slice index of the flatten tensor. 
        """
        if isinstance(index, tuple) or isinstance(index, list):
            self._tensor[self.getindex(index)] = value
        elif isinstance(index, int) or isinstance(index, float) or isinstance(index, slice):
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
            value = self._tensor[self.getindex(index)]
        elif isinstance(index, int) or isinstance(index, float):
            value =self._tensor[index] = value
        else:
            raise TypeError("Unsupported index type :" + str(type(index)))
        return value

    def fromNumpy(self, np_array):
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
        if isinstance(np_array, ndarray):
            if np_array.dtype is dtype("bool"):
                # N2D2.Tensor doesn't support transformation from numpy.array with a boolean data type
                # So we create a temporary tensor with a int datatype and we manually copy it into a tensor with a boolean data type.  
                tmp_tensor = self._tensor_generators[int](np_array)
                self._tensor = self._tensor_generators[bool](tmp_tensor.dims())
                for cpt in range(len(tmp_tensor)):
                    self._tensor[cpt] = tmp_tensor[cpt]
                self._dataType = bool
            elif np_array.dtype is dtype("int"):
                self._dataType = int
                self._tensor = self._tensor_generators[int](np_array)
            elif np_array.dtype is dtype("float"):
                self._dataType = float
                self._tensor = self._tensor_generators[float](np_array)
            else:
                raise TypeError("The numpy array data type is unsupported : " + str(np_array.dtype))
        else:
            raise TypeError("arg 1 must be a numpy array.")
        
    def __len__(self):
        return len(self._tensor)

    def __iter__(self):
        return self._tensor.__iter__()

    def __contains__(self, value):
        return self._tensor.__contains__(value)

    def __str__(self):
        return str(self._tensor)
    