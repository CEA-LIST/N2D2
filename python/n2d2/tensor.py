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
    
    _tensor_generators = {
        float: N2D2.Tensor_float,
        #'half': N2D2.Tensor_half,
        #'double': N2D2.Tensor_double,
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
            
    """
    Add basic methods like size based on N2D2::Tensor class
    """
    
    def N2D2(self):
        return self._tensor
        
    def dims(self):
        return self._tensor.dims()
    
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

    def __str__(self):
        return str(self._tensor)
    