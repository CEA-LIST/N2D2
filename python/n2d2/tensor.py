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
        'float': N2D2.Tensor_float,
        #'half': N2D2.Tensor_half,
        #'double': N2D2.Tensor_double,
        'int': N2D2.Tensor_int,
        #'float': N2D2.Cuda_Tensor_float,
        #'half': N2D2.Cuda_Tensor_half,
        #'double': N2D2.Cuda_Tensor_double
    }
    
    def __init__(self, dims, DefaultModel='', DefaultDataType='float'):
        self._model_key = DefaultModel + '<' + DefaultDataType + '>'
        self._tensor = self._tensor_generators[self._model_key](dims)
        #else:
        #    raise ValueError("Unrecognized Tensor datatype " + str(dtype))
            
    """
    Add basic methods like size based on N2D2::Tensor class
    """
    
    def N2D2(self):
        return self._tensor
        
    def dims(self):
        return self._tensor.dims()
        
    #def __str__(self):
    #    return self.tensor # For pytorch tensor
    