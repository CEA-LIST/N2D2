# distutils: language = c++

from libc.stdint cimport uint8_t, int8_t, uint32_t, int32_t

import numpy as np

cdef extern from "Network.hpp" namespace "N2D2_Export":
    cdef cppclass Network:
        void propagate[Input_T, Output_T](const Input_T* inputs, Output_T* outputs) const

cdef extern from "dnn/src/forward.cpp":
    pass

cdef extern from "env.hpp":
    cdef int ENV_SIZE_X
    cdef int ENV_SIZE_Y
    cdef int ENV_NB_OUTPUTS
    cdef int ENV_OUTPUTS_SIZE


def model_forward(inputs):

    if not isinstance(inputs, np.ndarray):
        raise Exception("Inputs should be a numpy array")

    if len(inputs.shape) > 1:
        raise Exception("Inputs must have a shape of 1 (array1D)")

    if inputs.size != ENV_OUTPUTS_SIZE:
        raise Exception(f"The network only accepts images of {ENV_NB_OUTPUTS}x{ENV_SIZE_X}x{ENV_SIZE_Y} ({ENV_OUTPUTS_SIZE})")


    cdef Network n
    cdef int32_t outputs = 0

    cdef uint8_t[::1] input_view = inputs

    n.propagate[uint8_t, int32_t](&input_view[0], &outputs)
    
    return outputs