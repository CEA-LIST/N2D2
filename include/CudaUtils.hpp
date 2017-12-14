/*
    (C) Copyright 2015 CEA LIST. All Rights Reserved.
    Contributor(s): Olivier BICHLER (olivier.bichler@cea.fr)

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
*/

#ifndef N2D2_CUDA_H
#define N2D2_CUDA_H

#include <string>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>

#include <cublas_v2.h>
#include <cuda.h>
#include <cudnn.h>

#define CHECK_CUDNN_STATUS(status)                                             \
    {                                                                          \
        if ((status) != CUDNN_STATUS_SUCCESS) {                                \
            std::stringstream error;                                           \
            error << "CUDNN failure: " << cudnnGetErrorString(status) << " ("  \
                  << (int)status << ") in " << __FILE__ << ':' << __LINE__;    \
            std::cerr << error.str() << std::endl;                             \
            cudaDeviceReset();                                                 \
            throw std::runtime_error(error.str());                             \
        }                                                                      \
    }

#define CHECK_CUDA_STATUS(status)                                              \
    {                                                                          \
        if ((status) != cudaSuccess) {                                         \
            std::stringstream error;                                           \
            error << "Cuda failure: " << cudaGetErrorString(status) << " ("    \
                  << (int)status << ") in " << __FILE__ << ':' << __LINE__;    \
            std::cerr << error.str() << std::endl;                             \
            cudaDeviceReset();                                                 \
            throw std::runtime_error(error.str());                             \
        }                                                                      \
    }

#define CHECK_CUBLAS_STATUS(status)                                            \
    {                                                                          \
        if ((status) != CUBLAS_STATUS_SUCCESS) {                               \
            std::stringstream error;                                           \
            error << "Cublas failure: "                                        \
                  << N2D2::Cuda::cublasGetErrorString(status) << " ("          \
                  << (int)status << ") in " << __FILE__ << ':' << __LINE__;    \
            std::cerr << error.str() << std::endl;                             \
            cudaDeviceReset();                                                 \
            throw std::runtime_error(error.str());                             \
        }                                                                      \
    }

namespace N2D2 {
namespace Cuda {
    const char* cublasGetErrorString(cublasStatus_t error);

    template <class T> void printDeviceVector(unsigned int size, T* devVec)
    {
        std::unique_ptr<float[]> vec(new T[size]);
        CHECK_CUDA_STATUS(cudaDeviceSynchronize());
        CHECK_CUDA_STATUS(
            cudaMemcpy(vec, devVec, size * sizeof(T), cudaMemcpyDeviceToHost));

        for (unsigned int i = 0; i < size; i++)
            std::cout << vec[i] << " ";

        std::cout << std::endl;
    }
}
}

#endif // N2D2_CUDA_H
