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
    do {                                                                       \
        const cudnnStatus_t e = (status);                                      \
        if (e != CUDNN_STATUS_SUCCESS) {                                       \
            std::stringstream error;                                           \
            error << "CUDNN failure: " << cudnnGetErrorString(e) << " ("       \
                  << (int)e << ") in " << __FILE__ << ':' << __LINE__;         \
            std::cerr << error.str() << std::endl;                             \
            cudaDeviceReset();                                                 \
            throw std::runtime_error(error.str());                             \
        }                                                                      \
    } while(0)

#define CHECK_CUDA_STATUS(status)                                              \
    do {                                                                       \
        const cudaError_t e = (status);                                        \
        if ((e) != cudaSuccess) {                                              \
            std::stringstream error;                                           \
            error << "Cuda failure: " << cudaGetErrorString(e) << " ("         \
                  << (int)e << ") in " << __FILE__ << ':' << __LINE__;         \
            std::cerr << error.str() << std::endl;                             \
            cudaDeviceReset();                                                 \
            throw std::runtime_error(error.str());                             \
        }                                                                      \
    } while(0)

#define CHECK_CUBLAS_STATUS(status)                                            \
    do {                                                                       \
        const cublasStatus_t e = (status);                                     \
        if (e != CUBLAS_STATUS_SUCCESS) {                                      \
            std::stringstream error;                                           \
            error << "Cublas failure: "                                        \
                  << N2D2::Cuda::cublasGetErrorString(e) << " ("               \
                  << (int)e << ") in " << __FILE__ << ':' << __LINE__;         \
            std::cerr << error.str() << std::endl;                             \
            cudaDeviceReset();                                                 \
            throw std::runtime_error(error.str());                             \
        }                                                                      \
    } while(0)

namespace N2D2 {
namespace Cuda {
    const char* cublasGetErrorString(cublasStatus_t error);

    template <class T> void printDeviceVector(unsigned int size, T* devVec)
    {
        std::unique_ptr<T[]> vec(new T[size]);
        CHECK_CUDA_STATUS(cudaDeviceSynchronize());
        CHECK_CUDA_STATUS(cudaMemcpy(vec.get(),
                                     devVec,
                                     size * sizeof(T),
                                     cudaMemcpyDeviceToHost));

        for (unsigned int i = 0; i < size; i++)
            std::cout << vec[i] << " ";

        std::cout << std::endl;
    }

    // CuDNN scaling parameters are typically "alpha" and "beta".
    // Their type must be "float" for HALF and FLOAT (default template)
    // and "double" for DOUBLE (specialized template)
    template <class T>
    struct cudnn_scaling_type {
        typedef float type;
    };

    template <>
    struct cudnn_scaling_type<double> {
        typedef double type;
    };
}
}

#endif // N2D2_CUDA_H
