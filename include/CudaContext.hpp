/*
    (C) Copyright 2015 CEA LIST. All Rights Reserved.
    Contributor(s): Olivier BICHLER (olivier.bichler@cea.fr)
                    Victor GACOIN

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

#ifndef N2D2_CUDA_CONTEXT_H
#define N2D2_CUDA_CONTEXT_H

#include <iostream>

#include "CudaUtils.hpp"

#include "third_party/half.hpp"

namespace N2D2 {
class CudaContext {
public:
    static void setDevice(int device = -1)
    {
        static int prevDevice = 0;

        if (device >= 0)
            prevDevice = device;
        else
            device = prevDevice;

        CHECK_CUDA_STATUS(cudaSetDevice(device));
    }

    static const cudaDeviceProp& getDeviceProp()
    {
        static std::vector<cudaDeviceProp> deviceProp;
        static std::vector<bool> init;

        if (deviceProp.empty()) {
#pragma omp critical(CudaContext__getDeviceProp)
            if (deviceProp.empty()) {
                int count = 1;
                CHECK_CUDA_STATUS(cudaGetDeviceCount(&count));

                deviceProp.resize(count);
                init.resize(count, false);
            }
        }

        int dev;
        CHECK_CUDA_STATUS(cudaGetDevice(&dev));

        if (!init[dev]) {
            CHECK_CUDA_STATUS(cudaGetDeviceProperties(&deviceProp[dev], dev));
            init[dev] = true;
        }

        return deviceProp[dev];
    }

    // Declare cublas handle
    static cublasHandle_t& cublasHandle()
    {
        static std::vector<cublasHandle_t> cublas_h;

        if (cublas_h.empty()) {
#pragma omp critical(CudaContext__cublasHandle)
            if (cublas_h.empty()) {
                int count = 1;
                CHECK_CUDA_STATUS(cudaGetDeviceCount(&count));

                cublas_h.resize(count, NULL);
            }
        }

        int dev;
        CHECK_CUDA_STATUS(cudaGetDevice(&dev));

        if (cublas_h[dev] == NULL) {
            CHECK_CUBLAS_STATUS(cublasCreate(&cublas_h[dev]));
            std::cout << "CUBLAS initialized on device #" << dev << std::endl;
        }

        return cublas_h[dev];
    }

    // Declare cudnn handle
    static cudnnHandle_t& cudnnHandle()
    {
        static std::vector<cudnnHandle_t> cudnn_h;

        if (cudnn_h.empty()) {
#pragma omp critical(CudaContext__cudnnHandle)
            if (cudnn_h.empty()) {
                int count = 1;
                CHECK_CUDA_STATUS(cudaGetDeviceCount(&count));

                cudnn_h.resize(count, NULL);
            }
        }

        int dev;
        CHECK_CUDA_STATUS(cudaGetDevice(&dev));

        if (cudnn_h[dev] == NULL) {
            CHECK_CUDNN_STATUS(cudnnCreate(&cudnn_h[dev]));
            std::cout << "CUDNN initialized on device #" << dev << std::endl;
        }

        return cudnn_h[dev];
    }

    template <class T>
    struct data_type {
        static const cudnnDataType_t value = CUDNN_DATA_FLOAT;
                                            // Dummy value by default
    };
};
}

namespace N2D2 {
    template <>
    struct CudaContext::data_type<half_float::half> {
        static const cudnnDataType_t value = CUDNN_DATA_HALF;
    };

    template <>
    struct CudaContext::data_type<float> {
        static const cudnnDataType_t value = CUDNN_DATA_FLOAT;
    };

    template <>
    struct CudaContext::data_type<double> {
        static const cudnnDataType_t value = CUDNN_DATA_DOUBLE;
    };
}

#endif // N2D2_CUDA_CONTEXT_H
