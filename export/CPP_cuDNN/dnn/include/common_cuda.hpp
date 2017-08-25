/*
    (C) Copyright 2015 CEA LIST. All Rights Reserved.
    Contributor(s): Olivier BICHLER (olivier.bichler@cea.fr)
                    David BRIAND (david.briand@cea.fr)

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

#ifndef COMMON_CUDA_H
#define COMMON_CUDA_H

#define EXIT_WAIVED 0

#define FatalError(s)                                                          \
    {                                                                          \
        std::stringstream _where, _message;                                    \
        _where << __FILE__ << ':' << __LINE__;                                 \
        _message << std::string(s) + "\n" << __FILE__ << ':' << __LINE__;      \
        std::cerr << _message.str() << "\nAborting...\n";                      \
        cudaDeviceReset();                                                     \
        exit(EXIT_FAILURE);                                                    \
    }

#define CHECK_CUDNN_STATUS(status)                                             \
    {                                                                          \
        std::stringstream _error;                                              \
        if (status != CUDNN_STATUS_SUCCESS) {                                  \
            _error << "CUDNN failure: " << cudnnGetErrorString(status);        \
            FatalError(_error.str());                                          \
        }                                                                      \
    }

#define CHECK_CUDA_STATUS(status)                                              \
    {                                                                          \
        std::stringstream _error;                                              \
        if (status != 0) {                                                     \
            _error << "Cuda failure: " << cudaGetErrorString(status);          \
            FatalError(_error.str());                                          \
        }                                                                      \
    }

#define CHECK_CUBLAS_STATUS(status)                                            \
    {                                                                          \
        std::stringstream _error;                                              \
        if (status != CUBLAS_STATUS_SUCCESS) {                                 \
            switch (status) {                                                  \
            case CUBLAS_STATUS_NOT_INITIALIZED:                                \
                _error << "Cublas failure: CUBLAS_STATUS_NOT_INITIALIZED";     \
            case CUBLAS_STATUS_ALLOC_FAILED:                                   \
                _error << "Cublas failure: CUBLAS_STATUS_ALLOC_FAILED";        \
            case CUBLAS_STATUS_INVALID_VALUE:                                  \
                _error << "Cublas failure: CUBLAS_STATUS_INVALID_VALUE";       \
            case CUBLAS_STATUS_ARCH_MISMATCH:                                  \
                _error << "Cublas failure: CUBLAS_STATUS_ARCH_MISMATCH";       \
            case CUBLAS_STATUS_MAPPING_ERROR:                                  \
                _error << "Cublas failure: CUBLAS_STATUS_MAPPING_ERROR";       \
            case CUBLAS_STATUS_EXECUTION_FAILED:                               \
                _error << "Cublas failure: CUBLAS_STATUS_EXECUTION_FAILED";    \
            case CUBLAS_STATUS_INTERNAL_ERROR:                                 \
                _error << "Cublas failure: CUBLAS_STATUS_INTERNAL_ERROR";      \
            case CUBLAS_STATUS_NOT_SUPPORTED:                                  \
                _error << "Cublas failure: CUBLAS_STATUS_NOT_SUPPORTED";       \
            case CUBLAS_STATUS_LICENSE_ERROR:                                  \
                _error << "Cublas failure: CUBLAS_STATUS_LICENSE_ERROR";       \
            default:                                                           \
                _error << "Cublas failure: Unknown Cublas Error";              \
            }                                                                  \
            FatalError(_error.str());                                          \
        }                                                                      \
    }

class CudaContext {
public:
    static void setDevice(int device)
    {
        CHECK_CUDA_STATUS(cudaSetDevice(device));
    }

    // Declare cublas handle
    static cublasHandle_t& cublasHandle()
    {
        static cublasHandle_t cublas_h = NULL;

        if (!cublas_h) {
            CHECK_CUBLAS_STATUS(cublasCreate(&cublas_h));
            std::cout << "CUBLAS initialized" << std::endl;
        }

        return cublas_h;
    }

    // Declare cudnn handle
    static cudnnHandle_t& cudnnHandle()
    {
        static cudnnHandle_t cudnn_h = NULL;

        if (!cudnn_h) {
            CHECK_CUDNN_STATUS(cudnnCreate(&cudnn_h));
            std::cout << "CUDNN initialized" << std::endl;
        }

        return cudnn_h;
    }

};
#endif // COMMON_CUDA_H
