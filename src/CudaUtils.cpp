/*
    (C) Copyright 2016 CEA LIST. All Rights Reserved.
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

#ifdef CUDA

#include "CudaUtils.hpp"

const char* N2D2::Cuda::cublasGetErrorString(cublasStatus_t error)
{
    switch (error) {
    case CUBLAS_STATUS_SUCCESS:
        return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED:
        return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED:
        return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE:
        return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH:
        return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR:
        return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED:
        return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR:
        return "CUBLAS_STATUS_INTERNAL_ERROR";
    case CUBLAS_STATUS_NOT_SUPPORTED:
        return "CUBLAS_STATUS_NOT_SUPPORTED";
    case CUBLAS_STATUS_LICENSE_ERROR:
        return "CUBLAS_STATUS_LICENSE_ERROR";
    }

    return "<unknown>";
}

void N2D2::Cuda::setMultiDevicePeerAccess(unsigned int size, unsigned int* devices)
{
    for (unsigned int i = 0; i < size; ++i) {
        for (unsigned int j = 0; j < size; ++j) {
            if (i != j) {
                int canAccessPeer = 0;
                CHECK_CUDA_STATUS(cudaDeviceCanAccessPeer(&canAccessPeer,
                                            devices[j], devices[i]));                     
                if (canAccessPeer) {
                    CHECK_CUDA_STATUS(cudaSetDevice(devices[j]));
                    const cudaError_t e = cudaDeviceEnablePeerAccess(devices[i], 0);
                    if (e == cudaErrorPeerAccessAlreadyEnabled) {
                        std::cout << "Peer access already enabled between ";
                        std::cout << "device " << devices[j] << " and ";
                        std::cout << "device " << devices[i] << std::endl;
                    } else {
                        CHECK_CUDA_STATUS(e);
                    }
                }
            }
        }
    }
}

#endif
