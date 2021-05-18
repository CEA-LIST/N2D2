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

#include <cmath>
#include <cstdlib>
#include <dirent.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <vector>

#ifndef COMMON_CUDA_H
#define COMMON_CUDA_H

#define EXIT_WAIVED 0
#define DEVICE_ID 0

#define FatalError(s)                                                          \
    {                                                                          \
        std::stringstream _where, _message;                                    \
        _where << __FILE__ << ':' << __LINE__;                                 \
        _message << std::string(s) + "\n" << __FILE__ << ':' << __LINE__;      \
        std::cerr << _message.str() << "\nAborting...\n";                      \
        cudaDeviceReset();                                                     \
        exit(EXIT_FAILURE);                                                    \
    }

#define checkCudaErrors(status)                                                \
    {                                                                          \
        std::stringstream _error;                                              \
        if (status != 0) {                                                     \
            _error << "Cuda failure: " << cudaGetErrorString(status);          \
            FatalError(_error.str());                                          \
        }                                                                      \
    }

#define checkCudaKernelsErrors()                                               \
    {                                                                          \
        cudaThreadSynchronize();                                               \
        cudaError_t error = cudaGetLastError();                                \
        if (error != cudaSuccess) {                                            \
            printf("CUDA error: %s\n", cudaGetErrorString(error));             \
            exit(-1);                                                          \
        }                                                                      \
    }

#endif // COMMON_CUDA_H
