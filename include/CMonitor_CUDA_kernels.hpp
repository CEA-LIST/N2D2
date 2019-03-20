/*
    (C) Copyright 2016 CEA LIST. All Rights Reserved.
    Contributor(s): Johannes Thiele (johannes.thiele@cea.fr)
                    Olivier BICHLER (olivier.bichler@cea.fr)

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

#ifndef N2D2_CMONITOR_CUDA_KERNELS_H
#define N2D2_CMONITOR_CUDA_KERNELS_H

#include <curand_kernel.h>
#include "CudaUtils.hpp"


namespace N2D2 {

void cudaUpdateActivity(int * inputs,
                        char * activity,
                        unsigned int * firingRate,
                        unsigned int * exampleFiringRate,
                        int * totalOutput,
                        unsigned long long int * firstEventTime,
                        unsigned long long int * lastEventTime,
                        unsigned int inputsDimX,
                        unsigned int inputsDimY,
                        unsigned int inputsDimZ,
                        unsigned long long int timestamp,
                        unsigned int batchSize,
                        unsigned int maxNbThreads,
                        unsigned int warpSize);

void cudaUpdateFiringRate(unsigned int * firingRate,
                        unsigned int * totalFiringRate,
                        unsigned int inputsDimX,
                        unsigned int inputsDimY,
                        unsigned int inputsDimZ,
                        unsigned int batchSize,
                        unsigned int maxNbThreads,
                        unsigned int warpSize);

void cudaUpdateFiringRate(int * firingRate,
                        int * totalFiringRate,
                        unsigned int inputsDimX,
                        unsigned int inputsDimY,
                        unsigned int inputsDimZ,
                        unsigned int batchSize,
                        unsigned int maxNbThreads,
                        unsigned int warpSize);

void cudaUpdateBatchFiringRate(unsigned int * firingRate,
                                    unsigned int * batchFiringRate,
                                    unsigned int inputsDimX,
                                    unsigned int inputsDimY,
                                    unsigned int inputsDimZ,
                                    unsigned int batchSize,
                                    unsigned int maxNbThreads,
                                    unsigned int warpSize);

void cudaUpdateMostActive(unsigned int * exampleFiringRate,
                            unsigned int * mostActiveId,
                            unsigned int inputsDimX,
                            unsigned int inputsDimY,
                            unsigned int inputsDimZ,
                            unsigned int batchSize,
                            unsigned int maxNbThreads,
                            unsigned int warpSize);

/*
void cudaUpdateMostActive(unsigned int * exampleIds,
                            unsigned int * exampleFiringRate,
                            unsigned int * mostActiveId,
                            unsigned int inputsDimX,
                            unsigned int inputsDimY,
                            unsigned int inputsDimZ,
                            unsigned int batchSize,
                            unsigned int maxNbThreads,
                            unsigned int warpSize);*/

}

#endif //N2D2_CMONITOR_CUDA_KERNELS_H
