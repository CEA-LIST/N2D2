/*
    (C) Copyright 2016 CEA LIST. All Rights Reserved.
    Contributor(s): Johannes THIELE (johannes.thiele@cea.fr)
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

#ifndef N2D2_CENVIRONMENT_CUDA_KERNELS_H
#define N2D2_CENVIRONMENT_CUDA_KERNELS_H

#include <curand_kernel.h>

namespace N2D2 {

void cudaNoConversion(float * data,
                    float * tickOutputsTraces,
                    float * tickOutputsTracesLearning,
                    float scaling,
                    unsigned int inputDimX,
                    unsigned int inputDimY,
                    unsigned int inputDimZ,
                    unsigned int nbBatches,
                    unsigned int maxNbThreads);

void cudaGenerateInitialSpikes(float * data,
                        unsigned long long int * nextEventTime,
                        int * nextEventType,
                        unsigned int inputDimX,
                        unsigned int inputDimY,
                        unsigned int inputDimZ,
                        unsigned long long int start,
                        unsigned long long int stop,
                        float discardedLateStimuli,
                        unsigned int stimulusType,
                        unsigned long long int periodMeanMin,
                        unsigned long long int periodMeanMax,
                        float periodRelStdDev,
                        unsigned long long int periodMin,
                        float mMaxFrequency,
                        unsigned int nbBatches,
                        curandState * state);

void cudaGenerateSpikes(float * data,
                        int * tickData,
                        int * tickOutputs,
                        unsigned long long int * nextEventTime,
                        int * nextEventType,
                        unsigned int inputDimX,
                        unsigned int inputDimY,
                        unsigned int inputDimZ,
                        unsigned long long int timestamp,
                        unsigned long long int start,
                        unsigned long long int stop,
                        float discardedLateStimuli,
                        unsigned int stimulusType,
                        unsigned long long int periodMeanMin,
                        unsigned long long int periodMeanMax,
                        float periodRelStdDev,
                        unsigned long long int periodMin,
                        float mMaxFrequency,
                        unsigned int nbSubStimuli,
                        unsigned int subStimulus,
                        unsigned int nbBatches,
                        curandState * state);

void cudaSetupRng(curandState * state,
                    unsigned int seed,
                    unsigned int nbBatches);

}

#endif //N2D2_CENVIRONMENT_CUDA_KERNELS_H
