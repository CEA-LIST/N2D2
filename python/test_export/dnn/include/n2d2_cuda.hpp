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

#ifndef N2D2_EXPORTC_DEEPNET_CUDA_H
#define N2D2_EXPORTC_DEEPNET_CUDA_H

#define N2D2_SECTION_ATTRIBUTE(sec)

#include <algorithm> // std::sort
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

#include "common_cuda.hpp"
#include "kernels.hpp"
#include "../../include/typedefs.h"
#include "../../include/utils.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <vector_types.h>

struct cudaProfiling {
    std::string name;
    float processTime;
    unsigned int bytesRead;
    unsigned int bytesWritten;
};

struct cudaHandleStruct {
    std::vector<int> devices;
    std::vector<std::string> kernels;
    std::vector<float> events;
    bool isProfiled = false;
    std::vector<cudaProfiling> profiling;
    bool populated;
};
extern struct cudaHandleStruct cudaHandles;

void set_profiling();

void report_per_layer_profiling(unsigned int nbIter);

static unsigned int nextDivisor(unsigned int target, unsigned int value)
{
    unsigned int v = value;
    while (target % v != 0)
        ++v;
    return v;
}

static unsigned int prevDivisor(unsigned int target, unsigned int value)
{
    unsigned int v = value;
    while (target % v != 0)
        --v;
    return v;
}

template<typename T>
inline void saveOutputs(const T* outputs_cuda, std::ostream& ostream, 
                        std::size_t nbChannels, std::size_t height, std::size_t width)
{
    std::vector<T> tmp(nbChannels*height*width);
    checkCudaErrors(cudaMemcpy((void*) tmp.data(), (const void*) outputs_cuda, tmp.size()*sizeof(T), cudaMemcpyDeviceToHost));

    std::size_t i = 0;
    for(std::size_t ch = 0; ch < nbChannels; ch++) {
        for(std::size_t h = 0; h < height; h++) {
            for(std::size_t w = 0; w < width; w++) {
                ostream.operator<<(tmp[i]);
                ostream << ", ";
                i++;
            }

            ostream << "\n";
        }

        ostream << "\n";
    }
}

void initCUDA(unsigned int devID);

void initGrid(unsigned int batchSize,
              unsigned int nbChannels,
              unsigned int channelsHeight,
              unsigned int channelsWidth,
              unsigned int nbOutputs,
              unsigned int outputsHeight,
              unsigned int outputsWidth,
              unsigned int kernelHeight,
              unsigned int kernelWidth,
              bool u,
              bool unitMap,
              kernel_T k,
              dim3& threadsPerBlocks,
              dim3& blocksPerGrid);

void cudaOffSetMemoryManager(unsigned int batchSize,
                             int sizeFirstChunk,
                             int sizeSecondChunk,
                             DATA_T* WorkSpace,
                             DATA_T* data);

void dumpMem(int size, DATA_T* data, std::string fileName);

//////////////////////////////////////CUDA
/// KERNELS/////////////////////////////////////////

void convcell_propagate(unsigned int nbChannels,
                        unsigned int channelsHeight,
                        unsigned int channelsWidth,
                        unsigned int paddingY,
                        unsigned int paddingX,
                        unsigned int strideY,
                        unsigned int strideX,
                        unsigned int subSampleY,
                        unsigned int subSampleX,
                        const DATA_T* inputs,
                        unsigned int oySize,
                        unsigned int oxSize,
                        unsigned int nbOutputs_,
                        unsigned int outputsHeight,
                        unsigned int outputsWidth,
                        unsigned int nbOutputs,
                        unsigned int outputOffset,
                        DATA_T* outputs,
                        unsigned int kernelHeight,
                        unsigned int kernelWidth,
                        const BDATA_T* bias,
                        const WDATA_T* weights,
                        ActivationFunction_T func,
                        int shift,
                        dim3 threadsPerBlocks,
                        dim3 blocksPerGrid);

void convcell_upropagate(unsigned int nbChannels,
                         unsigned int channelsHeight,
                         unsigned int channelsWidth,
                         unsigned int paddingY,
                         unsigned int paddingX,
                         unsigned int strideY,
                         unsigned int strideX,
                         unsigned int subSampleY,
                         unsigned int subSampleX,
                         const UDATA_T* inputs,
                         unsigned int oySize,
                         unsigned int oxSize,
                         unsigned int nbOutputs_,
                         unsigned int outputsHeight,
                         unsigned int outputsWidth,
                         unsigned int nbOutputs,
                         unsigned int outputOffset,
                         DATA_T* outputs,
                         unsigned int kernelHeight,
                         unsigned int kernelWidth,
                         const BDATA_T* bias,
                         const WDATA_T* weights,
                         ActivationFunction_T func,
                         int shift,
                         dim3 threadsPerBlocks,
                         dim3 blocksPerGrid);

void poolcell_propagate(unsigned int nbChannels,
                        unsigned int channelsHeight,
                        unsigned int channelsWidth,
                        unsigned int strideY,
                        unsigned int strideX,
                        const DATA_T* inputs,
                        unsigned int nbOutputs_,
                        unsigned int outputsHeight,
                        unsigned int outputsWidth,
                        unsigned int nbOutputs,
                        unsigned int outputOffset,
                        DATA_T* outputs,
                        unsigned int poolHeight,
                        unsigned int poolWidth,
                        const char* mapping,
                        Pooling_T pooling,
                        ActivationFunction_T func,
                        int shift,
                        dim3 threadsPerBlocks,
                        dim3 blocksPerGrid,
                        bool unitMap);

void poolcell_upropagate(unsigned int nbChannels,
                         unsigned int channelsHeight,
                         unsigned int channelsWidth,
                         unsigned int strideY,
                         unsigned int strideX,
                         const UDATA_T* inputs,
                         unsigned int nbOutputs_,
                         unsigned int outputsHeight,
                         unsigned int outputsWidth,
                         unsigned int nbOutputs,
                         unsigned int outputOffset,
                         DATA_T* outputs,
                         unsigned int poolHeight,
                         unsigned int poolWidth,
                         const char* mapping,
                         Pooling_T pooling,
                         ActivationFunction_T func,
                         int shift,
                         dim3 threadsPerBlocks,
                         dim3 blocksPerGrid,
                         bool unitMap);

void fccell_propagate_2d(unsigned int nbChannels,
                         unsigned int channelsHeight,
                         unsigned int channelsWidth,
                         const DATA_T* inputs,
                         unsigned int nbOutputs_,
                         unsigned int nbOutputs,
                         unsigned int outputOffset,
                         DATA_T* outputs,
                         unsigned int nbChannels_,
                         const BDATA_T* bias,
                         const DATA_T* weights,
                         ActivationFunction_T func,
                         int shift,
                         dim3 threadsPerBlocks,
                         dim3 blocksPerGrid);

void fccell_upropagate_2d(unsigned int nbChannels,
                          unsigned int channelsHeight,
                          unsigned int channelsWidth,
                          const UDATA_T* inputs,
                          unsigned int nbOutputs_,
                          unsigned int nbOutputs,
                          unsigned int outputOffset,
                          DATA_T* outputs,
                          unsigned int nbChannels_,
                          const BDATA_T* bias,
                          const DATA_T* weights,
                          ActivationFunction_T func,
                          int shift,
                          dim3 threadsPerBlocks,
                          dim3 blocksPerGrid);

void fccell_propagate(unsigned int nbChannels,
                      const DATA_T* inputs,
                      unsigned int nbOutputs_,
                      unsigned int nbOutputs,
                      unsigned int outputOffset,
                      DATA_T* outputs,
                      const BDATA_T* bias,
                      const DATA_T* weights,
                      ActivationFunction_T func,
                      int shift,
                      dim3 threadsPerBlocks,
                      dim3 blocksPerGrid);

void fccell_upropagate(unsigned int nbChannels,
                       const UDATA_T* inputs,
                       unsigned int nbOutputs_,
                       unsigned int nbOutputs,
                       unsigned int outputOffset,
                       DATA_T* outputs,
                       const BDATA_T* bias,
                       const DATA_T* weights,
                       ActivationFunction_T func,
                       int shift,
                       dim3 threadsPerBlocks,
                       dim3 blocksPerGrid);

void softmaxcell_propagate(unsigned int nbOutputs,
                           unsigned int outputsHeight,
                           unsigned int outputsWidth,
                           unsigned int batchSize,
                           const DATA_T* inputs,
                           DATA_T*outputs,
                           dim3 threadsPerBlocks,
                           dim3 blocksPerGrid);

void output_generation(unsigned int nbOutputs,
                       unsigned int batchSize,
                       DATA_T* outputs,
                       uint32_t* outputEstimated);


void spatial_output_generation(unsigned int nbOutputs,
                               unsigned int outputsHeight,
                               unsigned int outputsWidth,
                               unsigned int batchSize,
                               DATA_T* inputs,
                               DATA_T* outputs,
                               uint32_t* outputEstimated,
                               dim3& threadsPerBlocks,
                               dim3& blocksPerGrid);
///////////////////////////////////////////////////////////////////////////////////////////

#endif // N2D2_EXPORTC_DEEPNET_CUDA_H
