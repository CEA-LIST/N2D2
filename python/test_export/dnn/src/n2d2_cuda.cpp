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

#include "n2d2_cuda.hpp"

cudaHandleStruct cudaHandles;

void set_profiling()
{
    cudaHandles.isProfiled = true;
}

void report_per_layer_profiling(unsigned int nbIter)
{
    double totalProcessTime = 0.0;
    for (std::vector<cudaProfiling>::iterator it
         = cudaHandles.profiling.begin(),
         itEnd = cudaHandles.profiling.end();
         it != itEnd;
         ++it) {
        totalProcessTime += (*it).processTime / (nbIter);
    }

    for (std::vector<cudaProfiling>::iterator it
         = cudaHandles.profiling.begin(),
         itEnd = cudaHandles.profiling.end();
         it != itEnd;
         ++it) {
        const double processTimeUs = (*it).processTime / (nbIter);
        const double workLoad = (processTimeUs / totalProcessTime) * 100.0;
        std::string barrelLoad(((unsigned int)workLoad + 1) * 2, '*');
        std::cout << "(" << std::setfill('0') << std::setw(2)
                  << (unsigned int)workLoad << "%)  " << barrelLoad
                  << "    " << (*it).name << ": " << processTimeUs << " us"
                  << std::endl;
     }
}

void initCUDA(unsigned int devID)
{

    // Detect CUDA devices
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    std::cout << "Number of CUDA device found: " << device_count << std::endl;
    std::cout << "---------------------------------------" << std::endl;
    for (int it = 0; it < device_count; it++)
        cudaHandles.devices.push_back(it);

    for (std::vector<int>::iterator it = cudaHandles.devices.begin(),
                                    itEnd = cudaHandles.devices.end();
         it != itEnd;
         ++it) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, (*it));

        std::cout
            << "Found device: " << deviceProp.name
            << "\n"
               "    CUDA compute capability: " << deviceProp.major << "."
            << deviceProp.minor << "\n"
                                   "    Number of Multiprocessors: "
            << deviceProp.multiProcessorCount
            << "\n"
               "    Max. threads sizes-xyz: ( " << deviceProp.maxThreadsDim[0]
            << ", " << deviceProp.maxThreadsDim[1] << ", "
            << deviceProp.maxThreadsDim[2] << ")\n"
                                              "    Max. grid sizes-xyz: ( "
            << deviceProp.maxGridSize[0] << ", " << deviceProp.maxGridSize[1]
            << ", " << deviceProp.maxGridSize[2]
            << ")\n"
               "    Max. threads per blocks: " << deviceProp.maxThreadsPerBlock
            << "\n"
               "    warp size: " << deviceProp.warpSize
            << "\n"
               "    Max. clock frequency: " << deviceProp.clockRate / 1000
            << " MHz\n"
               "    Shared memory size: " << (int)(deviceProp.sharedMemPerBlock)
            << " Bytes\n"
               "    Global memory size: "
            << (int)(deviceProp.totalGlobalMem / 1024 / 1024)
            << " MB\n"
               "    Max. constant memory size: "
            << (int)(deviceProp.totalConstMem / 1024) << " KB" << std::endl;
    }

    cudaSetDevice(devID);
    cudaHandles.kernels.push_back("convcell_upropagate");
    cudaHandles.kernels.push_back("convcell_propagate");
    cudaHandles.kernels.push_back("poolcell_upropagate");
    cudaHandles.kernels.push_back("poolcell_upropagate_unitmap");
    cudaHandles.kernels.push_back("poolcell_propagate");
    cudaHandles.kernels.push_back("poolcell_propagate_unitmap");
    cudaHandles.kernels.push_back("fccell_upropagate_2d");
    cudaHandles.kernels.push_back("fccell_propagate_2d");
    cudaHandles.kernels.push_back("fccell_propagate");
    cudaHandles.kernels.push_back("fccell_upropagate");
    cudaHandles.kernels.push_back("softmaxcell_propagate");
    cudaHandles.kernels.push_back("spatial_outputs_max");
}
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
              dim3& blocksPerGrid)
{
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    if (k == Convolution) {
        const std::string kernelName = (u) ? cudaHandles.kernels[0]
                                           : cudaHandles.kernels[1];
        const unsigned int maxSize
            = (unsigned int)deviceProp.maxThreadsPerBlock;
        const unsigned int prefMultiple = (unsigned int)deviceProp.warpSize;

        const unsigned int groupSize = (outputsWidth * outputsHeight < maxSize)
                                           ? outputsWidth * outputsHeight
                                           : maxSize;
        const unsigned int groupWidth
            = std::min(prefMultiple, nextDivisor(groupSize, outputsWidth));
        const unsigned int nbInputs = nbChannels * channelsHeight
                                      * channelsWidth;
        const unsigned int nbBiases = nbOutputs;
        const unsigned int nbWeights = nbChannels * nbOutputs * kernelHeight
                                       * kernelWidth;
        if (!cudaHandles.populated) {
            if(cudaHandles.isProfiled) {
                const unsigned int nbBytesRead = nbInputs + nbBiases + nbWeights;
                const unsigned int nbBytesWritten = nbOutputs * outputsHeight
                                                    * outputsWidth;
                cudaHandles.profiling.push_back(
                    cudaProfiling({kernelName, 0.0, nbBytesRead, nbBytesWritten}));
            }
            threadsPerBlocks = {groupWidth, groupSize / groupWidth, 1};
            blocksPerGrid = {nbOutputs, 1, batchSize};

            std::cout << kernelName
                      << ":\n"
                         "    Max. Threads per Blocks = " << maxSize
                      << "\n"
                         "    Preferred Blocks Size multiple = " << prefMultiple
                      << "\n"
                         "    Blocks size = (" << threadsPerBlocks.x << ", "
                      << threadsPerBlocks.y << ", " << threadsPerBlocks.z
                      << ") = "
                      << std::max<unsigned long>(threadsPerBlocks.x, 1UL)
                         * std::max<unsigned long>(threadsPerBlocks.y, 1UL)
                         * std::max<unsigned long>(threadsPerBlocks.z, 1UL)
                      << "\n"
                         "    Grid size = (" << blocksPerGrid.x << ", "
                      << blocksPerGrid.y << ", " << blocksPerGrid.z << ") = "
                      << std::max<unsigned long>(blocksPerGrid.x, 1UL)
                         * std::max<unsigned long>(blocksPerGrid.y, 1UL)
                         * std::max<unsigned long>(blocksPerGrid.z, 1UL) << "\n"
                      << "    Multi-Processors used = "
                      << (blocksPerGrid.x)
                         * (std::max<unsigned long>(blocksPerGrid.y, 1UL))
                         * (std::max<unsigned long>(blocksPerGrid.z, 1UL))
                      << std::endl;

            if (nbWeights + nbBiases > deviceProp.totalConstMem) {
                std::cout << ESC_FG_RED << "Warning: synaptic parameters don't "
                                           "fit in device constant memory"
                          << ESC_ALL_OFF << std::endl;
            }
        }
    } else if (k == Pooling) {
        const std::string kernelName = (u) ? (!unitMap)
            ? cudaHandles.kernels[2]
            : cudaHandles.kernels[3]
            : (!unitMap) ? cudaHandles.kernels[4] : cudaHandles.kernels[5];
        const unsigned int maxSize
            = (unsigned int)deviceProp.maxThreadsPerBlock;
        const unsigned int prefMultiple = (unsigned int)deviceProp.warpSize;

        const unsigned int groupSize = (outputsWidth * outputsHeight < maxSize)
                                           ? outputsWidth * outputsHeight
                                           : maxSize;
        const unsigned int groupWidth
            = std::min(prefMultiple, nextDivisor(groupSize, outputsWidth));

        if (!cudaHandles.populated) {
            if(cudaHandles.isProfiled) {
                const unsigned int nbBytesRead = nbChannels * channelsHeight
                                                 * channelsWidth;
                const unsigned int nbBytesWritten = nbOutputs * outputsHeight
                                                    * outputsWidth;
                cudaHandles.profiling.push_back(
                    cudaProfiling({kernelName, 0.0, nbBytesRead, nbBytesWritten}));
            }
            threadsPerBlocks = {groupWidth, groupSize / groupWidth, 1};
            blocksPerGrid = {nbOutputs, 1, batchSize};

            std::cout << kernelName
                      << ":\n"
                         "    Max. Threads per Blocks = " << maxSize
                      << "\n"
                         "    Preferred Blocks Size multiple = " << prefMultiple
                      << "\n"
                         "    Blocks size = (" << threadsPerBlocks.x << ", "
                      << threadsPerBlocks.y << ", " << threadsPerBlocks.z
                      << ") = "
                      << std::max<unsigned long>(threadsPerBlocks.x, 1UL)
                         * std::max<unsigned long>(threadsPerBlocks.y, 1UL)
                         * std::max<unsigned long>(threadsPerBlocks.z, 1UL)
                      << "\n"
                         "    Grid size = (" << blocksPerGrid.x << ", "
                      << blocksPerGrid.y << ", " << blocksPerGrid.z << ") = "
                      << std::max<unsigned long>(blocksPerGrid.x, 1UL)
                         * std::max<unsigned long>(blocksPerGrid.y, 1UL)
                         * std::max<unsigned long>(blocksPerGrid.z, 1UL) << "\n"
                      << "    Multi-Processors used = "
                      << (blocksPerGrid.x)
                         * (std::max<unsigned long>(blocksPerGrid.y, 1UL))
                         * (std::max<unsigned long>(blocksPerGrid.z, 1UL))
                      << std::endl;
        }
    } else if (k == fcCellProp2D) {
        const std::string kernelName = (u) ? cudaHandles.kernels[6]
                                           : cudaHandles.kernels[7];
        const unsigned int nbInputs = nbChannels * channelsHeight
                                      * channelsWidth;
        const unsigned int nbBiases = nbOutputs;
        const unsigned int nbWeights = nbChannels * nbOutputs;
        const unsigned int maxSize
            = (unsigned int)deviceProp.maxThreadsPerBlock;
        const unsigned int prefMultiple = (unsigned int)deviceProp.warpSize;

        if (!cudaHandles.populated) {
            if(cudaHandles.isProfiled) {
                const unsigned int nbBytesRead = nbInputs + nbBiases + nbWeights;
                const unsigned int nbBytesWritten = nbOutputs;
                cudaHandles.profiling.push_back(
                    cudaProfiling({kernelName, 0.0, nbBytesRead, nbBytesWritten}));
            }
            threadsPerBlocks = {prefMultiple, 1, 1};
            blocksPerGrid = {nbOutputs, batchSize, 1};
            std::cout << kernelName
                      << ":\n"
                         "    Max. Threads per Blocks = " << maxSize
                      << "\n"
                         "    Preferred Blocks Size multiple = " << prefMultiple
                      << "\n"
                         "    Blocks size = (" << threadsPerBlocks.x << ", "
                      << threadsPerBlocks.y << ", " << threadsPerBlocks.z
                      << ") = "
                      << std::max<unsigned long>(threadsPerBlocks.x, 1UL)
                         * std::max<unsigned long>(threadsPerBlocks.y, 1UL)
                         * std::max<unsigned long>(threadsPerBlocks.z, 1UL)
                      << "\n"
                         "    Grid size = (" << blocksPerGrid.x << ", "
                      << blocksPerGrid.y << ", " << blocksPerGrid.z << ") = "
                      << std::max<unsigned long>(blocksPerGrid.x, 1UL)
                         * std::max<unsigned long>(blocksPerGrid.y, 1UL)
                         * std::max<unsigned long>(blocksPerGrid.z, 1UL) << "\n"
                      << "    Multi-Processors used = "
                      << (blocksPerGrid.x)
                         * (std::max<unsigned long>(blocksPerGrid.y, 1UL))
                         * (std::max<unsigned long>(blocksPerGrid.z, 1UL))
                      << std::endl;
            if (nbWeights + nbBiases > deviceProp.totalConstMem) {
                std::cout << ESC_FG_RED << "Warning: synaptic parameters don't "
                                           "fit in device constant memory"
                          << ESC_ALL_OFF << std::endl;
            }
        }
    } else if (k == fcCellProp) {
        const std::string kernelName = cudaHandles.kernels[8];
        const unsigned int nbInputs = nbChannels * channelsHeight
                                      * channelsWidth;
        const unsigned int nbBiases = nbOutputs;
        const unsigned int nbWeights = nbChannels * nbOutputs;
        const unsigned int maxSize
            = (unsigned int)deviceProp.maxThreadsPerBlock;
        const unsigned int prefMultiple = (unsigned int)deviceProp.warpSize;

        if (!cudaHandles.populated) {
            if(cudaHandles.isProfiled) {
                const unsigned int nbBytesRead = nbChannels + nbBiases + nbWeights;
                const unsigned int nbBytesWritten = nbOutputs;
                cudaHandles.profiling.push_back(
                    cudaProfiling({kernelName, 0.0, nbBytesRead, nbBytesWritten}));
            }
            threadsPerBlocks = {prefMultiple, 1, 1};
            blocksPerGrid = {nbOutputs, batchSize, 1};
            std::cout << kernelName
                      << ":\n"
                         "    Max. Threads per Blocks = " << maxSize
                      << "\n"
                         "    Preferred Blocks Size multiple = " << prefMultiple
                      << "\n"
                         "    Blocks size = (" << threadsPerBlocks.x << ", "
                      << threadsPerBlocks.y << ", " << threadsPerBlocks.z
                      << ") = "
                      << std::max<unsigned long>(threadsPerBlocks.x, 1UL)
                         * std::max<unsigned long>(threadsPerBlocks.y, 1UL)
                         * std::max<unsigned long>(threadsPerBlocks.z, 1UL)
                      << "\n"
                         "    Grid size = (" << blocksPerGrid.x << ", "
                      << blocksPerGrid.y << ", " << blocksPerGrid.z << ") = "
                      << std::max<unsigned long>(blocksPerGrid.x, 1UL)
                         * std::max<unsigned long>(blocksPerGrid.y, 1UL)
                         * std::max<unsigned long>(blocksPerGrid.z, 1UL) << "\n"
                      << "    Multi-Processors used = "
                      << (blocksPerGrid.x)
                         * (std::max<unsigned long>(blocksPerGrid.y, 1UL))
                         * (std::max<unsigned long>(blocksPerGrid.z, 1UL))
                      << std::endl;
            if (nbWeights + nbBiases > deviceProp.totalConstMem) {
                std::cout << ESC_FG_RED << "Warning: synaptic parameters don't "
                                           "fit in device constant memory"
                          << ESC_ALL_OFF << std::endl;
            }
        }
    } else if (k == Softmax) {
        const std::string kernelName = cudaHandles.kernels[9];
        unsigned int groupSize = nbOutputs;
        threadsPerBlocks = {groupSize, 1, 1};
        blocksPerGrid = {outputsWidth, outputsHeight, batchSize};

        if (!cudaHandles.populated) {
            if(cudaHandles.isProfiled) {
                const unsigned int nbBytesRead = nbOutputs * outputsHeight
                                                 * outputsWidth;
                const unsigned int nbBytesWritten = outputsHeight * outputsWidth;
                cudaHandles.profiling.push_back(cudaProfiling(
                    {kernelName, 0.0, nbBytesRead, nbBytesWritten}));

            }
            std::cout << "softmaxcell_propagate:\n"
                         "    Blocks size = (" << threadsPerBlocks.x << ", "
                      << threadsPerBlocks.y << ", " << threadsPerBlocks.z
                      << ") = "
                      << std::max<unsigned long>(threadsPerBlocks.x, 1UL)
                         * std::max<unsigned long>(threadsPerBlocks.y, 1UL)
                         * std::max<unsigned long>(threadsPerBlocks.z, 1UL)
                      << "\n"
                         "    Grid size = (" << blocksPerGrid.x << ", "
                      << blocksPerGrid.y << ", " << blocksPerGrid.z << ") = "
                      << std::max<unsigned long>(blocksPerGrid.x, 1UL)
                         * std::max<unsigned long>(blocksPerGrid.y, 1UL)
                         * std::max<unsigned long>(blocksPerGrid.z, 1UL) << "\n"
                      << "    Multi-Processors used = "
                      << (blocksPerGrid.x)
                         * (std::max<unsigned long>(blocksPerGrid.y, 1UL))
                         * (std::max<unsigned long>(blocksPerGrid.z, 1UL))
                      << std::endl;
        }
    } else if (k == SpatialOutputs) {

        unsigned int groupSize = nbOutputs;
        threadsPerBlocks = {groupSize, 1, 1};
        blocksPerGrid = {outputsWidth, outputsHeight, batchSize};

        if (!cudaHandles.populated) {
            if(cudaHandles.isProfiled) {
                const unsigned int nbBytesRead = nbOutputs * outputsHeight
                                                 * outputsWidth;
                const unsigned int nbBytesWritten = outputsHeight * outputsWidth;
                cudaHandles.profiling.push_back(cudaProfiling(
                    {"spatial_outputs_max", 0.0, nbBytesRead, nbBytesWritten}));
            }
            std::cout << "spatial_outputs_max:\n"
                         "    Blocks size = (" << threadsPerBlocks.x << ", "
                      << threadsPerBlocks.y << ", " << threadsPerBlocks.z
                      << ") = "
                      << std::max<unsigned long>(threadsPerBlocks.x, 1UL)
                         * std::max<unsigned long>(threadsPerBlocks.y, 1UL)
                         * std::max<unsigned long>(threadsPerBlocks.z, 1UL)
                      << "\n"
                         "    Grid size = (" << blocksPerGrid.x << ", "
                      << blocksPerGrid.y << ", " << blocksPerGrid.z << ") = "
                      << std::max<unsigned long>(blocksPerGrid.x, 1UL)
                         * std::max<unsigned long>(blocksPerGrid.y, 1UL)
                         * std::max<unsigned long>(blocksPerGrid.z, 1UL) << "\n"
                      << "    Multi-Processors used = "
                      << (blocksPerGrid.x)
                         * (std::max<unsigned long>(blocksPerGrid.y, 1UL))
                         * (std::max<unsigned long>(blocksPerGrid.z, 1UL))
                      << std::endl;
        }
    }
}

void output_generation(unsigned int nbOutputs,
                       unsigned int batchSize,
                       DATA_T* outputs,
                       uint32_t* outputEstimated)
{
    static DATA_T* outputsData = NULL;

    if (outputsData == NULL) {
        outputsData = (DATA_T*)malloc(sizeof(DATA_T) * nbOutputs * batchSize);

        if (!outputsData)
            throw std::runtime_error(
                "fccell_output_max(): could not allocate memory");
    }
    cudaDeviceSynchronize();
    checkCudaErrors(cudaMemcpy(outputsData,
                               outputs,
                               batchSize * nbOutputs * sizeof(DATA_T),
                               cudaMemcpyDeviceToHost));

    if(cudaHandles.isProfiled) {
        for (std::vector<float>::iterator it = cudaHandles.events.begin(),
                                          itBegin = cudaHandles.events.begin(),
                                          itEnd = cudaHandles.events.end();
             it != itEnd;
             ++it) {
            cudaHandles.profiling[it - itBegin].processTime += 1.0e3
                                                               * (double)(*it);
        }

        cudaHandles.events.clear();
    }
    for (unsigned int i = 0; i < batchSize; i++) {

        DATA_T maxVal = outputsData[i * nbOutputs];
        unsigned int outputMax = 0;

        for (unsigned int output = 1 + nbOutputs * i;
             output < nbOutputs * (i + 1);
             ++output) {
            if (outputsData[output] > maxVal) {
                maxVal = outputsData[output];
                outputMax = output - i * nbOutputs;
            }
        }
        outputEstimated[i] = outputMax;
    }
}

void dumpMem(int size, DATA_T* data, std::string fileName)
{

    std::ofstream file;
    file.open(fileName.c_str());

    DATA_T* watch_eagle(NULL);
    watch_eagle = new DATA_T[size];

    checkCudaErrors(cudaMemcpy(
        watch_eagle, data, sizeof(DATA_T) * size, cudaMemcpyDeviceToHost));

    for (unsigned int i = 0; i < size; i++)
        file << "data[" << i << "]= " << watch_eagle[i] << "\n";

    std::cout << "dump mem in file " << fileName.c_str() << "done"
              << "\n";
    file.close();
    delete[] watch_eagle;
}

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
                        dim3 blocksPerGrid)
{
    if(cudaHandles.isProfiled) {
        float event;
        cuda_convcell_propagate(nbChannels,
                                channelsHeight,
                                channelsWidth,
                                paddingY,
                                paddingX,
                                strideY,
                                strideX,
                                subSampleY,
                                subSampleX,
                                inputs,
                                oySize,
                                oxSize,
                                nbOutputs_,
                                outputsHeight,
                                outputsWidth,
                                nbOutputs,
                                outputOffset,
                                outputs,
                                kernelHeight,
                                kernelWidth,
                                bias,
                                weights,
                                func,
                                shift,
                                threadsPerBlocks,
                                blocksPerGrid,
                                true,
                                &event);
        cudaHandles.events.push_back(event);
    }
    else {
        cuda_convcell_propagate(nbChannels,
                                channelsHeight,
                                channelsWidth,
                                paddingY,
                                paddingX,
                                strideY,
                                strideX,
                                subSampleY,
                                subSampleX,
                                inputs,
                                oySize,
                                oxSize,
                                nbOutputs_,
                                outputsHeight,
                                outputsWidth,
                                nbOutputs,
                                outputOffset,
                                outputs,
                                kernelHeight,
                                kernelWidth,
                                bias,
                                weights,
                                func,
                                shift,
                                threadsPerBlocks,
                                blocksPerGrid,
                                false,
                                NULL);
    }
}


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
                         dim3 blocksPerGrid)
{
    if(cudaHandles.isProfiled) {
        float event;
        cuda_convcell_upropagate(nbChannels,
                                 channelsHeight,
                                 channelsWidth,
                                 paddingY,
                                 paddingX,
                                 strideY,
                                 strideX,
                                 subSampleY,
                                 subSampleX,
                                 inputs,
                                 oySize,
                                 oxSize,
                                 nbOutputs_,
                                 outputsHeight,
                                 outputsWidth,
                                 nbOutputs,
                                 outputOffset,
                                 outputs,
                                 kernelHeight,
                                 kernelWidth,
                                 bias,
                                 weights,
                                 func,
                                 shift,
                                 threadsPerBlocks,
                                 blocksPerGrid,
                                 true,
                                 &event);
        cudaHandles.events.push_back(event);
    }
    else {
        cuda_convcell_upropagate(nbChannels,
                                 channelsHeight,
                                 channelsWidth,
                                 paddingY,
                                 paddingX,
                                 strideY,
                                 strideX,
                                 subSampleY,
                                 subSampleX,
                                 inputs,
                                 oySize,
                                 oxSize,
                                 nbOutputs_,
                                 outputsHeight,
                                 outputsWidth,
                                 nbOutputs,
                                 outputOffset,
                                 outputs,
                                 kernelHeight,
                                 kernelWidth,
                                 bias,
                                 weights,
                                 func,
                                 shift,
                                 threadsPerBlocks,
                                 blocksPerGrid,
                                 false,
                                 NULL);
    }
}

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
                        bool unitMap)
{
    if(cudaHandles.isProfiled) {
        float event;
        cuda_poolcell_propagate(nbChannels,
                                channelsHeight,
                                channelsWidth,
                                strideY,
                                strideX,
                                inputs,
                                nbOutputs_,
                                outputsHeight,
                                outputsWidth,
                                nbOutputs,
                                outputOffset,
                                outputs,
                                poolHeight,
                                poolWidth,
                                mapping,
                                pooling,
                                func,
                                shift,
                                threadsPerBlocks,
                                blocksPerGrid,
                                true,
                                &event,
                                unitMap);
        cudaHandles.events.push_back(event);
    }
    else {
        cuda_poolcell_propagate(nbChannels,
                                channelsHeight,
                                channelsWidth,
                                strideY,
                                strideX,
                                inputs,
                                nbOutputs_,
                                outputsHeight,
                                outputsWidth,
                                nbOutputs,
                                outputOffset,
                                outputs,
                                poolHeight,
                                poolWidth,
                                mapping,
                                pooling,
                                func,
                                shift,
                                threadsPerBlocks,
                                blocksPerGrid,
                                false,
                                NULL,
                                unitMap);
    }
}

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
                         bool unitMap)
{
    if(cudaHandles.isProfiled) {
        float event;
        cuda_poolcell_upropagate(nbChannels,
                                 channelsHeight,
                                 channelsWidth,
                                 strideY,
                                 strideX,
                                 inputs,
                                 nbOutputs_,
                                 outputsHeight,
                                 outputsWidth,
                                 nbOutputs,
                                 outputOffset,
                                 outputs,
                                 poolHeight,
                                 poolWidth,
                                 mapping,
                                 pooling,
                                 func,
                                 shift,
                                 threadsPerBlocks,
                                 blocksPerGrid,
                                 true,
                                 &event,
                                 unitMap);
        cudaHandles.events.push_back(event);
    }
    else {
        cuda_poolcell_upropagate(nbChannels,
                                 channelsHeight,
                                 channelsWidth,
                                 strideY,
                                 strideX,
                                 inputs,
                                 nbOutputs_,
                                 outputsHeight,
                                 outputsWidth,
                                 nbOutputs,
                                 outputOffset,
                                 outputs,
                                 poolHeight,
                                 poolWidth,
                                 mapping,
                                 pooling,
                                 func,
                                 shift,
                                 threadsPerBlocks,
                                 blocksPerGrid,
                                 false,
                                 NULL,
                                 unitMap);
    }
}

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
                         dim3 blocksPerGrid)
{
    if(cudaHandles.isProfiled) {
        float event;
        cuda_fccell_propagate_2d(nbChannels,
                                 channelsHeight,
                                 channelsWidth,
                                 inputs,
                                 nbOutputs_,
                                 nbOutputs,
                                 outputOffset,
                                 outputs,
                                 nbChannels_,
                                 bias,
                                 weights,
                                 func,
                                 shift,
                                 threadsPerBlocks,
                                 blocksPerGrid,
                                 true,
                                 &event);
        cudaHandles.events.push_back(event);
    }
    else {
        cuda_fccell_propagate_2d(nbChannels,
                                 channelsHeight,
                                 channelsWidth,
                                 inputs,
                                 nbOutputs_,
                                 nbOutputs,
                                 outputOffset,
                                 outputs,
                                 nbChannels_,
                                 bias,
                                 weights,
                                 func,
                                 shift,
                                 threadsPerBlocks,
                                 blocksPerGrid,
                                 false,
                                 NULL);
    }
}

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
                          dim3 blocksPerGrid)
{
    if(cudaHandles.isProfiled) {
        float event;
        cuda_fccell_upropagate_2d(nbChannels,
                                  channelsHeight,
                                  channelsWidth,
                                  inputs,
                                  nbOutputs_,
                                  nbOutputs,
                                  outputOffset,
                                  outputs,
                                  nbChannels_,
                                  bias,
                                  weights,
                                  func,
                                  shift,
                                  threadsPerBlocks,
                                  blocksPerGrid,
                                  true,
                                  &event);
        cudaHandles.events.push_back(event);
    }
    else {
        cuda_fccell_upropagate_2d(nbChannels,
                                  channelsHeight,
                                  channelsWidth,
                                  inputs,
                                  nbOutputs_,
                                  nbOutputs,
                                  outputOffset,
                                  outputs,
                                  nbChannels_,
                                  bias,
                                  weights,
                                  func,
                                  shift,
                                  threadsPerBlocks,
                                  blocksPerGrid,
                                  false,
                                  NULL);
    }
}

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
                      dim3 blocksPerGrid)
{

    if(cudaHandles.isProfiled) {
        float event = 0;
        cuda_fccell_propagate(nbChannels,
                              inputs,
                              nbOutputs_,
                              nbOutputs,
                              outputOffset,
                              outputs,
                              bias,
                              weights,
                              func,
                              shift,
                              threadsPerBlocks,
                              blocksPerGrid,
                              true,
                              &event);
        cudaHandles.events.push_back(event);
    }
    else {
        cuda_fccell_propagate(nbChannels,
                              inputs,
                              nbOutputs_,
                              nbOutputs,
                              outputOffset,
                              outputs,
                              bias,
                              weights,
                              func,
                              shift,
                              threadsPerBlocks,
                              blocksPerGrid,
                              false,
                              NULL);
    }
}

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
                       dim3 blocksPerGrid)
{

    if(cudaHandles.isProfiled) {
        float event = 0;
        cuda_fccell_upropagate(nbChannels,
                              inputs,
                              nbOutputs_,
                              nbOutputs,
                              outputOffset,
                              outputs,
                              bias,
                              weights,
                              func,
                              shift,
                              threadsPerBlocks,
                              blocksPerGrid,
                              true,
                              &event);
        cudaHandles.events.push_back(event);
    }
    else {
        cuda_fccell_upropagate(nbChannels,
                              inputs,
                              nbOutputs_,
                              nbOutputs,
                              outputOffset,
                              outputs,
                              bias,
                              weights,
                              func,
                              shift,
                              threadsPerBlocks,
                              blocksPerGrid,
                              false,
                              NULL);
    }
}

void softmaxcell_propagate(unsigned int nbOutputs,
                           unsigned int outputsHeight,
                           unsigned int outputsWidth,
                           unsigned int batchSize,
                           const DATA_T* inputs,
                           DATA_T*outputs,
                           dim3 threadsPerBlocks,
                           dim3 blocksPerGrid)
{
    if(cudaHandles.isProfiled) {
        float event = 0;
        cuda_softmaxcell_propagate(nbOutputs,
                                  outputsHeight,
                                  outputsWidth,
                                  batchSize,
                                  inputs,
                                  outputs,
                                  threadsPerBlocks,
                                  blocksPerGrid,
                                  true,
                                  &event);
        cudaHandles.events.push_back(event);
    }
    else {
        cuda_softmaxcell_propagate(nbOutputs,
                                  outputsHeight,
                                  outputsWidth,
                                  batchSize,
                                  inputs,
                                  outputs,
                                  threadsPerBlocks,
                                  blocksPerGrid,
                                  false,
                                  NULL);
    }
}

void spatial_output_generation(unsigned int nbOutputs,
                               unsigned int outputsHeight,
                               unsigned int outputsWidth,
                               unsigned int batchSize,
                               DATA_T* inputs,
                               DATA_T* outputs,
                               uint32_t* outputEstimated,
                               dim3& threadsPerBlocks,
                               dim3& blocksPerGrid)
{
    if(cudaHandles.isProfiled) {
        float event;
        cuda_spatial_outputs_max(nbOutputs,
                                 outputsHeight,
                                 outputsWidth,
                                 batchSize,
                                 inputs,
                                 outputs,
                                 threadsPerBlocks,
                                 blocksPerGrid,
                                 true,
                                 &event);
        cudaHandles.events.push_back(event);
    }
    else {
        cuda_spatial_outputs_max(nbOutputs,
                                 outputsHeight,
                                 outputsWidth,
                                 batchSize,
                                 inputs,
                                 outputs,
                                 threadsPerBlocks,
                                 blocksPerGrid,
                                 false,
                                 NULL);
    }

    const unsigned int size = outputsHeight * outputsWidth * batchSize;
    static DATA_T* outputsData = NULL;
    if (outputsData == NULL) {
        outputsData = (DATA_T*)malloc(sizeof(DATA_T) * size);

        if (!outputsData)
            throw std::runtime_error(
                "spatial_output_max(): could not allocate memory");
    }


    checkCudaErrors(cudaMemcpy(
        outputsData, outputs, size * sizeof(DATA_T), cudaMemcpyDeviceToHost));

    if(cudaHandles.isProfiled) {
        for (std::vector<float>::iterator it = cudaHandles.events.begin(),
                                          itBegin = cudaHandles.events.begin(),
                                          itEnd = cudaHandles.events.end();
             it != itEnd;
             ++it) {
            cudaHandles.profiling[it - itBegin].processTime += 1.0e3
                                                               * (double)(*it);
        }

        cudaHandles.events.clear();
    }

    for (unsigned int n = 0; n < batchSize; ++n)
        for (unsigned int oy = 0; oy < outputsHeight; ++oy)
            for (unsigned int ox = 0; ox < outputsWidth; ++ox)
                outputEstimated
                    [ox + oy * outputsWidth + n * outputsWidth * outputsHeight]
                    = outputsData[ox + oy * outputsWidth + n * outputsWidth
                                                           * outputsHeight];
}



/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
