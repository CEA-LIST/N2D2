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

#include "Cell/Cell_Frame_CUDA_Kernels.hpp"

__global__
void cudaPopulateNbTargetOutputs_kernel(int* targets,
    unsigned int* nbTargetOutputs,
    unsigned int nbOutputs,
    unsigned int outputsHeight,
    unsigned int outputsWidth,
    unsigned int batchSize,
    int* err)
{
    const unsigned int batchTargetOffset = blockIdx.z
                                           * outputsHeight * outputsWidth;
    const unsigned int batchNbTargetOutputsOffset = blockIdx.z
                                           * ((nbOutputs > 1) ? nbOutputs : 2);

    for (unsigned int oy = threadIdx.y; oy < outputsHeight;
            oy += blockDim.y) {
        for (unsigned int ox = threadIdx.x; ox < outputsWidth;
                ox += blockDim.x)
        {
            const unsigned int targetsIdx = ox + oy * outputsWidth
                + batchTargetOffset;

            if (targets[targetsIdx] >= 0) {
                if ((nbOutputs > 1 && targets[targetsIdx] >= nbOutputs)
                    || (nbOutputs == 1 && (targets[targetsIdx] < 0
                                           || targets[targetsIdx] > 1)))
                {
                    err[0] = targets[targetsIdx];
                    err[1] = ox;
                    err[2] = oy;
                    return;
                }

                const unsigned int nbTargetOutputsIdx = targets[targetsIdx]
                    + batchNbTargetOutputsOffset;

                atomicAdd(nbTargetOutputs + nbTargetOutputsIdx, 1U);
            }
        }
    }
}

//Half
__global__ void cudaHReduce_kernel(__half* idata, __half* odata,
                                   unsigned int size)
{
    __shared__ __half sdata[256];

    // each thread loads one element from global to shared mem
    const unsigned int tid = threadIdx.x;
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size)
        sdata[tid] = idata[index];
    else
        sdata[tid] = __float2half(0.0f);

    __syncthreads();

    // do reduction in shared mem
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride>>= 1) {
        if (tid < stride) {
#if __CUDA_ARCH__ >= 530
            sdata[tid] = __hadd(sdata[tid], sdata[tid + stride]);
#else
            sdata[tid] = __float2half(__half2float(sdata[tid])
                                        + __half2float(sdata[tid + stride]));
#endif

        }
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) {
#if __CUDA_ARCH__ >= 700
        atomicAdd(odata, sdata[0]);
// Work on open source Docker, but fail in internal Docker...
// error: no instance of overloaded function "atomicAdd" matches the argument list
// argument types are: (__half2 *, __half2)
//#elif __CUDA_ARCH__ >= 600
//        // Not sure it works!
//        atomicAdd((__half2*)odata, *((__half2*)sdata));
#else
        // Not supported!
        asm("trap;");
#endif
    }
}

__global__
void cudaHSetOutputTargets_kernel(int* targets,
    unsigned int* nbTargetOutputs,
    __half* diffInputs,
    unsigned int nbOutputs,
    unsigned int outputsHeight,
    unsigned int outputsWidth,
    unsigned int batchSize)
{
    const unsigned int batchOutputOffset = blockIdx.z * nbOutputs
                                           * outputsHeight * outputsWidth;
    const unsigned int batchTargetOffset = blockIdx.z
                                           * outputsHeight * outputsWidth;
    const unsigned int batchNbTargetOutputsOffset = blockIdx.z
                                           * ((nbOutputs > 1) ? nbOutputs : 2);

    for (unsigned int output = blockIdx.x; output < nbOutputs;
        output += gridDim.x) {                                
        for (unsigned int oy = threadIdx.y; oy < outputsHeight;
                oy += blockDim.y) {
            for (unsigned int ox = threadIdx.x; ox < outputsWidth;
                    ox += blockDim.x)
            {
                const unsigned int outputsIdx
                    = ox + (oy + output * outputsHeight) * outputsWidth
                        + batchOutputOffset;
                const unsigned int targetsIdx = ox + oy * outputsWidth
                    + batchTargetOffset;

                if (targets[targetsIdx] >= 0) {
                    const unsigned int nbTargetOutputsIdx = targets[targetsIdx]
                        + batchNbTargetOutputsOffset;

                    diffInputs[outputsIdx] = ((nbOutputs > 1
                                && targets[targetsIdx] == (int)output)
                            || (nbOutputs == 1 && targets[targetsIdx] == 1))
                        ? __float2half(1.0f
                            / (float)nbTargetOutputs[nbTargetOutputsIdx])
                        : __float2half(-1.0f
                            / (float)nbTargetOutputs[nbTargetOutputsIdx]);
                }
                else
                    diffInputs[outputsIdx] = __float2half(0.0f);
            }
        }
    }
}

//Float
__global__ void cudaSReduce_kernel(float* idata, float* odata,
                                   unsigned int size)
{
    __shared__ float sdata[256];

    // each thread loads one element from global to shared mem
    const unsigned int tid = threadIdx.x;
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size)
        sdata[tid] = idata[index];
    else
        sdata[tid] = 0.0f;

    __syncthreads();

    // do reduction in shared mem
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride>>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0)
        atomicAdd(odata, sdata[0]);
}

__global__
void cudaSSetOutputTargets_kernel(int* targets,
    unsigned int* nbTargetOutputs,
    float* diffInputs,
    unsigned int nbOutputs,
    unsigned int outputsHeight,
    unsigned int outputsWidth,
    unsigned int batchSize)
{
    const unsigned int batchOutputOffset = blockIdx.z * nbOutputs
                                           * outputsHeight * outputsWidth;
    const unsigned int batchTargetOffset = blockIdx.z
                                           * outputsHeight * outputsWidth;
    const unsigned int batchNbTargetOutputsOffset = blockIdx.z
                                           * ((nbOutputs > 1) ? nbOutputs : 2);

    for (unsigned int output = blockIdx.x; output < nbOutputs;
        output += gridDim.x) {                                
        for (unsigned int oy = threadIdx.y; oy < outputsHeight;
                oy += blockDim.y) {
            for (unsigned int ox = threadIdx.x; ox < outputsWidth;
                    ox += blockDim.x)
            {
                const unsigned int outputsIdx
                    = ox + (oy + output * outputsHeight) * outputsWidth
                        + batchOutputOffset;
                const unsigned int targetsIdx = ox + oy * outputsWidth
                    + batchTargetOffset;

                if (targets[targetsIdx] >= 0) {
                    const unsigned int nbTargetOutputsIdx = targets[targetsIdx]
                        + batchNbTargetOutputsOffset;

                    diffInputs[outputsIdx] = ((nbOutputs > 1
                                && targets[targetsIdx] == (int)output)
                            || (nbOutputs == 1 && targets[targetsIdx] == 1))
                        ? 1.0f / nbTargetOutputs[nbTargetOutputsIdx]
                        : -1.0f / nbTargetOutputs[nbTargetOutputsIdx];
                }
                else
                    diffInputs[outputsIdx] = 0.0f;
            }
        }
    }
}

//Double
#if __CUDA_ARCH__ < 600
// Prototype declaration
__device__ __forceinline__ double atomicAdd(double* address, double val);
#endif

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__device__ __forceinline__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull =
                             (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif

__global__ void cudaDReduce_kernel(double* idata, double* odata,
                                   unsigned int size)
{
    __shared__ double sdata[256];

    // each thread loads one element from global to shared mem
    const unsigned int tid = threadIdx.x;
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size)
        sdata[tid] = idata[index];
    else
        sdata[tid] = 0.0;

    __syncthreads();

    // do reduction in shared mem
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride>>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0)
        atomicAdd(odata, sdata[0]);
}

__global__
void cudaDSetOutputTargets_kernel(int* targets,
    unsigned int* nbTargetOutputs,
    double* diffInputs,
    unsigned int nbOutputs,
    unsigned int outputsHeight,
    unsigned int outputsWidth,
    unsigned int batchSize)
{
    const unsigned int batchOutputOffset = blockIdx.z * nbOutputs
                                           * outputsHeight * outputsWidth;
    const unsigned int batchTargetOffset = blockIdx.z
                                           * outputsHeight * outputsWidth;
    const unsigned int batchNbTargetOutputsOffset = blockIdx.z
                                           * ((nbOutputs > 1) ? nbOutputs : 2);

    for (unsigned int output = blockIdx.x; output < nbOutputs;
        output += gridDim.x) {                                
        for (unsigned int oy = threadIdx.y; oy < outputsHeight;
                oy += blockDim.y) {
            for (unsigned int ox = threadIdx.x; ox < outputsWidth;
                    ox += blockDim.x)
            {
                const unsigned int outputsIdx
                    = ox + (oy + output * outputsHeight) * outputsWidth
                        + batchOutputOffset;
                const unsigned int targetsIdx = ox + oy * outputsWidth
                    + batchTargetOffset;

                if (targets[targetsIdx] >= 0) {
                    const unsigned int nbTargetOutputsIdx = targets[targetsIdx]
                        + batchNbTargetOutputsOffset;

                    diffInputs[outputsIdx] = ((nbOutputs > 1
                                && targets[targetsIdx] == (int)output)
                            || (nbOutputs == 1 && targets[targetsIdx] == 1))
                        ? 1.0 / nbTargetOutputs[nbTargetOutputsIdx]
                        : -1.0 / nbTargetOutputs[nbTargetOutputsIdx];
                }
                else
                    diffInputs[outputsIdx] = 0.0;
            }
        }
    }
}

//Half
#if __CUDA_ARCH__ >= 530 && (!defined(CUDART_VERSION) || CUDART_VERSION < 10020)
// __habs was introduced with CUDA 10.2
__device__ __half __habs(__half x) {
    return __float2half(fabs(__half2float(x)));
}
#endif

__global__
void cudaHApplyLoss_kernel(__half* lossMem,
    __half* outputs,
    __half* diffInputs,
    unsigned int nbOutputs,
    unsigned int outputsHeight,
    unsigned int outputsWidth,
    unsigned int batchSize,
    __half targetVal,
    __half defaultVal)
{
    const unsigned int batchOutputOffset = blockIdx.z * nbOutputs
                                           * outputsHeight * outputsWidth;
    const unsigned int batchTargetOffset = blockIdx.z
                                           * outputsHeight * outputsWidth;
    const unsigned int batchNbTargetOutputsOffset = blockIdx.z
                                           * ((nbOutputs > 1) ? nbOutputs : 2);

    for (unsigned int output = blockIdx.x; output < nbOutputs;
        output += gridDim.x) {                                
        for (unsigned int oy = threadIdx.y; oy < outputsHeight;
                oy += blockDim.y) {
            for (unsigned int ox = threadIdx.x; ox < outputsWidth;
                    ox += blockDim.x)
            {
                const unsigned int outputsIdx
                    = ox + (oy + output * outputsHeight) * outputsWidth
                        + batchOutputOffset;
                const unsigned int targetsIdx = ox + oy * outputsWidth
                    + batchTargetOffset;

#if __CUDA_ARCH__ >= 530
                if (__heq(diffInputs[outputsIdx], __float2half(0.0f))) {
                    const __half val
                        = (__hgt(diffInputs[outputsIdx], __float2half(0.0f)))
                                            ? targetVal : defaultVal;

                    const __half error = __hmul(
                        __hsub(val, outputs[outputsIdx]),
                        __habs(diffInputs[outputsIdx]));
                    lossMem[outputsIdx] = __hmul(error, error);
                    diffInputs[outputsIdx] = error;
                }
                else
                    lossMem[outputsIdx] = __float2half(0.0f);
#else
                if (__half2float(diffInputs[outputsIdx]) != 0.0f) {
                    const __half val
                        = (__half2float(diffInputs[outputsIdx]) > 0.0f)
                                            ? targetVal : defaultVal;

                    const float error = (__half2float(val)
                        - __half2float(outputs[outputsIdx]))
                            * abs(__half2float(diffInputs[outputsIdx]));
                    lossMem[outputsIdx] = __float2half(error * error);
                    diffInputs[outputsIdx] = __float2half(error);
                }
                else
                    lossMem[outputsIdx] = __float2half(0.0f);
#endif
            }
        }
    }
}

//Float
__global__
void cudaSApplyLoss_kernel(float* lossMem,
    float* outputs,
    float* diffInputs,
    unsigned int nbOutputs,
    unsigned int outputsHeight,
    unsigned int outputsWidth,
    unsigned int batchSize,
    float targetVal,
    float defaultVal)
{
    const unsigned int batchOutputOffset = blockIdx.z * nbOutputs
                                           * outputsHeight * outputsWidth;
    const unsigned int batchTargetOffset = blockIdx.z
                                           * outputsHeight * outputsWidth;
    const unsigned int batchNbTargetOutputsOffset = blockIdx.z
                                           * ((nbOutputs > 1) ? nbOutputs : 2);

    for (unsigned int output = blockIdx.x; output < nbOutputs;
        output += gridDim.x) {                                
        for (unsigned int oy = threadIdx.y; oy < outputsHeight;
                oy += blockDim.y) {
            for (unsigned int ox = threadIdx.x; ox < outputsWidth;
                    ox += blockDim.x)
            {
                const unsigned int outputsIdx
                    = ox + (oy + output * outputsHeight) * outputsWidth
                        + batchOutputOffset;
                const unsigned int targetsIdx = ox + oy * outputsWidth
                    + batchTargetOffset;

                if (diffInputs[outputsIdx] != 0.0f) {
                    const float val = (diffInputs[outputsIdx] > 0.0f)
                                            ? targetVal : defaultVal;
                    const float error = (val - outputs[outputsIdx])
                                            * abs(diffInputs[outputsIdx]);
                    diffInputs[outputsIdx] = error;
                    lossMem[outputsIdx] = error * error;
                }
                else
                    lossMem[outputsIdx] = 0.0f;
            }
        }
    }
}

//Double
__global__
void cudaDApplyLoss_kernel(double* lossMem,
    double* outputs,
    double* diffInputs,
    unsigned int nbOutputs,
    unsigned int outputsHeight,
    unsigned int outputsWidth,
    unsigned int batchSize,
    double targetVal,
    double defaultVal)
{
    const unsigned int batchOutputOffset = blockIdx.z * nbOutputs
                                           * outputsHeight * outputsWidth;
    const unsigned int batchTargetOffset = blockIdx.z
                                           * outputsHeight * outputsWidth;
    const unsigned int batchNbTargetOutputsOffset = blockIdx.z
                                           * ((nbOutputs > 1) ? nbOutputs : 2);

    for (unsigned int output = blockIdx.x; output < nbOutputs;
        output += gridDim.x) {                                
        for (unsigned int oy = threadIdx.y; oy < outputsHeight;
                oy += blockDim.y) {
            for (unsigned int ox = threadIdx.x; ox < outputsWidth;
                    ox += blockDim.x)
            {
                const unsigned int outputsIdx
                    = ox + (oy + output * outputsHeight) * outputsWidth
                        + batchOutputOffset;
                const unsigned int targetsIdx = ox + oy * outputsWidth
                    + batchTargetOffset;

                if (diffInputs[outputsIdx] != 0.0) {
                    const double val = (diffInputs[outputsIdx] > 0.0)
                                            ? targetVal : defaultVal;
                    const double error = (val - outputs[outputsIdx])
                                            * abs(diffInputs[outputsIdx]);
                    diffInputs[outputsIdx] = error;
                    lossMem[outputsIdx] = error * error;
                }
                else
                    lossMem[outputsIdx] = 0.0;
            }
        }
    }
}

void N2D2::cudaPopulateNbTargetOutputs(const cudaDeviceProp& deviceProp,
    int* targets,
    unsigned int* nbTargetOutputs,
    unsigned int nbOutputs,
    unsigned int outputsHeight,
    unsigned int outputsWidth,
    unsigned int batchSize)
{
    const unsigned int maxSize = (unsigned int)deviceProp.maxThreadsPerBlock;
    const unsigned int prefMultiple = (unsigned int)deviceProp.warpSize;

    const unsigned int groupSize = (outputsWidth * outputsHeight < maxSize)
                                       ? outputsWidth * outputsHeight
                                       : maxSize;
    const unsigned int reqWidth
        = (unsigned int)ceilf((float)groupSize / (float)outputsWidth);

    const unsigned int groupWidth = min(prefMultiple, reqWidth);


    const dim3 blocksPerGrid
        = {1, 1, batchSize};
    const dim3 threadsPerBlocks = {groupWidth, groupSize / groupWidth, 1};

    int* hostErr = new int[3];
    hostErr[0] = 0;
    hostErr[1] = 0;
    hostErr[2] = 0;

    int* devErr;
    CHECK_CUDA_STATUS(cudaMalloc(&devErr, 3 * sizeof(int)));
    CHECK_CUDA_STATUS(cudaMemcpy(devErr,
                                 hostErr,
                                 3 * sizeof(int),
                                 cudaMemcpyHostToDevice));

    cudaPopulateNbTargetOutputs_kernel<<<blocksPerGrid, threadsPerBlocks>>>
        (targets,
           nbTargetOutputs,
           nbOutputs,
           outputsHeight,
           outputsWidth,
           batchSize,
           devErr);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());

    CHECK_CUDA_STATUS(cudaMemcpy(hostErr,
                                 devErr,
                                 3 * sizeof(int),
                                 cudaMemcpyDeviceToHost));

    CHECK_CUDA_STATUS(cudaFree(devErr));

    if (hostErr[0] > 0) {
        std::stringstream errorMsg;
        errorMsg << "Cell_Frame_CUDA<T>:: "
            "setOutputTargets(): "
            "output target (" << hostErr[0] << ") out "
            "of range [0," << (nbOutputs
                            - (nbOutputs > 1)) << "].";

        delete[] hostErr;
        throw std::domain_error(errorMsg.str());
    }

    delete[] hostErr;
}

//Half
void N2D2::cudaHSetOutputTargets(const cudaDeviceProp& deviceProp,
    int* targets,
    unsigned int* nbTargetOutputs,
    half_float::half* diffInputs,
    unsigned int nbOutputs,
    unsigned int outputsHeight,
    unsigned int outputsWidth,
    unsigned int batchSize)
{
    const unsigned int maxSize = (unsigned int)deviceProp.maxThreadsPerBlock;
    const unsigned int prefMultiple = (unsigned int)deviceProp.warpSize;

    const unsigned int groupSize = (outputsWidth * outputsHeight < maxSize)
                                       ? outputsWidth * outputsHeight
                                       : maxSize;
    const unsigned int reqWidth
        = (unsigned int)ceilf((float)groupSize / (float)outputsWidth);

    const unsigned int groupWidth = min(prefMultiple, reqWidth);

    const dim3 blocksPerGrid = {nbOutputs, 1, batchSize};
    const dim3 threadsPerBlocks = {groupWidth, groupSize / groupWidth, 1};

    cudaHSetOutputTargets_kernel<<<blocksPerGrid, threadsPerBlocks>>>
        (targets,
           nbTargetOutputs,
           reinterpret_cast<__half*>(diffInputs),
           nbOutputs,
           outputsHeight,
           outputsWidth,
           batchSize);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}

//Float
void N2D2::cudaSSetOutputTargets(const cudaDeviceProp& deviceProp,
    int* targets,
    unsigned int* nbTargetOutputs,
    float* diffInputs,
    unsigned int nbOutputs,
    unsigned int outputsHeight,
    unsigned int outputsWidth,
    unsigned int batchSize)
{
    const unsigned int maxSize = (unsigned int)deviceProp.maxThreadsPerBlock;
    const unsigned int prefMultiple = (unsigned int)deviceProp.warpSize;

    const unsigned int groupSize = (outputsWidth * outputsHeight < maxSize)
                                       ? outputsWidth * outputsHeight
                                       : maxSize;
    const unsigned int reqWidth
        = (unsigned int)ceilf((float)groupSize / (float)outputsWidth);

    const unsigned int groupWidth = min(prefMultiple, reqWidth);

    const dim3 blocksPerGrid = {nbOutputs, 1, batchSize};
    const dim3 threadsPerBlocks = {groupWidth, groupSize / groupWidth, 1};

    cudaSSetOutputTargets_kernel<<<blocksPerGrid, threadsPerBlocks>>>
        (targets,
           nbTargetOutputs,
           diffInputs,
           nbOutputs,
           outputsHeight,
           outputsWidth,
           batchSize);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}

//Double
void N2D2::cudaDSetOutputTargets(const cudaDeviceProp& deviceProp,
    int* targets,
    unsigned int* nbTargetOutputs,
    double* diffInputs,
    unsigned int nbOutputs,
    unsigned int outputsHeight,
    unsigned int outputsWidth,
    unsigned int batchSize)
{
    const unsigned int maxSize = (unsigned int)deviceProp.maxThreadsPerBlock;
    const unsigned int prefMultiple = (unsigned int)deviceProp.warpSize;

    const unsigned int groupSize = (outputsWidth * outputsHeight < maxSize)
                                       ? outputsWidth * outputsHeight
                                       : maxSize;
    const unsigned int reqWidth
        = (unsigned int)ceilf((float)groupSize / (float)outputsWidth);

    const unsigned int groupWidth = min(prefMultiple, reqWidth);

    const dim3 blocksPerGrid = {nbOutputs, 1, batchSize};
    const dim3 threadsPerBlocks = {groupWidth, groupSize / groupWidth, 1};

    cudaDSetOutputTargets_kernel<<<blocksPerGrid, threadsPerBlocks>>>
        (targets,
           nbTargetOutputs,
           diffInputs,
           nbOutputs,
           outputsHeight,
           outputsWidth,
           batchSize);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}

//Half
double N2D2::cudaHApplyLoss(const cudaDeviceProp& deviceProp,
    half_float::half* lossMem,
    half_float::half* outputs,
    half_float::half* diffInputs,
    unsigned int nbOutputs,
    unsigned int outputsHeight,
    unsigned int outputsWidth,
    unsigned int batchSize,
    half_float::half targetVal,
    half_float::half defaultVal)
{
    const unsigned int maxSize = (unsigned int)deviceProp.maxThreadsPerBlock;
    const unsigned int prefMultiple = (unsigned int)deviceProp.warpSize;

    const unsigned int groupSize = (outputsWidth * outputsHeight < maxSize)
                                       ? outputsWidth * outputsHeight
                                       : maxSize;
    const unsigned int reqWidth
        = (unsigned int)ceilf((float)groupSize / (float)outputsWidth);

    const unsigned int groupWidth = min(prefMultiple, reqWidth);

    const dim3 blocksPerGrid = {nbOutputs, 1, batchSize};
    const dim3 threadsPerBlocks = {groupWidth, groupSize / groupWidth, 1};

    cudaHApplyLoss_kernel<<<blocksPerGrid, threadsPerBlocks>>>
        (reinterpret_cast<__half*>(lossMem),
           reinterpret_cast<__half*>(outputs),
           reinterpret_cast<__half*>(diffInputs),
           nbOutputs,
           outputsHeight,
           outputsWidth,
           batchSize,
           reinterpret_cast<__half&>(targetVal),
           reinterpret_cast<__half&>(defaultVal));
    CHECK_CUDA_STATUS(cudaPeekAtLastError());

    const unsigned int size = outputsWidth * outputsHeight * nbOutputs
                                * batchSize;
    cudaHReduce_kernel<<<(size + 255) / 256, 256>>>(
        reinterpret_cast<__half*>(lossMem),
        reinterpret_cast<__half*>(lossMem),
        size);

    half_float::half hostLoss;
    CHECK_CUDA_STATUS(cudaMemcpy(&hostLoss,
                                 lossMem,
                                 sizeof(half_float::half),
                                 cudaMemcpyDeviceToHost));

    return (float)hostLoss;
}

//Float
double N2D2::cudaSApplyLoss(const cudaDeviceProp& deviceProp,
    float* lossMem,
    float* outputs,
    float* diffInputs,
    unsigned int nbOutputs,
    unsigned int outputsHeight,
    unsigned int outputsWidth,
    unsigned int batchSize,
    float targetVal,
    float defaultVal)
{
    const unsigned int maxSize = (unsigned int)deviceProp.maxThreadsPerBlock;
    const unsigned int prefMultiple = (unsigned int)deviceProp.warpSize;

    const unsigned int groupSize = (outputsWidth * outputsHeight < maxSize)
                                       ? outputsWidth * outputsHeight
                                       : maxSize;
    const unsigned int reqWidth
        = (unsigned int)ceilf((float)groupSize / (float)outputsWidth);

    const unsigned int groupWidth = min(prefMultiple, reqWidth);

    const dim3 blocksPerGrid = {nbOutputs, 1, batchSize};
    const dim3 threadsPerBlocks = {groupWidth, groupSize / groupWidth, 1};

    cudaSApplyLoss_kernel<<<blocksPerGrid, threadsPerBlocks>>>
        (lossMem,
           outputs,
           diffInputs,
           nbOutputs,
           outputsHeight,
           outputsWidth,
           batchSize,
           targetVal,
           defaultVal);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());

    const unsigned int size = outputsWidth * outputsHeight * nbOutputs
                                * batchSize;
    cudaSReduce_kernel<<<(size + 255) / 256, 256>>>(lossMem, lossMem, size);

    float hostLoss;
    CHECK_CUDA_STATUS(cudaMemcpy(&hostLoss,
                                 lossMem,
                                 sizeof(float),
                                 cudaMemcpyDeviceToHost));

    return hostLoss;
}

//Double
double N2D2::cudaDApplyLoss(const cudaDeviceProp& deviceProp,
    double* lossMem,
    double* outputs,
    double* diffInputs,
    unsigned int nbOutputs,
    unsigned int outputsHeight,
    unsigned int outputsWidth,
    unsigned int batchSize,
    double targetVal,
    double defaultVal)
{
    const unsigned int maxSize = (unsigned int)deviceProp.maxThreadsPerBlock;
    const unsigned int prefMultiple = (unsigned int)deviceProp.warpSize;

    const unsigned int groupSize = (outputsWidth * outputsHeight < maxSize)
                                       ? outputsWidth * outputsHeight
                                       : maxSize;
    const unsigned int reqWidth
        = (unsigned int)ceilf((float)groupSize / (float)outputsWidth);

    const unsigned int groupWidth = min(prefMultiple, reqWidth);

    const dim3 blocksPerGrid = {nbOutputs, 1, batchSize};
    const dim3 threadsPerBlocks = {groupWidth, groupSize / groupWidth, 1};

    cudaDApplyLoss_kernel<<<blocksPerGrid, threadsPerBlocks>>>
        (lossMem,
           outputs,
           diffInputs,
           nbOutputs,
           outputsHeight,
           outputsWidth,
           batchSize,
           targetVal,
           defaultVal);
    CHECK_CUDA_STATUS(cudaPeekAtLastError());

    const unsigned int size = outputsWidth * outputsHeight * nbOutputs
                                * batchSize;
    cudaDReduce_kernel<<<(size + 255) / 256, 256>>>(lossMem, lossMem, size);

    double hostLoss;
    CHECK_CUDA_STATUS(cudaMemcpy(&hostLoss,
                                 lossMem,
                                 sizeof(double),
                                 cudaMemcpyDeviceToHost));

    return hostLoss;
}
