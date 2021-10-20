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

#include "Cell/DistanceCell_Frame_CUDA_Kernels.hpp"

////Forward kernels
template <class T>
__global__
void cudaDistanceL2Forward_kernel(unsigned int size, // Batch * nb_class 
                                    unsigned int nb_class, 
                                    unsigned int feat_dim,
                                    const T *inputs,
                                    const T *means,
                                    //const T *sigma,
                                    T *dist,
                                    T *outputs)
{
    const unsigned int start_index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int index = start_index; index < size; index += stride)
    {
        const int i = index / nb_class; // i -> index over the Batch
        const int j = index % nb_class; // j -> index over the number of classes
        T t = 0;
        for (int k = 0; k < feat_dim; ++k) // k -> index over the feature dimension
        {
            T d = inputs[i*feat_dim + k] - means[j*feat_dim + k];
            t += d*d;
        }
        dist[index] = t;
        //outputs[index] = T(-0.5) * t / (max(sigma[j], T(0)) + T(0.0001));
        outputs[index] = T(-0.5) * t; // case sigma = 1
    }
}

template <class T>
__global__ 
void cudaMargin_kernel(unsigned int size, // Batch * nb_class 
                        const T *label, 
                        const T margin,
                        T* outputs)
{
    const unsigned int start_index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int index = start_index; index < size; index += stride) {
        if (label[index] > 0.0f)
            outputs[index] += outputs[index]*margin;
    }
}

////Backward kernels
template <class T>
__global__
void cudaDistanceL2Backward_input_kernel(unsigned int size, // feat_dim * Batch
                                            unsigned int nb_class, 
                                            unsigned int feat_dim,
                                            const T margin,
                                            const T centercoef,
                                            const T *label,
                                            const T *inputs,
                                            const T *means, 
                                            const T *diffInputs, 
                                            T *diffOutputs) {

    const unsigned int start_index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int index = start_index; index < size; index += stride) {
        const int i = index / feat_dim; // i -> index over the Batch
        const int k = index % feat_dim; // k -> index over the feature dimension
        T t = 0;
        for (int j = 0; j < nb_class; ++j) // j -> index over the number of classes
        {
            if (label[i*nb_class + j] > 0.0f)
                t += (means[j*feat_dim + k] - inputs[index]) * (((T)1.0 + margin) * diffInputs[i*nb_class + j] - centercoef);
            else
                t += (means[j*feat_dim + k] - inputs[index]) * diffInputs[i*nb_class + j];
        }
        diffOutputs[index] = t;
    }
}

template <class T>
__global__
void cudaDistanceL2Backward_mean_kernel(unsigned int size, // feat_dim * nb_class
                                            unsigned int nb_class, 
                                            unsigned int feat_dim,
                                            unsigned int batchsize,
                                            const T margin,
                                            const T centercoef,
                                            const T *label,
                                            const T *inputs,
                                            const T *means, 
                                            const T *diffInputs, 
                                            T *diffMeans) {

    const unsigned int start_index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int index = start_index; index < size; index += stride) {
        const int j = index / feat_dim; // j -> index over the number of classes
        const int k = index % feat_dim; // k -> index over the feature dimension
        T t = 0;
        for (int i = 0; i < batchsize; ++i) // i -> index over the Batch
        {
            if (label[i*nb_class + j] > 0.0f)
                t += (inputs[i*feat_dim + k] - means[index]) * (((T)1.0 + margin) * diffInputs[i*nb_class + j] - centercoef);
            else
                t += (inputs[i*feat_dim + k] - means[index]) * diffInputs[i*nb_class + j];
        }
        diffMeans[index] = t;
    }
}


// Forward and Backward callable functions

namespace N2D2 {

template <class T>
void cudaDistanceL2Forward(unsigned int size, // Batch * nb_class 
                            unsigned int nb_class, 
                            unsigned int feat_dim,
                            const T *inputs,
                            const T *means,
                            //const T *sigma,
                            T *dist,
                            T *outputs)
{

    cudaDistanceL2Forward_kernel<<<(size + 255) / 256, 256>>>(size, nb_class, feat_dim,
        reinterpret_cast<const typename Cuda::cuda_type<T>::type*>(inputs), 
        reinterpret_cast<const typename Cuda::cuda_type<T>::type*>(means),
        //reinterpret_cast<typename Cuda::cuda_type<T>::type*>(sigma),
        reinterpret_cast<typename Cuda::cuda_type<T>::type*>(dist),
        reinterpret_cast<typename Cuda::cuda_type<T>::type*>(outputs));
    
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}

template <class T>
void cudaMargin(unsigned int size, 
                const T* label, 
                const T margin,
                T* outputs)
{

    cudaMargin_kernel<<<(size + 255) / 256, 256>>>(size,  
        reinterpret_cast<const typename Cuda::cuda_type<T>::type*>(label), 
        reinterpret_cast<const typename Cuda::cuda_type<T>::type&>(margin),
        reinterpret_cast<typename Cuda::cuda_type<T>::type*>(outputs));
    
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}


template <class T>
void cudaDistanceL2Backward_input(unsigned int size,
                                    unsigned int nb_class, 
                                    unsigned int feat_dim,
                                    const T margin,
                                    const T centercoef,
                                    const T *label,
                                    const T *inputs,
                                    const T *means, 
                                    const T *diffInputs, 
                                    T *diffOutputs) 
{
    cudaDistanceL2Backward_input_kernel<<<(size + 255) / 256, 256>>>(size, nb_class, feat_dim,
        reinterpret_cast<const typename Cuda::cuda_type<T>::type&>(margin),
        reinterpret_cast<const typename Cuda::cuda_type<T>::type&>(centercoef),
        reinterpret_cast<const typename Cuda::cuda_type<T>::type*>(label),
        reinterpret_cast<const typename Cuda::cuda_type<T>::type*>(inputs),
        reinterpret_cast<const typename Cuda::cuda_type<T>::type*>(means), 
        reinterpret_cast<const typename Cuda::cuda_type<T>::type*>(diffInputs),
        reinterpret_cast<typename Cuda::cuda_type<T>::type*>(diffOutputs));
    
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}

template <class T>
void cudaDistanceL2Backward_mean(unsigned int size,
                                    unsigned int nb_class, 
                                    unsigned int feat_dim,
                                    unsigned int batchsize,
                                    const T margin,
                                    const T centercoef,
                                    const T *label,
                                    const T *inputs,
                                    const T *means, 
                                    const T *diffInputs, 
                                    T *diffMeans) 
{
    cudaDistanceL2Backward_mean_kernel<<<(size + 255) / 256, 256>>>(size, nb_class, feat_dim, batchsize,
        reinterpret_cast<const typename Cuda::cuda_type<T>::type&>(margin),
        reinterpret_cast<const typename Cuda::cuda_type<T>::type&>(centercoef),
        reinterpret_cast<const typename Cuda::cuda_type<T>::type*>(label),
        reinterpret_cast<const typename Cuda::cuda_type<T>::type*>(inputs),
        reinterpret_cast<const typename Cuda::cuda_type<T>::type*>(means), 
        reinterpret_cast<const typename Cuda::cuda_type<T>::type*>(diffInputs),
        reinterpret_cast<typename Cuda::cuda_type<T>::type*>(diffMeans));
    
    CHECK_CUDA_STATUS(cudaPeekAtLastError());
}


template void cudaDistanceL2Forward(unsigned int size,
                                    unsigned int nb_class, 
                                    unsigned int feat_dim,
                                    const float *inputs,
                                    const float *means,
                                    //const float *sigma,
                                    float *dist,
                                    float *outputs);

template void cudaMargin(unsigned int size, 
                            const float *label, 
                            const float margin,
                            float* outputs);
                           
template void cudaDistanceL2Backward_input(unsigned int size,
                                unsigned int nb_class, 
                                unsigned int feat_dim,
                                const float margin,
                                const float centercoef,
                                const float *label,
                                const float *inputs,
                                const float *means, 
                                const float *diffInputs, 
                                float *diffOutputs);

template void cudaDistanceL2Backward_mean(unsigned int size,
                                    unsigned int nb_class, 
                                    unsigned int feat_dim,
                                    unsigned int batchsize,
                                    const float margin,
                                    const float centercoef,
                                    const float *label,
                                    const float *inputs,
                                    const float *means, 
                                    const float *diffInputs, 
                                    float *diffMeans);                                
}
