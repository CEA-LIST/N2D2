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

#ifndef N2D2_CUDATENSOR_H
#define N2D2_CUDATENSOR_H

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include <thrust/device_ptr.h>
#include <thrust/fill.h>

#include "CudaUtils.hpp"
#include "containers/Tensor.hpp"

namespace N2D2 {
template <typename T> void thrust_fill(T* devData, size_t size, T value);

template <typename T> class CudaTensor : public Tensor<T> {
public:
    using Tensor<T>::size;
    using Tensor<T>::synchronizeDToH;
    using Tensor<T>::synchronizeHToD;

    CudaTensor();
    CudaTensor(Tensor<T>* base);
    CudaTensor(const CudaTensor<T>& tensor);
    CudaTensor(std::initializer_list<size_t> dims);
    CudaTensor(const std::vector<size_t>& dims);
    inline void reserve(std::initializer_list<size_t> dims);
    inline void reserve(const std::vector<size_t>& dims);
    inline void resize(std::initializer_list<size_t> dims,
                       const T& value = T());
    inline void resize(const std::vector<size_t>& dims,
                       const T& value = T());
    inline void assign(std::initializer_list<size_t> dims,
                       const T& value);
    inline void assign(const std::vector<size_t>& dims,
                       const T& value);
    inline void push_back(const std::vector<T>& vec);
    inline void push_back(const Tensor<T>& frame);
    inline void clear();
    inline CudaTensor<T> operator[](size_t i);
    inline const CudaTensor<T> operator[](size_t i) const;
    CudaTensor<T>& operator=(const Tensor<T>& tensor);

    /** Synchronize Device To Host */
    void synchronizeDToH() const;
    void synchronizeDToH(std::initializer_list<size_t> indexAndLength)
        const;

    /** Synchronize Host To Device */
    void synchronizeHToD() const;
    void synchronizeHToD(std::initializer_list<size_t> indexAndLength)
        const;

    /** Synchronize Device To Host-based data  */
    void synchronizeDToHBased() const;

    /** Synchronize Host-based data To Device */
    void synchronizeHBasedToD() const;

    /** Synchronize Device-based data To Host  */
    void synchronizeDBasedToH() const;

    /** Synchronize Host data To Device-based */
    void synchronizeHToDBased() const;

    void setDevicePtr(T* dataDevice)
    {
        mDataDevice = dataDevice;
    }
    T* getDevicePtr()
    {
        return mDataDevice;
    }
    cudnnTensorDescriptor_t& getCudnnTensorDesc()
    {
        return mTensor;
    }

    ~CudaTensor();

protected:
    CudaTensor(const std::vector<size_t>& dims,
               const std::shared_ptr<std::vector<T> >& data,
               const std::shared_ptr<bool>& valid,
               size_t dataOffset,
               size_t size,
               size_t sizeM1,
               T* dataDevice,
               bool hostBased);
    void setCudnnTensor();
    template <typename U>
    void syncFill(typename std::enable_if<std::is_pod<U>::value, U>::type value);
    template <typename U>
    void syncFill(typename std::enable_if<!std::is_pod<U>::value, U>::type value);

    using Tensor<T>::mDims;
    using Tensor<T>::mData;
    using Tensor<T>::mValid;
    using Tensor<T>::mDataOffset;
    using Tensor<T>::mSize;
    using Tensor<T>::mSizeM1;

    cudnnTensorDescriptor_t mTensor;
    T* mDataDevice;
    const bool mDataDeviceOwner;
    const bool mHostBased;
};
}

template <typename T>
N2D2::CudaTensor<T>::CudaTensor()
    : Tensor<T>(),
      mDataDevice(NULL),
      mDataDeviceOwner(true),
      mHostBased(false)
{
    // ctor
    CHECK_CUDNN_STATUS(cudnnCreateTensorDescriptor(&mTensor));
}

template <typename T>
N2D2::CudaTensor<T>::CudaTensor(Tensor<T>* base)
    : Tensor<T>(*base),
      mDataDevice(NULL),
      mDataDeviceOwner(true),
      mHostBased(true)
{
    // ctor
    CHECK_CUDNN_STATUS(cudnnCreateTensorDescriptor(&mTensor));
    setCudnnTensor();
}

template <typename T>
N2D2::CudaTensor<T>::CudaTensor(const CudaTensor<T>& tensor)
    : Tensor<T>(tensor.dims(),
                  tensor.begin(),
                  tensor.end()),
      mDataDevice(NULL),
      mDataDeviceOwner(true),
      mHostBased(tensor.mHostBased)
{
    // copy-ctor
    CHECK_CUDNN_STATUS(cudnnCreateTensorDescriptor(&mTensor));
    setCudnnTensor();
}

template <typename T>
N2D2::CudaTensor<T>::CudaTensor(std::initializer_list<size_t> dims)
    : Tensor<T>(dims),
      mDataDevice(NULL),
      mDataDeviceOwner(true),
      mHostBased(false)
{
    // ctor
    CHECK_CUDNN_STATUS(cudnnCreateTensorDescriptor(&mTensor));
    setCudnnTensor();
}

template <typename T>
N2D2::CudaTensor<T>::CudaTensor(const std::vector<size_t>& dims)
    : Tensor<T>(dims),
      mDataDevice(NULL),
      mDataDeviceOwner(true),
      mHostBased(false)
{
    // ctor
    CHECK_CUDNN_STATUS(cudnnCreateTensorDescriptor(&mTensor));
    setCudnnTensor();
}

template <class T>
N2D2::CudaTensor<T>::CudaTensor(const std::vector<size_t>& dims,
                                const std::shared_ptr<std::vector<T> >& data,
                                const std::shared_ptr<bool>& valid,
                                size_t dataOffset,
                                size_t size,
                                size_t sizeM1,
                                T* dataDevice,
                                bool hostBased)
    : Tensor<T>(dims, data, valid, dataOffset, size, sizeM1),
      mDataDevice(dataDevice),
      mDataDeviceOwner(false),
      mHostBased(hostBased)
{
    // ctor
    CHECK_CUDNN_STATUS(cudnnCreateTensorDescriptor(&mTensor));
    setCudnnTensor();
}

template <typename T>
void N2D2::CudaTensor<T>::reserve(std::initializer_list<size_t> dims)
{
    reserve(std::vector<size_t>(dims));
}

template <typename T>
void N2D2::CudaTensor<T>::reserve(const std::vector<size_t>& dims)
{
    assert(mDataDeviceOwner);
    Tensor<T>::reserve(dims);

    if (mDataDevice != NULL) {
        CHECK_CUDA_STATUS(cudaFree(mDataDevice));
        mDataDevice = NULL;
    }

    setCudnnTensor();
}

template <typename T>
void N2D2::CudaTensor<T>::resize(std::initializer_list<size_t> dims,
                                   const T& value)
{
    resize(std::vector<size_t>(dims), value);
}

template <typename T>
void N2D2::CudaTensor<T>::resize(const std::vector<size_t>& dims,
                                 const T& value)
{
    assert(mDataDeviceOwner);
    Tensor<T>::resize(dims, value);

    if (mDataDevice != NULL) {
        CHECK_CUDA_STATUS(cudaFree(mDataDevice));
        mDataDevice = NULL;
    }

    setCudnnTensor();
    syncFill<T>(value);
}

template <typename T>
void N2D2::CudaTensor<T>::assign(std::initializer_list<size_t> dims,
                                   const T& value)
{
    assign(std::vector<size_t>(dims), value);
}

template <typename T>
void N2D2::CudaTensor<T>::assign(const std::vector<size_t>& dims,
                                   const T& value)
{
    assert(mDataDeviceOwner);
    Tensor<T>::assign(dims, value);

    if (mDataDevice != NULL) {
        CHECK_CUDA_STATUS(cudaFree(mDataDevice));
        mDataDevice = NULL;
    }

    setCudnnTensor();
    syncFill<T>(value);
}

template <typename T>
template <typename U>
void N2D2::CudaTensor<T>::syncFill(typename std::enable_if<std::is_pod<U>::value, U>::type value)
{
    thrust_fill(mDataDevice, size(), value);
}

template <typename T>
template <typename U>
void N2D2::CudaTensor<T>::syncFill(typename std::enable_if<!std::is_pod<U>::value, U>::type /*value*/)
{
    synchronizeHToD();
}

template <typename T>
void N2D2::CudaTensor<T>::setCudnnTensor() {
    const size_t size_ = size();

    if (size_ > 0) {
/**
**      cudNN Tensors are restricted to having at least 4 dimensions :
**      When working with lower dimensionsal data, unused dimensions are set to 1.
**      Referes to the cudnnSetTensorNdDescriptor documentation from :
**      https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html
**/
        std::vector<int> dims(4,1);
        std::vector<int> strides(4,1);
        int stride = 1;

        for (unsigned int dim = 0; dim < 4; ++dim) {
            if(dim < mDims.size()) {
                dims[dim] = mDims[dim];
                strides[dim] = stride;
                stride  *= mDims[dim];
            }
        }

        for (unsigned int dim = 4; dim < mDims.size(); ++dim) {
            dims.push_back(mDims[dim]);
            strides.push_back(stride);
            stride *= mDims[dim];
        }

        std::reverse(dims.begin(), dims.end());
        std::reverse(strides.begin(), strides.end());

        if (mDataDeviceOwner)
            CHECK_CUDA_STATUS(cudaMalloc(&mDataDevice, size_ * sizeof(T)));

        CHECK_CUDNN_STATUS(cudnnSetTensorNdDescriptor(mTensor,
                                                      CUDNN_DATA_FLOAT,
                                                      /*mDims.size(),*/
                                                      dims.size(),
                                                      &dims[0],
                                                      &strides[0]));
    }
}

template <typename T>
void N2D2::CudaTensor<T>::push_back(const std::vector<T>& vec)
{
    Tensor<T>::push_back(vec);

    reserve(mDims); // Resize device tensor accordingly
    synchronizeHToD(); // Copy data into device memory
}

template <typename T>
void N2D2::CudaTensor<T>::push_back(const Tensor<T>& frame)
{
    Tensor<T>::push_back(frame);

    reserve(mDims); // Resize device tensor accordingly
    synchronizeHToD(); // Copy data into device memory
}

template <typename T> void N2D2::CudaTensor<T>::clear()
{
    assert(mDataDeviceOwner);
    Tensor<T>::clear();

    if (mDataDevice != NULL) {
        CHECK_CUDA_STATUS(cudaFree(mDataDevice));
        mDataDevice = NULL;
    }
}

template <class T>
N2D2::CudaTensor<T> N2D2::CudaTensor<T>::operator[](size_t i)
{
    assert(mDims.size() > 1);
    std::vector<size_t> newDims = mDims;
    newDims.pop_back();
    return CudaTensor<T>(newDims, mData, mValid, mDataOffset + i * mSizeM1,
                mSizeM1, (newDims.back() > 0) ? mSizeM1 / newDims.back() : 0,
                mDataDevice + i * mSizeM1, mHostBased);
}

template <class T>
const N2D2::CudaTensor<T> N2D2::CudaTensor<T>::operator[](size_t i) const
{
    assert(mDims.size() > 1);
    std::vector<size_t> newDims = mDims;
    newDims.pop_back();
    return CudaTensor<T>(newDims, mData, mValid, mDataOffset + i * mSizeM1,
                mSizeM1, (newDims.back() > 0) ? mSizeM1 / newDims.back() : 0,
                mDataDevice + i * mSizeM1, mHostBased);
}

template <class T>
N2D2::CudaTensor<T>& N2D2::CudaTensor<T>::operator=(const Tensor<T>& tensor)
{
    Tensor<T>::operator=(tensor);
    return *this;
}

/**
*   Synchronize data from device to host / Par morceau
*/
template <typename T> void N2D2::CudaTensor<T>::synchronizeDToH() const
{
    CHECK_CUDA_STATUS(cudaMemcpy(&(*mData)[mDataOffset],
                                 mDataDevice,
                                 size() * sizeof(T),
                                 cudaMemcpyDeviceToHost));
}

template <typename T>
void N2D2::CudaTensor<T>::synchronizeDToH(
    std::initializer_list<size_t> indexAndLength) const
{
    assert(indexAndLength.size() == mDims.size() + 1
           || indexAndLength.size() == 2);
    const std::vector<size_t> vec(indexAndLength.begin(), indexAndLength.end());
    size_t offset = 0;

    for (int dim = indexAndLength.size() - 2; dim >= 0; --dim) {
        assert(vec[dim] < mDims[dim]
               || (indexAndLength.size() == 2 && vec[dim] < size()));
        offset = vec[dim] + mDims[dim] * offset;
    }

    CHECK_CUDA_STATUS(cudaMemcpy(&(*mData)[mDataOffset] + offset,
                                 mDataDevice + offset,
                                 vec.back() * sizeof(T),
                                 cudaMemcpyDeviceToHost));
}

/**
*   Synchronize data from host to device / Par morceau
*/
template <typename T> void N2D2::CudaTensor<T>::synchronizeHToD() const
{
    CHECK_CUDA_STATUS(cudaMemcpy(mDataDevice,
                                 &(*mData)[mDataOffset],
                                 size() * sizeof(T),
                                 cudaMemcpyHostToDevice));
}

template <typename T>
void N2D2::CudaTensor<T>::synchronizeHToD(
    std::initializer_list<size_t> indexAndLength) const
{
    assert(indexAndLength.size() == mDims.size() + 1
           || indexAndLength.size() == 2);
    const std::vector<size_t> vec(indexAndLength.begin(), indexAndLength.end());
    size_t offset = 0;

    for (int dim = indexAndLength.size() - 2; dim >= 0; --dim) {
        assert(vec[dim] < mDims[dim]
               || (indexAndLength.size() == 2 && vec[dim] < size()));
        offset = vec[dim] + mDims[dim] * offset;
    }

    CHECK_CUDA_STATUS(cudaMemcpy(mDataDevice + offset,
                                 &(*mData)[mDataOffset] + offset,
                                 vec.back() * sizeof(T),
                                 cudaMemcpyHostToDevice));
}

/**
*   Synchronize data from valid host to device / Par morceau
*/
template <typename T> void N2D2::CudaTensor<T>::synchronizeDToHBased() const
{
    if (mHostBased)
        synchronizeDToH();
}

template <typename T> void N2D2::CudaTensor<T>::synchronizeHBasedToD() const
{
    if (mHostBased)
        synchronizeHToD();
}

template <typename T> void N2D2::CudaTensor<T>::synchronizeDBasedToH() const
{
    if (!mHostBased)
        synchronizeDToH();
}

template <typename T> void N2D2::CudaTensor<T>::synchronizeHToDBased() const
{
    if (!mHostBased)
        synchronizeHToD();
}

template <typename T> N2D2::CudaTensor<T>::~CudaTensor()
{
    CHECK_CUDNN_STATUS(cudnnDestroyTensorDescriptor(mTensor));

    if (mDataDeviceOwner && mDataDevice != NULL) {
        CHECK_CUDA_STATUS(cudaFree(mDataDevice));
        mDataDevice = NULL;
    }
}

#endif // N2D2_CUDATENSOR_H
