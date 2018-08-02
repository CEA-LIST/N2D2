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
#include "CudaContext.hpp"
#include "containers/Tensor.hpp"

namespace N2D2 {
template <typename T> void thrust_fill(T* devData, size_t size, T value);

template <typename T, typename U>
void thrust_copy(T* /*srcData*/, U* /*dstData*/, size_t /*size*/) {}

template <> void thrust_copy(double* srcData, float* dstData,
                             size_t size);
template <> void thrust_copy(double* srcData, half_float::half* dstData,
                             size_t size);
template <> void thrust_copy(float* srcData, double* dstData,
                             size_t size);
template <> void thrust_copy(float* srcData, half_float::half* dstData,
                             size_t size);
template <> void thrust_copy(half_float::half* srcData, float* dstData,
                             size_t size);
template <> void thrust_copy(half_float::half* srcData, double* dstData,
                             size_t size);

class CudaBaseTensor;
template <typename T> class CudaTensor;

class CudaBaseDeviceTensor {
public:
    inline CudaBaseDeviceTensor(const CudaBaseTensor& base);
    virtual CudaBaseDeviceTensor& operator=(
        const CudaBaseDeviceTensor& baseDevice) = 0;
    virtual const std::type_info* getType() const = 0;
    const CudaBaseTensor& getCudaTensor() const { return mCudaBaseTensor; }
    virtual ~CudaBaseDeviceTensor() {};

protected:
    const CudaBaseTensor& mCudaBaseTensor;
};

template <typename T> class CudaDeviceTensor : public CudaBaseDeviceTensor {
public:
    inline CudaDeviceTensor(const CudaBaseTensor& base, T* dataDevice = NULL);
    inline void fill(const T& value);
    T* getDevicePtr() const
    {
        return mDataDevice;
    }
    bool isOwner() const
    {
        return mDataDeviceOwner;
    }
    const cudnnTensorDescriptor_t& getCudnnTensorDesc() const
    {
        return mTensor;
    }
    const std::type_info* getType() const
    {
        return &typeid(T);
    };
    inline CudaBaseDeviceTensor& operator=(
                                        const CudaBaseDeviceTensor& baseDevice);
    virtual ~CudaDeviceTensor();

protected:
    T* mDataDevice;
    const bool mDataDeviceOwner;
    cudnnTensorDescriptor_t mTensor;
};

class CudaBaseTensor : public virtual BaseTensor {
public:
    inline CudaBaseTensor(bool hostBased);
    inline virtual void reserve(std::initializer_list<size_t> dims);
    virtual void reserve(const std::vector<size_t>& dims) = 0;
    inline virtual void resize(std::initializer_list<size_t> dims);
    virtual void resize(const std::vector<size_t>& dims) = 0;
    virtual void clear() = 0;

    /** Synchronize Device To Host */
    virtual void synchronizeDToH() const = 0;
    virtual void synchronizeDToH(std::initializer_list<size_t> indexAndLength)
        const = 0;
    virtual void synchronizeDToH(const Index& index,
                         size_t length) const = 0;

    /** Synchronize Host To Device */
    virtual void synchronizeHToD() const = 0;
    virtual void synchronizeHToD(std::initializer_list<size_t> indexAndLength)
        const = 0;
    virtual void synchronizeHToD(const Index& index,
                         size_t length) const = 0;

    /** Synchronize Device To Host-based data  */
    virtual void synchronizeDToHBased() const = 0;

    /** Synchronize Host-based data To Device */
    virtual void synchronizeHBasedToD() const = 0;

    /** Synchronize Device-based data To Host  */
    virtual void synchronizeDBasedToH() const = 0;

    /** Synchronize Host data To Device-based */
    virtual void synchronizeHToDBased() const = 0;

    virtual CudaBaseDeviceTensor& deviceTensor() = 0;
    virtual const cudnnTensorDescriptor_t& getCudnnTensorDesc() const = 0;
    virtual ~CudaBaseTensor() {};

protected:
    const bool mHostBased;
    mutable std::map<const std::type_info*,
             std::shared_ptr<CudaBaseDeviceTensor> > mDeviceTensors;

    template <class U> friend std::shared_ptr<CudaDeviceTensor<U> >
        cuda_device_tensor_cast(const CudaBaseTensor& base);
    template <class U> friend std::shared_ptr<CudaDeviceTensor<U> >
        cuda_device_tensor_cast_nocopy(const CudaBaseTensor& base);
    template <class U>
    friend CudaTensor<U> cuda_tensor_cast(const CudaBaseTensor& base);
    template <class U>
    friend CudaTensor<U> cuda_tensor_cast_nocopy(const CudaBaseTensor& base);
};

template <typename T> class CudaTensor : public Tensor<T>,
                                         public CudaBaseTensor {
public:
    using Tensor<T>::size;
    using Tensor<T>::synchronizeDToH;
    using Tensor<T>::synchronizeHToD;
    using Tensor<T>::operator=;
    using CudaBaseTensor::reserve;
    using CudaBaseTensor::resize;

    CudaTensor();
    CudaTensor(const Tensor<T>& base);
    CudaTensor(std::initializer_list<size_t> dims);
    CudaTensor(const std::vector<size_t>& dims);
    inline void reserve(const std::vector<size_t>& dims);
    inline void resize(const std::vector<size_t>& dims);
    inline void resize(std::initializer_list<size_t> dims,
                       const T& value);
    inline void resize(const std::vector<size_t>& dims,
                       const T& value);
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
    template <class U> CudaTensor<T>& operator=(const Tensor<U>& tensor);

    /** Synchronize Device To Host */
    void synchronizeDToH() const;
    void synchronizeDToH(std::initializer_list<size_t> indexAndLength)
        const;
    void synchronizeDToH(const typename Tensor<T>::Index& index,
                         size_t length) const;

    /** Synchronize Host To Device */
    void synchronizeHToD() const;
    void synchronizeHToD(std::initializer_list<size_t> indexAndLength)
        const;
    void synchronizeHToD(const typename Tensor<T>::Index& index,
                         size_t length) const;

    /** Synchronize Device To Host-based data  */
    void synchronizeDToHBased() const;

    /** Synchronize Host-based data To Device */
    void synchronizeHBasedToD() const;

    /** Synchronize Device-based data To Host  */
    void synchronizeDBasedToH() const;

    /** Synchronize Host data To Device-based */
    void synchronizeHToDBased() const;

    CudaBaseDeviceTensor& deviceTensor()
    {
        return (*mDeviceTensor);
    }
    T* getDevicePtr() const
    {
        return mDeviceTensor->getDevicePtr();
    }
    const cudnnTensorDescriptor_t& getCudnnTensorDesc() const
    {
        return mDeviceTensor->getCudnnTensorDesc();
    }

    virtual ~CudaTensor() {};

protected:
    CudaTensor(const Tensor<T>& base,
               const std::shared_ptr<CudaDeviceTensor<T> >& deviceTensor,
               bool hostBased);
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

    std::shared_ptr<CudaDeviceTensor<T> > mDeviceTensor;

    template <class U> friend std::shared_ptr<CudaDeviceTensor<U> >
        cuda_device_tensor_cast(const CudaBaseTensor& base);
    template <class U> friend std::shared_ptr<CudaDeviceTensor<U> >
        cuda_device_tensor_cast_nocopy(const CudaBaseTensor& base);
    template <class U>
    friend CudaTensor<U> cuda_tensor_cast(const CudaBaseTensor& base);
    template <class U>
    friend CudaTensor<U> cuda_tensor_cast_nocopy(const CudaBaseTensor& base);
};

template <> class CudaTensor<bool> {
private:
    CudaTensor();
};

template <class T>
std::shared_ptr<CudaDeviceTensor<T> >
cuda_device_tensor_cast(const CudaBaseTensor& base)
{
    if (base.getType() == &typeid(T))
        return static_cast<const CudaTensor<T>&>(base).mDeviceTensor;

    std::map<const std::type_info*, std::shared_ptr<CudaBaseDeviceTensor> >
        ::const_iterator it = base.mDeviceTensors.find(&typeid(T));
    std::shared_ptr<CudaDeviceTensor<T> > deviceTensor;

    if (it != base.mDeviceTensors.end()) {
        deviceTensor
            = std::static_pointer_cast<CudaDeviceTensor<T> >((*it).second);
    }
    else {
        deviceTensor = std::make_shared<CudaDeviceTensor<T> >(base);
        base.mDeviceTensors[&typeid(T)] = deviceTensor;
    }

    if (base.getType() == &typeid(float)) {
        const CudaTensor<float>& tensor
            = dynamic_cast<const CudaTensor<float>&>(base);

        thrust_copy(tensor.mDeviceTensor->getDevicePtr(),
                    deviceTensor->getDevicePtr(),
                    base.size());
    }
    else if (base.getType() == &typeid(half_float::half)) {
        const CudaTensor<half_float::half>& tensor
            = dynamic_cast<const CudaTensor<half_float::half>&>(base);

        thrust_copy(tensor.mDeviceTensor->getDevicePtr(),
                    deviceTensor->getDevicePtr(),
                    base.size());
    }
    else if (base.getType() == &typeid(double)) {
        const CudaTensor<double>& tensor
            = dynamic_cast<const CudaTensor<double>&>(base);

        thrust_copy(tensor.mDeviceTensor->getDevicePtr(),
                    deviceTensor->getDevicePtr(),
                    base.size());
    }
    else {
        throw std::runtime_error("cuda_device_tensor_cast(): "
                                 "tensor type not supported!");
    }

    return deviceTensor;
}

template <class T>
std::shared_ptr<CudaDeviceTensor<T> >
cuda_device_tensor_cast_nocopy(const CudaBaseTensor& base)
{
    if (base.getType() == &typeid(T))
        return static_cast<const CudaTensor<T>&>(base).mDeviceTensor;

    std::map<const std::type_info*, std::shared_ptr<CudaBaseDeviceTensor> >
        ::const_iterator it = base.mDeviceTensors.find(&typeid(T));
    std::shared_ptr<CudaDeviceTensor<T> > deviceTensor;

    if (it != base.mDeviceTensors.end()) {
        deviceTensor
            = std::static_pointer_cast<CudaDeviceTensor<T> >((*it).second);
    }
    else {
        deviceTensor = std::make_shared<CudaDeviceTensor<T> >(base);
        base.mDeviceTensors[&typeid(T)] = deviceTensor;
    }

    return deviceTensor;
}

template <class T>
CudaTensor<T> cuda_tensor_cast(const CudaBaseTensor& base)
{
    if (base.getType() == &typeid(T))
        return static_cast<const CudaTensor<T>&>(base);

    return CudaTensor<T>(
        tensor_cast<T>(base),
        cuda_device_tensor_cast<T>(base),
        base.mHostBased);
}

template <class T>
CudaTensor<T> cuda_tensor_cast_nocopy(const CudaBaseTensor& base)
{
    if (base.getType() == &typeid(T))
        return static_cast<const CudaTensor<T>&>(base);

    return CudaTensor<T>(
        tensor_cast_nocopy<T>(base),
        cuda_device_tensor_cast_nocopy<T>(base),
        base.mHostBased);
}
}

void N2D2::CudaBaseTensor::reserve(std::initializer_list<size_t> dims)
{
    reserve(std::vector<size_t>(dims));
}

void N2D2::CudaBaseTensor::resize(std::initializer_list<size_t> dims)
{
    resize(std::vector<size_t>(dims));
}

N2D2::CudaBaseDeviceTensor::CudaBaseDeviceTensor(const CudaBaseTensor& base)
    : mCudaBaseTensor(base)
{
    //ctor
}

template <typename T>
N2D2::CudaDeviceTensor<T>::CudaDeviceTensor(const CudaBaseTensor& base,
                                            T* dataDevice)
    : CudaBaseDeviceTensor(base),
      mDataDevice(dataDevice),
      mDataDeviceOwner(dataDevice == NULL)
{
    // ctor
    CHECK_CUDNN_STATUS(cudnnCreateTensorDescriptor(&mTensor));

    const size_t size_ = base.size();

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
            if(dim < base.nbDims()) {
                dims[dim] = base.dims()[dim];
                strides[dim] = stride;
                stride  *= base.dims()[dim];
            }
        }

        for (unsigned int dim = 4; dim < base.nbDims(); ++dim) {
            dims.push_back(base.dims()[dim]);
            strides.push_back(stride);
            stride *= base.dims()[dim];
        }

        std::reverse(dims.begin(), dims.end());
        std::reverse(strides.begin(), strides.end());

        if (mDataDeviceOwner)
            CHECK_CUDA_STATUS(cudaMalloc(&mDataDevice, size_ * sizeof(T)));

        CHECK_CUDNN_STATUS(cudnnSetTensorNdDescriptor(mTensor,
                                                      CudaContext::data_type<T>::value,
                                                      dims.size(),
                                                      &dims[0],
                                                      &strides[0]));
    }
}

template <typename T>
void N2D2::CudaDeviceTensor<T>::fill(const T& value) {
    thrust_fill(mDataDevice, mCudaBaseTensor.size(), value);
}

template <typename T>
N2D2::CudaBaseDeviceTensor& N2D2::CudaDeviceTensor<T>::operator=(
    const N2D2::CudaBaseDeviceTensor& device)
{
    if (device.getType() == &typeid(T)) {
        assert(&(device.getCudaTensor()) == &mCudaBaseTensor);
        return *this;
    }

    if (device.getType() == &typeid(float)) {
        const CudaDeviceTensor<float>& deviceTensor
            = dynamic_cast<const CudaDeviceTensor<float>&>(device);

        thrust_copy(deviceTensor.getDevicePtr(),
                    mDataDevice,
                    mCudaBaseTensor.size());
    }
    else if (device.getType() == &typeid(half_float::half)) {
        const CudaDeviceTensor<half_float::half>& deviceTensor
            = dynamic_cast<const CudaDeviceTensor<half_float::half>&>(device);

        thrust_copy(deviceTensor.getDevicePtr(),
                    mDataDevice,
                    mCudaBaseTensor.size());
    }
    else if (device.getType() == &typeid(double)) {
        const CudaDeviceTensor<double>& deviceTensor
            = dynamic_cast<const CudaDeviceTensor<double>&>(device);

        thrust_copy(deviceTensor.getDevicePtr(),
                    mDataDevice,
                    mCudaBaseTensor.size());
    }
    else {
        throw std::runtime_error("CudaDeviceTensor<T>::operator=(): "
                                 "tensor type not supported!");
    }

    return *this;
}

template <typename T> N2D2::CudaDeviceTensor<T>::~CudaDeviceTensor()
{
    if (mDataDeviceOwner && mDataDevice != NULL) {
        CHECK_CUDA_STATUS(cudaFree(mDataDevice));
        mDataDevice = NULL;
    }

    CHECK_CUDNN_STATUS(cudnnDestroyTensorDescriptor(mTensor));
}

N2D2::CudaBaseTensor::CudaBaseTensor(bool hostBased):
    mHostBased(hostBased)
{
    //ctor
}

template <typename T>
N2D2::CudaTensor<T>::CudaTensor()
    : BaseTensor(),
      Tensor<T>(),
      CudaBaseTensor(false)
{
    // ctor
    mDeviceTensor = std::make_shared<CudaDeviceTensor<T> >(*this);
}

template <typename T>
N2D2::CudaTensor<T>::CudaTensor(const Tensor<T>& base)
    : BaseTensor(base),
      Tensor<T>(base),
      CudaBaseTensor(true)
{
    // ctor
    mDeviceTensor = std::make_shared<CudaDeviceTensor<T> >(*this);
}

template <typename T>
N2D2::CudaTensor<T>::CudaTensor(std::initializer_list<size_t> dims)
    : BaseTensor(dims),
      Tensor<T>(dims),
      CudaBaseTensor(false)
{
    // ctor
    mDeviceTensor = std::make_shared<CudaDeviceTensor<T> >(*this);
}

template <typename T>
N2D2::CudaTensor<T>::CudaTensor(const std::vector<size_t>& dims)
    : BaseTensor(dims),
      Tensor<T>(dims),
      CudaBaseTensor(false)
{
    // ctor
    mDeviceTensor = std::make_shared<CudaDeviceTensor<T> >(*this);
}

template <typename T>
N2D2::CudaTensor<T>::CudaTensor(const Tensor<T>& base,
                    const std::shared_ptr<CudaDeviceTensor<T> >& deviceTensor,
                    bool hostBased)
    : BaseTensor(base),
      Tensor<T>(base),
      CudaBaseTensor(hostBased),
      mDeviceTensor(deviceTensor)
{
    // ctor
}

template <typename T>
void N2D2::CudaTensor<T>::reserve(const std::vector<size_t>& dims)
{
    assert(mDeviceTensor->isOwner());
    Tensor<T>::reserve(dims);

    mDeviceTensor = std::make_shared<CudaDeviceTensor<T> >(*this);
}

template <typename T>
void N2D2::CudaTensor<T>::resize(const std::vector<size_t>& dims)
{
    assert(mDeviceTensor->isOwner());
    Tensor<T>::resize(dims);

    mDeviceTensor = std::make_shared<CudaDeviceTensor<T> >(*this);
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
    assert(mDeviceTensor->isOwner());
    Tensor<T>::resize(dims, value);

    mDeviceTensor = std::make_shared<CudaDeviceTensor<T> >(*this);
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
    assert(mDeviceTensor->isOwner());
    Tensor<T>::assign(dims, value);

    mDeviceTensor = std::make_shared<CudaDeviceTensor<T> >(*this);
    syncFill<T>(value);
}

template <typename T>
template <typename U>
void N2D2::CudaTensor<T>::syncFill(typename std::enable_if<std::is_pod<U>::value, U>::type value)
{
    thrust_fill(mDeviceTensor->getDevicePtr(), size(), value);
}

template <typename T>
template <typename U>
void N2D2::CudaTensor<T>::syncFill(typename std::enable_if<!std::is_pod<U>::value, U>::type /*value*/)
{
    synchronizeHToD();
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
    assert(mDeviceTensor->isOwner());
    Tensor<T>::clear();

    mDeviceTensor = std::make_shared<CudaDeviceTensor<T> >(*this);
}

template <class T>
N2D2::CudaTensor<T> N2D2::CudaTensor<T>::operator[](size_t i)
{
    return CudaTensor<T>(
        Tensor<T>::operator[](i),
        std::make_shared<CudaDeviceTensor<T> >(*this,
                                mDeviceTensor->getDevicePtr() + i * mSizeM1),
        mHostBased);
}

template <class T>
const N2D2::CudaTensor<T> N2D2::CudaTensor<T>::operator[](size_t i) const
{
    return CudaTensor<T>(
        Tensor<T>::operator[](i),
        std::make_shared<CudaDeviceTensor<T> >(*this,
                                mDeviceTensor->getDevicePtr() + i * mSizeM1),
        mHostBased);
}

template <class T>
N2D2::CudaTensor<T>& N2D2::CudaTensor<T>::operator=(const Tensor<T>& tensor)
{
    Tensor<T>::operator=(tensor);
    return *this;
}

template <class T>
template <class U>
N2D2::CudaTensor<T>& N2D2::CudaTensor<T>::operator=(const Tensor<U>& tensor)
{
    Tensor<T>::operator=(tensor);
    return *this;
}

/**
*   Synchronize data from device to host / Par morceau
*/
template <typename T> void N2D2::CudaTensor<T>::synchronizeDToH() const
{
    CHECK_CUDA_STATUS(cudaMemcpy(&(*mData)()[mDataOffset],
                                 mDeviceTensor->getDevicePtr(),
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

    CHECK_CUDA_STATUS(cudaMemcpy(&(*mData)()[mDataOffset] + offset,
                                 mDeviceTensor->getDevicePtr() + offset,
                                 vec.back() * sizeof(T),
                                 cudaMemcpyDeviceToHost));
}

template <typename T>
void N2D2::CudaTensor<T>::synchronizeDToH(const typename Tensor<T>::Index& index,
                                          size_t length) const
{
    assert(mDims.size() == index.index.size());

    size_t offset = 0;

    for (int dim = mDims.size() - 1; dim >= 0; --dim) {
        assert(index[dim] < mDims[dim]);
        offset = index[dim] + mDims[dim] * offset;
    }

    CHECK_CUDA_STATUS(cudaMemcpy(&(*mData)()[mDataOffset] + offset,
                                 mDeviceTensor->getDevicePtr() + offset,
                                 length * sizeof(T),
                                 cudaMemcpyDeviceToHost));
}

/**
*   Synchronize data from host to device / Par morceau
*/
template <typename T> void N2D2::CudaTensor<T>::synchronizeHToD() const
{
    CHECK_CUDA_STATUS(cudaMemcpy(mDeviceTensor->getDevicePtr(),
                                 &(*mData)()[mDataOffset],
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

    CHECK_CUDA_STATUS(cudaMemcpy(mDeviceTensor->getDevicePtr() + offset,
                                 &(*mData)()[mDataOffset] + offset,
                                 vec.back() * sizeof(T),
                                 cudaMemcpyHostToDevice));
}

template <typename T>
void N2D2::CudaTensor<T>::synchronizeHToD(const typename Tensor<T>::Index& index,
                                          size_t length) const
{
    assert(mDims.size() == index.index.size());

    size_t offset = 0;

    for (int dim = mDims.size() - 1; dim >= 0; --dim) {
        assert(index[dim] < mDims[dim]);
        offset = index[dim] + mDims[dim] * offset;
    }

    CHECK_CUDA_STATUS(cudaMemcpy(mDeviceTensor->getDevicePtr() + offset,
                                 &(*mData)()[mDataOffset] + offset,
                                 length * sizeof(T),
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

#endif // N2D2_CUDATENSOR_H
