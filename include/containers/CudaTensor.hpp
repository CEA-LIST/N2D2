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
#include <thrust/functional.h>
#include <thrust/transform.h>

#include "CudaUtils.hpp"
#include "CudaContext.hpp"
#include "containers/Tensor.hpp"

namespace N2D2 {

enum class DeviceState { Excluded, Banned, Debanned, Ready, Connected };

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
// Copy to same type
template <> void thrust_copy(half_float::half* srcData, half_float::half* dstData,
                             size_t size);
template <> void thrust_copy(float* srcData, float* dstData,
                             size_t size);
template <> void thrust_copy(double* srcData, double* dstData,
                             size_t size);

template <typename T>
void thrust_aggregate(T* /*srcData*/, T* /*dstData*/, size_t /*size*/) {}

template <> void thrust_aggregate(half_float::half* srcData,
                                  half_float::half* dstData,
                                  size_t size);
template <> void thrust_aggregate(float* srcData,
                                  float* dstData,
                                  size_t size);
template <> void thrust_aggregate(double* srcData,
                                  double* dstData,
                                  size_t size);

std::vector<std::pair<int, int>> pairDevices(std::vector<int>& /*array*/);

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
    inline CudaDeviceTensor(const CudaBaseTensor& base,
        const std::shared_ptr<CudaDeviceTensor<T> >& dataDeviceOwner
            = std::shared_ptr<CudaDeviceTensor<T> >(),
        size_t dataDeviceOffset = 0);
    inline void fill(const T& value);
    T* getDevicePtr() const;
    T* getDevicePtr(int dev) const;
    bool isDevicePtr(int dev) const;
    void setDevicePtr(T* dataDevice)
    {
        if (mDataDeviceOwner) {
            throw std::runtime_error("setDevicePtr(): "
                                     "data device owner is not null!");
        }

        int dev;
        CHECK_CUDA_STATUS(cudaGetDevice(&dev));

        assert(dev < (int)mDataDevice.size());
        mDataDevice[dev] = dataDevice;
    }
    bool isOwner() const
    {
        return (!mDataDeviceOwner);
    }
    const cudnnTensorDescriptor_t& getCudnnTensorDesc() const;
    const std::type_info* getType() const
    {
        return &typeid(T);
    };
    inline CudaBaseDeviceTensor& operator=(
                                        const CudaBaseDeviceTensor& baseDevice);
    inline CudaDeviceTensor<T>& operator=(
                                        const CudaDeviceTensor<T>& device);

    void broadcast(int srcDev, int dstDev) const;
    void broadcastAllFrom(int srcDev) const;
    void broadcastAllFrom(int srcDev, std::vector<DeviceState> devices) const;
    void broadcastAnyTo(int dstDev) const;

    void aggregate(int srcDev, int dstDev) const;
    void aggregateAllTo(int dstDev) const;
    void aggregateAllTo(int dstDev, std::vector<DeviceState> devices) const;

    virtual ~CudaDeviceTensor();

protected:
    mutable std::vector<T*> mDataDevice;
    mutable std::vector<T*> mForeignDataDevice;
    const std::shared_ptr<CudaDeviceTensor<T> > mDataDeviceOwner;
    const size_t mDataDeviceOffset;
    mutable cudnnTensorDescriptor_t mTensor;
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

    virtual void synchronizeToH(BaseTensor& tensor) const = 0;

    virtual void broadcast(int srcDev, int dstDev) const = 0;
    virtual void broadcastAllFrom(int srcDev) const = 0;
    virtual void broadcastAllFrom(int srcDev,
                    std::vector<DeviceState> devices) const = 0;
    virtual void broadcastAnyTo(int dstDev) const = 0;

    virtual void aggregate(int srcDev, int dstDev) const = 0;
    virtual void aggregateAllTo(int dstDev) const = 0;
    virtual void aggregateAllTo(int dstDev,
                    std::vector<DeviceState> devices) const = 0;

    virtual CudaBaseDeviceTensor& deviceTensor() = 0;
    virtual const CudaBaseDeviceTensor& deviceTensor() const = 0;
    inline bool& hostBased() { return mHostBased; }
    virtual const cudnnTensorDescriptor_t& getCudnnTensorDesc() const = 0;
    virtual ~CudaBaseTensor() {};

protected:
    bool mHostBased;
    mutable std::map<const std::type_info*,
             std::shared_ptr<CudaBaseDeviceTensor> > mDeviceTensors;

    template <class U> friend std::shared_ptr<CudaDeviceTensor<U> >
        cuda_device_tensor_cast(const CudaBaseTensor& base);
    template <class U> friend std::shared_ptr<CudaDeviceTensor<U> >
        cuda_device_tensor_cast_nocopy(const CudaBaseTensor& base);
    template <class U>
    friend CudaTensor<U> cuda_tensor_cast(const BaseTensor& base);
    template <class U>
    friend CudaTensor<U> cuda_tensor_cast(const CudaBaseTensor& base);
    template <class U>
    friend CudaTensor<U> cuda_tensor_cast_nocopy(const BaseTensor& base);
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

    CudaTensor(bool hostBased = false);
    CudaTensor(const Tensor<T>& base, bool hostBased = true);
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
    inline void push_back(const T& value);
    inline void push_back(const std::vector<T>& vec);
    inline void push_back(const Tensor<T>& frame);
    inline void append(const std::vector<T>& vec);
    inline void append(const Tensor<T>& frame, int towardsDim = -1);
    inline void clear();
    inline void swap(CudaTensor<T>& tensor);
    inline CudaTensor<T> clone() const;
    inline CudaTensor<T> operator[](size_t i);
    inline const CudaTensor<T> operator[](size_t i) const;
    inline CudaTensor<T> rows(size_t j0, size_t nb);
    inline const CudaTensor<T> rows(size_t j0, size_t nb) const;
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

    void synchronizeToD(const Tensor<T>& tensor) const;
    void synchronizeToH(BaseTensor& tensor) const;
    void synchronizeToH(Tensor<T>& tensor) const;
    Tensor<T> synchronizeToH() const;

    void broadcast(int srcDev, int dstDev) const;
    void broadcastAllFrom(int srcDev) const;
    void broadcastAllFrom(int srcDev,
                std::vector<DeviceState> devices) const;
    void broadcastAnyTo(int dstDev) const;

    void aggregate(int srcDev, int dstDev) const;
    void aggregateAllTo(int dstDev) const;
    void aggregateAllTo(int dstDev,
                std::vector<DeviceState> devices) const;

    CudaDeviceTensor<T>& deviceTensor()
    {
        return (*mDeviceTensor);
    }
    const CudaDeviceTensor<T>& deviceTensor() const
    {
        return (*mDeviceTensor);
    }
    T* getDevicePtr() const
    {
        return mDeviceTensor->getDevicePtr();
    }
    void setDevicePtr(T* dataDevice)
    {
        mDeviceTensor->setDevicePtr(dataDevice);
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
    CudaTensor(const Tensor<T>& base,
               const std::shared_ptr<CudaDeviceTensor<T> >& dataDeviceOwner,
               size_t dataDeviceOffset,
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
    friend CudaTensor<U> cuda_tensor_cast(const BaseTensor& base);
    template <class U>
    friend CudaTensor<U> cuda_tensor_cast(const CudaBaseTensor& base);
    template <class U>
    friend CudaTensor<U> cuda_tensor_cast_nocopy(const BaseTensor& base);
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
CudaTensor<T> cuda_tensor_cast(const BaseTensor& base)
{
    const CudaBaseTensor& cudaBase = dynamic_cast<const CudaBaseTensor&>(base);
    return cuda_tensor_cast<T>(cudaBase);
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
CudaTensor<T> cuda_tensor_cast_nocopy(const BaseTensor& base)
{
    const CudaBaseTensor& cudaBase = dynamic_cast<const CudaBaseTensor&>(base);
    return cuda_tensor_cast_nocopy<T>(cudaBase);
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
    const std::shared_ptr<CudaDeviceTensor<T> >& dataDeviceOwner,
    size_t dataDeviceOffset)
    : CudaBaseDeviceTensor(base),
      mDataDeviceOwner(dataDeviceOwner),
      mDataDeviceOffset(dataDeviceOffset),
      mTensor(NULL)
{
    // ctor
    int count;
    CHECK_CUDA_STATUS(cudaGetDeviceCount(&count));

    mDataDevice.resize(count, NULL);
    mForeignDataDevice.resize(count, NULL);
}

template <typename T>
T* N2D2::CudaDeviceTensor<T>::getDevicePtr() const
{
    int dev;
    CHECK_CUDA_STATUS(cudaGetDevice(&dev));

    return getDevicePtr(dev);
}

template <typename T>
T* N2D2::CudaDeviceTensor<T>::getDevicePtr(int dev) const
{
    if (mDataDeviceOwner)
        return mDataDeviceOwner->getDevicePtr(dev) + mDataDeviceOffset;
    else {
        assert(dev < (int)mDataDevice.size());

        if (mDataDevice[dev] == NULL) {
            // Lazy memory allocation
            CHECK_CUDA_STATUS(cudaMalloc(&(mDataDevice[dev]),
                                         mCudaBaseTensor.size() * sizeof(T)));
        }

        return mDataDevice[dev];
    }
}

template <typename T>
bool N2D2::CudaDeviceTensor<T>::isDevicePtr(int dev) const
{
    if (mDataDeviceOwner)
        return mDataDeviceOwner->isDevicePtr(dev);
    else {
        assert(dev < (int)mDataDevice.size());
        return (mDataDevice[dev] != NULL);
    }
}

template <typename T>
const cudnnTensorDescriptor_t& N2D2::CudaDeviceTensor<T>::getCudnnTensorDesc()
    const
{
    if (mTensor == NULL) {
        CHECK_CUDNN_STATUS(cudnnCreateTensorDescriptor(&mTensor));

        const CudaBaseTensor& base = getCudaTensor();

        if (base.size() > 0) {
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

            CHECK_CUDNN_STATUS(cudnnSetTensorNdDescriptor(mTensor,
                                          CudaContext::data_type<T>::value,
                                          dims.size(),
                                          &dims[0],
                                          &strides[0]));
        }
    }

    return mTensor;
}

template <typename T>
void N2D2::CudaDeviceTensor<T>::fill(const T& value) {
    thrust_fill(getDevicePtr(), mCudaBaseTensor.size(), value);
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
                    getDevicePtr(),
                    mCudaBaseTensor.size());
    }
    else if (device.getType() == &typeid(half_float::half)) {
        const CudaDeviceTensor<half_float::half>& deviceTensor
            = dynamic_cast<const CudaDeviceTensor<half_float::half>&>(device);

        thrust_copy(deviceTensor.getDevicePtr(),
                    getDevicePtr(),
                    mCudaBaseTensor.size());
    }
    else if (device.getType() == &typeid(double)) {
        const CudaDeviceTensor<double>& deviceTensor
            = dynamic_cast<const CudaDeviceTensor<double>&>(device);

        thrust_copy(deviceTensor.getDevicePtr(),
                    getDevicePtr(),
                    mCudaBaseTensor.size());
    }
    else {
        throw std::runtime_error("CudaDeviceTensor<T>::operator=(): "
                                 "tensor type not supported!");
    }

    return *this;
}

template <typename T>
N2D2::CudaDeviceTensor<T>& N2D2::CudaDeviceTensor<T>::operator=(
    const N2D2::CudaDeviceTensor<T>& device)
{
    assert(mCudaBaseTensor.nbDims() == device.getCudaTensor().nbDims());

    for (unsigned int dim = 0; dim < mCudaBaseTensor.nbDims(); ++dim) {
        assert(mCudaBaseTensor.dims()[dim]
            == device.getCudaTensor().dims()[dim]);
    }

    if (getDevicePtr() != device.getDevicePtr()) {
        // Actual copy only if data is different
        thrust_copy(device.getDevicePtr(),
                    getDevicePtr(),
                    mCudaBaseTensor.size());
    }

    return *this;
}
template <typename T> void N2D2::CudaDeviceTensor<T>::broadcast(
    int srcDev,
    int dstDev) const
{
    CHECK_CUDA_STATUS(cudaMemcpyPeer(
        getDevicePtr(dstDev), dstDev,
        getDevicePtr(srcDev), srcDev,
        mCudaBaseTensor.size() * sizeof(T)));
}

template <typename T> void N2D2::CudaDeviceTensor<T>::broadcastAllFrom(
    int srcDev) const
{
    assert(mCudaBaseTensor.size() == 0 || isDevicePtr(srcDev));
    
    for (int dev = 0; dev < (int)mDataDevice.size(); ++dev) {
        if (dev != srcDev && isDevicePtr(dev))
            broadcast(srcDev, dev);
    }
}

template <typename T> void N2D2::CudaDeviceTensor<T>::broadcastAllFrom(
    int srcDev, std::vector<DeviceState> devices) const
{
    assert(mCudaBaseTensor.size() == 0 || isDevicePtr(srcDev));
    assert(devices.size() == mDataDevice.size());

    for (int dev = 0; dev < (int)mDataDevice.size(); ++dev) {
        if (dev != srcDev && isDevicePtr(dev)) {
            if (devices[dev] == N2D2::DeviceState::Connected
                || devices[dev] == N2D2::DeviceState::Ready) {
                    broadcast(srcDev, dev);
            }
        }
    }
}

template <typename T> void N2D2::CudaDeviceTensor<T>::broadcastAnyTo(
    int dstDev) const
{
    // TODO: this method may be optimized by selecting the source device
    // for the fastest transfer (NVLink pair for example)
    for (int dev = 0; dev < (int)mDataDevice.size(); ++dev) {
        if (dev != dstDev && isDevicePtr(dev)) {
            broadcast(dev, dstDev);
            break;
        }
    }
}

template <typename T> 
void N2D2::CudaDeviceTensor<T>::aggregate(int srcDev, int dstDev) const
{
    if (mForeignDataDevice[dstDev] == NULL) {
		// Lazy allocation
		CHECK_CUDA_STATUS(cudaMalloc(
		&mForeignDataDevice[dstDev], mCudaBaseTensor.size() * sizeof(T)));
    }

    CHECK_CUDA_STATUS(cudaMemcpyPeer(
        mForeignDataDevice[dstDev], dstDev,
        getDevicePtr(srcDev), srcDev,
        mCudaBaseTensor.size() * sizeof(T)));

    thrust_aggregate(mForeignDataDevice[dstDev], getDevicePtr(dstDev),
                     mCudaBaseTensor.size());
}

template <typename T> 
void N2D2::CudaDeviceTensor<T>::aggregateAllTo(int dstDev) const
{
    assert(mCudaBaseTensor.size() == 0 || isDevicePtr(dstDev));

    int currentDev;
    CHECK_CUDA_STATUS(cudaGetDevice(&currentDev));

    if (currentDev != dstDev)
        CHECK_CUDA_STATUS(cudaSetDevice(dstDev));

    for (int dev = 0; dev < (int)mDataDevice.size(); ++dev) {
        if (dev != dstDev && isDevicePtr(dev)) {
            aggregate(dev, dstDev); 
        }
    }

    if (currentDev != dstDev)
        CHECK_CUDA_STATUS(cudaSetDevice(currentDev));
}

template <typename T> 
void N2D2::CudaDeviceTensor<T>::aggregateAllTo(int dstDev, 
                                std::vector<DeviceState> devices) const
{
    assert(mCudaBaseTensor.size() == 0 || isDevicePtr(dstDev));
    assert(devices.size() == mDataDevice.size());

    int currentDev;
    CHECK_CUDA_STATUS(cudaGetDevice(&currentDev));

    if (currentDev != dstDev)
        CHECK_CUDA_STATUS(cudaSetDevice(dstDev));

    for (int dev = 0; dev < (int)mDataDevice.size(); ++dev) {
        if (dev != dstDev && isDevicePtr(dev)) {
            if (devices[dev] == N2D2::DeviceState::Connected) {
                aggregate(dev, dstDev);
            }
        }
    }

    if (currentDev != dstDev)
        CHECK_CUDA_STATUS(cudaSetDevice(currentDev));
}

/* 
// Prototype to improve the aggregateAllTo function 
// by performing pairwise aggregations (complexity in O(log(n))
// However the function decreases the accuracy of the whole network
// TODO : find the issue and implement this function
// The function uses pairDevices, defined in CudaTensor.cpp
template <typename T> 
void N2D2::CudaDeviceTensor<T>::aggregateAllTo(int dstDev, 
                                std::vector<DeviceState> devices) const
{
    assert(mCudaBaseTensor.size() == 0 || isDevicePtr(dstDev));
    assert(devices.size() == mDataDevice.size());

    int currentDev;
    CHECK_CUDA_STATUS(cudaGetDevice(&currentDev));

    std::vector<int> availableDevices;

    for (int dev = 0; dev < (int)mDataDevice.size(); ++dev) {
        if (devices[dev] == N2D2::DeviceState::Connected) {
            availableDevices.push_back(dev);
        }
    }

    while (availableDevices.size() > 1) {
        std::vector<std::pair<int, int>> pairDev = pairDevices(availableDevices);
        
#pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < (int)pairDev.size(); ++i) {
            std::pair<int, int> p = pairDev[i];

            if (p.first == dstDev) {
                CHECK_CUDA_STATUS(cudaSetDevice(p.first));
                aggregate(p.second, p.first);
                availableDevices.push_back(p.first);
            } else if (p.first == -1) {
                availableDevices.push_back(p.second);
            } else {
                CHECK_CUDA_STATUS(cudaSetDevice(p.second));
                aggregate(p.first, p.second);
                availableDevices.push_back(p.second);
            }

        }
    }

    // Mandatory to resynchronize the master device
    CHECK_CUDA_STATUS(cudaSetDevice(currentDev));
}
*/

template <typename T> N2D2::CudaDeviceTensor<T>::~CudaDeviceTensor()
{
    for (size_t dev = 0; dev < mDataDevice.size(); ++dev) {
        if (mDataDevice[dev] != NULL) {
            cudaSetDevice(dev);
            cudaFree(mDataDevice[dev]);
            mDataDevice[dev] = NULL;
        }

        if (mForeignDataDevice[dev] != NULL) {
            // BUG: current device may not be the one on which memory was allocated!
            cudaFree(mForeignDataDevice[dev]);
            mForeignDataDevice[dev] = NULL;
        }
    }


    if (mTensor != NULL)
        cudnnDestroyTensorDescriptor(mTensor);
}

N2D2::CudaBaseTensor::CudaBaseTensor(bool hostBased):
    mHostBased(hostBased)
{
    //ctor
}

template <typename T>
N2D2::CudaTensor<T>::CudaTensor(bool hostBased)
    : BaseTensor(),
      Tensor<T>(),
      CudaBaseTensor(hostBased)
{
    // ctor
    mDeviceTensor = std::make_shared<CudaDeviceTensor<T> >(*this);
}

template <typename T>
N2D2::CudaTensor<T>::CudaTensor(const Tensor<T>& base, bool hostBased)
    : BaseTensor(base),
      Tensor<T>(base),
      CudaBaseTensor(hostBased)
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
N2D2::CudaTensor<T>::CudaTensor(const Tensor<T>& base,
    const std::shared_ptr<CudaDeviceTensor<T> >& dataDeviceOwner,
    size_t dataDeviceOffset,
    bool hostBased)
    : BaseTensor(base),
      Tensor<T>(base),
      CudaBaseTensor(hostBased)
{
    // ctor
    // Constructor used to extract a sub-tensor with operator[] and rows().
    // It ensures that the CudaDeviceTensor base is correct (*this).
    // The constructor with deviceTensor only is not usable, as there is no
    // way to pass to the constructor a CudaDeviceTensor with the right base.
    // As the base size() is used, it must be correct for sub-tensors.
    mDeviceTensor = std::make_shared<CudaDeviceTensor<T> >(*this,
                                                           dataDeviceOwner,
                                                           dataDeviceOffset);
}

template <typename T>
void N2D2::CudaTensor<T>::reserve(const std::vector<size_t>& dims)
{
    assert(mDeviceTensor->isOwner());
    const bool dimsMatch = (dims == mDims);

    Tensor<T>::reserve(dims);

    if (!dimsMatch)
        mDeviceTensor = std::make_shared<CudaDeviceTensor<T> >(*this);
}

template <typename T>
void N2D2::CudaTensor<T>::resize(const std::vector<size_t>& dims)
{
    assert(mDeviceTensor->isOwner());
    const bool dimsMatch = (dims == mDims);

    Tensor<T>::resize(dims);

    if (!dimsMatch)
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
    const bool dimsMatch = (dims == mDims);

    Tensor<T>::resize(dims, value);

    if (!dimsMatch)
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
    const bool dimsMatch = (dims == mDims);

    Tensor<T>::assign(dims, value);

    if (!dimsMatch)
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
void N2D2::CudaTensor<T>::push_back(const T& value)
{
    Tensor<T>::push_back(value);

    // Reallocation is needed
    // Note: don't use reserve(), as mDims is already changed!
    mDeviceTensor = std::make_shared<CudaDeviceTensor<T> >(*this);
    synchronizeHToD(); // Copy data into device memory
}

template <typename T>
void N2D2::CudaTensor<T>::push_back(const std::vector<T>& vec)
{
    Tensor<T>::push_back(vec);

    mDeviceTensor = std::make_shared<CudaDeviceTensor<T> >(*this);
    synchronizeHToD(); // Copy data into device memory
}

template <typename T>
void N2D2::CudaTensor<T>::push_back(const Tensor<T>& frame)
{
    Tensor<T>::push_back(frame);

    mDeviceTensor = std::make_shared<CudaDeviceTensor<T> >(*this);
    synchronizeHToD(); // Copy data into device memory
}

template <typename T>
void N2D2::CudaTensor<T>::append(const std::vector<T>& vec)
{
    Tensor<T>::append(vec);

    mDeviceTensor = std::make_shared<CudaDeviceTensor<T> >(*this);
    synchronizeHToD(); // Copy data into device memory
}

template <typename T>
void N2D2::CudaTensor<T>::append(const Tensor<T>& frame, int towardsDim)
{
    Tensor<T>::append(frame, towardsDim);

    mDeviceTensor = std::make_shared<CudaDeviceTensor<T> >(*this);
    synchronizeHToD(); // Copy data into device memory
}

template <typename T> void N2D2::CudaTensor<T>::clear()
{
    assert(mDeviceTensor->isOwner());
    Tensor<T>::clear();

    mDeviceTensor = std::make_shared<CudaDeviceTensor<T> >(*this);
}

template <class T>
void N2D2::CudaTensor<T>::swap(CudaTensor<T>& tensor)
{
    Tensor<T>::swap(tensor);

    // CudaBaseTensor
    std::swap(mHostBased, tensor.mHostBased);
    mDeviceTensors.swap(tensor.mDeviceTensors);

    // CudaTensor<T>
    mDeviceTensor.swap(tensor.mDeviceTensor);
}

template <class T> N2D2::CudaTensor<T> N2D2::CudaTensor<T>::clone() const {
    return CudaTensor<T>(Tensor<T>::clone(), mHostBased);
}

template <class T>
N2D2::CudaTensor<T> N2D2::CudaTensor<T>::operator[](size_t i)
{
    return CudaTensor<T>(
        Tensor<T>::operator[](i),
        mDeviceTensor,
        i * mSizeM1,
        mHostBased);
}

template <class T>
const N2D2::CudaTensor<T> N2D2::CudaTensor<T>::operator[](size_t i) const
{
    return CudaTensor<T>(
        Tensor<T>::operator[](i),
        mDeviceTensor,
        i * mSizeM1,
        mHostBased);
}

template <class T>
N2D2::CudaTensor<T> N2D2::CudaTensor<T>::rows(size_t j0, size_t nb)
{
    return CudaTensor<T>(
        Tensor<T>::rows(j0, nb),
        mDeviceTensor,
        j0 * mSizeM1,
        mHostBased);
}

template <class T>
const N2D2::CudaTensor<T> N2D2::CudaTensor<T>::rows(size_t j0, size_t nb) const
{
    return CudaTensor<T>(
        Tensor<T>::rows(j0, nb),
        mDeviceTensor,
        j0 * mSizeM1,
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

template <typename T> void N2D2::CudaTensor<T>::synchronizeToD(
    const Tensor<T>& tensor) const
{
    assert(size() == tensor.size());
    assert(dims() == tensor.dims());

    CHECK_CUDA_STATUS(cudaMemcpy(mDeviceTensor->getDevicePtr(),
                                 &(*tensor.begin()),
                                 size() * sizeof(T),
                                 cudaMemcpyHostToDevice));
}

template <typename T> void N2D2::CudaTensor<T>::synchronizeToH(
    BaseTensor& baseTensor) const
{
    Tensor<T> tensor(dims());
    CHECK_CUDA_STATUS(cudaMemcpy(&(*tensor.begin()),
                                 mDeviceTensor->getDevicePtr(),
                                 size() * sizeof(T),
                                 cudaMemcpyDeviceToHost));

    baseTensor.resize(dims());
    baseTensor = tensor;
}

template <typename T> void N2D2::CudaTensor<T>::synchronizeToH(
    Tensor<T>& tensor) const
{
    assert(size() == tensor.size());
    assert(dims() == tensor.dims());

    CHECK_CUDA_STATUS(cudaMemcpy(&(*tensor.begin()),
                                 mDeviceTensor->getDevicePtr(),
                                 size() * sizeof(T),
                                 cudaMemcpyDeviceToHost));
}

template <typename T> N2D2::Tensor<T> N2D2::CudaTensor<T>::synchronizeToH() 
    const
{
    Tensor<T> tensor(dims());
    CHECK_CUDA_STATUS(cudaMemcpy(&(*tensor.begin()),
                                 mDeviceTensor->getDevicePtr(),
                                 size() * sizeof(T),
                                 cudaMemcpyDeviceToHost));
    return tensor;
}

template <typename T> void N2D2::CudaTensor<T>::broadcast(int srcDev,
                                                          int dstDev) const
{
    mDeviceTensor->broadcast(srcDev, dstDev);
}

template <typename T> void N2D2::CudaTensor<T>::broadcastAllFrom(int srcDev)
    const
{
    mDeviceTensor->broadcastAllFrom(srcDev);
}

template <typename T> void N2D2::CudaTensor<T>::broadcastAllFrom(int srcDev,
    std::vector<DeviceState> devices) const
{
    mDeviceTensor->broadcastAllFrom(srcDev, devices);
}

template <typename T> void N2D2::CudaTensor<T>::broadcastAnyTo(int dstDev)
    const
{
    mDeviceTensor->broadcastAnyTo(dstDev);
}

template <typename T> void N2D2::CudaTensor<T>::aggregate(int srcDev,
                                                          int dstDev) const
{
    mDeviceTensor->aggregate(srcDev, dstDev);
}

template <typename T> void N2D2::CudaTensor<T>::aggregateAllTo(int dstDev)
    const
{
    mDeviceTensor->aggregateAllTo(dstDev);    
}

template <typename T> void N2D2::CudaTensor<T>::aggregateAllTo(int dstDev,
    std::vector<DeviceState> devices) const
{
    mDeviceTensor->aggregateAllTo(dstDev, devices);    
}

#endif // N2D2_CUDATENSOR_H
