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

/**
 * @file      Tensor.h
 * @author    Olivier BICHLER (olivier.bichler@cea.fr)
 * @brief     Define Tensor.
 *
 * @details   This class is an upper representation of a STL vector.
*/

#ifndef N2D2_TENSOR_H
#define N2D2_TENSOR_H

#include <algorithm>
#include <cassert>
#include <cctype>
#include <complex>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <stdexcept>
#include <type_traits>
#include <vector>

#ifdef OPENCV_USE_OLD_HEADERS       //  before OpenCV 2.2.0
    #include "cv.h"
    #include "highgui.h"
#else
    #include "opencv2/core/version.hpp"
    #if CV_MAJOR_VERSION == 2
        #include "opencv2/core/core.hpp"
        #include "opencv2/imgproc/imgproc.hpp"
        #include "opencv2/highgui/highgui.hpp"
    #elif CV_MAJOR_VERSION >= 3
        #include "opencv2/core.hpp"
        #include "opencv2/imgproc.hpp"
        #include "opencv2/highgui.hpp"
    #endif
#endif

#ifdef CUDA
#include "CudaUtils.hpp"
#endif

#include "third_party/half.hpp"

namespace N2D2 {
template <class T> class Tensor;

/**
 * BaseDataTensor is a simple polymorphic wrapper around std::vector
 * Its purpose is to be able to store a pointer to any type of std::vector.
 * It is used for storing casted tensor data, in mDataTensors:
 * mutable std::map<const std::type_info*,
 *            std::shared_ptr<BaseDataTensor> > mDataTensors;
*/
class BaseDataTensor {
public:
    virtual ~BaseDataTensor() {};
};

/**
 * DataTensor<T> is a simple wrapper around std::vector<T>, which inherit from
 * BaseDataTensor
*/
template <class T>
class DataTensor : public BaseDataTensor {
public:
    DataTensor(const std::vector<T>& data) : mUnallocatedSize(0), mData(data) {}
    DataTensor(size_t size) : mUnallocatedSize(size), mData() {}
    std::vector<T>& operator()() {
        if (mUnallocatedSize > 0) {
            // Lazy memory allocation, useful to avoid host memory allocation
            // when casting CudaTensor types on GPU only.
            mData.resize(mUnallocatedSize);
            mUnallocatedSize = 0;
        }

        return mData;
    }
    virtual ~DataTensor() {};

protected:
    size_t mUnallocatedSize;
    std::vector<T> mData;
};

class BaseTensor {
public:
    struct Index {
        std::vector<size_t> index;

        Index(const std::vector<size_t>& index_)
            : index(index_) {}
        Index(std::initializer_list<size_t> index_)
            : index(index_) {}
        template <typename... Args>
        Index(Args... args) : index({static_cast<size_t>(args)...}) {}
        Index& operator+=(const Index& offsets) {
            assert(offsets.index.size() <= index.size());
            std::transform(offsets.index.begin(), offsets.index.end(),
                           index.begin(), index.begin(),
                           std::plus<size_t>());
            return *this;
        }
        Index& operator+=(const std::vector<size_t>& offsets) {
            assert(offsets.size() <= index.size());
            std::transform(offsets.begin(), offsets.end(), index.begin(),
                index.begin(), std::plus<size_t>());
            return *this;
        }
        Index& operator+=(std::initializer_list<size_t> offsets) {
            assert(offsets.size() <= index.size());
            std::transform(offsets.begin(), offsets.end(), index.begin(),
                index.begin(), std::plus<size_t>());
            return *this;
        }
        size_t& operator[](unsigned int dim)
        {
            assert(dim < index.size());
            return index[dim];
        }
        size_t operator[](unsigned int dim) const
        {
            assert(dim < index.size());
            return index[dim];
        }

        friend Index operator+(Index index, const Index& offsets)
        {
            index += offsets;
            return index;
        }
    };

    bool empty() const
    {
        return (mSize == 0);
    }
    size_t dimX() const
    {
        return (mDims.size() > 0) ? mDims[0] : 0;
    }
    size_t dimY() const
    {
        return (mDims.size() > 1) ? mDims[1] : 0;
    }
    size_t dimD() const
    {
        return (mDims.size() > 2) ? mDims[2] : 0;
    }
    /// Historically, for Tensor3d, dimZ() is the last dimension
    /// For Tensor4d, it is the third dimension (one but last)
    /// But many Tensor4d were really 2D tensors, with 1st and 2nd dimensions
    /// equal 1. When converted to true 2D tensors, dimZ() should correspond to
    /// the first dimension!
    /// dimZ() is generalized as being the one but last dimension, except for 3D
    /// tensors
    size_t dimZ() const
    {
        return (mDims.size() > 3) ? mDims[mDims.size() - 2] :
               (mDims.size() == 3) ? mDims[2] :
               (mDims.size() == 2) ? mDims[0] : 0;
    }
    /// Historically, for Tensor4d, dimB is the fourth and last dimension
    /// For other tensor dimensions, dimB() is generalized as always being the
    /// last dimension
    size_t dimB() const
    {
        return (!mDims.empty()) ? mDims.back() : 0;
    }
    size_t size() const
    {
        return mSize;
    }

    virtual void reserve(std::initializer_list<size_t> dims);
    virtual void reserve(const std::vector<size_t>& dims) = 0;
    virtual void resize(std::initializer_list<size_t> dims);
    virtual void resize(const std::vector<size_t>& dims) = 0;
    virtual void reshape(std::initializer_list<size_t> dims);
    virtual void reshape(const std::vector<size_t>& dims);
    virtual void clear() = 0;
    virtual void save(std::ostream& data) const = 0;
    virtual void load(std::istream& data) = 0;

    /** Synchronize Device To Host */
    virtual void synchronizeDToH() const {};
    virtual void synchronizeDToH(
        std::initializer_list<size_t> /*indexAndLength*/) const {};
    virtual void synchronizeDToH(const Index& /*index*/, size_t /*length*/)
        const {};
    // This is just a helper that call virtual
    // synchronizeDToH(std::initializer_list):
    template <typename... Args> void synchronizeDToH(Args... args) const;

    /** Synchronize Host To Device */
    virtual void synchronizeHToD() const {};
    virtual void synchronizeHToD(
        std::initializer_list<size_t> /*indexAndLength*/) const {};
    virtual void synchronizeHToD(const Index& /*index*/, size_t /*length*/)
        const {};
    // This is just a helper that call virtual
    // synchronizeHToD(std::initializer_list):
    template <typename... Args> void synchronizeHToD(Args... args) const;

    /** Synchronize Device To Host-based data  */
    virtual void synchronizeDToHBased() const {};

    /** Synchronize Host-based data To Device */
    virtual void synchronizeHBasedToD() const {};

    /** Synchronize Device-based data To Host  */
    virtual void synchronizeDBasedToH() const {};

    /** Synchronize Host data To Device-based */
    virtual void synchronizeHToDBased() const {};

    virtual void synchronizeToH(BaseTensor& tensor) const = 0;

    virtual BaseTensor& operator=(const BaseTensor& base) = 0;

    size_t nbDims() const
    {
        return mDims.size();
    };
    const std::vector<size_t>& dims() const
    {
        return mDims;
    };
    bool isValid(int dev = -1) const
    {
#ifdef CUDA
        if (dev == -1)
            CHECK_CUDA_STATUS(cudaGetDevice(&dev));
#else
        dev = 0;
#endif
        return (*mValid)[dev];
    };
    void setValid(int dev = -1)
    {
#ifdef CUDA
        if (dev == -1)
            CHECK_CUDA_STATUS(cudaGetDevice(&dev));
#else
        dev = 0;
#endif
        (*mValid)[dev] = true;
    };
    void clearValid(int dev = -1)
    {
#ifdef CUDA
        if (dev == -1)
            CHECK_CUDA_STATUS(cudaGetDevice(&dev));
#else
        dev = 0;
#endif
        (*mValid)[dev] = false;
    };
    virtual const std::type_info* getType() const = 0;
#ifdef CUDA
    virtual BaseTensor* newCuda() const = 0;
#endif
    virtual ~BaseTensor() {};

protected:
    BaseTensor(const std::vector<size_t>& dims = std::vector<size_t>(),
               const std::shared_ptr<std::vector<char> >& valid
                    = std::make_shared<std::vector<char> >(),
               size_t size = 0,
               size_t sizeM1 = 0)
        : mDims(dims),
          mValid(valid),
          mSize(size),
          mSizeM1(sizeM1)
    {
        int count = 1;
#ifdef CUDA
        CHECK_CUDA_STATUS(cudaGetDeviceCount(&count));
#endif
        (*mValid).resize(count, false);
    }
    size_t computeSize()
    {
        mSizeM1 = (!mDims.empty()) ? std::accumulate(mDims.begin(),
                                               --mDims.end(),
                                               1U,
                                               std::multiplies<size_t>()) : 0U;
        mSize = (!mDims.empty()) ? mSizeM1 * mDims.back() : 0U;
        return mSize;
    }

    size_t getOffset(unsigned int dim, size_t i) const;
    template <typename... Args>
    size_t getOffset(unsigned int dim, size_t i, Args... args) const;
    size_t getOffsetAt(unsigned int dim, size_t i) const;
    template <typename... Args>
    size_t getOffsetAt(unsigned int dim, size_t i, Args... args) const;


    template <class U> friend
        typename std::enable_if<std::is_convertible<float,U>::value
            || std::is_convertible<half_float::half,U>::value
            || std::is_convertible<double,U>::value, Tensor<U> >::type
            tensor_cast(const BaseTensor& base);
    template <class U> friend
        typename std::enable_if<!std::is_convertible<float,U>::value
            && !std::is_convertible<half_float::half,U>::value
            && !std::is_convertible<double,U>::value, Tensor<U> >::type
            tensor_cast(const BaseTensor& base);
    template <class U>
    friend Tensor<U> tensor_cast_nocopy(const BaseTensor& base);


protected:
    std::vector<size_t> mDims;
    const std::shared_ptr<std::vector<char> > mValid;

    // Cached data
    size_t mSize;
    size_t mSizeM1;

    mutable std::map<const std::type_info*,
             std::shared_ptr<BaseDataTensor> > mDataTensors;
};

template <class T> 
class Tensor : public virtual BaseTensor {
public:
    typedef typename std::vector<T>::iterator iterator;
    typedef typename std::vector<T>::const_iterator const_iterator;
    typedef typename std::vector<T>::reference reference;
    typedef typename std::vector<T>::const_reference const_reference;
    typedef T value_type;

    using BaseTensor::reserve;
    using BaseTensor::resize;

    Tensor();
    Tensor(std::initializer_list<size_t> dims,
             const T& value = T());
    Tensor(const std::vector<size_t>& dims,
             const T& value = T());
    Tensor(const std::vector<unsigned int>& dims,
             const T& value = T());
    template <typename InputIterator>
    Tensor(std::initializer_list<size_t> dims,
             InputIterator first,
             InputIterator last);
    template <typename InputIterator>
    Tensor(const std::vector<size_t>& dims,
             InputIterator first,
             InputIterator last);
    explicit Tensor(const std::vector<size_t>& dims, T* dataPtr);
    Tensor(const cv::Mat& mat, bool signedMapping = false);
    iterator begin()
    {
        return (*mData)().begin() + mDataOffset;
    }
    const_iterator begin() const
    {
        return (*mData)().begin() + mDataOffset;
    }
    iterator end()
    {
        return (*mData)().begin() + mDataOffset + size();
    }
    const_iterator end() const
    {
        return (*mData)().begin() + mDataOffset + size();
    }
    virtual void reserve(const std::vector<size_t>& dims);
    virtual void resize(const std::vector<size_t>& dims);
    virtual void resize(std::initializer_list<size_t> dims,
                               const T& value);
    virtual void resize(const std::vector<size_t>& dims,
                               const T& value);
    virtual void assign(std::initializer_list<size_t> dims,
                               const T& value);
    virtual void assign(const std::vector<size_t>& dims,
                               const T& value);
    virtual void fill(const T& value);
    virtual void push_back(const T& value);
    virtual void push_back(const std::vector<T>& vec);
    virtual void push_back(const Tensor<T>& frame);
    virtual void append(const std::vector<T>& vec);
    virtual void append(const Tensor<T>& frame);
    virtual void clear();
    virtual void save(std::ostream& stream) const;
    virtual void load(std::istream& stream);
    void swap(Tensor<T>& tensor);
    Tensor<T> clone() const;
    // Return type should be "reference" (not T&), in order to ensure it works
    // for std::vector<bool>, which is a special case...
    template <typename... Args> reference operator()(Args... args);
    template <typename... Args> const_reference operator()(Args... args) const;
    reference operator()(const Index& index);
    const_reference operator()(const Index& index) const;
    template <typename... Args> reference at(Args... args);
    template <typename... Args> const_reference at(Args... args) const;
    Tensor<T> operator[](size_t i);
    const Tensor<T> operator[](size_t i) const;
    Tensor<T> rows(size_t j0, size_t nb);
    const Tensor<T> rows(size_t j0, size_t nb) const;
    double sum() const;
    double mean() const;
    virtual void synchronizeToH(BaseTensor& tensor) const;
    BaseTensor& operator=(const BaseTensor& base);
    Tensor<T>& operator=(const Tensor<T>& tensor);
    template <class U> Tensor<T>& operator=(const Tensor<U>& tensor);

    /**
     * Return true if `other` has the same dimensions and the same data
     * as the current Tensor.
     */
    bool operator==(const Tensor& other) const;
    bool operator!=(const Tensor& other) const;


    operator cv::Mat() const;
    std::vector<T>& data()
    {
        return (*mData)();
    };
    const std::vector<T>& data() const
    {
        return (*mData)();
    };
    const std::type_info* getType() const
    {
        return &typeid(T);
    };
#ifdef CUDA
    // Create a CudaTensor<T>*, but due to a compiler bug in MSVC 2015, we return
    // a BaseTensor* that must be dynamic_casted to CudaTensor<T>.
    BaseTensor* newCuda() const;
#endif
    virtual ~Tensor() {};

protected:
    Tensor(const std::vector<size_t>& dims,
             const std::shared_ptr<DataTensor<T> >& data,
             const std::shared_ptr<std::vector<char> >& valid,
             size_t dataOffset,
             size_t size,
             size_t sizeM1);

    template <class CV_T, class U,
              typename std::enable_if<std::is_arithmetic<U>::value && 
                                      !std::is_same<U, bool>::value>::type* = nullptr>
    static void convert(const cv::Mat& mat, std::vector<U>& data,
                        bool signedMapping = false);
    
    template <class CV_T, class U,
              typename std::enable_if<!(std::is_arithmetic<U>::value && 
                                        !std::is_same<U, bool>::value)>::type* = nullptr>
    static void convert(const cv::Mat& mat,
                        std::vector<U>& data,
                        bool signedMapping = false);

protected:
    template <class U>
    friend typename std::enable_if<std::is_convertible<float,U>::value
            || std::is_convertible<half_float::half,U>::value
            || std::is_convertible<double,U>::value, Tensor<U> >::type
            tensor_cast(const BaseTensor& base);
    template <class U>
    friend typename std::enable_if<!std::is_convertible<float,U>::value
            && !std::is_convertible<half_float::half,U>::value
            && !std::is_convertible<double,U>::value, Tensor<U> >::type
            tensor_cast(const BaseTensor& base);
    
    template <class U>
    friend Tensor<U> tensor_cast_nocopy(const BaseTensor& base);

    // Needed for Tensor<T>& operator=(const Tensor<U>& tensor)
    template <class U> friend class Tensor;

protected:
    const std::shared_ptr<DataTensor<T> > mData;
    const size_t mDataOffset;
};

template <class T>
typename std::enable_if<std::is_convertible<float,T>::value
                     || std::is_convertible<half_float::half,T>::value
                     || std::is_convertible<double,T>::value, Tensor<T> >::type
tensor_cast(const BaseTensor& base)
{
    if (base.getType() == &typeid(T))
        return dynamic_cast<const Tensor<T>&>(base);

    std::map<const std::type_info*, std::shared_ptr<BaseDataTensor> >
        ::const_iterator it = base.mDataTensors.find(&typeid(T));
    std::shared_ptr<DataTensor<T> > dataTensor;

    if (it != base.mDataTensors.end())
        dataTensor = std::static_pointer_cast<DataTensor<T> >((*it).second);
    else {
        dataTensor
            = std::make_shared<DataTensor<T> >(base.mSize);
        base.mDataTensors[&typeid(T)] = dataTensor;
    }

    if (base.getType() == &typeid(float)) {
        const Tensor<float>& tensor
            = dynamic_cast<const Tensor<float>&>(base);

        std::copy(tensor.begin(), tensor.end(), (*dataTensor)().begin());
    }
    else if (base.getType() == &typeid(half_float::half)) {
        const Tensor<half_float::half>& tensor
            = dynamic_cast<const Tensor<half_float::half>&>(base);

        std::copy(tensor.begin(), tensor.end(), (*dataTensor)().begin());
    }
    else if (base.getType() == &typeid(double)) {
        const Tensor<double>& tensor
            = dynamic_cast<const Tensor<double>&>(base);

        std::copy(tensor.begin(), tensor.end(), (*dataTensor)().begin());
    }
    else if (base.getType() == &typeid(int8_t)) {
        const Tensor<int8_t>& tensor
            = dynamic_cast<const Tensor<int8_t>&>(base);

        std::copy(tensor.begin(), tensor.end(), (*dataTensor)().begin());
    }
    else if (base.getType() == &typeid(uint8_t)) {
        const Tensor<uint8_t>& tensor
            = dynamic_cast<const Tensor<uint8_t>&>(base);

        std::copy(tensor.begin(), tensor.end(), (*dataTensor)().begin());
    }
    else if (base.getType() == &typeid(int16_t)) {
        const Tensor<int16_t>& tensor
            = dynamic_cast<const Tensor<int16_t>&>(base);

        std::copy(tensor.begin(), tensor.end(), (*dataTensor)().begin());
    }
    else if (base.getType() == &typeid(uint16_t)) {
        const Tensor<uint16_t>& tensor
            = dynamic_cast<const Tensor<uint16_t>&>(base);

        std::copy(tensor.begin(), tensor.end(), (*dataTensor)().begin());
    }
    else if (base.getType() == &typeid(int32_t)) {
        const Tensor<int32_t>& tensor
            = dynamic_cast<const Tensor<int32_t>&>(base);

        std::copy(tensor.begin(), tensor.end(), (*dataTensor)().begin());
    }
    else if (base.getType() == &typeid(uint32_t)) {
        const Tensor<uint32_t>& tensor
            = dynamic_cast<const Tensor<uint32_t>&>(base);

        std::copy(tensor.begin(), tensor.end(), (*dataTensor)().begin());
    }
    else if (base.getType() == &typeid(int64_t)) {
        const Tensor<int64_t>& tensor
            = dynamic_cast<const Tensor<int64_t>&>(base);

        std::copy(tensor.begin(), tensor.end(), (*dataTensor)().begin());
    }
    else if (base.getType() == &typeid(uint64_t)) {
        const Tensor<uint64_t>& tensor
            = dynamic_cast<const Tensor<uint64_t>&>(base);

        std::copy(tensor.begin(), tensor.end(), (*dataTensor)().begin());
    }
    else {
        throw std::runtime_error("tensor_cast(): "
                                 "tensor type not supported!");
    }

    return Tensor<T>(
        base.mDims,
        dataTensor,
        base.mValid,
        0,
        base.mSize,
        base.mSizeM1);
}

template <class T>
typename std::enable_if<!std::is_convertible<float,T>::value
                     && !std::is_convertible<half_float::half,T>::value
                     && !std::is_convertible<double,T>::value, Tensor<T> >::type
tensor_cast(const BaseTensor& base)
{
    if (base.getType() == &typeid(T))
        return dynamic_cast<const Tensor<T>&>(base);

    throw std::runtime_error("tensor_cast(): "
                             "tensor type not supported (not assignable)!");
}

template <class T>
Tensor<T> tensor_cast_nocopy(const BaseTensor& base)
{
    if (base.getType() == &typeid(T))
        return dynamic_cast<const Tensor<T>&>(base);

    std::map<const std::type_info*, std::shared_ptr<BaseDataTensor> >
        ::const_iterator it = base.mDataTensors.find(&typeid(T));
    std::shared_ptr<DataTensor<T> > dataTensor;

    if (it != base.mDataTensors.end())
        dataTensor = std::static_pointer_cast<DataTensor<T> >((*it).second);
    else {
        dataTensor
            = std::make_shared<DataTensor<T> >(base.mSize);
        base.mDataTensors[&typeid(T)] = dataTensor;
    }

    return Tensor<T>(
        base.mDims,
        dataTensor,
        base.mValid,
        0,
        base.mSize,
        base.mSizeM1);
}
} // End namespace N2D2


template <typename... Args>
inline void N2D2::BaseTensor::synchronizeHToD(Args... args) const
{
    assert(sizeof...(args) == mDims.size() + 1 || sizeof...(args) == 2);
    synchronizeHToD(std::initializer_list<size_t>(
                                            {static_cast<size_t>(args)...}));
}

template <typename... Args>
inline void N2D2::BaseTensor::synchronizeDToH(Args... args) const
{
    assert(sizeof...(args) == mDims.size() + 1 || sizeof...(args) == 2);
    synchronizeDToH(std::initializer_list<size_t>(
                                            {static_cast<size_t>(args)...}));
}

template <typename... Args>
inline size_t N2D2::BaseTensor::getOffset(unsigned int dim, size_t i,
                                   Args... args) const
{
    assert(mDims.size() > dim);
    assert(i < mDims[dim]);
    return i + mDims[dim] * getOffset(dim + 1, args...);
}

inline size_t N2D2::BaseTensor::getOffset(unsigned int dim, size_t i) const
{
    (void) dim; // discard warning about unused parameter
    assert(mDims.size() > dim);
    assert(i < mDims[dim]);
    return i;
}

template <typename... Args>
inline size_t N2D2::BaseTensor::getOffsetAt(unsigned int dim, size_t i,
                                     Args... args) const
{
    assert(mDims.size() > dim);

    if (i >= mDims[dim])
        throw std::runtime_error("Out of range!");

    return i + mDims[dim] * getOffset(dim + 1, args...);
}

inline size_t N2D2::BaseTensor::getOffsetAt(unsigned int dim, size_t i) const
{
    assert(mDims.size() > dim);

    if (i >= mDims[dim])
        throw std::runtime_error("Out of range!");

    return i;
}


template <class T>
template <typename InputIterator>
inline N2D2::Tensor<T>::Tensor(std::initializer_list<size_t> dims,
                               InputIterator first,
                               InputIterator last)
    : BaseTensor(dims),
      mData(std::make_shared<DataTensor<T> >(std::vector<T>(first, last))),
      mDataOffset(0)
{
    // ctor
    if (computeSize() != (*mData)().size())
        throw std::runtime_error("Invalid size.");
}

template <class T>
template <typename InputIterator>
inline N2D2::Tensor<T>::Tensor(const std::vector<size_t>& dims,
                               InputIterator first,
                               InputIterator last)
    : BaseTensor(dims),
      mData(std::make_shared<DataTensor<T> >(std::vector<T>(first, last))),
      mDataOffset(0)
{
    // ctor
    if (computeSize() != (*mData)().size())
        throw std::runtime_error("Invalid size.");
}


template <class T>
template <typename... Args>
inline typename N2D2::Tensor<T>::reference N2D2::Tensor<T>::
operator()(Args... args)
{
    if (sizeof...(args) == 1) {
        const size_t i[sizeof...(args)] = {static_cast<size_t>(args)...};
        assert(i[0] < size());
        return (*mData)()[mDataOffset + i[0]];
    }
    else if (sizeof...(args) == 2) {
        const size_t i[sizeof...(args)] = {static_cast<size_t>(args)...};
        assert(mDims.size() > 1);
        assert(i[0] < mSizeM1);
        assert(i[1] < mDims.back());
        return (*mData)()[mDataOffset + i[0] + mSizeM1 * i[1]];
    }
    else {
        assert(sizeof...(args) == mDims.size());
        return (*mData)()[mDataOffset + getOffset(0U, args...)];
    }
}

template <class T>
template <typename... Args>
inline typename N2D2::Tensor<T>::const_reference N2D2::Tensor<T>::
operator()(Args... args) const
{
    if (sizeof...(args) == 1) {
        const size_t i[sizeof...(args)] = {static_cast<size_t>(args)...};
        assert(i[0] < size());
        return (*mData)()[mDataOffset + i[0]];
    }
    else if (sizeof...(args) == 2) {
        const size_t i[sizeof...(args)] = {static_cast<size_t>(args)...};
        assert(mDims.size() > 1);
        assert(i[0] < mSizeM1);
        assert(i[1] < mDims.back());
        return (*mData)()[mDataOffset + i[0] + mSizeM1 * i[1]];
    }
    else {
        assert(sizeof...(args) == mDims.size());
        return (*mData)()[mDataOffset + getOffset(0U, args...)];
    }
}

template <class T>
template <typename... Args>
inline typename N2D2::Tensor<T>::reference N2D2::Tensor<T>::at(Args... args)
{
    if (sizeof...(args) == 1) {
        const size_t i[sizeof...(args)] = {static_cast<size_t>(args)...};

        if (i[0] >= size())
            throw std::runtime_error("Tensor<T>::at(): Out of range!");

        return (*mData)()[mDataOffset + i[0]];
    }
    else if (sizeof...(args) == 2) {
        const size_t i[sizeof...(args)] = {static_cast<size_t>(args)...};

        if (mDims.size() == 1) {
            std::stringstream errorStr;
            errorStr << "Tensor<T>::at(): trying to access a 1D tensor (size: "
                << mDims[0] << ") with two dimensions (" << i[0] << ", " << i[1]
                << ")." << std::endl;

            throw std::runtime_error(errorStr.str());
        }

        if (i[0] >= mSizeM1)
            throw std::runtime_error("Tensor<T>::at(): Out of range!");

        if (i[1] >= mDims.back())
            throw std::runtime_error("Tensor<T>::at(): Out of range!");

        return (*mData)()[mDataOffset + i[0] + mSizeM1 * i[1]];
    }
    else {
        if (sizeof...(args) != mDims.size())
            throw std::runtime_error("Tensor<T>::at(): Argument count must "
                                     "match tensor dimension");

        return (*mData)()[mDataOffset + getOffset(0U, args...)];
    }
}

template <class T>
template <typename... Args>
inline typename N2D2::Tensor<T>::const_reference N2D2::Tensor<T>::at(Args... args)
    const
{
    if (sizeof...(args) == 1) {
        const size_t i[sizeof...(args)] = {static_cast<size_t>(args)...};

        if (i[0] >= size())
            throw std::runtime_error("Tensor<T>::at(): Out of range!");

        return (*mData)()[mDataOffset + i[0]];
    }
    else if (sizeof...(args) == 2) {
        const size_t i[sizeof...(args)] = {static_cast<size_t>(args)...};

        if (mDims.size() == 1) {
            std::stringstream errorStr;
            errorStr << "Tensor<T>::at(): trying to access a 1D tensor (size: "
                << mDims[0] << ") with two dimensions (" << i[0] << ", " << i[1]
                << ")." << std::endl;

            throw std::runtime_error(errorStr.str());
        }

        if (i[0] >= mSizeM1)
            throw std::runtime_error("Tensor<T>::at(): Out of range!");

        if (i[1] >= mDims.back())
            throw std::runtime_error("Tensor<T>::at(): Out of range!");

        return (*mData)()[mDataOffset + i[0] + mSizeM1 * i[1]];
    }
    else {
        if (sizeof...(args) != mDims.size())
            throw std::runtime_error("Tensor<T>::at(): Argument count must "
                                     "match tensor dimension");

        return (*mData)()[mDataOffset + getOffset(0U, args...)];
    }
}

template <class T>
template <class U>
inline N2D2::Tensor<T>& N2D2::Tensor<T>::operator=(const Tensor<U>& tensor)
{
    assert(mDims.size() == tensor.nbDims());

    for (unsigned int dim = 0; dim < mDims.size(); ++dim) {
        assert(mDims[dim] == tensor.dims()[dim]);
    }

    // No need to cast here, std::copy() can work with two different types,
    // as long as type U is assignable to T&.

    if ((void*)tensor.mData.get() != (void*)mData.get()
        || tensor.mDataOffset != mDataOffset)
    {
        // Actual copy only if data is different
        std::copy(tensor.begin(), tensor.end(),
                  (*mData)().begin() + mDataOffset);
    }

    return *this;
}


namespace N2D2 {
template <class T>
N2D2::Tensor<T>& operator<<(Tensor<T>& tensor, const std::string& data)
{
    std::stringstream dataStr(data);

    if (!(dataStr >> tensor) && !dataStr.eof())
        throw std::runtime_error("Tensor<T>::operator <<: Missing value or "
                                 "unreadable data.");

    // Discard trailing whitespaces
    while (std::isspace(dataStr.peek()))
        dataStr.ignore();

    if (dataStr.get() != std::stringstream::traits_type::eof())
        throw std::runtime_error("Tensor<T>::operator <<: Unread additional "
                                 "data remaining.");

    return tensor;
}

template <class T>
std::ostream& operator<<(std::ostream& os, const Tensor<T>& tensor)
{
    if (tensor.nbDims() > 2) {
        for (size_t i = 0; i < tensor.dims().back(); ++i) {
            os << "[" << i << "]";

            if (tensor.dims().size() == 3)
                os << ":\n";

            os << Tensor<T>(tensor[i]);
        }
    }
    else {
        const size_t dimY = (tensor.dims().size() > 1) ? tensor.dimY() : 1;

        for (size_t i = 0; i < dimY; ++i) {
            // Assume row-major storage
            std::copy(tensor.begin() + i * tensor.dimX(),
                      tensor.begin() + (i + 1) * tensor.dimX(),
                      std::ostream_iterator<T>(os, " "));

            os << "\n";
        }
    }

    return os;
}

template <class T>
std::istream& operator>>(std::istream& is, Tensor<T>& tensor)
{
    std::string line;
    size_t nbRows = 0;
    size_t nbCols = 0;

    while (std::getline(is, line)) {
        std::stringstream values(line);

        size_t nbValues = 0;
        T value;

        while (values >> value) {
            ++nbValues;
            tensor.push_back(value);
        }

#if defined(__GNUC__)                                                          \
    && (__GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 6))
        // Bug in libstdc++: "complex type operator>> does not set eofbit for
        // input streams"
        // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=59568
        // Replicated on GCC 4.6.3
        if (!std::is_same<T, std::complex<float> >::value
            && !std::is_same<T, std::complex<double> >::value
            && !std::is_same<T, std::complex<long double> >::value) {
            if (!values.eof())
                throw std::runtime_error(
                    "Tensor<T>::operator >>: Extra data at end of line");
        }
#else
        if (!values.eof())
            throw std::runtime_error(
                "Tensor<T>::operator >>: Extra data at end of line");
#endif

        if (nbCols == 0)
            nbCols = nbValues;
        else if (nbValues != nbCols)
            throw std::runtime_error(
                "Tensor<T>::operator >>: Wrong number of columns");

        ++nbRows;
    }

    assert(tensor.data().size() == nbCols * nbRows);

    tensor.resize({nbCols, nbRows});
    return is;
}
} // namespace N2D2 

#endif // N2D2_TENSOR_H
