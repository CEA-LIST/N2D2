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

#include <cassert>
#include <iostream>
#include <iterator>
#include <memory>
#include <stdexcept>
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
    #elif CV_MAJOR_VERSION == 3
        #include "opencv2/core.hpp"
        #include "opencv2/imgproc.hpp"
        #include "opencv2/highgui.hpp"
    #endif
#endif

#include "utils/Utils.hpp"

namespace N2D2 {
template <class T> class Tensor {
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
        inline size_t& operator[](unsigned int dim)
        {
            assert(dim < index.size());
            return index[dim];
        }
        inline size_t operator[](unsigned int dim) const
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

    typedef typename std::vector<T>::iterator iterator;
    typedef typename std::vector<T>::const_iterator const_iterator;
    typedef typename std::vector<T>::reference reference;
    typedef typename std::vector<T>::const_reference const_reference;

    Tensor();
    Tensor(std::initializer_list<size_t> dims,
             const T& value = T());
    Tensor(const std::vector<size_t>& dims,
             const T& value = T());
    template <typename InputIterator>
    Tensor(std::initializer_list<size_t> dims,
             InputIterator first,
             InputIterator last);
    template <typename InputIterator>
    Tensor(const std::vector<size_t>& dims,
             InputIterator first,
             InputIterator last);
    Tensor(const std::vector<size_t>& dims,
             const std::shared_ptr<std::vector<T> >& data,
             const std::shared_ptr<bool>& valid,
             size_t dataOffset);
    Tensor(const cv::Mat& mat);
    bool empty() const
    {
        return (*mData).empty();
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
        return (!mDims.empty()) ? std::accumulate(mDims.begin(),
                                               mDims.end(),
                                               1U,
                                               std::multiplies<size_t>()) : 0U;
    }
    iterator begin()
    {
        return (*mData).begin() + mDataOffset;
    }
    const_iterator begin() const
    {
        return (*mData).begin() + mDataOffset;
    }
    iterator end()
    {
        return (*mData).begin() + mDataOffset + size();
    }
    const_iterator end() const
    {
        return (*mData).begin() + mDataOffset + size();
    }
    inline virtual void reserve(std::initializer_list<size_t> dims);
    inline virtual void reserve(const std::vector<size_t>& dims);
    inline virtual void reshape(std::initializer_list<size_t> dims);
    inline virtual void reshape(const std::vector<size_t>& dims);
    inline virtual void resize(std::initializer_list<size_t> dims,
                               const T& value = T());
    inline virtual void resize(const std::vector<size_t>& dims,
                               const T& value = T());
    inline virtual void assign(std::initializer_list<size_t> dims,
                               const T& value);
    inline virtual void assign(const std::vector<size_t>& dims,
                               const T& value);
    inline virtual void fill(const T& value);
    inline void push_back(const T& value);
    inline virtual void push_back(const std::vector<T>& vec);
    inline virtual void push_back(const Tensor<T>& frame);
    inline virtual void clear();
    inline void swap(Tensor<T>& tensor);
    // Return type should be "reference" (not T&), in order to ensure it works
    // for std::vector<bool>, which is a special case...
    template <typename... Args> reference operator()(Args... args);
    template <typename... Args> const_reference operator()(Args... args) const;
    inline reference operator()(const Index& index);
    inline const_reference operator()(const Index& index) const;
    template <typename... Args> reference at(Args... args);
    template <typename... Args> const_reference at(Args... args) const;
    inline Tensor<T> operator[](size_t i);
    inline const Tensor<T> operator[](size_t i) const;
    inline Tensor<T> rows(size_t j0, size_t nb);
    inline const Tensor<T> rows(size_t j0, size_t nb) const;
    Tensor<T>& operator=(const Tensor<T>& tensor);

    /** Synchronize Device To Host */
    virtual void synchronizeDToH() const {};
    virtual void synchronizeDToH(
        std::initializer_list<size_t> /*indexAndLength*/) const {};
    // This is just a helper that call virtual
    // synchronizeDToH(std::initializer_list):
    template <typename... Args> void synchronizeDToH(Args... args) const;

    /** Synchronize Host To Device */
    virtual void synchronizeHToD() const {};
    virtual void synchronizeHToD(
        std::initializer_list<size_t> /*indexAndLength*/) const {};
    // This is just a helper that call virtual
    // synchronizeHToD(std::initializer_list):
    template <typename... Args> void synchronizeHToD(Args... args) const;

    size_t nbDims() const
    {
        return mDims.size();
    };
    const std::vector<size_t>& dims() const
    {
        return mDims;
    };
    inline operator cv::Mat() const;
    inline std::vector<T>& data()
    {
        return (*mData);
    };
    inline const std::vector<T>& data() const
    {
        return (*mData);
    };
    bool isValid() const
    {
        return (*mValid);
    };
    void setValid()
    {
        (*mValid) = true;
    };
    void clearValid()
    {
        (*mValid) = false;
    };
    virtual ~Tensor() {};

protected:
    template <class CV_T>
    static void convert(const cv::Mat& mat, std::vector<T>& data);

    size_t getOffset(unsigned int dim, size_t i) const;
    template <typename... Args>
    size_t getOffset(unsigned int dim, size_t i, Args... args) const;
    size_t getOffsetAt(unsigned int dim, size_t i) const;
    template <typename... Args>
    size_t getOffsetAt(unsigned int dim, size_t i, Args... args) const;

    std::vector<size_t> mDims;
    const std::shared_ptr<std::vector<T> > mData;
    const std::shared_ptr<bool> mValid;
    const size_t mDataOffset;
};
}

template <class T>
N2D2::Tensor<T>::Tensor()
    : mData(std::make_shared<std::vector<T> >()),
      mValid(std::make_shared<bool>(false)),
      mDataOffset(0)
{
    // ctor
}

template <class T>
N2D2::Tensor<T>::Tensor(std::initializer_list<size_t> dims,
                            const T& value)
    : mDims(dims),
      mData(std::make_shared<std::vector<T> >(size(), value)),
      mValid(std::make_shared<bool>(false)),
      mDataOffset(0)
{
    // ctor
}

template <class T>
N2D2::Tensor<T>::Tensor(const std::vector<size_t>& dims,
                            const T& value)
    : mDims(dims),
      mData(std::make_shared<std::vector<T> >(size(), value)),
      mValid(std::make_shared<bool>(false)),
      mDataOffset(0)
{
    // ctor
}

template <class T>
template <typename InputIterator>
N2D2::Tensor<T>::Tensor(std::initializer_list<size_t> dims,
                            InputIterator first,
                            InputIterator last)
    : mDims(dims),
      mData(std::make_shared<std::vector<T> >(first, last)),
      mValid(std::make_shared<bool>(false)),
      mDataOffset(0)
{
    // ctor
    if (size() != (*mData).size())
        throw std::runtime_error("Invalid size.");
}

template <class T>
template <typename InputIterator>
N2D2::Tensor<T>::Tensor(const std::vector<size_t>& dims,
                            InputIterator first,
                            InputIterator last)
    : mDims(dims),
      mData(std::make_shared<std::vector<T> >(first, last)),
      mValid(std::make_shared<bool>(false)),
      mDataOffset(0)
{
    // ctor
    if (size() != (*mData).size())
        throw std::runtime_error("Invalid size.");
}

template <class T>
N2D2::Tensor<T>::Tensor(const std::vector<size_t>& dims,
                            const std::shared_ptr<std::vector<T> >& data,
                            const std::shared_ptr<bool>& valid,
                            size_t dataOffset)
    : mDims(dims),
      mData(data),
      mValid(valid),
      mDataOffset(dataOffset)
{
    // ctor
}

template <class T>
N2D2::Tensor<T>::Tensor(const cv::Mat& mat)
    : mData(std::make_shared<std::vector<T> >()),
      mValid(std::make_shared<bool>(true)),
      mDataOffset(0)
{
    // ctor
    mDims.reserve(2);
    mDims.push_back(mat.cols);
    mDims.push_back(mat.rows);

    if (mat.channels() > 1)
        mDims.push_back(mat.channels());

    std::vector<cv::Mat> channels;
    cv::split(mat, channels);

    for (std::vector<cv::Mat>::const_iterator itChannel = channels.begin();
         itChannel != channels.end();
         ++itChannel)
    {
        switch ((*itChannel).depth()) {
        case CV_8U:
            convert<unsigned char>(*itChannel, *mData);
            break;
        case CV_8S:
            convert<char>(*itChannel, *mData);
            break;
        case CV_16U:
            convert<unsigned short>(*itChannel, *mData);
            break;
        case CV_16S:
            convert<short>(*itChannel, *mData);
            break;
        case CV_32S:
            convert<int>(*itChannel, *mData);
            break;
        case CV_32F:
            convert<float>(*itChannel, *mData);
            break;
        case CV_64F:
            convert<double>(*itChannel, *mData);
            break;
        default:
            throw std::runtime_error(
                "Cannot convert cv::Mat to Tensor: incompatible types.");
        }
    }

    assert((*mData).size() == mat.rows * mat.cols * mat.channels());
    assert((*mData).size() == size());
}

template <class T>
template <class CV_T>
void N2D2::Tensor<T>::convert(const cv::Mat& mat, std::vector<T>& data)
{
    const CV_T srcRange = (std::numeric_limits<CV_T>::is_integer)
                              ? std::numeric_limits<CV_T>::max()
                              : (CV_T)1.0;
    const T dstRange = (std::numeric_limits<T>::is_integer)
                           ? std::numeric_limits<T>::max()
                           : (T)1.0;

    data.reserve(data.size() + mat.rows * mat.cols);

    for (int i = 0; i < mat.rows; ++i) {
        const CV_T* rowPtr = mat.ptr<CV_T>(i);

        for (int j = 0; j < mat.cols; ++j) {
            data.push_back(static_cast<T>(
                ((std::numeric_limits<CV_T>::is_integer && std::numeric_limits
                  <T>::is_integer)
                     ? static_cast<long long int>(dstRange)
                     : static_cast<double>(dstRange)) * rowPtr[j] / srcRange));
        }
    }
}

template <class T>
void N2D2::Tensor<T>::reserve(std::initializer_list<size_t> dims)
{
    reserve(std::vector<size_t>(dims));
}

template <class T>
void N2D2::Tensor<T>::reserve(const std::vector<size_t>& dims)
{
    assert(mData.unique());

    mDims = dims;
    (*mData).reserve(size());
}

template <class T>
void N2D2::Tensor<T>::reshape(std::initializer_list<size_t> dims)
{
    reshape(std::vector<size_t>(dims));
}

template <class T>
void N2D2::Tensor<T>::reshape(const std::vector<size_t>& dims)
{
    const size_t oldSize = size();
    const std::vector<size_t> oldDims = mDims;
    mDims = dims;
    const size_t newSize = size();

    if (newSize != oldSize) {
        std::stringstream errorStr;
        errorStr << "Tensor<T>::reshape(): new size (" << mDims << " = "
            << newSize << ") does not match current size (" << oldDims << " = "
            << oldSize << ")" << std::endl;

        throw std::runtime_error(errorStr.str());
    }
}

template <class T>
void N2D2::Tensor<T>::resize(std::initializer_list<size_t> dims,
                               const T& value)
{
    resize(std::vector<size_t>(dims), value);
}

template <class T>
void N2D2::Tensor<T>::resize(const std::vector<size_t>& dims,
                               const T& value)
{
    assert(mData.unique());

    mDims = dims;
    (*mData).resize(size(), value);
}

template <class T>
void N2D2::Tensor<T>::assign(std::initializer_list<size_t> dims,
                               const T& value)
{
    assign(std::vector<size_t>(dims), value);
}

template <class T>
void N2D2::Tensor<T>::assign(const std::vector<size_t>& dims,
                               const T& value)
{
    assert(mData.unique());

    mDims = dims;
    (*mData).assign(size(), value);
}

template <typename T> void N2D2::Tensor<T>::fill(const T& value)
{
    std::fill((*mData).begin() + mDataOffset,
              (*mData).begin() + mDataOffset + size(), value);
}

template <class T> void N2D2::Tensor<T>::push_back(const T& value)
{
    assert(mData.unique());

    if (mDims.empty() || std::all_of(mDims.begin(), mDims.end(),
                                     Utils::IsZero<size_t>()))
    {
        mDims.resize(1, 1);
    }
    else if (mDims.size() == 1 || (mDims.size() == 2 && mDims[1] <= 1)) {
        ++mDims[0];
        mDims[1] = 1;
    }
    else {
        std::stringstream errorStr;
        errorStr << "Tensor<T>::push_back(): tensor second dimension must be"
            " 0 or 1 to push back a single value, but tensor dimension is "
            << mDims << std::endl;

        throw std::runtime_error(errorStr.str());
    }

    (*mData).push_back(value);
}

template <class T> void N2D2::Tensor<T>::push_back(const std::vector<T>& vec)
{
    assert(mData.unique());

    if (mDims.empty() || std::all_of(mDims.begin(), mDims.end(),
                                     Utils::IsZero<size_t>()))
    {
        mDims.resize(1, vec.size());
        mDims.push_back(0);
    }
    else if (mDims.size() != 2) {
        std::stringstream errorStr;
        errorStr << "Tensor<T>::push_back(): tensor must be 2D to push back a"
            " vector, but tensor dimension is " << mDims << std::endl;

        throw std::runtime_error(errorStr.str());
    }
    else {
        if (mDims[0] != vec.size()) {
            std::stringstream errorStr;
            errorStr << "Tensor<T>::push_back(): tensor first dimension must"
                " match the vector size (" << vec.size() << "), but is "
                << mDims[0] << " (" << mDims << ")" << std::endl;

            throw std::runtime_error(errorStr.str());
        }
    }

    ++mDims.back();
    (*mData).insert((*mData).end(), vec.begin(), vec.end());
}

template <class T> void N2D2::Tensor<T>::push_back(const Tensor<T>& frame)
{
    assert(mData.unique());

    if (mDims.empty() || std::all_of(mDims.begin(), mDims.end(),
                                     Utils::IsZero<size_t>()))
    {
        mDims = frame.dims();
        mDims.push_back(0);
    }
    else if (mDims.size() != frame.nbDims() + 1) {
        std::stringstream errorStr;
        errorStr << "Tensor<T>::push_back(): tensor must be "
            << (frame.nbDims() + 1) << "D to push back a " << frame.nbDims()
            << "D tensor, but tensor dimension is " << mDims << std::endl;

        throw std::runtime_error(errorStr.str());
    }
    else {
        for (unsigned int dim = 0; dim < frame.nbDims(); ++dim) {
            if (mDims[dim] != frame.dims()[dim]) {
                std::stringstream errorStr;
                errorStr << "Tensor<T>::push_back(): tensors dimension #"
                    << dim << " must match, but tensor dimension is "
                    << mDims << " and tensor to push back is "
                    << frame.dims() << std::endl;

                throw std::runtime_error(errorStr.str());
            }
        }
    }

    ++mDims.back();
    (*mData).insert((*mData).end(), frame.begin(), frame.end());
}

template <class T> void N2D2::Tensor<T>::clear()
{
    assert(mData.unique());

    mDims.clear();
    (*mData).clear();
}

template <class T> void N2D2::Tensor<T>::swap(Tensor<T>& tensor)
{
    std::swap(mDims, tensor.mDims);
    (*mData).swap((*tensor.mData));

    assert((*mData).size() == size());
    assert((*tensor.mData).size() == tensor.size());
}

template <class T>
size_t N2D2::Tensor<T>::getOffset(unsigned int dim, size_t i) const
{
    (void) dim; // discard warning about unused parameter
    assert(mDims.size() > dim);
    assert(i < mDims[dim]);
    return i;
}

template <class T>
template <typename... Args>
size_t N2D2::Tensor<T>::getOffset(unsigned int dim,
                                    size_t i,
                                    Args... args) const
{
    assert(mDims.size() > dim);
    assert(i < mDims[dim]);
    return i + mDims[dim] * getOffset(dim + 1, args...);
}

template <class T>
template <typename... Args>
typename N2D2::Tensor<T>::reference N2D2::Tensor<T>::
operator()(Args... args)
{
    if (sizeof...(args) == 1) {
        const size_t i[sizeof...(args)] = {static_cast<size_t>(args)...};
        assert(i[0] < size());
        return (*mData)[mDataOffset + i[0]];
    }
    else {
        assert(sizeof...(args) == mDims.size());
        return (*mData)[mDataOffset + getOffset(0U, args...)];
    }
}

template <class T>
template <typename... Args>
typename N2D2::Tensor<T>::const_reference N2D2::Tensor<T>::
operator()(Args... args) const
{
    if (sizeof...(args) == 1) {
        const size_t i[sizeof...(args)] = {static_cast<size_t>(args)...};
        assert(i[0] < size());
        return (*mData)[mDataOffset + i[0]];
    }
    else {
        assert(sizeof...(args) == mDims.size());
        return (*mData)[mDataOffset + getOffset(0U, args...)];
    }
}

template <class T>
typename N2D2::Tensor<T>::reference N2D2::Tensor<T>::
operator()(const Index& index)
{
    assert(mDims.size() == index.index.size());

    size_t offset = 0;

    for (int dim = mDims.size() - 1; dim >= 0; --dim) {
        assert(index[dim] < mDims[dim]);
        offset = index[dim] + mDims[dim] * offset;
    }

    return (*mData)[mDataOffset + offset];
}

template <class T>
typename N2D2::Tensor<T>::const_reference N2D2::Tensor<T>::
operator()(const Index& index) const
{
    assert(mDims.size() == index.index.size());

    size_t offset = 0;

    for (int dim = mDims.size() - 1; dim >= 0; --dim) {
        assert(index[dim] < mDims[dim]);
        offset = index[dim] + mDims[dim] * offset;
    }

    return (*mData)[mDataOffset + offset];
}

template <class T>
size_t N2D2::Tensor<T>::getOffsetAt(unsigned int dim, size_t i) const
{
    assert(mDims.size() > dim);

    if (i >= mDims[dim])
        throw std::runtime_error("Out of range!");

    return i;
}

template <class T>
template <typename... Args>
size_t N2D2::Tensor<T>::getOffsetAt(unsigned int dim,
                                    size_t i,
                                    Args... args) const
{
    assert(mDims.size() > dim);

    if (i >= mDims[dim])
        throw std::runtime_error("Out of range!");

    return i + mDims[dim] * getOffset(dim + 1, args...);
}

template <class T>
template <typename... Args>
typename N2D2::Tensor<T>::reference N2D2::Tensor<T>::at(Args... args)
{
    if (sizeof...(args) == 1) {
        const size_t i[sizeof...(args)] = {static_cast<size_t>(args)...};

        if (i[0] >= size())
            throw std::runtime_error("Out of range!");

        return (*mData)[mDataOffset + i[0]];
    }
    else {
        if (sizeof...(args) != mDims.size())
            throw std::runtime_error("Argument count must match "
                                     "tensor dimension");

        return (*mData)[mDataOffset + getOffset(0U, args...)];
    }
}

template <class T>
template <typename... Args>
typename N2D2::Tensor<T>::const_reference N2D2::Tensor<T>::at(Args... args)
    const
{
    if (sizeof...(args) == 1) {
        const size_t i[sizeof...(args)] = {static_cast<size_t>(args)...};

        if (i[0] >= size())
            throw std::runtime_error("Out of range!");

        return (*mData)[mDataOffset + i[0]];
    }
    else {
        if (sizeof...(args) != mDims.size())
            throw std::runtime_error("Argument count must match "
                                     "tensor dimension");

        return (*mData)[mDataOffset + getOffset(0U, args...)];
    }
}

template <class T>
N2D2::Tensor<T> N2D2::Tensor<T>::operator[](size_t i)
{
    assert(mDims.size() > 1);
    std::vector<size_t> newDims = mDims;
    newDims.pop_back();

    const size_t offset = std::accumulate(newDims.begin(),
                                                newDims.end(),
                                                1U,
                                            std::multiplies<size_t>());

    return Tensor<T>(newDims, mData, mValid, mDataOffset + i * offset);
}

template <class T>
const N2D2::Tensor<T> N2D2::Tensor<T>::operator[](size_t i) const
{
    assert(mDims.size() > 1);
    std::vector<size_t> newDims = mDims;
    newDims.pop_back();

    const size_t offset = std::accumulate(newDims.begin(),
                                                newDims.end(),
                                                1U,
                                            std::multiplies<size_t>());

    return Tensor<T>(newDims, mData, mValid, mDataOffset + i * offset);
}

template <class T>
N2D2::Tensor<T> N2D2::Tensor<T>::rows(size_t j0,
                                          size_t nb)
{
    assert(mDims.size() > 1);
    assert(j0 + nb <= mDims.back());

    std::vector<size_t> newDims = mDims;
    newDims.pop_back();

    const size_t offset = std::accumulate(newDims.begin(),
                                                newDims.end(),
                                                1U,
                                            std::multiplies<size_t>());
    newDims.push_back(nb);
    return Tensor<T>(newDims, mData, mValid, mDataOffset + j0 * offset);
}

template <class T>
const N2D2::Tensor<T> N2D2::Tensor<T>::rows(size_t j0,
                                                size_t nb) const
{
    assert(mDims.size() > 1);
    assert(j0 + nb <= mDims.back());

    std::vector<size_t> newDims = mDims;
    newDims.pop_back();

    const size_t offset = std::accumulate(newDims.begin(),
                                                newDims.end(),
                                                1U,
                                            std::multiplies<size_t>());
    newDims.push_back(nb);
    return Tensor<T>(newDims, mData, mValid, mDataOffset + j0 * offset);
}

template <class T>
template <typename... Args>
void N2D2::Tensor<T>::synchronizeHToD(Args... args) const
{
    assert(sizeof...(args) == mDims.size() + 1 || sizeof...(args) == 2);
    synchronizeHToD({static_cast<size_t>(args)...});
}

template <class T>
template <typename... Args>
void N2D2::Tensor<T>::synchronizeDToH(Args... args) const
{
    assert(sizeof...(args) == mDims.size() + 1 || sizeof...(args) == 2);
    synchronizeDToH({static_cast<size_t>(args)...});
}

template <class T>
N2D2::Tensor<T>& N2D2::Tensor<T>::operator=(const Tensor<T>& tensor)
{
    assert(mDims.size() == tensor.nbDims());

    for (unsigned int dim = 0; dim < mDims.size(); ++dim) {
        assert(mDims[dim] == tensor.dims()[dim]);
    }

    std::copy(tensor.begin(), tensor.end(), (*mData).begin() + mDataOffset);
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
}

template <class T> N2D2::Tensor<T>::operator cv::Mat() const
{
    const int type = (std::is_same<T, char>::value)         ? CV_8SC1 :
                 (std::is_same<T, unsigned char>::value)    ? CV_8UC1 :
                 (std::is_same<T, short>::value)            ? CV_16SC1 :
                 (std::is_same<T, unsigned short>::value)   ? CV_16UC1 :
                 (std::is_same<T, int>::value)              ? CV_32SC1 :
                 (std::is_same<T, float>::value)            ? CV_32FC1 :
                 (std::is_same<T, double>::value)           ? CV_64FC1 :
                                                              -1;
    if (type == -1) {
        std::stringstream errorStr;
        errorStr << "Tensor<T>::operator cv::Mat(): Cannot convert Tensor to "
            "cv::Mat: Tensor type is not supported (" << typeid(T).name()
            << ")." << std::endl;

        throw std::runtime_error(errorStr.str());
    }

    if (mDims.size() < 3) {
        return cv::Mat((int)((mDims.size() > 1) ? mDims[1] :
                             (mDims.size() > 0) ? 1 : 0),
                       (int)((mDims.size() > 0) ? mDims[0] : 0),
                       type,
                       &((*mData)[mDataOffset]));
    }
    else if (mDims.size() == 3) {
        std::vector<cv::Mat> channels;

        for (size_t k = 0; k < mDims[2]; ++k) {
            channels.push_back(cv::Mat((int)mDims[1],
                                       (int)mDims[0],
                                       type,
                                       &((*mData)[mDataOffset
                                            + k * mDims[0] * mDims[1]])));
        }

        cv::Mat mat;
        cv::merge(channels, mat);
        return mat;
    }
    else {
        std::stringstream errorStr;
        errorStr << "Tensor<T>::operator cv::Mat(): Cannot convert Tensor to "
            "cv::Mat: tensor dimension (" << mDims.size() << ": " << mDims
            << ") is > 3." << std::endl;

        throw std::runtime_error(errorStr.str());
    }
}

#endif // N2D2_TENSOR_H
