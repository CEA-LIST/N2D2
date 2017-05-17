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
 * @file      Tensor2d.h
 * @author    Olivier BICHLER (olivier.bichler@cea.fr)
 * @brief     Define Tensor2d.
 *
 * @details   This class is an upper representation of a STL vector.
*/

#ifndef N2D2_TENSOR2D_H
#define N2D2_TENSOR2D_H

#include <cassert>
#include <iostream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <vector>

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

namespace N2D2 {
/**
 * @class   Tensor2d
 * @brief   2 dimensional container which simplify access to multidimensional
 * data. Can be convert as an OpenCV matrix.
*/
template <class T> class Tensor2d {
public:
    typedef typename std::vector<T>::iterator iterator;
    typedef typename std::vector<T>::const_iterator const_iterator;
    typedef typename std::vector<T>::reference reference;
    typedef typename std::vector<T>::const_reference const_reference;

    Tensor2d();
    Tensor2d(unsigned int dimX, unsigned int dimY, const T& value = T());
    Tensor2d(unsigned int dimX,
             unsigned int dimY,
             const std::shared_ptr<std::vector<T> >& data,
             unsigned int dataOffset = 0);
    template <typename InputIterator>
    Tensor2d(unsigned int dimX,
             unsigned int dimY,
             InputIterator first,
             InputIterator last);
    Tensor2d(const cv::Mat& mat);
    bool empty() const
    {
        return (*mData).empty();
    }
    unsigned int dimX() const
    {
        return mDimX;
    }
    unsigned int dimY() const
    {
        return mDimY;
    }
    unsigned int size() const
    {
        return mDimX * mDimY;
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
        return (*mData).begin() + mDataOffset + mDimX * mDimY;
    }
    const_iterator end() const
    {
        return (*mData).begin() + mDataOffset + mDimX * mDimY;
    }
    inline void reserve(unsigned int dimX, unsigned int dimY);
    inline void
    resize(unsigned int dimX, unsigned int dimY, const T& value = T());
    inline void assign(unsigned int dimX, unsigned int dimY, const T& value);
    inline void push_back(const T& value);
    inline void clear();
    inline void swap(Tensor2d<T>& tensor);
    // Return type should be "reference" (not T&), in order to ensure it works
    // for std::vector<bool>, which is a special case...
    inline reference operator()(unsigned int i, unsigned int j);
    inline const_reference operator()(unsigned int i, unsigned int j) const;
    inline reference operator()(unsigned int index);
    inline const_reference operator()(unsigned int index) const;
    reference at(unsigned int i, unsigned int j)
    {
        return (*mData).at(mDataOffset + i + mDimX * j);
    }
    const_reference at(unsigned int i, unsigned int j) const
    {
        return (*mData).at(mDataOffset + i + mDimX * j);
    }
    reference at(unsigned int index)
    {
        return (*mData).at(mDataOffset + index);
    }
    const_reference at(unsigned int index) const
    {
        return (*mData).at(mDataOffset + index);
    }
    inline Tensor2d<T> operator[](unsigned int j);
    inline const Tensor2d<T> operator[](unsigned int j) const;
    inline Tensor2d<T> rows(unsigned int j0, unsigned int nb);
    inline const Tensor2d<T> rows(unsigned int j0, unsigned int nb) const;
    Tensor2d<T>& operator=(const Tensor2d<T>& tensor);

    template <class U>
    friend Tensor2d<U>& operator<<(Tensor2d<U>& tensor,
                                   const std::string& data);
    template <class U>
    friend std::ostream& operator<<(std::ostream& os,
                                    const Tensor2d<U>& tensor);
    template <class U>
    friend std::istream& operator>>(std::istream& is, Tensor2d<U>& tensor);

    inline operator cv::Mat() const;
    inline std::vector<T>& data()
    {
        return (*mData);
    };
    inline const std::vector<T>& data() const
    {
        return (*mData);
    };
    virtual ~Tensor2d()
    {
    }

protected:
    template <class CV_T>
    static void convert(const cv::Mat& mat, std::vector<T>& data);

    unsigned int mDimX;
    unsigned int mDimY;
    const std::shared_ptr<std::vector<T> > mData;
    const unsigned int mDataOffset;
};
}

template <class T>
N2D2::Tensor2d<T>::Tensor2d()
    : mDimX(0), mDimY(0), mData(new std::vector<T>()), mDataOffset(0)
{
    // ctor
}

template <class T>
N2D2::Tensor2d
    <T>::Tensor2d(unsigned int dimX, unsigned int dimY, const T& value)
    : mDimX(dimX),
      mDimY(dimY),
      mData(new std::vector<T>(dimX * dimY, value)),
      mDataOffset(0)
{
    // ctor
}

template <class T>
N2D2::Tensor2d<T>::Tensor2d(unsigned int dimX,
                            unsigned int dimY,
                            const std::shared_ptr<std::vector<T> >& data,
                            unsigned int dataOffset)
    : mDimX(dimX), mDimY(dimY), mData(data), mDataOffset(dataOffset)
{
    // ctor
}

template <class T>
template <typename InputIterator>
N2D2::Tensor2d<T>::Tensor2d(unsigned int dimX,
                            unsigned int dimY,
                            InputIterator first,
                            InputIterator last)
    : mDimX(dimX),
      mDimY(dimY),
      mData(new std::vector<T>(first, last)),
      mDataOffset(0)
{
    // ctor
    if (mDimX * mDimY != (*mData).size())
        throw std::runtime_error("Invalid size.");
}

template <class T>
N2D2::Tensor2d<T>::Tensor2d(const cv::Mat& mat)
    : mDimX(mat.cols),
      mDimY(mat.rows),
      mData(new std::vector<T>()),
      mDataOffset(0)
{
    // ctor
    if (mat.channels() != 1)
        throw std::runtime_error("Cannot convert cv::Mat to Tensor2d: cv::Mat "
                                 "must have a single channel.");

    switch (mat.depth()) {
    case CV_8U:
        convert<unsigned char>(mat, *mData);
        break;
    case CV_8S:
        convert<char>(mat, *mData);
        break;
    case CV_16U:
        convert<unsigned short>(mat, *mData);
        break;
    case CV_16S:
        convert<short>(mat, *mData);
        break;
    case CV_32S:
        convert<int>(mat, *mData);
        break;
    case CV_32F:
        convert<float>(mat, *mData);
        break;
    case CV_64F:
        convert<double>(mat, *mData);
        break;
    default:
        throw std::runtime_error(
            "Cannot convert cv::Mat to Tensor2d: incompatible types.");
    }
}

template <class T>
template <class CV_T>
void N2D2::Tensor2d<T>::convert(const cv::Mat& mat, std::vector<T>& data)
{
    const CV_T srcRange = (std::numeric_limits<CV_T>::is_integer)
                              ? std::numeric_limits<CV_T>::max()
                              : (CV_T)1.0;
    const T dstRange = (std::numeric_limits<T>::is_integer)
                           ? std::numeric_limits<T>::max()
                           : (T)1.0;

    data.resize(mat.rows * mat.cols);
    // data.reserve(mat.rows*mat.cols);

    for (int i = 0; i < mat.rows; ++i) {
        const CV_T* rowPtr = mat.ptr<CV_T>(i);

        for (int j = 0; j < mat.cols; ++j) {
            data[i * mat.cols + j] = static_cast<T>(
                ((std::numeric_limits<CV_T>::is_integer && std::numeric_limits
                  <T>::is_integer)
                     ? static_cast<long long int>(dstRange)
                     : static_cast<double>(dstRange)) * rowPtr[j] / srcRange);
        }

        // data.insert(data.end(), rowPtr, rowPtr + mat.cols);
    }
}

template <class T>
void N2D2::Tensor2d<T>::reserve(unsigned int dimX, unsigned int dimY)
{
    assert(mData.unique());

    mDimX = dimX;
    mDimY = dimY;
    (*mData).reserve(dimX * dimY);
}

template <class T>
void N2D2::Tensor2d
    <T>::resize(unsigned int dimX, unsigned int dimY, const T& value)
{
    assert(mData.unique());

    mDimX = dimX;
    mDimY = dimY;
    (*mData).resize(dimX * dimY, value);
}

template <class T>
void N2D2::Tensor2d
    <T>::assign(unsigned int dimX, unsigned int dimY, const T& value)
{
    assert(mData.unique());

    mDimX = dimX;
    mDimY = dimY;
    (*mData).assign(dimX * dimY, value);
}

template <class T> void N2D2::Tensor2d<T>::push_back(const T& value)
{
    assert(mData.unique());

    if (mDimY <= 1) {
        ++mDimX;
        mDimY = 1;
    } else
        throw std::runtime_error("Tensor2d<T>::push_back(): Tensor2d must be a "
                                 "vector (dimY should be 0 or 1).");

    (*mData).push_back(value);
}

template <class T> void N2D2::Tensor2d<T>::clear()
{
    assert(mData.unique());

    mDimX = 0;
    mDimY = 0;
    (*mData).clear();
}

template <class T> void N2D2::Tensor2d<T>::swap(Tensor2d<T>& tensor)
{
    assert(mData.unique());

    std::swap(mDimX, tensor.mDimX);
    std::swap(mDimY, tensor.mDimY);
    (*mData).swap(*(tensor.mData));

    assert((*mData).size() == mDimX * mDimY);
    assert((*tensor.mData).size() == tensor.mDimX * tensor.mDimY);
}

template <class T>
typename N2D2::Tensor2d<T>::reference N2D2::Tensor2d<T>::
operator()(unsigned int i, unsigned int j)
{
    assert(i < mDimX);
    assert(j < mDimY);

    return (*mData)[mDataOffset + i + mDimX * j];
}

template <class T>
typename N2D2::Tensor2d<T>::const_reference N2D2::Tensor2d<T>::
operator()(unsigned int i, unsigned int j) const
{
    assert(i < mDimX);
    assert(j < mDimY);

    return (*mData)[mDataOffset + i + mDimX * j];
}

template <class T>
typename N2D2::Tensor2d<T>::reference N2D2::Tensor2d<T>::
operator()(unsigned int index)
{
    assert(index < mDimX * mDimY);

    return (*mData)[mDataOffset + index];
}

template <class T>
typename N2D2::Tensor2d<T>::const_reference N2D2::Tensor2d<T>::
operator()(unsigned int index) const
{
    assert(index < mDimX * mDimY);

    return (*mData)[mDataOffset + index];
}

template <class T>
N2D2::Tensor2d<T> N2D2::Tensor2d<T>::operator[](unsigned int j)
{
    assert(j < mDimY);

    return Tensor2d<T>(mDimX, 1, mData, mDataOffset + j * mDimX);
}

template <class T>
const N2D2::Tensor2d<T> N2D2::Tensor2d<T>::operator[](unsigned int j) const
{
    assert(j < mDimY);

    return Tensor2d<T>(mDimX, 1, mData, mDataOffset + j * mDimX);
}

template <class T>
N2D2::Tensor2d<T> N2D2::Tensor2d<T>::rows(unsigned int j0,
                                                unsigned int nb)
{
    assert(j0 + nb <= mDimY);

    return Tensor2d<T>(mDimX, nb, mData, mDataOffset + j0 * mDimX);
}

template <class T>
const N2D2::Tensor2d<T> N2D2::Tensor2d<T>::rows(unsigned int j0,
                                                unsigned int nb) const
{
    assert(j0 + nb <= mDimY);

    return Tensor2d<T>(mDimX, nb, mData, mDataOffset + j0 * mDimX);
}

template <class T>
N2D2::Tensor2d<T>& N2D2::Tensor2d<T>::operator=(const Tensor2d<T>& tensor)
{
    assert(mDimX == tensor.mDimX);
    assert(mDimY == tensor.mDimY);

    std::copy(tensor.begin(), tensor.end(), (*mData).begin() + mDataOffset);
    return *this;
}

namespace N2D2 {
template <class T>
N2D2::Tensor2d<T>& operator<<(Tensor2d<T>& tensor, const std::string& data)
{
    std::stringstream dataStr(data);

    if (!(dataStr >> tensor) && !dataStr.eof())
        throw std::runtime_error("Missing value or unreadable data.");

    // Discard trailing whitespaces
    while (std::isspace(dataStr.peek()))
        dataStr.ignore();

    if (dataStr.get() != std::stringstream::traits_type::eof())
        throw std::runtime_error("Unread additional data remaining.");

    return tensor;
}

template <class T>
std::ostream& operator<<(std::ostream& os, const Tensor2d<T>& tensor)
{
    for (unsigned int i = 0; i < tensor.mDimY; ++i) {
        // Assume row-major storage
        std::copy(tensor.begin() + i * tensor.mDimX,
                  tensor.begin() + (i + 1) * tensor.mDimX,
                  std::ostream_iterator<T>(os, " "));

        os << "\n";
    }

    return os;
}

template <class T>
std::istream& operator>>(std::istream& is, Tensor2d<T>& tensor)
{
    std::string line;
    unsigned int nbRows = 0;
    unsigned int nbCols = 0;

    while (std::getline(is, line)) {
        std::stringstream values(line);

        unsigned int nbValues = 0;
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
                    "Tensor2d<T>::operator >>: Extra data at end of line");
        }
#else
        if (!values.eof())
            throw std::runtime_error(
                "Tensor2d<T>::operator >>: Extra data at end of line");
#endif

        if (nbCols == 0)
            nbCols = nbValues;
        else if (nbValues != nbCols)
            throw std::runtime_error(
                "Tensor2d<T>::operator >>: Wrong number of columns");

        ++nbRows;
    }

    assert(tensor.data().size() == nbCols * nbRows);

    tensor.resize(nbCols, nbRows);
    return is;
}
}

template <class T> N2D2::Tensor2d<T>::operator cv::Mat() const
{
    const int type
        = (std::is_same<T, char>::value)
              ? CV_8SC1
              : (std::is_same<T, unsigned char>::value)
                    ? CV_8UC1
                    : (std::is_same<T, short>::value)
                          ? CV_16SC1
                          : (std::is_same<T, unsigned short>::value)
                                ? CV_16UC1
                                : (std::is_same<T, int>::value)
                                      ? CV_32SC1
                                      : (std::is_same<T, float>::value)
                                            ? CV_32FC1
                                            : (std::is_same<T, double>::value)
                                                  ? CV_64FC1
                                                  : -1;

    if (type == -1)
        throw std::runtime_error(
            "Cannot convert Tensor2d to cv::Mat: incompatible types.");

    return cv::Mat((int)mDimY, (int)mDimX, type, &((*mData)[mDataOffset]));
}

#endif // N2D2_TENSOR2D_H
