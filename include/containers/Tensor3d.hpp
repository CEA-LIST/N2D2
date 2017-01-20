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
 * @file      Tensor3d.h
 * @author    Olivier BICHLER (olivier.bichler@cea.fr)
 * @brief     Define Tensor3d.
 *
 * @details   This class is an upper representation of a STL vector.
*/

#ifndef N2D2_TENSOR3D_H
#define N2D2_TENSOR3D_H

#include <cassert>
#include <iostream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <vector>

#include "containers/Tensor2d.hpp"

namespace N2D2 {
/**
 * @class   Tensor3d
 * @brief   3 dimensional container which simplify access to multidimensional
 * data. Can be convert as an OpenCV matrix.
*/
template <class T> class Tensor3d {
public:
    typedef typename std::vector<T>::iterator iterator;
    typedef typename std::vector<T>::const_iterator const_iterator;
    typedef typename std::vector<T>::reference reference;
    typedef typename std::vector<T>::const_reference const_reference;

    Tensor3d();
    Tensor3d(unsigned int dimX,
             unsigned int dimY,
             unsigned int dimZ,
             const T& value = T());
    Tensor3d(unsigned int dimX,
             unsigned int dimY,
             unsigned int dimZ,
             const std::shared_ptr<std::vector<T> >& data,
             unsigned int dataOffset);
    template <typename InputIterator>
    Tensor3d(unsigned int dimX,
             unsigned int dimY,
             unsigned int dimZ,
             InputIterator first,
             InputIterator last);
    Tensor3d(const cv::Mat& mat);
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
    unsigned int dimZ() const
    {
        return mDimZ;
    }
    unsigned int size() const
    {
        return mDimX * mDimY * mDimZ;
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
        return (*mData).begin() + mDataOffset + mDimX * mDimY * mDimZ;
    }
    const_iterator end() const
    {
        return (*mData).begin() + mDataOffset + mDimX * mDimY * mDimZ;
    }
    inline void
    reserve(unsigned int dimX, unsigned int dimY, unsigned int dimZ);
    inline void resize(unsigned int dimX,
                       unsigned int dimY,
                       unsigned int dimZ,
                       const T& value = T());
    inline void assign(unsigned int dimX,
                       unsigned int dimY,
                       unsigned int dimZ,
                       const T& value);
    inline void push_back(const Tensor2d<T>& frame);
    inline void clear();
    inline void swap(Tensor3d<T>& tensor);
    // Return type should be "reference" (not T&), in order to ensure it works
    // for std::vector<bool>, which is a special case...
    inline reference operator()(unsigned int i, unsigned int j, unsigned int k);
    inline const_reference
    operator()(unsigned int i, unsigned int j, unsigned int k) const;
    inline reference operator()(unsigned int index);
    inline const_reference operator()(unsigned int index) const;
    reference at(unsigned int i, unsigned int j, unsigned int k)
    {
        return (*mData).at(mDataOffset + i + j * mDimX + k * mDimX * mDimY);
    }
    const_reference at(unsigned int i, unsigned int j, unsigned int k) const
    {
        return (*mData).at(mDataOffset + i + j * mDimX + k * mDimX * mDimY);
    }
    reference at(unsigned int index)
    {
        return (*mData).at(mDataOffset + index);
    }
    const_reference at(unsigned int index) const
    {
        return (*mData).at(mDataOffset + index);
    }
    inline Tensor2d<T> operator[](unsigned int k);
    inline const Tensor2d<T> operator[](unsigned int k) const;
    Tensor3d<T>& operator=(const Tensor3d<T>& tensor);

    inline operator cv::Mat() const;
    inline std::vector<T>& data()
    {
        return (*mData);
    };
    inline const std::vector<T>& data() const
    {
        return (*mData);
    };
    virtual ~Tensor3d() {};

protected:
    unsigned int mDimX;
    unsigned int mDimY;
    unsigned int mDimZ;
    const std::shared_ptr<std::vector<T> > mData;
    const unsigned int mDataOffset;
};
}

template <class T>
N2D2::Tensor3d<T>::Tensor3d()
    : mDimX(0), mDimY(0), mDimZ(0), mData(new std::vector<T>()), mDataOffset(0)
{
    // ctor
}

template <class T>
N2D2::Tensor3d<T>::Tensor3d(unsigned int dimX,
                            unsigned int dimY,
                            unsigned int dimZ,
                            const std::shared_ptr<std::vector<T> >& data,
                            unsigned int dataOffset)
    : mDimX(dimX),
      mDimY(dimY),
      mDimZ(dimZ),
      mData(data),
      mDataOffset(dataOffset)
{
    // ctor
}

template <class T>
N2D2::Tensor3d<T>::Tensor3d(unsigned int dimX,
                            unsigned int dimY,
                            unsigned int dimZ,
                            const T& value)
    : mDimX(dimX),
      mDimY(dimY),
      mDimZ(dimZ),
      mData(new std::vector<T>(dimX * dimY * dimZ, value)),
      mDataOffset(0)
{
    // ctor
}

template <class T>
template <typename InputIterator>
N2D2::Tensor3d<T>::Tensor3d(unsigned int dimX,
                            unsigned int dimY,
                            unsigned int dimZ,
                            InputIterator first,
                            InputIterator last)
    : mDimX(dimX),
      mDimY(dimY),
      mDimZ(dimZ),
      mData(new std::vector<T>(first, last)),
      mDataOffset(0)
{
    // ctor
    if (mDimX * mDimY * mDimZ != (*mData).size())
        throw std::runtime_error("Invalid size.");
}

template <class T>
N2D2::Tensor3d<T>::Tensor3d(const cv::Mat& mat)
    : mDimX(mat.cols),
      mDimY(mat.rows),
      mDimZ(0), // Is incremented by push_back()
      mData(new std::vector<T>()),
      mDataOffset(0)
{
    // ctor
    std::vector<cv::Mat> channels;
    cv::split(mat, channels);

    for (std::vector<cv::Mat>::const_iterator itChannel = channels.begin();
         itChannel != channels.end();
         ++itChannel) {
        const Tensor2d<T> tensor(*itChannel);
        push_back(tensor);
    }
}

template <class T>
void N2D2::Tensor3d
    <T>::reserve(unsigned int dimX, unsigned int dimY, unsigned int dimZ)
{
    assert(mData.unique());

    mDimX = dimX;
    mDimY = dimY;
    mDimZ = dimZ;
    (*mData).reserve(dimX * dimY * dimZ);
}

template <class T>
void N2D2::Tensor3d<T>::resize(unsigned int dimX,
                               unsigned int dimY,
                               unsigned int dimZ,
                               const T& value)
{
    assert(mData.unique());

    mDimX = dimX;
    mDimY = dimY;
    mDimZ = dimZ;
    (*mData).resize(dimX * dimY * dimZ, value);
}

template <class T>
void N2D2::Tensor3d<T>::assign(unsigned int dimX,
                               unsigned int dimY,
                               unsigned int dimZ,
                               const T& value)
{
    assert(mData.unique());

    mDimX = dimX;
    mDimY = dimY;
    mDimZ = dimZ;
    (*mData).assign(dimX * dimY * dimZ, value);
}

template <class T> void N2D2::Tensor3d<T>::push_back(const Tensor2d<T>& frame)
{
    assert(mData.unique());

    if (mDimX == 0 && mDimY == 0) {
        mDimX = frame.dimX();
        mDimY = frame.dimY();
    } else {
        if (mDimX != frame.dimX() || mDimY != frame.dimY())
            throw std::runtime_error(
                "Tensor3d<T>::push_back(): tensor dimension must match");
    }

    ++mDimZ;
    (*mData).insert((*mData).end(), frame.begin(), frame.end());
}

template <class T> void N2D2::Tensor3d<T>::clear()
{
    assert(mData.unique());

    mDimX = 0;
    mDimY = 0;
    mDimZ = 0;
    (*mData).clear();
}

template <class T> void N2D2::Tensor3d<T>::swap(Tensor3d<T>& tensor)
{
    assert(mData.unique());

    std::swap(mDimX, tensor.mDimX);
    std::swap(mDimY, tensor.mDimY);
    std::swap(mDimZ, tensor.mDimZ);
    (*mData).swap((*tensor.mData));

    assert((*mData).size() == mDimX * mDimY * mDimZ);
    assert((*tensor.mData).size() == tensor.mDimX * tensor.mDimY
                                     * tensor.mDimZ);
}

template <class T>
typename N2D2::Tensor3d<T>::reference N2D2::Tensor3d<T>::
operator()(unsigned int i, unsigned int j, unsigned int k)
{
    assert(i < mDimX);
    assert(j < mDimY);
    assert(k < mDimZ);

    return (*mData)[mDataOffset + i + j * mDimX + k * mDimX * mDimY];
}

template <class T>
typename N2D2::Tensor3d<T>::const_reference N2D2::Tensor3d<T>::
operator()(unsigned int i, unsigned int j, unsigned int k) const
{
    assert(i < mDimX);
    assert(j < mDimY);
    assert(k < mDimZ);

    return (*mData)[mDataOffset + i + j * mDimX + k * mDimX * mDimY];
}

template <class T>
typename N2D2::Tensor3d<T>::reference N2D2::Tensor3d<T>::
operator()(unsigned int index)
{
    assert(index < mDimX * mDimY * mDimZ);

    return (*mData)[mDataOffset + index];
}

template <class T>
typename N2D2::Tensor3d<T>::const_reference N2D2::Tensor3d<T>::
operator()(unsigned int index) const
{
    assert(index < mDimX * mDimY * mDimZ);

    return (*mData)[mDataOffset + index];
}

template <class T>
N2D2::Tensor2d<T> N2D2::Tensor3d<T>::operator[](unsigned int k)
{
    return Tensor2d<T>(mDimX, mDimY, mData, mDataOffset + k * mDimX * mDimY);
}

template <class T>
const N2D2::Tensor2d<T> N2D2::Tensor3d<T>::operator[](unsigned int k) const
{
    return Tensor2d<T>(mDimX, mDimY, mData, mDataOffset + k * mDimX * mDimY);
}

template <class T>
N2D2::Tensor3d<T>& N2D2::Tensor3d<T>::operator=(const Tensor3d<T>& tensor)
{
    assert(mDimX == tensor.mDimX);
    assert(mDimY == tensor.mDimY);
    assert(mDimZ == tensor.mDimZ);

    std::copy(tensor.begin(), tensor.end(), (*mData).begin() + mDataOffset);
    return *this;
}

template <class T> N2D2::Tensor3d<T>::operator cv::Mat() const
{
    std::vector<cv::Mat> channels;

    for (unsigned int k = 0; k < mDimZ; ++k) {
        channels.push_back((cv::Mat)((*this)[k]));
        /*
                const std::vector<T> vec((*mData).begin() + mDataOffset +
           k*mDimY*mDimX,
                    (*mData).begin() + mDataOffset + (k + 1)*mDimY*mDimX);

                channels.push_back(cv::Mat(vec, true).reshape(0, mDimY));
        */
    }

    cv::Mat mat;
    cv::merge(channels, mat);
    return mat;
}

#endif // N2D2_TENSOR3D_H
