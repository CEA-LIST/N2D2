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
 * @file      Tensor4d.h
 * @author    Olivier BICHLER (olivier.bichler@cea.fr)
 * @brief     Define Tensor4d.
 *
 * @details   This class is an upper representation of a STL vector.
*/

#ifndef N2D2_TENSOR4D_H
#define N2D2_TENSOR4D_H

#include <cassert>
#include <iostream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <vector>

#include "containers/Tensor3d.hpp"

namespace N2D2 {
/**
 * @class   Tensor4d
 * @brief   4 dimensional container which simplify access to multidimensional
 * data.
*/
template <class T> class Tensor4d {
public:
    struct Index {
        unsigned int i;
        unsigned int j;
        unsigned int k;
        unsigned int b;

        Index(unsigned int i_ = 0,
               unsigned int j_ = 0,
               unsigned int k_ = 0,
               unsigned int b_ = 0)
            : i(i_), j(j_), k(k_), b(b_) {}
    };

    typedef typename std::vector<T>::iterator iterator;
    typedef typename std::vector<T>::const_iterator const_iterator;
    typedef typename std::vector<T>::reference reference;
    typedef typename std::vector<T>::const_reference const_reference;

    Tensor4d();
    Tensor4d(unsigned int dimX,
             unsigned int dimY,
             unsigned int dimZ,
             unsigned int dimB,
             const T& value = T());
    template <typename InputIterator>
    Tensor4d(unsigned int dimX,
             unsigned int dimY,
             unsigned int dimZ,
             unsigned int dimB,
             InputIterator first,
             InputIterator last);
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
    unsigned int dimB() const
    {
        return mDimB;
    }
    unsigned int size() const
    {
        return (*mData).size();
    }
    iterator begin()
    {
        return (*mData).begin();
    }
    const_iterator begin() const
    {
        return (*mData).begin();
    }
    iterator end()
    {
        return (*mData).end();
    }
    const_iterator end() const
    {
        return (*mData).end();
    }
    inline virtual void reserve(unsigned int dimX,
                                unsigned int dimY,
                                unsigned int dimZ,
                                unsigned int dimB);
    inline virtual void resize(unsigned int dimX,
                               unsigned int dimY = 1,
                               unsigned int dimZ = 1,
                               unsigned int dimB = 1,
                               const T& value = T());
    inline virtual void assign(unsigned int dimX,
                               unsigned int dimY,
                               unsigned int dimZ,
                               unsigned int dimB,
                               const T& value);
    inline virtual void fill(const T& value);
    inline virtual void push_back(const Tensor3d<T>& frame);
    inline virtual void clear();
    inline void swap(Tensor4d<T>& tensor);
    // Return type should be "reference" (not T&), in order to ensure it works
    // for std::vector<bool>, which is a special case...
    inline reference
    operator()(unsigned int i, unsigned int j, unsigned int k, unsigned int b);
    inline const_reference operator()(unsigned int i,
                                      unsigned int j,
                                      unsigned int k,
                                      unsigned int b) const;
    inline reference operator()(const Index& index);
    inline const_reference operator()(const Index& index) const;
    inline reference operator()(unsigned int ijk, unsigned int b);
    inline const_reference operator()(unsigned int ijk, unsigned int b) const;
    inline reference operator()(unsigned int index);
    inline const_reference operator()(unsigned int index) const;
    reference at(unsigned int i, unsigned int j, unsigned int k, unsigned int b)
    {
        return (*mData).at(i + mDimX * (j + mDimY * (k + mDimZ * b)));
    }
    const_reference
    at(unsigned int i, unsigned int j, unsigned int k, unsigned int b) const
    {
        return (*mData).at(i + mDimX * (j + mDimY * (k + mDimZ * b)));
    }
    reference at(unsigned int index)
    {
        return (*mData).at(index);
    }
    const_reference at(unsigned int index) const
    {
        return (*mData).at(index);
    }
    reference at(unsigned int ijk, unsigned int b)
    {
        return (*mData).at(ijk + b * mDimX * mDimY * mDimZ);
    }
    const_reference at(unsigned int ijk, unsigned int b) const
    {
        return (*mData).at(ijk + b * mDimX * mDimY * mDimZ);
    }
    inline Tensor3d<T> operator[](unsigned int b);
    inline const Tensor3d<T> operator[](unsigned int b) const;

    /** Synchronize Device To Host */
    virtual void synchronizeDToH() const {};
    virtual void synchronizeDToH(unsigned int /*index*/,
                                 unsigned int /*length*/) const {};
    virtual void synchronizeDToH(unsigned int /*i*/,
                                 unsigned int /*j*/,
                                 unsigned int /*k*/,
                                 unsigned int /*b*/,
                                 unsigned int /*length*/) const {};
    virtual void synchronizeDToH(unsigned int /*ijk*/,
                                 unsigned int /*b*/,
                                 unsigned int /*length*/) const {};

    /** Synchronize Host To Device */
    virtual void synchronizeHToD() const {};
    virtual void synchronizeHToD(unsigned int /*index*/,
                                 unsigned int /*length*/) const {};
    virtual void synchronizeHToD(unsigned int /*i*/,
                                 unsigned int /*j*/,
                                 unsigned int /*k*/,
                                 unsigned int /*b*/,
                                 unsigned int /*length*/) const {};
    virtual void synchronizeHToD(unsigned int /*ijk*/,
                                 unsigned int /*b*/,
                                 unsigned int /*length*/) const {};

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
    virtual ~Tensor4d() {};

protected:
    unsigned int mDimX;
    unsigned int mDimY;
    unsigned int mDimZ;
    unsigned int mDimB;
    const std::shared_ptr<std::vector<T> > mData;
    const std::shared_ptr<bool> mValid;
};
}

template <class T>
N2D2::Tensor4d<T>::Tensor4d()
    : mDimX(0),
      mDimY(0),
      mDimZ(0),
      mDimB(0),
      mData(std::make_shared<std::vector<T> >()),
      mValid(std::make_shared<bool>(false))
{
    // ctor
}

template <class T>
N2D2::Tensor4d<T>::Tensor4d(unsigned int dimX,
                            unsigned int dimY,
                            unsigned int dimZ,
                            unsigned int dimB,
                            const T& value)
    : mDimX(dimX),
      mDimY(dimY),
      mDimZ(dimZ),
      mDimB(dimB),
      mData(std::make_shared<std::vector<T> >(dimX * dimY * dimZ * dimB,
                                              value)),
      mValid(std::make_shared<bool>(false))
{
    // ctor
}

template <class T>
template <typename InputIterator>
N2D2::Tensor4d<T>::Tensor4d(unsigned int dimX,
                            unsigned int dimY,
                            unsigned int dimZ,
                            unsigned int dimB,
                            InputIterator first,
                            InputIterator last)
    : mDimX(dimX),
      mDimY(dimY),
      mDimZ(dimZ),
      mDimB(dimB),
      mData(std::make_shared<std::vector<T> >(first, last)),
      mValid(std::make_shared<bool>(false))
{
    // ctor
    if (mDimX * mDimY * mDimZ * dimB != (*mData).size())
        throw std::runtime_error("Invalid size.");
}

template <class T>
void N2D2::Tensor4d<T>::reserve(unsigned int dimX,
                                unsigned int dimY,
                                unsigned int dimZ,
                                unsigned int dimB)
{
    assert(mData.unique());

    mDimX = dimX;
    mDimY = dimY;
    mDimZ = dimZ;
    mDimB = dimB;
    (*mData).reserve(dimX * dimY * dimZ * dimB);
}

template <class T>
void N2D2::Tensor4d<T>::resize(unsigned int dimX,
                               unsigned int dimY,
                               unsigned int dimZ,
                               unsigned int dimB,
                               const T& value)
{
    assert(mData.unique());

    mDimX = dimX;
    mDimY = dimY;
    mDimZ = dimZ;
    mDimB = dimB;
    (*mData).resize(dimX * dimY * dimZ * dimB, value);
}

template <class T>
void N2D2::Tensor4d<T>::assign(unsigned int dimX,
                               unsigned int dimY,
                               unsigned int dimZ,
                               unsigned int dimB,
                               const T& value)
{
    assert(mData.unique());

    mDimX = dimX;
    mDimY = dimY;
    mDimZ = dimZ;
    mDimB = dimB;
    (*mData).assign(dimX * dimY * dimZ * dimB, value);
}

template <typename T> void N2D2::Tensor4d<T>::fill(const T& value)
{
    std::fill((*mData).begin(), (*mData).end(), value);
}

template <class T> void N2D2::Tensor4d<T>::push_back(const Tensor3d<T>& frame)
{
    assert(mData.unique());

    if (mDimX == 0 && mDimY == 0 && mDimZ == 0) {
        mDimX = frame.dimX();
        mDimY = frame.dimY();
        mDimZ = frame.dimZ();
    } else {
        if (mDimX != frame.dimX() || mDimY != frame.dimY() || mDimZ
                                                              != frame.dimZ())
            throw std::runtime_error(
                "Tensor4d<T>::push_back(): tensor dimension must match");
    }

    ++mDimB;
    (*mData).insert((*mData).end(), frame.begin(), frame.end());
}

template <class T> void N2D2::Tensor4d<T>::clear()
{
    assert(mData.unique());

    mDimX = 0;
    mDimY = 0;
    mDimZ = 0;
    mDimB = 0;
    (*mData).clear();
}

template <class T> void N2D2::Tensor4d<T>::swap(Tensor4d<T>& tensor)
{
    std::swap(mDimX, tensor.mDimX);
    std::swap(mDimY, tensor.mDimY);
    std::swap(mDimZ, tensor.mDimZ);
    std::swap(mDimB, tensor.mDimB);
    (*mData).swap((*tensor.mData));

    assert((*mData).size() == mDimX * mDimY * mDimZ * mDimB);
    assert((*tensor.mData).size() == tensor.mDimX * tensor.mDimY * tensor.mDimZ
                                     * tensor.mDimB);
}

template <class T>
typename N2D2::Tensor4d<T>::reference N2D2::Tensor4d<T>::
operator()(unsigned int i, unsigned int j, unsigned int k, unsigned int b)
{
    assert(i < mDimX);
    assert(j < mDimY);
    assert(k < mDimZ);
    assert(b < mDimB);

    return (*mData)[i + mDimX * (j + mDimY * (k + mDimZ * b))];
}

template <class T>
typename N2D2::Tensor4d<T>::const_reference N2D2::Tensor4d<T>::
operator()(unsigned int i, unsigned int j, unsigned int k, unsigned int b) const
{
    assert(i < mDimX);
    assert(j < mDimY);
    assert(k < mDimZ);
    assert(b < mDimB);

    return (*mData)[i + mDimX * (j + mDimY * (k + mDimZ * b))];
}

template <class T>
typename N2D2::Tensor4d<T>::reference N2D2::Tensor4d<T>::
operator()(const Index& index)
{
    assert(index.i < mDimX);
    assert(index.j < mDimY);
    assert(index.k < mDimZ);
    assert(index.b < mDimB);

    return (*mData)[index.i + mDimX * (index.j + mDimY * (index.k
            + mDimZ * index.b))];
}

template <class T>
typename N2D2::Tensor4d<T>::const_reference N2D2::Tensor4d<T>::
operator()(const Index& index) const
{
    assert(index.i < mDimX);
    assert(index.j < mDimY);
    assert(index.k < mDimZ);
    assert(index.b < mDimB);

    return (*mData)[index.i + mDimX * (index.j + mDimY * (index.k
            + mDimZ * index.b))];
}

template <class T>
typename N2D2::Tensor4d<T>::reference N2D2::Tensor4d<T>::
operator()(unsigned int ijk, unsigned int b)
{
    assert(ijk < mDimX * mDimY * mDimZ);
    assert(b < mDimB);

    return (*mData)[ijk + b * mDimX * mDimY * mDimZ];
}

template <class T>
typename N2D2::Tensor4d<T>::const_reference N2D2::Tensor4d<T>::
operator()(unsigned int ijk, unsigned int b) const
{
    assert(ijk < mDimX * mDimY * mDimZ);
    assert(b < mDimB);

    return (*mData)[ijk + b * mDimX * mDimY * mDimZ];
}

template <class T>
typename N2D2::Tensor4d<T>::reference N2D2::Tensor4d<T>::
operator()(unsigned int index)
{
    assert(index < (*mData).size());

    return (*mData)[index];
}

template <class T>
typename N2D2::Tensor4d<T>::const_reference N2D2::Tensor4d<T>::
operator()(unsigned int index) const
{
    assert(index < (*mData).size());

    return (*mData)[index];
}

template <class T>
N2D2::Tensor3d<T> N2D2::Tensor4d<T>::operator[](unsigned int b)
{
    return Tensor3d<T>(mDimX, mDimY, mDimZ, mData, b * mDimX * mDimY * mDimZ);
}

template <class T>
const N2D2::Tensor3d<T> N2D2::Tensor4d<T>::operator[](unsigned int b) const
{
    return Tensor3d<T>(mDimX, mDimY, mDimZ, mData, b * mDimX * mDimY * mDimZ);
}

#endif // N2D2_TENSOR4D_H
