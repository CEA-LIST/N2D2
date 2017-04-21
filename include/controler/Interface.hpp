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

#ifndef N2D2_INTERFACE_H
#define N2D2_INTERFACE_H

#include <cassert>
#include <iostream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <vector>

#include "containers/Tensor4d.hpp"

namespace N2D2 {
/**
 * @class   Interface
 * @brief   Merge virtually several Tensor4d through an unified data interface.
*/
template <class T> class Interface {
public:
    typedef typename std::vector<Tensor4d<T>*>::iterator iterator;
    typedef typename std::vector<Tensor4d<T>*>::const_iterator const_iterator;
    typedef typename std::vector<T>::reference reference;
    typedef typename std::vector<T>::const_reference const_reference;

    Interface();
    bool empty() const
    {
        return mData.empty();
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
        return mData.size();
    }
    unsigned int dataSize() const;
    iterator begin()
    {
        return mData.begin();
    }
    const_iterator begin() const
    {
        return mData.begin();
    }
    iterator end()
    {
        return mData.end();
    }
    const_iterator end() const
    {
        return mData.end();
    }
    inline void fill(const T& value);
    inline virtual void push_back(Tensor4d<T>* tensor);
    inline void clear();
    // Return type should be "reference" (not T&), in order to ensure it works
    // for std::vector<bool>, which is a special case...
    inline reference
    operator()(unsigned int i, unsigned int j, unsigned int k, unsigned int b);
    inline const_reference operator()(unsigned int i,
                                      unsigned int j,
                                      unsigned int k,
                                      unsigned int b) const;
    reference
    at(unsigned int i, unsigned int j, unsigned int k, unsigned int b);
    const_reference
    at(unsigned int i, unsigned int j, unsigned int k, unsigned int b) const;
    virtual Tensor4d<T>& back();
    virtual const Tensor4d<T>& back() const;
    virtual Tensor4d<T>& operator[](unsigned int t);
    virtual const Tensor4d<T>& operator[](unsigned int t) const;
    virtual Tensor4d<T>& getTensor4d(unsigned int k);

    /** Synchronize Device To Host */
    virtual void synchronizeDToH() const;
    virtual void synchronizeDToH(unsigned int i,
                                 unsigned int j,
                                 unsigned int k,
                                 unsigned int b,
                                 unsigned int length) const;

    /** Synchronize Host To Device */
    virtual void synchronizeHToD() const;
    virtual void synchronizeHToD(unsigned int i,
                                 unsigned int j,
                                 unsigned int k,
                                 unsigned int b,
                                 unsigned int length) const;

    inline void setValid();
    inline void clearValid();
    void matchingDimB(bool value)
    {
        mMatchingDimB = value;
    }
    virtual ~Interface() {};

protected:
    unsigned int mDimZ;
    unsigned int mDimB;
    bool mMatchingDimB;
    std::vector<Tensor4d<T>*> mData;
    std::vector<std::pair<unsigned int, unsigned int> > mDataOffset;
};
}

template <class T> N2D2::Interface<T>::Interface()
    : mDimZ(0),
      mDimB(0),
      mMatchingDimB(true)
{
    // ctor
}

template <class T> unsigned int N2D2::Interface<T>::dataSize() const
{
    unsigned int size = 0;

    for (typename std::vector<Tensor4d<T>*>::const_iterator it = mData.begin(),
                                                            itEnd = mData.end();
         it != itEnd;
         ++it)
        size += (*it)->size();

    return size;
}

template <class T> void N2D2::Interface<T>::fill(const T& value)
{
    for (typename std::vector<Tensor4d<T>*>::iterator it = mData.begin(),
                                                      itEnd = mData.end();
         it != itEnd;
         ++it)
        (*it)->fill(value);
}

template <class T> void N2D2::Interface<T>::push_back(Tensor4d<T>* tensor)
{
    if (mData.empty())
        mDimB = tensor->dimB();
    else {
        if (mDimB != tensor->dimB()) {
            if (mMatchingDimB) {
                throw std::runtime_error("Interface<T>::push_back(): "
                                         "tensor dimB dimension must match");
            }
            else
                mDimB = 0;
        }
    }

    const unsigned int tensorOffset = mData.size();

    for (unsigned int k = 0; k < tensor->dimZ(); ++k)
        mDataOffset.push_back(std::make_pair(tensorOffset, mDimZ));

    mDimZ += tensor->dimZ();
    mData.push_back(tensor);
}

template <class T> void N2D2::Interface<T>::clear()
{
    mDimZ = 0;
    mDimB = 0;
    mData.clear();
    mDataOffset.clear();
}

template <class T>
typename N2D2::Interface<T>::reference N2D2::Interface<T>::
operator()(unsigned int i, unsigned int j, unsigned int k, unsigned int b)
{
    assert(k < mDimZ);
    assert(mDimB == 0 || b < mDimB);

    const std::pair<unsigned int, unsigned int>& dataOffset = mDataOffset[k];
    return (*mData[dataOffset.first])(i, j, k - dataOffset.second, b);
}

template <class T>
typename N2D2::Interface<T>::const_reference N2D2::Interface<T>::
operator()(unsigned int i, unsigned int j, unsigned int k, unsigned int b) const
{
    assert(k < mDimZ);
    assert(mDimB == 0 || b < mDimB);

    const std::pair<unsigned int, unsigned int>& dataOffset = mDataOffset[k];
    return (*mData[dataOffset.first])(i, j, k - dataOffset.second, b);
}

template <class T>
typename N2D2::Interface<T>::reference N2D2::Interface
    <T>::at(unsigned int i, unsigned int j, unsigned int k, unsigned int b)
{
    assert(k < mDimZ);
    assert(mDimB == 0 || b < mDimB);

    const std::pair<unsigned int, unsigned int>& dataOffset = mDataOffset.at(k);
    return mData[dataOffset.first]->at(i, j, k - dataOffset.second, b);
}

template <class T>
typename N2D2::Interface<T>::const_reference N2D2::Interface
    <T>::at(unsigned int i, unsigned int j, unsigned int k, unsigned int b)
    const
{
    assert(k < mDimZ);
    assert(mDimB == 0 || b < mDimB);

    const std::pair<unsigned int, unsigned int>& dataOffset = mDataOffset.at(k);
    return mData[dataOffset.first]->at(i, j, k - dataOffset.second, b);
}

template <class T> N2D2::Tensor4d<T>& N2D2::Interface<T>::back()
{
    return *(mData.back());
}

template <class T> const N2D2::Tensor4d<T>& N2D2::Interface<T>::back() const
{
    return *(mData.back());
}

template <class T>
N2D2::Tensor4d<T>& N2D2::Interface<T>::operator[](unsigned int t)
{
    return *(mData.at(t));
}

template <class T>
const N2D2::Tensor4d<T>& N2D2::Interface<T>::operator[](unsigned int t) const
{
    return *(mData.at(t));
}

template <class T>
N2D2::Tensor4d<T>& N2D2::Interface<T>::getTensor4d(unsigned int k)
{
    assert(k < mDimZ);

    const std::pair<unsigned int, unsigned int>& dataOffset = mDataOffset.at(k);
    return *(mData[dataOffset.first]);
}

template <typename T> void N2D2::Interface<T>::synchronizeDToH() const
{
    for (typename std::vector<Tensor4d<T>*>::const_iterator it = mData.begin(),
                                                            itEnd = mData.end();
         it != itEnd;
         ++it)
        (*it)->synchronizeDToH();
}

template <typename T>
void N2D2::Interface<T>::synchronizeDToH(unsigned int i,
                                        unsigned int j,
                                        unsigned int k,
                                        unsigned int b,
                                        unsigned int length) const
{
    assert(k < mDimZ);
    assert(b < mDimB);

    const std::pair<unsigned int, unsigned int>& dataOffset = mDataOffset[k];
    mData[dataOffset.first]->synchronizeDToH(i, j, k - dataOffset.second, b,
                                             length);
}

template <typename T> void N2D2::Interface<T>::synchronizeHToD() const
{
    for (typename std::vector<Tensor4d<T>*>::const_iterator it = mData.begin(),
                                                            itEnd = mData.end();
         it != itEnd;
         ++it)
        (*it)->synchronizeHToD();
}

template <typename T>
void N2D2::Interface<T>::synchronizeHToD(unsigned int i,
                                        unsigned int j,
                                        unsigned int k,
                                        unsigned int b,
                                        unsigned int length) const
{
    assert(k < mDimZ);
    assert(b < mDimB);

    const std::pair<unsigned int, unsigned int>& dataOffset = mDataOffset[k];
    mData[dataOffset.first]->synchronizeHToD(i, j, k - dataOffset.second, b,
                                             length);
}

template <class T> void N2D2::Interface<T>::setValid()
{
    for (typename std::vector<Tensor4d<T>*>::iterator it = mData.begin(),
                                                      itEnd = mData.end();
         it != itEnd;
         ++it)
        (*it)->setValid();
}

template <class T> void N2D2::Interface<T>::clearValid()
{
    for (typename std::vector<Tensor4d<T>*>::iterator it = mData.begin(),
                                                      itEnd = mData.end();
         it != itEnd;
         ++it)
        (*it)->clearValid();
}

#endif // N2D2_INTERFACE_H
