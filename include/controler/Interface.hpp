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

#include "containers/Tensor.hpp"

namespace N2D2 {
template< size_t... Ns >
struct indices
{
  typedef indices< Ns..., sizeof...( Ns ) > next;
};

template< size_t N >
struct make_indices
{
  typedef typename make_indices< N - 1 >::type::next type;
};

template<>
struct make_indices< 0 >
{
  typedef indices<> type;
};

/**
 * @class   Interface
 * @brief   Merge virtually several Tensor4d through an unified data interface.
*/
template <class T, int STACKING_DIM = -2>
class Interface {
public:
    typedef typename std::vector<Tensor<T>*>::iterator iterator;
    typedef typename std::vector<Tensor<T>*>::const_iterator const_iterator;
    typedef typename std::vector<T>::reference reference;
    typedef typename std::vector<T>::const_reference const_reference;

    Interface(std::initializer_list<bool> matchingDim
                = std::initializer_list<bool>({true, true, false, true}));
    void matchingDims(std::initializer_list<bool> matchingDims_)
    {
        mMatchingDim = matchingDims_;
    }
    bool empty() const
    {
        return mData.empty();
    }
    /// Historically, for Interface, dimZ is always the stacking dimension
    unsigned int dimZ() const
    {
        return mDataOffset.size();
    }
    unsigned int dimB() const
    {
        return (!mData.empty()) ? mData.back()->dimB() : 0;
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
    inline virtual void push_back(Tensor<T>* tensor);
    inline void clear();
    // Return type should be "reference" (not T&), in order to ensure it works
    // for std::vector<bool>, which is a special case...
    template <typename... Args> reference operator()(Args... args);
    template <typename... Args> const_reference operator()(Args... args) const;
    template <typename... Args> reference at(Args... args);
    template <typename... Args> const_reference at(Args... args) const;
    virtual Tensor<T>& back();
    virtual const Tensor<T>& back() const;
    virtual Tensor<T>& operator[](unsigned int t);
    virtual const Tensor<T>& operator[](unsigned int t) const;
    virtual Tensor<T>& getTensor(unsigned int k, unsigned int* offset = NULL);
    virtual const Tensor<T>& getTensor(unsigned int k,
                                       unsigned int* offset = NULL) const;

    /** Synchronize Device To Host */
    virtual void synchronizeDToH() const;
    template <typename... Args> void synchronizeDToH(Args... args) const;

    /** Synchronize Host To Device */
    virtual void synchronizeHToD() const;
    template <typename... Args> void synchronizeHToD(Args... args) const;

    inline void setValid();
    inline void clearValid();
    virtual ~Interface() {};

protected:
    size_t getOffset(unsigned int dim, size_t i) const;
    template <typename R, typename... Args, size_t... Ns>
    R translateIndexes(unsigned int tensorOffset, indices<Ns...>, Args... args);
    template <typename R, typename... Args, size_t... Ns>
    R translateIndexesAt(unsigned int tensorOffset, indices<Ns...>,
                         Args... args);
    template <typename R, typename... Args, size_t... Ns>
    R translateIndexes(unsigned int tensorOffset, indices<Ns...>, Args... args)
        const;
    template <typename R, typename... Args, size_t... Ns>
    R translateIndexesAt(unsigned int tensorOffset, indices<Ns...>,
                         Args... args) const;
    template <typename R, typename... Args, size_t... Ns>
    R translateIndexesSynchronizeDToH(unsigned int tensorOffset, indices<Ns...>,
                         Args... args) const;
    template <typename R, typename... Args, size_t... Ns>
    R translateIndexesSynchronizeHToD(unsigned int tensorOffset, indices<Ns...>,
                         Args... args) const;

    std::vector<bool> mMatchingDim;
    std::vector<Tensor<T>*> mData;
    std::vector<std::pair<unsigned int, unsigned int> > mDataOffset;
};
}

template <class T, int STACKING_DIM>
N2D2::Interface<T, STACKING_DIM>::Interface(
    std::initializer_list<bool> matchingDim)
    : mMatchingDim(matchingDim)
{
    // ctor
}

template <class T, int STACKING_DIM>
unsigned int N2D2::Interface<T, STACKING_DIM>::dataSize() const
{
    unsigned int size = 0;

    for (typename std::vector<Tensor<T>*>::const_iterator it = mData.begin(),
                                                            itEnd = mData.end();
         it != itEnd;
         ++it)
        size += (*it)->size();

    return size;
}

template <class T, int STACKING_DIM>
void N2D2::Interface<T, STACKING_DIM>::fill(const T& value)
{
    for (typename std::vector<Tensor<T>*>::iterator it = mData.begin(),
                                                      itEnd = mData.end();
         it != itEnd;
         ++it)
        (*it)->fill(value);
}

template <class T, int STACKING_DIM>
void N2D2::Interface<T, STACKING_DIM>::push_back(Tensor<T>* tensor)
{
    if (!mData.empty()) {
        if (tensor->nbDims() != mData.back()->nbDims()) {
            throw std::runtime_error("Interface<T>::push_back(): "
                            "tensor must have the same number of dimensions");
        }

        for (unsigned int dim = 0; dim < tensor->nbDims(); ++dim) {
            if ((STACKING_DIM < 0 || dim != (unsigned int)STACKING_DIM)
                && (STACKING_DIM >= 0
                    || tensor->nbDims() - dim != -STACKING_DIM)
                && dim < mMatchingDim.size() && mMatchingDim[dim]
                && tensor->dims()[dim] != mData.back()->dims()[dim])
            {
                throw std::runtime_error("Interface<T>::push_back(): "
                                         "tensor dimension must match");
            }
        }
    }

    const unsigned int stackingDim = (STACKING_DIM >= 0)
        ? STACKING_DIM : tensor->nbDims() + STACKING_DIM;
    const size_t tensorOffset = mData.size();
    const size_t indexOffset = mDataOffset.size();

    for (size_t index = 0; index < tensor->dims()[stackingDim]; ++index)
        mDataOffset.push_back(std::make_pair(tensorOffset, indexOffset));

    mData.push_back(tensor);
}

template <class T, int STACKING_DIM>
void N2D2::Interface<T, STACKING_DIM>::clear()
{
    mData.clear();
    mDataOffset.clear();
}

template <class T, int STACKING_DIM>
size_t N2D2::Interface<T, STACKING_DIM>::getOffset(unsigned int dim, size_t i) const {
    if ((STACKING_DIM >= 0 && dim == (unsigned int)STACKING_DIM)
        || (STACKING_DIM < 0
            && mData.back()->nbDims() - dim == -STACKING_DIM))
    {
        const std::pair<unsigned int, unsigned int>& dataOffset = mDataOffset[i];
        return (i - dataOffset.second);
    }
    else
        return i;
}

template <class T, int STACKING_DIM>
template <typename R, typename... Args, size_t... Ns>
R N2D2::Interface<T, STACKING_DIM>::translateIndexes(unsigned int tensorOffset,
                                       indices<Ns...>,
                                       Args... args)
{
    return (*mData[tensorOffset])(getOffset(Ns, args)...);
}

template <class T, int STACKING_DIM>
template <typename R, typename... Args, size_t... Ns>
R N2D2::Interface<T, STACKING_DIM>::translateIndexes(unsigned int tensorOffset,
                                       indices<Ns...>,
                                       Args... args) const
{
    return (*mData[tensorOffset])(getOffset(Ns, args)...);
}

template <class T, int STACKING_DIM>
template <typename R, typename... Args, size_t... Ns>
R N2D2::Interface<T, STACKING_DIM>::translateIndexesAt(unsigned int tensorOffset,
                                         indices<Ns...>,
                                         Args... args)
{
    return (*mData[tensorOffset]).at(getOffset(Ns, args)...);
}

template <class T, int STACKING_DIM>
template <typename R, typename... Args, size_t... Ns>
R N2D2::Interface<T, STACKING_DIM>::translateIndexesAt(unsigned int tensorOffset,
                                         indices<Ns...>,
                                         Args... args) const
{
    return (*mData[tensorOffset]).at(getOffset(Ns, args)...);
}

template <class T, int STACKING_DIM>
template <typename R, typename... Args, size_t... Ns>
R N2D2::Interface<T, STACKING_DIM>::translateIndexesSynchronizeDToH(unsigned int tensorOffset,
                                         indices<Ns...>,
                                         Args... args) const
{
    return (*mData[tensorOffset]).synchronizeDToH(getOffset(Ns, args)...);
}

template <class T, int STACKING_DIM>
template <typename R, typename... Args, size_t... Ns>
R N2D2::Interface<T, STACKING_DIM>::translateIndexesSynchronizeHToD(unsigned int tensorOffset,
                                         indices<Ns...>,
                                         Args... args) const
{
    return (*mData[tensorOffset]).synchronizeHToD(getOffset(Ns, args)...);
}

template <class T, int STACKING_DIM>
template <typename... Args>
typename N2D2::Interface<T, STACKING_DIM>::reference N2D2::Interface<T, STACKING_DIM>::
operator()(Args... args)
{
    assert(!mData.empty());
    assert(sizeof...(args) == (*mData.back()).nbDims());

    const unsigned int stackingDim = (STACKING_DIM >= 0)
        ? STACKING_DIM : (*mData.back()).nbDims() + STACKING_DIM;
    const std::vector<size_t> indexes = {static_cast<size_t>(args)...};
    const std::pair<unsigned int, unsigned int>& dataOffset
        = mDataOffset[indexes[stackingDim]];

    return translateIndexes<typename N2D2::Interface<T, STACKING_DIM>::reference>(
                                dataOffset.first,
                                typename make_indices<sizeof...(Args)>::type(),
                                args...);
}

template <class T, int STACKING_DIM>
template <typename... Args>
typename N2D2::Interface<T, STACKING_DIM>::const_reference N2D2::Interface<T, STACKING_DIM>::
operator()(Args... args) const
{
    assert(!mData.empty());
    assert(sizeof...(args) == (*mData.back()).nbDims());

    const unsigned int stackingDim = (STACKING_DIM >= 0)
        ? STACKING_DIM : (*mData.back()).nbDims() + STACKING_DIM;
    const std::vector<size_t> indexes = {static_cast<size_t>(args)...};
    const std::pair<unsigned int, unsigned int>& dataOffset
        = mDataOffset[indexes[stackingDim]];

    return translateIndexes<typename N2D2::Interface<T, STACKING_DIM>::const_reference>(
                                dataOffset.first,
                                typename make_indices<sizeof...(Args)>::type(),
                                args...);
}

template <class T, int STACKING_DIM>
template <typename... Args>
typename N2D2::Interface<T, STACKING_DIM>::reference N2D2::Interface<T, STACKING_DIM>::at(Args... args)
{
    if (mData.empty() || sizeof...(args) != (*mData.back()).nbDims())
        throw std::runtime_error("Argument count must match tensor dimension");

    const unsigned int stackingDim = (STACKING_DIM >= 0)
        ? STACKING_DIM : (*mData.back()).nbDims() + STACKING_DIM;
    const std::vector<size_t> indexes = {static_cast<size_t>(args)...};
    const std::pair<unsigned int, unsigned int>& dataOffset
        = mDataOffset[indexes[stackingDim]];

    return translateIndexesAt<typename N2D2::Interface<T, STACKING_DIM>::reference>(
                                  dataOffset.first,
                                  typename make_indices<sizeof...(Args)>::type(),
                                  args...);
}

template <class T, int STACKING_DIM>
template <typename... Args>
typename N2D2::Interface<T, STACKING_DIM>::const_reference N2D2::Interface<T, STACKING_DIM>::
    at(Args... args) const
{
    if (mData.empty() || sizeof...(args) != (*mData.back()).nbDims())
        throw std::runtime_error("Argument count must match tensor dimension");

    const unsigned int stackingDim = (STACKING_DIM >= 0)
        ? STACKING_DIM : (*mData.back()).nbDims() + STACKING_DIM;
    const std::vector<size_t> indexes = {static_cast<size_t>(args)...};
    const std::pair<unsigned int, unsigned int>& dataOffset
        = mDataOffset[indexes[stackingDim]];

    return translateIndexesAt<typename N2D2::Interface<T, STACKING_DIM>::const_reference>(
                                  dataOffset.first,
                                  typename make_indices<sizeof...(Args)>::type(),
                                  args...);
}

template <class T, int STACKING_DIM>
N2D2::Tensor<T>& N2D2::Interface<T, STACKING_DIM>::back()
{
    return *(mData.back());
}

template <class T, int STACKING_DIM>
const N2D2::Tensor<T>& N2D2::Interface<T, STACKING_DIM>::back() const
{
    return *(mData.back());
}

template <class T, int STACKING_DIM>
N2D2::Tensor<T>& N2D2::Interface<T, STACKING_DIM>::operator[](unsigned int t)
{
    return *(mData.at(t));
}

template <class T, int STACKING_DIM>
const N2D2::Tensor<T>& N2D2::Interface<T, STACKING_DIM>::operator[](unsigned int t) const
{
    return *(mData.at(t));
}

template <class T, int STACKING_DIM>
N2D2::Tensor<T>& N2D2::Interface<T, STACKING_DIM>::getTensor(unsigned int k,
                                                        unsigned int* offset)
{
    assert(k < mDataOffset.size());

    const std::pair<unsigned int, unsigned int>& dataOffset = mDataOffset.at(k);

    if (offset != NULL)
        (*offset) = dataOffset.second;

    return *(mData[dataOffset.first]);
}

template <class T, int STACKING_DIM>
const N2D2::Tensor<T>& N2D2::Interface<T, STACKING_DIM>::getTensor(
    unsigned int k,
    unsigned int* offset) const
{
    assert(k < mDataOffset.size());

    const std::pair<unsigned int, unsigned int>& dataOffset = mDataOffset.at(k);

    if (offset != NULL)
        (*offset) = dataOffset.second;

    return *(mData[dataOffset.first]);
}

template <class T, int STACKING_DIM>
void N2D2::Interface<T, STACKING_DIM>::synchronizeDToH() const
{
    for (typename std::vector<Tensor<T>*>::const_iterator it = mData.begin(),
                                                            itEnd = mData.end();
         it != itEnd;
         ++it)
        (*it)->synchronizeDToH();
}

template <class T, int STACKING_DIM>
template <typename... Args>
void N2D2::Interface<T, STACKING_DIM>::synchronizeDToH(Args... args) const
{
    assert(!mData.empty());
    assert(sizeof...(args) == (*mData.back()).nbDims() + 1);

    const unsigned int stackingDim = (STACKING_DIM >= 0)
        ? STACKING_DIM : (*mData.back()).nbDims() + STACKING_DIM;
    const std::vector<size_t> indexes = {static_cast<size_t>(args)...};
    const std::pair<unsigned int, unsigned int>& dataOffset
        = mDataOffset[indexes[stackingDim]];

    return translateIndexesSynchronizeDToH<void>(
                                dataOffset.first,
                                typename make_indices<sizeof...(Args)>::type(),
                                args...);
}

template <class T, int STACKING_DIM>
void N2D2::Interface<T, STACKING_DIM>::synchronizeHToD() const
{
    for (typename std::vector<Tensor<T>*>::const_iterator it = mData.begin(),
                                                            itEnd = mData.end();
         it != itEnd;
         ++it)
        (*it)->synchronizeHToD();
}

template <class T, int STACKING_DIM>
template <typename... Args>
void N2D2::Interface<T, STACKING_DIM>::synchronizeHToD(Args... args) const
{
    assert(!mData.empty());
    assert(sizeof...(args) == (*mData.back()).nbDims() + 1);

    const unsigned int stackingDim = (STACKING_DIM >= 0)
        ? STACKING_DIM : (*mData.back()).nbDims() + STACKING_DIM;
    const std::vector<size_t> indexes = {static_cast<size_t>(args)...};
    const std::pair<unsigned int, unsigned int>& dataOffset
        = mDataOffset[indexes[stackingDim]];

    return translateIndexesSynchronizeHToD<void>(
                                dataOffset.first,
                                typename make_indices<sizeof...(Args)>::type(),
                                args...);
}

template <class T, int STACKING_DIM>
void N2D2::Interface<T, STACKING_DIM>::setValid()
{
    for (typename std::vector<Tensor<T>*>::iterator it = mData.begin(),
                                                      itEnd = mData.end();
         it != itEnd;
         ++it)
        (*it)->setValid();
}

template <class T, int STACKING_DIM>
void N2D2::Interface<T, STACKING_DIM>::clearValid()
{
    for (typename std::vector<Tensor<T>*>::iterator it = mData.begin(),
                                                      itEnd = mData.end();
         it != itEnd;
         ++it)
        (*it)->clearValid();
}

#endif // N2D2_INTERFACE_H
