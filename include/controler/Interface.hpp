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

template <typename SELF, int STACKING_DIM, typename T>
class TypedInterface {
public:
    typedef typename Tensor<T>::reference reference;
    typedef typename Tensor<T>::const_reference const_reference;

    inline void fill(const T& value);
    // Return type should be "reference" (not T&), in order to ensure it works
    // for std::vector<bool>, which is a special case...
    template <typename... Args> reference operator()(Args... args);
    template <typename... Args> const_reference operator()(Args... args) const;
    template <typename... Args> reference at(Args... args);
    template <typename... Args> const_reference at(Args... args) const;

protected:
    size_t getOffset(unsigned int dim, size_t i) const;
    template <typename... Args, size_t... Ns>
    reference translateIndexes(unsigned int tensorOffset, indices<Ns...>, Args... args);
    template <typename... Args, size_t... Ns>
    reference translateIndexesAt(unsigned int tensorOffset, indices<Ns...>,
                         Args... args);
    template <typename... Args, size_t... Ns>
    const_reference translateIndexes(unsigned int tensorOffset, indices<Ns...>, Args... args)
        const;
    template <typename... Args, size_t... Ns>
    const_reference translateIndexesAt(unsigned int tensorOffset, indices<Ns...>,
                         Args... args) const;

private:
    SELF* self() { return static_cast<SELF*>(this); };
    const SELF* self() const { return static_cast<const SELF*>(this); };
};

template <typename SELF, int STACKING_DIM>
class TypedInterface<SELF, STACKING_DIM, void> {

};

template <class T>
struct tensor_t {
    typedef Tensor<T> type;
};

template <>
struct tensor_t<void> {
    typedef BaseTensor type;
};

class BaseInterface {
public:
    virtual bool empty() const = 0;
    virtual size_t dimX() const = 0;
    virtual size_t dimY() const = 0;
    virtual size_t dimD() const = 0;
    virtual size_t dimZ() const = 0;
    virtual size_t dimB() const = 0;
    virtual size_t size() const = 0;
    virtual size_t dataSize() const = 0;
    virtual BaseTensor& back() = 0;
    virtual const BaseTensor& back() const = 0;
    virtual BaseTensor& operator[](unsigned int t) = 0;
    virtual const BaseTensor& operator[](unsigned int t) const = 0;
    virtual ~BaseInterface() {};
};

/**
 * @class   Interface
 * @brief   Merge virtually several Tensor through an unified data interface.
*/
template <class T = void, int STACKING_DIM = -2>
class Interface : public BaseInterface,
            public TypedInterface<Interface<T, STACKING_DIM>, STACKING_DIM, T> {
public:
    typedef typename tensor_t<T>::type tensor_type;
    typedef typename std::vector<tensor_type*>::iterator iterator;
    typedef typename std::vector<tensor_type*>::const_iterator const_iterator;

    template <class U, int V> Interface(const Interface<U, V>& interface);
    Interface(std::initializer_list<bool> matchingDim);
    Interface();

    void matchingDims(std::initializer_list<bool> matchingDims_)
    {
        mMatchingDim = matchingDims_;
    }
    bool empty() const
    {
        return mData.empty();
    }
    size_t dimX() const
    {
        return (!mData.empty() && mData.back()->nbDims() > 0) ? dataDim(0) : 0;
    }
    size_t dimY() const
    {
        return (!mData.empty() && mData.back()->nbDims() > 1) ? dataDim(1) : 0;
    }
    size_t dimD() const
    {
        return (!mData.empty() && mData.back()->nbDims() > 2) ? dataDim(2) : 0;
    }
    /// See Tensor for the special meaning of dimZ()
    size_t dimZ() const
    {
        return (!mData.empty() && mData.back()->nbDims() > 1)
            ? dataDim(
                (mData.back()->nbDims() > 3)    ? mData.back()->nbDims() - 2 :
                (mData.back()->nbDims() == 3)   ? 2
                                                : 0)
            : 0;
    }
    /// See Tensor for the special meaning of dimB()
    size_t dimB() const
    {
        return (!mData.empty() && mData.back()->nbDims() > 0)
            ? dataDim(mData.back()->nbDims() - 1) : 0;
    }
    size_t size() const
    {
        return mData.size();
    }
    size_t dataSize() const;
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
    inline virtual void push_back(tensor_type* tensor, size_t refs = 1);
    void swap(Interface<T, STACKING_DIM>& interface);
    inline void clear();

    virtual tensor_type& back();
    virtual const tensor_type& back() const;
    virtual tensor_type& operator[](unsigned int t);
    virtual const tensor_type& operator[](unsigned int t) const;
    virtual void replace(unsigned int t, tensor_type* tensor, size_t refs = 1);
    inline unsigned int getTensorIndex(unsigned int k) const;
    inline unsigned int getTensorDataOffset(unsigned int k) const;
    inline size_t getDataRefs(unsigned int t) const
    {
        return mDataRefs.at(t);
    }
    inline void setDataRefs(unsigned int t, size_t refs)
    {
        mDataRefs.at(t) = refs;
    }

    /** Synchronize Device To Host */
    virtual void synchronizeDToH() const;
    template <typename... Args> void synchronizeDToH(Args... args) const;

    /** Synchronize Host To Device */
    virtual void synchronizeHToD() const;
    template <typename... Args> void synchronizeHToD(Args... args) const;

    /** Synchronize Device To Host-based data  */
    virtual void synchronizeDToHBased() const;

    /** Synchronize Host-based data To Device */
    virtual void synchronizeHBasedToD() const;

    /** Synchronize Device-based data To Host  */
    virtual void synchronizeDBasedToH() const;

    /** Synchronize Host data To Device-based */
    virtual void synchronizeHToDBased() const;

    inline void setValid();
    inline void clearValid();
    virtual ~Interface();

protected:
    template <typename... Args, size_t... Ns>
    void translateIndexesSynchronizeDToH(unsigned int tensorOffset, indices<Ns...>,
                         Args... args) const;
    template <typename... Args, size_t... Ns>
    void translateIndexesSynchronizeHToD(unsigned int tensorOffset, indices<Ns...>,
                         Args... args) const;
    size_t dataDim(unsigned int dim) const {
        assert(!mData.empty());
        assert(dim < mData.back()->nbDims());

        const unsigned int stackingDim = (STACKING_DIM >= 0)
            ? STACKING_DIM : mData.back()->nbDims() + STACKING_DIM;

        return (stackingDim == dim) ? mDataOffset.size()
                                    : mData.back()->dims()[dim];

        // TODO: preferred behavior should be to return 0 if the dim is not a
        // matching dim. However, this is incompatible with current
        // ROIPoolingCell implementation, which rely on this behavior in
        // Cell_Frame*::addInput() (which uses mInputs.dimB())
/*
        return (stackingDim == dim) ? mDataOffset.size() :
            (mMatchingDim.size() > dim && mMatchingDim[dim])
                ? mData.back()->dims()[dim]
            : 0;
*/
    }

    std::vector<bool> mMatchingDim;
    std::vector<tensor_type*> mData;
    std::vector<size_t> mDataRefs;
    std::vector<std::pair<unsigned int, unsigned int> > mDataOffset;

    friend class TypedInterface<Interface<T, STACKING_DIM>, STACKING_DIM, T>;
    template <class, int> friend class Interface;
};
}

template <class T, int STACKING_DIM>
template <class U, int V>
N2D2::Interface<T, STACKING_DIM>::Interface(
    const Interface<U, V>& interface)
    : mMatchingDim(interface.mMatchingDim),
      mDataOffset(interface.mDataOffset)
{
    // copy-ctor
    for (size_t k = 0; k < interface.size(); ++k) {
        tensor_type* tensor = dynamic_cast<tensor_type*>(interface[k]);

        if (tensor == NULL) {
            throw std::runtime_error("Interface::Interface(): "
                            "incompatible tensor data type");
        }

        mData.push_back(tensor);
        mDataRefs.push_back(interface.mDataRefs[k] + 1);
    }
}

template <class T, int STACKING_DIM>
N2D2::Interface<T, STACKING_DIM>::Interface(
    std::initializer_list<bool> matchingDim)
    : mMatchingDim(matchingDim)
{
    // ctor
}
template <class T, int STACKING_DIM>
N2D2::Interface<T, STACKING_DIM>::Interface()
    : mMatchingDim({true, true, false, true})
{
    // ctor
}


template <class T, int STACKING_DIM>
size_t N2D2::Interface<T, STACKING_DIM>::dataSize() const
{
    size_t size = 0;

    for (typename std::vector<tensor_type*>::const_iterator it = mData.begin(),
                                                            itEnd = mData.end();
         it != itEnd;
         ++it)
        size += (*it)->size();

    return size;
}

template <typename SELF, int STACKING_DIM, typename T>
void N2D2::TypedInterface<SELF, STACKING_DIM, T>::fill(const T& value)
{
    for (auto it = self()->mData.begin(), itEnd = self()->mData.end();
        it != itEnd; ++it)
    {
        Tensor<T>* tensor = dynamic_cast<Tensor<T>*>(*it);

        if (tensor != NULL)
            tensor->fill(value);
        else {
            throw std::runtime_error("Interface::fill(): "
                            "incompatible tensor data type");
        }
    }
}

template <class T, int STACKING_DIM>
void N2D2::Interface<T, STACKING_DIM>::push_back(tensor_type* tensor,
                                                 size_t refs)
{
    // Find the first non-empty tensor in the interface
    tensor_type* prevTensor = NULL;

    for (unsigned int k = 0; k < mData.size(); ++k) {
        if (!mData[k]->empty()) {
            prevTensor = mData[k];
            break;
        }
    }

    // If there is already a non-empty tensor, check the dimensions
    if (prevTensor != NULL && !tensor->empty()) {
        if (tensor->nbDims() != prevTensor->nbDims()) {
            throw std::runtime_error("Interface::push_back(): "
                            "tensor must have the same number of dimensions");
        }

        for (unsigned int dim = 0; dim < tensor->nbDims(); ++dim) {
            if ((STACKING_DIM < 0 || dim != (unsigned int)STACKING_DIM)
                && (STACKING_DIM >= 0
                    || tensor->nbDims() - dim != -STACKING_DIM)
                && dim < mMatchingDim.size() && mMatchingDim[dim]
                && tensor->dims()[dim] != prevTensor->dims()[dim])
            {
                throw std::runtime_error("Interface::push_back(): "
                                         "tensor dimension must match");
            }
        }
    }

    if (!tensor->empty()) {
        const unsigned int stackingDim = (STACKING_DIM >= 0)
            ? STACKING_DIM : tensor->nbDims() + STACKING_DIM;
        const size_t tensorOffset = mData.size();
        const size_t indexOffset = mDataOffset.size();

        for (size_t index = 0; index < tensor->dims()[stackingDim]; ++index)
            mDataOffset.push_back(std::make_pair(tensorOffset, indexOffset));
    }

    mData.push_back(tensor);
    mDataRefs.push_back(refs);
}

template <class T, int STACKING_DIM>
void N2D2::Interface<T, STACKING_DIM>::swap(
    Interface<T, STACKING_DIM>& interface)
{
    mMatchingDim.swap(interface.mMatchingDim);
    mData.swap(interface.mData);
    mDataRefs.swap(interface.mDataRefs);
    mDataOffset.swap(interface.mDataOffset);
}

template <class T, int STACKING_DIM>
void N2D2::Interface<T, STACKING_DIM>::clear()
{
    for (size_t k = 0; k < mData.size(); ++k) {
        if (mDataRefs[k] == 0)
            delete mData[k];
    }

    mData.clear();
    mDataRefs.clear();
    mDataOffset.clear();
}

template <typename SELF, int STACKING_DIM, typename T>
size_t N2D2::TypedInterface<SELF, STACKING_DIM, T>::getOffset(unsigned int dim, size_t i) const {
    if ((STACKING_DIM >= 0 && dim == (unsigned int)STACKING_DIM)
        || (STACKING_DIM < 0
            && self()->mData.back()->nbDims() - dim == -STACKING_DIM))
    {
        const std::pair<unsigned int, unsigned int>& dataOffset = self()->mDataOffset[i];
        return (i - dataOffset.second);
    }
    else
        return i;
}

template <typename SELF, int STACKING_DIM, typename T>
template <typename... Args, size_t... Ns>
typename N2D2::Tensor<T>::reference N2D2::TypedInterface<SELF, STACKING_DIM, T>::translateIndexes(unsigned int tensorOffset,
                                       indices<Ns...>,
                                       Args... args)
{
    return (*static_cast<Tensor<T>*>(self()->mData[tensorOffset]))(getOffset(Ns, args)...);
}

template <typename SELF, int STACKING_DIM, typename T>
template <typename... Args, size_t... Ns>
typename N2D2::Tensor<T>::const_reference N2D2::TypedInterface<SELF, STACKING_DIM, T>::translateIndexes(unsigned int tensorOffset,
                                       indices<Ns...>,
                                       Args... args) const
{
    return (*static_cast<Tensor<T>*>(self()->mData[tensorOffset]))(getOffset(Ns, args)...);
}

template <typename SELF, int STACKING_DIM, typename T>
template <typename... Args, size_t... Ns>
typename N2D2::Tensor<T>::reference N2D2::TypedInterface<SELF, STACKING_DIM, T>::translateIndexesAt(unsigned int tensorOffset,
                                         indices<Ns...>,
                                         Args... args)
{
    return (*static_cast<Tensor<T>*>(self()->mData[tensorOffset])).at(getOffset(Ns, args)...);
}

template <typename SELF, int STACKING_DIM, typename T>
template <typename... Args, size_t... Ns>
typename N2D2::Tensor<T>::const_reference N2D2::TypedInterface<SELF, STACKING_DIM, T>::translateIndexesAt(unsigned int tensorOffset,
                                         indices<Ns...>,
                                         Args... args) const
{
    return (*static_cast<Tensor<T>*>(self()->mData[tensorOffset])).at(getOffset(Ns, args)...);
}

template <class T, int STACKING_DIM>
template <typename... Args, size_t... Ns>
void N2D2::Interface<T, STACKING_DIM>::translateIndexesSynchronizeDToH(unsigned int tensorOffset,
                                         indices<Ns...>,
                                         Args... args) const
{
    return (*mData[tensorOffset]).synchronizeDToH(this->getOffset(Ns, args)...);
}

template <class T, int STACKING_DIM>
template <typename... Args, size_t... Ns>
void N2D2::Interface<T, STACKING_DIM>::translateIndexesSynchronizeHToD(unsigned int tensorOffset,
                                         indices<Ns...>,
                                         Args... args) const
{
    return (*mData[tensorOffset]).synchronizeHToD(this->getOffset(Ns, args)...);
}

template <typename SELF, int STACKING_DIM, typename T>
template <typename... Args>
typename N2D2::TypedInterface<SELF, STACKING_DIM, T>::reference N2D2::TypedInterface<SELF, STACKING_DIM, T>::
operator()(Args... args)
{
    assert(!self()->mData.empty());
    assert(sizeof...(args) == (*self()->mData.back()).nbDims());

    const unsigned int stackingDim = (STACKING_DIM >= 0)
        ? STACKING_DIM : (*self()->mData.back()).nbDims() + STACKING_DIM;
    const std::vector<size_t> indexes = {static_cast<size_t>(args)...};
    const std::pair<unsigned int, unsigned int>& dataOffset
        = self()->mDataOffset[indexes[stackingDim]];

    return translateIndexes(
                                dataOffset.first,
                                typename make_indices<sizeof...(Args)>::type(),
                                args...);
}

template <typename SELF, int STACKING_DIM, typename T>
template <typename... Args>
typename N2D2::TypedInterface<SELF, STACKING_DIM, T>::const_reference N2D2::TypedInterface<SELF, STACKING_DIM, T>::
operator()(Args... args) const
{
    assert(!self()->mData.empty());
    assert(sizeof...(args) == (*self()->mData.back()).nbDims());

    const unsigned int stackingDim = (STACKING_DIM >= 0)
        ? STACKING_DIM : (*self()->mData.back()).nbDims() + STACKING_DIM;
    const std::vector<size_t> indexes = {static_cast<size_t>(args)...};
    const std::pair<unsigned int, unsigned int>& dataOffset
        = self()->mDataOffset[indexes[stackingDim]];

    return translateIndexes(
                                dataOffset.first,
                                typename make_indices<sizeof...(Args)>::type(),
                                args...);
}

template <typename SELF, int STACKING_DIM, typename T>
template <typename... Args>
typename N2D2::TypedInterface<SELF, STACKING_DIM, T>::reference N2D2::TypedInterface<SELF, STACKING_DIM, T>::at(Args... args)
{
    if (self()->mData.empty() || sizeof...(args) != (*self()->mData.back()).nbDims())
        throw std::runtime_error("Argument count must match tensor dimension");

    const unsigned int stackingDim = (STACKING_DIM >= 0)
        ? STACKING_DIM : (*self()->mData.back()).nbDims() + STACKING_DIM;
    const std::vector<size_t> indexes = {static_cast<size_t>(args)...};
    const std::pair<unsigned int, unsigned int>& dataOffset
        = self()->mDataOffset[indexes[stackingDim]];

    return translateIndexesAt(
                                  dataOffset.first,
                                  typename make_indices<sizeof...(Args)>::type(),
                                  args...);
}

template <typename SELF, int STACKING_DIM, typename T>
template <typename... Args>
typename N2D2::TypedInterface<SELF, STACKING_DIM, T>::const_reference N2D2::TypedInterface<SELF, STACKING_DIM, T>::
    at(Args... args) const
{
    if (self()->mData.empty() || sizeof...(args) != (*self()->mData.back()).nbDims())
        throw std::runtime_error("Argument count must match tensor dimension");

    const unsigned int stackingDim = (STACKING_DIM >= 0)
        ? STACKING_DIM : (*self()->mData.back()).nbDims() + STACKING_DIM;
    const std::vector<size_t> indexes = {static_cast<size_t>(args)...};
    const std::pair<unsigned int, unsigned int>& dataOffset
        = self()->mDataOffset[indexes[stackingDim]];

    return translateIndexesAt(
                                  dataOffset.first,
                                  typename make_indices<sizeof...(Args)>::type(),
                                  args...);
}

template <class T, int STACKING_DIM>
typename N2D2::Interface<T, STACKING_DIM>::tensor_type& N2D2::Interface<T, STACKING_DIM>::back()
{
    return *(mData.back());
}

template <class T, int STACKING_DIM>
const typename N2D2::Interface<T, STACKING_DIM>::tensor_type& N2D2::Interface<T, STACKING_DIM>::back() const
{
    return *(mData.back());
}

template <class T, int STACKING_DIM>
typename N2D2::Interface<T, STACKING_DIM>::tensor_type& N2D2::Interface<T, STACKING_DIM>::operator[](unsigned int t)
{
    return *(mData.at(t));
}

template <class T, int STACKING_DIM>
const typename N2D2::Interface<T, STACKING_DIM>::tensor_type& N2D2::Interface<T, STACKING_DIM>::operator[](unsigned int t) const
{
    return *(mData.at(t));
}

template <class T, int STACKING_DIM>
void N2D2::Interface<T, STACKING_DIM>::replace(unsigned int t,
                                               tensor_type* tensor,
                                               size_t refs)
{
    assert(t < mData.size());

    const unsigned int stackingDim = (STACKING_DIM >= 0)
        ? STACKING_DIM : tensor->nbDims() + STACKING_DIM;

    if (mData.size() > 1) {
        if (tensor->nbDims() != mData.back()->nbDims()) {
            throw std::runtime_error("Interface::replace(): "
                            "tensor must have the same number of dimensions");
        }

        for (unsigned int dim = 0; dim < tensor->nbDims(); ++dim) {
            if ((STACKING_DIM < 0 || dim != (unsigned int)STACKING_DIM)
                && (STACKING_DIM >= 0
                    || tensor->nbDims() - dim != -STACKING_DIM)
                && dim < mMatchingDim.size() && mMatchingDim[dim]
                && tensor->dims()[dim] != mData.back()->dims()[dim])
            {
                throw std::runtime_error("Interface::replace(): "
                                            "tensor dimension must match");
            }
        }

        if (tensor->dims()[stackingDim] != mData.at(t)->dims()[stackingDim]) {
            throw std::runtime_error("Interface::replace(): the new tensor must"
                                    " have the same number of element in the"
                                    " stacking dimension as the replaced one.");
        }
    }
    else {
        const size_t tensorOffset = 0;
        const size_t indexOffset = 0;

        mDataOffset.clear();
        for (size_t index = 0; index < tensor->dims()[stackingDim]; ++index)
            mDataOffset.push_back(std::make_pair(tensorOffset, indexOffset));
    }

    if (mDataRefs[t] == 0)
        delete mData[t];

    mData[t] = tensor;
    mDataRefs[t] = refs;
}

template <class T, int STACKING_DIM>
unsigned int N2D2::Interface<T, STACKING_DIM>::getTensorIndex(unsigned int k)
    const
{
    const std::pair<unsigned int, unsigned int>& dataOffset = mDataOffset.at(k);
    return dataOffset.first;
}

template <class T, int STACKING_DIM>
unsigned int N2D2::Interface<T, STACKING_DIM>::getTensorDataOffset(
    unsigned int k) const
{
    const std::pair<unsigned int, unsigned int>& dataOffset = mDataOffset.at(k);
    return dataOffset.second;
}

template <class T, int STACKING_DIM>
void N2D2::Interface<T, STACKING_DIM>::synchronizeDToH() const
{
    for (typename std::vector<tensor_type*>::const_iterator it = mData.begin(),
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

    translateIndexesSynchronizeDToH(
                                dataOffset.first,
                                typename make_indices<sizeof...(Args)>::type(),
                                args...);
}

template <class T, int STACKING_DIM>
void N2D2::Interface<T, STACKING_DIM>::synchronizeHToD() const
{
    for (typename std::vector<tensor_type*>::const_iterator it = mData.begin(),
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

    translateIndexesSynchronizeHToD(
                                dataOffset.first,
                                typename make_indices<sizeof...(Args)>::type(),
                                args...);
}

template <class T, int STACKING_DIM>
void N2D2::Interface<T, STACKING_DIM>::synchronizeDToHBased() const
{
    for (typename std::vector<tensor_type*>::const_iterator it = mData.begin(),
                                                            itEnd = mData.end();
         it != itEnd;
         ++it)
        (*it)->synchronizeDToHBased();
}

template <class T, int STACKING_DIM>
void N2D2::Interface<T, STACKING_DIM>::synchronizeHBasedToD() const
{
    for (typename std::vector<tensor_type*>::const_iterator it = mData.begin(),
                                                            itEnd = mData.end();
         it != itEnd;
         ++it)
        (*it)->synchronizeHBasedToD();
}

template <class T, int STACKING_DIM>
void N2D2::Interface<T, STACKING_DIM>::synchronizeDBasedToH() const
{
    for (typename std::vector<tensor_type*>::const_iterator it = mData.begin(),
                                                            itEnd = mData.end();
         it != itEnd;
         ++it)
        (*it)->synchronizeDBasedToH();
}

template <class T, int STACKING_DIM>
void N2D2::Interface<T, STACKING_DIM>::synchronizeHToDBased() const
{
    for (typename std::vector<tensor_type*>::const_iterator it = mData.begin(),
                                                            itEnd = mData.end();
         it != itEnd;
         ++it)
        (*it)->synchronizeHToDBased();
}

template <class T, int STACKING_DIM>
void N2D2::Interface<T, STACKING_DIM>::setValid()
{
    for (typename std::vector<tensor_type*>::iterator it = mData.begin(),
                                                      itEnd = mData.end();
         it != itEnd;
         ++it)
        (*it)->setValid();
}

template <class T, int STACKING_DIM>
void N2D2::Interface<T, STACKING_DIM>::clearValid()
{
    for (typename std::vector<tensor_type*>::iterator it = mData.begin(),
                                                      itEnd = mData.end();
         it != itEnd;
         ++it)
        (*it)->clearValid();
}

template <class T, int STACKING_DIM>
N2D2::Interface<T, STACKING_DIM>::~Interface()
{
    clear();
}

#endif // N2D2_INTERFACE_H
