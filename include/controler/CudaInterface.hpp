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

#ifndef N2D2_CUDAINTERFACE_H
#define N2D2_CUDAINTERFACE_H

#include <cassert>
#include <iostream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <vector>

#include "controler/Interface.hpp"

namespace N2D2 {
/**
 * @class   CudaInterface
 * @brief   Merge virtually several Tensor4d through an unified data interface.
*/
template <typename T> class CudaInterface : public Interface<T> {
public:
    using Interface<T>::push_back;

    CudaInterface();

    /**
    * Add a CudaTensor4d to the interface
    *
    * @param tensor   tensor to add to the interface.
    */
    void push_back(CudaTensor4d<T>* tensor);

    /** Synchronize Device To Host */
    void synchronizeDToH() const;
    void synchronizeDToH(unsigned int i,
                         unsigned int j,
                         unsigned int k,
                         unsigned int b,
                         unsigned int length) const;

    /** Synchronize Host To Device */
    void synchronizeHToD() const;
    void synchronizeHToD(unsigned int i,
                         unsigned int j,
                         unsigned int k,
                         unsigned int b,
                         unsigned int length) const;

    virtual CudaTensor4d<T>& back();
    virtual const CudaTensor4d<T>& back() const;
    virtual CudaTensor4d<T>& operator[](unsigned int t);
    virtual const CudaTensor4d<T>& operator[](unsigned int t) const;
    virtual CudaTensor4d<T>& getTensor4d(unsigned int k);
    ~CudaInterface() {};

private:
    using Interface<T>::mDimZ;
    using Interface<T>::mDimB;
    using Interface<T>::mData;
    using Interface<T>::mDataOffset;
};
}

template <typename T> N2D2::CudaInterface<T>::CudaInterface() : Interface<T>()
{
    // if (sizeof(T) != sizeof(float))
    //    throw std::runtime_error("Types must match");
}

template <typename T>
void N2D2::CudaInterface<T>::push_back(CudaTensor4d<T>* tensor)
{
    Interface<T>::push_back(tensor);
}

template <typename T> void N2D2::CudaInterface<T>::synchronizeDToH() const
{
    for (typename std::vector<Tensor4d<T>*>::const_iterator it = mData.begin(),
                                                            itEnd = mData.end();
         it != itEnd;
         ++it)
        static_cast<CudaTensor4d<T>*>(*it)->synchronizeDToH();
}

template <typename T>
void N2D2::CudaInterface<T>::synchronizeDToH(unsigned int i,
                                             unsigned int j,
                                             unsigned int k,
                                             unsigned int b,
                                             unsigned int length) const
{
    assert(k < mDimZ);
    assert(b < mDimB);

    const std::pair<unsigned int, unsigned int>& dataOffset = mDataOffset[k];
    CudaTensor4d<T>* tensor = static_cast
        <CudaTensor4d<T>*>(mData[dataOffset.first]);
    tensor->synchronizeDToH(i, j, k - dataOffset.second, b, length);
}

template <typename T> void N2D2::CudaInterface<T>::synchronizeHToD() const
{
    for (typename std::vector<Tensor4d<T>*>::const_iterator it = mData.begin(),
                                                            itEnd = mData.end();
         it != itEnd;
         ++it)
        static_cast<CudaTensor4d<T>*>(*it)->synchronizeHToD();
}

template <typename T>
void N2D2::CudaInterface<T>::synchronizeHToD(unsigned int i,
                                             unsigned int j,
                                             unsigned int k,
                                             unsigned int b,
                                             unsigned int length) const
{
    assert(k < mDimZ);
    assert(b < mDimB);

    const std::pair<unsigned int, unsigned int>& dataOffset = mDataOffset[k];
    CudaTensor4d<T>* tensor = static_cast
        <CudaTensor4d<T>*>(mData[dataOffset.first]);
    tensor->synchronizeHToD(i, j, k - dataOffset.second, b, length);
}

template <class T> N2D2::CudaTensor4d<T>& N2D2::CudaInterface<T>::back()
{
    return *static_cast<CudaTensor4d<T>*>(mData.back());
}

template <class T>
const N2D2::CudaTensor4d<T>& N2D2::CudaInterface<T>::back() const
{
    return *static_cast<CudaTensor4d<T>*>(mData.back());
}

template <class T>
N2D2::CudaTensor4d<T>& N2D2::CudaInterface<T>::operator[](unsigned int t)
{
    return *static_cast<CudaTensor4d<T>*>(mData.at(t));
}

template <class T>
const N2D2::CudaTensor4d<T>& N2D2::CudaInterface<T>::
operator[](unsigned int t) const
{
    return *static_cast<CudaTensor4d<T>*>(mData.at(t));
}

template <class T>
N2D2::CudaTensor4d<T>& N2D2::CudaInterface<T>::getTensor4d(unsigned int k)
{
    assert(k < mDimZ);

    const std::pair<unsigned int, unsigned int>& dataOffset = mDataOffset.at(k);
    return *static_cast<CudaTensor4d<T>*>(mData[dataOffset.first]);
}

#endif // N2D2_CUDAINTERFACE_H
