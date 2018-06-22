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

#include "containers/CudaTensor.hpp"
#include "controler/Interface.hpp"

namespace N2D2 {
/**
 * @class   CudaInterface
 * @brief   Merge virtually several Tensor4d through an unified data interface.
*/
template <class T, int STACKING_DIM = -2>
class CudaInterface : public Interface<T, STACKING_DIM> {
public:
    using Interface<T>::push_back;

    CudaInterface();

    void push_back(Tensor<T>* tensor);

    /**
    * Add a CudaTensor to the interface
    *
    * @param tensor   tensor to add to the interface.
    */
    void push_back(CudaTensor<T>* tensor);

    /** Synchronize Device To Host-based data  */
    void synchronizeDToHBased() const;

    /** Synchronize Host-based data To Device */
    void synchronizeHBasedToD() const;

    /** Synchronize Device-based data To Host  */
    void synchronizeDBasedToH() const;

    /** Synchronize Host data To Device-based */
    void synchronizeHToDBased() const;

    virtual CudaTensor<T>& back();
    virtual const CudaTensor<T>& back() const;
    virtual CudaTensor<T>& operator[](unsigned int t);
    virtual const CudaTensor<T>& operator[](unsigned int t) const;
    virtual CudaTensor<T>& getTensor(unsigned int k,
                                     unsigned int* offset = NULL);
    virtual const CudaTensor<T>& getTensor(unsigned int k,
                                           unsigned int* offset = NULL) const;
    ~CudaInterface() {};

private:
    using Interface<T>::mMatchingDim;
    using Interface<T>::mData;
    using Interface<T>::mDataOffset;
};
}

template <class T, int STACKING_DIM>
N2D2::CudaInterface<T, STACKING_DIM>::CudaInterface() : Interface<T>()
{
    // if (sizeof(T) != sizeof(float))
    //    throw std::runtime_error("Types must match");
}

template <class T, int STACKING_DIM>
void N2D2::CudaInterface<T, STACKING_DIM>::push_back(Tensor<T>* tensor)
{
    CudaTensor<T>* cudaTensor = dynamic_cast<CudaTensor<T>*>(tensor);

    if (cudaTensor != NULL)
        Interface<T>::push_back(tensor);
    else
        Interface<T>::push_back(new CudaTensor<T>(tensor));
}

template <class T, int STACKING_DIM>
void N2D2::CudaInterface<T, STACKING_DIM>::push_back(
    CudaTensor<T>* tensor)
{
    Interface<T>::push_back(tensor);
}

template <class T, int STACKING_DIM>
void N2D2::CudaInterface<T, STACKING_DIM>::synchronizeDToHBased() const
{
    for (typename std::vector<Tensor<T>*>::const_iterator it = mData.begin(),
                                                            itEnd = mData.end();
         it != itEnd;
         ++it)
    {
        static_cast<CudaTensor<T>*>(*it)->synchronizeDToHBased();
    }
}

template <class T, int STACKING_DIM>
void N2D2::CudaInterface<T, STACKING_DIM>::synchronizeHBasedToD() const
{
    for (typename std::vector<Tensor<T>*>::const_iterator it = mData.begin(),
                                                            itEnd = mData.end();
         it != itEnd;
         ++it)
    {
        static_cast<CudaTensor<T>*>(*it)->synchronizeHBasedToD();
    }
}

template <class T, int STACKING_DIM>
void N2D2::CudaInterface<T, STACKING_DIM>::synchronizeDBasedToH() const
{
    for (typename std::vector<Tensor<T>*>::const_iterator it = mData.begin(),
                                                            itEnd = mData.end();
         it != itEnd;
         ++it)
    {
        static_cast<CudaTensor<T>*>(*it)->synchronizeDBasedToH();
    }
}

template <class T, int STACKING_DIM>
void N2D2::CudaInterface<T, STACKING_DIM>::synchronizeHToDBased() const
{
    for (typename std::vector<Tensor<T>*>::const_iterator it = mData.begin(),
                                                            itEnd = mData.end();
         it != itEnd;
         ++it)
    {
        static_cast<CudaTensor<T>*>(*it)->synchronizeHToDBased();
    }
}

template <class T, int STACKING_DIM>
N2D2::CudaTensor<T>& N2D2::CudaInterface<T, STACKING_DIM>::back()
{
    return *static_cast<CudaTensor<T>*>(mData.back());
}

template <class T, int STACKING_DIM>
const N2D2::CudaTensor<T>& N2D2::CudaInterface<T, STACKING_DIM>::back() const
{
    return *static_cast<CudaTensor<T>*>(mData.back());
}

template <class T, int STACKING_DIM>
N2D2::CudaTensor<T>& N2D2::CudaInterface<T, STACKING_DIM>::operator[](
    unsigned int t)
{
    return *static_cast<CudaTensor<T>*>(mData.at(t));
}

template <class T, int STACKING_DIM>
const N2D2::CudaTensor<T>& N2D2::CudaInterface<T, STACKING_DIM>::
operator[](unsigned int t) const
{
    return *static_cast<CudaTensor<T>*>(mData.at(t));
}

template <class T, int STACKING_DIM>
N2D2::CudaTensor<T>& N2D2::CudaInterface<T, STACKING_DIM>::getTensor(
    unsigned int k,
    unsigned int* offset)
{
    assert(k < mDataOffset.size());

    const std::pair<unsigned int, unsigned int>& dataOffset = mDataOffset.at(k);

    if (offset != NULL)
        (*offset) = dataOffset.second;

    return *static_cast<CudaTensor<T>*>(mData[dataOffset.first]);
}

template <class T, int STACKING_DIM>
const N2D2::CudaTensor<T>& N2D2::CudaInterface<T, STACKING_DIM>::getTensor(
    unsigned int k,
    unsigned int* offset) const
{
    assert(k < mDataOffset.size());

    const std::pair<unsigned int, unsigned int>& dataOffset = mDataOffset.at(k);

    if (offset != NULL)
        (*offset) = dataOffset.second;

    return *static_cast<CudaTensor<T>*>(mData[dataOffset.first]);
}

#endif // N2D2_CUDAINTERFACE_H
