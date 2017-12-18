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

    void push_back(Tensor4d<T>* tensor);

    /**
    * Add a CudaTensor4d to the interface
    *
    * @param tensor   tensor to add to the interface.
    */
    void push_back(CudaTensor4d<T>* tensor);

    /** Synchronize Device To Host-based data  */
    void synchronizeDToHBased() const;

    /** Synchronize Host-based data To Device */
    void synchronizeHBasedToD() const;

    /** Synchronize Device-based data To Host  */
    void synchronizeDBasedToH() const;

    /** Synchronize Host data To Device-based */
    void synchronizeHToDBased() const;

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
void N2D2::CudaInterface<T>::push_back(Tensor4d<T>* tensor)
{
    CudaTensor4d<T>* cudaTensor = dynamic_cast<CudaTensor4d<T>*>(tensor);

    if (cudaTensor != NULL)
        Interface<T>::push_back(tensor);
    else
        Interface<T>::push_back(new CudaTensor4d<T>(tensor));
}

template <typename T>
void N2D2::CudaInterface<T>::push_back(CudaTensor4d<T>* tensor)
{
    Interface<T>::push_back(tensor);
}

template <typename T> void N2D2::CudaInterface<T>::synchronizeDToHBased() const
{
    for (typename std::vector<Tensor4d<T>*>::const_iterator it = mData.begin(),
                                                            itEnd = mData.end();
         it != itEnd;
         ++it)
    {
        static_cast<CudaTensor4d<T>*>(*it)->synchronizeDToHBased();
    }
}

template <typename T> void N2D2::CudaInterface<T>::synchronizeHBasedToD() const
{
    for (typename std::vector<Tensor4d<T>*>::const_iterator it = mData.begin(),
                                                            itEnd = mData.end();
         it != itEnd;
         ++it)
    {
        static_cast<CudaTensor4d<T>*>(*it)->synchronizeHBasedToD();
    }
}

template <typename T> void N2D2::CudaInterface<T>::synchronizeDBasedToH() const
{
    for (typename std::vector<Tensor4d<T>*>::const_iterator it = mData.begin(),
                                                            itEnd = mData.end();
         it != itEnd;
         ++it)
    {
        static_cast<CudaTensor4d<T>*>(*it)->synchronizeDBasedToH();
    }
}

template <typename T> void N2D2::CudaInterface<T>::synchronizeHToDBased() const
{
    for (typename std::vector<Tensor4d<T>*>::const_iterator it = mData.begin(),
                                                            itEnd = mData.end();
         it != itEnd;
         ++it)
    {
        static_cast<CudaTensor4d<T>*>(*it)->synchronizeHToDBased();
    }
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
