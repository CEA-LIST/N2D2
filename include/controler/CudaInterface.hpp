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
template <class T>
struct cuda_tensor_t {
    typedef CudaTensor<T> type;
};

template <>
struct cuda_tensor_t<void> {
    typedef CudaBaseTensor type;
};

/**
 * @class   CudaInterface
 * @brief   Merge virtually several Tensor through an unified data interface.
*/
template <class T = void, int STACKING_DIM = -2>
class CudaInterface : public Interface<T, STACKING_DIM> {
public:
    typedef typename cuda_tensor_t<T>::type cuda_tensor_type;

    CudaInterface();
    template <class U> CudaInterface(const CudaInterface<U, STACKING_DIM>&
                                        interface);

    void push_back(typename Interface<T, STACKING_DIM>::tensor_type* tensor,
                   size_t refs = 1);

    /**
    * Add a CudaTensor to the interface
    *
    * @param tensor   tensor to add to the interface.
    */
    void push_back(cuda_tensor_type* tensor, size_t refs = 1);

    virtual void broadcast(int srcDev, int dstDev) const;
    virtual void broadcastAllFrom(int srcDev) const;
    virtual void broadcastAllFrom(int srcDev,
                    std::vector<N2D2::DeviceState> devices) const;
    virtual void broadcastAnyTo(int dstDev) const;

    virtual void aggregate(int srcDev, int dstDev) const;
    virtual void aggregateAllTo(int dstDev) const;
    virtual void aggregateAllTo(int dstDev,
                    std::vector<N2D2::DeviceState> devices) const;

    virtual cuda_tensor_type& back();
    virtual const cuda_tensor_type& back() const;
    virtual cuda_tensor_type& operator[](unsigned int t);
    virtual const cuda_tensor_type& operator[](unsigned int t) const;
    ~CudaInterface() {};

private:
    using Interface<T, STACKING_DIM>::mMatchingDim;
    using Interface<T, STACKING_DIM>::mData;
    using Interface<T, STACKING_DIM>::mDataOffset;
};
}

template <class T, int STACKING_DIM>
N2D2::CudaInterface<T, STACKING_DIM>::CudaInterface()
    : Interface<T, STACKING_DIM>()
{
    // ctor
}

template <class T, int STACKING_DIM>
template <class U>
N2D2::CudaInterface<T, STACKING_DIM>::CudaInterface(
    const CudaInterface<U, STACKING_DIM>& interface)
    : Interface<T, STACKING_DIM>(interface)
{
    // copy-ctor
}

template <class T, int STACKING_DIM>
void N2D2::CudaInterface<T, STACKING_DIM>::push_back(
    typename Interface<T, STACKING_DIM>::tensor_type* tensor, size_t refs)
{
    cuda_tensor_type* cudaTensor = dynamic_cast<cuda_tensor_type*>(tensor);

    if (cudaTensor != NULL)
        Interface<T, STACKING_DIM>::push_back(tensor, refs);
    else
        Interface<T, STACKING_DIM>::push_back(dynamic_cast<cuda_tensor_type*>(tensor->newCuda()), 0);
}

template <class T, int STACKING_DIM>
void N2D2::CudaInterface<T, STACKING_DIM>::push_back(cuda_tensor_type* tensor,
                                                     size_t refs)
{
    Interface<T, STACKING_DIM>::push_back(tensor, refs);
}

template <class T, int STACKING_DIM>
typename N2D2::CudaInterface<T, STACKING_DIM>::cuda_tensor_type& N2D2::CudaInterface<T, STACKING_DIM>::back()
{
    return *dynamic_cast<cuda_tensor_type*>(mData.back());
}

template <class T, int STACKING_DIM>
const typename N2D2::CudaInterface<T, STACKING_DIM>::cuda_tensor_type& N2D2::CudaInterface<T, STACKING_DIM>::back() const
{
    return *dynamic_cast<cuda_tensor_type*>(mData.back());
}

template <class T, int STACKING_DIM>
typename N2D2::CudaInterface<T, STACKING_DIM>::cuda_tensor_type& N2D2::CudaInterface<T, STACKING_DIM>::operator[](
    unsigned int t)
{
    return *dynamic_cast<cuda_tensor_type*>(mData.at(t));
}

template <class T, int STACKING_DIM>
const typename N2D2::CudaInterface<T, STACKING_DIM>::cuda_tensor_type& N2D2::CudaInterface<T, STACKING_DIM>::
operator[](unsigned int t) const
{
    return *dynamic_cast<cuda_tensor_type*>(mData.at(t));
}

template <class T, int STACKING_DIM>
void N2D2::CudaInterface<T, STACKING_DIM>::broadcast(int srcDev, int dstDev) const
{
    for (auto it = mData.begin(), itEnd = mData.end(); it != itEnd; ++it)
    {
        dynamic_cast<cuda_tensor_type*>(*it)->broadcast(srcDev, dstDev);
    }
}

template <class T, int STACKING_DIM>
void N2D2::CudaInterface<T, STACKING_DIM>::broadcastAllFrom(int srcDev) const
{
    for (auto it = mData.begin(), itEnd = mData.end(); it != itEnd; ++it)
    {
        dynamic_cast<cuda_tensor_type*>(*it)->broadcastAllFrom(srcDev);
    }
}

template <class T, int STACKING_DIM>
void N2D2::CudaInterface<T, STACKING_DIM>::broadcastAllFrom(int srcDev,
    std::vector<N2D2::DeviceState> devices) const
{
    for (auto it = mData.begin(), itEnd = mData.end(); it != itEnd; ++it)
    {
        dynamic_cast<cuda_tensor_type*>(*it)->broadcastAllFrom(srcDev, devices);
    }
}

template <class T, int STACKING_DIM>
void N2D2::CudaInterface<T, STACKING_DIM>::broadcastAnyTo(int dstDev) const
{
    for (auto it = mData.begin(), itEnd = mData.end(); it != itEnd; ++it)
    {
        dynamic_cast<cuda_tensor_type*>(*it)->broadcastAnyTo(dstDev);
    }
}

template <class T, int STACKING_DIM>
void N2D2::CudaInterface<T, STACKING_DIM>::aggregate(int srcDev, int dstDev) const
{
    for (auto it = mData.begin(), itEnd = mData.end(); it != itEnd; ++it)
    {
        dynamic_cast<cuda_tensor_type*>(*it)->aggregate(srcDev, dstDev);
    }
}

template <class T, int STACKING_DIM>
void N2D2::CudaInterface<T, STACKING_DIM>::aggregateAllTo(int dstDev) const
{
    for (auto it = mData.begin(), itEnd = mData.end(); it != itEnd; ++it)
    {
        dynamic_cast<cuda_tensor_type*>(*it)->aggregateAllTo(dstDev);
    }
}

template <class T, int STACKING_DIM>
void N2D2::CudaInterface<T, STACKING_DIM>::aggregateAllTo(int dstDev,
    std::vector<N2D2::DeviceState> devices) const
{
    for (auto it = mData.begin(), itEnd = mData.end(); it != itEnd; ++it)
    {
        dynamic_cast<cuda_tensor_type*>(*it)->aggregateAllTo(dstDev, devices);
    }
}

#endif // N2D2_CUDAINTERFACE_H
