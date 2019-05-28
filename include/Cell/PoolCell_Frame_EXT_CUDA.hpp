/*
    (C) Copyright 2015 CEA LIST. All Rights Reserved.
    Contributor(s): Olivier BICHLER (olivier.bichler@cea.fr)
                    Victor GACOIN

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

#ifndef N2D2_POOLCELL_FRAME_EXT_CUDA_H
#define N2D2_POOLCELL_FRAME_EXT_CUDA_H

#include "Cell_Frame_CUDA.hpp"
#include "PoolCell.hpp"
#include "PoolCell_Frame_CUDA_Kernels.hpp"

#include "CudaContext.hpp"
#include "CudaUtils.hpp"
#include "containers/CudaTensor.hpp"
#include "DeepNet.hpp"

namespace N2D2 {
template <class T>
class PoolCell_Frame_EXT_CUDA : public virtual PoolCell, public Cell_Frame_CUDA<T> {
public:
    using Cell_Frame_CUDA<T>::mInputs;
    using Cell_Frame_CUDA<T>::mOutputs;
    using Cell_Frame_CUDA<T>::mDiffInputs;
    using Cell_Frame_CUDA<T>::mDiffOutputs;
    using Cell_Frame_CUDA<T>::mActivationDesc;

    PoolCell_Frame_EXT_CUDA(const DeepNet& deepNet, const std::string& name,
                        const std::vector<unsigned int>& poolDims,
                        unsigned int nbOutputs,
                        const std::vector<unsigned int>& strideDims
                           = std::vector<unsigned int>(2, 1U),
                        const std::vector<unsigned int>& paddingDims
                           = std::vector<unsigned int>(2, 0),
                        Pooling pooling = Max,
                        const std::shared_ptr<Activation>& activation
                        = std::shared_ptr<Activation>());
    static std::shared_ptr<PoolCell> create(Network& /*net*/,
        const DeepNet& deepNet, 
        const std::string& name,
        const std::vector<unsigned int>& poolDims,
        unsigned int nbOutputs,
        const std::vector<unsigned int>& strideDims
            = std::vector<unsigned int>(2, 1U),
        const std::vector<unsigned int>& paddingDims
            = std::vector<unsigned int>(2, 0),
        Pooling pooling = Max,
        const std::shared_ptr<Activation>& activation
            = std::shared_ptr<Activation>())
    {
        return std::make_shared<PoolCell_Frame_EXT_CUDA<T> >(deepNet, name,
                                                            poolDims,
                                                            nbOutputs,
                                                            strideDims,
                                                            paddingDims,
                                                            pooling,
                                                            activation);
    }

    virtual void initialize();
    virtual void propagate(bool inference = false);
    virtual void backPropagate();
    virtual void update();
    void checkGradient(double epsilon = 1.0e-4, double maxError = 1.0e-6);
    CudaInterface<PoolCell_Frame_Kernels::ArgMax>* getArgMax()
    {
        initialize();   // Make sure mArgMax is populated
        return &mArgMax;
    };
    virtual ~PoolCell_Frame_EXT_CUDA();

protected:
    std::vector<char*> mInputMap;
    PoolCell_Frame_Kernels::Descriptor* mPoolDesc;
    CudaInterface<PoolCell_Frame_Kernels::ArgMax> mArgMax;

private:
    static Registrar<PoolCell> mRegistrar;
};
}

namespace N2D2 {
template <>
void PoolCell_Frame_EXT_CUDA<half_float::half>::propagate(bool inference);
template <>
void PoolCell_Frame_EXT_CUDA<half_float::half>::backPropagate();

template <>
void PoolCell_Frame_EXT_CUDA<float>::propagate(bool inference);
template <>
void PoolCell_Frame_EXT_CUDA<float>::backPropagate();

template <>
void PoolCell_Frame_EXT_CUDA<double>::propagate(bool inference);
template <>
void PoolCell_Frame_EXT_CUDA<double>::backPropagate();
}


#endif // N2D2_POOLCELL_FRAME_EXT_CUDA_H
