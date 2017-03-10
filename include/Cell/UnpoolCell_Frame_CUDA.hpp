/*
    (C) Copyright 2014 CEA LIST. All Rights Reserved.
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

#ifndef N2D2_UNPOOLCELL_FRAME_CUDA_H
#define N2D2_UNPOOLCELL_FRAME_CUDA_H

#include "Cell_Frame_CUDA.hpp"
#include "UnpoolCell.hpp"
#include "PoolCell_Frame_CUDA_Kernels.hpp"

#include "CudaContext.hpp"
#include "CudaUtils.hpp"
#include "containers/CudaTensor4d.hpp"

namespace N2D2 {
class UnpoolCell_Frame_CUDA : public virtual UnpoolCell, public Cell_Frame_CUDA
{
public:
    UnpoolCell_Frame_CUDA(const std::string& name,
                   unsigned int poolWidth,
                   unsigned int poolHeight,
                   unsigned int nbOutputs,
                   unsigned int strideX = 1,
                   unsigned int strideY = 1,
                   unsigned int paddingX = 0,
                   unsigned int paddingY = 0,
                   Pooling pooling = Max,
                   const std::shared_ptr<Activation<Float_T> >& activation
                   = std::shared_ptr<Activation<Float_T> >());
    static std::shared_ptr<UnpoolCell> create(Network& /*net*/,
                                            const std::string& name,
                                            unsigned int poolWidth,
                                            unsigned int poolHeight,
                                            unsigned int nbOutputs,
                                            unsigned int strideX = 1,
                                            unsigned int strideY = 1,
                                            unsigned int paddingX = 0,
                                            unsigned int paddingY = 0,
                                            Pooling pooling = Max,
                                            const std::shared_ptr
                                            <Activation<Float_T> >& activation
                                            = std::shared_ptr
                                            <Activation<Float_T> >())
    {
        return std::make_shared<UnpoolCell_Frame_CUDA>(name,
                                                       poolWidth,
                                                       poolHeight,
                                                       nbOutputs,
                                                       strideX,
                                                       strideY,
                                                       paddingX,
                                                       paddingY,
                                                       pooling,
                                                       activation);
    }

    virtual void initialize();
    virtual void propagate(bool inference = false);
    virtual void backPropagate();
    virtual void update();
    void checkGradient(double epsilon = 1.0e-4, double maxError = 1.0e-6);
    void addArgMax(Tensor4d<PoolCell_Frame_Kernels::ArgMax>* argMax)
    {
        mArgMax.push_back(argMax);
    };
    void addArgMax(Interface<PoolCell_Frame_Kernels::ArgMax>* argMax);
    virtual ~UnpoolCell_Frame_CUDA();

protected:
    std::vector<char*> mInputMap;
    PoolCell_Frame_Kernels::Descriptor* mPoolDesc;
    CudaInterface<PoolCell_Frame_Kernels::ArgMax> mArgMax;

private:
    static Registrar<UnpoolCell> mRegistrar;
};
}

#endif // N2D2_UNPOOLCELL_FRAME_CUDA_H
