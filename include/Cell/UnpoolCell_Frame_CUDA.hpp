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
#include "DeepNet.hpp"
#include "UnpoolCell.hpp"
#include "PoolCell_Frame_CUDA_Kernels.hpp"

#include "CudaContext.hpp"
#include "CudaUtils.hpp"
#include "containers/CudaTensor.hpp"

namespace N2D2 {
class UnpoolCell_Frame_CUDA : public virtual UnpoolCell, public Cell_Frame_CUDA<Float_T>
{
public:
    UnpoolCell_Frame_CUDA(const DeepNet& deepNet, const std::string& name,
                   const std::vector<unsigned int>& poolDims,
                   unsigned int nbOutputs,
                   const std::vector<unsigned int>& strideDims
                      = std::vector<unsigned int>(2, 1U),
                   const std::vector<unsigned int>& paddingDims
                      = std::vector<unsigned int>(2, 0),
                   Pooling pooling = Max,
                   const std::shared_ptr<Activation>& activation
                   = std::shared_ptr<Activation>());
    static std::shared_ptr<UnpoolCell> create(Network& /*net*/,
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
        return std::make_shared<UnpoolCell_Frame_CUDA>(deepNet, name,
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
    void addArgMax(Tensor<PoolCell_Frame_Kernels::ArgMax>* argMax)
    {
        mArgMax.push_back(argMax);
    };
    void addArgMax(Interface<PoolCell_Frame_Kernels::ArgMax>* argMax);
    
    std::pair<double, double> getOutputsRange() const;

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
