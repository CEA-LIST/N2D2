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

#ifndef N2D2_POOLCELL_FRAME_CUDA_H
#define N2D2_POOLCELL_FRAME_CUDA_H

#include "Cell_Frame_CUDA.hpp"
#include "PoolCell.hpp"

#include "CudaContext.hpp"
#include "CudaUtils.hpp"
#include "containers/CudaTensor.hpp"

namespace N2D2 {
class PoolCell_Frame_CUDA : public virtual PoolCell, public Cell_Frame_CUDA<Float_T> {
public:
    PoolCell_Frame_CUDA(const std::string& name,
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
        return std::make_shared<PoolCell_Frame_CUDA>(name,
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
    virtual ~PoolCell_Frame_CUDA();

protected:
    std::vector<cudnnTensorDescriptor_t> mOutputDesc;

    cudnnPoolingDescriptor_t mPoolingDesc;

private:
    static Registrar<PoolCell> mRegistrar;
};
}

#endif // N2D2_POOLCELL_FRAME_CUDA_H
