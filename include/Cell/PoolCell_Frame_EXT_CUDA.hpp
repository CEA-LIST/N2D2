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
#include "containers/CudaTensor4d.hpp"

namespace N2D2 {
class PoolCell_Frame_EXT_CUDA : public virtual PoolCell, public Cell_Frame_CUDA {
public:
    PoolCell_Frame_EXT_CUDA(const std::string& name,
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
    static std::shared_ptr<PoolCell> create(Network& /*net*/,
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
        return std::make_shared<PoolCell_Frame_EXT_CUDA>(name,
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

#endif // N2D2_POOLCELL_FRAME_EXT_CUDA_H
