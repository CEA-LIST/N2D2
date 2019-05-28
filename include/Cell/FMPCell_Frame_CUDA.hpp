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

#ifndef N2D2_FMPCELL_FRAME_CUDA_H
#define N2D2_FMPCELL_FRAME_CUDA_H

#include "Cell_Frame_CUDA.hpp"
#include "FMPCell.hpp"
#include "FMPCell_Frame_CUDA_Kernels.hpp"

#include "CudaContext.hpp"
#include "CudaUtils.hpp"
#include "containers/CudaTensor.hpp"
#include "DeepNet.hpp"

namespace N2D2 {
class FMPCell_Frame_CUDA : public virtual FMPCell, public Cell_Frame_CUDA<Float_T> {
public:
    FMPCell_Frame_CUDA(const DeepNet& deepNet, const std::string& name,
                       double scalingRatio,
                       unsigned int nbOutputs,
                       const std::shared_ptr<Activation>& activation
                       = std::shared_ptr<Activation>());
    static std::shared_ptr<FMPCell> create(Network& /*net*/, const DeepNet& deepNet, 
                                           const std::string& name,
                                           double scalingRatio,
                                           unsigned int nbOutputs,
                                           const std::shared_ptr
                                           <Activation>& activation
                                           = std::shared_ptr
                                           <Activation>())
    {
        return std::make_shared
            <FMPCell_Frame_CUDA>(deepNet, name, scalingRatio, nbOutputs, activation);
    }

    virtual void initialize();
    virtual void propagate(bool inference = false);
    virtual void backPropagate();
    virtual void update();
    void checkGradient(double /*epsilon*/ = 1.0e-4,
                       double /*maxError*/ = 1.0e-6) {};
    virtual ~FMPCell_Frame_CUDA() {};

protected:
    void generateRegions(CudaTensor<unsigned int>& grid,
                         unsigned int sizeIn,
                         unsigned int sizeOut);

    CudaTensor<unsigned int> mGridX;
    CudaTensor<unsigned int> mGridY;

private:
    static Registrar<FMPCell> mRegistrar;
};
}

#endif // N2D2_FMPCELL_FRAME_CUDA_H
