/*
    (C) Copyright 2015 CEA LIST. All Rights Reserved.
    Contributor(s): Olivier BICHLER (olivier.bichler@cea.fr)
                    David BRIAND (david.briand@cea.fr)
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

#ifndef N2D2_ROIPOOLINGCELL_FRAME_CUDA_H
#define N2D2_ROIPOOLINGCELL_FRAME_CUDA_H

#include <tuple>
#include <unordered_map>
#include <vector>

#include "Cell_Frame_CUDA.hpp"
#include "PoolCell_Frame_Kernels.hpp"
#include "ROIPoolingCell_Frame_CUDA_Kernels.hpp"
#include "ROIPoolingCell.hpp"

#include "CudaContext.hpp"
#include "CudaUtils.hpp"
#include "containers/CudaTensor4d.hpp"

namespace N2D2 {
class ROIPoolingCell_Frame_CUDA : public virtual ROIPoolingCell,
                                  public Cell_Frame_CUDA
{
public:
    ROIPoolingCell_Frame_CUDA(const std::string& name,
                              StimuliProvider& sp,
                              unsigned int outputsWidth,
                              unsigned int outputsHeight,
                              unsigned int nbOutputs,
                              ROIPooling pooling);
    static std::shared_ptr<ROIPoolingCell> create(const std::string& name,
                                                  StimuliProvider& sp,
                                                  unsigned int outputsWidth,
                                                  unsigned int outputsHeight,
                                                  unsigned int nbOutputs,
                                                  ROIPooling pooling)
    {
        return std::make_shared<ROIPoolingCell_Frame_CUDA>(name,
                                                           sp,
                                                           outputsWidth,
                                                           outputsHeight,
                                                           nbOutputs,
                                                           pooling);
    }

    virtual void initialize();
    virtual void propagate(bool inference = false);
    virtual void backPropagate();
    virtual void update();
    void checkGradient(double epsilon = 1.0e-4, double maxError = 1.0e-6);
    virtual ~ROIPoolingCell_Frame_CUDA();

protected:
    CudaInterface<PoolCell_Frame_Kernels::ArgMax> mArgMax;

private:
    static Registrar<ROIPoolingCell> mRegistrar;
};
}

#endif // N2D2_ROIPOOLINGCELL_FRAME_CUDA_H
