/*
    (C) Copyright 2018 CEA LIST. All Rights Reserved.
    Contributor(s): David BRIAND (david.briand@cea.fr)

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

#ifndef N2D2_RESIZECELL_FRAME_H
#define N2D2_RESIZECELL_FRAME_H

#include <tuple>
#include <unordered_map>
#include <vector>

#include "CudaContext.hpp"
#include "CudaUtils.hpp"
#include "containers/CudaTensor.hpp"

#include "Cell_Frame_CUDA.hpp"
#include "DeepNet.hpp"
#include "ResizeCell.hpp"
#include "ResizeCell_Frame_Kernels_struct.hpp"
#include "ResizeCell_Frame_CUDA_Kernels.hpp"

namespace N2D2 {
class ResizeCell_Frame_CUDA : public virtual ResizeCell, public Cell_Frame_CUDA<Float_T> {
public:
    void BilinearInterpolation(const int out_size,
                                const int in_size,
                                const float scale,
                                CudaTensor<unsigned int>& LowIndex,
                                CudaTensor<unsigned int>& HightIndex,
                                CudaTensor<Float_T>& Interpolation);

    ResizeCell_Frame_CUDA(const DeepNet& deepNet, const std::string& name,
                         unsigned int outputsWidth,
                         unsigned int outputsHeight,
                         unsigned int nbOutputs,
                         ResizeMode resizeMode);
    static std::shared_ptr<ResizeCell> create(const DeepNet& deepNet, const std::string& name,
                                                  unsigned int outputsWidth,
                                                  unsigned int outputsHeight,
                                                  unsigned int nbOutputs,
                                                  ResizeMode resizeMode)
    {
        return std::make_shared<ResizeCell_Frame_CUDA>(deepNet, name,
                                                      outputsWidth,
                                                      outputsHeight,
                                                      nbOutputs,
                                                      resizeMode);
    }

    virtual void initialize();
    virtual void propagate(bool inference = false);
    virtual void backPropagate();
    virtual void update();
    void checkGradient(double epsilon = 1.0e-4, double maxError = 1.0e-6);
    virtual ~ResizeCell_Frame_CUDA() {};
protected:
    CudaTensor<unsigned int> mYStrideLowIndex;
    CudaTensor<unsigned int> mYStrideHightIndex;
    CudaTensor<Float_T> mYStrideInterpolation;

    CudaTensor<unsigned int> mXStrideLowIndex;
    CudaTensor<unsigned int> mXStrideHightIndex;
    CudaTensor<Float_T> mXStrideInterpolation;

    Float_T mScaleX;
    Float_T mScaleY;

private:
    static Registrar<ResizeCell> mRegistrar;
};
}

#endif // N2D2_RESIZECELL_FRAME_H
