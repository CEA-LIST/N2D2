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

#ifndef N2D2_SATURATIONACTIVATION_FRAME_CUDA_H
#define N2D2_SATURATIONACTIVATION_FRAME_CUDA_H

#include "CudaContext.hpp"
#include "CudaUtils.hpp"
#include "Activation/Activation_CUDA_Kernels.hpp"
#include "Activation/SaturationActivation.hpp"
#include "Cell/Cell.hpp"
#include "containers/CudaTensor.hpp"

namespace N2D2 {
template <class T>
class SaturationActivation_Frame_CUDA : public SaturationActivation {
public:
    static std::shared_ptr<SaturationActivation> create()
    {
        return std::make_shared<SaturationActivation_Frame_CUDA<T> >();
    }

    virtual void propagate(const Cell& cell,
                           BaseTensor& input,
                           BaseTensor& output,
                           bool inference = false);
    virtual void backPropagate(const Cell& cell,
                               BaseTensor& input,
                               BaseTensor& output,
                               BaseTensor& diffInput,
                               BaseTensor& diffOutput);
    virtual ~SaturationActivation_Frame_CUDA() {};

private:
    static Registrar<SaturationActivation> mRegistrar;
};
}

template <class T>
void N2D2::SaturationActivation_Frame_CUDA<T>::propagate(
    const Cell& cell, 
    BaseTensor& baseInput,
    BaseTensor& baseOutput,
    bool /*inference*/)
{
    CudaTensor<T>& input = dynamic_cast<CudaTensor<T>&>(baseInput);
    CudaTensor<T>& output = dynamic_cast<CudaTensor<T>&>(baseOutput);

    mScaling.propagate(cell, input, output);

    if (mThreshold != 0) {
        cudaSaturation_propagate(output.getDevicePtr(),
                                  output.getDevicePtr(),
                                  output.size(),
                                  T(mThreshold));
    }
}

template <class T>
void N2D2::SaturationActivation_Frame_CUDA<T>::backPropagate(
    const Cell& cell, 
    BaseTensor& /*baseInput*/,
    BaseTensor& baseOutput,
    BaseTensor& baseDiffInput,
    BaseTensor& baseDiffOutput)
{
    CudaTensor<T>& output = dynamic_cast<CudaTensor<T>&>(baseOutput);
    CudaTensor<T>& diffInput = dynamic_cast<CudaTensor<T>&>(baseDiffInput);
    CudaTensor<T>& diffOutput = dynamic_cast<CudaTensor<T>&>(baseDiffOutput);

    if (mThreshold != 0) {
        cudaSaturation_backPropagate(output.getDevicePtr(),
                                    diffInput.getDevicePtr(),
                                    diffOutput.getDevicePtr(),
                                    output.size(),
                                    (T)mThreshold);

        mScaling.backPropagate(cell, diffOutput, diffOutput);
    }
    else
        mScaling.backPropagate(cell, diffInput, diffOutput);
}

#endif // N2D2_SATURATIONACTIVATION_FRAME_CUDA_H
