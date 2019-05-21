/*
    (C) Copyright 2019 CEA LIST. All Rights Reserved.
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

#ifdef CUDA

#include "Cell/ThresholdCell_Frame_CUDA.hpp"
#include "Cell/ThresholdCell_Frame_CUDA_Kernels.hpp"

N2D2::Registrar<N2D2::ThresholdCell>
N2D2::ThresholdCell_Frame_CUDA::mRegistrar(
    "Frame_CUDA", N2D2::ThresholdCell_Frame_CUDA::create);

N2D2::ThresholdCell_Frame_CUDA::ThresholdCell_Frame_CUDA(
    const std::string& name,
    unsigned int nbOutputs,
    double threshold)
    : Cell(name, nbOutputs),
      ThresholdCell(name, nbOutputs, threshold),
      Cell_Frame_CUDA<Float_T>(name, nbOutputs)
{
    // ctor
}

void N2D2::ThresholdCell_Frame_CUDA::propagate(bool inference)
{
    mInputs.synchronizeHBasedToD();

    if (mInputs.size() > 1)
        throw std::runtime_error("ThresholdCell can only have one input");

    std::shared_ptr<CudaDeviceTensor<Float_T> > input
        = cuda_device_tensor_cast<Float_T>(mInputs[0]);

    cudaSThreshold(CudaContext::getDeviceProp(),
                   input->getDevicePtr(),
                   mInputs[0].dimX(),
                   mInputs[0].dimY(),
                   mInputs[0].dimZ(),
                   mInputs[0].dimB(),
                   mOutputs.getDevicePtr(),
                   mThreshold,
                   mOperation,
                   mMaxValue);

    Cell_Frame_CUDA<Float_T>::propagate(inference);
    mDiffInputs.clearValid();
}

#endif
