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

#include <stdexcept>
#include <string>

#include "Cell/Cell.hpp"
#include "Cell/Cell_Frame_CUDA.hpp"
#include "Cell/TargetBiasCell.hpp"
#include "Cell/TargetBiasCell_Frame_CUDA.hpp"
#include "Cell/TargetBiasCell_Frame_CUDA_Kernels.hpp"
#include "containers/Tensor.hpp"
#include "DeepNet.hpp"
#include "utils/Utils.hpp"


static const N2D2::Registrar<N2D2::TargetBiasCell> registrarHalfFloat(
                    "Frame_CUDA", N2D2::TargetBiasCell_Frame_CUDA<half_float::half>::create,
                    N2D2::Registrar<N2D2::TargetBiasCell>::Type<half_float::half>());

static const N2D2::Registrar<N2D2::TargetBiasCell> registrarFloat(
                    "Frame_CUDA", N2D2::TargetBiasCell_Frame_CUDA<float>::create,
                    N2D2::Registrar<N2D2::TargetBiasCell>::Type<float>());

static const N2D2::Registrar<N2D2::TargetBiasCell> registrarDouble(
                    "Frame_CUDA", N2D2::TargetBiasCell_Frame_CUDA<double>::create,
                    N2D2::Registrar<N2D2::TargetBiasCell>::Type<double>());


template<class T>
N2D2::TargetBiasCell_Frame_CUDA<T>::TargetBiasCell_Frame_CUDA(const DeepNet& deepNet, const std::string& name,
                                                        unsigned int nbOutputs, double bias)
    : Cell(deepNet, name, nbOutputs),
      TargetBiasCell(deepNet, name, nbOutputs, std::move(bias)),
      Cell_Frame_CUDA<T>(deepNet, name, nbOutputs)
{
}

template<class T>
void N2D2::TargetBiasCell_Frame_CUDA<T>::initialize() {
    if(mInputs.size() != 1) {
        throw std::runtime_error("There can only be one input for TargetBiasCell '" + mName + "'.");
    }

    if(mInputs[0].size() != mOutputs.size()) {
        throw std::runtime_error("The size of the input and output of cell '" + mName + "' must be the same");
    }
}

template<class T>
void N2D2::TargetBiasCell_Frame_CUDA<T>::propagate(bool inference) {
    mInputs.synchronizeHBasedToD();

    std::shared_ptr<CudaDeviceTensor<T> > input0
        = cuda_device_tensor_cast<T>(mInputs[0]);

    if (!inference) {
        cudaTargetBiasPropagate(CudaContext::getDeviceProp(),
                        (T)mBias,
                        input0->getDevicePtr(),
                        mDiffInputs.getDevicePtr(),
                        mOutputs.getDevicePtr(),
                        mInputs[0].dimX(),
                        mInputs[0].dimY(),
                        getNbChannels(),
                        mInputs.dimB());
    }
    else {
        CHECK_CUDA_STATUS(
            cudaMemcpy(mOutputs.getDevicePtr(),
                       input0->getDevicePtr(),
                       mOutputs.size() * sizeof(T),
                       cudaMemcpyDeviceToDevice));
    }

    Cell_Frame_CUDA<T>::propagate(inference);
    mDiffInputs.clearValid();
}

template <class T>
void N2D2::TargetBiasCell_Frame_CUDA<T>::backPropagate() {
    if (!mDiffInputs.isValid())
        return;

    Cell_Frame_CUDA<T>::backPropagate();

    const T alpha(1.0f);

    std::shared_ptr<CudaDeviceTensor<T> > diffOutput0
        = (mDiffOutputs[0].isValid())
            ? cuda_device_tensor_cast<T>(mDiffOutputs[0])
            : cuda_device_tensor_cast_nocopy<T>(mDiffOutputs[0]);

    if (mDiffOutputs[0].isValid()) {
        CHECK_CUBLAS_STATUS(
            cublasAxpy(CudaContext::cublasHandle(),
                        mDiffOutputs[0].size(), // size of data
                        &alpha,
                        mDiffInputs.getDevicePtr(),
                        1,
                        diffOutput0->getDevicePtr(),
                        1));
    } else {
        CHECK_CUDA_STATUS(
            cudaMemcpy(diffOutput0->getDevicePtr(),
                       mDiffInputs.getDevicePtr(),
                       mDiffOutputs[0].size() * sizeof(T),
                       cudaMemcpyDeviceToDevice));

        mDiffOutputs[0].setValid();
    }

    mDiffOutputs[0].deviceTensor() = *diffOutput0;
}

template<class T>
void N2D2::TargetBiasCell_Frame_CUDA<T>::update() {
    Cell_Frame_CUDA<T>::update();
}

template<class T>
double N2D2::TargetBiasCell_Frame_CUDA<T>::applyLoss(
    double /*targetVal*/,
    double /*defaultVal*/)
{
    // TargetBias takes a target for the forward pass only, which should no set
    // mDiffInputs
    return 0.0;
}

template<class T>
double N2D2::TargetBiasCell_Frame_CUDA<T>::applyLoss()
{
    // TargetBias takes a target for the forward pass only, which should no set
    // mDiffInputs
    return 0.0;
}

template<class T>
void N2D2::TargetBiasCell_Frame_CUDA<T>::checkGradient(double /*epsilon*/, double /*maxError*/) {
    throw std::runtime_error("checkGradient not supported yet.");
}

namespace N2D2 {
    template class TargetBiasCell_Frame_CUDA<half_float::half>;
    template class TargetBiasCell_Frame_CUDA<float>;
    template class TargetBiasCell_Frame_CUDA<double>;
}

#endif