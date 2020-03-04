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
#include "Cell/NormalizeCell.hpp"
#include "Cell/NormalizeCell_Frame_CUDA.hpp"
#include "Cell/NormalizeCell_Frame_CUDA_Kernels.hpp"
#include "containers/Tensor.hpp"
#include "DeepNet.hpp"
#include "utils/Utils.hpp"


static const N2D2::Registrar<N2D2::NormalizeCell> registrarHalfFloat(
                    "Frame_CUDA", N2D2::NormalizeCell_Frame_CUDA<half_float::half>::create,
                    N2D2::Registrar<N2D2::NormalizeCell>::Type<half_float::half>());

static const N2D2::Registrar<N2D2::NormalizeCell> registrarFloat(
                    "Frame_CUDA", N2D2::NormalizeCell_Frame_CUDA<float>::create,
                    N2D2::Registrar<N2D2::NormalizeCell>::Type<float>());

static const N2D2::Registrar<N2D2::NormalizeCell> registrarDouble(
                    "Frame_CUDA", N2D2::NormalizeCell_Frame_CUDA<double>::create,
                    N2D2::Registrar<N2D2::NormalizeCell>::Type<double>());


template<class T>
N2D2::NormalizeCell_Frame_CUDA<T>::NormalizeCell_Frame_CUDA(const DeepNet& deepNet, const std::string& name,
                                                        unsigned int nbOutputs, Norm norm)
    : Cell(deepNet, name, nbOutputs),
      NormalizeCell(deepNet, name, nbOutputs, std::move(norm)),
      Cell_Frame_CUDA<T>(deepNet, name, nbOutputs)
{
}

template<class T>
void N2D2::NormalizeCell_Frame_CUDA<T>::initialize() {
    if(mInputs.size() != 1) {
        throw std::runtime_error("There can only be one input for NormalizeCell '" + mName + "'.");
    }

    if(mInputs[0].size() != mOutputs.size()) {
        throw std::runtime_error("The size of the input and output of cell '" + mName + "' must be the same");
    }

    mNormData.resize(mOutputs.dims());
}

//Half type propagate for NormalizeCell_Frame_CUDA
template <>
void N2D2::NormalizeCell_Frame_CUDA<half_float::half>::propagate(bool inference)
{
    mInputs.synchronizeHBasedToD();

    const half_float::half alpha(1.0f);
    half_float::half beta(0.0f);

    for (unsigned int k = 0, size = mInputs.size(); k < size; ++k) {
        if (k > 0)
            beta = half_float::half(1.0f);

        std::shared_ptr<CudaDeviceTensor<half_float::half> > input
            = cuda_device_tensor_cast<half_float::half>(mInputs[k]);

        if (mNorm == L2) {
            cudaHNormalizeL2Forward(CudaContext::getDeviceProp(),
                                alpha,
                                input->getDevicePtr(),
                                mInputs[k].dimZ(),
                                mInputs[k].dimY(),
                                mInputs[k].dimX(),
                                mInputs[k].dimB(),
                                beta,
                                mOutputs.getDevicePtr(),
                                mNormData.getDevicePtr(),
                                mOutputs.dimZ(),
                                mOutputs.dimY(),
                                mOutputs.dimX());
        }
        else {
            throw std::runtime_error("Unsupported norm.");
        }
    }

    Cell_Frame_CUDA<half_float::half>::propagate(inference);
    mDiffInputs.clearValid();
}
//Float type propagate for NormalizeCell_Frame_CUDA
template <>
void N2D2::NormalizeCell_Frame_CUDA<float>::propagate(bool inference)
{
    mInputs.synchronizeHBasedToD();

    const float alpha = 1.0f;
    float beta = 0.0f;

    for (unsigned int k = 0, size = mInputs.size(); k < size; ++k) {
        if (k > 0)
            beta = 1.0f;

        std::shared_ptr<CudaDeviceTensor<float> > input
            = cuda_device_tensor_cast<float>(mInputs[k]);

        if (mNorm == L2) {
            cudaSNormalizeL2Forward(CudaContext::getDeviceProp(),
                                alpha,
                                input->getDevicePtr(),
                                mInputs[k].dimZ(),
                                mInputs[k].dimY(),
                                mInputs[k].dimX(),
                                mInputs[k].dimB(),
                                beta,
                                mOutputs.getDevicePtr(),
                                mNormData.getDevicePtr(),
                                mOutputs.dimZ(),
                                mOutputs.dimY(),
                                mOutputs.dimX());
        }
        else {
            throw std::runtime_error("Unsupported norm.");
        }
    }

    Cell_Frame_CUDA<float>::propagate(inference);
    mDiffInputs.clearValid();
}
//Double type propagate for NormalizeCell_Frame_CUDA
template <>
void N2D2::NormalizeCell_Frame_CUDA<double>::propagate(bool inference)
{
    mInputs.synchronizeHBasedToD();

    const double alpha = 1.0;
    double beta = 0.0;

    for (unsigned int k = 0, size = mInputs.size(); k < size; ++k) {
        if (k > 0)
            beta = 1.0;

        std::shared_ptr<CudaDeviceTensor<double> > input
            = cuda_device_tensor_cast<double>(mInputs[k]);

        if (mNorm == L2) {
            cudaDNormalizeL2Forward(CudaContext::getDeviceProp(),
                                alpha,
                                input->getDevicePtr(),
                                mInputs[k].dimZ(),
                                mInputs[k].dimY(),
                                mInputs[k].dimX(),
                                mInputs[k].dimB(),
                                beta,
                                mOutputs.getDevicePtr(),
                                mNormData.getDevicePtr(),
                                mOutputs.dimZ(),
                                mOutputs.dimY(),
                                mOutputs.dimX());
        }
        else {
            throw std::runtime_error("Unsupported norm.");
        }
    }

    Cell_Frame_CUDA<double>::propagate(inference);
    mDiffInputs.clearValid();
}


//Half type backpropagation for NormalizeCell_Frame_CUDA
template <>
void N2D2::NormalizeCell_Frame_CUDA<half_float::half>::backPropagate()
{
    if (mDiffOutputs.empty() || !mDiffInputs.isValid())
        return;

    mDiffInputs.synchronizeHBasedToD();
    Cell_Frame_CUDA<half_float::half>::backPropagate();

    const half_float::half alpha(1.0f);

    for (unsigned int k = 0, size = mInputs.size(); k < size; ++k) {
        const half_float::half beta((mDiffOutputs[k].isValid()) ? 1.0f : 0.0f);

        std::shared_ptr<CudaDeviceTensor<half_float::half> > diffOutput
            = (mDiffOutputs[k].isValid())
                ? cuda_device_tensor_cast<half_float::half>(mDiffOutputs[k])
                : cuda_device_tensor_cast_nocopy<half_float::half>(mDiffOutputs[k]);

        if (mNorm == L2) {
            cudaHNormalizeL2Backward(CudaContext::getDeviceProp(),
                                     alpha,
                                     mOutputs.getDevicePtr(),
                                     mNormData.getDevicePtr(),
                                     mDiffInputs.getDevicePtr(),
                                     mDiffInputs.dimZ(),
                                     mDiffInputs.dimY(),
                                     mDiffInputs.dimX(),
                                     mDiffInputs.dimB(),
                                     beta,
                                     diffOutput->getDevicePtr(),
                                     mDiffOutputs[k].dimZ(),
                                     mDiffOutputs[k].dimY(),
                                     mDiffOutputs[k].dimX());
        }
        else {
            throw std::runtime_error("Unsupported norm.");
        }

        mDiffOutputs[k].deviceTensor() = *diffOutput;
        mDiffOutputs[k].setValid();
    }
}
//Float type backpropagation for NormalizeCell_Frame_CUDA
template <>
void N2D2::NormalizeCell_Frame_CUDA<float>::backPropagate()
{
    if (mDiffOutputs.empty() || !mDiffInputs.isValid())
        return;

    mDiffInputs.synchronizeHBasedToD();
    Cell_Frame_CUDA<float>::backPropagate();

    const float alpha = 1.0f;

    for (unsigned int k = 0, size = mInputs.size(); k < size; ++k) {
        const float beta = (mDiffOutputs[k].isValid()) ? 1.0f : 0.0f;

        std::shared_ptr<CudaDeviceTensor<float> > diffOutput
            = (mDiffOutputs[k].isValid())
                ? cuda_device_tensor_cast<float>(mDiffOutputs[k])
                : cuda_device_tensor_cast_nocopy<float>(mDiffOutputs[k]);

        if (mNorm == L2) {
            cudaSNormalizeL2Backward(CudaContext::getDeviceProp(),
                                     alpha,
                                     mOutputs.getDevicePtr(),
                                     mNormData.getDevicePtr(),
                                     mDiffInputs.getDevicePtr(),
                                     mDiffInputs.dimZ(),
                                     mDiffInputs.dimY(),
                                     mDiffInputs.dimX(),
                                     mDiffInputs.dimB(),
                                     beta,
                                     diffOutput->getDevicePtr(),
                                     mDiffOutputs[k].dimZ(),
                                     mDiffOutputs[k].dimY(),
                                     mDiffOutputs[k].dimX());
        }
        else {
            throw std::runtime_error("Unsupported norm.");
        }

        mDiffOutputs[k].deviceTensor() = *diffOutput;
        mDiffOutputs[k].setValid();
    }
}
//Double type backpropagation for NormalizeCell_Frame_CUDA
template <>
void N2D2::NormalizeCell_Frame_CUDA<double>::backPropagate()
{
    if (mDiffOutputs.empty() || !mDiffInputs.isValid())
        return;

    mDiffInputs.synchronizeHBasedToD();
    Cell_Frame_CUDA<double>::backPropagate();

    const double alpha = 1.0;

    for (unsigned int k = 0, size = mInputs.size(); k < size; ++k) {
        const double beta = (mDiffOutputs[k].isValid())
                                            ? 1.0
                                            : 0.0;

        std::shared_ptr<CudaDeviceTensor<double> > diffOutput
            = (mDiffOutputs[k].isValid())
                ? cuda_device_tensor_cast<double>(mDiffOutputs[k])
                : cuda_device_tensor_cast_nocopy<double>(mDiffOutputs[k]);

        if (mNorm == L2) {
            cudaDNormalizeL2Backward(CudaContext::getDeviceProp(),
                                     alpha,
                                     mOutputs.getDevicePtr(),
                                     mNormData.getDevicePtr(),
                                     mDiffInputs.getDevicePtr(),
                                     mDiffInputs.dimZ(),
                                     mDiffInputs.dimY(),
                                     mDiffInputs.dimX(),
                                     mDiffInputs.dimB(),
                                     beta,
                                     diffOutput->getDevicePtr(),
                                     mDiffOutputs[k].dimZ(),
                                     mDiffOutputs[k].dimY(),
                                     mDiffOutputs[k].dimX());
        }
        else {
            throw std::runtime_error("Unsupported norm.");
        }

        mDiffOutputs[k].deviceTensor() = *diffOutput;
        mDiffOutputs[k].setValid();
    }
}

template<class T>
void N2D2::NormalizeCell_Frame_CUDA<T>::update() {
    // Nothing to update
}

template<class T>
void N2D2::NormalizeCell_Frame_CUDA<T>::checkGradient(double epsilon, double maxError) {
    GradientCheck<T> gc(epsilon, maxError);
    gc.initialize(mInputs,
                  mOutputs,
                  mDiffInputs,
                  std::bind(&NormalizeCell_Frame_CUDA<T>::propagate, this, false),
                  std::bind(&NormalizeCell_Frame_CUDA<T>::backPropagate, this));

    if (!mDiffOutputs.empty()) {
        for (unsigned int in = 0; in < mInputs.size(); ++in) {
            std::stringstream name;
            name << mName + "_mDiffOutputs[" << in << "]";

            gc.check(name.str(), mInputs[in], mDiffOutputs[in]);
        }
    } else {
        std::cout << Utils::cwarning << "Empty diff. outputs for cell " << mName
                  << ", could not check the gradient!" << Utils::cdef
                  << std::endl;
    }
}

namespace N2D2 {
    template class NormalizeCell_Frame_CUDA<half_float::half>;
    template class NormalizeCell_Frame_CUDA<float>;
    template class NormalizeCell_Frame_CUDA<double>;
}

#endif