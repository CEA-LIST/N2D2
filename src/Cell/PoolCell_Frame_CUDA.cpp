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

#ifdef CUDA

#include "GradientCheck.hpp"
#include "Cell/PoolCell_Frame_CUDA.hpp"
#include "DeepNet.hpp"
#include "third_party/half.hpp"

template <>
N2D2::Registrar<N2D2::PoolCell>
N2D2::PoolCell_Frame_CUDA<half_float::half>::mRegistrar("Frame_CUDA",
        N2D2::PoolCell_Frame_CUDA<half_float::half>::create,
        N2D2::Registrar<N2D2::PoolCell>::Type<half_float::half>());

template <>
N2D2::Registrar<N2D2::PoolCell>
N2D2::PoolCell_Frame_CUDA<float>::mRegistrar("Frame_CUDA",
        N2D2::PoolCell_Frame_CUDA<float>::create,
        N2D2::Registrar<N2D2::PoolCell>::Type<float>());

template <>
N2D2::Registrar<N2D2::PoolCell>
N2D2::PoolCell_Frame_CUDA<double>::mRegistrar("Frame_CUDA",
    N2D2::PoolCell_Frame_CUDA<double>::create,
    N2D2::Registrar<N2D2::PoolCell>::Type<double>());

template <class T>
N2D2::PoolCell_Frame_CUDA<T>::PoolCell_Frame_CUDA(
    const DeepNet& deepNet, 
    const std::string& name,
    const std::vector<unsigned int>& poolDims,
    unsigned int nbOutputs,
    const std::vector<unsigned int>& strideDims,
    const std::vector<unsigned int>& paddingDims,
    Pooling pooling,
    const std::shared_ptr<Activation>& activation)
    : Cell(deepNet, name, nbOutputs),
      PoolCell(deepNet, name,
               poolDims,
               nbOutputs,
               strideDims,
               paddingDims,
               pooling),
      Cell_Frame_CUDA<T>(deepNet, name, nbOutputs, activation)
{
    // ctor
    assert(poolDims.size() <= POOL_KERNEL_MAX_DIMS);

    if (strideDims.size() != poolDims.size()) {
        throw std::domain_error("PoolCell_Frame_CUDA: the number of dimensions"
                                " of stride must match the number of"
                                " dimensions of the pooling.");
    }

    if (paddingDims.size() != poolDims.size()) {
        throw std::domain_error("PoolCell_Frame_CUDA: the number of dimensions"
                                " of padding must match the number of"
                                " dimensions of the pooling.");
    }

    CHECK_CUDNN_STATUS(cudnnCreatePoolingDescriptor(&mPoolingDesc));
}

template <class T>
void N2D2::PoolCell_Frame_CUDA<T>::initialize()
{
    if (!isUnitMap()) {
        throw std::domain_error(
            "PoolCell_Frame_CUDA::initialize(): only unit maps are "
                "supported for cell " + mName + ".");
    }

    for (unsigned int k = 0, size = mInputs.size(); k < size; ++k) {
        if (mInputs[k].size() == 0)
            throw std::runtime_error("Zero-sized input for PoolCell " + mName);

        if (k < mOutputDesc.size())
            continue;  // already initialized, skip!

        mOutputDesc.push_back(cudnnTensorDescriptor_t());

        std::vector<int> dims(mOutputs.dims().begin(), mOutputs.dims().end());
        dims[dims.size() - 2] = mInputs[k].dimZ();

        std::vector<int> strides;
        unsigned int stride = 1;

        for (unsigned int dim = 0; dim < mOutputs.nbDims(); ++dim) {
            strides.push_back(stride);

            if (dim < mOutputs.nbDims() - 2)
                stride *= mOutputs.dims()[dim];
            else
                stride *= mInputs.dimZ();
        }

        std::reverse(dims.begin(), dims.end());
        std::reverse(strides.begin(), strides.end());

        CHECK_CUDNN_STATUS(cudnnCreateTensorDescriptor(&mOutputDesc.back()));
        CHECK_CUDNN_STATUS(cudnnSetTensorNdDescriptor(
            mOutputDesc.back(),
            CudaContext::data_type<T>::value,
            mOutputs.nbDims(),
            &dims[0],
            &strides[0]));
    }

    const cudnnPoolingMode_t poolingMode
        = (mPooling == Max) ? CUDNN_POOLING_MAX
                            : CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;

    const std::vector<int> pools(mPoolDims.rbegin(), mPoolDims.rend());
    const std::vector<int> paddings(mPaddingDims.rbegin(), mPaddingDims.rend());
    const std::vector<int> strides(mStrideDims.rbegin(), mStrideDims.rend());

#if CUDNN_VERSION >= 5000
    CHECK_CUDNN_STATUS(cudnnSetPoolingNdDescriptor(
        mPoolingDesc,
        poolingMode,
        CUDNN_PROPAGATE_NAN,
        //CUDNN_NOT_PROPAGATE_NAN,
        mPoolDims.size(),
        &pools[0],
        &paddings[0],
        &strides[0]));
#else
    CHECK_CUDNN_STATUS(cudnnSetPoolingNdDescriptor(
        mPoolingDesc,
        poolingMode,
        mPoolDims.size(),
        &pools[0],
        &paddings[0],
        &strides[0]));
#endif
}

template <class T>
void N2D2::PoolCell_Frame_CUDA<T>::propagate(bool inference)
{
    mInputs.synchronizeHBasedToD();

    const typename Cuda::cudnn_scaling_type<T>::type alpha = 1.0f;
    const typename Cuda::cudnn_scaling_type<T>::type beta = 0.0f;

    unsigned int offset = 0;

    for (unsigned int k = 0, size = mInputs.size(); k < size; ++k) {
        std::shared_ptr<CudaDeviceTensor<T> > input
            = cuda_device_tensor_cast<T>(mInputs[k]);

        CHECK_CUDNN_STATUS(
            cudnnPoolingForward(CudaContext::cudnnHandle(),
                                mPoolingDesc,
                                &alpha,
                                input->getCudnnTensorDesc(),
                                input->getDevicePtr(),
                                &beta,
                                mOutputDesc[k],
                                mOutputs.getDevicePtr() + offset));

        offset += mOutputs.dimX() * mOutputs.dimY() * mInputs[k].dimZ();
    }

    Cell_Frame_CUDA<T>::propagate(inference);
    mDiffInputs.clearValid();
}

template <class T>
void N2D2::PoolCell_Frame_CUDA<T>::backPropagate()
{
    if (mDiffOutputs.empty() || !mDiffInputs.isValid())
        return;

    Cell_Frame_CUDA<T>::backPropagate();

    const typename Cuda::cudnn_scaling_type<T>::type alpha = 1.0f;

    unsigned int offset = 0;

    for (unsigned int k = 0, size = mInputs.size(); k < size; ++k) {
        const typename Cuda::cudnn_scaling_type<T>::type beta
                            = (mDiffOutputs[k].isValid()) ? 1.0f : 0.0f;

        std::shared_ptr<CudaDeviceTensor<T> > input
            = cuda_device_tensor_cast_nocopy<T>(mInputs[k]);
        std::shared_ptr<CudaDeviceTensor<T> > diffOutput
            = (mDiffOutputs[k].isValid())
                ? cuda_device_tensor_cast<T>(mDiffOutputs[k])
                : cuda_device_tensor_cast_nocopy<T>(mDiffOutputs[k]);

        CHECK_CUDNN_STATUS(
            cudnnPoolingBackward(CudaContext::cudnnHandle(),
                                 mPoolingDesc,
                                 &alpha,
                                 mOutputDesc[k],
                                 mOutputs.getDevicePtr() + offset,
                                 mOutputDesc[k],
                                 mDiffInputs.getDevicePtr() + offset,
                                 input->getCudnnTensorDesc(),
                                 input->getDevicePtr(),
                                 &beta,
                                 diffOutput->getCudnnTensorDesc(),
                                 diffOutput->getDevicePtr()));

        offset += mOutputs.dimX() * mOutputs.dimY() * mInputs[k].dimZ();

        mDiffOutputs[k].deviceTensor() = *diffOutput;
        mDiffOutputs[k].setValid();
    }

    mDiffOutputs.synchronizeDToHBased();
}

template <class T>
void N2D2::PoolCell_Frame_CUDA<T>::update()
{
}

template <class T>
void N2D2::PoolCell_Frame_CUDA<T>::checkGradient(double epsilon, double maxError)
{
    GradientCheck<T> gc(epsilon, maxError);
    gc.initialize(mInputs,
                  mOutputs,
                  mDiffInputs,
                  std::bind(&PoolCell_Frame_CUDA<T>::propagate, this, false),
                  std::bind(&PoolCell_Frame_CUDA<T>::backPropagate, this),
                  (mPooling == Max));

    if (!mDiffOutputs.empty()) {
        for (unsigned int k = 0, size = mInputs.size(); k < size; ++k) {
            std::stringstream name;
            name << mName + "_mDiffOutputs[" << k << "]";

            gc.check(name.str(), mInputs[k], mDiffOutputs[k]);
        }
    } else {
        std::cout << Utils::cwarning << "Empty diff. outputs for cell " << mName
                  << ", could not check the gradient!" << Utils::cdef
                  << std::endl;
    }
}

template <class T>
std::pair<double, double> N2D2::PoolCell_Frame_CUDA<T>::getOutputsRange() const {
    const auto& activation = Cell_Frame_CUDA<T>::getActivation();
    return activation?activation->getOutputRange():PoolCell::getOutputsRange();
}

template <class T>
N2D2::PoolCell_Frame_CUDA<T>::~PoolCell_Frame_CUDA()
{
    for (unsigned int k = 0, size = mOutputDesc.size(); k < size; ++k)
        cudnnDestroyTensorDescriptor(mOutputDesc[k]);

    cudnnDestroyPoolingDescriptor(mPoolingDesc);
}

namespace N2D2 {
    template class PoolCell_Frame_CUDA<half_float::half>;
    template class PoolCell_Frame_CUDA<float>;
    template class PoolCell_Frame_CUDA<double>;
}


#endif
