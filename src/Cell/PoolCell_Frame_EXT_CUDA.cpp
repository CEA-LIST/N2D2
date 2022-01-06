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
#include "Cell/PoolCell_Frame_EXT_CUDA.hpp"
#include "DeepNet.hpp"
#include "third_party/half.hpp"

template <>
N2D2::Registrar<N2D2::PoolCell>
N2D2::PoolCell_Frame_EXT_CUDA<half_float::half>::mRegistrar("Frame_EXT_CUDA",
        N2D2::PoolCell_Frame_EXT_CUDA<half_float::half>::create,
        N2D2::Registrar<N2D2::PoolCell>::Type<half_float::half>());
template <>
N2D2::Registrar<N2D2::PoolCell>
N2D2::PoolCell_Frame_EXT_CUDA<float>::mRegistrar("Frame_EXT_CUDA",
        N2D2::PoolCell_Frame_EXT_CUDA<float>::create,
        N2D2::Registrar<N2D2::PoolCell>::Type<float>());
template <>
N2D2::Registrar<N2D2::PoolCell>
N2D2::PoolCell_Frame_EXT_CUDA<double>::mRegistrar("Frame_EXT_CUDA",
        N2D2::PoolCell_Frame_EXT_CUDA<double>::create,
        N2D2::Registrar<N2D2::PoolCell>::Type<double>());

template <class T>
N2D2::PoolCell_Frame_EXT_CUDA<T>::PoolCell_Frame_EXT_CUDA(
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
      Cell_Frame_CUDA<T>(deepNet, name, nbOutputs, activation),
      mPoolDesc(NULL)
{
    // ctor
    if (mPoolDims.size() != 2) {
        throw std::domain_error("PoolCell_Frame_EXT_CUDA: only 2D pooling is"
                                " supported");
    }

    if (strideDims.size() != poolDims.size()) {
        throw std::domain_error("PoolCell_Frame_EXT_CUDA: the number of dimensions"
                                " of stride must match the number of"
                                " dimensions of the pooling.");
    }

    if (paddingDims.size() != poolDims.size()) {
        throw std::domain_error("PoolCell_Frame_EXT_CUDA: the number of dimensions"
                                " of padding must match the number of"
                                " dimensions of the pooling.");
    }

    const PoolCell_Frame_Kernels::Descriptor poolDesc(mPoolDims.size(),
                                                      &mPoolDims[0],
                                                      &mStrideDims[0],
                                                      &mPaddingDims[0]);

    CHECK_CUDA_STATUS(cudaMalloc((void**)&mPoolDesc, sizeof(poolDesc)));
    CHECK_CUDA_STATUS(cudaMemcpy(mPoolDesc,
                                 &poolDesc,
                                 sizeof(poolDesc),
                                 cudaMemcpyHostToDevice));
}

template <class T>
void N2D2::PoolCell_Frame_EXT_CUDA<T>::setExtendedPadding(
    const std::vector<int>& paddingDims)
{
    PoolCell::setExtendedPadding(paddingDims);

    PoolCell_Frame_Kernels::Descriptor poolDesc(mPoolDims.size(),
                                                &mPoolDims[0],
                                                &mStrideDims[0],
                                                &mPaddingDims[0]);

    for (std::size_t dim = 0; dim < mPaddingDims.size(); ++dim) {
        // Don't care about bottom/right padding, not used anywhere
        poolDesc.padding[dim] = mPaddingDims[dim] + paddingDims[dim];
    }

    CHECK_CUDA_STATUS(cudaMemcpy(mPoolDesc,
                                 &poolDesc,
                                 sizeof(poolDesc),
                                 cudaMemcpyHostToDevice));
}

template <class T>
void N2D2::PoolCell_Frame_EXT_CUDA<T>::initialize()
{
    unsigned int offset = 0;

    for (unsigned int k = 0, size = mInputs.size(); k < size; ++k) {
        if (mInputs[k].size() == 0)
            throw std::runtime_error("Zero-sized input for PoolCell " + mName);

        if (mInputMap.size() == k) {
            mInputMap.push_back(NULL);

            if (!mMapping.empty()) {
                const Tensor<bool> inputMap = mMapping.rows(offset,
                                                           mInputs[k].dimZ());

                std::vector<char> inputMapData;
                std::copy(inputMap.begin(), inputMap.end(),
                          std::back_inserter(inputMapData));

                CHECK_CUDA_STATUS(cudaMalloc(&mInputMap[k],
                                             inputMapData.size()
                                                * sizeof(char)));
                CHECK_CUDA_STATUS(cudaMemcpy(mInputMap[k],
                                             &inputMapData[0],
                                             inputMapData.size() * sizeof(char),
                                             cudaMemcpyHostToDevice));
            }
        }

        if (mArgMax.size() == k) {
            mArgMax.push_back(new CudaTensor<PoolCell_Frame_Kernels::ArgMax>
                              (mOutputs.dims()));
        }

        offset += mInputs[k].dimZ();
    }
}

template <class T>
void N2D2::PoolCell_Frame_EXT_CUDA<T>::propagate(bool inference)
{
    mInputs.synchronizeHBasedToD();

    const T alpha(1.0f);
    T beta(0.0f);

    for (unsigned int k = 0, size = mInputs.size(); k < size; ++k) {
        if (k > 0)
            beta = T(1.0f);

        std::shared_ptr<CudaDeviceTensor<T> > input
            = cuda_device_tensor_cast<T>(mInputs[k]);

        if (mPooling == Max) {
            cudaPoolForwardMax(CudaContext::getDeviceProp(),
                                alpha,
                                input->getDevicePtr(),
                                mInputs[k].dimZ(),
                                mInputs[k].dimY(),
                                mInputs[k].dimX(),
                                mInputs[k].dimB(),
                                mPoolDesc,
                                beta,
                                mOutputs.getDevicePtr(),
                                mOutputs.dimZ(),
                                mOutputs.dimY(),
                                mOutputs.dimX(),
                                mArgMax[k].getDevicePtr(),
                                false,
                                mInputMap[k]);
        }
        else {
            cudaPoolForwardAverage(CudaContext::getDeviceProp(),
                                    alpha,
                                    input->getDevicePtr(),
                                    mInputs[k].dimZ(),
                                    mInputs[k].dimY(),
                                    mInputs[k].dimX(),
                                    mInputs[k].dimB(),
                                    mPoolDesc,
                                    beta,
                                    mOutputs.getDevicePtr(),
                                    mOutputs.dimZ(),
                                    mOutputs.dimY(),
                                    mOutputs.dimX(),
                                    true,
                                    mInputMap[k]);
        }
    }

    Cell_Frame_CUDA<T>::propagate(inference);
    mDiffInputs.clearValid();
}

template <class T>
void N2D2::PoolCell_Frame_EXT_CUDA<T>::backPropagate()
{
    if (!mDiffInputs.isValid())
        return;

    mDiffInputs.synchronizeHBasedToD();
    Cell_Frame_CUDA<T>::backPropagate();

    const T alpha(1.0f);

    for (unsigned int k = 0, size = mInputs.size(); k < size; ++k) {
        if (mDiffOutputs[k].empty())
            continue;

        const T beta((mDiffOutputs[k].isValid()) ? 1.0f : 0.0f);

        std::shared_ptr<CudaDeviceTensor<T> > diffOutput
            = (mDiffOutputs[k].isValid())
                ? cuda_device_tensor_cast<T>(mDiffOutputs[k])
                : cuda_device_tensor_cast_nocopy<T>(mDiffOutputs[k]);

        if (mPooling == Max) {
            cudaPoolBackwardMax(CudaContext::getDeviceProp(),
                                 alpha,
                                 mDiffInputs.getDevicePtr(),
                                 mDiffInputs.dimZ(),
                                 mDiffInputs.dimY(),
                                 mDiffInputs.dimX(),
                                 mDiffInputs.dimB(),
                                 mPoolDesc,
                                 beta,
                                 diffOutput->getDevicePtr(),
                                 mDiffOutputs[k].dimZ(),
                                 mDiffOutputs[k].dimY(),
                                 mDiffOutputs[k].dimX(),
                                 mArgMax[k].getDevicePtr(),
                                 mInputMap[k]);
        }
        else {
            cudaPoolBackwardAverage(CudaContext::getDeviceProp(),
                                     alpha,
                                     mDiffInputs.getDevicePtr(),
                                     mDiffInputs.dimZ(),
                                     mDiffInputs.dimY(),
                                     mDiffInputs.dimX(),
                                     mDiffInputs.dimB(),
                                     mPoolDesc,
                                     beta,
                                     diffOutput->getDevicePtr(),
                                     mDiffOutputs[k].dimZ(),
                                     mDiffOutputs[k].dimY(),
                                     mDiffOutputs[k].dimX(),
                                     true,
                                     mInputMap[k]);
        }

        mDiffOutputs[k].deviceTensor() = *diffOutput;
        mDiffOutputs[k].setValid();
    }

    mDiffOutputs.synchronizeDToHBased();
}

template <class T>
void N2D2::PoolCell_Frame_EXT_CUDA<T>::update()
{
}

template <class T>
void N2D2::PoolCell_Frame_EXT_CUDA<T>::checkGradient(double epsilon, double maxError)
{
    GradientCheck<T> gc(epsilon, maxError);
    gc.initialize(mInputs,
                  mOutputs,
                  mDiffInputs,
                  std::bind(&PoolCell_Frame_EXT_CUDA<T>::propagate, this, false),
                  std::bind(&PoolCell_Frame_EXT_CUDA<T>::backPropagate, this),
                  (mPooling == Max));

    for (unsigned int k = 0; k < mInputs.size(); ++k) {
        if (mDiffOutputs[k].empty()) {
            std::cout << Utils::cwarning << "Empty diff. outputs #" << k
                    << " for cell " << mName
                    << ", could not check the gradient!" << Utils::cdef
                    << std::endl;
            continue;
        }

        std::stringstream name;
        name << mName + "_mDiffOutputs[" << k << "]";

        gc.check(name.str(), mInputs[k], mDiffOutputs[k]);
    }
}

template <class T>
N2D2::PoolCell_Frame_EXT_CUDA<T>::~PoolCell_Frame_EXT_CUDA()
{
    if (mPoolDesc != NULL)
        cudaFree(mPoolDesc);

    for (unsigned int k = 0, size = mArgMax.size(); k < size; ++k)
        delete &mArgMax[k];

    for (unsigned int k = 0, size = mInputMap.size(); k < size; ++k) {
        if (mInputMap[k] != NULL) {
            cudaFree(mInputMap[k]);
            mInputMap[k] = NULL;
        }
    }
}
namespace N2D2 {
    template class PoolCell_Frame_EXT_CUDA<half_float::half>;
    template class PoolCell_Frame_EXT_CUDA<float>;
    template class PoolCell_Frame_EXT_CUDA<double>;
}
#endif
