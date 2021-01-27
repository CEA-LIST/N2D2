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

#include "Filler/Filler.hpp"
#include "Filler/NormalFiller.hpp"
#include "GradientCheck.hpp"
#include "Cell/FcCell_Frame_CUDA.hpp"
#include "Cell/FcCell_Frame_CUDA_Kernels.hpp"
#include "DeepNet.hpp"
#include "third_party/half.hpp"

template <>
N2D2::Registrar<N2D2::FcCell>
N2D2::FcCell_Frame_CUDA<half_float::half>::mRegistrar("Frame_CUDA",
    N2D2::FcCell_Frame_CUDA<half_float::half>::create,
    N2D2::Registrar<N2D2::FcCell>::Type<half_float::half>());

template <>
N2D2::Registrar<N2D2::FcCell>
N2D2::FcCell_Frame_CUDA<float>::mRegistrar("Frame_CUDA",
    N2D2::FcCell_Frame_CUDA<float>::create,
    N2D2::Registrar<N2D2::FcCell>::Type<float>());

template <>
N2D2::Registrar<N2D2::FcCell>
N2D2::FcCell_Frame_CUDA<double>::mRegistrar("Frame_CUDA",
    N2D2::FcCell_Frame_CUDA<double>::create,
    N2D2::Registrar<N2D2::FcCell>::Type<double>());

template <class T>
N2D2::FcCell_Frame_CUDA<T>::FcCell_Frame_CUDA(const DeepNet& deepNet, const std::string& name,
                                           unsigned int nbOutputs,
                                           const std::shared_ptr
                                           <Activation>& activation)
    : Cell(deepNet, name, nbOutputs),
      FcCell(deepNet, name, nbOutputs),
      Cell_Frame_CUDA<T>(deepNet, name, nbOutputs, activation),
      // IMPORTANT: Do not change the value of the parameters here! Use
      // setParameter() or loadParameters().
      mOnesVector(NULL)
{
    // ctor
    mWeightsFiller = std::make_shared<NormalFiller<T> >(0.0, 0.05);
    mBiasFiller = std::make_shared<NormalFiller<T> >(0.0, 0.05);
    mWeightsSolver = std::make_shared<SGDSolver_Frame_CUDA<T> >();
    mBiasSolver = std::make_shared<SGDSolver_Frame_CUDA<T> >();
}

template <class T>
void N2D2::FcCell_Frame_CUDA<T>::initialize()
{
    if (!mNoBias && mBias.empty()) {
        mBias.resize({mOutputs.dimZ(), 1, 1, 1});
        mDiffBias.resize({mOutputs.dimZ(), 1, 1, 1});
        mBiasFiller->apply(mBias);
        mBias.synchronizeHToD();

        if (mOnesVector != NULL)
            cudaFree(mOnesVector);

        //  1   <-->    batch   <-->    mInputs.b()
        CHECK_CUDA_STATUS(
            cudaMalloc(&mOnesVector, mInputs.dimB() * sizeof(T)));
        std::vector<T> onesVec(mInputs.dimB(), T(1.0));
        CHECK_CUDA_STATUS(cudaMemcpy(mOnesVector,
                                    &onesVec[0],
                                    mInputs.dimB() * sizeof(T),
                                    cudaMemcpyHostToDevice));
    }

    for (unsigned int k = 0, size = mInputs.size(); k < size; ++k) {
        if (mInputs[k].size() == 0)
            throw std::runtime_error("Zero-sized input for FcCell " + mName);

        if (k < mWeightsSolvers.size())
            continue;  // already initialized, skip!

        mWeightsSolvers.push_back(mWeightsSolver->clone());
        mSynapses.push_back(new CudaTensor<T>(
            {1, 1, mInputs[k].size() / mInputs.dimB(), mOutputs.dimZ()}), 0);
        mDiffSynapses.push_back(new CudaTensor<T>(
            {1, 1, mInputs[k].size() / mInputs.dimB(), mOutputs.dimZ()}), 0);
        mWeightsFiller->apply(mSynapses.back());
        mSynapses.back().synchronizeHToD();
    }

    if (mNormalize)
        mSynapsesNorm.resize({mOutputs.dimZ()});
}

template <class T>
void N2D2::FcCell_Frame_CUDA<T>::save(const std::string& dirName) const
{
    Cell_Frame_CUDA<T>::save(dirName);

    for (unsigned int k = 0, size = mSynapses.size(); k < size; ++k) {
        std::stringstream solverName;
        solverName << "WeightsSolver-" << k;

        mWeightsSolvers[k]->save(dirName + "/" + solverName.str());
    }

    if (!mNoBias)
        mBiasSolver->save(dirName + "/BiasSolver");
}

template <class T>
void N2D2::FcCell_Frame_CUDA<T>::load(const std::string& dirName)
{
    Cell_Frame_CUDA<T>::load(dirName);

    for (unsigned int k = 0, size = mSynapses.size(); k < size; ++k) {
        std::stringstream solverName;
        solverName << "WeightsSolver-" << k;

        mWeightsSolvers[k]->load(dirName + "/" + solverName.str());
    }

    if (!mNoBias)
        mBiasSolver->load(dirName + "/BiasSolver");
}

template <class T>
void N2D2::FcCell_Frame_CUDA<T>::propagate(bool inference)
{
    mInputs.synchronizeHBasedToD();

    if (mNormalize) {
        mSynapsesNorm.deviceTensor().fill(T(0.0f));

        for (unsigned int k = 0, size = mInputs.size(); k < size; ++k) {
            cudaFcWeightsSumSq(CudaContext::getDeviceProp(),
                                mSynapses[k].getDevicePtr(),
                                mSynapsesNorm.getDevicePtr(),
                                mInputs[k].size() / mInputs.dimB(),
                                mOutputs.dimZ());
        }

        cudaFcWeightsSqrt(CudaContext::getDeviceProp(),
                            mSynapsesNorm.getDevicePtr(),
                            mOutputs.dimZ(),
                            T(1.0e-6));

        for (unsigned int k = 0, size = mInputs.size(); k < size; ++k) {
            cudaFcWeightsNormalize(CudaContext::getDeviceProp(),
                                mSynapses[k].getDevicePtr(),
                                mSynapsesNorm.getDevicePtr(),
                                mInputs[k].size() / mInputs.dimB(),
                                mOutputs.dimZ());
        }
    }

    const T alpha(1.0f);
    T beta(0.0f);

    for (unsigned int k = 0, size = mInputs.size(); k < size; ++k) {
        const unsigned int inputSize = mInputs[k].dimX() * mInputs[k].dimY()
                                       * mInputs[k].dimZ();
        if (k > 0)
            beta = 1.0f;

        std::shared_ptr<CudaDeviceTensor<T> > input
            = cuda_device_tensor_cast<T>(mInputs[k]);

        // Computes mOutputs = alpha*mSynapses'*mInputs + beta*mOutputs
        CHECK_CUBLAS_STATUS(cublasGemm(
            CudaContext::cublasHandle(),
            CUBLAS_OP_T, // mSynapses'
            CUBLAS_OP_N, // mInputs
            mOutputs.dimZ(), // nb rows in mSynapses' and mOutputs
            mInputs.dimB(), // nb cols in mInputs and mOutputs
            inputSize, // nb cols in mSynapses' and nb rows in mInputs
            reinterpret_cast<const typename Cuda::cuda_type<T>::type*>(&alpha),
            reinterpret_cast<const typename Cuda::cuda_type<T>::type*>(mSynapses[k].getDevicePtr()),
            inputSize,
            reinterpret_cast<const typename Cuda::cuda_type<T>::type*>(input->getDevicePtr()),
            inputSize,
            reinterpret_cast<const typename Cuda::cuda_type<T>::type*>(&beta),
            reinterpret_cast<typename Cuda::cuda_type<T>::type*>(mOutputs.getDevicePtr()),
            mOutputs.dimZ()));
    }

    if (!mNoBias) {
        // Computes mOutputs = alpha*mBias*mOnesVector + alpha*mOutputs
        CHECK_CUBLAS_STATUS(cublasGemm(
            CudaContext::cublasHandle(),
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            mOutputs.dimZ(),
            mInputs.dimB(),
            1,
            reinterpret_cast<const typename Cuda::cuda_type<T>::type*>(&alpha),
            reinterpret_cast<const typename Cuda::cuda_type<T>::type*>(mBias.getDevicePtr()),
            mOutputs.dimZ(),
            reinterpret_cast<const typename Cuda::cuda_type<T>::type*>(mOnesVector),
            1,
            reinterpret_cast<const typename Cuda::cuda_type<T>::type*>(&alpha),
            reinterpret_cast<typename Cuda::cuda_type<T>::type*>(mOutputs.getDevicePtr()),
            mOutputs.dimZ()));
    }

    Cell_Frame_CUDA<T>::propagate(inference);
    mDiffInputs.clearValid();
    mDiffSynapses.clearValid();
    mDiffBias.clearValid();
}

template <class T>
void N2D2::FcCell_Frame_CUDA<T>::backPropagate()
{
    if (!mDiffInputs.isValid())
        return;

    Cell_Frame_CUDA<T>::backPropagate();

    //  1   <-->    batch   <-->    mInputs.b()

    const T alpha(1.0f);

    for (unsigned int k = 0, size = mInputs.size(); k < size; ++k) {
        const unsigned int inputSize = mInputs[k].dimX() * mInputs[k].dimY()
                                       * mInputs[k].dimZ();
        const T beta((mWeightsSolvers[k]->isNewIteration()) ? 0.0f : 1.0f);

        std::shared_ptr<CudaDeviceTensor<T> > input
            = cuda_device_tensor_cast_nocopy<T>(mInputs[k]);

        // mDiffSynapses.getDevicePtr() = mInputs.getDevicePtr *
        // mDiffInputs.getDevicePtr*
        CHECK_CUBLAS_STATUS(cublasGemm(
            CudaContext::cublasHandle(),
            CUBLAS_OP_N,
            CUBLAS_OP_T,
            inputSize,
            mOutputs.dimZ(),
            mInputs.dimB(),
            reinterpret_cast<const typename Cuda::cuda_type<T>::type*>(&alpha),
            reinterpret_cast<const typename Cuda::cuda_type<T>::type*>(input->getDevicePtr()),
            inputSize,
            reinterpret_cast<const typename Cuda::cuda_type<T>::type*>(mDiffInputs.getDevicePtr()),
            mOutputs.dimZ(),
            reinterpret_cast<const typename Cuda::cuda_type<T>::type*>(&beta),
            reinterpret_cast<typename Cuda::cuda_type<T>::type*>(mDiffSynapses[k].getDevicePtr()),
            inputSize));

        mDiffSynapses[k].setValid();
    }

    if (!mNoBias) {
        const T beta((mBiasSolver->isNewIteration()) ? 0.0f : 1.0f);

        // mDiffBias.getDevicePtr() = mDiffInputs.getDevicePtr * mOnesVector
        CHECK_CUBLAS_STATUS(cublasGemv(
            CudaContext::cublasHandle(),
            CUBLAS_OP_N,
            mOutputs.dimZ(),
            mInputs.dimB(),
            reinterpret_cast<const typename Cuda::cuda_type<T>::type*>(&alpha),
            reinterpret_cast<const typename Cuda::cuda_type<T>::type*>(mDiffInputs.getDevicePtr()),
            mOutputs.dimZ(),
            reinterpret_cast<const typename Cuda::cuda_type<T>::type*>(mOnesVector),
            1,
            reinterpret_cast<const typename Cuda::cuda_type<T>::type*>(&beta),
            reinterpret_cast<typename Cuda::cuda_type<T>::type*>(mDiffBias.getDevicePtr()),
            1));

        mDiffBias.setValid();
    }

    if (!mDiffOutputs.empty() && mBackPropagate) {
        for (unsigned int k = 0, size = mInputs.size(); k < size; ++k) {
            const T betaData((mDiffOutputs[k].isValid()) ? 1.0f : 0.0f);
            const unsigned int diffOutputSize = mDiffOutputs[k].dimX()
                                                * mDiffOutputs[k].dimY()
                                                * mDiffOutputs[k].dimZ();

            std::shared_ptr<CudaDeviceTensor<T> > diffOutput
                = (mDiffOutputs[k].isValid())
                    ? cuda_device_tensor_cast<T>(mDiffOutputs[k])
                    : cuda_device_tensor_cast_nocopy<T>(mDiffOutputs[k]);

            // mDiffOutputs.getDevicePtr = mSynapses.getDevicePtr() *
            // mDiffInputs.getDevicePtr
            CHECK_CUBLAS_STATUS(cublasGemm(
                CudaContext::cublasHandle(),
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                diffOutputSize,
                mInputs.dimB(),
                mOutputs.dimZ(),
                reinterpret_cast<const typename Cuda::cuda_type<T>::type*>(&alpha),
                reinterpret_cast<const typename Cuda::cuda_type<T>::type*>(mSynapses[k].getDevicePtr()),
                diffOutputSize,
                reinterpret_cast<const typename Cuda::cuda_type<T>::type*>(mDiffInputs.getDevicePtr()),
                mOutputs.dimZ(),
                reinterpret_cast<const typename Cuda::cuda_type<T>::type*>(&betaData),
                reinterpret_cast<typename Cuda::cuda_type<T>::type*>(diffOutput->getDevicePtr()),
                diffOutputSize));

            mDiffOutputs[k].deviceTensor() = *diffOutput;
            mDiffOutputs[k].setValid();
        }

        mDiffOutputs.synchronizeDToHBased();
    } // Sinon il s'agit de la première couche, inutile de calculer
}

template <class T>
void N2D2::FcCell_Frame_CUDA<T>::update()
{
    for (unsigned int k = 0, size = mSynapses.size(); k < size; ++k) {
        if (mDiffSynapses[k].isValid()) {
            mWeightsSolvers[k]
                ->update(mSynapses[k], mDiffSynapses[k], mInputs.dimB());
        }
    }

    if (!mNoBias && mDiffBias.isValid())
        mBiasSolver->update(mBias, mDiffBias, mInputs.dimB());
}

template <class T>
void N2D2::FcCell_Frame_CUDA<T>::checkGradient(double epsilon, double maxError)
{
    GradientCheck<T> gc(epsilon, maxError);
    gc.initialize(mInputs,
                  mOutputs,
                  mDiffInputs,
                  std::bind(&FcCell_Frame_CUDA<T>::propagate, this, false),
                  std::bind(&FcCell_Frame_CUDA<T>::backPropagate, this));

    for (unsigned int k = 0, size = mSynapses.size(); k < size; ++k) {
        std::stringstream name;
        name << mName + "mDiffSynapses[" << k << "]";

        gc.check(name.str(), mSynapses[k], mDiffSynapses[k]);
    }

    if (!mNoBias)
        gc.check(mName + "_mDiffBias", mBias, mDiffBias);

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

template <class T>
void N2D2::FcCell_Frame_CUDA<T>::logFreeParameters(const std::string& fileName,
                                                unsigned int output) const
{
    synchronizeToH(false);
    FcCell::logFreeParameters(fileName, output);
    keepInSync(true);
}

template <class T>
void N2D2::FcCell_Frame_CUDA<T>::logFreeParameters(const std::string
                                                & dirName) const
{
    synchronizeToH(false);
    FcCell::logFreeParameters(dirName);
    keepInSync(true);
}

template <class T>
void N2D2::FcCell_Frame_CUDA<T>::saveFreeParameters(const std::string
                                                 & fileName) const
{
    std::ofstream syn(fileName.c_str(), std::fstream::binary);

    if (!syn.good())
        throw std::runtime_error("Could not create synaptic file (.SYN): "
                                 + fileName);

    mSynapses.synchronizeDToH();

    for (unsigned int k = 0; k < mSynapses.size(); ++k)
        mSynapses[k].save(syn);

    if (!mNoBias) {
        mBias.synchronizeDToH();
        mBias.save(syn);
    }

    if (!syn.good())
        throw std::runtime_error("Error writing synaptic file: " + fileName);
}

template <class T>
void N2D2::FcCell_Frame_CUDA<T>::loadFreeParameters(const std::string& fileName,
                                                 bool ignoreNotExists)
{
    std::ifstream syn(fileName.c_str(), std::fstream::binary);

    if (!syn.good()) {
        if (ignoreNotExists) {
            std::cout << Utils::cnotice
                      << "Notice: Could not open synaptic file (.SYN): "
                      << fileName << Utils::cdef << std::endl;
            return;
        } else
            throw std::runtime_error("Could not open synaptic file (.SYN): "
                                     + fileName);
    }

    for (unsigned int k = 0; k < mSynapses.size(); ++k)
        mSynapses[k].load(syn);

    mSynapses.synchronizeHToD();

    if (!mNoBias) {
        mBias.load(syn);
        mBias.synchronizeHToD();
    }

    if (syn.eof())
        throw std::runtime_error(
            "End-of-file reached prematurely in synaptic file (.SYN): "
            + fileName);
    else if (!syn.good())
        throw std::runtime_error("Error while reading synaptic file (.SYN): "
                                 + fileName);
    else if (syn.get() != std::fstream::traits_type::eof())
        throw std::runtime_error(
            "Synaptic file (.SYN) size larger than expected: " + fileName);
}

template <class T>
void N2D2::FcCell_Frame_CUDA<T>::exportFreeParameters(const std::string
                                                   & fileName) const
{
    synchronizeToH(false);
    FcCell::exportFreeParameters(fileName);
    keepInSync(true);
}

template <class T>
void N2D2::FcCell_Frame_CUDA<T>::importFreeParameters(const std::string& fileName,
                                                   bool ignoreNotExists)
{
    keepInSync(false);
    FcCell::importFreeParameters(fileName, ignoreNotExists);
    synchronizeToD(true);
}

template <class T>
void N2D2::FcCell_Frame_CUDA<T>::logFreeParametersDistrib(const std::string
                                                       & fileName) const
{
    synchronizeToH(false);
    FcCell::logFreeParametersDistrib(fileName);
    keepInSync(true);
}

template <class T>
std::pair<N2D2::Float_T, N2D2::Float_T>
N2D2::FcCell_Frame_CUDA<T>::getFreeParametersRange(bool withAdditiveParameters) const
{
    const bool keepInSyncTop(mKeepInSync);

    if (keepInSyncTop)
        synchronizeToH(false);

    const std::pair<Float_T, Float_T> range = FcCell::getFreeParametersRange(withAdditiveParameters);

    if (keepInSyncTop)
        keepInSync(true);

    return range;
}

template <class T>
std::pair<N2D2::Float_T, N2D2::Float_T>
N2D2::FcCell_Frame_CUDA<T>::getFreeParametersRangePerOutput(std::size_t output, 
                                                            bool withAdditiveParameters) const
{
    const bool keepInSyncTop(mKeepInSync);

    if (keepInSyncTop)
        synchronizeToH(false);

    const std::pair<Float_T, Float_T> range = FcCell::getFreeParametersRangePerOutput(output, withAdditiveParameters);

    if (keepInSyncTop)
        keepInSync(true);

    return range;
}

template <class T>
std::pair<N2D2::Float_T, N2D2::Float_T>
N2D2::FcCell_Frame_CUDA<T>::getFreeParametersRangePerChannel(std::size_t channel) const
{
    const bool keepInSyncTop(mKeepInSync);

    if (keepInSyncTop)
        synchronizeToH(false);

    const std::pair<Float_T, Float_T> range = FcCell::getFreeParametersRangePerChannel(channel);

    if (keepInSyncTop)
        keepInSync(true);

    return range;
}

template <class T>
void N2D2::FcCell_Frame_CUDA<T>::processFreeParameters(std::function<Float_T(Float_T)> func,
                                                       FreeParametersType type)
{
    const bool keepInSyncTop(mKeepInSync);

    if (keepInSyncTop)
        synchronizeToH(false);

    FcCell::processFreeParameters(func, type);

    if (keepInSyncTop)
        synchronizeToD(true);
}

template <class T>
void N2D2::FcCell_Frame_CUDA<T>::processFreeParametersPerOutput(std::function<Float_T(Float_T)> func,
                                                                std::size_t output,
                                                                FreeParametersType type)
{
    const bool keepInSyncTop(mKeepInSync);

    if (keepInSyncTop)
        synchronizeToH(false);

    FcCell::processFreeParametersPerOutput(func, output, type);

    if (keepInSyncTop)
        synchronizeToD(true);
}

template <class T>
void N2D2::FcCell_Frame_CUDA<T>::processFreeParametersPerChannel(std::function<Float_T(Float_T)> func,
                                                                std::size_t channel)
{
    const bool keepInSyncTop(mKeepInSync);

    if (keepInSyncTop)
        synchronizeToH(false);

    FcCell::processFreeParametersPerChannel(func, channel);

    if (keepInSyncTop)
        synchronizeToD(true);
}

template <class T>
void N2D2::FcCell_Frame_CUDA<T>::synchronizeToH(bool keepInSync_) const
{
    mSynapses.synchronizeDToH();
    mBias.synchronizeDToH();
    keepInSync(keepInSync_);
}

template <class T>
void N2D2::FcCell_Frame_CUDA<T>::synchronizeToD(bool keepInSync_)
{
    mSynapses.synchronizeHToD();
    mBias.synchronizeHToD();
    keepInSync(keepInSync_);
}

template <class T>
N2D2::FcCell_Frame_CUDA<T>::~FcCell_Frame_CUDA()
{
    if (mOnesVector != NULL) {
        cudaFree(mOnesVector);
        mOnesVector = NULL;
    }
}

namespace N2D2 {
    template class FcCell_Frame_CUDA<half_float::half>;
    template class FcCell_Frame_CUDA<float>;
    template class FcCell_Frame_CUDA<double>;
}

#endif
