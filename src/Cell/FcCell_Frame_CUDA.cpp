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
      mOnesVector(NULL),
      mSynchronized(false)
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

    if (mQuantizer) {
        for (unsigned int k = 0, size = mSynapses.size(); k < size; ++k) {
            mQuantizer->addWeights(mSynapses[k], mDiffSynapses[k]);
        }
        if (!mNoBias) {
            mQuantizer->addBiases(mBias, mDiffBias);
        }
        if (!mDiffOutputs.empty()){
            for (unsigned int k = 0, size = mInputs.size(); k < size; ++k) {
                mQuantizer->addActivations(mInputs[k], mDiffOutputs[k]);
            }
        }
        else {
            for (unsigned int k = 0, size = mInputs.size(); k < size; ++k) {
                mQuantizer->addActivations(mInputs[k]);
            }
        }
        mQuantizer->initialize();
    }

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

namespace N2D2 {
template <>
void N2D2::FcCell_Frame_CUDA<half_float::half>::propagate(bool inference)
{
    if (mNormalize) {
        throw std::runtime_error("FcCell_Frame_CUDA<half_float::half>"
            "::propagate(): normalization not implemented in half precision.");
    }

    mInputs.synchronizeHBasedToD();

    const half_float::half alpha(1.0f);
    half_float::half beta(0.0f);

    for (unsigned int k = 0, size = mInputs.size(); k < size; ++k) {
        const unsigned int inputSize = mInputs[k].dimX() * mInputs[k].dimY()
                                       * mInputs[k].dimZ();
        if (k > 0)
            beta = 1.0f;

        std::shared_ptr<CudaDeviceTensor<half_float::half> > input
            = cuda_device_tensor_cast<half_float::half>(mInputs[k]);

        // Computes mOutputs = alpha*mSynapses'*mInputs + beta*mOutputs
        CHECK_CUBLAS_STATUS(cublasHgemm(
            CudaContext::cublasHandle(),
            CUBLAS_OP_T, // mSynapses'
            CUBLAS_OP_N, // mInputs
            mOutputs.dimZ(), // nb rows in mSynapses' and mOutputs
            mInputs.dimB(), // nb cols in mInputs and mOutputs
            inputSize, // nb cols in mSynapses' and nb rows in mInputs
            reinterpret_cast<const __half*>(&alpha),
            reinterpret_cast<const __half*>(mSynapses[k].getDevicePtr()),
            inputSize,
            reinterpret_cast<const __half*>(input->getDevicePtr()),
            inputSize,
            reinterpret_cast<const __half*>(&beta),
            reinterpret_cast<__half*>(mOutputs.getDevicePtr()),
            mOutputs.dimZ()));
    }

    if (!mNoBias) {
        // Computes mOutputs = alpha*mBias*mOnesVector + alpha*mOutputs
        CHECK_CUBLAS_STATUS(cublasHgemm(
            CudaContext::cublasHandle(),
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            mOutputs.dimZ(),
            mInputs.dimB(),
            1,
            reinterpret_cast<const __half*>(&alpha),
            reinterpret_cast<const __half*>(mBias.getDevicePtr()),
            mOutputs.dimZ(),
            reinterpret_cast<const __half*>(mOnesVector),
            1,
            reinterpret_cast<const __half*>(&alpha),
            reinterpret_cast<__half*>(mOutputs.getDevicePtr()),
            mOutputs.dimZ()));
    }

    Cell_Frame_CUDA<half_float::half>::propagate(inference);
    mDiffInputs.clearValid();
}

template <>
void N2D2::FcCell_Frame_CUDA<float>::propagate(bool inference)
{
    mInputs.synchronizeHBasedToD();

    if (mNormalize) {
        mSynapsesNorm.deviceTensor().fill(0.0f);

        for (unsigned int k = 0, size = mInputs.size(); k < size; ++k) {
            cudaSFcWeightsSumSq(CudaContext::getDeviceProp(),
                                mSynapses[k].getDevicePtr(),
                                mSynapsesNorm.getDevicePtr(),
                                mInputs[k].size() / mInputs.dimB(),
                                mOutputs.dimZ());
        }

        cudaSFcWeightsSqrt(CudaContext::getDeviceProp(),
                            mSynapsesNorm.getDevicePtr(),
                            mOutputs.dimZ(),
                            1.0e-6);

        for (unsigned int k = 0, size = mInputs.size(); k < size; ++k) {
            cudaSFcWeightsNormalize(CudaContext::getDeviceProp(),
                                mSynapses[k].getDevicePtr(),
                                mSynapsesNorm.getDevicePtr(),
                                mInputs[k].size() / mInputs.dimB(),
                                mOutputs.dimZ());
        }
    }

    if (mQuantizer) {
        mQuantizer->propagate();
    }

    const float alpha = 1.0f;
    float beta = 0.0f;

    for (unsigned int k = 0, size = mInputs.size(); k < size; ++k) {
        const unsigned int inputSize = mInputs[k].dimX() * mInputs[k].dimY()
                                       * mInputs[k].dimZ();
        if (k > 0)
            beta = 1.0f;

        std::shared_ptr<CudaDeviceTensor<float> > input;
        std::shared_ptr<CudaDeviceTensor<float> > synapses;

        if (mQuantizer) {
            input = cuda_device_tensor_cast<float>
                (cuda_tensor_cast<float>(mQuantizer->getQuantizedActivations(k)));
            synapses = cuda_device_tensor_cast<float>
                (cuda_tensor_cast<float>(mQuantizer->getQuantizedWeights(k)));
        }
        else {
            input = cuda_device_tensor_cast<float>(mInputs[k]);
            synapses = cuda_device_tensor_cast<float>(mSynapses[k]);
        }

        // Computes mOutputs = alpha*mSynapses'*mInputs + beta*mOutputs
        CHECK_CUBLAS_STATUS(cublasSgemm(
            CudaContext::cublasHandle(),
            CUBLAS_OP_T, // mSynapses'
            CUBLAS_OP_N, // mInputs
            mOutputs.dimZ(), // nb rows in mSynapses' and mOutputs
            mInputs.dimB(), // nb cols in mInputs and mOutputs
            inputSize, // nb cols in mSynapses' and nb rows in mInputs
            &alpha,
            synapses->getDevicePtr(),
            inputSize,
            input->getDevicePtr(),
            inputSize,
            &beta,
            mOutputs.getDevicePtr(),
            mOutputs.dimZ()));
    }

    if (!mNoBias) {

        std::shared_ptr<CudaDeviceTensor<float> > biases;

        if (mQuantizer) {
            biases = cuda_device_tensor_cast<float>
                (cuda_tensor_cast<float>(mQuantizer->getQuantizedBiases()));
        }
        else {
            biases = cuda_device_tensor_cast<float>(mBias);
        }

        // Computes mOutputs = alpha*mBias*mOnesVector + alpha*mOutputs
        CHECK_CUBLAS_STATUS(cublasSgemm(CudaContext::cublasHandle(),
                                        CUBLAS_OP_N,
                                        CUBLAS_OP_N,
                                        mOutputs.dimZ(),
                                        mInputs.dimB(),
                                        1,
                                        &alpha,
                                        biases->getDevicePtr(),
                                        mOutputs.dimZ(),
                                        mOnesVector,
                                        1,
                                        &alpha,
                                        mOutputs.getDevicePtr(),
                                        mOutputs.dimZ()));
    }

    Cell_Frame_CUDA<float>::propagate(inference);
    mDiffInputs.clearValid();
}

template <>
void N2D2::FcCell_Frame_CUDA<double>::propagate(bool inference)
{
    mInputs.synchronizeHBasedToD();

    if (mNormalize) {
        mSynapsesNorm.deviceTensor().fill(0.0);

        for (unsigned int k = 0, size = mInputs.size(); k < size; ++k) {
            cudaDFcWeightsSumSq(CudaContext::getDeviceProp(),
                                mSynapses[k].getDevicePtr(),
                                mSynapsesNorm.getDevicePtr(),
                                mInputs[k].size() / mInputs.dimB(),
                                mOutputs.dimZ());
        }

        cudaDFcWeightsSqrt(CudaContext::getDeviceProp(),
                            mSynapsesNorm.getDevicePtr(),
                            mOutputs.dimZ(),
                            1.0e-6);

        for (unsigned int k = 0, size = mInputs.size(); k < size; ++k) {
            cudaDFcWeightsNormalize(CudaContext::getDeviceProp(),
                                mSynapses[k].getDevicePtr(),
                                mSynapsesNorm.getDevicePtr(),
                                mInputs[k].size() / mInputs.dimB(),
                                mOutputs.dimZ());
        }
    }

    const double alpha = 1.0;
    double beta = 0.0;

    if (mQuantizer) {
        mQuantizer->propagate();
    }

    for (unsigned int k = 0, size = mInputs.size(); k < size; ++k) {
        const unsigned int inputSize = mInputs[k].dimX() * mInputs[k].dimY()
                                       * mInputs[k].dimZ();
        if (k > 0)
            beta = 1.0;

        std::shared_ptr<CudaDeviceTensor<double> > input;
        std::shared_ptr<CudaDeviceTensor<double> > synapses;

        if (mQuantizer) {
            input = cuda_device_tensor_cast<double>
                (cuda_tensor_cast<double>(mQuantizer->getQuantizedActivations(k)));
            synapses = cuda_device_tensor_cast<double>
                (cuda_tensor_cast<double>(mQuantizer->getQuantizedWeights(k)));
        }
        else {
            input = cuda_device_tensor_cast<double>(mInputs[k]);
            synapses = cuda_device_tensor_cast<double>(mSynapses[k]);
        }

        // Computes mOutputs = alpha*mSynapses'*mInputs + beta*mOutputs
        CHECK_CUBLAS_STATUS(cublasDgemm(
            CudaContext::cublasHandle(),
            CUBLAS_OP_T, // mSynapses'
            CUBLAS_OP_N, // mInputs
            mOutputs.dimZ(), // nb rows in mSynapses' and mOutputs
            mInputs.dimB(), // nb cols in mInputs and mOutputs
            inputSize, // nb cols in mSynapses' and nb rows in mInputs
            &alpha,
            synapses->getDevicePtr(),
            inputSize,
            input->getDevicePtr(),
            inputSize,
            &beta,
            mOutputs.getDevicePtr(),
            mOutputs.dimZ()));
    }

    if (!mNoBias) {
        // Computes mOutputs = alpha*mBias*mOnesVector + alpha*mOutputs
        CHECK_CUBLAS_STATUS(cublasDgemm(CudaContext::cublasHandle(),
                                        CUBLAS_OP_N,
                                        CUBLAS_OP_N,
                                        mOutputs.dimZ(),
                                        mInputs.dimB(),
                                        1,
                                        &alpha,
                                        mBias.getDevicePtr(),
                                        mOutputs.dimZ(),
                                        mOnesVector,
                                        1,
                                        &alpha,
                                        mOutputs.getDevicePtr(),
                                        mOutputs.dimZ()));
    }

    Cell_Frame_CUDA<double>::propagate(inference);
    mDiffInputs.clearValid();
}

template <>
void N2D2::FcCell_Frame_CUDA<half_float::half>::backPropagate()
{
    if (!mDiffInputs.isValid())
        return;

    Cell_Frame_CUDA<half_float::half>::backPropagate();

    //  1   <-->    batch   <-->    mInputs.b()

    const half_float::half alpha(1.0f);

    for (unsigned int k = 0, size = mInputs.size(); k < size; ++k) {
        const unsigned int inputSize = mInputs[k].dimX() * mInputs[k].dimY()
                                       * mInputs[k].dimZ();
        const half_float::half beta((mWeightsSolvers[k]->isNewIteration())
                                    ? 0.0f : 1.0f);

        std::shared_ptr<CudaDeviceTensor<half_float::half> > input
            = cuda_device_tensor_cast_nocopy<half_float::half>(mInputs[k]);

        // mDiffSynapses.getDevicePtr() = mInputs.getDevicePtr *
        // mDiffInputs.getDevicePtr*
        CHECK_CUBLAS_STATUS(cublasHgemm(
            CudaContext::cublasHandle(),
            CUBLAS_OP_N,
            CUBLAS_OP_T,
            inputSize,
            mOutputs.dimZ(),
            mInputs.dimB(),
            reinterpret_cast<const __half*>(&alpha),
            reinterpret_cast<const __half*>(input->getDevicePtr()),
            inputSize,
            reinterpret_cast<const __half*>(mDiffInputs.getDevicePtr()),
            mOutputs.dimZ(),
            reinterpret_cast<const __half*>(&beta),
            reinterpret_cast<__half*>(mDiffSynapses[k].getDevicePtr()),
            inputSize));

        mDiffSynapses[k].setValid();
    }

    if (!mNoBias) {
        const half_float::half beta((mBiasSolver->isNewIteration())
                                    ? 0.0f : 1.0f);

        // mDiffBias.getDevicePtr() = mDiffInputs.getDevicePtr * mOnesVector
        // Using cublasHgemm() because there is no cublasHgemv() yet
        CHECK_CUBLAS_STATUS(cublasHgemm(
            CudaContext::cublasHandle(),
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            mOutputs.dimZ(),
            1,
            mInputs.dimB(),
            reinterpret_cast<const __half*>(&alpha),
            reinterpret_cast<const __half*>(mDiffInputs.getDevicePtr()),
            mOutputs.dimZ(),
            reinterpret_cast<const __half*>(mOnesVector),
            1,
            reinterpret_cast<const __half*>(&beta),
            reinterpret_cast<__half*>(mDiffBias.getDevicePtr()),
            1));

        mDiffBias.setValid();
    }

    if (!mDiffOutputs.empty() && mBackPropagate) {
        for (unsigned int k = 0, size = mInputs.size(); k < size; ++k) {
            const half_float::half betaData((mDiffOutputs[k].isValid())
                                            ? 1.0f : 0.0f);
            const unsigned int diffOutputSize = mDiffOutputs[k].dimX()
                                                * mDiffOutputs[k].dimY()
                                                * mDiffOutputs[k].dimZ();

            std::shared_ptr<CudaDeviceTensor<half_float::half> > diffOutput
                = (mDiffOutputs[k].isValid())
                    ? cuda_device_tensor_cast<half_float::half>(mDiffOutputs[k])
                    : cuda_device_tensor_cast_nocopy<half_float::half>(mDiffOutputs[k]);

            // mDiffOutputs.getDevicePtr = mSynapses.getDevicePtr() *
            // mDiffInputs.getDevicePtr
            CHECK_CUBLAS_STATUS(cublasHgemm(
                CudaContext::cublasHandle(),
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                diffOutputSize,
                mInputs.dimB(),
                mOutputs.dimZ(),
                reinterpret_cast<const __half*>(&alpha),
                reinterpret_cast<const __half*>(mSynapses[k].getDevicePtr()),
                diffOutputSize,
                reinterpret_cast<const __half*>(mDiffInputs.getDevicePtr()),
                mOutputs.dimZ(),
                reinterpret_cast<const __half*>(&betaData),
                reinterpret_cast<__half*>(diffOutput->getDevicePtr()),
                diffOutputSize));

            mDiffOutputs[k].deviceTensor() = *diffOutput;
            mDiffOutputs[k].setValid();
        }

        mDiffOutputs.synchronizeDToHBased();
    } // Otherwise it is the first layer, no need to calculate
}

template <>
void N2D2::FcCell_Frame_CUDA<float>::backPropagate()
{
    if (!mDiffInputs.isValid())
        return;

    Cell_Frame_CUDA<float>::backPropagate();

    //  1   <-->    batch   <-->    mInputs.b()

    const float alpha = 1.0f;

    for (unsigned int k = 0, size = mInputs.size(); k < size; ++k) {
        const unsigned int inputSize = mInputs[k].dimX() * mInputs[k].dimY()
                                       * mInputs[k].dimZ();
        const float beta = (mWeightsSolvers[k]->isNewIteration()) ? 0.0f : 1.0f;

        std::shared_ptr<CudaDeviceTensor<float> > inputs;
        std::shared_ptr<CudaDeviceTensor<float> > diffSynapses;

        if (mQuantizer) {
            inputs = cuda_device_tensor_cast<float>
                (cuda_tensor_cast<float>(mQuantizer->getQuantizedActivations(k)));
            diffSynapses = cuda_device_tensor_cast<float>
                (cuda_tensor_cast<float>(mQuantizer->getDiffQuantizedWeights(k)));
        }
        else {
            inputs = cuda_device_tensor_cast<float>(mInputs[k]);
            diffSynapses = cuda_device_tensor_cast<float>(mDiffSynapses[k]); 
        }


        // mDiffSynapses.getDevicePtr() = mInputs.getDevicePtr *
        // mDiffInputs.getDevicePtr*
        CHECK_CUBLAS_STATUS(cublasSgemm(CudaContext::cublasHandle(),
                                        CUBLAS_OP_N,
                                        CUBLAS_OP_T,
                                        inputSize,
                                        mOutputs.dimZ(),
                                        mInputs.dimB(),
                                        &alpha,
                                        inputs->getDevicePtr(),
                                        inputSize,
                                        mDiffInputs.getDevicePtr(),
                                        mOutputs.dimZ(),
                                        &beta,
                                        diffSynapses->getDevicePtr(),
                                        inputSize));
       
        mDiffSynapses[k].setValid();

    }

    if (!mNoBias) {
        const float beta = (mBiasSolver->isNewIteration()) ? 0.0f : 1.0f;

        std::shared_ptr<CudaDeviceTensor<float> > diffBiases;

        if (mQuantizer) {
            diffBiases = cuda_device_tensor_cast<float>
                (cuda_tensor_cast<float>(mQuantizer->getDiffQuantizedBiases()));
        }
        else {
            diffBiases = cuda_device_tensor_cast<float>(mDiffBias);
        }

        // mDiffBias.getDevicePtr() = mDiffInputs.getDevicePtr * mOnesVector
        CHECK_CUBLAS_STATUS(cublasSgemv(CudaContext::cublasHandle(),
                                        CUBLAS_OP_N,
                                        mOutputs.dimZ(),
                                        mInputs.dimB(),
                                        &alpha,
                                        mDiffInputs.getDevicePtr(),
                                        mOutputs.dimZ(),
                                        mOnesVector,
                                        1,
                                        &beta,
                                        diffBiases->getDevicePtr(),
                                        1));

        mDiffBias.setValid();

    }

    if (!mDiffOutputs.empty() && mBackPropagate) {
        for (unsigned int k = 0, size = mInputs.size(); k < size; ++k) {
            const float betaData = (mDiffOutputs[k].isValid()) ? 1.0f : 0.0f;
            const unsigned int diffOutputSize = mDiffOutputs[k].dimX()
                                                * mDiffOutputs[k].dimY()
                                                * mDiffOutputs[k].dimZ();

            std::shared_ptr<CudaDeviceTensor<float> > diffOutputs;
            std::shared_ptr<CudaDeviceTensor<float> > synapses;

            if (mQuantizer) {
                synapses = cuda_device_tensor_cast<float>
                    (cuda_tensor_cast<float>(mQuantizer->getQuantizedWeights(k)));
                diffOutputs = (mDiffOutputs[k].isValid())
                    ? cuda_device_tensor_cast<float>
                        (cuda_tensor_cast<float>(mQuantizer->getDiffQuantizedActivations(k)))
                    : cuda_device_tensor_cast_nocopy<float>
                        (cuda_tensor_cast<float>(mQuantizer->getDiffQuantizedActivations(k)));
            }
            else {
                synapses = cuda_device_tensor_cast<float>(mSynapses[k]);
                diffOutputs = (mDiffOutputs[k].isValid())
                    ? cuda_device_tensor_cast<float>(mDiffOutputs[k])
                    : cuda_device_tensor_cast_nocopy<float>(mDiffOutputs[k]);
            }

            // mDiffOutputs.getDevicePtr = mSynapses.getDevicePtr() *
            // mDiffInputs.getDevicePtr
            CHECK_CUBLAS_STATUS(cublasSgemm(CudaContext::cublasHandle(),
                                            CUBLAS_OP_N,
                                            CUBLAS_OP_N,
                                            diffOutputSize,
                                            mInputs.dimB(),
                                            mOutputs.dimZ(),
                                            &alpha,
                                            synapses->getDevicePtr(),
                                            diffOutputSize,
                                            mDiffInputs.getDevicePtr(),
                                            mOutputs.dimZ(),
                                            &betaData,
                                            diffOutputs->getDevicePtr(),
                                            diffOutputSize));
            
            //mDiffOutputs[k].deviceTensor() = *diffOutputs;
            //mDiffOutputs[k].setValid();

        }

    } // Otherwise it is the first layer, no need to calculate
    
    // Calculate full precision weights and activation gradients
    if (mQuantizer && mBackPropagate) {
        mQuantizer->back_propagate();
    }

    if (!mDiffOutputs.empty() && mBackPropagate)
    {
        std::shared_ptr<CudaDeviceTensor<float> > diffOutputs;
        for (unsigned int k = 0, size = mInputs.size(); k < size; ++k) {
            if (mQuantizer) {
                 diffOutputs = (mDiffOutputs[k].isValid())
                    ? cuda_device_tensor_cast<float>
                        (cuda_tensor_cast<float>(mQuantizer->getDiffQuantizedActivations(k)))
                    : cuda_device_tensor_cast_nocopy<float>
                        (cuda_tensor_cast<float>(mQuantizer->getDiffQuantizedActivations(k)));
            }
            else {
                diffOutputs = (mDiffOutputs[k].isValid())
                    ? cuda_device_tensor_cast<float>(mDiffOutputs[k])
                    : cuda_device_tensor_cast_nocopy<float>(mDiffOutputs[k]);
            }

            mDiffOutputs[k].deviceTensor() = *diffOutputs;
            mDiffOutputs[k].setValid();
        }
        mDiffOutputs.synchronizeDToHBased();
    }
}

template <>
void N2D2::FcCell_Frame_CUDA<double>::backPropagate()
{
    if (!mDiffInputs.isValid())
        return;

    Cell_Frame_CUDA<double>::backPropagate();

    //  1   <-->    batch   <-->    mInputs.b()

    const double alpha = 1.0;

    for (unsigned int k = 0, size = mInputs.size(); k < size; ++k) {
        const unsigned int inputSize = mInputs[k].dimX() * mInputs[k].dimY()
                                       * mInputs[k].dimZ();
        const double beta = (mWeightsSolvers[k]->isNewIteration()) ? 0.0 : 1.0;

        std::shared_ptr<CudaDeviceTensor<double> > inputs;
        std::shared_ptr<CudaDeviceTensor<double> > diffSynapses;

        if (mQuantizer) {
            inputs = cuda_device_tensor_cast<double>
                (cuda_tensor_cast<double>(mQuantizer->getQuantizedActivations(k)));
            diffSynapses = cuda_device_tensor_cast<double>
                (cuda_tensor_cast<double>(mQuantizer->getDiffQuantizedWeights(k)));
        }
        else {
            inputs = cuda_device_tensor_cast<double>(mInputs[k]);
            diffSynapses = cuda_device_tensor_cast<double>(mDiffSynapses[k]); 
        }

        // mDiffSynapses.getDevicePtr() = mInputs.getDevicePtr *
        // mDiffInputs.getDevicePtr*
        CHECK_CUBLAS_STATUS(cublasDgemm(CudaContext::cublasHandle(),
                                        CUBLAS_OP_N,
                                        CUBLAS_OP_T,
                                        inputSize,
                                        mOutputs.dimZ(),
                                        mInputs.dimB(),
                                        &alpha,
                                        inputs->getDevicePtr(),
                                        inputSize,
                                        mDiffInputs.getDevicePtr(),
                                        mOutputs.dimZ(),
                                        &beta,
                                        diffSynapses->getDevicePtr(),
                                        inputSize));

        mDiffSynapses[k].setValid();
    }

    if (!mNoBias) {
        const double beta = (mBiasSolver->isNewIteration()) ? 0.0 : 1.0;

        // mDiffBias.getDevicePtr() = mDiffInputs.getDevicePtr * mOnesVector
        CHECK_CUBLAS_STATUS(cublasDgemv(CudaContext::cublasHandle(),
                                        CUBLAS_OP_N,
                                        mOutputs.dimZ(),
                                        mInputs.dimB(),
                                        &alpha,
                                        mDiffInputs.getDevicePtr(),
                                        mOutputs.dimZ(),
                                        mOnesVector,
                                        1,
                                        &beta,
                                        mDiffBias.getDevicePtr(),
                                        1));

        mDiffBias.setValid();
    }

    if (!mDiffOutputs.empty() && mBackPropagate) {
        for (unsigned int k = 0, size = mInputs.size(); k < size; ++k) {
            const double betaData = (mDiffOutputs[k].isValid()) ? 1.0 : 0.0;
            const unsigned int diffOutputSize = mDiffOutputs[k].dimX()
                                                * mDiffOutputs[k].dimY()
                                                * mDiffOutputs[k].dimZ();

            std::shared_ptr<CudaDeviceTensor<double> > diffOutputs;
            std::shared_ptr<CudaDeviceTensor<double> > synapses;

            if (mQuantizer) {
                synapses = cuda_device_tensor_cast<double>
                    (cuda_tensor_cast<double>(mQuantizer->getQuantizedWeights(k)));
                diffOutputs = (mDiffOutputs[k].isValid())
                    ? cuda_device_tensor_cast<double>
                        (cuda_tensor_cast<double>(mQuantizer->getDiffQuantizedActivations(k)))
                    : cuda_device_tensor_cast_nocopy<double>
                        (cuda_tensor_cast<double>(mQuantizer->getDiffQuantizedActivations(k)));
            }
            else {
                synapses = cuda_device_tensor_cast<double>(mSynapses[k]);
                diffOutputs = (mDiffOutputs[k].isValid())
                    ? cuda_device_tensor_cast<double>(mDiffOutputs[k])
                    : cuda_device_tensor_cast_nocopy<double>(mDiffOutputs[k]);
            }

            // mDiffOutputs.getDevicePtr = mSynapses.getDevicePtr() *
            // mDiffInputs.getDevicePtr
            CHECK_CUBLAS_STATUS(cublasDgemm(CudaContext::cublasHandle(),
                                            CUBLAS_OP_N,
                                            CUBLAS_OP_N,
                                            diffOutputSize,
                                            mInputs.dimB(),
                                            mOutputs.dimZ(),
                                            &alpha,
                                            synapses->getDevicePtr(),
                                            diffOutputSize,
                                            mDiffInputs.getDevicePtr(),
                                            mOutputs.dimZ(),
                                            &betaData,
                                            diffOutputs->getDevicePtr(),
                                            diffOutputSize));
        }
    } // Otherwise it is the first layer, no need to calculate

    if (mQuantizer && mBackPropagate) {
        mQuantizer->back_propagate();
    }

    if (!mDiffOutputs.empty() && mBackPropagate)
    {
        std::shared_ptr<CudaDeviceTensor<double> > diffOutputs;
        for (unsigned int k = 0, size = mInputs.size(); k < size; ++k) {
            if (mQuantizer) {
                 diffOutputs = (mDiffOutputs[k].isValid())
                    ? cuda_device_tensor_cast<double>
                        (cuda_tensor_cast<double>(mQuantizer->getDiffQuantizedActivations(k)))
                    : cuda_device_tensor_cast_nocopy<double>
                        (cuda_tensor_cast<double>(mQuantizer->getDiffQuantizedActivations(k)));
            }
            else {
                diffOutputs = (mDiffOutputs[k].isValid())
                    ? cuda_device_tensor_cast<double>(mDiffOutputs[k])
                    : cuda_device_tensor_cast_nocopy<double>(mDiffOutputs[k]);
            }

            mDiffOutputs[k].deviceTensor() = *diffOutputs;
            mDiffOutputs[k].setValid();
        }
        mDiffOutputs.synchronizeDToHBased();
    }
}
}

template <class T>
void N2D2::FcCell_Frame_CUDA<T>::update()
{
    for (unsigned int k = 0, size = mSynapses.size(); k < size; ++k) {
        if (mDiffSynapses[k].isValid() && !mQuantizer) {
            mWeightsSolvers[k]
                ->update(mSynapses[k], mDiffSynapses[k], mInputs.dimB());
        }
        else if (mDiffSynapses[k].isValid() && mQuantizer) {
            mWeightsSolvers[k]->update(
                mSynapses[k], mQuantizer->getDiffFullPrecisionWeights(k), mInputs.dimB());
        }
    }

    if (mActivation) {
        double minVal, maxVal;
        //TODO: implement common scaling for all the solvers in the cell
        std::tie(minVal, maxVal) = mWeightsSolvers.back()->getQuantizedRange();
        mActivation->setPreQuantizeScaling(maxVal);
    }

    if (!mNoBias && mDiffBias.isValid()){
        if(!mQuantizer) {
            mBiasSolver->update(mBias, mDiffBias, mInputs.dimB());
        }
        else {
            mBiasSolver->update(mBias, mQuantizer->getDiffFullPrecisionBiases(), mInputs.dimB());
        }
    }

    if(mQuantizer){
        if (strcmp(mQuantizer->getType(),"LSQ") == 0) std::cout << getName() << ": ";
        mQuantizer->update();
    }
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
    mSynapses.synchronizeDToH();
    mBias.synchronizeDToH();

    mSynchronized = true;
    FcCell::logFreeParameters(fileName, output);
    mSynchronized = false;
}

template <class T>
void N2D2::FcCell_Frame_CUDA<T>::logFreeParameters(const std::string
                                                & dirName) const
{
    mSynapses.synchronizeDToH();
    mBias.synchronizeDToH();

    mSynchronized = true;
    FcCell::logFreeParameters(dirName);
    mSynchronized = false;
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
    mSynapses.synchronizeDToH();
    mBias.synchronizeDToH();

    mSynchronized = true;
    FcCell::exportFreeParameters(fileName);
    mSynchronized = false;
    if(mQuantizer) {
        mQuantizer->exportFreeParameters(fileName);
    }

}

template <class T>
void N2D2::FcCell_Frame_CUDA<T>::importFreeParameters(const std::string& fileName,
                                                   bool ignoreNotExists)
{
    mSynchronized = true;
    FcCell::importFreeParameters(fileName, ignoreNotExists);
    mSynchronized = false;

    mSynapses.synchronizeHToD();
    mBias.synchronizeHToD();
    if(mQuantizer) {
        mQuantizer->importFreeParameters(fileName, ignoreNotExists);
    }
}

template <class T>
void N2D2::FcCell_Frame_CUDA<T>::logFreeParametersDistrib(const std::string
                                                       & fileName) const
{
    mSynapses.synchronizeDToH();
    mBias.synchronizeDToH();

    mSynchronized = true;
    FcCell::logFreeParametersDistrib(fileName);
    mSynchronized = false;
}

template <class T>
void N2D2::FcCell_Frame_CUDA<T>::discretizeFreeParameters(unsigned int nbLevels)
{
    mSynapses.synchronizeDToH();
    mBias.synchronizeDToH();

    mSynchronized = true;
    FcCell::discretizeFreeParameters(nbLevels);
    mSynchronized = false;

    mSynapses.synchronizeHToD();
    mBias.synchronizeHToD();
}

template <class T>
std::pair<N2D2::Float_T, N2D2::Float_T>
N2D2::FcCell_Frame_CUDA<T>::getFreeParametersRange(bool withAdditiveParameters) const
{
    mSynapses.synchronizeDToH();
    mBias.synchronizeDToH();

    mSynchronized = true;
    const std::pair<Float_T, Float_T> range = FcCell::getFreeParametersRange(withAdditiveParameters);
    mSynchronized = false;

    return range;
}

template <class T>
std::pair<N2D2::Float_T, N2D2::Float_T>
N2D2::FcCell_Frame_CUDA<T>::getFreeParametersRangePerOutput(std::size_t output, 
                                                            bool withAdditiveParameters) const
{
    mSynapses.synchronizeDToH();
    mBias.synchronizeDToH();

    mSynchronized = true;
    const std::pair<Float_T, Float_T> range = FcCell::getFreeParametersRangePerOutput(output, withAdditiveParameters);
    mSynchronized = false;

    return range;
}

template <class T>
std::pair<N2D2::Float_T, N2D2::Float_T>
N2D2::FcCell_Frame_CUDA<T>::getFreeParametersRangePerChannel(std::size_t channel) const
{
    mSynapses.synchronizeDToH();
    mBias.synchronizeDToH();

    mSynchronized = true;
    const std::pair<Float_T, Float_T> range = FcCell::getFreeParametersRangePerChannel(channel);
    mSynchronized = false;

    return range;
}

template <class T>
void N2D2::FcCell_Frame_CUDA<T>::processFreeParameters(std::function<Float_T(Float_T)> func,
                                                       FreeParametersType type)
{
    mSynapses.synchronizeDToH();
    mBias.synchronizeDToH();

    mSynchronized = true;
    FcCell::processFreeParameters(func, type);
    mSynchronized = false;

    mSynapses.synchronizeHToD();
    mBias.synchronizeHToD();
}

template <class T>
void N2D2::FcCell_Frame_CUDA<T>::processFreeParametersPerOutput(std::function<Float_T(Float_T)> func,
                                                                std::size_t output,
                                                                FreeParametersType type)
{
    mSynapses.synchronizeDToH();
    mBias.synchronizeDToH();

    mSynchronized = true;
    FcCell::processFreeParametersPerOutput(func, output, type);
    mSynchronized = false;

    mSynapses.synchronizeHToD();
    mBias.synchronizeHToD();
}

template <class T>
void N2D2::FcCell_Frame_CUDA<T>::processFreeParametersPerChannel(std::function<Float_T(Float_T)> func,
                                                                std::size_t channel)
{
    mSynapses.synchronizeDToH();
    mBias.synchronizeDToH();

    mSynchronized = true;
    FcCell::processFreeParametersPerChannel(func, channel);
    mSynchronized = false;

    mSynapses.synchronizeHToD();
    mBias.synchronizeHToD();
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
