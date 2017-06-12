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

#ifdef CUDA

#include "Solver/SGDSolver_Frame_CUDA.hpp"

template <>
N2D2::Registrar<N2D2::SGDSolver<N2D2::Float_T> > N2D2::SGDSolver_Frame_CUDA
    <N2D2::Float_T>::mRegistrar(N2D2::SGDSolver_Frame_CUDA
                                <N2D2::Float_T>::create,
                                "Frame_CUDA",
                                "Transcode_CUDA",
                                NULL);

namespace N2D2 {
template <>
void SGDSolver_Frame_CUDA<float>::update(Tensor4d<float>* data,
                                         Tensor4d<float>* diffData,
                                         unsigned int batchSize)
{
    CudaTensor4d<float>* cudaData = static_cast<CudaTensor4d<float>*>(data);
    CudaTensor4d<float>* cudaDiffData = static_cast
        <CudaTensor4d<float>*>(diffData);

    if (mQuantizationLevels > 0 && mContinuousData.empty()) {
        mContinuousData.resize(
            data->dimX(), data->dimY(), data->dimZ(), data->dimB());
        CHECK_CUDA_STATUS(cudaMemcpy(mContinuousData.getDevicePtr(),
                                     cudaData->getDevicePtr(),
                                     cudaData->size() * sizeof(float),
                                     cudaMemcpyDeviceToDevice));
    }

    CudaTensor4d<float>* cudaContinuousData
        = (mQuantizationLevels > 0) ? &mContinuousData : cudaData;

    float rate = mLearningRate;
    const unsigned int itFactor = mNbIterations / mLearningRateStepSize;

    if (mLearningRatePolicy == SGDSolver<float>::StepDecay) {
        rate *= std::pow(mLearningRateDecay, (double)itFactor);

        if (mNbIterations > 0 && (mNbIterations - batchSize)
                                 / mLearningRateStepSize != itFactor)
            std::cout << "Learning rate after " << mNbIterations
                      << " iteration(s): " << rate << std::endl;
    } else if (mLearningRatePolicy == SGDSolver<float>::ExponentialDecay)
        rate = mLearningRate * std::exp(-mLearningRateDecay * itFactor);
    else if (mLearningRatePolicy == SGDSolver<float>::InvTDecay)
        rate = mLearningRate / (1.0 + mLearningRateDecay * itFactor);
    else if (mLearningRatePolicy == SGDSolver<float>::PolyDecay) {
        float power = mPower;
        float maxIterations = mMaxIterations;
        rate = mLearningRate
               * std::pow(1.0 - (mNbIterations / maxIterations), power);
    }
    else if (mLearningRatePolicy == SGDSolver<float>::InvDecay) {
        float power = mPower;
        rate = mLearningRate
               * std::pow(1.0 + (mLearningRateDecay * mNbIterations), -power);
    }

    mNbIterations += batchSize;

    // Normalize in function of the batch size
    float rateDiff = rate / (float)batchSize;
    float momentum = mMomentum;
    float decay = mDecay;
    float unit = 1.0f;

    if (momentum == 0.0f && decay == 0.0f) {
        // data = data + diffData*rate
        CHECK_CUBLAS_STATUS(cublasSaxpy(CudaContext::cublasHandle(),
                                        diffData->size(), // size of data
                                        &rateDiff,
                                        cudaDiffData->getDevicePtr(),
                                        1,
                                        cudaContinuousData->getDevicePtr(),
                                        1));
    } else {
        if (mMomentumData.empty()) {
            mMomentumData.resize(
                data->dimX(), data->dimY(), data->dimZ(), data->dimB());
            mMomentumData.fill(0.0);
            mMomentumData.synchronizeHToD();
        }

        // mMomentumData = mMomentumData*momentum
        CHECK_CUBLAS_STATUS(cublasSscal(CudaContext::cublasHandle(),
                                        mMomentumData.size(),
                                        &momentum,
                                        mMomentumData.getDevicePtr(),
                                        1));

        // mMomentumData = mMomentumData + diffData*rate
        CHECK_CUBLAS_STATUS(cublasSaxpy(CudaContext::cublasHandle(),
                                        diffData->size(),
                                        &rateDiff,
                                        cudaDiffData->getDevicePtr(),
                                        1,
                                        mMomentumData.getDevicePtr(),
                                        1));

        if (decay != 0.0f) {
            float alpha = -decay * rate;
            // mMomentumData = mMomentumData - decay*rate*data
            CHECK_CUBLAS_STATUS(cublasSaxpy(CudaContext::cublasHandle(),
                                            data->size(),
                                            &alpha,
                                            cudaContinuousData->getDevicePtr(),
                                            1,
                                            mMomentumData.getDevicePtr(),
                                            1));
        }

        // data = data + mMomentumData
        CHECK_CUBLAS_STATUS(cublasSaxpy(CudaContext::cublasHandle(),
                                        mMomentumData.size(),
                                        &unit,
                                        mMomentumData.getDevicePtr(),
                                        1,
                                        cudaContinuousData->getDevicePtr(),
                                        1));
    }

    if (mClamping)
        cudaSclamp(cudaContinuousData->getDevicePtr(), data->size(), -1.0, 1.0);

    if (mQuantizationLevels > 0)
        cudaSquantize(cudaData->getDevicePtr(),
                      cudaContinuousData->getDevicePtr(),
                      data->size(),
                      mQuantizationLevels);
}

template <>
void SGDSolver_Frame_CUDA<double>::update(Tensor4d<double>* data,
                                          Tensor4d<double>* diffData,
                                          unsigned int batchSize)
{
    CudaTensor4d<double>* cudaData = static_cast<CudaTensor4d<double>*>(data);
    CudaTensor4d<double>* cudaDiffData = static_cast
        <CudaTensor4d<double>*>(diffData);

    if (mQuantizationLevels > 0 && mContinuousData.empty()) {
        mContinuousData.resize(
            data->dimX(), data->dimY(), data->dimZ(), data->dimB());
        CHECK_CUDA_STATUS(cudaMemcpy(mContinuousData.getDevicePtr(),
                                     cudaData->getDevicePtr(),
                                     cudaData->size() * sizeof(double),
                                     cudaMemcpyDeviceToDevice));
    }

    CudaTensor4d<double>* cudaContinuousData
        = (mQuantizationLevels > 0) ? &mContinuousData : cudaData;

    double rate = mLearningRate;
    const unsigned int itFactor = mNbIterations / mLearningRateStepSize;

    if (mLearningRatePolicy == SGDSolver<double>::StepDecay) {
        rate *= std::pow(mLearningRateDecay, (double)itFactor);

        if (mNbIterations > 0 && (mNbIterations - batchSize)
                                 / mLearningRateStepSize != itFactor)
            std::cout << "Learning rate after " << mNbIterations
                      << " iteration(s): " << rate << std::endl;
    } else if (mLearningRatePolicy == SGDSolver<double>::ExponentialDecay)
        rate = mLearningRate * std::exp(-mLearningRateDecay * itFactor);
    else if (mLearningRatePolicy == SGDSolver<double>::InvTDecay)
        rate = mLearningRate / (1.0 + mLearningRateDecay * itFactor);
    else if (mLearningRatePolicy == SGDSolver<double>::PolyDecay) {
        double power = mPower;
        double maxIterations = mMaxIterations;
        rate = mLearningRate
               * std::pow(1.0 - (mNbIterations / maxIterations), power);
    }
    else if (mLearningRatePolicy == SGDSolver<double>::InvDecay) {
        double power = mPower;
        rate = mLearningRate
               * std::pow(1.0 + (mLearningRateDecay * mNbIterations), -power);
    }

    mNbIterations += batchSize;

    // Normalize in function of the batch size
    double rateDiff = rate / (double)batchSize;
    double momentum = mMomentum;
    double decay = mDecay;
    double unit = 1.0;

    if (momentum == 0.0 && decay == 0.0) {
        // data = data + diffData*rate
        CHECK_CUBLAS_STATUS(cublasDaxpy(CudaContext::cublasHandle(),
                                        diffData->size(), // size of data
                                        &rateDiff,
                                        cudaDiffData->getDevicePtr(),
                                        1,
                                        cudaContinuousData->getDevicePtr(),
                                        1));
    } else {
        if (mMomentumData.empty()) {
            mMomentumData.resize(
                data->dimX(), data->dimY(), data->dimZ(), data->dimB());
            mMomentumData.fill(0.0);
            mMomentumData.synchronizeHToD();
        }

        // mMomentumData = mMomentumData*momentum
        CHECK_CUBLAS_STATUS(cublasDscal(CudaContext::cublasHandle(),
                                        mMomentumData.size(),
                                        &momentum,
                                        mMomentumData.getDevicePtr(),
                                        1));

        // mMomentumData = mMomentumData + diffData*rate
        CHECK_CUBLAS_STATUS(cublasDaxpy(CudaContext::cublasHandle(),
                                        diffData->size(),
                                        &rateDiff,
                                        cudaDiffData->getDevicePtr(),
                                        1,
                                        mMomentumData.getDevicePtr(),
                                        1));

        if (decay != 0.0) {
            double alpha = -decay * rate;
            // mMomentumData = mMomentumData - decay*rate*data
            CHECK_CUBLAS_STATUS(cublasDaxpy(CudaContext::cublasHandle(),
                                            data->size(),
                                            &alpha,
                                            cudaContinuousData->getDevicePtr(),
                                            1,
                                            mMomentumData.getDevicePtr(),
                                            1));
        }

        // data = data + mMomentumData
        CHECK_CUBLAS_STATUS(cublasDaxpy(CudaContext::cublasHandle(),
                                        mMomentumData.size(),
                                        &unit,
                                        mMomentumData.getDevicePtr(),
                                        1,
                                        cudaContinuousData->getDevicePtr(),
                                        1));
    }

    if (mClamping)
        cudaDclamp(cudaContinuousData->getDevicePtr(), data->size(), -1.0, 1.0);

    if (mQuantizationLevels > 0)
        cudaDquantize(cudaData->getDevicePtr(),
                      cudaContinuousData->getDevicePtr(),
                      data->size(),
                      mQuantizationLevels);
}

template <>
void SGDSolver_Frame_CUDA
    <float>::exportFreeParameters(const std::string& fileName) const
{
    float momentum = mMomentum;

    if (momentum != 0.0 && mMomentumData.size() > 0) {
        std::ofstream syn(fileName);

        if (!syn.good())
            throw std::runtime_error("Could not create synaptic file : "
                                     + fileName);

        mMomentumData.synchronizeDToH();

        for (std::vector<float>::const_iterator it = mMomentumData.begin();
             it != mMomentumData.end();
             ++it) {
            syn << (*it) << " ";
            syn << "\n";
        }

        if (!syn.good())
            throw std::runtime_error("Error writing synaptic file: "
                                     + fileName);
    }
}

template <>
void SGDSolver_Frame_CUDA
    <double>::exportFreeParameters(const std::string& fileName) const
{
    double momentum = mMomentum;

    if (momentum != 0.0) {
        std::ofstream syn(fileName.c_str());

        if (!syn.good())
            throw std::runtime_error("Could not create synaptic file : "
                                     + fileName);

        mMomentumData.synchronizeDToH();

        for (std::vector<double>::const_iterator it = mMomentumData.begin();
             it != mMomentumData.end();
             ++it)
            syn << (*it) << " ";

        if (!syn.good())
            throw std::runtime_error("Error writing synaptic file: "
                                     + fileName);
    }
}
/*
    template <>
    void SGDSolver_Frame_CUDA<float>::importFreeParameters(const std::string&
   fileName, bool ignoreNotExists) {

    }

    template <>
    void SGDSolver_Frame_CUDA<double>::importFreeParameters(const std::string&
   fileName, bool ignoreNotExists) {

    }
    */
}

#endif
