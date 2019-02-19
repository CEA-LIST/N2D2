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

#include "Solver/AdamSolver_Frame_CUDA.hpp"
#include "third_party/half.hpp"

template <>
N2D2::Registrar<N2D2::AdamSolver>
N2D2::AdamSolver_Frame_CUDA<half_float::half>::mRegistrar(
    {"Frame_CUDA",
     "Transcode_CUDA",
     "CSpike_BP_CUDA"},
    N2D2::AdamSolver_Frame_CUDA<half_float::half>::create,
    N2D2::Registrar<N2D2::AdamSolver>::Type<half_float::half>());

template <>
N2D2::Registrar<N2D2::AdamSolver>
N2D2::AdamSolver_Frame_CUDA<float>::mRegistrar(
    {"Frame_CUDA",
     "Transcode_CUDA",
     "CSpike_BP_CUDA"},
    N2D2::AdamSolver_Frame_CUDA<float>::create,
    N2D2::Registrar<N2D2::AdamSolver>::Type<float>());

template <>
N2D2::Registrar<N2D2::AdamSolver>
N2D2::AdamSolver_Frame_CUDA<double>::mRegistrar(
    {"Frame_CUDA",
     "Transcode_CUDA",
     "CSpike_BP_CUDA"},
    N2D2::AdamSolver_Frame_CUDA<double>::create,
    N2D2::Registrar<N2D2::AdamSolver>::Type<double>());

namespace N2D2 {
template <>
void AdamSolver_Frame_CUDA<half_float::half>::update(
    CudaTensor<half_float::half>& data,
    CudaTensor<half_float::half>& diffData,
    unsigned int /*batchSize*/)
{
    ++mNbSteps;

    if (mMomentum1Data.empty())
        mMomentum1Data.resize(data.dims(), half_float::half(0.0));

    if (mMomentum2Data.empty())
        mMomentum2Data.resize(data.dims(), half_float::half(0.0));

    if (mTmpData.empty())
        mTmpData.resize(diffData.dims(), half_float::half(0.0));

    if (mQuantizationLevels > 0 && mContinuousData.empty()) {
        mContinuousData.resize(data.dims());
        CHECK_CUDA_STATUS(cudaMemcpy(mContinuousData.getDevicePtr(),
                                     data.getDevicePtr(),
                                     data.size() * sizeof(half_float::half),
                                     cudaMemcpyDeviceToDevice));
    }

    CudaTensor<half_float::half>& cudaContinuousData
        = (mQuantizationLevels > 0) ? mContinuousData : data;

    const double learningRate = (mGlobalLearningRate > 0.0)
        ? mGlobalLearningRate : mLearningRate;
    const half_float::half alpha(learningRate
        * std::sqrt(1.0 - std::pow((double)mBeta2, (double)mNbSteps))
            / (1.0 - std::pow((double)mBeta1, (double)mNbSteps)));
    const half_float::half epsilon(mEpsilon
        * std::sqrt(1.0 - std::pow((double)mBeta2, (double)mNbSteps)));

    // Update biased first moment estimate
    const half_float::half beta1(mBeta1);
    const half_float::half beta1m(1.0 - mBeta1);

    // mMomentum1Data = mBeta1 * mMomentum1Data
    cudaHscal(mMomentum1Data.size(),
                beta1,
                mMomentum1Data.getDevicePtr());

    // mMomentum1Data = mMomentum1Data + (1.0 - mBeta1) * diffData
    cudaHaxpy(diffData.size(),
              beta1m,
              diffData.getDevicePtr(),
              mMomentum1Data.getDevicePtr());

    // Update biased second raw moment estimate
    const half_float::half beta2(mBeta2);
    const half_float::half beta2m(1.0 - mBeta2);

    // mMomentum2Data = mBeta2 * mMomentum2Data
    cudaHscal(mMomentum2Data.size(),
              beta2,
              mMomentum2Data.getDevicePtr());

    // mTmpData = diffData ^ 2
    cudaHpow(diffData.size(),
               half_float::half(2.0f),
               diffData.getDevicePtr(),
               mTmpData.getDevicePtr());

    // mMomentum2Data = mMomentum2Data + (1.0 - mBeta2) * mTmpData
    cudaHaxpy(diffData.size(),
              beta2m,
              mTmpData.getDevicePtr(),
              mMomentum2Data.getDevicePtr());

    // 3. continuousData(index) += alpha * mMomentum1Data(index)
    //                           / (std::sqrt(mMomentum2Data(index)) + epsilon);
    // mTmpData = sqrt(mMomentum2Data)
    cudaHpow(mMomentum2Data.size(),
               half_float::half(0.5f),
               mMomentum2Data.getDevicePtr(),
               mTmpData.getDevicePtr());

    // mTmpData = mTmpData + epsilon
    cudaHadd(mTmpData.size(),
             epsilon,
             mTmpData.getDevicePtr(),
             mTmpData.getDevicePtr());

    // mTmpData = 1 / mTmpData
    cudaHinv(mTmpData.size(),
             mTmpData.getDevicePtr(),
             mTmpData.getDevicePtr());

    // mTmpData = mTmpData * mMomentum1Data
    cudaHmult(mTmpData.size(),
              mMomentum1Data.getDevicePtr(),
              mTmpData.getDevicePtr(),
              mTmpData.getDevicePtr());

    // continuousData = continuousData + alpha * mTmpData
    cudaHaxpy(mTmpData.size(),
              alpha,
              mTmpData.getDevicePtr(),
              cudaContinuousData.getDevicePtr());

    if (mClamping) {
        cudaHclamp(cudaContinuousData.getDevicePtr(),
                   data.size(),
                   half_float::half(-1.0f),
                   half_float::half(1.0f));
    }

    if (mQuantizationLevels > 0) {
        std::tie(mMinVal, mMaxVal)
            = cudaHminMax(cudaContinuousData.getDevicePtr(),
                          cudaContinuousData.size());

        rangeZeroAlign(mMinVal, mMaxVal,
                       mMinValQuant, mMaxValQuant, mQuantizationLevels);

        cudaHquantize(cudaContinuousData.getDevicePtr(),
                      data.getDevicePtr(),
                      data.size(),
                      half_float::half(mMinValQuant),
                      half_float::half(mMaxValQuant),
                      mQuantizationLevels);
    }
}

template <>
void AdamSolver_Frame_CUDA<float>::update(CudaTensor<float>& data,
                                         CudaTensor<float>& diffData,
                                         unsigned int /*batchSize*/)
{
    ++mNbSteps;

    if (mMomentum1Data.empty())
        mMomentum1Data.resize(data.dims(), 0.0);

    if (mMomentum2Data.empty())
        mMomentum2Data.resize(data.dims(), 0.0);

    if (mTmpData.empty())
        mTmpData.resize(diffData.dims(), 0.0);

    if (mQuantizationLevels > 0 && mContinuousData.empty()) {
        mContinuousData.resize(data.dims());
        CHECK_CUDA_STATUS(cudaMemcpy(mContinuousData.getDevicePtr(),
                                     data.getDevicePtr(),
                                     data.size() * sizeof(float),
                                     cudaMemcpyDeviceToDevice));
    }

    CudaTensor<float>& cudaContinuousData
        = (mQuantizationLevels > 0) ? mContinuousData : data;

    const double learningRate = (mGlobalLearningRate > 0.0)
        ? mGlobalLearningRate : mLearningRate;
    const float alpha = learningRate
        * std::sqrt(1.0 - std::pow((double)mBeta2, (double)mNbSteps))
            / (1.0 - std::pow((double)mBeta1, (double)mNbSteps));
    const float epsilon = mEpsilon
        * std::sqrt(1.0 - std::pow((double)mBeta2, (double)mNbSteps));

    // Update biased first moment estimate
    const float beta1 = mBeta1;
    const float beta1m = 1.0 - mBeta1;

    // mMomentum1Data = mBeta1 * mMomentum1Data
    CHECK_CUBLAS_STATUS(cublasSscal(CudaContext::cublasHandle(),
                                    mMomentum1Data.size(),
                                    &beta1,
                                    mMomentum1Data.getDevicePtr(),
                                    1));

    // mMomentum1Data = mMomentum1Data + (1.0 - mBeta1) * diffData
    CHECK_CUBLAS_STATUS(cublasSaxpy(CudaContext::cublasHandle(),
                                    diffData.size(),
                                    &beta1m,
                                    diffData.getDevicePtr(),
                                    1,
                                    mMomentum1Data.getDevicePtr(),
                                    1));

    // Update biased second raw moment estimate
    const float beta2 = mBeta2;
    const float beta2m = 1.0 - mBeta2;

    // mMomentum2Data = mBeta2 * mMomentum2Data
    CHECK_CUBLAS_STATUS(cublasSscal(CudaContext::cublasHandle(),
                                    mMomentum2Data.size(),
                                    &beta2,
                                    mMomentum2Data.getDevicePtr(),
                                    1));

    // mTmpData = diffData ^ 2
    cudaSpow(diffData.size(),
               2.0f,
               diffData.getDevicePtr(),
               mTmpData.getDevicePtr());

    // mMomentum2Data = mMomentum2Data + (1.0 - mBeta2) * mTmpData
    CHECK_CUBLAS_STATUS(cublasSaxpy(CudaContext::cublasHandle(),
                                    diffData.size(),
                                    &beta2m,
                                    mTmpData.getDevicePtr(),
                                    1,
                                    mMomentum2Data.getDevicePtr(),
                                    1));

    // 3. continuousData(index) += alpha * mMomentum1Data(index)
    //                           / (std::sqrt(mMomentum2Data(index)) + epsilon);
    // mTmpData = sqrt(mMomentum2Data)
    cudaSpow(mMomentum2Data.size(),
               0.5f,
               mMomentum2Data.getDevicePtr(),
               mTmpData.getDevicePtr());

    // mTmpData = mTmpData + epsilon
    cudaSadd(mTmpData.size(),
             epsilon,
             mTmpData.getDevicePtr(),
             mTmpData.getDevicePtr());

    // mTmpData = 1 / mTmpData
    cudaSinv(mTmpData.size(),
             mTmpData.getDevicePtr(),
             mTmpData.getDevicePtr());

    // mTmpData = mTmpData * mMomentum1Data
    cudaSmult(mTmpData.size(),
              mMomentum1Data.getDevicePtr(),
              mTmpData.getDevicePtr(),
              mTmpData.getDevicePtr());

    // continuousData = continuousData + alpha * mTmpData
    CHECK_CUBLAS_STATUS(cublasSaxpy(CudaContext::cublasHandle(),
                                    mTmpData.size(),
                                    &alpha,
                                    mTmpData.getDevicePtr(),
                                    1,
                                    cudaContinuousData.getDevicePtr(),
                                    1));

    if (mClamping)
        cudaSclamp(cudaContinuousData.getDevicePtr(), data.size(), -1.0f, 1.0f);

    if (mQuantizationLevels > 0) {
        std::tie(mMinVal, mMaxVal)
            = cudaSminMax(cudaContinuousData.getDevicePtr(),
                          cudaContinuousData.size());

        rangeZeroAlign(mMinVal, mMaxVal,
                       mMinValQuant, mMaxValQuant, mQuantizationLevels);

        cudaSquantize(cudaContinuousData.getDevicePtr(),
                      data.getDevicePtr(),
                      data.size(),
                      mMinValQuant,
                      mMaxValQuant,
                      mQuantizationLevels);
    }
}

template <>
void AdamSolver_Frame_CUDA<double>::update(CudaTensor<double>& data,
                                          CudaTensor<double>& diffData,
                                          unsigned int /*batchSize*/)
{
    ++mNbSteps;

    if (mMomentum1Data.empty())
        mMomentum1Data.resize(data.dims(), 0.0);

    if (mMomentum2Data.empty())
        mMomentum2Data.resize(data.dims(), 0.0);

    if (mTmpData.empty())
        mTmpData.resize(diffData.dims(), 0.0);

    if (mQuantizationLevels > 0 && mContinuousData.empty()) {
        mContinuousData.resize(data.dims());
        CHECK_CUDA_STATUS(cudaMemcpy(mContinuousData.getDevicePtr(),
                                     data.getDevicePtr(),
                                     data.size() * sizeof(double),
                                     cudaMemcpyDeviceToDevice));
    }

    CudaTensor<double>& cudaContinuousData
        = (mQuantizationLevels > 0) ? mContinuousData : data;

    const double learningRate = (mGlobalLearningRate > 0.0)
        ? mGlobalLearningRate : mLearningRate;
    const double alpha = learningRate
        * std::sqrt(1.0 - std::pow((double)mBeta2, (double)mNbSteps))
            / (1.0 - std::pow((double)mBeta1, (double)mNbSteps));
    const double epsilon = mEpsilon
        * std::sqrt(1.0 - std::pow((double)mBeta2, (double)mNbSteps));

    // Update biased first moment estimate
    const double beta1 = mBeta1;
    const double beta1m = 1.0 - mBeta1;

    // mMomentum1Data = mBeta1 * mMomentum1Data
    CHECK_CUBLAS_STATUS(cublasDscal(CudaContext::cublasHandle(),
                                    mMomentum1Data.size(),
                                    &beta1,
                                    mMomentum1Data.getDevicePtr(),
                                    1));

    // mMomentum1Data = mMomentum1Data + (1.0 - mBeta1) * diffData
    CHECK_CUBLAS_STATUS(cublasDaxpy(CudaContext::cublasHandle(),
                                    diffData.size(),
                                    &beta1m,
                                    diffData.getDevicePtr(),
                                    1,
                                    mMomentum1Data.getDevicePtr(),
                                    1));

    // Update biased second raw moment estimate
    const double beta2 = mBeta2;
    const double beta2m = 1.0 - mBeta2;

    // mMomentum2Data = mBeta2 * mMomentum2Data
    CHECK_CUBLAS_STATUS(cublasDscal(CudaContext::cublasHandle(),
                                    mMomentum2Data.size(),
                                    &beta2,
                                    mMomentum2Data.getDevicePtr(),
                                    1));

    // mTmpData = diffData ^ 2
    cudaDpow(diffData.size(),
               2.0,
               diffData.getDevicePtr(),
               mTmpData.getDevicePtr());

    // mMomentum2Data = mMomentum2Data + (1.0 - mBeta2) * mTmpData
    CHECK_CUBLAS_STATUS(cublasDaxpy(CudaContext::cublasHandle(),
                                    diffData.size(),
                                    &beta2m,
                                    mTmpData.getDevicePtr(),
                                    1,
                                    mMomentum2Data.getDevicePtr(),
                                    1));

    // 3. continuousData(index) += alpha * mMomentum1Data(index)
    //                           / (std::sqrt(mMomentum2Data(index)) + epsilon);
    // mTmpData = sqrt(mMomentum2Data)
    cudaDpow(mMomentum2Data.size(),
               0.5,
               mMomentum2Data.getDevicePtr(),
               mTmpData.getDevicePtr());

    // mTmpData = mTmpData + epsilon
    cudaDadd(mTmpData.size(),
             epsilon,
             mTmpData.getDevicePtr(),
             mTmpData.getDevicePtr());

    // mTmpData = 1 / mTmpData
    cudaDinv(mTmpData.size(),
             mTmpData.getDevicePtr(),
             mTmpData.getDevicePtr());

    // mTmpData = mTmpData * mMomentum1Data
    cudaDmult(mTmpData.size(),
              mMomentum1Data.getDevicePtr(),
              mTmpData.getDevicePtr(),
              mTmpData.getDevicePtr());

    // continuousData = continuousData + alpha * mTmpData
    CHECK_CUBLAS_STATUS(cublasDaxpy(CudaContext::cublasHandle(),
                                    mTmpData.size(),
                                    &alpha,
                                    mTmpData.getDevicePtr(),
                                    1,
                                    cudaContinuousData.getDevicePtr(),
                                    1));

    if (mClamping)
        cudaDclamp(cudaContinuousData.getDevicePtr(), data.size(), -1.0, 1.0);

    if (mQuantizationLevels > 0) {
        std::tie(mMinVal, mMaxVal)
            = cudaDminMax(cudaContinuousData.getDevicePtr(),
                          cudaContinuousData.size());

        rangeZeroAlign(mMinVal, mMaxVal,
                       mMinValQuant, mMaxValQuant, mQuantizationLevels);

        cudaDquantize(cudaContinuousData.getDevicePtr(),
                      data.getDevicePtr(),
                      data.size(),
                      mMinValQuant,
                      mMaxValQuant,
                      mQuantizationLevels);
    }
}
}

#endif
