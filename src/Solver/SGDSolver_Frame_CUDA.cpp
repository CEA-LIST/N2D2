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
#include "third_party/half.hpp"

template <>
N2D2::Registrar<N2D2::SGDSolver>
N2D2::SGDSolver_Frame_CUDA<half_float::half>::mRegistrar(
    {"Frame_CUDA",
     "Transcode_CUDA"},
    N2D2::SGDSolver_Frame_CUDA<half_float::half>::create,
    N2D2::Registrar<N2D2::SGDSolver>::Type<half_float::half>());

template <>
N2D2::Registrar<N2D2::SGDSolver>
N2D2::SGDSolver_Frame_CUDA<float>::mRegistrar(
    {"Frame_CUDA",
     "Transcode_CUDA"},
    N2D2::SGDSolver_Frame_CUDA<float>::create,
    N2D2::Registrar<N2D2::SGDSolver>::Type<float>());

template <>
N2D2::Registrar<N2D2::SGDSolver>
N2D2::SGDSolver_Frame_CUDA<double>::mRegistrar(
    {"Frame_CUDA",
     "Transcode_CUDA"},
    N2D2::SGDSolver_Frame_CUDA<double>::create,
    N2D2::Registrar<N2D2::SGDSolver>::Type<double>());

namespace N2D2 {
template <>
void SGDSolver_Frame_CUDA<half_float::half>::update(
    CudaTensor<half_float::half>& data,
    CudaTensor<half_float::half>& diffData,
    unsigned int batchSize)
{
    const float rate = SGDSolver::getLearningRate(batchSize);

    if (rate == 0.0)
        return;

    if (mQuantizationLevels > 0 && mContinuousData.empty()) {
        mContinuousData.resize(data.dims());
        CHECK_CUDA_STATUS(cudaMemcpy(mContinuousData.getDevicePtr(),
                                     data.getDevicePtr(),
                                     data.size() * sizeof(half_float::half),
                                     cudaMemcpyDeviceToDevice));
    }

    CudaTensor<half_float::half>& cudaContinuousData
        = (mQuantizationLevels > 0) ? mContinuousData : data;

    // Normalize in function of the iteration size
    const half_float::half rateDiff(rate / (batchSize * (float)mIterationSize));

    if (mQuantizationLevels > 0) {
        cudaHclamp(diffData.getDevicePtr(),
                   diffData.size(),
                   half_float::half(-1.0f),
                   half_float::half(1.0f));
    }

    if (mMomentum == 0.0 && mDecay == 0.0) {
        // data = data + diffData*rate
        cudaHaxpy(diffData.size(), // size of data
                  rateDiff,
                  diffData.getDevicePtr(),
                  cudaContinuousData.getDevicePtr());
    } else {
        const half_float::half momentum(mMomentum);
        const half_float::half decay(mDecay);
        const half_float::half unit(1.0f);

        if (mMomentumData.empty())
            mMomentumData.resize(data.dims(), half_float::half(0.0));

        // mMomentumData = mMomentumData*momentum
        cudaHscal(mMomentumData.size(),
                  momentum,
                  mMomentumData.getDevicePtr());

        // mMomentumData = mMomentumData + diffData*rate
        cudaHaxpy(diffData.size(),
                  rateDiff,
                  diffData.getDevicePtr(),
                  mMomentumData.getDevicePtr());

        if (decay != 0.0f) {
            const half_float::half alpha(-decay * rate);
            // mMomentumData = mMomentumData - decay*rate*data
            cudaHaxpy(data.size(),
                      alpha,
                      cudaContinuousData.getDevicePtr(),
                      mMomentumData.getDevicePtr());
        }

        // data = data + mMomentumData
        cudaHaxpy(mMomentumData.size(),
                  unit,
                  mMomentumData.getDevicePtr(),
                  cudaContinuousData.getDevicePtr());
    }

    if (mClamping) {
        cudaHclamp(cudaContinuousData.getDevicePtr(), data.size(),
                   half_float::half(-1.0), half_float::half(1.0));
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
void SGDSolver_Frame_CUDA<float>::update(CudaTensor<float>& data,
                                         CudaTensor<float>& diffData,
                                         unsigned int batchSize)
{
    const float rate = SGDSolver::getLearningRate(batchSize);

    if (rate == 0.0)
        return;

    if (mQuantizationLevels > 0 && mContinuousData.empty()) {
        mContinuousData.resize(data.dims());
        CHECK_CUDA_STATUS(cudaMemcpy(mContinuousData.getDevicePtr(),
                                     data.getDevicePtr(),
                                     data.size() * sizeof(float),
                                     cudaMemcpyDeviceToDevice));
    }

    CudaTensor<float>& cudaContinuousData
        = (mQuantizationLevels > 0) ? mContinuousData : data;

    // Normalize in function of the iteration size
    const float rateDiff = rate / (batchSize * (float)mIterationSize);

    if (mQuantizationLevels > 0)
        cudaSclamp(diffData.getDevicePtr(), diffData.size(), -1.0f, 1.0f);

    if (mMomentum == 0.0 && mDecay == 0.0) {
        // data = data + diffData*rate
        CHECK_CUBLAS_STATUS(cublasSaxpy(CudaContext::cublasHandle(),
                                        diffData.size(), // size of data
                                        &rateDiff,
                                        diffData.getDevicePtr(),
                                        1,
                                        cudaContinuousData.getDevicePtr(),
                                        1));
    } else {
        const float momentum = mMomentum;
        const float decay = mDecay;
        const float unit = 1.0f;

        if (mMomentumData.empty())
            mMomentumData.resize(data.dims(), 0.0f);

        // mMomentumData = mMomentumData*momentum
        CHECK_CUBLAS_STATUS(cublasSscal(CudaContext::cublasHandle(),
                                        mMomentumData.size(),
                                        &momentum,
                                        mMomentumData.getDevicePtr(),
                                        1));

        // mMomentumData = mMomentumData + diffData*rate
        CHECK_CUBLAS_STATUS(cublasSaxpy(CudaContext::cublasHandle(),
                                        diffData.size(),
                                        &rateDiff,
                                        diffData.getDevicePtr(),
                                        1,
                                        mMomentumData.getDevicePtr(),
                                        1));

        if (decay != 0.0f) {
            const float alpha = -decay * rate;
            // mMomentumData = mMomentumData - decay*rate*data
            CHECK_CUBLAS_STATUS(cublasSaxpy(CudaContext::cublasHandle(),
                                            data.size(),
                                            &alpha,
                                            cudaContinuousData.getDevicePtr(),
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
                                        cudaContinuousData.getDevicePtr(),
                                        1));
    }

    if (mClamping)
        cudaSclamp(cudaContinuousData.getDevicePtr(), data.size(), -1.0, 1.0);

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
void SGDSolver_Frame_CUDA<double>::update(CudaTensor<double>& data,
                                          CudaTensor<double>& diffData,
                                          unsigned int batchSize)
{
    const double rate = SGDSolver::getLearningRate(batchSize);

    if (rate == 0.0)
        return;

    if (mQuantizationLevels > 0 && mContinuousData.empty()) {
        mContinuousData.resize(data.dims());
        CHECK_CUDA_STATUS(cudaMemcpy(mContinuousData.getDevicePtr(),
                                     data.getDevicePtr(),
                                     data.size() * sizeof(double),
                                     cudaMemcpyDeviceToDevice));
    }

    CudaTensor<double>& cudaContinuousData
        = (mQuantizationLevels > 0) ? mContinuousData : data;

    // Normalize in function of the iteration size
    const double rateDiff = rate / (batchSize * (double)mIterationSize);

    if (mQuantizationLevels > 0)
        cudaDclamp(diffData.getDevicePtr(), diffData.size(), -1.0, 1.0);

    if (mMomentum == 0.0 && mDecay == 0.0) {
        // data = data + diffData*rate
        CHECK_CUBLAS_STATUS(cublasDaxpy(CudaContext::cublasHandle(),
                                        diffData.size(), // size of data
                                        &rateDiff,
                                        diffData.getDevicePtr(),
                                        1,
                                        cudaContinuousData.getDevicePtr(),
                                        1));
    } else {
        const double momentum = mMomentum;
        const double decay = mDecay;
        const double unit = 1.0;

        if (mMomentumData.empty())
            mMomentumData.resize(data.dims(), 0.0);

        // mMomentumData = mMomentumData*momentum
        CHECK_CUBLAS_STATUS(cublasDscal(CudaContext::cublasHandle(),
                                        mMomentumData.size(),
                                        &momentum,
                                        mMomentumData.getDevicePtr(),
                                        1));

        // mMomentumData = mMomentumData + diffData*rate
        CHECK_CUBLAS_STATUS(cublasDaxpy(CudaContext::cublasHandle(),
                                        diffData.size(),
                                        &rateDiff,
                                        diffData.getDevicePtr(),
                                        1,
                                        mMomentumData.getDevicePtr(),
                                        1));

        if (decay != 0.0) {
            const double alpha = -decay * rate;
            // mMomentumData = mMomentumData - decay*rate*data
            CHECK_CUBLAS_STATUS(cublasDaxpy(CudaContext::cublasHandle(),
                                            data.size(),
                                            &alpha,
                                            cudaContinuousData.getDevicePtr(),
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
                                        cudaContinuousData.getDevicePtr(),
                                        1));
    }

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
