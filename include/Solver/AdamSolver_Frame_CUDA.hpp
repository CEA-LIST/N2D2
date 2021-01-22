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

#ifndef N2D2_ADAMSOLVER_FRAME_CUDA_H
#define N2D2_ADAMSOLVER_FRAME_CUDA_H

#include "Solver/AdamSolver.hpp"
#include "Solver/SGDSolver_Kernels.hpp"
#include "Solver/SGDSolver_CUDA_Kernels.hpp"

#include "CudaContext.hpp"
#include "CudaUtils.hpp"
#include "CublasUtils.hpp"
#include "containers/CudaTensor.hpp"

namespace N2D2 {
template <class T> class AdamSolver_Frame_CUDA : public AdamSolver {
public:
    static std::shared_ptr<AdamSolver> create()
    {
        return std::make_shared<AdamSolver_Frame_CUDA<T> >();
    }

    AdamSolver_Frame_CUDA();
    AdamSolver_Frame_CUDA(const AdamSolver_Frame_CUDA<T>& solver);
    void update(BaseTensor& data, BaseTensor& diffData, unsigned int batchSize);
    std::shared_ptr<AdamSolver_Frame_CUDA<T> > clone() const
    {
        return std::shared_ptr<AdamSolver_Frame_CUDA<T> >(doClone());
    }
    virtual ~AdamSolver_Frame_CUDA() {};

protected:
    void saveInternal(std::ostream& state, std::ostream& log) const;
    void loadInternal(std::istream& state);

    CudaTensor<T> mMomentum1Data;
    CudaTensor<T> mMomentum2Data;

    // Temporary
    CudaTensor<T> mTmpData;

private:
    virtual AdamSolver_Frame_CUDA<T>* doClone() const
    {
        return new AdamSolver_Frame_CUDA<T>(*this);
    }

    static Registrar<AdamSolver> mRegistrar;
};
}

template <class T>
N2D2::AdamSolver_Frame_CUDA<T>::AdamSolver_Frame_CUDA()
    : AdamSolver()
{
    // ctor
}

template <class T>
N2D2::AdamSolver_Frame_CUDA<T>::AdamSolver_Frame_CUDA(
    const AdamSolver_Frame_CUDA<T>& solver)
    : AdamSolver(solver)
{
    // copy-ctor
}

template <class T>
void N2D2::AdamSolver_Frame_CUDA<T>::update(BaseTensor& baseData,
                                           BaseTensor& baseDiffData,
                                           unsigned int /*batchSize*/)
{
    CudaTensor<T>& data = dynamic_cast<CudaTensor<T>&>(baseData);
    CudaTensor<T>& diffData = dynamic_cast<CudaTensor<T>&>(baseDiffData);

    ++mNbSteps;

    if (mMomentum1Data.empty())
        mMomentum1Data.resize(data.dims(), T(0.0));

    if (mMomentum2Data.empty())
        mMomentum2Data.resize(data.dims(), T(0.0));

    if (mTmpData.empty())
        mTmpData.resize(diffData.dims(), T(0.0));

    T clampMin, clampMax;
    std::tie(clampMin, clampMax) = getClamping<T>();

    const double learningRate = (mGlobalLearningRate > 0.0)
        ? mGlobalLearningRate : mLearningRate;
    const T alpha(learningRate
        * std::sqrt(1.0 - std::pow((double)mBeta2, (double)mNbSteps))
            / (1.0 - std::pow((double)mBeta1, (double)mNbSteps)));
    const T epsilon(mEpsilon
        * std::sqrt(1.0 - std::pow((double)mBeta2, (double)mNbSteps)));

    // Update biased first moment estimate
    const T beta1(mBeta1);
    const T beta1m(1.0 - mBeta1);

    // mMomentum1Data = mBeta1 * mMomentum1Data
    CHECK_CUBLAS_STATUS(cublasScal(CudaContext::cublasHandle(),
                                    mMomentum1Data.size(),
                                    &beta1,
                                    mMomentum1Data.getDevicePtr(),
                                    1));

    // mMomentum1Data = mMomentum1Data + (1.0 - mBeta1) * diffData
    CHECK_CUBLAS_STATUS(cublasAxpy(CudaContext::cublasHandle(),
                                    diffData.size(),
                                    &beta1m,
                                    diffData.getDevicePtr(),
                                    1,
                                    mMomentum1Data.getDevicePtr(),
                                    1));

    // Update biased second raw moment estimate
    const T beta2(mBeta2);
    const T beta2m(1.0 - mBeta2);

    // mMomentum2Data = mBeta2 * mMomentum2Data
    CHECK_CUBLAS_STATUS(cublasScal(CudaContext::cublasHandle(),
                                    mMomentum2Data.size(),
                                    &beta2,
                                    mMomentum2Data.getDevicePtr(),
                                    1));

    // mTmpData = diffData ^ 2
    cudaPow(diffData.size(),
               T(2.0f),
               diffData.getDevicePtr(),
               mTmpData.getDevicePtr());

    // mMomentum2Data = mMomentum2Data + (1.0 - mBeta2) * mTmpData
    CHECK_CUBLAS_STATUS(cublasAxpy(CudaContext::cublasHandle(),
                                    diffData.size(),
                                    &beta2m,
                                    mTmpData.getDevicePtr(),
                                    1,
                                    mMomentum2Data.getDevicePtr(),
                                    1));

    // 3. continuousData(index) += alpha * mMomentum1Data(index)
    //                           / (std::sqrt(mMomentum2Data(index)) + epsilon);
    // mTmpData = sqrt(mMomentum2Data)
    cudaPow(mMomentum2Data.size(),
               T(0.5f),
               mMomentum2Data.getDevicePtr(),
               mTmpData.getDevicePtr());

    // mTmpData = mTmpData + epsilon
    cudaAdd(mTmpData.size(),
             epsilon,
             mTmpData.getDevicePtr(),
             mTmpData.getDevicePtr());

    // mTmpData = 1 / mTmpData
    cudaInv(mTmpData.size(),
             mTmpData.getDevicePtr(),
             mTmpData.getDevicePtr());

    // mTmpData = mTmpData * mMomentum1Data
    cudaMult(mTmpData.size(),
              mMomentum1Data.getDevicePtr(),
              mTmpData.getDevicePtr(),
              mTmpData.getDevicePtr());

    // continuousData = continuousData + alpha * mTmpData
    CHECK_CUBLAS_STATUS(cublasAxpy(CudaContext::cublasHandle(),
                                    mTmpData.size(),
                                    &alpha,
                                    mTmpData.getDevicePtr(),
                                    1,
                                    data.getDevicePtr(),
                                    1));

    if (clampMin != std::numeric_limits<T>::lowest()
        || clampMax != std::numeric_limits<T>::max())
    {
        cudaClamp(data.getDevicePtr(), data.size(),
                   clampMin, clampMax);
    }
}

template <class T>
void N2D2::AdamSolver_Frame_CUDA<T>::saveInternal(std::ostream& state,
                                                 std::ostream& log) const
{
    AdamSolver::saveInternal(state, log);

    mMomentum1Data.synchronizeDToH();
    mMomentum1Data.save(state);
    mMomentum2Data.synchronizeDToH();
    mMomentum2Data.save(state);
}

template <class T>
void N2D2::AdamSolver_Frame_CUDA<T>::loadInternal(std::istream& state)
{
    AdamSolver::loadInternal(state);

    mMomentum1Data.load(state);
    mMomentum1Data.synchronizeHToD();
    mMomentum2Data.load(state);
    mMomentum2Data.synchronizeHToD();
}

#endif // N2D2_ADAMSOLVER_FRAME_CUDA_H
