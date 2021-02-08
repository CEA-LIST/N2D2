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

#ifndef N2D2_SGDSOLVER_FRAME_CUDA_H
#define N2D2_SGDSOLVER_FRAME_CUDA_H

#include "Solver/SGDSolver.hpp"
#include "Solver/SGDSolver_Kernels.hpp"
#include "Solver/SGDSolver_CUDA_Kernels.hpp"

#include "CudaContext.hpp"
#include "CudaUtils.hpp"
#include "CublasUtils.hpp"
#include "containers/CudaTensor.hpp"

namespace N2D2 {
template <class T> class SGDSolver_Frame_CUDA : public SGDSolver {
public:
    static std::shared_ptr<SGDSolver> create()
    {
        return std::make_shared<SGDSolver_Frame_CUDA<T> >();
    }

    SGDSolver_Frame_CUDA();
    SGDSolver_Frame_CUDA(const SGDSolver_Frame_CUDA<T>& solver);
    void update(BaseTensor& data, BaseTensor& diffData, unsigned int batchSize);
    std::shared_ptr<SGDSolver_Frame_CUDA<T> > clone() const
    {
        return std::shared_ptr<SGDSolver_Frame_CUDA<T> >(doClone());
    }
    virtual ~SGDSolver_Frame_CUDA() {};

protected:
    void saveInternal(std::ostream& state, std::ostream& log) const;
    void loadInternal(std::istream& state);

    // inline void setMomentum(unsigned int output, unsigned int channel,
    // unsigned int sx, unsigned int sy, float value);
    CudaTensor<T> mMomentumData;

private:
    virtual SGDSolver_Frame_CUDA<T>* doClone() const
    {
        return new SGDSolver_Frame_CUDA<T>(*this);
    }

    static Registrar<SGDSolver> mRegistrar;
};
}

template <class T>
N2D2::SGDSolver_Frame_CUDA<T>::SGDSolver_Frame_CUDA()
    : SGDSolver()
{
    // ctor
}

template <class T>
N2D2::SGDSolver_Frame_CUDA<T>::SGDSolver_Frame_CUDA(
    const SGDSolver_Frame_CUDA<T>& solver)
    : SGDSolver(solver)
{
    // copy-ctor
}

template <class T>
void N2D2::SGDSolver_Frame_CUDA<T>::update(BaseTensor& baseData,
                                           BaseTensor& baseDiffData,
                                           unsigned int batchSize)
{
    CudaTensor<T>& data = dynamic_cast<CudaTensor<T>&>(baseData);
    CudaTensor<T>& diffData = dynamic_cast<CudaTensor<T>&>(baseDiffData);

    const float rate = SGDSolver::getLearningRate(batchSize, true);

    if (rate == 0.0)
        return;

    T clampMin, clampMax;
    std::tie(clampMin, clampMax) = getClamping<T>();

    // Normalize in function of the iteration size
    const T rateDiff(rate / (batchSize * (float)mIterationSize));

    if (mMomentum == 0.0 && mDecay == 0.0) {
        // data = data + diffData*rate
        CHECK_CUBLAS_STATUS(cublasAxpy(CudaContext::cublasHandle(),
                                        diffData.size(), // size of data
                                        &rateDiff,
                                        diffData.getDevicePtr(),
                                        1,
                                        data.getDevicePtr(),
                                        1));
    } else {
        const T momentum(mMomentum);
        const T decay(mDecay);
        const T unit(1.0f);

        if (mMomentumData.empty())
            mMomentumData.resize(data.dims(), T(0.0));

        // mMomentumData = mMomentumData*momentum
        CHECK_CUBLAS_STATUS(cublasScal(CudaContext::cublasHandle(),
                                        mMomentumData.size(),
                                        &momentum,
                                        mMomentumData.getDevicePtr(),
                                        1));

        // mMomentumData = mMomentumData + diffData*rate
        CHECK_CUBLAS_STATUS(cublasAxpy(CudaContext::cublasHandle(),
                                        diffData.size(),
                                        &rateDiff,
                                        diffData.getDevicePtr(),
                                        1,
                                        mMomentumData.getDevicePtr(),
                                        1));

        if (decay != 0.0f) {
            const T alpha(-decay * rate);
            // mMomentumData = mMomentumData - decay*rate*data
            CHECK_CUBLAS_STATUS(cublasAxpy(CudaContext::cublasHandle(),
                                            data.size(),
                                            &alpha,
                                            data.getDevicePtr(),
                                            1,
                                            mMomentumData.getDevicePtr(),
                                            1));
        }

        // data = data + mMomentumData
        CHECK_CUBLAS_STATUS(cublasAxpy(CudaContext::cublasHandle(),
                                        mMomentumData.size(),
                                        &unit,
                                        mMomentumData.getDevicePtr(),
                                        1,
                                        data.getDevicePtr(),
                                        1));
    }

    if (clampMin != std::numeric_limits<T>::lowest()
        || clampMax != std::numeric_limits<T>::max())
    {
        cudaClamp(data.getDevicePtr(), data.size(),
                   clampMin, clampMax);
    }
}

template <class T>
void N2D2::SGDSolver_Frame_CUDA<T>::saveInternal(std::ostream& state,
                                                 std::ostream& log) const
{
    SGDSolver::saveInternal(state, log);

    mMomentumData.synchronizeDToH();
    mMomentumData.save(state);
}

template <class T>
void N2D2::SGDSolver_Frame_CUDA<T>::loadInternal(std::istream& state)
{
    SGDSolver::loadInternal(state);

    mMomentumData.load(state);
    mMomentumData.synchronizeHToD();
}

#endif // N2D2_SGDSOLVER_FRAME_CUDA_H
