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
    void update(CudaTensor<T>& data, CudaTensor<T>& diffData,
                unsigned int batchSize);
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
    CudaTensor<T> mContinuousData;

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
void N2D2::AdamSolver_Frame_CUDA<T>::update(BaseTensor& data,
                                           BaseTensor& diffData,
                                           unsigned int batchSize)
{
    update(dynamic_cast<CudaTensor<T>&>(data),
           dynamic_cast<CudaTensor<T>&>(diffData),
           batchSize);
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
    mContinuousData.synchronizeDToH();
    mContinuousData.save(state);
}

template <class T>
void N2D2::AdamSolver_Frame_CUDA<T>::loadInternal(std::istream& state)
{
    AdamSolver::loadInternal(state);

    mMomentum1Data.load(state);
    mMomentum1Data.synchronizeHToD();
    mMomentum2Data.load(state);
    mMomentum2Data.synchronizeHToD();
    mContinuousData.load(state);
    mContinuousData.synchronizeHToD();
}

namespace N2D2 {
template <>
void AdamSolver_Frame_CUDA<half_float::half>::update(CudaTensor<half_float::half>& data,
                                         CudaTensor<half_float::half>& diffData,
                                         unsigned int batchSize);

template <>
void AdamSolver_Frame_CUDA<float>::update(CudaTensor<float>& data,
                                         CudaTensor<float>& diffData,
                                         unsigned int batchSize);

template <>
void AdamSolver_Frame_CUDA<double>::update(CudaTensor<double>& data,
                                          CudaTensor<double>& diffData,
                                          unsigned int batchSize);
}

#endif // N2D2_ADAMSOLVER_FRAME_CUDA_H
