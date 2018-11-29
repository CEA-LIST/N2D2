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

#ifndef N2D2_ADAMSOLVER_FRAME_H
#define N2D2_ADAMSOLVER_FRAME_H

#include "Solver/AdamSolver.hpp"
#include "Solver/SGDSolver_Kernels.hpp"

namespace N2D2 {
template <class T> class AdamSolver_Frame : public AdamSolver {
public:
    static std::shared_ptr<AdamSolver> create()
    {
        return std::make_shared<AdamSolver_Frame<T> >();
    }

    AdamSolver_Frame();
    AdamSolver_Frame(const AdamSolver_Frame<T>& solver);
    void update(BaseTensor& data, BaseTensor& diffData, unsigned int batchSize);
    std::shared_ptr<AdamSolver_Frame<T> > clone() const
    {
        return std::shared_ptr<AdamSolver_Frame<T> >(doClone());
    }
    virtual ~AdamSolver_Frame() {};

protected:
    void saveInternal(std::ostream& state, std::ostream& log) const;
    void loadInternal(std::istream& state);

    Tensor<T> mMomentum1Data;
    Tensor<T> mMomentum2Data;
    Tensor<T> mContinuousData;

private:
    virtual AdamSolver_Frame<T>* doClone() const
    {
        return new AdamSolver_Frame<T>(*this);
    }

    static Registrar<AdamSolver> mRegistrar;
};
}

template <class T>
N2D2::AdamSolver_Frame<T>::AdamSolver_Frame()
    : AdamSolver()
{
    // ctor
}

template <class T>
N2D2::AdamSolver_Frame<T>::AdamSolver_Frame(const AdamSolver_Frame<T>& solver)
    : AdamSolver(solver)
{
    // copy-ctor
}

template <class T>
void N2D2::AdamSolver_Frame<T>::update(BaseTensor& baseData,
                                      BaseTensor& baseDiffData,
                                      unsigned int /*batchSize*/)
{
    Tensor<T>& data = dynamic_cast<Tensor<T>&>(baseData);
    Tensor<T>& diffData = dynamic_cast<Tensor<T>&>(baseDiffData);

    ++mNbSteps;

    if (mMomentum1Data.empty())
        mMomentum1Data.resize(data.dims(), T(0.0));

    if (mMomentum2Data.empty())
        mMomentum2Data.resize(data.dims(), T(0.0));

    if (mQuantizationLevels > 0 && mContinuousData.empty()) {
        mContinuousData.resize(data.dims());
        std::copy(data.begin(), data.end(), mContinuousData.begin());
    }

    Tensor<T>& continuousData
        = (mQuantizationLevels > 0) ? mContinuousData : data;

    const double alpha = mLearningRate
        * std::sqrt(1.0 - std::pow((double)mBeta2, (double)mNbSteps))
            / (1.0 - std::pow((double)mBeta1, (double)mNbSteps));
    const double epsilon = mEpsilon
        * std::sqrt(1.0 - std::pow((double)mBeta2, (double)mNbSteps));

#pragma omp parallel for if (data.size() > 1024)
    for (int index = 0; index < (int)data.size(); ++index) {
        // Update biased first moment estimate
        mMomentum1Data(index) = mBeta1 * mMomentum1Data(index)
                                + (1.0 - mBeta1) * diffData(index);

        // Update biased second raw moment estimate
        mMomentum2Data(index) = mBeta2 * mMomentum2Data(index)
                        + (1.0 - mBeta2) * (diffData(index) * diffData(index));

        continuousData(index) += alpha * mMomentum1Data(index)
            / (std::sqrt(mMomentum2Data(index)) + epsilon);

        if (mClamping) {
            continuousData(index) = Utils::clamp<T>(continuousData(index),
                T(-1.0), T(1.0));
        }
    }

    if (mQuantizationLevels > 0) {
        std::tie(mMinVal, mMaxVal) = minMax(continuousData);

        rangeZeroAlign(mMinVal, mMaxVal,
                       mMinValQuant, mMaxValQuant, mQuantizationLevels);

        quantize(data,
                 continuousData,
                 T(mMinValQuant),
                 T(mMaxValQuant),
                 mQuantizationLevels);
    }
}

template <class T>
void N2D2::AdamSolver_Frame<T>::saveInternal(std::ostream& state,
                                            std::ostream& log) const
{
    AdamSolver::saveInternal(state, log);

    mMomentum1Data.save(state);
    mMomentum2Data.save(state);
    mContinuousData.save(state);
}

template <class T>
void N2D2::AdamSolver_Frame<T>::loadInternal(std::istream& state)
{
    AdamSolver::loadInternal(state);

    mMomentum1Data.load(state);
    mMomentum2Data.load(state);
    mContinuousData.load(state);
}

#endif // N2D2_ADAMSOLVER_FRAME_H
