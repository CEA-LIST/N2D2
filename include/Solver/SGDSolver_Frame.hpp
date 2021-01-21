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

#ifndef N2D2_SGDSOLVER_FRAME_H
#define N2D2_SGDSOLVER_FRAME_H

#include "Solver/SGDSolver.hpp"
#include "Solver/SGDSolver_Kernels.hpp"
#include "utils/Registrar.hpp"

namespace N2D2 {
template <class T> class SGDSolver_Frame : public SGDSolver {
public:
    static std::shared_ptr<SGDSolver> create()
    {
        return std::make_shared<SGDSolver_Frame<T> >();
    }

    SGDSolver_Frame();
    SGDSolver_Frame(const SGDSolver_Frame<T>& solver);
    void update(BaseTensor& data, BaseTensor& diffData, unsigned int batchSize);
    std::shared_ptr<SGDSolver_Frame<T> > clone() const
    {
        return std::shared_ptr<SGDSolver_Frame<T> >(doClone());
    }
    virtual ~SGDSolver_Frame() {};

protected:
    void saveInternal(std::ostream& state, std::ostream& log) const;
    void loadInternal(std::istream& state);

    Tensor<T> mMomentumData;
    Tensor<T> mContinuousData;

private:
    virtual SGDSolver_Frame<T>* doClone() const
    {
        return new SGDSolver_Frame<T>(*this);
    }

    static Registrar<SGDSolver> mRegistrar;
};
}

template <class T>
N2D2::SGDSolver_Frame<T>::SGDSolver_Frame()
    : SGDSolver()
{
    // ctor
}

template <class T>
N2D2::SGDSolver_Frame<T>::SGDSolver_Frame(const SGDSolver_Frame<T>& solver)
    : SGDSolver(solver)
{
    // copy-ctor
}

template <class T>
void N2D2::SGDSolver_Frame<T>::update(BaseTensor& baseData,
                                      BaseTensor& baseDiffData,
                                      unsigned int batchSize)
{
    Tensor<T>& data = dynamic_cast<Tensor<T>&>(baseData);
    Tensor<T>& diffData = dynamic_cast<Tensor<T>&>(baseDiffData);

    const T rate(SGDSolver::getLearningRate(batchSize));

    if (rate == 0.0)
        return;

    // Normalize in function of the iteration size
    const T rateDiff(rate / (batchSize * (T)mIterationSize));

    T clampMin, clampMax;
    std::tie(clampMin, clampMax) = getClamping<T>();

    if (mMomentum == 0.0 && mDecay == 0.0) {
        // if outside the loop for better performance
        // Clamping
        if (clampMin != std::numeric_limits<T>::lowest()
            || clampMax != std::numeric_limits<T>::max())
        {
            for (int index = 0; index < (int)data.size(); ++index) {
                data(index) = Utils::clamp<T>(
                    data(index)
                        + rateDiff * diffData(index), clampMin, clampMax);
            }
        }
        else {
            //#pragma omp parallel for
            for (int index = 0; index < (int)data.size(); ++index)
                data(index) += rateDiff * diffData(index);
        }
    } else {
        const T momentum(mMomentum);

        if (mMomentumData.empty())
            mMomentumData.resize(data.dims(), T(0.0));

#pragma omp parallel for if (mMomentumData.size() > 1024)
        for (int index = 0; index < (int)mMomentumData.size(); ++index) {
            // mMomentumData = mMomentumData*momentum
            mMomentumData(index) *= momentum;

            // mMomentumData = mMomentumData + diffData*mWeightsLearningRate
            mMomentumData(index) += rateDiff * diffData(index);

            if (mDecay != 0.0) {
                const T decay(mDecay);
                const T alpha = -decay * rate;

                // mMomentumData = mMomentumData - decay*rate*data
                mMomentumData(index) += alpha * data(index);
            }

            // data = data + mMomentumData
            if (clampMin != std::numeric_limits<T>::lowest()
                || clampMax != std::numeric_limits<T>::max())
            {
                data(index) = Utils::clamp<T>(
                    data(index) + mMomentumData(index),
                        clampMin, clampMax);
            }
            else
                data(index) += mMomentumData(index);
        }
    }
}

template <class T>
void N2D2::SGDSolver_Frame<T>::saveInternal(std::ostream& state,
                                            std::ostream& log) const
{
    SGDSolver::saveInternal(state, log);

    mMomentumData.save(state);
    mContinuousData.save(state);
}

template <class T>
void N2D2::SGDSolver_Frame<T>::loadInternal(std::istream& state)
{
    SGDSolver::loadInternal(state);

    mMomentumData.load(state);
    mContinuousData.load(state);
}

#endif // N2D2_SGDSOLVER_FRAME_H
