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

#ifndef N2D2_RECTIFIERACTIVATION_FRAME_H
#define N2D2_RECTIFIERACTIVATION_FRAME_H

#include "Activation/Activation_Kernels.hpp"
#include "Activation/RectifierActivation.hpp"
#include "Cell/Cell.hpp"
#include "containers/Tensor.hpp"
#include "Solver/SGDSolver_Kernels.hpp"

namespace N2D2 {
template <class T>
class RectifierActivation_Frame : public RectifierActivation {
public:
    static std::shared_ptr<RectifierActivation> create()
    {
        return std::make_shared<RectifierActivation_Frame<T> >();
    }

    RectifierActivation_Frame();
    virtual void propagate(const Cell& cell, BaseTensor& data, bool inference = false);
    virtual void backPropagate(const Cell& cell, BaseTensor& data, BaseTensor& diffData);
    virtual ~RectifierActivation_Frame() {};

private:
    static Registrar<RectifierActivation> mRegistrar;
};
}

template <class T>
N2D2::RectifierActivation_Frame<T>::RectifierActivation_Frame():
    RectifierActivation()
{
    //ctor
}

template <class T>
void N2D2::RectifierActivation_Frame<T>::propagate(const Cell& cell, BaseTensor& baseData,
                                                   bool inference)
{
    Tensor<T>& data = dynamic_cast<Tensor<T>&>(baseData);

    mScaling.propagate(cell, data);

    if (mClipping > 0.0 && !cell.isQuantized()) {
#pragma omp parallel for if (data.size() > 1024)
        for (int index = 0; index < (int)data.size(); ++index) {
            data(index) = (data(index) > 0)
                ? std::min<T>(data(index), (T)mClipping)
                : (T)mLeakSlope * data(index);
        }
    } else {
#pragma omp parallel for if (data.size() > 1024)
        for (int index = 0; index < (int)data.size(); ++index) {
            data(index) = (data(index) > 0)
                ? data(index)
                : (T)mLeakSlope * data(index);
        }
    }

    if (mQuantizationLevels > 0) {
        if (!inference) {
            T minVal, maxVal;
            std::tie(minVal, maxVal) = minMax(data);

            double minValMA_unused;
            rangeAveraging(T(0.0f), maxVal, minValMA_unused, mMaxValMA,
                           mNbSteps, mMovingAverage, mMA_Window, mEMA_Alpha);

            if (mLog2RoundingRate > 0.0) {
                mMaxValQuant = log2Round(mMaxValMA / mPreQuantizeScaling,
                                         mLog2RoundingRate, mLog2RoundingPower)
                                            * mPreQuantizeScaling;
            }
        }

        if (mNbSteps > mQuantizationDelay || inference) {
            quantize(data, data, T(0.0f), T(mMaxValQuant),
                     (unsigned int)mQuantizationLevels);
        }
    }
}

template <class T>
void N2D2::RectifierActivation_Frame<T>::backPropagate(const Cell& cell, 
                                                       BaseTensor& baseData, BaseTensor& baseDiffData)
{
    Tensor<T>& data = dynamic_cast<Tensor<T>&>(baseData);
    Tensor<T>& diffData = dynamic_cast<Tensor<T>&>(baseDiffData);


    if (mQuantizationLevels > 0) {
#pragma omp parallel for if (diffData.size() > 1024)
        for (int index = 0; index < (int)diffData.size(); ++index) {
            diffData(index) = Utils::clamp<T>(diffData(index),
                                              T(-1.0f), T(1.0f));
        }
    }


    if (mClipping > 0.0 && !cell.isQuantized()) {
#pragma omp parallel for if (data.size() > 1024)
        for (int index = 0; index < (int)diffData.size(); ++index) {
            diffData(index) *= (data(index) > (T)mClipping)
                                      ? 0.0f
                                      : (data(index) > 0) ? 1.0f
                                                             : (T)mLeakSlope;
        }
    } else {
#pragma omp parallel for if (data.size() > 1024)
        for (int index = 0; index < (int)diffData.size(); ++index)
            diffData(index) *= (data(index) > 0) ? 1.0f : (T)mLeakSlope;
    }
    
    mScaling.backPropagate(cell, data, diffData);
}

#endif // N2D2_RECTIFIERACTIVATION_FRAME_H
