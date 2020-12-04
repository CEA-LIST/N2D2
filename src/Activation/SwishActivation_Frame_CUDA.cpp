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

#include "Activation/SwishActivation_Frame_CUDA.hpp"
#include "Cell/Cell.hpp"
#include "third_party/half.hpp"

template <>
N2D2::Registrar<N2D2::SwishActivation>
N2D2::SwishActivation_Frame_CUDA<half_float::half>::mRegistrar(
    {"Frame_CUDA",
    "Transcode_CUDA",
    "CSpike_CUDA",
    "CSpike_BP_CUDA",
    "CSpike_LIF_CUDA"},
    N2D2::SwishActivation_Frame_CUDA<half_float::half>::create,
    N2D2::Registrar<N2D2::SwishActivation>::Type<half_float::half>());

template <>
N2D2::Registrar<N2D2::SwishActivation>
N2D2::SwishActivation_Frame_CUDA<float>::mRegistrar(
    {"Frame_CUDA",
    "Transcode_CUDA",
    "CSpike_CUDA",
    "CSpike_BP_CUDA",
    "CSpike_LIF_CUDA"},
    N2D2::SwishActivation_Frame_CUDA<float>::create,
    N2D2::Registrar<N2D2::SwishActivation>::Type<float>());

template <>
N2D2::Registrar<N2D2::SwishActivation>
N2D2::SwishActivation_Frame_CUDA<double>::mRegistrar(
    { "Frame_CUDA",
    "Transcode_CUDA",
    "CSpike_CUDA",
    "CSpike_BP_CUDA",
    "CSpike_LIF_CUDA" },
    N2D2::SwishActivation_Frame_CUDA<double>::create,
    N2D2::Registrar<N2D2::SwishActivation>::Type<double>());


namespace N2D2 {

template <>
void SwishActivation_Frame_CUDA<half_float::half>::propagate(const Cell& cell, 
                                                                 CudaTensor<half_float::half>& data,
                                                                 bool inference)
{
    mScaling.propagate(cell, data);

    mSigmoid.resize(data.dims());
    cudaHSwish_propagate(
        data.getDevicePtr(),
        data.getDevicePtr(),
        mSigmoid.getDevicePtr(),
        data.size());

    if (mQuantizationLevels > 0) {
        if (!inference) {
            half_float::half minVal, maxVal;
            std::tie(minVal, maxVal) = cudaHminMax(data.getDevicePtr(),
                                                   data.size());

            double minValMA_unused;
            rangeAveraging(minVal, maxVal, minValMA_unused, mMaxValMA,
                           mNbSteps, mMovingAverage, mMA_Window, mEMA_Alpha);

            if (mLog2RoundingRate > 0.0) {
                mMaxValQuant = log2Round(mMaxValMA / mPreQuantizeScaling,
                                         mLog2RoundingRate, mLog2RoundingPower)
                                            * mPreQuantizeScaling;
            }
            else
                mMaxValQuant = mMaxValMA;
        }

        if (mNbSteps > mQuantizationDelay || inference) {
            cudaHquantize(data.getDevicePtr(),
                          data.getDevicePtr(),
                          data.size(),
                          half_float::half(0.0f),
                          half_float::half(mMaxValQuant),
                          mQuantizationLevels);
        }
    }
}

template <>
void SwishActivation_Frame_CUDA<float>::propagate(const Cell& cell, 
                                                      CudaTensor<float>& data, bool inference)
{
    mScaling.propagate(cell, data);

    mSigmoid.resize(data.dims());
    cudaSSwish_propagate(
        data.getDevicePtr(),
        data.getDevicePtr(),
        mSigmoid.getDevicePtr(),
        data.size());

    if (mQuantizationLevels > 0) {
        if (!inference) {
            float minVal, maxVal;
            std::tie(minVal, maxVal) = cudaSminMax(data.getDevicePtr(),
                                                   data.size());

            double minValMA_unused;
            rangeAveraging(minVal, maxVal, minValMA_unused, mMaxValMA,
                           mNbSteps, mMovingAverage, mMA_Window, mEMA_Alpha);

            if (mLog2RoundingRate > 0.0) {
                mMaxValQuant = log2Round(mMaxValMA / mPreQuantizeScaling,
                                         mLog2RoundingRate, mLog2RoundingPower)
                                            * mPreQuantizeScaling;
            }
            else
                mMaxValQuant = mMaxValMA;
        }

        if (mNbSteps > mQuantizationDelay || inference) {
            cudaSquantize(data.getDevicePtr(),
                          data.getDevicePtr(),
                          data.size(),
                          0.0f,
                          mMaxValQuant,
                          mQuantizationLevels);
        }
    }
}

template <>
void SwishActivation_Frame_CUDA<double>::propagate(const Cell& cell, 
                                                       CudaTensor<double>& data, bool inference)
{
    mScaling.propagate(cell, data);

    mSigmoid.resize(data.dims());
    cudaDSwish_propagate(
        data.getDevicePtr(),
        data.getDevicePtr(),
        mSigmoid.getDevicePtr(),
        data.size());

    if (mQuantizationLevels > 0) {
        if (!inference) {
            double minVal, maxVal;
            std::tie(minVal, maxVal) = cudaDminMax(data.getDevicePtr(),
                                                   data.size());

            double minValMA_unused;
            rangeAveraging(minVal, maxVal, minValMA_unused, mMaxValMA,
                           mNbSteps, mMovingAverage, mMA_Window, mEMA_Alpha);

            if (mLog2RoundingRate > 0.0) {
                mMaxValQuant = log2Round(mMaxValMA / mPreQuantizeScaling,
                                         mLog2RoundingRate, mLog2RoundingPower)
                                            * mPreQuantizeScaling;
            }
            else
                mMaxValQuant = mMaxValMA;
        }

        if (mNbSteps > mQuantizationDelay || inference) {
            cudaDquantize(data.getDevicePtr(),
                          data.getDevicePtr(),
                          data.size(),
                          0.0,
                          mMaxValQuant,
                          mQuantizationLevels);
        }
    }
}

template <>
void SwishActivation_Frame_CUDA<half_float::half>::backPropagate(const Cell& cell, 
                                                                     CudaTensor<half_float::half>& data,
                                                                     CudaTensor<half_float::half>& diffData)
{
    if (mQuantizationLevels > 0) {
        cudaHclamp(diffData.getDevicePtr(),
                   diffData.size(),
                   half_float::half(-1.0f),
                   half_float::half(1.0f));
    }

    cudaHSwish_backPropagate(
        data.getDevicePtr(),
        diffData.getDevicePtr(),
        mSigmoid.getDevicePtr(),
        data.size());

    mScaling.backPropagate(cell, data, diffData);
}

template <>
void SwishActivation_Frame_CUDA<float>::backPropagate(const Cell& cell, 
                                                          CudaTensor<float>& data,
                                                          CudaTensor<float>& diffData)
{
    if (mQuantizationLevels > 0)
        cudaSclamp(diffData.getDevicePtr(), diffData.size(), -1.0f, 1.0f);

    cudaSSwish_backPropagate(
        data.getDevicePtr(),
        diffData.getDevicePtr(),
        mSigmoid.getDevicePtr(),
        data.size());
    
    mScaling.backPropagate(cell, data, diffData);
}

template <>
void SwishActivation_Frame_CUDA<double>::backPropagate(const Cell& cell, 
                                                           CudaTensor<double>& data,
                                                           CudaTensor<double>& diffData)
{
    if (mQuantizationLevels > 0)
        cudaDclamp(diffData.getDevicePtr(), diffData.size(), -1.0, 1.0);

    cudaDSwish_backPropagate(
        data.getDevicePtr(),
        diffData.getDevicePtr(),
        mSigmoid.getDevicePtr(),
        data.size());
    
    mScaling.backPropagate(cell, data, diffData);
}
}

#endif
