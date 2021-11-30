/*
    (C) Copyright 2018 CEA LIST. All Rights Reserved.
    Contributor(s): Johannes THIELE (johannes.thiele@cea.fr)
                    Olivier BICHLER (olivier.bichler@cea.fr)

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

/// Implements the weight initialization from He et al. 2015 for ReLUs

#ifndef N2D2_HEFILLER_H
#define N2D2_HEFILLER_H

#include "Filler.hpp"
#include "containers/Tensor.hpp"
#include "utils/Random.hpp"

namespace N2D2 {
template <class T> class HeFiller : public Filler {
public:
    enum VarianceNorm {
        FanIn,
        Average,
        FanOut
    };

    HeFiller(VarianceNorm varianceNorm = FanIn, T meanNorm=0.0, T scaling = 1.0);
    void apply(BaseTensor& data, bool restrictPositive=false);
    const char* getType() const
    {
        return "He";
    };
    VarianceNorm getVarianceNorm(){
        return mVarianceNorm;
    };
    T getMeanNorm(){
        return mMeanNorm;
    }; 
    T getScaling(){
        return mScaling;
    };
    virtual ~HeFiller() {};

private:
    VarianceNorm mVarianceNorm;
    T mMeanNorm;
    T mScaling;
};
}

namespace {
template <>
const char* const EnumStrings<N2D2::HeFiller<half_float::half>
    ::VarianceNorm>::data[]
    = {"FanIn", "Average", "FanOut"};
template <>
const char* const EnumStrings<N2D2::HeFiller<float>::VarianceNorm>::data[]
    = {"FanIn", "Average", "FanOut"};
template <>
const char* const EnumStrings<N2D2::HeFiller<double>::VarianceNorm>::data[]
    = {"FanIn", "Average", "FanOut"};
}

template <class T>
N2D2::HeFiller
    <T>::HeFiller(VarianceNorm varianceNorm, T meanNorm, T scaling)
    : mVarianceNorm(varianceNorm),
    mMeanNorm(meanNorm),
    mScaling(scaling)
{
    // ctor
}

template <class T> void N2D2::HeFiller<T>::apply(BaseTensor& baseData,
                                                 bool restrictPositive)
{
    Tensor<T>& data = dynamic_cast<Tensor<T>&>(baseData);

    const unsigned int fanIn = data.size() / data.dimB();
    const unsigned int fanOut = data.size() / data.dimZ();

    const T n((mVarianceNorm == FanIn) ? fanIn : (mVarianceNorm == Average)
                                                       ? (fanIn + fanOut) / 2.0
                                                       : fanOut);
    const T stdDev(std::sqrt(2.0 / n));

    const T mean(mVarianceNorm == FanIn ? mMeanNorm/fanIn :
                (mVarianceNorm == Average) ? mMeanNorm/((fanIn + fanOut)/2.0)
                : mMeanNorm/fanOut);

    for (typename Tensor<T>::iterator it = data.begin(),
                                        itEnd = data.end();
         it != itEnd; ++it)
    {
            (*it) = mScaling * Random::randNormal(mean, stdDev);

            if (restrictPositive)
                (*it) = ((*it) < 0) ? 0 : (*it);
    }
}

#endif // N2D2_HEFILLER_H
