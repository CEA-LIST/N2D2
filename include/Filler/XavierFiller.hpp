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

#ifndef N2D2_XAVIERFILLER_H
#define N2D2_XAVIERFILLER_H

#include "Filler.hpp"
#include "containers/Tensor.hpp"
#include "utils/Random.hpp"
#include "third_party/half.hpp"

namespace N2D2 {
template <class T> class XavierFiller : public Filler {
public:
    enum VarianceNorm {
        FanIn,
        Average,
        FanOut
    };
    enum Distribution {
        Uniform,
        Normal
    };

    XavierFiller(VarianceNorm varianceNorm = FanIn,
                 Distribution distribution = Uniform,
                 T scaling = 1.0);
    void apply(BaseTensor& data, bool restrictPositive=false);
    const char* getType() const
    {
        return "Xavier";
    };
    VarianceNorm getVarianceNorm(){
        return mVarianceNorm;
    };
    Distribution getDistribution(){
        return mDistribution;
    }; 
    T getScaling(){
        return mScaling;
    };
    virtual ~XavierFiller() {};

private:
    VarianceNorm mVarianceNorm;
    Distribution mDistribution;
    T mScaling;
};
}

namespace {
template <>
const char* const EnumStrings<N2D2::XavierFiller<half_float::half>
    ::VarianceNorm>::data[]
    = {"FanIn", "Average", "FanOut"};
template <>
const char* const EnumStrings<N2D2::XavierFiller<float>::VarianceNorm>::data[]
    = {"FanIn", "Average", "FanOut"};
template <>
const char* const EnumStrings<N2D2::XavierFiller<double>::VarianceNorm>::data[]
    = {"FanIn", "Average", "FanOut"};
template <>
const char* const EnumStrings<N2D2::XavierFiller<half_float::half>
    ::Distribution>::data[]
    = {"Uniform", "Normal"};
template <>
const char* const EnumStrings<N2D2::XavierFiller<float>::Distribution>::data[]
    = {"Uniform", "Normal"};
template <>
const char* const EnumStrings<N2D2::XavierFiller<double>::Distribution>::data[]
    = {"Uniform", "Normal"};
}

template <class T>
N2D2::XavierFiller
    <T>::XavierFiller(VarianceNorm varianceNorm,
                      Distribution distribution,
                      T scaling)
    : mVarianceNorm(varianceNorm),
      mDistribution(distribution),
      mScaling(scaling)
{
    // ctor
}

template <class T> void N2D2::XavierFiller<T>::apply(BaseTensor& baseData,
                                                     bool restrictPositive)
{
    Tensor<T>& data = dynamic_cast<Tensor<T>&>(baseData);

    const unsigned int fanIn = data.size() / data.dimB();
    const unsigned int fanOut = data.size() / data.dimZ();

    const T n((mVarianceNorm == FanIn) ? fanIn : (mVarianceNorm == Average)
                                                       ? (fanIn + fanOut) / 2.0
                                                       : fanOut);

    if (mDistribution == Uniform) {
        // Variance of uniform distribution between [a,b] is (1/12)*((b-a)^2)
        // for [-scale,scale], variance is therefore (1/3)*(scale^2)
        // in order to have a variance of 1/n, the scale is therefore:
        const T scale(std::sqrt(3.0 / n));

        for (typename Tensor<T>::iterator it = data.begin(),
                                            itEnd = data.end();
             it != itEnd; ++it)
        {
                (*it) = mScaling * Random::randUniform(-scale, scale);

                if (restrictPositive)
                    (*it) = ((*it) < 0) ? 0 : (*it);
        }
    } else {
        // randNormal() takes the std. dev., which is the square root of the
        // variance
        const T stdDev(std::sqrt(1.0 / n));

        for (typename Tensor<T>::iterator it = data.begin(),
                                            itEnd = data.end();
             it != itEnd; ++it)
        {
                (*it) = mScaling * Random::randNormal(0.0, stdDev);

                if (restrictPositive)
                    (*it) = ((*it) < 0) ? 0 : (*it);
        }
    }
}

#endif // N2D2_XAVIERFILLER_H
