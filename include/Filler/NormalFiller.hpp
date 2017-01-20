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

#ifndef N2D2_NORMALFILLER_H
#define N2D2_NORMALFILLER_H

#include "Filler.hpp"
#include "utils/Random.hpp"

namespace N2D2 {
template <class T> class NormalFiller : public Filler<T> {
public:
    NormalFiller(double mean = 0.0, double stdDev = 1.0);
    void apply(Tensor4d<T>& data);
    virtual ~NormalFiller() {};

private:
    double mMean;
    double mStdDev;
};
}

template <class T>
N2D2::NormalFiller<T>::NormalFiller(double mean, double stdDev)
    : mMean(mean), mStdDev(stdDev)
{
    // ctor
}

template <class T> void N2D2::NormalFiller<T>::apply(Tensor4d<T>& data)
{
    for (typename Tensor4d<T>::iterator it = data.begin(), itEnd = data.end();
         it != itEnd;
         ++it)
        (*it) = Random::randNormal(mMean, mStdDev);
}

#endif // N2D2_NORMALFILLER_H
