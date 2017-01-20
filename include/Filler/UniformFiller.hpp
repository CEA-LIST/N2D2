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

#ifndef N2D2_UNIFORMFILLER_H
#define N2D2_UNIFORMFILLER_H

#include "Filler.hpp"
#include "utils/Random.hpp"

namespace N2D2 {
template <class T> class UniformFiller : public Filler<T> {
public:
    UniformFiller(T min = 0.0, T max = 1.0);
    void apply(Tensor4d<T>& data);
    virtual ~UniformFiller() {};

private:
    T mMin;
    T mMax;
};
}

template <class T>
N2D2::UniformFiller<T>::UniformFiller(T min, T max)
    : mMin(min), mMax(max)
{
    // ctor
}

template <class T> void N2D2::UniformFiller<T>::apply(Tensor4d<T>& data)
{
    for (typename Tensor4d<T>::iterator it = data.begin(), itEnd = data.end();
         it != itEnd;
         ++it)
        (*it) = Random::randUniform(mMin, mMax);
}

#endif // N2D2_UNIFORMFILLER_H
