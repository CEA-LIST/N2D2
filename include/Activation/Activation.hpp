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

#ifndef N2D2_ACTIVATION_H
#define N2D2_ACTIVATION_H

#include "Environment.hpp"
#include "containers/Tensor4d.hpp"
#include "utils/Parameterizable.hpp"
#include "utils/Utils.hpp"

namespace N2D2 {
template <class T> class Activation : public Parameterizable {
public:
    Activation();
    virtual const char* getType() const = 0;
    virtual void propagate(Tensor4d<T>* data) = 0;
    virtual void backPropagate(Tensor4d<T>* data, Tensor4d<T>* diffData) = 0;
    virtual ~Activation() {};

protected:
    /// Shifting
    Parameter<int> mShifting;
};
}

template <class T>
N2D2::Activation<T>::Activation()
    : mShifting(this, "Shifting", 0)
{
    // ctor
}

#endif // N2D2_ACTIVATION_H
