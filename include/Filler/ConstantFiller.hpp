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

#ifndef N2D2_CONSTANTFILLER_H
#define N2D2_CONSTANTFILLER_H

#include "Filler.hpp"
#include "containers/Tensor.hpp"

namespace N2D2 {
template <class T> class ConstantFiller : public Filler {
public:
    ConstantFiller(T value = 0.0);
    void apply(BaseTensor& data, bool restrictPositive=false);
    const char* getType() const
    {
        return "Constant";
    };
    T getValue(){
        return mValue;
    };
    virtual ~ConstantFiller() {};

private:
    T mValue;
};
}

template <class T>
N2D2::ConstantFiller<T>::ConstantFiller(T value)
    : mValue(value)
{
    // ctor
}

template <class T> void N2D2::ConstantFiller<T>::apply(BaseTensor& baseData,
                                                       bool restrictPositive)
{
    Tensor<T>& data = dynamic_cast<Tensor<T>&>(baseData);

    for (typename Tensor<T>::iterator it = data.begin(), itEnd = data.end();
         it != itEnd;
         ++it) {
        (*it) = mValue;
        if (restrictPositive)
            (*it) = (*it) < 0 ? 0 : (*it);
    }
}

#endif // N2D2_CONSTANTFILLER_H
