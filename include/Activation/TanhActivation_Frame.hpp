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

#ifndef N2D2_TANHACTIVATION_FRAME_H
#define N2D2_TANHACTIVATION_FRAME_H

#include "Activation/TanhActivation.hpp"

namespace N2D2 {
template <class T> class TanhActivation_Frame : public TanhActivation {
public:
    static std::shared_ptr<TanhActivation> create()
    {
        return std::make_shared<TanhActivation_Frame<T> >();
    }

    virtual void propagate(BaseTensor& data);
    virtual void backPropagate(BaseTensor& data, BaseTensor& diffData);
    virtual ~TanhActivation_Frame() {};

private:
    static Registrar<TanhActivation> mRegistrar;
};
}

template <class T>
void N2D2::TanhActivation_Frame<T>::propagate(BaseTensor& baseData)
{
    Tensor<T>& data = dynamic_cast<Tensor<T>&>(baseData);

    if (mAlpha != 1.0) {
        const T alpha(mAlpha);

#pragma omp parallel for if (data.size() > 1024)
        for (int index = 0; index < (int)data.size(); ++index)
            data(index) = std::tanh(alpha * data(index));
    } else {
#pragma omp parallel for if (data.size() > 1024)
        for (int index = 0; index < (int)data.size(); ++index)
            data(index) = std::tanh(data(index));
    }
}

template <class T>
void N2D2::TanhActivation_Frame
    <T>::backPropagate(BaseTensor& baseData, BaseTensor& baseDiffData)
{
    Tensor<T>& data = dynamic_cast<Tensor<T>&>(baseData);
    Tensor<T>& diffData = dynamic_cast<Tensor<T>&>(baseDiffData);

    if (mAlpha != 1.0) {
        const T alpha(mAlpha);

#pragma omp parallel for if (data.size() > 1024)
        for (int index = 0; index < (int)diffData.size(); ++index)
            diffData(index) *= alpha * (1.0f - data(index) * data(index));
    } else {
#pragma omp parallel for if (data.size() > 1024)
        for (int index = 0; index < (int)diffData.size(); ++index)
            diffData(index) *= (1.0f - data(index) * data(index));
    }
}

#endif // N2D2_TANHACTIVATION_FRAME_H
