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

#ifndef N2D2_SWISHACTIVATION_FRAME_H
#define N2D2_SWISHACTIVATION_FRAME_H

#include "Activation/SwishActivation.hpp"
#include "Cell/Cell.hpp"
#include "containers/Tensor.hpp"
#include "Solver/SGDSolver_Kernels.hpp"

namespace N2D2 {
template <class T>
class SwishActivation_Frame : public SwishActivation {
public:
    static std::shared_ptr<SwishActivation> create()
    {
        return std::make_shared<SwishActivation_Frame<T> >();
    }

    SwishActivation_Frame();
    virtual void propagate(const Cell& cell, BaseTensor& data, bool inference = false);
    virtual void backPropagate(const Cell& cell, BaseTensor& data, BaseTensor& diffData);
    virtual ~SwishActivation_Frame() {};

protected:
    Tensor<T> mSigmoid;

private:
    static Registrar<SwishActivation> mRegistrar;
};
}

template <class T>
N2D2::SwishActivation_Frame<T>::SwishActivation_Frame():
    SwishActivation()
{
    //ctor
}

template <class T>
void N2D2::SwishActivation_Frame<T>::propagate(const Cell& cell, BaseTensor& baseData,
                                                   bool /*inference*/)
{
    Tensor<T>& data = dynamic_cast<Tensor<T>&>(baseData);

    mScaling.propagate(cell, data);

    mSigmoid.resize(data.dims());

#pragma omp parallel for if (data.size() > 1024)
    for (int index = 0; index < (int)data.size(); ++index) {
        const T sig(1.0f / (1.0f + std::exp(-data(index))));
        mSigmoid(index) = sig;
        data(index) = data(index) * sig;
    }
}

template <class T>
void N2D2::SwishActivation_Frame<T>::backPropagate(const Cell& cell, 
                                                       BaseTensor& baseData, BaseTensor& baseDiffData)
{
    Tensor<T>& data = dynamic_cast<Tensor<T>&>(baseData);
    Tensor<T>& diffData = dynamic_cast<Tensor<T>&>(baseDiffData);

#pragma omp parallel for if (data.size() > 1024)
    for (int index = 0; index < (int)diffData.size(); ++index) {
        diffData(index) *= mSigmoid(index)
            + data(index) * (1.0f - mSigmoid(index));
    }
    
    mScaling.backPropagate(cell, data, diffData);
}

#endif // N2D2_SWISHACTIVATION_FRAME_H
