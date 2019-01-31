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

#ifndef N2D2_SOFTPLUSACTIVATION_FRAME_H
#define N2D2_SOFTPLUSACTIVATION_FRAME_H

#ifndef WIN32
#include <fenv.h>
#endif

#include "Activation/SoftplusActivation.hpp"
#include "containers/Tensor.hpp"

namespace N2D2 {
template <class T>
class SoftplusActivation_Frame : public SoftplusActivation {
public:
    static std::shared_ptr<SoftplusActivation> create()
    {
        return std::make_shared<SoftplusActivation_Frame<T> >();
    }

    virtual void propagate(BaseTensor& data, bool inference = false);
    virtual void backPropagate(BaseTensor& data, BaseTensor& diffData);
    virtual ~SoftplusActivation_Frame() {};

private:
    static Registrar<SoftplusActivation> mRegistrar;
};
}

template <class T>
void N2D2::SoftplusActivation_Frame<T>::propagate(BaseTensor& baseData,
                                                  bool /*inference*/)
{
    Tensor<T>& data = dynamic_cast<Tensor<T>&>(baseData);

#pragma omp parallel for if (data.size() > 1024)
    for (int index = 0; index < (int)data.size(); ++index) {
#if !defined(WIN32) && !defined(__APPLE__) && !defined(__CYGWIN__) && !defined(_WIN32)
        // Disabling of FE_OVERFLOW must be INSIDE THE LOOP, because else it
        // only applies to the main thread when using OpenMP
        const int excepts = fegetexcept();
        fedisableexcept(FE_OVERFLOW);
#endif

        data(index) = std::log(1.0f + std::exp(data(index)));

#if !defined(WIN32) && !defined(__APPLE__) && !defined(__CYGWIN__) && !defined(_WIN32)
        feenableexcept(excepts);
#endif
    }

    if (mQuantizationLevels > 0) {
        throw std::runtime_error("SoftplusActivation_Frame::propagate: "
                                 "quantization is not yet supported.");
    }
}

template <class T>
void N2D2::SoftplusActivation_Frame
    <T>::backPropagate(BaseTensor& baseData, BaseTensor& baseDiffData)
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

#pragma omp parallel for if (data.size() > 1024)
    for (int index = 0; index < (int)diffData.size(); ++index)
        diffData(index) *= (1.0f - std::exp(-data(index)));
}

#endif // N2D2_SOFTPLUSACTIVATION_FRAME_H
