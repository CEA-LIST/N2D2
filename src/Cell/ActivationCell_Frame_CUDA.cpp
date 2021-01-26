/*
    (C) Copyright 2013 CEA LIST. All Rights Reserved.
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

#include "GradientCheck.hpp"
#include "Cell/ActivationCell_Frame_CUDA.hpp"
#include "DeepNet.hpp"
#include "third_party/half.hpp"

template <>
N2D2::Registrar<N2D2::ActivationCell>
N2D2::ActivationCell_Frame_CUDA<half_float::half>::mRegistrar("Frame_CUDA",
        N2D2::ActivationCell_Frame_CUDA<half_float::half>::create,
        N2D2::Registrar<N2D2::ActivationCell>::Type<half_float::half>());

template <>
N2D2::Registrar<N2D2::ActivationCell>
N2D2::ActivationCell_Frame_CUDA<float>::mRegistrar("Frame_CUDA",
        N2D2::ActivationCell_Frame_CUDA<float>::create,
        N2D2::Registrar<N2D2::ActivationCell>::Type<float>());

template <>
N2D2::Registrar<N2D2::ActivationCell>
N2D2::ActivationCell_Frame_CUDA<double>::mRegistrar("Frame_CUDA",
    N2D2::ActivationCell_Frame_CUDA<double>::create,
    N2D2::Registrar<N2D2::ActivationCell>::Type<double>());

template <class T>
N2D2::ActivationCell_Frame_CUDA<T>::ActivationCell_Frame_CUDA(
    const DeepNet& deepNet, 
    const std::string& name,
    unsigned int nbOutputs,
    const std::shared_ptr<Activation>& activation)
    : Cell(deepNet, name, nbOutputs),
      ActivationCell(deepNet, name, nbOutputs),
      Cell_Frame_CUDA<T>(deepNet, name, nbOutputs, activation)
{

}

template <class T>
void N2D2::ActivationCell_Frame_CUDA<T>::initialize()
{
    if (mInputs.size() > 1)
        throw std::domain_error("ActivationCell_Frame_CUDA<T>::initialize(): "
                                "inputs concatenation is not supported.");

    if (mInputs.dimZ() != mOutputs.dimZ()) {
        throw std::domain_error("ActivationCell_Frame_CUDA<T>::initialize():"
                                " the number of output channels must be equal "
                                "to the sum of inputs channels.");
    }
}

template <class T>
void N2D2::ActivationCell_Frame_CUDA<T>::propagate(bool inference)
{
    mInputs.synchronizeHBasedToD();

    const CudaTensor<T>& input = cuda_tensor_cast<T>(mInputs[0]);
    mActivation->propagate(*this, input, mOutputs, inference);

    mDiffInputs.clearValid();
}

template <class T>
void N2D2::ActivationCell_Frame_CUDA<T>::backPropagate()
{
    if (!mDiffInputs.isValid())
        return;

    if (!mDiffOutputs.empty()) {
        const CudaTensor<T>& input = cuda_tensor_cast<T>(mInputs[0]);
        CudaTensor<T> diffOutput = (mDiffOutputs[0].isValid())
            ? cuda_tensor_cast<T>(mDiffOutputs[0])
            : cuda_tensor_cast_nocopy<T>(mDiffOutputs[0]);

        mActivation->backPropagate(*this, input, mOutputs, mDiffInputs,
                                                        diffOutput);

        mDiffOutputs[0].deviceTensor() = diffOutput.deviceTensor();

        mDiffOutputs[0].setValid();
        mDiffOutputs[0].synchronizeDToHBased();
    }
}

template <class T>
void N2D2::ActivationCell_Frame_CUDA<T>::update()
{
    mActivation->update(mInputs.dimB());
}

template <class T>
void N2D2::ActivationCell_Frame_CUDA<T>::checkGradient(double epsilon, double maxError)
{
    GradientCheck<T> gc(epsilon, maxError);
    gc.initialize(mInputs,
                  mOutputs,
                  mDiffInputs,
                  std::bind(&ActivationCell_Frame_CUDA<T>::propagate, this, false),
                  std::bind(&ActivationCell_Frame_CUDA<T>::backPropagate, this));

    if (!mDiffOutputs.empty()) {
        for (unsigned int k = 0; k < mInputs.size(); ++k) {
            std::stringstream name;
            name << mName + "_mDiffOutputs[" << k << "]";

            gc.check(name.str(), mInputs[k], mDiffOutputs[k]);
        }
    } else {
        std::cout << Utils::cwarning << "Empty diff. outputs for cell " << mName
                  << ", could not check the gradient!" << Utils::cdef
                  << std::endl;
    }
}

template <class T>
N2D2::ActivationCell_Frame_CUDA<T>::~ActivationCell_Frame_CUDA()
{
    
}

namespace N2D2 {
    template class ActivationCell_Frame_CUDA<half_float::half>;
    template class ActivationCell_Frame_CUDA<float>;
    template class ActivationCell_Frame_CUDA<double>;
}

#endif
