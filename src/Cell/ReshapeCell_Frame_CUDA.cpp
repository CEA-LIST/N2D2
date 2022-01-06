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
#include "Cell/ReshapeCell_Frame_CUDA.hpp"
#include "DeepNet.hpp"
#include "third_party/half.hpp"

template <>
N2D2::Registrar<N2D2::ReshapeCell>
N2D2::ReshapeCell_Frame_CUDA<half_float::half>::mRegistrar("Frame_CUDA",
        N2D2::ReshapeCell_Frame_CUDA<half_float::half>::create,
        N2D2::Registrar<N2D2::ReshapeCell>::Type<half_float::half>());

template <>
N2D2::Registrar<N2D2::ReshapeCell>
N2D2::ReshapeCell_Frame_CUDA<float>::mRegistrar("Frame_CUDA",
        N2D2::ReshapeCell_Frame_CUDA<float>::create,
        N2D2::Registrar<N2D2::ReshapeCell>::Type<float>());

template <>
N2D2::Registrar<N2D2::ReshapeCell>
N2D2::ReshapeCell_Frame_CUDA<double>::mRegistrar("Frame_CUDA",
    N2D2::ReshapeCell_Frame_CUDA<double>::create,
    N2D2::Registrar<N2D2::ReshapeCell>::Type<double>());

template <class T>
N2D2::ReshapeCell_Frame_CUDA<T>::ReshapeCell_Frame_CUDA(
    const DeepNet& deepNet, 
    const std::string& name,
    unsigned int nbOutputs,
    const std::vector<int>& dims)
    : Cell(deepNet, name, nbOutputs),
      ReshapeCell(deepNet, name, nbOutputs, dims),
      Cell_Frame_CUDA<T>(deepNet, name, nbOutputs)
{

}

template <class T>
void N2D2::ReshapeCell_Frame_CUDA<T>::initialize()
{
    if (mInputs.size() > 1) {
        throw std::domain_error("ReshapeCell_Frame_CUDA<T>::initialize(): "
                                "inputs concatenation is not supported.");
    }
}

template <class T>
void N2D2::ReshapeCell_Frame_CUDA<T>::initializeDataDependent()
{
    Cell_Frame_CUDA<T>::initializeDataDependent();
    initialize();
}

template <class T>
void N2D2::ReshapeCell_Frame_CUDA<T>::propagate(bool inference)
{
    mInputs.synchronizeHBasedToD();

    const std::shared_ptr<CudaDeviceTensor<T> > input
        = cuda_device_tensor_cast<T>(mInputs[0]);
    mOutputs.setDevicePtr(input->getDevicePtr());

    Cell_Frame_CUDA<T>::propagate(inference);
    mDiffInputs.clearValid();
}

template <class T>
void N2D2::ReshapeCell_Frame_CUDA<T>::backPropagate()
{
    if (!mDiffInputs.isValid())
        return;

    Cell_Frame_CUDA<T>::backPropagate();

    if (!mDiffOutputs[0].empty()) {
        mDiffInputs.synchronizeHBasedToD();

        std::shared_ptr<CudaDeviceTensor<T> > diffOutput
            = (mDiffOutputs[0].isValid())
                ? cuda_device_tensor_cast<T>(mDiffOutputs[0])
                : cuda_device_tensor_cast_nocopy<T>(mDiffOutputs[0]);

        diffOutput->setDevicePtr(mDiffInputs.getDevicePtr());

        mDiffOutputs[0].deviceTensor() = *diffOutput;
        mDiffOutputs[0].setValid();
    }
}

template <class T>
void N2D2::ReshapeCell_Frame_CUDA<T>::update()
{

}

template <class T>
void N2D2::ReshapeCell_Frame_CUDA<T>::checkGradient(double epsilon, double maxError)
{
    GradientCheck<T> gc(epsilon, maxError);
    gc.initialize(mInputs,
                  mOutputs,
                  mDiffInputs,
                  std::bind(&ReshapeCell_Frame_CUDA<T>::propagate, this, false),
                  std::bind(&ReshapeCell_Frame_CUDA<T>::backPropagate, this));

    for (unsigned int k = 0; k < mInputs.size(); ++k) {
        if (mDiffOutputs[k].empty()) {
            std::cout << Utils::cwarning << "Empty diff. outputs #" << k
                    << " for cell " << mName
                    << ", could not check the gradient!" << Utils::cdef
                    << std::endl;
            continue;
        }

        std::stringstream name;
        name << mName + "_mDiffOutputs[" << k << "]";

        gc.check(name.str(), mInputs[k], mDiffOutputs[k]);
    }
}

template <class T>
N2D2::ReshapeCell_Frame_CUDA<T>::~ReshapeCell_Frame_CUDA()
{
    
}

namespace N2D2 {
    template class ReshapeCell_Frame_CUDA<half_float::half>;
    template class ReshapeCell_Frame_CUDA<float>;
    template class ReshapeCell_Frame_CUDA<double>;
}

#endif
