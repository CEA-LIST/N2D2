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

#include "GradientCheck.hpp"
#include "Cell/ActivationCell_Frame.hpp"
#include "DeepNet.hpp"
#include "third_party/half.hpp"

template <>
N2D2::Registrar<N2D2::ActivationCell>
N2D2::ActivationCell_Frame<half_float::half>::mRegistrar("Frame",
    N2D2::ActivationCell_Frame<half_float::half>::create,
    N2D2::Registrar<N2D2::ActivationCell>::Type<half_float::half>());

template <>
N2D2::Registrar<N2D2::ActivationCell>
N2D2::ActivationCell_Frame<float>::mRegistrar("Frame",
    N2D2::ActivationCell_Frame<float>::create,
    N2D2::Registrar<N2D2::ActivationCell>::Type<float>());

template <>
N2D2::Registrar<N2D2::ActivationCell>
N2D2::ActivationCell_Frame<double>::mRegistrar("Frame",
    N2D2::ActivationCell_Frame<double>::create,
    N2D2::Registrar<N2D2::ActivationCell>::Type<double>());

template <class T>
N2D2::ActivationCell_Frame<T>::ActivationCell_Frame(const DeepNet& deepNet, const std::string& name,
                                 unsigned int nbOutputs,
                                 const std::shared_ptr
                                 <Activation>& activation)
    : Cell(deepNet, name, nbOutputs),
      ActivationCell(deepNet, name, nbOutputs),
      Cell_Frame<T>(deepNet, name, nbOutputs, activation)
{
    // ctor
}

template <class T>
void N2D2::ActivationCell_Frame<T>::initialize()
{

    if (mInputs.dimZ() != mOutputs.dimZ()) {
        throw std::domain_error("ActivationCell_Frame<T>::initialize():"
                                " the number of output channels must be equal "
                                "to the sum of inputs channels.");
    }
}


template <class T>
void N2D2::ActivationCell_Frame<T>::initializeDataDependent()
{
    Cell_Frame<T>::initializeDataDependent();
    initialize();
}


template <class T>
void N2D2::ActivationCell_Frame<T>::propagate(bool inference)
{
    mInputs.synchronizeDBasedToH();

    if(!inference) {
        if(mInputs.size() > 1) {
            if(mWorkspaceCPU.empty()) {
                mWorkspaceCPU.resize({  mDiffInputs.dimX(), 
                                        mDiffInputs.dimY(), 
                                        mDiffInputs.dimZ(), 
                                        mDiffInputs.dimB()});
                mWorkspaceCPU.fill(T(0.0));
                mWorkspaceCPU.synchronizeHToD();
            }
        }
    }
    // Copy data following inputs size and batch size to allow 
    // in-place operation in activation step
    if(mInputs.size() > 1) {
        size_t outStrideOffset = 0;
        for(size_t b = 0; b < mOutputs.dimB(); ++b) {
            for(size_t k = 0; k < mInputs.size(); ++k) {
                const Tensor<T>& input = tensor_cast<T>(mInputs[k]);
                const size_t chrunkSize 
                    = (input.dimX()*input.dimY()*input.dimZ()) ;
                std::copy_n(input.begin() + (b*chrunkSize),
                            chrunkSize,
                            mOutputs.begin() + (outStrideOffset));
                outStrideOffset += chrunkSize;
            }
        }
    }
    else {
        const Tensor<T>& input = tensor_cast<T>(mInputs[0]);
        const size_t size 
                    = (input.dimX()*input.dimY()*input.dimZ()*input.dimB()) ;
        std::copy_n(input.begin(),
                    size,
                    mOutputs.begin());
    }

    mActivation->propagate(*this, mOutputs, inference);
    mDiffInputs.clearValid();
}

template <class T>
void N2D2::ActivationCell_Frame<T>::backPropagate()
{
    if (!mDiffInputs.isValid())
        return;

    if (!mDiffOutputs.empty()) {
        if(mInputs.size() > 1) {
            //backpropagate 
            mActivation->backPropagate(*this, 
                                        mOutputs, 
                                        mOutputs, 
                                        mDiffInputs,
                                        mWorkspaceCPU);

            size_t diffOutStride = 0;
            for(size_t k = 0; k < mDiffOutputs.size(); ++k) {
                Tensor<T> diffOutput = (mDiffOutputs[k].isValid())
                                                    ? tensor_cast<T>(mDiffOutputs[k])
                                                    : tensor_cast_nocopy<T>(mDiffOutputs[k]);

                const size_t chrunkSize 
                    = (diffOutput.dimX()*diffOutput.dimY()*diffOutput.dimZ()) ;

                for(size_t b = 0; b < mWorkspaceCPU.dimB(); ++b) {
                    const size_t batchOffset 
                        = mWorkspaceCPU.dimX() * mWorkspaceCPU.dimY() *mWorkspaceCPU.dimZ();
                    std::copy_n(mWorkspaceCPU.begin() + (b*batchOffset + diffOutStride),
                                chrunkSize,
                                diffOutput.begin() + (chrunkSize*b));
                    
                }

                mDiffOutputs[k] = diffOutput;
                mDiffOutputs[k].setValid();
                mDiffOutputs[k].synchronizeHToD();

                diffOutStride += chrunkSize;
            }
        } 
        else {
            const Tensor<T>& input = tensor_cast<T>(mInputs[0]);
            Tensor<T> diffOutput = (mDiffOutputs[0].isValid())
                ? tensor_cast<T>(mDiffOutputs[0])
                : tensor_cast_nocopy<T>(mDiffOutputs[0]);

            mActivation->backPropagate(*this, input, mOutputs, mDiffInputs,
                                                            diffOutput);

            mDiffOutputs[0] = diffOutput;

            mDiffOutputs[0].setValid();
            mDiffOutputs[0].synchronizeHToD();

        }

    }
}

template <class T>
void N2D2::ActivationCell_Frame<T>::update()
{
    mActivation->update(mInputs.dimB());
}

template <class T>
void N2D2::ActivationCell_Frame<T>::checkGradient(double epsilon, double maxError)
{
    GradientCheck<T> gc(epsilon, maxError);
    gc.initialize(mInputs,
                  mOutputs,
                  mDiffInputs,
                  std::bind(&ActivationCell_Frame<T>::propagate, this, false),
                  std::bind(&ActivationCell_Frame<T>::backPropagate, this));

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
N2D2::ActivationCell_Frame<T>::~ActivationCell_Frame()
{
    //dtor
}

namespace N2D2 {
    template class ActivationCell_Frame<half_float::half>;
    template class ActivationCell_Frame<float>;
    template class ActivationCell_Frame<double>;
}
