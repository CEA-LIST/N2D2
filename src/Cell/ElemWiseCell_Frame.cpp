/*
    (C) Copyright 2014 CEA LIST. All Rights Reserved.
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
#include "Cell/ElemWiseCell_Frame.hpp"
#include "DeepNet.hpp"

N2D2::Registrar<N2D2::ElemWiseCell>
N2D2::ElemWiseCell_Frame::mRegistrar("Frame", N2D2::ElemWiseCell_Frame::create);

N2D2::ElemWiseCell_Frame::ElemWiseCell_Frame(const DeepNet& deepNet, const std::string& name,
                                     unsigned int nbOutputs,
                                     Operation operation,
                                     CoeffMode mode,
                                     const std::vector<Float_T>& weights,
                                     const std::vector<Float_T>& shifts,
                                     const std::shared_ptr
                                     <Activation>& activation)
    : Cell(deepNet, name, nbOutputs),
      ElemWiseCell(deepNet, name,
               nbOutputs,
               operation,
               mode,
               weights,
               shifts),
      Cell_Frame<Float_T>(deepNet, name, nbOutputs, activation)
{
    // ctor
}

void N2D2::ElemWiseCell_Frame::initialize()
{
    for (unsigned int k = 0, size = mInputs.size(); k < size; ++k) {
        if (mInputs[k].dimZ() != mOutputs.dimZ()
            || mInputs[k].dimB() != mOutputs.dimB())
        {
            std::stringstream errorMsg;
            errorMsg << "ElemWiseCell_Frame::initialize(): for cell "
                << mName << ": the input tensor dimensions ("
                << mInputs[k].dims() << ") must match the output dimensions ("
                << mOutputs.dims() << ") for input #" << k << ".";

            throw std::runtime_error(errorMsg.str());
        }
    }

    if(mWeights.empty()) {
        std::cout << Utils::cwarning << "ElemWiseCell_Frame::initialize(): Empty weights for cell " 
                    << mName << ": Initialize weights to default value(1.0)" 
                    << Utils::cdef << std::endl;
    }
    if(mShifts.empty()) {
        std::cout << Utils::cwarning << "ElemWiseCell_Frame::initialize(): Empty shifts for cell " 
                    << mName << ": Initialize shifts to default value(0.0)" 
                    << Utils::cdef << std::endl;
    }

    if(mCoeffMode == ElemWiseCell::PerChannel) {
        std::size_t nbChannels = 0;
        for (unsigned int k = 0, size = mInputs.size(); k < size; ++k) {
            nbChannels += mInputs[k].dimZ();
        }

        mWeights.resize(nbChannels, 1.0);
        mShifts.resize(nbChannels, 0.0);
    } 
    else {
        mWeights.resize(mInputs.size(), 1.0);
        mShifts.resize(mInputs.size(), 0.0);
    }

    if (mOperation == Max)
        mArgMax.resize(mOutputs.dims());

    if (mOperation == EuclideanSum)
        mInterTerm.resize(mOutputs.dims());
}

void N2D2::ElemWiseCell_Frame::propagate(bool inference)
{
    const unsigned int nbInputs = mInputs.size();
    const unsigned int nbElems = mInputs[0].size();

    mInputs.synchronizeDBasedToH();

    std::vector<Tensor<Float_T> > inputs;

    for (unsigned int k = 0; k < nbInputs; ++k)
        inputs.push_back(tensor_cast<Float_T>(mInputs[k]));

    if (mOperation == Sum) {
        for(unsigned int b = 0; b < inputs[0].dimB(); ++b) {
            for(std::size_t ch = 0; ch < inputs[0].dimZ(); ++ch) {
                for(std::size_t y = 0; y < inputs[0].dimY(); ++y) {
                    for(std::size_t x = 0; x < inputs[0].dimX(); ++x) {
                        std::size_t c0 = (mCoeffMode == ElemWiseCell::PerChannel) ? 
                                                ch : 0;
                        mOutputs(x,y,ch,b) = mWeights[c0] * inputs[0](x,y,ch,b)
                                            + mShifts[c0];
                        
                        for (unsigned int k = 1; k < nbInputs; ++k) {
                            std::size_t c1 = (mCoeffMode == ElemWiseCell::PerChannel) ? 
                                                c0 : k;
                            mOutputs(x,y,ch,b) += mWeights[c1] * inputs[k](x,y,ch,b)
                                                + mShifts[c1];
                        }
                    }
                }
            }
        }
    }
    else if (mOperation == AbsSum) {
        for (unsigned int n = 0; n < nbElems; ++n) {
            mOutputs(n) = mWeights[0] * std::abs(inputs[0](n));

            for (unsigned int k = 1; k < nbInputs; ++k)
                mOutputs(n) += mWeights[k] * std::abs(inputs[k](n));
        }
    }
    else if (mOperation == EuclideanSum) {
        for (unsigned int n = 0; n < nbElems; ++n) {
            mInterTerm(n) = (mWeights[0] * mWeights[0])
                * (inputs[0](n) * inputs[0](n))
                + (mShifts[0]*mShifts[0]);

            for (unsigned int k = 1; k < nbInputs; ++k) {
                mInterTerm(n) += (mWeights[k] * mWeights[k])
                    * (inputs[k](n) * inputs[k](n))
                    + (mShifts[k]*mShifts[k]);
            }

            mInterTerm(n) = std::sqrt(mInterTerm(n));
            mOutputs(n) = mInterTerm(n);
        }
    }
    else if (mOperation == Prod) {
        for (unsigned int n = 0; n < nbElems; ++n) {
            mOutputs(n) = inputs[0](n);

            for (unsigned int k = 1; k < nbInputs; ++k)
                mOutputs(n) *= inputs[k](n);
        }
    }
    else if (mOperation == Max) {
        for (unsigned int n = 0; n < nbElems; ++n) {
            Float_T maxVal = inputs[0](n);
            unsigned int argMax = 0;

            for (unsigned int k = 1; k < nbInputs; ++k) {
                if (inputs[k](n) > maxVal) {
                    maxVal = inputs[k](n);
                    argMax = k;
                }
            }

            mOutputs(n) = maxVal;
            mArgMax(n) = argMax;
        }
    }
    else {
        throw std::runtime_error("ElemWiseCell_Frame::propagate(): "
                                 "unknown operation type.");
    }

    Cell_Frame<Float_T>::propagate(inference);
    mDiffInputs.clearValid();
}

void N2D2::ElemWiseCell_Frame::backPropagate()
{
    if (mDiffOutputs.empty() || !mDiffInputs.isValid())
        return;

    const unsigned int nbInputs = mInputs.size();
    const unsigned int nbElems = mInputs[0].size();

    Cell_Frame<Float_T>::backPropagate();

    std::vector<Tensor<Float_T> > inputs;

    for (unsigned int k = 0; k < nbInputs; ++k)
        inputs.push_back(tensor_cast_nocopy<Float_T>(mInputs[k]));

    #pragma omp parallel for
    for (int k = 0; k < (int)nbInputs; ++k) {
        const float beta = (mDiffOutputs[k].isValid()) ? 1.0f : 0.0f;

        Tensor<Float_T> diffOutput = (mDiffOutputs[k].isValid())
            ? tensor_cast<Float_T>(mDiffOutputs[k])
            : tensor_cast_nocopy<Float_T>(mDiffOutputs[k]);

        if (mOperation == Sum) {
            for(unsigned int b = 0; b < inputs[k].dimB(); ++b) {
                for(std::size_t ch = 0; ch < inputs[k].dimZ(); ++ch) {
                    for(std::size_t y = 0; y < inputs[k].dimY(); ++y) {
                        for(std::size_t x = 0; x < inputs[k].dimX(); ++x) {
                            std::size_t coeffIdx = mCoeffMode == ElemWiseCell::PerChannel ? 
                                                    ch : k;
                            diffOutput(x,y,ch,b) = mWeights[coeffIdx] * mDiffInputs(x,y,ch,b)
                                                + beta * diffOutput(x,y,ch,b);
                        }
                    }
                }
            }
        }
        else if (mOperation == AbsSum) {
            for (unsigned int n = 0; n < nbElems; ++n) {
                const Float_T sign = (inputs[k](n) >= 0.0) ? 1.0 : -1.0;
                diffOutput(n) = mWeights[k] * sign * mDiffInputs(n)
                                    + beta * diffOutput(n);
            }
        }
        else if (mOperation == EuclideanSum) {
            for (unsigned int n = 0; n < nbElems; ++n) {
                diffOutput(n) = (mInterTerm(n) != 0.0)
                    ? (mWeights[k] * mWeights[k])
                        * (inputs[k](n) / mInterTerm(n))
                        * mDiffInputs(n) + beta * diffOutput(n)
                    : beta * diffOutput(n);
            }
        }
        else if (mOperation == Prod) {
            for (unsigned int n = 0; n < nbElems; ++n) {
                Float_T prodTerm = 1.0;

                for (unsigned int i = 0; i < nbInputs; ++i) {
                    if (i != (unsigned int)k)
                        prodTerm *= inputs[i](n);
                }

                diffOutput(n) = prodTerm * mDiffInputs(n)
                                    + beta * diffOutput(n);
            }
        }
        else if (mOperation == Max) {
            for (unsigned int n = 0; n < nbElems; ++n) {
                diffOutput(n) = (mArgMax(n) == (unsigned int)k)
                    ? (mDiffInputs(n) + beta * diffOutput(n))
                    : beta * diffOutput(n);
            }
        }
        else {
            throw std::runtime_error("ElemWiseCell_Frame::propagate(): "
                                     "unknown operation type.");
        }

        mDiffOutputs[k] = diffOutput;
        mDiffOutputs[k].setValid();
    }

    mDiffOutputs.synchronizeHToD();
}

void N2D2::ElemWiseCell_Frame::update()
{
    Cell_Frame<float>::update();
}

void N2D2::ElemWiseCell_Frame::checkGradient(double epsilon, double maxError)
{
    GradientCheck<Float_T> gc(epsilon, maxError);
    gc.initialize(mInputs,
                  mOutputs,
                  mDiffInputs,
                  std::bind(&ElemWiseCell_Frame::propagate, this, false),
                  std::bind(&ElemWiseCell_Frame::backPropagate, this),
                  (mOperation == Max));

    if (!mDiffOutputs.empty()) {
        for (unsigned int in = 0; in < mInputs.size(); ++in) {
            std::stringstream name;
            name << mName + "_mDiffOutputs[" << in << "]";

            gc.check(name.str(), mInputs[in], mDiffOutputs[in]);
        }
    } else {
        std::cout << Utils::cwarning << "Empty diff. outputs for cell " << mName
                  << ", could not check the gradient!" << Utils::cdef
                  << std::endl;
    }
}

std::pair<double, double> N2D2::ElemWiseCell_Frame::getOutputsRange() const {
    const auto& activation = Cell_Frame<Float_T>::getActivation();
    return activation?activation->getOutputRange():ElemWiseCell::getOutputsRange();
}

N2D2::ElemWiseCell_Frame::~ElemWiseCell_Frame()
{

}
