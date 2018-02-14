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

#include "Cell/ElemWiseCell_Frame.hpp"

N2D2::Registrar<N2D2::ElemWiseCell>
N2D2::ElemWiseCell_Frame::mRegistrar("Frame", N2D2::ElemWiseCell_Frame::create);

N2D2::ElemWiseCell_Frame::ElemWiseCell_Frame(const std::string& name,
                                     unsigned int nbOutputs,
                                     Operation operation,
                                     const std::vector<Float_T>& weights,
                                     const std::shared_ptr
                                     <Activation<Float_T> >& activation)
    : Cell(name, nbOutputs),
      ElemWiseCell(name,
               nbOutputs,
               operation,
               weights),
      Cell_Frame(name, nbOutputs, activation)
{
    // ctor
}

void N2D2::ElemWiseCell_Frame::initialize()
{
    for (unsigned int k = 0, size = mInputs.size(); k < size; ++k) {
        if (mInputs[k].dimZ() != mOutputs.dimZ()
            || mInputs[k].dimB() != mOutputs.dimB())
        {
            throw std::runtime_error("ElemWiseCell_Frame::initialize(): "
                                     "the input tensors dimensions must match "
                                     "the output dimensions.");
        }
    }

    mWeights.resize(mInputs.size(), 1.0);

    if (mOperation == Max) {
        mArgMax.resize(mOutputs.dimX(),
                       mOutputs.dimY(),
                       mOutputs.dimZ(),
                       mOutputs.dimB());
    }

    if (mOperation == EuclideanSum) {
        mInterTerm.resize(mOutputs.dimX(),
                          mOutputs.dimY(),
                          mOutputs.dimZ(),
                          mOutputs.dimB());
    }
}

void N2D2::ElemWiseCell_Frame::propagate(bool /*inference*/)
{
    const unsigned int nbInputs = mInputs.size();
    const unsigned int nbElems = mInputs[0].size();

    mInputs.synchronizeDToH();

    if (mOperation == Sum) {
        for (unsigned int n = 0; n < nbElems; ++n) {
            mOutputs(n) = mWeights[0] * mInputs[0](n);

            for (unsigned int k = 1; k < nbInputs; ++k)
                mOutputs(n) += mWeights[k] * mInputs[k](n);
        }
    }
    else if (mOperation == AbsSum) {
        for (unsigned int n = 0; n < nbElems; ++n) {
            mOutputs(n) = mWeights[0] * std::abs(mInputs[0](n));

            for (unsigned int k = 1; k < nbInputs; ++k)
                mOutputs(n) += mWeights[k] * std::abs(mInputs[k](n));
        }
    }
    else if (mOperation == EuclideanSum) {
        for (unsigned int n = 0; n < nbElems; ++n) {
            mInterTerm(n) = (mWeights[0] * mWeights[0])
                * (mInputs[0](n) * mInputs[0](n));

            for (unsigned int k = 1; k < nbInputs; ++k) {
                mInterTerm(n) += (mWeights[k] * mWeights[k])
                    * (mInputs[k](n) * mInputs[k](n));
            }

            mInterTerm(n) = std::sqrt(mInterTerm(n));
            mOutputs(n) = mInterTerm(n);
        }
    }
    else if (mOperation == Prod) {
        for (unsigned int n = 0; n < nbElems; ++n) {
            mOutputs(n) = mInputs[0](n);

            for (unsigned int k = 1; k < nbInputs; ++k)
                mOutputs(n) *= mInputs[k](n);
        }
    }
    else if (mOperation == Max) {
        for (unsigned int n = 0; n < nbElems; ++n) {
            Float_T maxVal = mInputs[0](n);
            unsigned int argMax = 0;

            for (unsigned int k = 1; k < nbInputs; ++k) {
                if (mInputs[k](n) > maxVal) {
                    maxVal = mInputs[k](n);
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

    Cell_Frame::propagate();
    mDiffInputs.clearValid();
}

void N2D2::ElemWiseCell_Frame::backPropagate()
{
    if (mDiffOutputs.empty())
        return;

    const unsigned int nbInputs = mInputs.size();
    const unsigned int nbElems = mInputs[0].size();

    Cell_Frame::backPropagate();

    #pragma omp parallel for
    for (int k = 0; k < (int)nbInputs; ++k) {
        Tensor4d<Float_T>& diffOutputs = mDiffOutputs[k];
        const float beta = (diffOutputs.isValid()) ? 1.0f : 0.0f;

        if (mOperation == Sum) {
            for (unsigned int n = 0; n < nbElems; ++n) {
                diffOutputs(n) = mWeights[k] * mDiffInputs(n)
                                    + beta * diffOutputs(n);
            }
        }
        else if (mOperation == AbsSum) {
            for (unsigned int n = 0; n < nbElems; ++n) {
                const Float_T sign = (mInputs[k](n) >= 0.0) ? 1.0 : -1.0;
                diffOutputs(n) = mWeights[k] * sign * mDiffInputs(n)
                                    + beta * diffOutputs(n);
            }
        }
        else if (mOperation == EuclideanSum) {
            for (unsigned int n = 0; n < nbElems; ++n) {
                diffOutputs(n) = (mInterTerm(n) != 0.0)
                    ? (mWeights[k] * mWeights[k])
                        * (mInputs[k](n) / mInterTerm(n))
                        * mDiffInputs(n) + beta * diffOutputs(n)
                    : beta * diffOutputs(n);
            }
        }
        else if (mOperation == Prod) {
            for (unsigned int n = 0; n < nbElems; ++n) {
                Float_T prodTerm = 1.0;

                for (unsigned int i = 0; i < nbInputs; ++i) {
                    if (i != (unsigned int)k)
                        prodTerm *= mInputs[i](n);
                }

                diffOutputs(n) = prodTerm * mDiffInputs(n)
                                    + beta * diffOutputs(n);
            }
        }
        else if (mOperation == Max) {
            for (unsigned int n = 0; n < nbElems; ++n) {
                diffOutputs(n) = (mArgMax(n) == (unsigned int)k)
                    ? (mDiffInputs(n) + beta * diffOutputs(n))
                    : beta * diffOutputs(n);
            }
        }
        else {
            throw std::runtime_error("ElemWiseCell_Frame::propagate(): "
                                     "unknown operation type.");
        }

        diffOutputs.setValid();
    }

    mDiffOutputs.synchronizeHToD();
}

void N2D2::ElemWiseCell_Frame::update()
{
}

void N2D2::ElemWiseCell_Frame::checkGradient(double epsilon, double maxError)
{
    GradientCheck gc(epsilon, maxError);
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

N2D2::ElemWiseCell_Frame::~ElemWiseCell_Frame()
{

}
