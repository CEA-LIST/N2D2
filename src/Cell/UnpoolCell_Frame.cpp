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

#include "Cell/UnpoolCell_Frame.hpp"

N2D2::Registrar<N2D2::UnpoolCell>
N2D2::UnpoolCell_Frame::mRegistrar("Frame", N2D2::UnpoolCell_Frame::create);

N2D2::UnpoolCell_Frame::UnpoolCell_Frame(const std::string& name,
                                     unsigned int poolWidth,
                                     unsigned int poolHeight,
                                     unsigned int nbOutputs,
                                     unsigned int strideX,
                                     unsigned int strideY,
                                     unsigned int paddingX,
                                     unsigned int paddingY,
                                     Pooling pooling,
                                     const std::shared_ptr
                                     <Activation<Float_T> >& activation)
    : Cell(name, nbOutputs),
      UnpoolCell(name,
               poolWidth,
               poolHeight,
               nbOutputs,
               strideX,
               strideY,
               paddingX,
               paddingY,
               pooling),
      Cell_Frame(name, nbOutputs, activation),
      mPoolDesc(poolWidth, poolHeight, strideX, strideY, paddingX, paddingY)
{
    // ctor
}

void N2D2::UnpoolCell_Frame::initialize()
{
    if (mArgMax.size() == 0)
        throw std::runtime_error("ArgMax missing for UnpoolCell " + mName);
    else if (mArgMax.size() != mInputs.size()) {
        throw std::runtime_error("Wrong number of ArgMax tensors for "
                                 "UnpoolCell " + mName);
    }

    for (unsigned int k = 0, size = mInputs.size(); k < size; ++k) {
        if (mInputs[k].size() == 0)
            throw std::runtime_error("Zero-sized input for UnpoolCell "
                                     + mName);
    }
}

void N2D2::UnpoolCell_Frame::propagate(bool /*inference*/)
{
    mInputs.synchronizeDToH();

    const float alpha = 1.0f;
    float beta = 0.0f;

    unsigned int offset = 0;

    for (unsigned int k = 0, size = mInputs.size(); k < size; ++k) {
        if (k > 0)
            beta = 1.0;

        if (mPooling == Max) {
            PoolCell_Frame_Kernels::backwardMax(&alpha,
                                                mInputs[k],
                                                mPoolDesc,
                                                &beta,
                                                mOutputs,
                                                mArgMax[k],
                                                mMaps.rows(offset,
                                                           mInputs[k].dimZ()));
        }
        else {
            PoolCell_Frame_Kernels::backwardAverage(&alpha,
                                                    mInputs[k],
                                                    mPoolDesc,
                                                    &beta,
                                                    mOutputs,
                                                    true,
                                                    mMaps.rows(offset,
                                                        mInputs[k].dimZ()));
        }

        offset += mInputs[k].dimZ();
    }

    Cell_Frame::propagate();
    mDiffInputs.clearValid();
}

void N2D2::UnpoolCell_Frame::backPropagate()
{
    if (mDiffOutputs.empty())
        return;

    Cell_Frame::backPropagate();

    const Float_T alpha = 1.0;

    unsigned int offset = 0;

    for (unsigned int k = 0, size = mInputs.size(); k < size; ++k) {
        const Float_T beta = (mDiffOutputs[k].isValid()) ? 1.0 : 0.0;

        if (mPooling == Max) {
            PoolCell_Frame_Kernels::forwardMax(&alpha,
                                               mDiffInputs,
                                               mPoolDesc,
                                               &beta,
                                               mDiffOutputs[k],
                                               mArgMax[k],
                                               true,
                                               mMaps.rows(offset,
                                                          mInputs[k].dimZ()));
        }
        else {
            PoolCell_Frame_Kernels::forwardAverage(&alpha,
                                                   mDiffInputs,
                                                   mPoolDesc,
                                                   &beta,
                                                   mDiffOutputs[k],
                                                   true,
                                                   mMaps.rows(offset,
                                                       mInputs[k].dimZ()));
        }

        offset += mInputs[k].dimZ();
        mDiffOutputs[k].setValid();
    }

    mDiffOutputs.synchronizeHToD();
}

void N2D2::UnpoolCell_Frame::update()
{
}

void N2D2::UnpoolCell_Frame::checkGradient(double epsilon, double maxError)
{
    GradientCheck gc(epsilon, maxError);
    gc.initialize(mInputs,
                  mOutputs,
                  mDiffInputs,
                  std::bind(&UnpoolCell_Frame::propagate, this, false),
                  std::bind(&UnpoolCell_Frame::backPropagate, this),
                  (mPooling == Max));

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

void N2D2::UnpoolCell_Frame::addArgMax(
    Interface<PoolCell_Frame_Kernels::ArgMax>* argMax)
{
    for (unsigned int k = 0; k < argMax->size(); ++k)
        mArgMax.push_back(&(*argMax)[k]);
}
