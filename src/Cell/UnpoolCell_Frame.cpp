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
#include "Cell/UnpoolCell_Frame.hpp"
#include "DeepNet.hpp"

N2D2::Registrar<N2D2::UnpoolCell>
N2D2::UnpoolCell_Frame::mRegistrar("Frame", N2D2::UnpoolCell_Frame::create);

N2D2::UnpoolCell_Frame::UnpoolCell_Frame(const DeepNet& deepNet, const std::string& name,
    const std::vector<unsigned int>& poolDims,
    unsigned int nbOutputs,
    const std::vector<unsigned int>& strideDims,
    const std::vector<unsigned int>& paddingDims,
    Pooling pooling,
    const std::shared_ptr<Activation>& activation)
    : Cell(deepNet, name, nbOutputs),
      UnpoolCell(deepNet, name,
               poolDims,
               nbOutputs,
               strideDims,
               paddingDims,
               pooling),
      Cell_Frame<Float_T>(deepNet, name, nbOutputs, activation),
      mPoolDesc(poolDims.size(),
                &poolDims[0],
                &strideDims[0],
                &paddingDims[0])
    ,mArgMax({true,true,false,true})
{
    // ctor
    assert(poolDims.size() <= POOL_KERNEL_MAX_DIMS);

    if (poolDims.size() != 2) {
        throw std::domain_error("UnpoolCell_Frame: only 2D pooling is"
                                " supported");
    }

    if (strideDims.size() != poolDims.size()) {
        throw std::domain_error("UnpoolCell_Frame: the number of dimensions"
                                " of stride must match the number of"
                                " dimensions of the pooling.");
    }

    if (paddingDims.size() != poolDims.size()) {
        throw std::domain_error("UnpoolCell_Frame: the number of dimensions"
                                " of padding must match the number of"
                                " dimensions of the pooling.");
    }
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

void N2D2::UnpoolCell_Frame::propagate(bool inference)
{
    mInputs.synchronizeDBasedToH();

    const float alpha = 1.0f;
    float beta = 0.0f;

    unsigned int offset = 0;

    for (unsigned int k = 0, size = mInputs.size(); k < size; ++k) {
        if (k > 0)
            beta = 1.0;

        const Tensor<Float_T>& input = tensor_cast<Float_T>(mInputs[k]);

        if (mPooling == Max) {
            PoolCell_Frame_Kernels::backwardMax(&alpha,
                                                input,
                                                mPoolDesc,
                                                &beta,
                                                mOutputs,
                                                mArgMax[k],
                                                mMapping.rows(offset,
                                                           mInputs[k].dimZ()));
        }
        else {
            PoolCell_Frame_Kernels::backwardAverage(&alpha,
                                                    input,
                                                    mPoolDesc,
                                                    &beta,
                                                    mOutputs,
                                                    true,
                                                    mMapping.rows(offset,
                                                        mInputs[k].dimZ()));
        }

        offset += mInputs[k].dimZ();
    }

    Cell_Frame<Float_T>::propagate(inference);
    mDiffInputs.clearValid();
}

void N2D2::UnpoolCell_Frame::backPropagate()
{
    if (mDiffOutputs.empty())
        return;

    Cell_Frame<Float_T>::backPropagate();

    const Float_T alpha = 1.0;

    unsigned int offset = 0;

    for (unsigned int k = 0, size = mInputs.size(); k < size; ++k) {
        const Float_T beta = (mDiffOutputs[k].isValid()) ? 1.0 : 0.0;

        Tensor<Float_T> diffOutput = (mDiffOutputs[k].isValid())
            ? tensor_cast<Float_T>(mDiffOutputs[k])
            : tensor_cast_nocopy<Float_T>(mDiffOutputs[k]);

        if (mPooling == Max) {
            PoolCell_Frame_Kernels::forwardMax(&alpha,
                                               mDiffInputs,
                                               mPoolDesc,
                                               &beta,
                                               diffOutput,
                                               mArgMax[k],
                                               true,
                                               mMapping.rows(offset,
                                                          mInputs[k].dimZ()));
        }
        else {
            PoolCell_Frame_Kernels::forwardAverage(&alpha,
                                                   mDiffInputs,
                                                   mPoolDesc,
                                                   &beta,
                                                   diffOutput,
                                                   true,
                                                   mMapping.rows(offset,
                                                       mInputs[k].dimZ()));
        }

        offset += mInputs[k].dimZ();

        mDiffOutputs[k] = diffOutput;
        mDiffOutputs[k].setValid();
    }

    mDiffOutputs.synchronizeHToD();
}

void N2D2::UnpoolCell_Frame::update()
{
}

void N2D2::UnpoolCell_Frame::checkGradient(double epsilon, double maxError)
{
    GradientCheck<Float_T> gc(epsilon, maxError);
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
