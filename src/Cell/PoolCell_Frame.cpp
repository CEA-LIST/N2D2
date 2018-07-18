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

#include "Cell/PoolCell_Frame.hpp"

N2D2::Registrar<N2D2::PoolCell>
N2D2::PoolCell_Frame::mRegistrar("Frame", N2D2::PoolCell_Frame::create);

N2D2::PoolCell_Frame::PoolCell_Frame(const std::string& name,
    const std::vector<unsigned int>& poolDims,
    unsigned int nbOutputs,
    const std::vector<unsigned int>& strideDims,
    const std::vector<unsigned int>& paddingDims,
    Pooling pooling,
    const std::shared_ptr<Activation<Float_T> >& activation)
    : Cell(name, nbOutputs),
      PoolCell(name,
               poolDims,
               nbOutputs,
               strideDims,
               paddingDims,
               pooling),
      Cell_Frame(name, nbOutputs, activation),
      mPoolDesc(poolDims.size(),
                &poolDims[0],
                &strideDims[0],
                &paddingDims[0])
{
    // ctor
    assert(poolDims.size() <= POOL_KERNEL_MAX_DIMS);

    if (poolDims.size() != 2) {
        throw std::domain_error("PoolCell_Frame: only 2D pooling is"
                                " supported");
    }

    if (strideDims.size() != poolDims.size()) {
        throw std::domain_error("PoolCell_Frame: the number of dimensions"
                                " of stride must match the number of"
                                " dimensions of the pooling.");
    }

    if (paddingDims.size() != poolDims.size()) {
        throw std::domain_error("PoolCell_Frame: the number of dimensions"
                                " of padding must match the number of"
                                " dimensions of the pooling.");
    }
}

void N2D2::PoolCell_Frame::initialize()
{
    for (unsigned int k = 0, size = mInputs.size(); k < size; ++k) {
        if (mInputs[k].size() == 0)
            throw std::runtime_error("Zero-sized input for PoolCell " + mName);

        if (mArgMax.size() == k) {
            mArgMax.push_back(new Tensor<PoolCell_Frame_Kernels::ArgMax>
                              (mOutputs.dims()));
        }
    }
}

void N2D2::PoolCell_Frame::propagate(bool /*inference*/)
{
    mInputs.synchronizeDToH();

    const float alpha = 1.0f;
    float beta = 0.0f;

    unsigned int offset = 0;

    for (unsigned int k = 0, size = mInputs.size(); k < size; ++k) {
        if (k > 0)
            beta = 1.0;

        const Tensor<Float_T>& input = tensor_cast<Float_T>(mInputs[k]);

        if (mPooling == Max) {
            PoolCell_Frame_Kernels::forwardMax(&alpha,
                                               input,
                                               mPoolDesc,
                                               &beta,
                                               mOutputs,
                                               mArgMax[k],
                                               false,
                                               mMaps.rows(offset,
                                                          mInputs[k].dimZ()));
        }
        else {
            PoolCell_Frame_Kernels::forwardAverage(&alpha,
                                                   input,
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

void N2D2::PoolCell_Frame::backPropagate()
{
    if (mDiffOutputs.empty())
        return;

    Cell_Frame::backPropagate();

    const Float_T alpha = 1.0;

    unsigned int offset = 0;

    for (unsigned int k = 0, size = mInputs.size(); k < size; ++k) {
        const Float_T beta = (mDiffOutputs[k].isValid()) ? 1.0 : 0.0;

        Tensor<Float_T> diffOutput = (mDiffOutputs[k].isValid())
            ? tensor_cast<Float_T>(mDiffOutputs[k])
            : tensor_cast_nocopy<Float_T>(mDiffOutputs[k]);

        if (mPooling == Max) {
            PoolCell_Frame_Kernels::backwardMax(&alpha,
                                                mDiffInputs,
                                                mPoolDesc,
                                                &beta,
                                                diffOutput,
                                                mArgMax[k],
                                                mMaps.rows(offset,
                                                           mInputs[k].dimZ()));
        }
        else {
            PoolCell_Frame_Kernels::backwardAverage(&alpha,
                                                    mDiffInputs,
                                                    mPoolDesc,
                                                    &beta,
                                                    diffOutput,
                                                    true,
                                                    mMaps.rows(offset,
                                                          mInputs[k].dimZ()));
        }

        offset += mInputs[k].dimZ();

        mDiffOutputs[k] = diffOutput;
        mDiffOutputs[k].setValid();
    }

    mDiffOutputs.synchronizeHToD();
}

void N2D2::PoolCell_Frame::update()
{
}

void N2D2::PoolCell_Frame::checkGradient(double epsilon, double maxError)
{
    GradientCheck<Float_T> gc(epsilon, maxError);
    gc.initialize(mInputs,
                  mOutputs,
                  mDiffInputs,
                  std::bind(&PoolCell_Frame::propagate, this, false),
                  std::bind(&PoolCell_Frame::backPropagate, this),
                  (mPooling == Max));

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

N2D2::PoolCell_Frame::~PoolCell_Frame()
{
    for (unsigned int k = 0, size = mArgMax.size(); k < size; ++k)
        delete &mArgMax[k];
}
