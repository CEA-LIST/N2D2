/*
    (C) Copyright 2018 CEA LIST. All Rights Reserved.
    Contributor(s): David BRIAND (david.briand@cea.fr)

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
#include "Cell/PaddingCell_Frame.hpp"
#include "DeepNet.hpp"

N2D2::Registrar<N2D2::PaddingCell>
N2D2::PaddingCell_Frame::mRegistrar("Frame",
                                     N2D2::PaddingCell_Frame::create);

N2D2::PaddingCell_Frame::PaddingCell_Frame(const DeepNet& deepNet, const std::string& name,
                                            unsigned int nbOutputs,
                                            int topPad,
                                            int botPad,
                                            int leftPad,
                                            int rightPad)
    : Cell(deepNet, name, nbOutputs),
      PaddingCell(deepNet, name,
                  nbOutputs,
                  topPad,
                  botPad,
                  leftPad,
                  rightPad),
      Cell_Frame<Float_T>(deepNet, name, nbOutputs),
      mPaddingDesc(mLeftPad, mRightPad, mTopPad, mBotPad)

{
    // ctor
}

void N2D2::PaddingCell_Frame::initialize()
{
    unsigned int inputZ = mInputs[0].dimZ();
    for(unsigned int k = 1; k < mInputs.size(); ++k)
    {
        inputZ += mInputs[k].dimZ();

    }

    if (inputZ != mOutputs.dimZ()) {
        throw std::domain_error("PaddingCell_Frame::initialize():"
                                " the number of output channels must be equal "
                                "to the sum of inputs channels.");
    }
}



void N2D2::PaddingCell_Frame::initializeDataDependent(){
    Cell_Frame<Float_T>::initializeDataDependent();

    initialize();
}


void N2D2::PaddingCell_Frame::propagate(bool inference)
{
    mInputs.synchronizeDBasedToH();

    unsigned int offset = 0;

    for (unsigned int k = 0, size = mInputs.size(); k < size; ++k) {
        const Tensor<Float_T>& input = tensor_cast<Float_T>(mInputs[k]);

        PaddingCell_Frame_Kernels::forward( input,
                                            mPaddingDesc,
                                            mInputs[k].dimZ(),
                                            0,
                                            offset,
                                            mOutputs);

        offset += mInputs[k].dimZ();
    }

    Cell_Frame<Float_T>::propagate(inference);
    mDiffInputs.clearValid();
}

void N2D2::PaddingCell_Frame::backPropagate()
{
    if (!mDiffInputs.isValid())
        return;

    Cell_Frame<Float_T>::backPropagate();

    unsigned int offset = 0;

    PaddingCell_Frame_Kernels::Descriptor
            backwardPaddingDesc(-mPaddingDesc.leftPad,
                                -mPaddingDesc.rightPad,
                                -mPaddingDesc.topPad,
                                -mPaddingDesc.botPad);

    for (unsigned int k = 0, size = mInputs.size(); k < size; ++k) {
        if (mDiffOutputs[k].empty())
            continue;

        Tensor<Float_T> diffOutput
            = tensor_cast_nocopy<Float_T>(mDiffOutputs[k]);

        PaddingCell_Frame_Kernels::forward( mDiffInputs,
                                            backwardPaddingDesc,
                                            mDiffOutputs[k].dimZ(),
                                            offset,
                                            0,
                                            diffOutput);

        offset += mDiffOutputs[k].dimZ();

        mDiffOutputs[k] = diffOutput;
        mDiffOutputs[k].setValid();
    }

    mDiffOutputs.synchronizeHToD();
}

void N2D2::PaddingCell_Frame::update()
{
    Cell_Frame<float>::update();
}


void N2D2::PaddingCell_Frame::checkGradient(double epsilon, double maxError)
{
    GradientCheck<Float_T> gc(epsilon, maxError);
    gc.initialize(mInputs,
                  mOutputs,
                  mDiffInputs,
                  std::bind(&PaddingCell_Frame::propagate, this, false),
                  std::bind(&PaddingCell_Frame::backPropagate, this));

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

std::pair<double, double> N2D2::PaddingCell_Frame::getOutputsRange() const {
    return PaddingCell::getOutputsRange();
}

N2D2::PaddingCell_Frame::~PaddingCell_Frame()
{

}

