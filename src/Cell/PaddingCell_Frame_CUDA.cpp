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

#ifdef CUDA
#include <cudnn.h>

#include "Cell/PaddingCell_Frame_CUDA.hpp"

N2D2::Registrar<N2D2::PaddingCell>
N2D2::PaddingCell_Frame_CUDA::mRegistrar("Frame_CUDA",
                                     N2D2::PaddingCell_Frame_CUDA::create);

N2D2::PaddingCell_Frame_CUDA::PaddingCell_Frame_CUDA(const std::string& name,
                                                     unsigned int nbOutputs,
                                                    int topPad,
                                                    int botPad,
                                                    int leftPad,
                                                    int rightPad)
    : Cell(name, nbOutputs),
      PaddingCell(name, 
                  nbOutputs,
                  topPad,
                  botPad,
                  leftPad,
                  rightPad),
      Cell_Frame_CUDA(name, nbOutputs)
{
    // ctor
}

void N2D2::PaddingCell_Frame_CUDA::initialize()
{
    unsigned int inputX = mInputs[0].dimX();
    unsigned int inputY = mInputs[0].dimY();
    unsigned int inputZ = mInputs[0].dimZ();
    for(unsigned int k = 1; k < mInputs.size(); ++k)
    {
        if(inputX != mInputs[k].dimX())
            throw std::domain_error("PaddingCell_Frame_CUDA::initialize():"
                            " Input layers must have the same width dimension for layer " + k);
    
        if(inputY != mInputs[k].dimY())
            throw std::domain_error("PaddingCell_Frame_CUDA::initialize():"
                            " Input layers must have the same height dimension for layer " + k);

        inputZ += mInputs[k].dimZ();

    }

    if (inputZ != mOutputs.dimZ()) {
        throw std::domain_error("PaddingCell_Frame_CUDA::initialize():"
                                " the number of output channels must be equal "
                                "to the sum of inputs channels.");
    }

}

void N2D2::PaddingCell_Frame_CUDA::propagate(bool /*inference*/)
{
    mInputs.synchronizeHBasedToD();

    unsigned int outputOffset = 0;

    for(unsigned int k = 0; k < mInputs.size(); ++k)
    {

        cudaSPadding(mOutputs.dimX(),
                    mOutputs.dimY(),
                    mInputs[k].dimZ(),
                    mOutputs.dimB(),
                    mInputs[k].dimX(),
                    mInputs[k].dimY(),
                    mLeftPad,
                    mRightPad,
                    mTopPad,
                    mBotPad,
                    mInputs[k].getDevicePtr(),
                    mOutputs.getDevicePtr() + outputOffset);

        outputOffset += mInputs[k].dimZ()*mOutputs.dimX()*mOutputs.dimY()*mOutputs.dimB();
    }

    Cell_Frame_CUDA::propagate();

    mDiffInputs.clearValid();

}

void N2D2::PaddingCell_Frame_CUDA::backPropagate()
{
    if (mDiffOutputs.empty())
        return;

    unsigned int outputOffset = 0;
    Cell_Frame_CUDA::backPropagate();

    for(unsigned int k = 0; k < mInputs.size(); ++k)
    {

        cudaSPadding(mDiffOutputs[k].dimX(),
                    mDiffOutputs[k].dimY(),
                    mDiffOutputs[k].dimZ(),
                    mDiffOutputs[k].dimB(),
                    mDiffInputs.dimX(),
                    mDiffInputs.dimY(),
                    -mLeftPad,
                    -mRightPad,
                    -mTopPad,
                    -mBotPad,
                    mDiffInputs.getDevicePtr() + outputOffset,
                    mDiffOutputs[k].getDevicePtr());

        outputOffset += mDiffOutputs[k].dimZ()
                            *mDiffInputs.dimX()
                            *mDiffInputs.dimY()
                            *mDiffInputs.dimB();

        mDiffOutputs[k].setValid();
    }

    mDiffOutputs.synchronizeDToHBased();

}

void N2D2::PaddingCell_Frame_CUDA::update()
{
}

void N2D2::PaddingCell_Frame_CUDA::checkGradient(double epsilon, double maxError)
{
    GradientCheck gc(epsilon, maxError);
    gc.initialize(mInputs,
                  mOutputs,
                  mDiffInputs,
                  std::bind(&PaddingCell_Frame_CUDA::propagate, this, false),
                  std::bind(&PaddingCell_Frame_CUDA::backPropagate, this));

    if (!mDiffOutputs.empty()) {
        for (unsigned int k = 0, size = mInputs.size(); k < size; ++k) {
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


N2D2::PaddingCell_Frame_CUDA::~PaddingCell_Frame_CUDA()
{

}

#endif
