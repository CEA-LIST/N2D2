/*
    (C) Copyright 2016 CEA LIST. All Rights Reserved.
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
                                             unsigned int nbOutputs)
    : Cell(name, nbOutputs),
      PaddingCell(name, nbOutputs),
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

    //unsigned int inputLineSize = mOutputs.dimX() - mLeftPad - mRightPad;
    unsigned int outOffset = 0;

    if(mLeftPad < 0)
        outOffset -= mLeftPad;

    for (unsigned int k = 0, size = mInputs.size(); k < size; ++k) 
    {
        unsigned int inOffset = 0;

        //if(mLeftPad < 0)
        //    inOffset -= mLeftPad;
        if(mTopPad < 0)
            inOffset = -mTopPad*mInputs[k].dimX();

        unsigned int cpySize = mInputs[k].dimX() + mLeftPad + mRightPad;
        
        cpySize = Utils::clamp<unsigned int>(cpySize, 0, mInputs[k].dimX());

        for(int batchPos = 0; batchPos < (int) mOutputs.dimB(); ++batchPos)
        {
            for(int zIdx = 0; zIdx < (int) mOutputs.dimZ(); ++zIdx)
            {
                for(int yIdx = -mTopPad; yIdx < (int)mInputs[k].dimY() + mBotPad; ++yIdx)
                {

                    if(yIdx >= 0 && yIdx < (int) mInputs[k].dimY())
                    {
                        if(mLeftPad < 0)
                            inOffset -= mLeftPad;
            
                        CHECK_CUDA_STATUS(cudaMemcpy(mOutputs.getDevicePtr() + outOffset + mLeftPad,
                                                     mInputs[k].getDevicePtr() + inOffset,
                                                     cpySize*sizeof(Float_T),
                                                     cudaMemcpyDeviceToDevice));
                        if(mRightPad < 0)
                            inOffset -= mRightPad;
                         
                        inOffset += cpySize;   
                        
                    }

                    outOffset += mOutputs.dimX();

                }
            }
        }
    }

    mDiffInputs.clearValid();
}

void N2D2::PaddingCell_Frame_CUDA::backPropagate()
{

}

void N2D2::PaddingCell_Frame_CUDA::update()
{
}

N2D2::PaddingCell_Frame_CUDA::~PaddingCell_Frame_CUDA()
{

}

#endif
