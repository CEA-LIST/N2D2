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

#include "Cell/ResizeCell_Frame_CUDA.hpp"

N2D2::Registrar<N2D2::ResizeCell>
N2D2::ResizeCell_Frame_CUDA::mRegistrar("Frame_CUDA",
                                       N2D2::ResizeCell_Frame_CUDA::create);

N2D2::ResizeCell_Frame_CUDA::ResizeCell_Frame_CUDA(const std::string& name,
                                                 unsigned int outputsWidth,
                                                 unsigned int outputsHeight,
                                                 unsigned int nbOutputs,
                                                 ResizeMode resizeMode)
    : Cell(name, nbOutputs),
      ResizeCell(name, outputsWidth, outputsHeight, nbOutputs, resizeMode),
      Cell_Frame_CUDA(name, nbOutputs)
{
    // ctor
}

void N2D2::ResizeCell_Frame_CUDA::BilinearInterpolation(const int out_size,
                                                        const int in_size,
                                                        const float scale,
                                                        CudaTensor<unsigned int>& LowIndex,
                                                        CudaTensor<unsigned int>& HightIndex,
                                                        CudaTensor<Float_T>& Interpolation) 
{
  LowIndex(out_size) = 0;
  HightIndex(out_size) = 0;

  for (int i = out_size - 1; i >= 0; --i) {
    const float in = i * scale;
    LowIndex(i) = (unsigned int) in;
    HightIndex(i) = std::min((int) in + 1, in_size - 1);
    Interpolation(i) = in - LowIndex(i);
  }

}

void N2D2::ResizeCell_Frame_CUDA::initialize()
{
    
    for(unsigned int input = 1; input < mInputs.size(); ++input)
        if (mInputs[input].dimX() != mInputs[0].dimX() || 
            mInputs[input].dimY() != mInputs[0].dimY()) 
            throw std::runtime_error("Input must have the same dimensions in ResizeCell_Frame_CUDA " + mName);

    if(mResizeMode == BilinearTF)
    {
        const unsigned int outputDimX = mOutputs.dimX();
        const unsigned int outputDimY = mOutputs.dimY();
        const unsigned int inputDimX = mInputs[0].dimX();
        const unsigned int inputDimY = mInputs[0].dimY();

        mScaleX = mAlignCorners ? (inputDimX - 1) / (float) (outputDimX - 1)
                    : (inputDimX) / (float) (outputDimX);
                    
        mScaleY = mAlignCorners ? (inputDimY - 1) / (float) (outputDimY - 1)
                    : (inputDimY) / (float) (outputDimY);

        mYStrideLowIndex.resize({outputDimY + 1});
        mYStrideHightIndex.resize({outputDimY + 1});
        mYStrideInterpolation.resize({outputDimY + 1});

        mXStrideLowIndex.resize({outputDimX + 1});
        mXStrideHightIndex.resize({outputDimX + 1});
        mXStrideInterpolation.resize({outputDimX + 1});

        // Compute the cached interpolation weights on the x and y dimensions.
        BilinearInterpolation(outputDimY, inputDimY, mScaleY, mYStrideLowIndex, mYStrideHightIndex, mYStrideInterpolation);
        mYStrideLowIndex.synchronizeHToD();
        mYStrideHightIndex.synchronizeHToD();
        mYStrideInterpolation.synchronizeHToD();

        BilinearInterpolation(outputDimX, inputDimX, mScaleX, mXStrideLowIndex, mXStrideHightIndex, mXStrideInterpolation);
        mXStrideLowIndex.synchronizeHToD();
        mXStrideHightIndex.synchronizeHToD();
        mXStrideInterpolation.synchronizeHToD();


    }
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    const unsigned int maxSize = (unsigned int)deviceProp.maxThreadsPerBlock;
    const unsigned int prefMultiple = (unsigned int)deviceProp.warpSize;

    const unsigned int groupSize = (mOutputs.dimX() * mOutputs.dimY() < maxSize)
                                       ? mOutputs.dimX() * mOutputs.dimY()
                                       : maxSize;
    const unsigned int reqWidth = (unsigned int) ceilf((float) groupSize / (float) mOutputs.dimX());

    const unsigned int groupWidth = std::min(prefMultiple, reqWidth);

    for(unsigned int i = 0; i < mInputs.size(); ++i)
    {
        dim3 block_size = {(unsigned int)mInputs[i].dimZ(), 1, (unsigned int)mOutputs.dimB()};
        dim3 thread_size = {groupWidth, groupSize / groupWidth, 1};

        GPU_THREAD_GRID.push_back(thread_size);
        GPU_BLOCK_GRID.push_back(block_size);
    }
}

void N2D2::ResizeCell_Frame_CUDA::propagate(bool /*inference*/)
{
    mInputs.synchronizeHBasedToD();

    unsigned int outputOffset = 0;

    for(unsigned int k = 0; k < mInputs.size(); ++k)
    {
        std::shared_ptr<CudaDeviceTensor<Float_T> > input
            = cuda_device_tensor_cast_nocopy<Float_T>(mInputs[k]);

        cudaSResizeBilinearTF(  mOutputs.dimX(),
                                mOutputs.dimY(),
                                mInputs[k].dimZ(),
                                mOutputs.dimB(),
                                mInputs[k].dimX(),
                                mInputs[k].dimY(),
                                mYStrideLowIndex.getDevicePtr(),
                                mYStrideHightIndex.getDevicePtr(),
                                mYStrideInterpolation.getDevicePtr(),
                                mXStrideLowIndex.getDevicePtr(),
                                mXStrideHightIndex.getDevicePtr(),
                                mXStrideInterpolation.getDevicePtr(),
                                input->getDevicePtr(),
                                mOutputs.getDevicePtr() + outputOffset,
                                GPU_BLOCK_GRID[k],
                                GPU_THREAD_GRID[k]);

        outputOffset += mInputs[k].dimZ()*mOutputs.dimX()*mOutputs.dimY()*mOutputs.dimB();
    }

    Cell_Frame_CUDA::propagate();

    mDiffInputs.clearValid();

}

void N2D2::ResizeCell_Frame_CUDA::backPropagate()
{
    throw std::runtime_error(
        "ResizeCell_Frame_CUDA::backPropagate(): not implemented.");
}

void N2D2::ResizeCell_Frame_CUDA::update()
{
    // Nothing to update
}

void N2D2::ResizeCell_Frame_CUDA::checkGradient(double /*epsilon*/, double /*maxError*/)
{
}

#endif