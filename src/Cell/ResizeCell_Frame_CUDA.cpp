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

#include "Cell/ResizeCell_Frame_CUDA.hpp"
#include "DeepNet.hpp"

#include <cudnn.h>
#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>

N2D2::Registrar<N2D2::ResizeCell>
N2D2::ResizeCell_Frame_CUDA::mRegistrar("Frame_CUDA",
                                       N2D2::ResizeCell_Frame_CUDA::create);

N2D2::ResizeCell_Frame_CUDA::ResizeCell_Frame_CUDA(const DeepNet& deepNet, 
                                                 const std::string& name,
                                                 unsigned int outputsWidth,
                                                 unsigned int outputsHeight,
                                                 unsigned int nbOutputs,
                                                 ResizeMode resizeMode)
    : Cell(deepNet, name, nbOutputs),
      ResizeCell(deepNet, name, outputsWidth, outputsHeight, nbOutputs, resizeMode),
      Cell_Frame_CUDA<Float_T>(deepNet, name, nbOutputs)
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
    for(unsigned int input = 1; input < mInputs.size(); ++input) {
        if (mInputs[input].dimX() != mInputs[0].dimX() ||
            mInputs[input].dimY() != mInputs[0].dimY())
        {
            throw std::runtime_error("Input must have the same dimensions in ResizeCell_Frame_CUDA " + mName);
        }
    }

    if(mResizeMode == BilinearTF)
    {
        const unsigned int outputDimX = mOutputs.dimX();
        const unsigned int outputDimY = mOutputs.dimY();
        const unsigned int inputDimX = mInputs[0].dimX();
        const unsigned int inputDimY = mInputs[0].dimY();

        mScaleX = (mAlignCorners && outputDimX > 1) ? (inputDimX - 1) / (float) (outputDimX - 1)
                    : (inputDimX) / (float) (outputDimX);

        mScaleY = (mAlignCorners && outputDimY > 1) ? (inputDimY - 1) / (float) (outputDimY - 1)
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
}

void N2D2::ResizeCell_Frame_CUDA::propagate(bool inference)
{
    mInputs.synchronizeHBasedToD();

    unsigned int outputOffset = 0;

    for(unsigned int k = 0; k < mInputs.size(); ++k)
    {
        std::shared_ptr<CudaDeviceTensor<Float_T> > input
            = cuda_device_tensor_cast_nocopy<Float_T>(mInputs[k]);

        switch(mResizeMode) {
            case Bilinear:
                throw std::runtime_error("ResizeCell_Frame_CUDA: Bilinear resize not supported yet.");
            case BilinearTF:
                cudaSResizeFWBilinearTF(CudaContext::getDeviceProp(), mOutputs.dimX(), mOutputs.dimY(), mInputs[k].dimZ(), mOutputs.dimB(),
                                        mInputs[k].dimX(), mInputs[k].dimY(),
                                        mYStrideLowIndex.getDevicePtr(), mYStrideHightIndex.getDevicePtr(),
                                        mYStrideInterpolation.getDevicePtr(), mXStrideLowIndex.getDevicePtr(),
                                        mXStrideHightIndex.getDevicePtr(), mXStrideInterpolation.getDevicePtr(),
                                        input->getDevicePtr(), mOutputs.getDevicePtr() + outputOffset);
                break;
            case NearestNeighbor:
                cudaSResizeFWNearestNeighbor(CudaContext::getDeviceProp(), input->getDevicePtr(), mInputs[k].dimX(), mInputs[k].dimY(),
                                             mOutputs.getDevicePtr() + outputOffset, mOutputs.dimX(), mOutputs.dimY(), 
                                             mOutputs.dimZ(), mOutputs.dimB());
                break;
            default:
                throw std::runtime_error("ResizeCell_Frame_CUDA: Unknown resize mode.");
        }

        outputOffset += mInputs[k].dimZ()*mOutputs.dimX()*mOutputs.dimY()*mOutputs.dimB();
    }

    Cell_Frame_CUDA<Float_T>::propagate(inference);

    mDiffInputs.clearValid();

}

void N2D2::ResizeCell_Frame_CUDA::backPropagate()
{
    if (mDiffOutputs.empty() || !mDiffInputs.isValid())
        return;

    Cell_Frame_CUDA<Float_T>::backPropagate();
    unsigned int diffInputOffset = 0;
    
    for(unsigned int k = 0; k < mInputs.size(); ++k)
    {

        std::shared_ptr<CudaDeviceTensor<Float_T> > diffOutput
            = cuda_device_tensor_cast_nocopy<Float_T>(mDiffOutputs[k]);

        diffOutput->fill(0.0f);

        switch(mResizeMode) {
            case Bilinear:
                throw std::runtime_error("ResizeCell_Frame_CUDA: Bilinear resize not supported yet.");
            case BilinearTF:
                cudaSResizeBWBilinearTF(CudaContext::getDeviceProp(), mDiffInputs.dimX(), mDiffInputs.dimY(), mDiffInputs.dimZ(),
                                        mDiffInputs.dimB(), mDiffOutputs[k].dimX(), mDiffOutputs[k].dimY(),
                                        mScaleX, mScaleY, mDiffInputs.getDevicePtr() + diffInputOffset,
                                        diffOutput->getDevicePtr());
                break;
            case NearestNeighbor:
                cudaSResizeBWNearestNeighbor(CudaContext::getDeviceProp(), mDiffInputs.getDevicePtr() + diffInputOffset, mDiffInputs.dimX(), mDiffInputs.dimY(), 
                                             diffOutput->getDevicePtr(), mDiffOutputs[k].dimX(), mDiffOutputs[k].dimY(),
                                             mDiffInputs.dimZ(), mDiffInputs.dimB());
                break;
            default:
                throw std::runtime_error("ResizeCell_Frame_CUDA: Unknown resize mode.");
        }

        mDiffOutputs[k].deviceTensor() = *diffOutput;
        mDiffOutputs[k].setValid();

        diffInputOffset += mDiffOutputs[k].dimZ()
                            *mDiffInputs.dimX()
                            *mDiffInputs.dimY()
                            *mDiffInputs.dimB();

    }

    mDiffOutputs.synchronizeDToHBased();

}

void N2D2::ResizeCell_Frame_CUDA::update()
{
    // Nothing to update
}

void N2D2::ResizeCell_Frame_CUDA::checkGradient(double /*epsilon*/, double /*maxError*/)
{
}

#endif
