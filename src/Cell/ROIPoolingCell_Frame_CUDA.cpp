/*
    (C) Copyright 2015 CEA LIST. All Rights Reserved.
    Contributor(s): Olivier BICHLER (olivier.bichler@cea.fr)
                    David BRIAND (david.briand@cea.fr)
                    
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

#include "Cell/ROIPoolingCell_Frame_CUDA.hpp"

N2D2::Registrar<N2D2::ROIPoolingCell>
N2D2::ROIPoolingCell_Frame_CUDA::mRegistrar("Frame_CUDA",
                                       N2D2::ROIPoolingCell_Frame_CUDA::create);

N2D2::ROIPoolingCell_Frame_CUDA::ROIPoolingCell_Frame_CUDA(const std::string& name,
                                                 StimuliProvider& sp,
                                                 unsigned int outputsWidth,
                                                 unsigned int outputsHeight,
                                                 unsigned int nbOutputs,
                                                 ROIPooling pooling)
    : Cell(name, nbOutputs),
      ROIPoolingCell(name, sp, outputsWidth, outputsHeight, nbOutputs, pooling),
      Cell_Frame_CUDA(name, nbOutputs)
{
    // ctor
    mInputs.matchingDimB(false);
    mDiffOutputs.matchingDimB(false);
}

void N2D2::ROIPoolingCell_Frame_CUDA::initialize()
{
    if (mInputs.size() < 2) {
        throw std::runtime_error("At least two inputs are required for"
                                 " ROIPoolingCell " + mName);
    }

    if (mInputs[0].dimX() * mInputs[0].dimY() * mInputs[0].dimZ() != 4) {
        throw std::runtime_error("The first input must have a XYZ size of 4 for"
                                 " ROIPoolingCell " + mName);
    }

    unsigned int kRef = 1;
    const unsigned int inputBatch = mInputs[kRef].dimB();
    unsigned int nbChannels = 0;

    for (unsigned int k = kRef, size = mInputs.size(); k < size; ++k) {
        if (mInputs[k].size() == 0) {
            throw std::runtime_error("Zero-sized input for ROIPoolingCell "
                                      + mName);
        }

        if (mInputs[k].dimB() != inputBatch) {
            throw std::runtime_error("Input batch size must match for"
                                     " ROIPoolingCell" + mName);
        }

        if (mArgMax.size() == (k - 1)) {
            mArgMax.push_back(new CudaTensor4d<PoolCell_Frame_Kernels::ArgMax>(
                mOutputs.dimX(),
                mOutputs.dimY(),
                mOutputs.dimZ(),
                mOutputs.dimB()));
        }

        nbChannels += mInputs[k].dimZ();
    }

    if (nbChannels != mOutputs.dimZ()) {
        throw std::runtime_error("The number of output channels must match the "
            "total number of input channels for ROIPoolingCell" + mName);
    }
    mParentProposals = mInputs[kRef-1].dimB()/mInputs[kRef].dimB();
}

void N2D2::ROIPoolingCell_Frame_CUDA::propagate(bool /*inference*/)
{
    mInputs.synchronizeHBasedToD();

    const float alpha = 1.0f;
    float beta = 0.0f;

    unsigned int outputOffset = 0;
    unsigned int kFeatureMap = 1;

    for (unsigned int k = kFeatureMap, size = mInputs.size(); k < size; ++k) {

        if (mPooling == Max) {
            cudaSROIPoolingForwardMax(alpha,
                                      mInputs[0].getDevicePtr(),
                                      mInputs[0].dimB(),
                                      mStimuliProvider.getSizeY(),
                                      mStimuliProvider.getSizeX(),
                                      mInputs[k].getDevicePtr(),
                                      mInputs[k].dimZ(),
                                      mInputs[k].dimY(),
                                      mInputs[k].dimX(),
                                      mOutputs.dimB(),
                                      beta,
                                      mOutputs.getDevicePtr(),
                                      mOutputs.dimZ(),
                                      mOutputs.dimY(),
                                      mOutputs.dimX(),
                                      outputOffset,
                                      mArgMax[k-1].getDevicePtr());
        }
        else if (mPooling == Average) {
            cudaSROIPoolingForwardAverage(alpha,
                                          mInputs[0].getDevicePtr(),
                                          mInputs[0].dimB(),
                                          mStimuliProvider.getSizeY(),
                                          mStimuliProvider.getSizeX(),
                                          mInputs[k].getDevicePtr(),
                                          mInputs[k].dimZ(),
                                          mInputs[k].dimY(),
                                          mInputs[k].dimX(),
                                          mOutputs.dimB(),
                                          beta,
                                          mOutputs.getDevicePtr(),
                                          mOutputs.dimZ(),
                                          mOutputs.dimY(),
                                          mOutputs.dimX(),
                                          outputOffset);
        }
        else if (mPooling == Bilinear || mPooling == BilinearTF) {
            const Float_T xRatio = std::ceil(mStimuliProvider.getSizeX()
                                            / (Float_T) mInputs[k].dimX());
            const Float_T yRatio =  std::ceil(mStimuliProvider.getSizeY()
                                            / (Float_T)mInputs[k].dimY());
            Float_T xOffset = 0.0;
            Float_T yOffset = 0.0;

            if (mFlip) {
                xOffset = (mStimuliProvider.getSizeX() - 1) / xRatio
                            - (mInputs[k].dimX() - 1);
                yOffset = (mStimuliProvider.getSizeY() - 1) / yRatio
                            - (mInputs[k].dimY() - 1);
            }

            cudaSROIPoolingForwardBilinear(alpha,
                                                mInputs[kFeatureMap-1].getDevicePtr(),
                                                mInputs[kFeatureMap-1].dimB(),
                                                mStimuliProvider.getSizeY(),
                                                mStimuliProvider.getSizeX(),
                                                mInputs[k].getDevicePtr(),
                                                mInputs[k].dimZ(),
                                                mInputs[k].dimY(),
                                                mInputs[k].dimX(),
                                                mOutputs.dimB(),
                                                beta,
                                                mOutputs.getDevicePtr(),
                                                mOutputs.dimZ(),
                                                mOutputs.dimY(),
                                                mOutputs.dimX(),
                                                outputOffset,
                                                (mPooling == BilinearTF )
                                                    ? true: false,
                                                mIgnorePad,
                                                xOffset,
                                                yOffset,
                                                xRatio,
                                                yRatio);  
        }
        else {
            throw std::runtime_error("ROIPoolingCell_Frame_CUDA::propagate():"
                                     " only Max and Average pooling "
                                     "propagation are implemented");
        }

        outputOffset += mInputs[k].dimZ();
    }

    Cell_Frame_CUDA::propagate();
    mDiffInputs.clearValid();
}

void N2D2::ROIPoolingCell_Frame_CUDA::backPropagate()
{
    if (mDiffOutputs.empty())
        return;

    Cell_Frame_CUDA::backPropagate();

    const Float_T alpha = 1.0;
    const Float_T beta = 1.0;

    unsigned int outputOffset = 0;

    for (unsigned int k = 1, size = mInputs.size(); k < size; ++k) {
        if (!mDiffOutputs[k].isValid()) {
            mDiffOutputs[k].fill(0.0);
            mDiffOutputs[k].synchronizeHToD();
            mDiffOutputs[k].setValid();
        }

        if (mPooling == Max) {
            cudaSROIPoolingBackwardMax(alpha,
                                       mInputs[0].getDevicePtr(),
                                       mInputs[0].dimB(),
                                       mStimuliProvider.getSizeY(),
                                       mStimuliProvider.getSizeX(),
                                       mDiffInputs.getDevicePtr(),
                                       mDiffInputs.dimZ(),
                                       mDiffInputs.dimY(),
                                       mDiffInputs.dimX(),
                                       mOutputs.dimB(),
                                       outputOffset,
                                       beta,
                                       mDiffOutputs[k].getDevicePtr(),
                                       mDiffOutputs[k].dimZ(),
                                       mDiffOutputs[k].dimY(),
                                       mDiffOutputs[k].dimX(),
                                       mArgMax[k-1].getDevicePtr());
        }
        else if (mPooling == Average) {
            cudaSROIPoolingBackwardAverage(alpha,
                                           mInputs[0].getDevicePtr(),
                                           mInputs[0].dimB(),
                                           mStimuliProvider.getSizeY(),
                                           mStimuliProvider.getSizeX(),
                                           mDiffInputs.getDevicePtr(),
                                           mDiffInputs.dimZ(),
                                           mDiffInputs.dimY(),
                                           mDiffInputs.dimX(),
                                           mOutputs.dimB(),
                                           outputOffset,
                                           beta,
                                           mDiffOutputs[k].getDevicePtr(),
                                           mDiffOutputs[k].dimZ(),
                                           mDiffOutputs[k].dimY(),
                                           mDiffOutputs[k].dimX());
        }
        else {
            throw std::runtime_error("ROIPoolingCell_Frame_CUDA::"
                                     "backPropagate(): only Max and Average "
                                     "pooling back-propagation are "
                                     "implemented");
        }

        outputOffset += mInputs[k].dimZ();
    }

    mDiffOutputs.synchronizeDToHBased();
}

void N2D2::ROIPoolingCell_Frame_CUDA::update()
{
    // Nothing to update
}

void N2D2::ROIPoolingCell_Frame_CUDA::checkGradient(double epsilon,
                                                    double maxError)
{
    GradientCheck gc(epsilon, maxError);

    mInputs[0].setValid();
    gc.initialize(mInputs,
                  mOutputs,
                  mDiffInputs,
                  std::bind(&ROIPoolingCell_Frame_CUDA::propagate, this, false),
                  std::bind(&ROIPoolingCell_Frame_CUDA::backPropagate, this),
                  (mPooling == Max));
    mInputs[0].clearValid();

    if (!mDiffOutputs.empty()) {
        for (unsigned int in = 1; in < mInputs.size(); ++in) {
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

N2D2::ROIPoolingCell_Frame_CUDA::~ROIPoolingCell_Frame_CUDA()
{
    for (unsigned int k = 0, size = mArgMax.size(); k < size; ++k)
        delete &mArgMax[k];
}

#endif
