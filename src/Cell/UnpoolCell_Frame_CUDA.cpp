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

#ifdef CUDA

#include "Cell/UnpoolCell_Frame_CUDA.hpp"

N2D2::Registrar<N2D2::UnpoolCell>
N2D2::UnpoolCell_Frame_CUDA::mRegistrar("Frame_CUDA",
                                        N2D2::UnpoolCell_Frame_CUDA::create);

N2D2::UnpoolCell_Frame_CUDA::UnpoolCell_Frame_CUDA(const std::string& name,
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
      Cell_Frame_CUDA(name, nbOutputs, activation),
      mPoolDesc(NULL)
{
    // ctor
    const PoolCell_Frame_Kernels::Descriptor poolDesc(poolWidth,
                                                      poolHeight,
                                                      strideX,
                                                      strideY,
                                                      paddingX,
                                                      paddingY);

    CHECK_CUDA_STATUS(cudaMalloc((void**)&mPoolDesc, sizeof(poolDesc)));
    CHECK_CUDA_STATUS(cudaMemcpy(mPoolDesc,
                                 &poolDesc,
                                 sizeof(poolDesc),
                                 cudaMemcpyHostToDevice));
}

void N2D2::UnpoolCell_Frame_CUDA::initialize()
{
    if (mArgMax.size() == 0)
        throw std::runtime_error("ArgMax missing for UnpoolCell " + mName);
    else if (mArgMax.size() != mInputs.size()) {
        throw std::runtime_error("Wrong number of ArgMax tensors for "
                                 "UnpoolCell " + mName);
    }

    unsigned int offset = 0;

    for (unsigned int k = 0, size = mInputs.size(); k < size; ++k) {
        if (mInputs[k].size() == 0)
            throw std::runtime_error("Zero-sized input for UnpoolCell "
                                     + mName);

        mInputMap.push_back(NULL);

        if (!mMaps.empty()) {
            const Tensor2d<bool> inputMap = mMaps.rows(offset,
                                                       mInputs[k].dimZ());

            std::vector<char> inputMapData;
            std::copy(inputMap.begin(), inputMap.end(),
                      std::back_inserter(inputMapData));

            CHECK_CUDA_STATUS(cudaMalloc(&mInputMap[k],
                                         inputMapData.size() * sizeof(char)));
            CHECK_CUDA_STATUS(cudaMemcpy(mInputMap[k],
                                         &inputMapData[0],
                                         inputMapData.size() * sizeof(char),
                                         cudaMemcpyHostToDevice));
        }

        offset += mInputs[k].dimZ();
    }
}

void N2D2::UnpoolCell_Frame_CUDA::propagate(bool /*inference*/)
{
    mInputs.synchronizeHBasedToD();

    const float alpha = 1.0f;
    float beta = 0.0f;

    for (unsigned int k = 0, size = mInputs.size(); k < size; ++k) {
        if (k > 0)
            beta = 1.0;

        if (mPooling == Max) {
            cudaSPoolBackwardMax(alpha,
                                 mInputs[k].getDevicePtr(),
                                 mInputs[k].dimZ(),
                                 mInputs[k].dimY(),
                                 mInputs[k].dimX(),
                                 mInputs[k].dimB(),
                                 mPoolDesc,
                                 beta,
                                 mOutputs.getDevicePtr(),
                                 mOutputs.dimZ(),
                                 mOutputs.dimY(),
                                 mOutputs.dimX(),
                                 mArgMax[k].getDevicePtr(),
                                 mInputMap[k]);
        }
        else {
            cudaSPoolBackwardAverage(alpha,
                                     mInputs[k].getDevicePtr(),
                                     mInputs[k].dimZ(),
                                     mInputs[k].dimY(),
                                     mInputs[k].dimX(),
                                     mInputs[k].dimB(),
                                     mPoolDesc,
                                     beta,
                                     mOutputs.getDevicePtr(),
                                     mOutputs.dimZ(),
                                     mOutputs.dimY(),
                                     mOutputs.dimX(),
                                     true,
                                     mInputMap[k]);
        }
    }

    Cell_Frame_CUDA::propagate();
    mDiffInputs.clearValid();
}

void N2D2::UnpoolCell_Frame_CUDA::backPropagate()
{
    if (mDiffOutputs.empty())
        return;

    Cell_Frame_CUDA::backPropagate();

    const Float_T alpha = 1.0;

    for (unsigned int k = 0, size = mInputs.size(); k < size; ++k) {
        const Float_T beta = (mDiffOutputs[k].isValid()) ? 1.0 : 0.0;

        if (mPooling == Max) {
            cudaSPoolForwardMax(alpha,
                                mDiffInputs.getDevicePtr(),
                                mDiffInputs.dimZ(),
                                mDiffInputs.dimY(),
                                mDiffInputs.dimX(),
                                mDiffInputs.dimB(),
                                mPoolDesc,
                                beta,
                                mDiffOutputs[k].getDevicePtr(),
                                mDiffOutputs[k].dimZ(),
                                mDiffOutputs[k].dimY(),
                                mDiffOutputs[k].dimX(),
                                mArgMax[k].getDevicePtr(),
                                true,
                                mInputMap[k]);
        }
        else {
            cudaSPoolForwardAverage(alpha,
                                    mDiffInputs.getDevicePtr(),
                                    mDiffInputs.dimZ(),
                                    mDiffInputs.dimY(),
                                    mDiffInputs.dimX(),
                                    mDiffInputs.dimB(),
                                    mPoolDesc,
                                    beta,
                                    mDiffOutputs[k].getDevicePtr(),
                                    mDiffOutputs[k].dimZ(),
                                    mDiffOutputs[k].dimY(),
                                    mDiffOutputs[k].dimX(),
                                    true,
                                    mInputMap[k]);
        }

        mDiffOutputs[k].setValid();
    }

    mDiffOutputs.synchronizeDToHBased();
}

void N2D2::UnpoolCell_Frame_CUDA::update()
{
}

void N2D2::UnpoolCell_Frame_CUDA::checkGradient(double epsilon, double maxError)
{
    GradientCheck gc(epsilon, maxError);
    gc.initialize(mInputs,
                  mOutputs,
                  mDiffInputs,
                  std::bind(&UnpoolCell_Frame_CUDA::propagate, this, false),
                  std::bind(&UnpoolCell_Frame_CUDA::backPropagate, this),
                  (mPooling == Max));

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

void N2D2::UnpoolCell_Frame_CUDA::addArgMax(
    Interface<PoolCell_Frame_Kernels::ArgMax>* argMax)
{
    for (unsigned int k = 0; k < argMax->size(); ++k)
        mArgMax.push_back(&(*argMax)[k]);
}

N2D2::UnpoolCell_Frame_CUDA::~UnpoolCell_Frame_CUDA()
{
    if (mPoolDesc != NULL)
        CHECK_CUDA_STATUS(cudaFree(mPoolDesc));

    for (unsigned int k = 0, size = mInputMap.size(); k < size; ++k) {
        if (mInputMap[k] != NULL) {
            CHECK_CUDA_STATUS(cudaFree(mInputMap[k]));
            mInputMap[k] = NULL;
        }
    }
}

#endif
