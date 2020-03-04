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

#include "Cell/ResizeCell_Frame.hpp"
#include <algorithm>
#include <cassert>
#include <iostream>
#include <stdexcept>
#include <string>

#include "DeepNet.hpp"
#include "GradientCheck.hpp"

N2D2::Registrar<N2D2::ResizeCell>
N2D2::ResizeCell_Frame::mRegistrar("Frame",
                                       N2D2::ResizeCell_Frame::create);

N2D2::ResizeCell_Frame::ResizeCell_Frame(const DeepNet& deepNet, const std::string& name,
                                         unsigned int outputsWidth,
                                         unsigned int outputsHeight,
                                         unsigned int nbOutputs,
                                         ResizeMode resizeMode)
    : Cell(deepNet, name, nbOutputs),
      ResizeCell(deepNet, name, outputsWidth, outputsHeight, nbOutputs, resizeMode),
      Cell_Frame<Float_T>(deepNet, name, nbOutputs)
{
    // ctor
}

void N2D2::ResizeCell_Frame::BilinearInterpolation(const int out_size,
                                                    const int in_size,
                                                    const Float_T scale,
                                                    ResizeCell_Frame_Kernels::PreComputed* interpolation)
{
  interpolation[out_size].low_index = 0;
  interpolation[out_size].high_index = 0;
  for (int i = out_size - 1; i >= 0; --i) {
    const Float_T in = i * scale;
    interpolation[i].low_index = static_cast<int> (in);
    interpolation[i].high_index = std::min(interpolation[i].low_index + 1, in_size - 1);
    interpolation[i].interpolation = in - interpolation[i].low_index;
  }

}

void N2D2::ResizeCell_Frame::initialize()
{
    for(unsigned int input = 1; input < mInputs.size(); ++input) {
        if (mInputs[input].dimX() != mInputs[0].dimX() ||
            mInputs[input].dimY() != mInputs[0].dimY())
        {
            throw std::runtime_error("Input must have the same dimensions in ResizeCell_Frame " + mName);
        }
    }

    if(mResizeMode == BilinearTF)
    {
        const unsigned int outputDimX = mOutputs.dimX();
        const unsigned int outputDimY = mOutputs.dimY();
        const unsigned int inputDimX = mInputs[0].dimX();
        const unsigned int inputDimY = mInputs[0].dimY();

        mScaleX = (mAlignCorners && outputDimX > 1) ?
                      (inputDimX - 1) / (Float_T) (outputDimX - 1)
                    : (inputDimX) / (Float_T) (outputDimX);

        mScaleY = (mAlignCorners && outputDimY > 1) ?
                      (inputDimY - 1) / (Float_T) (outputDimY - 1)
                    : (inputDimY) / (Float_T) (outputDimY);

        mYStride.resize(outputDimY + 1);
        mXStride.resize(outputDimX + 1);

        // Compute the cached interpolation weights on the x and y dimensions.
        BilinearInterpolation(outputDimY, inputDimY, mScaleY, mYStride.data());
        BilinearInterpolation(outputDimX, inputDimX, mScaleX, mXStride.data());
    }
    else if (mResizeMode == Bilinear) {
        throw std::runtime_error("ResizeCell_Frame: Bilinear interpolation is not yet implemented.");
    }
}

void N2D2::ResizeCell_Frame::propagate(bool inference)
{
    mInputs.synchronizeDBasedToH();

    switch(mResizeMode) {
        case Bilinear:
            throw std::runtime_error("ResizeCell_Frame: Bilinear interpolation is not yet implemented.");
        case BilinearTF:
            propagateBilinearTF(inference);
            break;
        case NearestNeighbor:
            propagateNearestNeighbor(inference);
            break;
        default:
            throw std::runtime_error("ResizeCell_Frame: Unknown resize mode.");
    }

    Cell_Frame<Float_T>::propagate();
    mDiffInputs.clearValid();
}

void N2D2::ResizeCell_Frame::propagateBilinearTF(bool /*inference*/)
{
    const Tensor<Float_T>& input = tensor_cast<Float_T>(mInputs[0]);

#pragma omp parallel for if (mOutputs.dimB() > 3)
    for (int batchPos = 0; batchPos < (int)mOutputs.dimB(); ++batchPos) {
#pragma omp parallel for if (mOutputs.size() > 2)
        for(int oy = 0; oy < (int)mOutputs.dimY(); ++oy) {
            for(std::size_t ox = 0; ox < mOutputs.dimX(); ++ox) {
                for(std::size_t channel = 0; channel < mOutputs.dimZ(); ++channel) {
                    const Float_T top_left = input( mXStride[ox].low_index,
                                                    mYStride[oy].low_index,
                                                    channel,
                                                    batchPos);
                    const Float_T top_right = input( mXStride[ox].high_index,
                                                        mYStride[oy].low_index,
                                                        channel,
                                                        batchPos);
                    const Float_T bottom_left = input( mXStride[ox].low_index,
                                                        mYStride[oy].high_index,
                                                        channel,
                                                        batchPos);
                    const Float_T bottom_right = input( mXStride[ox].high_index,
                                                        mYStride[oy].high_index,
                                                        channel,
                                                        batchPos);

                    const Float_T top = top_left
                                            + (top_right - top_left) * mXStride[ox].interpolation;
                    const Float_T bottom = bottom_left
                                            + (bottom_right - bottom_left) * mXStride[ox].interpolation;

                    mOutputs(ox, oy, channel, batchPos)
                        = top + (bottom - top) * mYStride[oy].interpolation;
                }
            }
        }
    }
}

void N2D2::ResizeCell_Frame::propagateNearestNeighbor(bool /*inference*/) {
    assert(mInputs.size() == 1);
    nearestNeighbor(tensor_cast_nocopy<Float_T>(mInputs[0]), mOutputs);
}

void N2D2::ResizeCell_Frame::backPropagate()
{
    if (mDiffOutputs.empty() || !mDiffInputs.isValid())
        return;

    Cell_Frame<Float_T>::backPropagate();

    switch(mResizeMode) {
        case Bilinear:
            throw std::runtime_error("ResizeCell_Frame: Bilinear interpolation is not yet implemented.");
        case BilinearTF:
            backPropagateBilinearTF();
            break;
        case NearestNeighbor:
            backPropagateNearestNeighbor();
            break;
        default:
            throw std::runtime_error("ResizeCell_Frame: Unknown resize mode.");
    }

    mDiffOutputs.setValid();
    mDiffOutputs.synchronizeHToD();
}

void N2D2::ResizeCell_Frame::backPropagateBilinearTF()
{
    Tensor<Float_T> diffOutput
            = tensor_cast_nocopy<Float_T>(mDiffOutputs[0]);
    for (std::size_t idx = 0; idx < diffOutput.size(); ++idx) {
        diffOutput(idx) = 0.0f;
    }

    #pragma omp parallel for if (mInputs.dimB() > 1)
    for (int batchPos = 0; batchPos < (int)mInputs.dimB(); ++batchPos) {
        for (std::size_t oy = 0; oy < mDiffInputs.dimY(); ++oy) {
            const Float_T in_y = oy * mScaleY;
            const int top_y_index = (int)(floorf(in_y));
            const int bottom_y_index =
                std::min((int)(ceilf(in_y)), (int) (mInputs[0].dimY() - 1) ) ;
            const Float_T y_lerp = in_y - top_y_index;
            const Float_T inverse_y_lerp = (1.0f - y_lerp);

            for (std::size_t ox = 0; ox < mDiffInputs.dimX(); ++ox) {
                const Float_T in_x = ox * mScaleX;
                const int left_x_index = (int)(floorf(in_x));
                const int right_x_index = std::min((int)(ceilf(in_x)), (int)(mInputs[0].dimX() - 1));
                const Float_T x_lerp = in_x - left_x_index;
                const Float_T inverse_x_lerp = (1.0f - x_lerp);

                #pragma omp parallel for if (mDiffInputs.dimZ() > 2)
                for (int channel = 0; channel < (int)mDiffInputs.dimZ(); ++channel) {
                    diffOutput(left_x_index,
                                    top_y_index,
                                    channel,
                                    batchPos) += mDiffInputs(ox, oy, channel, batchPos) * inverse_y_lerp * inverse_x_lerp;
                    diffOutput(right_x_index,
                                    top_y_index,
                                    channel,
                                    batchPos) += mDiffInputs(ox, oy, channel, batchPos) * inverse_y_lerp * x_lerp;
                    diffOutput(left_x_index,
                                    bottom_y_index,
                                    channel,
                                    batchPos) += mDiffInputs(ox, oy, channel, batchPos) * y_lerp * inverse_x_lerp;
                    diffOutput(right_x_index,
                                    bottom_y_index,
                                    channel,
                                    batchPos) += mDiffInputs(ox, oy, channel, batchPos) * y_lerp * x_lerp;

                }
            }
        }
    }

    mDiffOutputs[0] = diffOutput;
}

void N2D2::ResizeCell_Frame::backPropagateNearestNeighbor() {
    assert(mDiffOutputs.size() == 1);
    Tensor<Float_T> diffOutput = tensor_cast_nocopy<Float_T>(mDiffOutputs[0]);
    nearestNeighbor(mDiffInputs, diffOutput);

    mDiffOutputs[0] = diffOutput;
}

void N2D2::ResizeCell_Frame::update()
{
    // Nothing to update
}

void N2D2::ResizeCell_Frame::checkGradient(double epsilon, double maxError)
{
    GradientCheck<Float_T> gc(epsilon, maxError);
    gc.initialize(mInputs,
                  mOutputs,
                  mDiffInputs,
                  std::bind(&ResizeCell_Frame::propagate, this, false),
                  std::bind(&ResizeCell_Frame::backPropagate, this));

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

void N2D2::ResizeCell_Frame::nearestNeighbor(const Tensor<Float_T>& inputs, Tensor<Float_T>& outputs) {
    assert(inputs.dimB() == outputs.dimB());
    assert(inputs.dimZ() == outputs.dimZ());
    assert(inputs.nbDims() == outputs.nbDims());
    assert(inputs.nbDims() == 4);

    const Float_T multy = ((Float_T) inputs.dimY())/((Float_T) outputs.dimY());
    const Float_T multx = ((Float_T) inputs.dimX())/((Float_T) outputs.dimX());

    #pragma omp parallel for if (outputs.dimB() > 1)
    for(int batch = 0; batch < (int)outputs.dimB(); batch++) {
        #pragma omp parallel for if (outputs.dimZ() > 3)
        for(int channel = 0; channel < (int)outputs.dimZ(); channel++) {
            for(std::size_t oy = 0; oy < outputs.dimY(); oy++) {
                for(std::size_t ox = 0; ox < outputs.dimX(); ox++) {
                    const std::size_t iy = static_cast<std::size_t>(oy*multy);
                    const std::size_t ix = static_cast<std::size_t>(ox*multx);

                    outputs(ox, oy, channel, batch) = inputs(ix, iy, channel, batch);
                }
            }
        }
    }
}
