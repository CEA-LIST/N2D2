/*
    (C) Copyright 2013 CEA LIST. All Rights Reserved.
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

#include "Cell/ConvCell_Frame_Kernels.hpp"

void N2D2::ConvCell_Frame_Kernels::forward(const Float_T* alpha,
                                           const Tensor4d<Float_T>& inputs,
                                           const Tensor4d
                                           <Float_T>& sharedSynapses,
                                           const Descriptor& desc,
                                           const Float_T* beta,
                                           Tensor4d<Float_T>& outputs,
                                           const Tensor2d<bool>& maps)
{
    const unsigned int oxSize
        = (unsigned int)((inputs.dimX() + 2 * desc.paddingX
                          - sharedSynapses.dimX() + desc.strideX)
                         / (double)desc.strideX);
    const unsigned int oySize
        = (unsigned int)((inputs.dimY() + 2 * desc.paddingY
                          - sharedSynapses.dimY() + desc.strideY)
                         / (double)desc.strideY);
    const bool subSample = (desc.subSampleX > 1 || desc.subSampleY > 1);

    if (subSample) {
        for (unsigned int index = 0; index < outputs.size(); ++index)
            outputs(index) *= (*beta);
    }

    const unsigned int size = inputs.dimB() * outputs.dimZ();

#if defined(_OPENMP) && _OPENMP >= 200805
#pragma omp parallel for collapse(2) if (size > 16)
#else
#pragma omp parallel for if (inputs.dimB() > 4 && size > 16)
#endif
    for (int batchPos = 0; batchPos < (int)inputs.dimB(); ++batchPos) {
        for (unsigned int output = 0; output < outputs.dimZ(); ++output) {
            for (unsigned int oy = 0; oy < oySize; ++oy) {
                for (unsigned int ox = 0; ox < oxSize; ++ox) {
                    const unsigned int sxMin = (unsigned int)std::max(
                        desc.paddingX - (int)(ox * desc.strideX), 0);
                    const unsigned int syMin = (unsigned int)std::max(
                        desc.paddingY - (int)(oy * desc.strideY), 0);
                    const unsigned int sxMax = Utils::clamp
                        <int>(inputs.dimX() + desc.paddingX - ox * desc.strideX,
                              0,
                              sharedSynapses.dimX());
                    const unsigned int syMax = Utils::clamp
                        <int>(inputs.dimY() + desc.paddingY - oy * desc.strideY,
                              0,
                              sharedSynapses.dimY());

                    const int ix = (int)(ox * desc.strideX) - desc.paddingX;
                    const int iy = (int)(oy * desc.strideY) - desc.paddingY;

                    // For each output, compute the weighted sum
                    Float_T weightedSum = 0.0;

                    for (unsigned int channel = 0; channel < inputs.dimZ();
                         ++channel) {
                        if (!maps.empty() && !maps(output, channel))
                            continue;

                        if (sxMin == 0 && syMin == 0 && sxMax == 3 && syMax
                                                                      == 3) {
                            // Loop unrolling for 3x3 conv
                            weightedSum
                                = weightedSum
                                  + sharedSynapses(0, 0, channel, output)
                                    * inputs(ix + 0, iy + 0, channel, batchPos)
                                  + sharedSynapses(1, 0, channel, output)
                                    * inputs(ix + 1, iy + 0, channel, batchPos)
                                  + sharedSynapses(2, 0, channel, output)
                                    * inputs(ix + 2, iy + 0, channel, batchPos)
                                  + sharedSynapses(0, 1, channel, output)
                                    * inputs(ix + 0, iy + 1, channel, batchPos)
                                  + sharedSynapses(1, 1, channel, output)
                                    * inputs(ix + 1, iy + 1, channel, batchPos)
                                  + sharedSynapses(2, 1, channel, output)
                                    * inputs(ix + 2, iy + 1, channel, batchPos)
                                  + sharedSynapses(0, 2, channel, output)
                                    * inputs(ix + 0, iy + 2, channel, batchPos)
                                  + sharedSynapses(1, 2, channel, output)
                                    * inputs(ix + 1, iy + 2, channel, batchPos)
                                  + sharedSynapses(2, 2, channel, output)
                                    * inputs(ix + 2, iy + 2, channel, batchPos);
                        } else {
                            for (unsigned int sy = syMin; sy < syMax; ++sy) {
                                for (unsigned int sx = sxMin; sx < sxMax;
                                     ++sx) {
                                    weightedSum += sharedSynapses(
                                                       sx, sy, channel, output)
                                                   * inputs(ix + sx,
                                                            iy + sy,
                                                            channel,
                                                            batchPos);
                                }
                            }
                        }
                    }

                    if (subSample) {
#pragma omp critical
                        outputs(ox / desc.subSampleX,
                                oy / desc.subSampleY,
                                output,
                                batchPos) += (*alpha) * weightedSum;
                    } else
                        outputs(ox, oy, output, batchPos)
                            = (*alpha) * weightedSum
                              + (*beta) * outputs(ox, oy, output, batchPos);
                }
            }
        }
    }
}

void N2D2::ConvCell_Frame_Kernels::forwardBias(const Float_T* alpha,
                                               const Tensor4d<Float_T>& bias,
                                               const Float_T* beta,
                                               Tensor4d<Float_T>& outputs)
{
    const unsigned int size = outputs.dimB() * outputs.dimZ();

#if defined(_OPENMP) && _OPENMP >= 200805
#pragma omp parallel for collapse(2) if (size > 16)
#else
#pragma omp parallel for if (outputs.dimB() > 4 && size > 16)
#endif
    for (int batchPos = 0; batchPos < (int)outputs.dimB(); ++batchPos) {
        for (unsigned int output = 0; output < outputs.dimZ(); ++output) {
            for (unsigned int oy = 0; oy < outputs.dimY(); ++oy) {
                for (unsigned int ox = 0; ox < outputs.dimX(); ++ox) {
                    outputs(ox, oy, output, batchPos) = (*alpha) * bias(output)
                        + (*beta) * outputs(ox, oy, output, batchPos);
                }
            }
        }
    }
}

void N2D2::ConvCell_Frame_Kernels::backwardData(const Float_T* alpha,
                                                const Tensor4d
                                                <Float_T>& sharedSynapses,
                                                const Tensor4d
                                                <Float_T>& diffInputs,
                                                const Descriptor& desc,
                                                const Float_T* beta,
                                                Tensor4d<Float_T>& diffOutputs,
                                                const Tensor2d<bool>& maps)
{
    const unsigned int oxStride
        = desc.strideX * (unsigned int)((diffOutputs.dimX() + 2 * desc.paddingX
                                         - sharedSynapses.dimX() + desc.strideX)
                                        / (double)desc.strideX);
    const unsigned int oyStride
        = desc.strideY * (unsigned int)((diffOutputs.dimY() + 2 * desc.paddingY
                                         - sharedSynapses.dimY() + desc.strideY)
                                        / (double)desc.strideY);

    const unsigned int size = diffOutputs.dimB() * diffOutputs.dimZ();

#if defined(_OPENMP) && _OPENMP >= 200805
#pragma omp parallel for collapse(2) if (size > 16)
#else
#pragma omp parallel for if (diffOutputs.dimB() > 4 && size > 16)
#endif
    for (int batchPos = 0; batchPos < (int)diffOutputs.dimB(); ++batchPos) {
        for (unsigned int channel = 0; channel < diffOutputs.dimZ();
             ++channel) {
            for (unsigned int iy = 0; iy < diffOutputs.dimY(); ++iy) {
                for (unsigned int ix = 0; ix < diffOutputs.dimX(); ++ix) {
                    const unsigned int ixPad = ix + desc.paddingX;
                    const unsigned int iyPad = iy + desc.paddingY;
                    const unsigned int sxMax
                        = std::min(sharedSynapses.dimX(), ixPad + 1);
                    const unsigned int syMax
                        = std::min(sharedSynapses.dimY(), iyPad + 1);

                    Float_T gradient = 0.0;

                    for (unsigned int sy = iyPad % desc.strideY,
                                      sx0 = ixPad % desc.strideX;
                         sy < syMax;
                         sy += desc.strideY) {
                        if (iyPad >= oyStride + sy)
                            continue;

                        for (unsigned int sx = sx0; sx < sxMax;
                             sx += desc.strideX) {
                            // Border conditions
                            if (ixPad >= oxStride + sx)
                                continue;

                            // Output node coordinates
                            const unsigned int ox = (ixPad - sx) / desc.strideX;
                            const unsigned int oy = (iyPad - sy) / desc.strideY;

                            for (unsigned int output = 0;
                                 output < diffInputs.dimZ();
                                 ++output) {
                                if (!maps.empty() && !maps(output, channel))
                                    continue;

                                gradient
                                    += sharedSynapses(sx, sy, channel, output)
                                       * diffInputs(ox / desc.subSampleX,
                                                    oy / desc.subSampleY,
                                                    output,
                                                    batchPos);
                            }
                        }
                    }

                    diffOutputs(ix, iy, channel, batchPos)
                        = (*alpha) * gradient
                          + (*beta) * diffOutputs(ix, iy, channel, batchPos);
                }
            }
        }
    }
}

void N2D2::ConvCell_Frame_Kernels::backwardFilter(const Float_T* alpha,
                                                  const Tensor4d
                                                  <Float_T>& inputs,
                                                  const Tensor4d
                                                  <Float_T>& diffInputs,
                                                  const Descriptor& desc,
                                                  const Float_T* beta,
                                                  Tensor4d
                                                  <Float_T>& diffSharedSynapses,
                                                  const Tensor2d<bool>& maps)
{
    const unsigned int oxSize
        = (unsigned int)((inputs.dimX() + 2 * desc.paddingX
                          - diffSharedSynapses.dimX() + desc.strideX)
                         / (double)desc.strideX);
    const unsigned int oySize
        = (unsigned int)((inputs.dimY() + 2 * desc.paddingY
                          - diffSharedSynapses.dimY() + desc.strideY)
                         / (double)desc.strideY);

    const unsigned int size = diffInputs.dimZ() * inputs.dimZ();

#if defined(_OPENMP) && _OPENMP >= 200805
#pragma omp parallel for collapse(2) if (size > 16)
#else
#pragma omp parallel for if (diffInputs.dimZ() > 4 && size > 16)
#endif
    for (int output = 0; output < (int)diffInputs.dimZ(); ++output) {
        for (unsigned int channel = 0; channel < inputs.dimZ(); ++channel) {
            if (!maps.empty() && !maps(output, channel))
                continue;

            for (unsigned int sy = 0; sy < diffSharedSynapses.dimY(); ++sy) {
                for (unsigned int sx = 0; sx < diffSharedSynapses.dimX();
                     ++sx) {
                    const unsigned int oxMin = (unsigned int)std::max(
                        (int)std::ceil((desc.paddingX - (int)sx)
                                       / (double)desc.strideX),
                        0);
                    const unsigned int oyMin = (unsigned int)std::max(
                        (int)std::ceil((desc.paddingY - (int)sy)
                                       / (double)desc.strideY),
                        0);
                    const unsigned int oxMax = std::min(
                        (unsigned int)std::ceil((inputs.dimX() + desc.paddingX
                                                 - sx) / (double)desc.strideX),
                        oxSize);
                    const unsigned int oyMax = std::min(
                        (unsigned int)std::ceil((inputs.dimY() + desc.paddingY
                                                 - sy) / (double)desc.strideY),
                        oySize);

                    Float_T gradient = 0.0;

                    for (unsigned int batchPos = 0; batchPos < inputs.dimB();
                         ++batchPos) {
                        for (unsigned int oy = oyMin; oy < oyMax; ++oy) {
                            for (unsigned int ox = oxMin; ox < oxMax; ++ox) {
                                const unsigned int ix
                                    = (int)(ox * desc.strideX + sx)
                                      - desc.paddingX;
                                const unsigned int iy
                                    = (int)(oy * desc.strideY + sy)
                                      - desc.paddingY;

                                gradient += inputs(ix, iy, channel, batchPos)
                                            * diffInputs(ox / desc.subSampleX,
                                                         oy / desc.subSampleY,
                                                         output,
                                                         batchPos);
                            }
                        }
                    }

                    diffSharedSynapses(sx, sy, channel, output)
                        = (*alpha) * gradient
                          + (*beta)
                            * diffSharedSynapses(sx, sy, channel, output);
                }
            }
        }
    }
}

void N2D2::ConvCell_Frame_Kernels::backwardBias(const Float_T* alpha,
                                                const Tensor4d
                                                <Float_T>& diffInputs,
                                                const Float_T* beta,
                                                Tensor4d<Float_T>& diffBias)
{
#pragma omp parallel for if (diffBias.dimZ() > 16)
    for (int output = 0; output < (int)diffBias.dimZ(); ++output) {
        Float_T sum = 0.0;

        for (unsigned int batchPos = 0; batchPos < diffInputs.dimB();
             ++batchPos) {
            for (unsigned int oy = 0; oy < diffInputs.dimY(); ++oy) {
                for (unsigned int ox = 0; ox < diffInputs.dimX(); ++ox)
                    sum += diffInputs(ox, oy, output, batchPos);
            }
        }

        diffBias(output) = (*alpha) * sum + (*beta) * diffBias(output);
    }
}
