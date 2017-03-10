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

#include "Cell/PoolCell_Frame_Kernels.hpp"

void N2D2::PoolCell_Frame_Kernels::forwardAverage(const Float_T* alpha,
                                                  const Tensor4d<Float_T>&
                                                  inputs,
                                                  const Descriptor& desc,
                                                  const Float_T* beta,
                                                  Tensor4d<Float_T>& outputs,
                                                  bool countIncludePadding,
                                                  const Tensor2d<bool>& maps)
{
    const unsigned int size = inputs.dimB() * outputs.dimZ();

#if defined(_OPENMP) && _OPENMP >= 200805
#pragma omp parallel for collapse(2) if (size > 16)
#else
#pragma omp parallel for if (inputs.dimB() > 4 && size > 16)
#endif
    for (int batchPos = 0; batchPos < (int)inputs.dimB(); ++batchPos) {
        for (unsigned int output = 0; output < outputs.dimZ(); ++output) {
            for (unsigned int oy = 0; oy < outputs.dimY(); ++oy) {
                for (unsigned int ox = 0; ox < outputs.dimX(); ++ox) {
                    const unsigned int sxMin = (unsigned int)std::max(
                        desc.paddingX - (int)(ox * desc.strideX), 0);
                    const unsigned int syMin = (unsigned int)std::max(
                        desc.paddingY - (int)(oy * desc.strideY), 0);
                    const unsigned int sxMax = Utils::clamp
                        <int>(inputs.dimX() + desc.paddingX - ox * desc.strideX,
                              0,
                              desc.poolWidth);
                    const unsigned int syMax = Utils::clamp
                        <int>(inputs.dimY() + desc.paddingY - oy * desc.strideY,
                              0,
                              desc.poolHeight);

                    const int ix = (int)(ox * desc.strideX) - desc.paddingX;
                    const int iy = (int)(oy * desc.strideY) - desc.paddingY;

                    // For each output, compute the pool value
                    Float_T poolValue = 0.0;
                    unsigned int poolCount = 0;

                    for (unsigned int channel = 0; channel < inputs.dimZ();
                         ++channel)
                    {
                        if (!maps.empty() && !maps(output, channel))
                            continue;

                        for (unsigned int sy = syMin; sy < syMax; ++sy) {
                            for (unsigned int sx = sxMin; sx < sxMax; ++sx) {
                                poolValue += inputs(ix + sx,
                                                    iy + sy,
                                                    channel,
                                                    batchPos);
                            }
                        }

                        poolCount += (countIncludePadding)
                            ? (desc.poolWidth * desc.poolHeight)
                            : (sxMax - sxMin)*(syMax - syMin);
                    }

                    outputs(ox, oy, output, batchPos)
                        = (*alpha) * ((poolCount > 0) ?
                                      (poolValue / poolCount) : 0.0)
                          + (*beta) * outputs(ox, oy, output, batchPos);
                }
            }
        }
    }
}

void N2D2::PoolCell_Frame_Kernels::forwardMax(const Float_T* alpha,
                                              const Tensor4d<Float_T>&
                                              inputs,
                                              const Descriptor& desc,
                                              const Float_T* beta,
                                              Tensor4d<Float_T>& outputs,
                                              Tensor4d<ArgMax>& argMax,
                                              bool useArgMax,
                                              const Tensor2d<bool>& maps)
{
    const unsigned int size = inputs.dimB() * outputs.dimZ();

#if defined(_OPENMP) && _OPENMP >= 200805
#pragma omp parallel for collapse(2) if (size > 16)
#else
#pragma omp parallel for if (inputs.dimB() > 4 && size > 16)
#endif
    for (int batchPos = 0; batchPos < (int)inputs.dimB(); ++batchPos) {
        for (unsigned int output = 0; output < outputs.dimZ(); ++output) {
            for (unsigned int oy = 0; oy < outputs.dimY(); ++oy) {
                for (unsigned int ox = 0; ox < outputs.dimX(); ++ox) {
                    const unsigned int sxMin = (unsigned int)std::max(
                        desc.paddingX - (int)(ox * desc.strideX), 0);
                    const unsigned int syMin = (unsigned int)std::max(
                        desc.paddingY - (int)(oy * desc.strideY), 0);
                    const unsigned int sxMax = Utils::clamp
                        <int>(inputs.dimX() + desc.paddingX - ox * desc.strideX,
                              0,
                              desc.poolWidth);
                    const unsigned int syMax = Utils::clamp
                        <int>(inputs.dimY() + desc.paddingY - oy * desc.strideY,
                              0,
                              desc.poolHeight);

                    const int ix = (int)(ox * desc.strideX) - desc.paddingX;
                    const int iy = (int)(oy * desc.strideY) - desc.paddingY;

                    Float_T poolValue = 0.0;

                    // For each output, compute the pool value
                    if (useArgMax) {
                        const ArgMax inputMax
                            = argMax(ox, oy, output, batchPos);

                        if (inputMax.valid) {
                            poolValue = inputs(inputMax.ix,
                                               inputMax.iy,
                                               inputMax.channel,
                                               batchPos);
                        }
                    }
                    else {
                        unsigned int ixMax = 0;
                        unsigned int iyMax = 0;
                        unsigned int channelMax = 0;
                        bool valid = false;

                        for (unsigned int channel = 0; channel < inputs.dimZ();
                             ++channel)
                        {
                            if (!maps.empty() && !maps(output, channel))
                                continue;

                            for (unsigned int sy = syMin; sy < syMax; ++sy) {
                                for (unsigned int sx = sxMin; sx < sxMax; ++sx)
                                {
                                    const Float_T value = inputs(ix + sx,
                                                                 iy + sy,
                                                                 channel,
                                                                 batchPos);

                                    if (!valid || value > poolValue) {
                                        poolValue = value;
                                        valid = true;

                                        ixMax = ix + sx;
                                        iyMax = iy + sy;
                                        channelMax = channel;
                                    }
                                }
                            }
                        }

                        argMax(ox, oy, output, batchPos)
                            = ArgMax(ixMax, iyMax, channelMax, valid);
                    }

                    outputs(ox, oy, output, batchPos)
                        = (*alpha) * poolValue
                          + (*beta) * outputs(ox, oy, output, batchPos);
                }
            }
        }
    }
}

void N2D2::PoolCell_Frame_Kernels::backwardAverage(const Float_T* alpha,
                                                   const Tensor4d
                                                   <Float_T>& diffInputs,
                                                   const Descriptor& desc,
                                                   const Float_T* beta,
                                                   Tensor4d<Float_T>&
                                                   diffOutputs,
                                                   bool countIncludePadding,
                                                   const Tensor2d<bool>& maps)
{
    if (!countIncludePadding) {
        throw std::runtime_error("PoolCell_Frame_Kernels::backwardAverage()"
            " exclude padding not implemented");
    }

    const unsigned int oxStride = desc.strideX * diffInputs.dimX();
    const unsigned int oyStride = desc.strideY * diffInputs.dimY();
    const unsigned int size = diffOutputs.dimB() * diffOutputs.dimZ();

    std::vector<unsigned int> poolChannelsCount(diffInputs.dimZ(), 0);

    for (unsigned int output = 0; output < diffInputs.dimZ(); ++output) {
        for (unsigned int channel = 0; channel < diffOutputs.dimZ(); ++channel)
            poolChannelsCount[output] += (maps.empty()
                                          || maps(output, channel));
    }

#if defined(_OPENMP) && _OPENMP >= 200805
#pragma omp parallel for collapse(2) if (size > 16)
#else
#pragma omp parallel for if (diffOutputs.dimB() > 4 && size > 16)
#endif
    for (int batchPos = 0; batchPos < (int)diffOutputs.dimB(); ++batchPos) {
        for (unsigned int channel = 0; channel < diffOutputs.dimZ();
             ++channel)
        {
            for (unsigned int iy = 0; iy < diffOutputs.dimY(); ++iy) {
                for (unsigned int ix = 0; ix < diffOutputs.dimX(); ++ix) {
                    const unsigned int ixPad = ix + desc.paddingX;
                    const unsigned int iyPad = iy + desc.paddingY;
                    const unsigned int sxMax
                        = std::min(desc.poolWidth, ixPad + 1);
                    const unsigned int syMax
                        = std::min(desc.poolHeight, iyPad + 1);

                    Float_T poolGradient = 0.0;

                    for (unsigned int sy = iyPad % desc.strideY,
                                      sx0 = ixPad % desc.strideX;
                         sy < syMax;
                         sy += desc.strideY)
                    {
                        if (iyPad >= oyStride + sy)
                            continue;

                        for (unsigned int sx = sx0; sx < sxMax;
                             sx += desc.strideX)
                        {
                            // Border conditions
                            if (ixPad >= oxStride + sx)
                                continue;

                            // Output node coordinates
                            const unsigned int ox = (ixPad - sx) / desc.strideX;
                            const unsigned int oy = (iyPad - sy) / desc.strideY;

                            for (unsigned int output = 0;
                                 output < diffInputs.dimZ();
                                 ++output)
                            {
                                if (!maps.empty() && !maps(output, channel))
                                    continue;

                                poolGradient += diffInputs(ox,
                                                           oy,
                                                           output,
                                                           batchPos)
                                                / poolChannelsCount[output];
                            }
                        }
                    }

                    const unsigned int poolCount
                        = desc.poolWidth * desc.poolHeight;

                    diffOutputs(ix, iy, channel, batchPos)
                        = (*alpha) * (poolGradient / poolCount)
                          + (*beta) * diffOutputs(ix, iy, channel, batchPos);
                }
            }
        }
    }
}

void N2D2::PoolCell_Frame_Kernels::backwardMax(const Float_T* alpha,
                                               const Tensor4d
                                               <Float_T>& diffInputs,
                                               const Descriptor& desc,
                                               const Float_T* beta,
                                               Tensor4d<Float_T>&
                                               diffOutputs,
                                               const Tensor4d<ArgMax>& argMax,
                                               const Tensor2d<bool>& maps)
{
    const unsigned int oxStride = desc.strideX * diffInputs.dimX();
    const unsigned int oyStride = desc.strideY * diffInputs.dimY();
    const unsigned int size = diffOutputs.dimB() * diffOutputs.dimZ();

#if defined(_OPENMP) && _OPENMP >= 200805
#pragma omp parallel for collapse(2) if (size > 16)
#else
#pragma omp parallel for if (diffOutputs.dimB() > 4 && size > 16)
#endif
    for (int batchPos = 0; batchPos < (int)diffOutputs.dimB(); ++batchPos) {
        for (unsigned int channel = 0; channel < diffOutputs.dimZ();
             ++channel)
        {
            for (unsigned int iy = 0; iy < diffOutputs.dimY(); ++iy) {
                for (unsigned int ix = 0; ix < diffOutputs.dimX(); ++ix) {
                    const unsigned int ixPad = ix + desc.paddingX;
                    const unsigned int iyPad = iy + desc.paddingY;
                    const unsigned int sxMax
                        = std::min(desc.poolWidth, ixPad + 1);
                    const unsigned int syMax
                        = std::min(desc.poolHeight, iyPad + 1);

                    Float_T poolGradient = 0.0;

                    for (unsigned int sy = iyPad % desc.strideY,
                                      sx0 = ixPad % desc.strideX;
                         sy < syMax;
                         sy += desc.strideY)
                    {
                        if (iyPad >= oyStride + sy)
                            continue;

                        for (unsigned int sx = sx0; sx < sxMax;
                             sx += desc.strideX)
                        {
                            // Border conditions
                            if (ixPad >= oxStride + sx)
                                continue;

                            // Output node coordinates
                            const unsigned int ox = (ixPad - sx) / desc.strideX;
                            const unsigned int oy = (iyPad - sy) / desc.strideY;

                            for (unsigned int output = 0;
                                 output < diffInputs.dimZ();
                                 ++output)
                            {
                                if (!maps.empty() && !maps(output, channel))
                                    continue;

                                const ArgMax inputMax
                                    = argMax(ox, oy, output, batchPos);

                                if (ix == inputMax.ix
                                    && iy == inputMax.iy
                                    && channel == inputMax.channel
                                    && inputMax.valid)
                                {
                                    poolGradient += diffInputs(ox,
                                                               oy,
                                                               output,
                                                               batchPos);
                                }
                            }
                        }
                    }

                    diffOutputs(ix, iy, channel, batchPos)
                        = (*alpha) * poolGradient
                          + (*beta) * diffOutputs(ix, iy, channel, batchPos);
                }
            }
        }
    }
}
