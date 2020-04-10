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
#include "containers/Tensor.hpp"
#include "third_party/half.hpp"
#include "utils/Utils.hpp"

template <class T>
void N2D2::ConvCell_Frame_Kernels::forward(const T* alpha,
                                           const Tensor<T>& inputs,
                                           const Tensor
                                           <T>& sharedSynapses,
                                           const Descriptor& desc,
                                           const T* beta,
                                           Tensor<T>& outputs,
                                           const Tensor<bool>& maps)
{
    const unsigned int oxSize
        = (unsigned int)((inputs.dimX() + desc.padding[0] + desc.padding[2]
                          - sharedSynapses.dimX() + desc.stride[0])
                         / (double)desc.stride[0]);
    const unsigned int oySize
        = (unsigned int)((inputs.dimY() + desc.padding[1] + desc.padding[3]
                          - sharedSynapses.dimY() + desc.stride[1])
                         / (double)desc.stride[1]);
    const bool subSample = (desc.subSample[0] > 1 || desc.subSample[1] > 1);

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
                        desc.padding[0] - (int)(ox * desc.stride[0]), 0);
                    const unsigned int syMin = (unsigned int)std::max(
                        desc.padding[1] - (int)(oy * desc.stride[1]), 0);
                    const unsigned int sxMax = Utils::clamp
                        <int>(inputs.dimX() + desc.padding[0] - ox * desc.stride[0],
                              0,
                              sharedSynapses.dimX());
                    const unsigned int syMax = Utils::clamp
                        <int>(inputs.dimY() + desc.padding[1] - oy * desc.stride[1],
                              0,
                              sharedSynapses.dimY());

                    const int ix = (int)(ox * desc.stride[0]) - desc.padding[0];
                    const int iy = (int)(oy * desc.stride[1]) - desc.padding[1];

                    // For each output, compute the weighted sum
                    T weightedSum(0.0);

                    for (unsigned int channel = 0; channel < inputs.dimZ();
                         ++channel) {
                        if (!maps.empty() && !maps(output, channel))
                            continue;

                        if (sxMin == 0 && syMin == 0
                            && sxMax == 3 && syMax == 3)
                        {
                            // Loop unrolling for 3x3 conv
                            weightedSum = weightedSum
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
#pragma omp critical(ConvCell_Frame_Kernels__forward)
                        outputs(ox / desc.subSample[0],
                                oy / desc.subSample[1],
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

template <class T>
void N2D2::ConvCell_Frame_Kernels::forwardBias(const T* alpha,
                                               const Tensor<T>& bias,
                                               const T* beta,
                                               Tensor<T>& outputs)
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

template <class T>
void N2D2::ConvCell_Frame_Kernels::backwardData(const T* alpha,
                                                const Tensor
                                                <T>& sharedSynapses,
                                                const Tensor
                                                <T>& diffInputs,
                                                const Descriptor& desc,
                                                const T* beta,
                                                Tensor<T>& diffOutputs,
                                                const Tensor<bool>& maps)
{
    const unsigned int oxStride
        = desc.stride[0] * (unsigned int)((diffOutputs.dimX() + desc.padding[0]
            + desc.padding[2] - sharedSynapses.dimX() + desc.stride[0])
                                        / (double)desc.stride[0]);
    const unsigned int oyStride
        = desc.stride[1] * (unsigned int)((diffOutputs.dimY() + desc.padding[1]
            + desc.padding[3] - sharedSynapses.dimY() + desc.stride[1])
                                        / (double)desc.stride[1]);
    const bool noSubSample = (desc.subSample[0] == 1 && desc.subSample[1] == 1);

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
                    const unsigned int ixPad = ix + desc.padding[0];
                    const unsigned int iyPad = iy + desc.padding[1];
                    unsigned int sxMin = ixPad % desc.stride[0]
                        + std::max<int>(ixPad - (ixPad % desc.stride[0])
                            - oxStride + desc.stride[0], 0);
                    unsigned int syMin = iyPad % desc.stride[1]
                        + std::max<int>(iyPad - (iyPad % desc.stride[1])
                            - oyStride + desc.stride[1], 0);
                    const unsigned int sxMax
                        = std::min<unsigned int>(sharedSynapses.dimX(), ixPad + 1);
                    const unsigned int syMax
                        = std::min<unsigned int>(sharedSynapses.dimY(), iyPad + 1);

                    T gradient(0.0);

                    for (unsigned int output = 0;
                            output < diffInputs.dimZ();
                            ++output)
                    {
                        if (!maps.empty() && !maps(output, channel))
                            continue;

                        if (noSubSample
                            && syMax - syMin == 3 && desc.stride[1] == 1
                            && sxMax - sxMin == 3 && desc.stride[0] == 1)
                        {
                            // Loop unrolling for 3x3 conv, with stride 1
                            const unsigned int ox = (ixPad - sxMin);
                            const unsigned int oy = (iyPad - syMin);

                            gradient = gradient
                                + sharedSynapses(sxMin + 0, syMin + 0, channel, output)
                                    * diffInputs((ox - 0),
                                                (oy - 0),
                                                output,
                                                batchPos)
                                + sharedSynapses(sxMin + 1, syMin + 0, channel, output)
                                    * diffInputs((ox - 1),
                                                (oy - 0),
                                                output,
                                                batchPos)
                                + sharedSynapses(sxMin + 2, syMin + 0, channel, output)
                                    * diffInputs((ox - 2),
                                                (oy - 0),
                                                output,
                                                batchPos)
                                + sharedSynapses(sxMin + 0, syMin + 1, channel, output)
                                    * diffInputs((ox - 0),
                                                (oy - 1),
                                                output,
                                                batchPos)
                                + sharedSynapses(sxMin + 1, syMin + 1, channel, output)
                                    * diffInputs((ox - 1),
                                                (oy - 1),
                                                output,
                                                batchPos)
                                + sharedSynapses(sxMin + 2, syMin + 1, channel, output)
                                    * diffInputs((ox - 2),
                                                (oy - 1),
                                                output,
                                                batchPos)
                                + sharedSynapses(sxMin + 0, syMin + 2, channel, output)
                                    * diffInputs((ox - 0),
                                                (oy - 2),
                                                output,
                                                batchPos)
                                + sharedSynapses(sxMin + 1, syMin + 2, channel, output)
                                    * diffInputs((ox - 1),
                                                (oy - 2),
                                                output,
                                                batchPos)
                                + sharedSynapses(sxMin + 2, syMin + 2, channel, output)
                                    * diffInputs((ox - 2),
                                                (oy - 2),
                                                output,
                                                batchPos);
                        }
                        else {
                            for (unsigned int sy = syMin; sy < syMax;
                                sy += desc.stride[1])
                            {
                                for (unsigned int sx = sxMin; sx < sxMax;
                                    sx += desc.stride[0])
                                {
                                    // Output node coordinates
                                    const unsigned int ox = (ixPad - sx) / desc.stride[0];
                                    const unsigned int oy = (iyPad - sy) / desc.stride[1];

                                    if (noSubSample) {
                                        gradient
                                            += sharedSynapses(sx, sy, channel, output)
                                                * diffInputs(ox,
                                                            oy,
                                                            output,
                                                            batchPos);
                                    }
                                    else {
                                        gradient
                                            += sharedSynapses(sx, sy, channel, output)
                                                * diffInputs(ox / desc.subSample[0],
                                                            oy / desc.subSample[1],
                                                            output,
                                                            batchPos);
                                    }
                                }
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

template <class T>
void N2D2::ConvCell_Frame_Kernels::backwardFilter(const T* alpha,
                                                  const Tensor
                                                  <T>& inputs,
                                                  const Tensor
                                                  <T>& diffInputs,
                                                  const Descriptor& desc,
                                                  const T* beta,
                                                  Tensor
                                                  <T>& diffSharedSynapses,
                                                  const Tensor<bool>& maps)
{
    const unsigned int oxSize
        = (unsigned int)((inputs.dimX() + desc.padding[0] + desc.padding[2]
                          - diffSharedSynapses.dimX() + desc.stride[0])
                         / (double)desc.stride[0]);
    const unsigned int oySize
        = (unsigned int)((inputs.dimY() + desc.padding[1] + desc.padding[3]
                          - diffSharedSynapses.dimY() + desc.stride[1])
                         / (double)desc.stride[1]);
    const bool noSubSample = (desc.subSample[0] == 1 && desc.subSample[1] == 1);

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
                        (int)std::ceil((desc.padding[0] - (int)sx)
                                       / (double)desc.stride[0]),
                        0);
                    const unsigned int oyMin = (unsigned int)std::max(
                        (int)std::ceil((desc.padding[1] - (int)sy)
                                       / (double)desc.stride[1]),
                        0);
                    const unsigned int oxMax = std::min(
                        (unsigned int)std::ceil((inputs.dimX() + desc.padding[0]
                                                 - sx) / (double)desc.stride[0]),
                        oxSize);
                    const unsigned int oyMax = std::min(
                        (unsigned int)std::ceil((inputs.dimY() + desc.padding[1]
                                                 - sy) / (double)desc.stride[1]),
                        oySize);

                    T gradient(0.0);

                    for (unsigned int batchPos = 0; batchPos < inputs.dimB();
                         ++batchPos) {
                        for (unsigned int oy = oyMin; oy < oyMax; ++oy) {
                            for (unsigned int ox = oxMin; ox < oxMax; ++ox) {
                                const unsigned int ix
                                    = (int)(ox * desc.stride[0] + sx)
                                      - desc.padding[0];
                                const unsigned int iy
                                    = (int)(oy * desc.stride[1] + sy)
                                      - desc.padding[1];

                                if (noSubSample) {
                                    gradient += inputs(ix, iy, channel, batchPos)
                                                * diffInputs(ox,
                                                            oy,
                                                            output,
                                                            batchPos);
                                }
                                else {
                                    gradient += inputs(ix, iy, channel, batchPos)
                                                * diffInputs(ox / desc.subSample[0],
                                                            oy / desc.subSample[1],
                                                            output,
                                                            batchPos);
                                }
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

template <class T>
void N2D2::ConvCell_Frame_Kernels::backwardBias(const T* alpha,
                                                const Tensor
                                                <T>& diffInputs,
                                                const T* beta,
                                                Tensor<T>& diffBias)
{
#pragma omp parallel for if (diffBias.dimZ() > 16)
    for (int output = 0; output < (int)diffBias.dimZ(); ++output) {
        T sum(0.0);

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

namespace N2D2 {
    template void ConvCell_Frame_Kernels::forward<half_float::half>(const half_float::half* alpha,
                                           const Tensor<half_float::half>& inputs,
                                           const Tensor
                                           <half_float::half>& sharedSynapses,
                                           const Descriptor& desc,
                                           const half_float::half* beta,
                                           Tensor<half_float::half>& outputs,
                                           const Tensor<bool>& maps);
    template void ConvCell_Frame_Kernels::forward<float>(const float* alpha,
                                           const Tensor<float>& inputs,
                                           const Tensor
                                           <float>& sharedSynapses,
                                           const Descriptor& desc,
                                           const float* beta,
                                           Tensor<float>& outputs,
                                           const Tensor<bool>& maps);
    template void ConvCell_Frame_Kernels::forward<double>(const double* alpha,
                                           const Tensor<double>& inputs,
                                           const Tensor
                                           <double>& sharedSynapses,
                                           const Descriptor& desc,
                                           const double* beta,
                                           Tensor<double>& outputs,
                                           const Tensor<bool>& maps);

    template void ConvCell_Frame_Kernels::forwardBias<half_float::half>(const half_float::half* alpha,
                                               const Tensor<half_float::half>& bias,
                                               const half_float::half* beta,
                                               Tensor<half_float::half>& outputs);
    template void ConvCell_Frame_Kernels::forwardBias<float>(const float* alpha,
                                               const Tensor<float>& bias,
                                               const float* beta,
                                               Tensor<float>& outputs);
    template void ConvCell_Frame_Kernels::forwardBias<double>(const double* alpha,
                                               const Tensor<double>& bias,
                                               const double* beta,
                                               Tensor<double>& outputs);

    template void ConvCell_Frame_Kernels::backwardData<half_float::half>(const half_float::half* alpha,
                                                const Tensor
                                                <half_float::half>& sharedSynapses,
                                                const Tensor
                                                <half_float::half>& diffInputs,
                                                const Descriptor& desc,
                                                const half_float::half* beta,
                                                Tensor<half_float::half>& diffOutputs,
                                                const Tensor<bool>& maps);
    template void ConvCell_Frame_Kernels::backwardData<float>(const float* alpha,
                                                const Tensor
                                                <float>& sharedSynapses,
                                                const Tensor
                                                <float>& diffInputs,
                                                const Descriptor& desc,
                                                const float* beta,
                                                Tensor<float>& diffOutputs,
                                                const Tensor<bool>& maps);
    template void ConvCell_Frame_Kernels::backwardData<double>(const double* alpha,
                                                const Tensor
                                                <double>& sharedSynapses,
                                                const Tensor
                                                <double>& diffInputs,
                                                const Descriptor& desc,
                                                const double* beta,
                                                Tensor<double>& diffOutputs,
                                                const Tensor<bool>& maps);

    template void ConvCell_Frame_Kernels::backwardFilter<half_float::half>(const half_float::half* alpha,
                                                  const Tensor
                                                  <half_float::half>& inputs,
                                                  const Tensor
                                                  <half_float::half>& diffInputs,
                                                  const Descriptor& desc,
                                                  const half_float::half* beta,
                                                  Tensor
                                                  <half_float::half>& diffSharedSynapses,
                                                  const Tensor<bool>& maps);
    template void ConvCell_Frame_Kernels::backwardFilter<float>(const float* alpha,
                                                  const Tensor
                                                  <float>& inputs,
                                                  const Tensor
                                                  <float>& diffInputs,
                                                  const Descriptor& desc,
                                                  const float* beta,
                                                  Tensor
                                                  <float>& diffSharedSynapses,
                                                  const Tensor<bool>& maps);
    template void ConvCell_Frame_Kernels::backwardFilter<double>(const double* alpha,
                                                  const Tensor
                                                  <double>& inputs,
                                                  const Tensor
                                                  <double>& diffInputs,
                                                  const Descriptor& desc,
                                                  const double* beta,
                                                  Tensor
                                                  <double>& diffSharedSynapses,
                                                  const Tensor<bool>& maps);

    template void ConvCell_Frame_Kernels::backwardBias<half_float::half>(const half_float::half* alpha,
                                                const Tensor
                                                <half_float::half>& diffInputs,
                                                const half_float::half* beta,
                                                Tensor<half_float::half>& diffBias);
    template void ConvCell_Frame_Kernels::backwardBias<float>(const float* alpha,
                                                const Tensor
                                                <float>& diffInputs,
                                                const float* beta,
                                                Tensor<float>& diffBias);
    template void ConvCell_Frame_Kernels::backwardBias<double>(const double* alpha,
                                                const Tensor
                                                <double>& diffInputs,
                                                const double* beta,
                                                Tensor<double>& diffBias);
}
