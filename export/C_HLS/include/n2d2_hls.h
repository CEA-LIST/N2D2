/*
    (C) Copyright 2015 CEA LIST. All Rights Reserved.
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

#ifndef N2D2_EXPORTC_DEEPNET_HLS_H
#define N2D2_EXPORTC_DEEPNET_HLS_H

//#define ENABLE_X_Y_LOOPS

#include "n2d2.h"
#include "typedefs.h"

#ifdef ENABLE_X_Y_LOOPS
#define CONVCELL_PROPAGATE_TYPE(PREFIX,                                        \
                                TYPE,                                          \
                                NAME,                                          \
                                M_nbChannels,                                  \
                                M_channelsHeight,                              \
                                M_channelsWidth,                               \
                                M_nbOutputs,                                   \
                                M_outputsHeight,                               \
                                M_outputsWidth,                                \
                                M_nbKernels,                                   \
                                M_kernelHeight,                                \
                                M_kernelWidth)                                 \
    void NAME##_##PREFIX##propagate(                                           \
        unsigned int nbChannels,                                               \
        unsigned int channelsHeight,                                           \
        unsigned int channelsWidth,                                            \
        int paddingY,                                                          \
        int paddingX,                                                          \
        unsigned int strideY,                                                  \
        unsigned int strideX,                                                  \
        unsigned int __attribute__((unused)) subSampleY,                       \
        __attribute__((unused)) unsigned int subSampleX,                       \
        DATA_T inputs[M_nbChannels][M_channelsHeight][M_channelsWidth],        \
        unsigned int oySize,                                                   \
        unsigned int oxSize,                                                   \
        __attribute__((unused)) unsigned int nbOutputs_,                       \
        __attribute__((unused)) unsigned int outputsHeight,                    \
        __attribute__((unused)) unsigned int outputsWidth,                     \
        unsigned int nbOutputs,                                                \
        unsigned int outputOffset,                                             \
        DATA_T outputs[M_nbOutputs][M_outputsHeight][M_outputsWidth],          \
        unsigned int kernelHeight,                                             \
        unsigned int kernelWidth,                                              \
        const BDATA_T bias[M_nbOutputs],                                       \
        const WDATA_T weights_full[M_nbOutputs][M_nbChannels][M_kernelHeight]  \
                                  [M_kernelWidth],                             \
        ActivationFunction_T func,                                             \
        int shift)                                                             \
    {                                                                          \
        SUM_T weightedSum = 0;                                                 \
                                                                               \
    LOOP_CONVCELL_OY:                                                          \
        for (unsigned int oy = 0; oy < oySize; ++oy) {                         \
        LOOP_CONVCELL_OX:                                                      \
            for (unsigned int ox = 0; ox < oxSize; ++ox) {                     \
            LOOP_CONVCELL_OUTPUT:                                              \
                for (unsigned int output = 0; output < nbOutputs; ++output) {  \
                LOOP_CONVCELL_CHANNEL:                                         \
                    for (unsigned int channel = 0; channel < nbChannels;       \
                         ++channel) {                                          \
                    LOOP_CONVCELL_SY:                                          \
                        for (unsigned int sy = 0; sy < kernelHeight; ++sy) {   \
                        LOOP_CONVCELL_SX:                                      \
                            for (unsigned int sx = 0; sx < kernelWidth;        \
                                 ++sx) {                                       \
                                if (channel == 0 && sx == 0 && sy == 0)        \
                                    weightedSum = bias[output];                \
                                                                               \
                                const int ix = (int)(ox * strideX + sx)        \
                                               - (int)paddingX;                \
                                const int iy = (int)(oy * strideY + sy)        \
                                               - (int)paddingY;                \
                                                                               \
                                if (ix >= 0 && ix < (int)channelsWidth         \
                                    && iy >= 0 && iy < (int)channelsHeight)    \
                                    weightedSum                                \
                                        += weights_full[output][channel][sy]   \
                                                       [sx]                    \
                                           * (SUM_T)((TYPE)                    \
                                                     inputs[channel][iy][ix]); \
                                                                               \
                                if (channel == nbChannels - 1                  \
                                    && sx == kernelWidth - 1                   \
                                    && sy == kernelHeight - 1)                 \
                                    outputs[outputOffset + output][oy][ox]     \
                                        = PREFIX##sat(                         \
                                            weightedSum, func, shift);         \
                            }                                                  \
                        }                                                      \
                    }                                                          \
                }                                                              \
            }                                                                  \
        }                                                                      \
    }
#else
#define CONVCELL_PROPAGATE_TYPE(PREFIX,                                        \
                                TYPE,                                          \
                                NAME,                                          \
                                M_nbChannels,                                  \
                                M_channelsHeight,                              \
                                M_channelsWidth,                               \
                                M_nbOutputs,                                   \
                                M_outputsHeight,                               \
                                M_outputsWidth,                                \
                                M_nbKernels,                                   \
                                M_kernelHeight,                                \
                                M_kernelWidth)                                 \
    void NAME##_##PREFIX##propagate(                                           \
        unsigned int nbChannels,                                               \
        unsigned int channelsHeight,                                           \
        unsigned int channelsWidth,                                            \
        int paddingY,                                                          \
        int paddingX,                                                          \
        unsigned int strideY,                                                  \
        unsigned int strideX,                                                  \
        unsigned int __attribute__((unused)) subSampleY,                       \
        __attribute__((unused)) unsigned int subSampleX,                       \
        DATA_T inputs[M_nbChannels][M_channelsHeight][M_channelsWidth],        \
        unsigned int oySize,                                                   \
        unsigned int oxSize,                                                   \
        __attribute__((unused)) unsigned int nbOutputs_,                       \
        __attribute__((unused)) unsigned int outputsHeight,                    \
        __attribute__((unused)) unsigned int outputsWidth,                     \
        unsigned int nbOutputs,                                                \
        unsigned int outputOffset,                                             \
        DATA_T outputs[M_nbOutputs][M_outputsHeight][M_outputsWidth],          \
        unsigned int kernelHeight,                                             \
        unsigned int kernelWidth,                                              \
        const BDATA_T bias[M_nbOutputs],                                       \
        const WDATA_T weights_full[M_nbOutputs][M_nbChannels][M_kernelHeight]  \
                                  [M_kernelWidth],                             \
        ActivationFunction_T func,                                             \
        int shift)                                                             \
    {                                                                          \
        const unsigned int oSize = oySize * oxSize;                            \
        const unsigned int sSize = kernelHeight * kernelWidth;                 \
        SUM_T weightedSum = 0;                                                 \
                                                                               \
    LOOP_CONVCELL_O:                                                           \
        for (unsigned int o = 0; o < oSize; ++o) {                             \
        LOOP_CONVCELL_OUTPUT:                                                  \
            for (unsigned int output = 0; output < nbOutputs; ++output) {      \
            LOOP_CONVCELL_CHANNEL:                                             \
                for (unsigned int channel = 0; channel < nbChannels;           \
                     ++channel) {                                              \
                LOOP_CONVCELL_S:                                               \
                    for (unsigned int s = 0; s < sSize; ++s) {                 \
                        const unsigned int ox = o % oxSize;                    \
                        const unsigned int oy = o / oxSize;                    \
                                                                               \
                        if (channel == 0 && s == 0)                            \
                            weightedSum = bias[output];                        \
                                                                               \
                        const unsigned int sx = s % kernelWidth;               \
                        const unsigned int sy = s / kernelWidth;               \
                        const int ix = (int)(ox * strideX + sx)                \
                                       - (int)paddingX;                        \
                        const int iy = (int)(oy * strideY + sy)                \
                                       - (int)paddingY;                        \
                                                                               \
                        if (ix >= 0 && ix < (int)channelsWidth && iy >= 0      \
                            && iy < (int)channelsHeight)                       \
                            weightedSum                                        \
                                += weights_full[output][channel][sy][sx]       \
                                   * (SUM_T)((TYPE)inputs[channel][iy][ix]);   \
                                                                               \
                        if (channel == nbChannels - 1 && s == sSize - 1)       \
                            outputs[outputOffset + output][oy][ox]             \
                                = PREFIX##sat(weightedSum, func, shift);       \
                    }                                                          \
                }                                                              \
            }                                                                  \
        }                                                                      \
    }
#endif

#define CONVCELL_PROPAGATE(NAME,                                               \
                           M_nbChannels,                                       \
                           M_channelsHeight,                                   \
                           M_channelsWidth,                                    \
                           M_nbOutputs,                                        \
                           M_outputsHeight,                                    \
                           M_outputsWidth,                                     \
                           M_nbKernels,                                        \
                           M_kernelHeight,                                     \
                           M_kernelWidth)                                      \
    CONVCELL_PROPAGATE_TYPE(,                                                  \
                            DATA_T,                                            \
                            NAME,                                              \
                            M_nbChannels,                                      \
                            M_channelsHeight,                                  \
                            M_channelsWidth,                                   \
                            M_nbOutputs,                                       \
                            M_outputsHeight,                                   \
                            M_outputsWidth,                                    \
                            M_nbKernels,                                       \
                            M_kernelHeight,                                    \
                            M_kernelWidth)

#define CONVCELL_UPROPAGATE(NAME,                                              \
                            M_nbChannels,                                      \
                            M_channelsHeight,                                  \
                            M_channelsWidth,                                   \
                            M_nbOutputs,                                       \
                            M_outputsHeight,                                   \
                            M_outputsWidth,                                    \
                            M_nbKernels,                                       \
                            M_kernelHeight,                                    \
                            M_kernelWidth)                                     \
    CONVCELL_PROPAGATE_TYPE(u,                                                 \
                            UDATA_T,                                           \
                            NAME,                                              \
                            M_nbChannels,                                      \
                            M_channelsHeight,                                  \
                            M_channelsWidth,                                   \
                            M_nbOutputs,                                       \
                            M_outputsHeight,                                   \
                            M_outputsWidth,                                    \
                            M_nbKernels,                                       \
                            M_kernelHeight,                                    \
                            M_kernelWidth)

#ifdef ENABLE_X_Y_LOOPS
#define CONVCELL_PROPAGATE_COMPACT_TYPE(PREFIX,                                \
                                        TYPE,                                  \
                                        NAME,                                  \
                                        M_nbChannels,                          \
                                        M_channelsHeight,                      \
                                        M_channelsWidth,                       \
                                        M_nbOutputs,                           \
                                        M_outputsHeight,                       \
                                        M_outputsWidth,                        \
                                        M_nbKernels,                           \
                                        M_kernelHeight,                        \
                                        M_kernelWidth)                         \
    void NAME##_##PREFIX##propagate(                                           \
        unsigned int nbChannels,                                               \
        unsigned int channelsHeight,                                           \
        unsigned int channelsWidth,                                            \
        int paddingY,                                                          \
        int paddingX,                                                          \
        unsigned int strideY,                                                  \
        unsigned int strideX,                                                  \
        unsigned int __attribute__((unused)) subSampleY,                       \
        __attribute__((unused)) unsigned int subSampleX,                       \
        DATA_T inputs[M_nbChannels][M_channelsHeight][M_channelsWidth],        \
        unsigned int oySize,                                                   \
        unsigned int oxSize,                                                   \
        __attribute__((unused)) unsigned int nbOutputs_,                       \
        __attribute__((unused)) unsigned int outputsHeight,                    \
        __attribute__((unused)) unsigned int outputsWidth,                     \
        unsigned int nbOutputs,                                                \
        unsigned int outputOffset,                                             \
        DATA_T outputs[M_nbOutputs][M_outputsHeight][M_outputsWidth],          \
        unsigned int kernelHeight,                                             \
        unsigned int kernelWidth,                                              \
        const BDATA_T bias[M_nbOutputs],                                       \
        const int weights_map[M_nbOutputs][M_nbChannels],                      \
        const WDATA_T weights_compact[M_nbKernels][M_kernelHeight]             \
                                     [M_kernelWidth],                          \
        ActivationFunction_T func,                                             \
        int shift)                                                             \
    {                                                                          \
        SUM_T weightedSum = 0;                                                 \
                                                                               \
    LOOP_CONVCELL_OY:                                                          \
        for (unsigned int oy = 0; oy < oySize; ++oy) {                         \
        LOOP_CONVCELL_OX:                                                      \
            for (unsigned int ox = 0; ox < oxSize; ++ox) {                     \
            LOOP_CONVCELL_OUTPUT:                                              \
                for (unsigned int output = 0; output < nbOutputs; ++output) {  \
                LOOP_CONVCELL_CHANNEL:                                         \
                    for (unsigned int channel = 0; channel < nbChannels;       \
                         ++channel) {                                          \
                    LOOP_CONVCELL_SY:                                          \
                        for (unsigned int sy = 0; sy < kernelHeight; ++sy) {   \
                        LOOP_CONVCELL_SX:                                      \
                            for (unsigned int sx = 0; sx < kernelWidth;        \
                                 ++sx) {                                       \
                                if (channel == 0 && sx == 0 && sy == 0)        \
                                    weightedSum = bias[output];                \
                                                                               \
                                if (weights_map[output][channel] >= 0) {       \
                                    const int ix = (int)(ox * strideX + sx)    \
                                                   - (int)paddingX;            \
                                    const int iy = (int)(oy * strideY + sy)    \
                                                   - (int)paddingY;            \
                                                                               \
                                    if (ix >= 0 && ix < (int)channelsWidth     \
                                        && iy >= 0                             \
                                        && iy < (int)channelsHeight) {         \
                                        weightedSum                            \
                                            += weights_compact                 \
                                                   [weights_map[output]        \
                                                               [channel]][sy]  \
                                                   [sx]                        \
                                               * (SUM_T)(                      \
                                                     (TYPE)                    \
                                                     inputs[channel][iy][ix]); \
                                    }                                          \
                                }                                              \
                                                                               \
                                if (channel == nbChannels - 1                  \
                                    && sx == kernelWidth - 1                   \
                                    && sy == kernelHeight - 1)                 \
                                    outputs[outputOffset + output][oy][ox]     \
                                        = PREFIX##sat(                         \
                                            weightedSum, func, shift);         \
                            }                                                  \
                        }                                                      \
                    }                                                          \
                }                                                              \
            }                                                                  \
        }                                                                      \
    }
#else
#define CONVCELL_PROPAGATE_COMPACT_TYPE(PREFIX,                                \
                                        TYPE,                                  \
                                        NAME,                                  \
                                        M_nbChannels,                          \
                                        M_channelsHeight,                      \
                                        M_channelsWidth,                       \
                                        M_nbOutputs,                           \
                                        M_outputsHeight,                       \
                                        M_outputsWidth,                        \
                                        M_nbKernels,                           \
                                        M_kernelHeight,                        \
                                        M_kernelWidth)                         \
    void NAME##_##PREFIX##propagate(                                           \
        unsigned int nbChannels,                                               \
        unsigned int channelsHeight,                                           \
        unsigned int channelsWidth,                                            \
        int paddingY,                                                          \
        int paddingX,                                                          \
        unsigned int strideY,                                                  \
        unsigned int strideX,                                                  \
        unsigned int __attribute__((unused)) subSampleY,                       \
        __attribute__((unused)) unsigned int subSampleX,                       \
        DATA_T inputs[M_nbChannels][M_channelsHeight][M_channelsWidth],        \
        unsigned int oySize,                                                   \
        unsigned int oxSize,                                                   \
        __attribute__((unused)) unsigned int nbOutputs_,                       \
        __attribute__((unused)) unsigned int outputsHeight,                    \
        __attribute__((unused)) unsigned int outputsWidth,                     \
        unsigned int nbOutputs,                                                \
        unsigned int outputOffset,                                             \
        DATA_T outputs[M_nbOutputs][M_outputsHeight][M_outputsWidth],          \
        unsigned int kernelHeight,                                             \
        unsigned int kernelWidth,                                              \
        const BDATA_T bias[M_nbOutputs],                                       \
        const int weights_map[M_nbOutputs][M_nbChannels],                      \
        const WDATA_T weights_compact[M_nbKernels][M_kernelHeight]             \
                                     [M_kernelWidth],                          \
        ActivationFunction_T func,                                             \
        int shift)                                                             \
    {                                                                          \
        const unsigned int oSize = oySize * oxSize;                            \
        const unsigned int sSize = kernelHeight * kernelWidth;                 \
        SUM_T weightedSum = 0;                                                 \
                                                                               \
    LOOP_CONVCELL_O:                                                           \
        for (unsigned int o = 0; o < oSize; ++o) {                             \
        LOOP_CONVCELL_OUTPUT:                                                  \
            for (unsigned int output = 0; output < nbOutputs; ++output) {      \
            LOOP_CONVCELL_CHANNEL:                                             \
                for (unsigned int channel = 0; channel < nbChannels;           \
                     ++channel) {                                              \
                LOOP_CONVCELL_S:                                               \
                    for (unsigned int s = 0; s < sSize; ++s) {                 \
                        const unsigned int ox = o % oxSize;                    \
                        const unsigned int oy = o / oxSize;                    \
                                                                               \
                        if (channel == 0 && s == 0)                            \
                            weightedSum = bias[output];                        \
                                                                               \
                        if (weights_map[output][channel] >= 0) {               \
                            const unsigned int sx = s % kernelWidth;           \
                            const unsigned int sy = s / kernelWidth;           \
                            const int ix = (int)(ox * strideX + sx)            \
                                           - (int)paddingX;                    \
                            const int iy = (int)(oy * strideY + sy)            \
                                           - (int)paddingY;                    \
                                                                               \
                            if (ix >= 0 && ix < (int)channelsWidth && iy >= 0  \
                                && iy < (int)channelsHeight) {                 \
                                weightedSum                                    \
                                    += weights_compact                         \
                                           [weights_map[output][channel]][sy]  \
                                           [sx]                                \
                                       * (SUM_T)(                              \
                                             (TYPE)inputs[channel][iy][ix]);   \
                            }                                                  \
                        }                                                      \
                                                                               \
                        if (channel == nbChannels - 1 && s == sSize - 1)       \
                            outputs[outputOffset + output][oy][ox]             \
                                = PREFIX##sat(weightedSum, func, shift);       \
                    }                                                          \
                }                                                              \
            }                                                                  \
        }                                                                      \
    }
#endif

#define CONVCELL_PROPAGATE_COMPACT(NAME,                                       \
                                   M_nbChannels,                               \
                                   M_channelsHeight,                           \
                                   M_channelsWidth,                            \
                                   M_nbOutputs,                                \
                                   M_outputsHeight,                            \
                                   M_outputsWidth,                             \
                                   M_nbKernels,                                \
                                   M_kernelHeight,                             \
                                   M_kernelWidth)                              \
    CONVCELL_PROPAGATE_COMPACT_TYPE(,                                          \
                                    DATA_T,                                    \
                                    NAME,                                      \
                                    M_nbChannels,                              \
                                    M_channelsHeight,                          \
                                    M_channelsWidth,                           \
                                    M_nbOutputs,                               \
                                    M_outputsHeight,                           \
                                    M_outputsWidth,                            \
                                    M_nbKernels,                               \
                                    M_kernelHeight,                            \
                                    M_kernelWidth)

#define CONVCELL_UPROPAGATE_COMPACT(NAME,                                      \
                                    M_nbChannels,                              \
                                    M_channelsHeight,                          \
                                    M_channelsWidth,                           \
                                    M_nbOutputs,                               \
                                    M_outputsHeight,                           \
                                    M_outputsWidth,                            \
                                    M_nbKernels,                               \
                                    M_kernelHeight,                            \
                                    M_kernelWidth)                             \
    CONVCELL_PROPAGATE_COMPACT_TYPE(u,                                         \
                                    UDATA_T,                                   \
                                    NAME,                                      \
                                    M_nbChannels,                              \
                                    M_channelsHeight,                          \
                                    M_channelsWidth,                           \
                                    M_nbOutputs,                               \
                                    M_outputsHeight,                           \
                                    M_outputsWidth,                            \
                                    M_nbKernels,                               \
                                    M_kernelHeight,                            \
                                    M_kernelWidth)

#ifdef ENABLE_X_Y_LOOPS
#define POOLCELL_PROPAGATE_TYPE(PREFIX,                                        \
                                TYPE,                                          \
                                NAME,                                          \
                                M_nbChannels,                                  \
                                M_channelsHeight,                              \
                                M_channelsWidth,                               \
                                M_nbOutputs,                                   \
                                M_outputsHeight,                               \
                                M_outputsWidth)                                \
    void NAME##_##PREFIX##propagate(                                           \
        unsigned int nbChannels,                                               \
        unsigned int channelsHeight,                                           \
        unsigned int channelsWidth,                                            \
        unsigned int strideY,                                                  \
        unsigned int strideX,                                                  \
        DATA_T inputs[M_nbChannels][M_channelsHeight][M_channelsWidth],        \
        __attribute__((unused)) unsigned int nbOutputs_,                       \
        unsigned int outputsHeight,                                            \
        unsigned int outputsWidth,                                             \
        unsigned int nbOutputs,                                                \
        unsigned int outputOffset,                                             \
        DATA_T outputs[M_nbOutputs][M_outputsHeight][M_outputsWidth],          \
        unsigned int poolHeight,                                               \
        unsigned int poolWidth,                                                \
        const char mapping[M_nbOutputs][M_nbChannels],                         \
        __attribute__((unused)) Pooling_T pooling,                             \
        __attribute__((unused)) ActivationFunction_T func,                     \
        __attribute__((unused)) int shift)                                     \
    {                                                                          \
        TYPE poolValue = TYPE##_MIN;                                           \
                                                                               \
    LOOP_POOLCELL_OY:                                                          \
        for (unsigned int oy = 0; oy < outputsHeight; ++oy) {                  \
        LOOP_POOLCELL_OX:                                                      \
            for (unsigned int ox = 0; ox < outputsWidth; ++ox) {               \
            LOOP_POOLCELL_OUTPUT:                                              \
                for (unsigned int output = 0; output < nbOutputs; ++output) {  \
                LOOP_POOLCELL_CHANNEL:                                         \
                    for (unsigned int channel = 0; channel < nbChannels;       \
                         ++channel) {                                          \
                    LOOP_POOLCELL_SY:                                          \
                        for (unsigned int sy = 0; sy < poolHeight; ++sy) {     \
                        LOOP_POOLCELL_SX:                                      \
                            for (unsigned int sx = 0; sx < poolWidth; ++sx) {  \
                                const unsigned int sxMax = uint_min(           \
                                    channelsWidth - ox * strideX, poolWidth);  \
                                const unsigned int syMax                       \
                                    = uint_min(channelsHeight - oy * strideY,  \
                                               poolHeight);                    \
                                                                               \
                                if (channel == 0 && sx == 0 && sy == 0)        \
                                    poolValue = TYPE##_MIN;                    \
                                                                               \
                                if (mapping[output][channel]) {                \
                                    if (sy < syMax && sx < sxMax) {            \
                                        const unsigned int ix = ox * strideX   \
                                                                + sx;          \
                                        const unsigned int iy = oy * strideY   \
                                                                + sy;          \
                                                                               \
                                        if (((TYPE)inputs[channel][iy][ix])    \
                                            > poolValue)                       \
                                            poolValue = ((                     \
                                                TYPE)inputs[channel][iy][ix]); \
                                    }                                          \
                                }                                              \
                                                                               \
                                if (channel == nbChannels - 1                  \
                                    && sx == poolWidth - 1                     \
                                    && sy == poolHeight - 1)                   \
                                    outputs[outputOffset + output][oy][ox]     \
                                        = poolValue;                           \
                            }                                                  \
                        }                                                      \
                    }                                                          \
                }                                                              \
            }                                                                  \
        }                                                                      \
    }                                                                          \
    void NAME##_##PREFIX##propagate_average(                                   \
        unsigned int nbChannels,                                               \
        unsigned int channelsHeight,                                           \
        unsigned int channelsWidth,                                            \
        unsigned int strideY,                                                  \
        unsigned int strideX,                                                  \
        DATA_T inputs[M_nbChannels][M_channelsHeight][M_channelsWidth],        \
        __attribute__((unused)) unsigned int nbOutputs_,                       \
        unsigned int outputsHeight,                                            \
        unsigned int outputsWidth,                                             \
        unsigned int nbOutputs,                                                \
        unsigned int outputOffset,                                             \
        DATA_T outputs[M_nbOutputs][M_outputsHeight][M_outputsWidth],          \
        unsigned int poolHeight,                                               \
        unsigned int poolWidth,                                                \
        const char mapping[M_nbOutputs][M_nbChannels],                         \
        __attribute__((unused)) Pooling_T pooling,                             \
        __attribute__((unused)) ActivationFunction_T func,                     \
        int shift)                                                             \
    {                                                                          \
        SUM_T sum = 0;                                                         \
        unsigned int poolNbChannels = 0;                                       \
                                                                               \
    LOOP_POOLCELL_OY:                                                          \
        for (unsigned int oy = 0; oy < outputsHeight; ++oy) {                  \
        LOOP_POOLCELL_OX:                                                      \
            for (unsigned int ox = 0; ox < outputsWidth; ++ox) {               \
            LOOP_POOLCELL_OUTPUT:                                              \
                for (unsigned int output = 0; output < nbOutputs; ++output) {  \
                LOOP_POOLCELL_CHANNEL:                                         \
                    for (unsigned int channel = 0; channel < nbChannels;       \
                         ++channel) {                                          \
                    LOOP_POOLCELL_SY:                                          \
                        for (unsigned int sy = 0; sy < poolHeight; ++sy) {     \
                        LOOP_POOLCELL_SX:                                      \
                            for (unsigned int sx = 0; sx < poolWidth; ++sx) {  \
                                const unsigned int sxMax = uint_min(           \
                                    channelsWidth - ox * strideX, poolWidth);  \
                                const unsigned int syMax                       \
                                    = uint_min(channelsHeight - oy * strideY,  \
                                               poolHeight);                    \
                                                                               \
                                if (channel == 0 && sx == 0 && sy == 0) {      \
                                    sum = 0;                                   \
                                    poolNbChannels = 0;                        \
                                }                                              \
                                                                               \
                                if (mapping[output][channel]) {                \
                                    if (sy < syMax && sx < sxMax) {            \
                                        const unsigned int ix = ox * strideX   \
                                                                + sx;          \
                                        const unsigned int iy = oy * strideY   \
                                                                + sy;          \
                                                                               \
                                        sum += ((                              \
                                            TYPE)inputs[channel][iy][ix]);     \
                                    }                                          \
                                                                               \
                                    if (sx == 0 && sy == 0)                    \
                                        ++poolNbChannels;                      \
                                }                                              \
                                                                               \
                                if (channel == nbChannels - 1                  \
                                    && sx == poolWidth - 1                     \
                                    && sy == poolHeight - 1) {                 \
                                    sum /= poolWidth * poolHeight              \
                                           * poolNbChannels;                   \
                                    outputs[outputOffset + output][oy][ox]     \
                                        = sht(sum, shift);                     \
                                }                                              \
                            }                                                  \
                        }                                                      \
                    }                                                          \
                }                                                              \
            }                                                                  \
        }                                                                      \
    }                                                                          \
    void NAME##_##PREFIX##propagate_unitmap(                                   \
        __attribute__((unused)) unsigned int nbChannels,                       \
        unsigned int channelsHeight,                                           \
        unsigned int channelsWidth,                                            \
        unsigned int strideY,                                                  \
        unsigned int strideX,                                                  \
        DATA_T inputs[M_nbChannels][M_channelsHeight][M_channelsWidth],        \
        __attribute__((unused)) unsigned int nbOutputs_,                       \
        unsigned int outputsHeight,                                            \
        unsigned int outputsWidth,                                             \
        unsigned int nbOutputs,                                                \
        unsigned int outputOffset,                                             \
        DATA_T outputs[M_nbOutputs][M_outputsHeight][M_outputsWidth],          \
        unsigned int poolHeight,                                               \
        unsigned int poolWidth,                                                \
        __attribute__((unused)) Pooling_T pooling,                             \
        __attribute__((unused)) ActivationFunction_T func,                     \
        __attribute__((unused)) int shift)                                     \
    {                                                                          \
        TYPE poolValue = TYPE##_MIN;                                           \
                                                                               \
    LOOP_POOLCELL_OY:                                                          \
        for (unsigned int oy = 0; oy < outputsHeight; ++oy) {                  \
        LOOP_POOLCELL_OX:                                                      \
            for (unsigned int ox = 0; ox < outputsWidth; ++ox) {               \
            LOOP_POOLCELL_OUTPUT:                                              \
                for (unsigned int output = 0; output < nbOutputs; ++output) {  \
                LOOP_POOLCELL_SY:                                              \
                    for (unsigned int sy = 0; sy < poolHeight; ++sy) {         \
                    LOOP_POOLCELL_SX:                                          \
                        for (unsigned int sx = 0; sx < poolWidth; ++sx) {      \
                            const unsigned int sxMax = uint_min(               \
                                channelsWidth - ox * strideX, poolWidth);      \
                            const unsigned int syMax = uint_min(               \
                                channelsHeight - oy * strideY, poolHeight);    \
                                                                               \
                            if (sx == 0 && sy == 0)                            \
                                poolValue = TYPE##_MIN;                        \
                                                                               \
                            if (sy < syMax && sx < sxMax) {                    \
                                const unsigned int ix = ox * strideX + sx;     \
                                const unsigned int iy = oy * strideY + sy;     \
                                                                               \
                                if (((TYPE)inputs[output][iy][ix])             \
                                    > poolValue)                               \
                                    poolValue                                  \
                                        = ((TYPE)inputs[output][iy][ix]);      \
                            }                                                  \
                                                                               \
                            if (sx == poolWidth - 1 && sy == poolHeight - 1)   \
                                outputs[outputOffset + output][oy][ox]         \
                                    = poolValue;                               \
                        }                                                      \
                    }                                                          \
                }                                                              \
            }                                                                  \
        }                                                                      \
    }                                                                          \
    void NAME##_##PREFIX##propagate_unitmap_average(                           \
        __attribute__((unused)) unsigned int nbChannels,                       \
        unsigned int channelsHeight,                                           \
        unsigned int channelsWidth,                                            \
        unsigned int strideY,                                                  \
        unsigned int strideX,                                                  \
        DATA_T inputs[M_nbChannels][M_channelsHeight][M_channelsWidth],        \
        __attribute__((unused)) unsigned int nbOutputs_,                       \
        unsigned int outputsHeight,                                            \
        unsigned int outputsWidth,                                             \
        unsigned int nbOutputs,                                                \
        unsigned int outputOffset,                                             \
        DATA_T outputs[M_nbOutputs][M_outputsHeight][M_outputsWidth],          \
        unsigned int poolHeight,                                               \
        unsigned int poolWidth,                                                \
        __attribute__((unused)) Pooling_T pooling,                             \
        __attribute__((unused)) ActivationFunction_T func,                     \
        int shift)                                                             \
    {                                                                          \
        SUM_T sum = 0;                                                         \
                                                                               \
    LOOP_POOLCELL_OY:                                                          \
        for (unsigned int oy = 0; oy < outputsHeight; ++oy) {                  \
        LOOP_POOLCELL_OX:                                                      \
            for (unsigned int ox = 0; ox < outputsWidth; ++ox) {               \
            LOOP_POOLCELL_OUTPUT:                                              \
                for (unsigned int output = 0; output < nbOutputs; ++output) {  \
                LOOP_POOLCELL_SY:                                              \
                    for (unsigned int sy = 0; sy < poolHeight; ++sy) {         \
                    LOOP_POOLCELL_SX:                                          \
                        for (unsigned int sx = 0; sx < poolWidth; ++sx) {      \
                            const unsigned int sxMax = uint_min(               \
                                channelsWidth - ox * strideX, poolWidth);      \
                            const unsigned int syMax = uint_min(               \
                                channelsHeight - oy * strideY, poolHeight);    \
                                                                               \
                            if (sx == 0 && sy == 0)                            \
                                sum = 0;                                       \
                                                                               \
                            if (sy < syMax && sx < sxMax) {                    \
                                const unsigned int ix = ox * strideX + sx;     \
                                const unsigned int iy = oy * strideY + sy;     \
                                                                               \
                                sum += ((TYPE)inputs[output][iy][ix]);         \
                            }                                                  \
                                                                               \
                            if (sx == poolWidth - 1 && sy == poolHeight - 1) { \
                                sum /= poolWidth * poolHeight;                 \
                                outputs[outputOffset + output][oy][ox]         \
                                    = sht(sum, shift);                         \
                            }                                                  \
                        }                                                      \
                    }                                                          \
                }                                                              \
            }                                                                  \
        }                                                                      \
    }
#else
#define POOLCELL_PROPAGATE_TYPE(PREFIX,                                        \
                                TYPE,                                          \
                                NAME,                                          \
                                M_nbChannels,                                  \
                                M_channelsHeight,                              \
                                M_channelsWidth,                               \
                                M_nbOutputs,                                   \
                                M_outputsHeight,                               \
                                M_outputsWidth)                                \
    void NAME##_##PREFIX##propagate(                                           \
        unsigned int nbChannels,                                               \
        unsigned int channelsHeight,                                           \
        unsigned int channelsWidth,                                            \
        unsigned int strideY,                                                  \
        unsigned int strideX,                                                  \
        DATA_T inputs[M_nbChannels][M_channelsHeight][M_channelsWidth],        \
        __attribute__((unused)) unsigned int nbOutputs_,                       \
        unsigned int outputsHeight,                                            \
        unsigned int outputsWidth,                                             \
        unsigned int nbOutputs,                                                \
        unsigned int outputOffset,                                             \
        DATA_T outputs[M_nbOutputs][M_outputsHeight][M_outputsWidth],          \
        unsigned int poolHeight,                                               \
        unsigned int poolWidth,                                                \
        const char mapping[M_nbOutputs][M_nbChannels],                         \
        __attribute__((unused)) Pooling_T pooling,                             \
        __attribute__((unused)) ActivationFunction_T func,                     \
        __attribute__((unused)) int shift)                                     \
    {                                                                          \
        const unsigned int oSize = outputsHeight * outputsWidth;               \
        const unsigned int sSize = poolHeight * poolWidth;                     \
        TYPE poolValue = TYPE##_MIN;                                           \
                                                                               \
    LOOP_POOLCELL_O:                                                           \
        for (unsigned int o = 0; o < oSize; ++o) {                             \
        LOOP_POOLCELL_OUTPUT:                                                  \
            for (unsigned int output = 0; output < nbOutputs; ++output) {      \
            LOOP_POOLCELL_CHANNEL:                                             \
                for (unsigned int channel = 0; channel < nbChannels;           \
                     ++channel) {                                              \
                LOOP_POOLCELL_S:                                               \
                    for (unsigned int s = 0; s < sSize; ++s) {                 \
                        const unsigned int ox = o % outputsWidth;              \
                        const unsigned int oy = o / outputsWidth;              \
                        const unsigned int sxMax = uint_min(                   \
                            channelsWidth - ox * strideX, poolWidth);          \
                        const unsigned int syMax = uint_min(                   \
                            channelsHeight - oy * strideY, poolHeight);        \
                                                                               \
                        if (channel == 0 && s == 0)                            \
                            poolValue = TYPE##_MIN;                            \
                                                                               \
                        if (mapping[output][channel]) {                        \
                            const unsigned int sx = s % poolWidth;             \
                            const unsigned int sy = s / poolWidth;             \
                                                                               \
                            if (sy < syMax && sx < sxMax) {                    \
                                const unsigned int ix = ox * strideX + sx;     \
                                const unsigned int iy = oy * strideY + sy;     \
                                                                               \
                                if (((TYPE)inputs[channel][iy][ix])            \
                                    > poolValue)                               \
                                    poolValue                                  \
                                        = ((TYPE)inputs[channel][iy][ix]);     \
                            }                                                  \
                        }                                                      \
                                                                               \
                        if (channel == nbChannels - 1 && s == sSize - 1)       \
                            outputs[outputOffset + output][oy][ox]             \
                                = poolValue;                                   \
                    }                                                          \
                }                                                              \
            }                                                                  \
        }                                                                      \
    }                                                                          \
    void NAME##_##PREFIX##propagate_average(                                   \
        unsigned int nbChannels,                                               \
        unsigned int channelsHeight,                                           \
        unsigned int channelsWidth,                                            \
        unsigned int strideY,                                                  \
        unsigned int strideX,                                                  \
        DATA_T inputs[M_nbChannels][M_channelsHeight][M_channelsWidth],        \
        __attribute__((unused)) unsigned int nbOutputs_,                       \
        unsigned int outputsHeight,                                            \
        unsigned int outputsWidth,                                             \
        unsigned int nbOutputs,                                                \
        unsigned int outputOffset,                                             \
        DATA_T outputs[M_nbOutputs][M_outputsHeight][M_outputsWidth],          \
        unsigned int poolHeight,                                               \
        unsigned int poolWidth,                                                \
        const char mapping[M_nbOutputs][M_nbChannels],                         \
        __attribute__((unused)) Pooling_T pooling,                             \
        __attribute__((unused)) ActivationFunction_T func,                     \
        int shift)                                                             \
    {                                                                          \
        const unsigned int oSize = outputsHeight * outputsWidth;               \
        const unsigned int sSize = poolHeight * poolWidth;                     \
        SUM_T sum = 0;                                                         \
        unsigned int poolNbChannels = 0;                                       \
                                                                               \
    LOOP_POOLCELL_O:                                                           \
        for (unsigned int o = 0; o < oSize; ++o) {                             \
        LOOP_POOLCELL_OUTPUT:                                                  \
            for (unsigned int output = 0; output < nbOutputs; ++output) {      \
            LOOP_POOLCELL_CHANNEL:                                             \
                for (unsigned int channel = 0; channel < nbChannels;           \
                     ++channel) {                                              \
                LOOP_POOLCELL_S:                                               \
                    for (unsigned int s = 0; s < sSize; ++s) {                 \
                        const unsigned int ox = o % outputsWidth;              \
                        const unsigned int oy = o / outputsWidth;              \
                        const unsigned int sxMax = uint_min(                   \
                            channelsWidth - ox * strideX, poolWidth);          \
                        const unsigned int syMax = uint_min(                   \
                            channelsHeight - oy * strideY, poolHeight);        \
                                                                               \
                        if (channel == 0 && s == 0) {                          \
                            sum = 0;                                           \
                            poolNbChannels = 0;                                \
                        }                                                      \
                                                                               \
                        if (mapping[output][channel]) {                        \
                            const unsigned int sx = s % poolWidth;             \
                            const unsigned int sy = s / poolWidth;             \
                                                                               \
                            if (sy < syMax && sx < sxMax) {                    \
                                const unsigned int ix = ox * strideX + sx;     \
                                const unsigned int iy = oy * strideY + sy;     \
                                                                               \
                                sum += ((TYPE)inputs[channel][iy][ix]);        \
                            }                                                  \
                                                                               \
                            if (s == 0)                                        \
                                ++poolNbChannels;                              \
                        }                                                      \
                                                                               \
                        if (channel == nbChannels - 1 && s == sSize - 1) {     \
                            sum /= sSize * poolNbChannels;                     \
                            outputs[outputOffset + output][oy][ox]             \
                                = sht(sum, shift);                             \
                        }                                                      \
                    }                                                          \
                }                                                              \
            }                                                                  \
        }                                                                      \
    }                                                                          \
    void NAME##_##PREFIX##propagate_unitmap(                                   \
        __attribute__((unused)) unsigned int nbChannels,                       \
        unsigned int channelsHeight,                                           \
        unsigned int channelsWidth,                                            \
        unsigned int strideY,                                                  \
        unsigned int strideX,                                                  \
        DATA_T inputs[M_nbChannels][M_channelsHeight][M_channelsWidth],        \
        __attribute__((unused)) unsigned int nbOutputs_,                       \
        unsigned int outputsHeight,                                            \
        unsigned int outputsWidth,                                             \
        unsigned int nbOutputs,                                                \
        unsigned int outputOffset,                                             \
        DATA_T outputs[M_nbOutputs][M_outputsHeight][M_outputsWidth],          \
        unsigned int poolHeight,                                               \
        unsigned int poolWidth,                                                \
        __attribute__((unused)) Pooling_T pooling,                             \
        __attribute__((unused)) ActivationFunction_T func,                     \
        __attribute__((unused)) int shift)                                     \
    {                                                                          \
        const unsigned int oSize = outputsHeight * outputsWidth;               \
        const unsigned int sSize = poolHeight * poolWidth;                     \
        TYPE poolValue = TYPE##_MIN;                                           \
                                                                               \
    LOOP_POOLCELL_O:                                                           \
        for (unsigned int o = 0; o < oSize; ++o) {                             \
        LOOP_POOLCELL_OUTPUT:                                                  \
            for (unsigned int output = 0; output < nbOutputs; ++output) {      \
            LOOP_POOLCELL_S:                                                   \
                for (unsigned int s = 0; s < sSize; ++s) {                     \
                    const unsigned int ox = o % outputsWidth;                  \
                    const unsigned int oy = o / outputsWidth;                  \
                    const unsigned int sxMax                                   \
                        = uint_min(channelsWidth - ox * strideX, poolWidth);   \
                    const unsigned int syMax                                   \
                        = uint_min(channelsHeight - oy * strideY, poolHeight); \
                                                                               \
                    if (s == 0)                                                \
                        poolValue = TYPE##_MIN;                                \
                                                                               \
                    const unsigned int sx = s % poolWidth;                     \
                    const unsigned int sy = s / poolWidth;                     \
                                                                               \
                    if (sy < syMax && sx < sxMax) {                            \
                        const unsigned int ix = ox * strideX + sx;             \
                        const unsigned int iy = oy * strideY + sy;             \
                                                                               \
                        if (((TYPE)inputs[output][iy][ix]) > poolValue)        \
                            poolValue = ((TYPE)inputs[output][iy][ix]);        \
                    }                                                          \
                                                                               \
                    if (s == sSize - 1)                                        \
                        outputs[outputOffset + output][oy][ox] = poolValue;    \
                }                                                              \
            }                                                                  \
        }                                                                      \
    }                                                                          \
    void NAME##_##PREFIX##propagate_unitmap_average(                           \
        __attribute__((unused)) unsigned int nbChannels,                       \
        unsigned int channelsHeight,                                           \
        unsigned int channelsWidth,                                            \
        unsigned int strideY,                                                  \
        unsigned int strideX,                                                  \
        DATA_T inputs[M_nbChannels][M_channelsHeight][M_channelsWidth],        \
        __attribute__((unused)) unsigned int nbOutputs_,                       \
        unsigned int outputsHeight,                                            \
        unsigned int outputsWidth,                                             \
        unsigned int nbOutputs,                                                \
        unsigned int outputOffset,                                             \
        DATA_T outputs[M_nbOutputs][M_outputsHeight][M_outputsWidth],          \
        unsigned int poolHeight,                                               \
        unsigned int poolWidth,                                                \
        __attribute__((unused)) Pooling_T pooling,                             \
        __attribute__((unused)) ActivationFunction_T func,                     \
        int shift)                                                             \
    {                                                                          \
        const unsigned int oSize = outputsHeight * outputsWidth;               \
        const unsigned int sSize = poolHeight * poolWidth;                     \
        SUM_T sum = 0;                                                         \
                                                                               \
    LOOP_POOLCELL_O:                                                           \
        for (unsigned int o = 0; o < oSize; ++o) {                             \
        LOOP_POOLCELL_OUTPUT:                                                  \
            for (unsigned int output = 0; output < nbOutputs; ++output) {      \
            LOOP_POOLCELL_S:                                                   \
                for (unsigned int s = 0; s < sSize; ++s) {                     \
                    const unsigned int ox = o % outputsWidth;                  \
                    const unsigned int oy = o / outputsWidth;                  \
                    const unsigned int sxMax                                   \
                        = uint_min(channelsWidth - ox * strideX, poolWidth);   \
                    const unsigned int syMax                                   \
                        = uint_min(channelsHeight - oy * strideY, poolHeight); \
                                                                               \
                    if (s == 0)                                                \
                        sum = 0;                                               \
                                                                               \
                    const unsigned int sx = s % poolWidth;                     \
                    const unsigned int sy = s / poolWidth;                     \
                                                                               \
                    if (sy < syMax && sx < sxMax) {                            \
                        const unsigned int ix = ox * strideX + sx;             \
                        const unsigned int iy = oy * strideY + sy;             \
                                                                               \
                        sum += ((TYPE)inputs[output][iy][ix]);                 \
                    }                                                          \
                                                                               \
                    if (s == sSize - 1) {                                      \
                        sum /= sSize;                                          \
                        outputs[outputOffset + output][oy][ox]                 \
                            = sht(sum, shift);                                 \
                    }                                                          \
                }                                                              \
            }                                                                  \
        }                                                                      \
    }
#endif

#define POOLCELL_PROPAGATE(NAME,                                               \
                           M_nbChannels,                                       \
                           M_channelsHeight,                                   \
                           M_channelsWidth,                                    \
                           M_nbOutputs,                                        \
                           M_outputsHeight,                                    \
                           M_outputsWidth)                                     \
    POOLCELL_PROPAGATE_TYPE(,                                                  \
                            DATA_T,                                            \
                            NAME,                                              \
                            M_nbChannels,                                      \
                            M_channelsHeight,                                  \
                            M_channelsWidth,                                   \
                            M_nbOutputs,                                       \
                            M_outputsHeight,                                   \
                            M_outputsWidth)

#define POOLCELL_UPROPAGATE(NAME,                                              \
                            M_nbChannels,                                      \
                            M_channelsHeight,                                  \
                            M_channelsWidth,                                   \
                            M_nbOutputs,                                       \
                            M_outputsHeight,                                   \
                            M_outputsWidth)                                    \
    POOLCELL_PROPAGATE_TYPE(u,                                                 \
                            UDATA_T,                                           \
                            NAME,                                              \
                            M_nbChannels,                                      \
                            M_channelsHeight,                                  \
                            M_channelsWidth,                                   \
                            M_nbOutputs,                                       \
                            M_outputsHeight,                                   \
                            M_outputsWidth)

#define FCCELL_PROPAGATE_2D_TYPE(PREFIX,                                       \
                                 TYPE,                                         \
                                 NAME,                                         \
                                 M_nbChannels,                                 \
                                 M_channelsHeight,                             \
                                 M_channelsWidth,                              \
                                 M_nbOutputs,                                  \
                                 M_nbChannels_,                                \
                                 M_nbWeights)                                  \
    void NAME##_##PREFIX##propagate_2d(                                        \
        unsigned int nbChannels,                                               \
        unsigned int channelsHeight,                                           \
        unsigned int channelsWidth,                                            \
        DATA_T inputs[M_nbChannels][M_channelsHeight][M_channelsWidth],        \
        __attribute__((unused)) unsigned int nbOutputs_,                       \
        unsigned int nbOutputs,                                                \
        unsigned int outputOffset,                                             \
        DATA_T outputs[M_nbOutputs],                                           \
        __attribute__((unused)) unsigned int nbChannels_,                      \
        const BDATA_T bias[M_nbOutputs],                                       \
        const WDATA_T weights[M_nbOutputs][M_nbChannels_],                     \
        ActivationFunction_T func,                                             \
        int shift)                                                             \
    {                                                                          \
        const unsigned int iSize = channelsHeight * channelsWidth;             \
        SUM_T weightedSum = 0;                                                 \
                                                                               \
    LOOP_FCCELL_OUTPUT:                                                        \
        for (unsigned int output = 0; output < nbOutputs; ++output) {          \
        LOOP_FCCELL_CHANNEL:                                                   \
            for (unsigned int channel = 0; channel < nbChannels; ++channel) {  \
            LOOP_FCCELL_I:                                                     \
                for (unsigned int i = 0; i < iSize; ++i) {                     \
                    if (channel == 0 && i == 0)                                \
                        weightedSum = bias[output];                            \
                                                                               \
                    const unsigned int ix = i % channelsWidth;                 \
                    const unsigned int iy = i / channelsWidth;                 \
                    const unsigned int c = i + channel * iSize;                \
                                                                               \
                    weightedSum += weights[output][c]                          \
                                   * (SUM_T)((TYPE)inputs[channel][iy][ix]);   \
                                                                               \
                    if (channel == nbChannels - 1 && i == iSize - 1)           \
                        outputs[outputOffset + output]                         \
                            = PREFIX##sat(weightedSum, func, shift);           \
                }                                                              \
            }                                                                  \
        }                                                                      \
    }                                                                          \
    void NAME##_##PREFIX##propagate_2d_sparse(                                 \
        unsigned int nbChannels,                                               \
        unsigned int channelsHeight,                                           \
        unsigned int channelsWidth,                                            \
        DATA_T inputs[M_nbChannels][M_channelsHeight][M_channelsWidth],        \
        __attribute__((unused)) unsigned int nbOutputs_,                       \
        unsigned int nbOutputs,                                                \
        unsigned int outputOffset,                                             \
        DATA_T outputs[M_nbOutputs],                                           \
        __attribute__((unused)) unsigned int nbChannels_,                      \
        const BDATA_T bias[M_nbOutputs],                                       \
        unsigned int nbWeights,                                                \
        const WDATA_T weights[M_nbWeights],                                    \
        const unsigned short offsets[M_nbWeights],                             \
        ActivationFunction_T func,                                             \
        int shift)                                                             \
    {                                                                          \
        const unsigned int channelSize = channelsHeight * channelsWidth;       \
        const unsigned int channelsSize = nbChannels * channelSize;            \
                                                                               \
        unsigned int w = 0;                                                    \
        unsigned int c = offsets[0];                                           \
                                                                               \
    LOOP_FCCELL_OUTPUT:                                                        \
        for (unsigned int output = 0; output < nbOutputs; ++output) {          \
            SUM_T weightedSum = bias[output];                                  \
                                                                               \
        LOOP_FCCELL_CHANNEL:                                                   \
            while (c < channelsSize && w < nbWeights) {                        \
                const unsigned int channel = c / channelSize;                  \
                const unsigned int i = c % channelSize;                        \
                const unsigned int ix = i % channelsWidth;                     \
                const unsigned int iy = i / channelsWidth;                     \
                                                                               \
                weightedSum += weights[w]                                      \
                               * (SUM_T)((TYPE)inputs[channel][iy][ix]);       \
                                                                               \
                ++w;                                                           \
                c += offsets[w];                                               \
            }                                                                  \
                                                                               \
            c -= channelsSize;                                                 \
                                                                               \
            outputs[outputOffset + output]                                     \
                = PREFIX##sat(weightedSum, func, shift);                       \
        }                                                                      \
    }

#define FCCELL_PROPAGATE_2D(NAME,                                              \
                            M_nbChannels,                                      \
                            M_channelsHeight,                                  \
                            M_channelsWidth,                                   \
                            M_nbOutputs,                                       \
                            M_nbChannels_,                                     \
                            M_nbWeights)                                       \
    FCCELL_PROPAGATE_2D_TYPE(,                                                 \
                             DATA_T,                                           \
                             NAME,                                             \
                             M_nbChannels,                                     \
                             M_channelsHeight,                                 \
                             M_channelsWidth,                                  \
                             M_nbOutputs,                                      \
                             M_nbChannels_,                                    \
                             M_nbWeights)

#define FCCELL_UPROPAGATE_2D(NAME,                                             \
                             M_nbChannels,                                     \
                             M_channelsHeight,                                 \
                             M_channelsWidth,                                  \
                             M_nbOutputs,                                      \
                             M_nbChannels_,                                    \
                             M_nbWeights)                                      \
    FCCELL_PROPAGATE_2D_TYPE(u,                                                \
                             UDATA_T,                                          \
                             NAME,                                             \
                             M_nbChannels,                                     \
                             M_channelsHeight,                                 \
                             M_channelsWidth,                                  \
                             M_nbOutputs,                                      \
                             M_nbChannels_,                                    \
                             M_nbWeights)

#define FCCELL_PROPAGATE_TYPE(PREFIX,                                                       \
                              TYPE,                                                         \
                              NAME,                                                         \
                              M_nbChannels,                                                 \
                              M_nbOutputs,                                                  \
                              M_nbWeights)                                                  \
    void NAME##_##PREFIX##propagate(unsigned int nbChannels,                                \
                                    DATA_T inputs[M_nbChannels],                            \
                                    __attribute__((unused)) unsigned int nbOutputs_,        \
                                    unsigned int nbOutputs,                                 \
                                    unsigned int outputOffset,                              \
                                    DATA_T outputs[M_nbOutputs],                            \
                                    const BDATA_T bias[M_nbOutputs],                        \
                                    const WDATA_T weights[M_nbOutputs][M_nbChannels],       \
                                    ActivationFunction_T func,                              \
                                    int shift)                                              \
    {                                                                                       \
        SUM_T weightedSum = 0;                                                              \
                                                                                            \
    LOOP_FCCELL_OUTPUT:                                                                     \
        for (unsigned int output = 0; output < nbOutputs; ++output)                         \
        {                                                                                   \
        LOOP_FCCELL_CHANNEL:                                                                \
            for (unsigned int channel = 0; channel < nbChannels; ++channel)                 \
            {                                                                               \
                if (channel == 0)                                                           \
                    weightedSum = bias[output];                                             \
                                                                                            \
                weightedSum += weights[output][channel] * (SUM_T)((TYPE)inputs[channel]);   \
                                                                                            \
                if (channel == nbChannels - 1)                                              \
                    outputs[outputOffset + output] = PREFIX##sat(weightedSum, func, shift); \
            }                                                                               \
        }                                                                                   \
    }                                                                                       \
    void NAME##_##PREFIX##propagate_sparse(                                                 \
        unsigned int nbChannels,                                                            \
        DATA_T inputs[M_nbChannels],                                                        \
        __attribute__((unused)) unsigned int nbOutputs_,                                    \
        unsigned int nbOutputs,                                                             \
        unsigned int outputOffset,                                                          \
        DATA_T outputs[M_nbOutputs],                                                        \
        const BDATA_T bias[M_nbOutputs],                                                    \
        unsigned int nbWeights,                                                             \
        const WDATA_T weights[M_nbWeights],                                                 \
        const unsigned short offsets[M_nbWeights],                                          \
        ActivationFunction_T func,                                                          \
        int shift)                                                                          \
    {                                                                                       \
        unsigned int w = 0;                                                                 \
        unsigned int channel = offsets[0];                                                  \
                                                                                            \
    LOOP_FCCELL_OUTPUT:                                                                     \
        for (unsigned int output = 0; output < nbOutputs; ++output)                         \
        {                                                                                   \
            SUM_T weightedSum = bias[output];                                               \
                                                                                            \
        LOOP_FCCELL_CHANNEL:                                                                \
            while (channel < nbChannels && w < nbWeights)                                   \
            {                                                                               \
                weightedSum += weights[w] * (SUM_T)((TYPE)inputs[channel]);                 \
                                                                                            \
                ++w;                                                                        \
                channel += offsets[w];                                                      \
            }                                                                               \
                                                                                            \
            channel -= nbChannels;                                                          \
                                                                                            \
            outputs[outputOffset + output] = PREFIX##sat(weightedSum, func, shift);         \
        }                                                                                   \
    }

#define FCCELL_PROPAGATE(NAME,              \
                         M_nbChannels,      \
                         M_nbOutputs,       \
                         M_nbWeights)       \
    FCCELL_PROPAGATE_TYPE(,                 \
                          DATA_T,           \
                          NAME,             \
                          M_nbChannels,     \
                          M_nbOutputs,      \
                          M_nbWeights)

#define FCCELL_UPROPAGATE(NAME,             \
                          M_nbChannels,     \
                          M_nbOutputs,      \
                          M_nbWeights)      \
    FCCELL_PROPAGATE_TYPE(u,                \
                          UDATA_T,          \
                          NAME,             \
                          M_nbChannels,     \
                          M_nbOutputs,      \
                          M_nbWeights)

#endif // N2D2_EXPORTC_DEEPNET_HLS_H
