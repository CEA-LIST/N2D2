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

#ifndef N2D2_EXPORTC_DEEPNET_H
#define N2D2_EXPORTC_DEEPNET_H

// For logistic function, should be 0.5, but 0 gives more dynamic range
#define BINARY_THRESHOLD 0
//#define ACC_NB_BITS 16

#ifndef NO_DIRENT
#include <dirent.h>
#else
#include "getline.h"
#endif

#include <stddef.h> // NULL def
#include <stdint.h> // (u)intx_t typedef
#include <stdio.h> // printf()
#include <sys/time.h>

#if defined(NL) || NB_BITS < 0
#include <math.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#include "typedefs.h"
#include "utils.h"

#ifdef ACC_NB_BITS
    #define ACC_MAX ((1LL << (ACC_NB_BITS - 1)) - 1)
    #define ACC_MIN (-(1LL << (ACC_NB_BITS - 1)))
    #define ADD_SAT(x, y) MAX(ACC_MIN, MIN(ACC_MAX, ((x) + (y))))
#else
    #define ADD_SAT(x, y) ((x) + (y))
#endif

typedef struct RUNNING_MEAN {
    double mean;
    unsigned long long int count;
} RUNNING_MEAN_T;

typedef enum ACCS_REPORT { CHW, HWC } ACCS_REPORT_T;

#ifdef NL
static inline DATA_T nl32_tanh(SUM_T x)
{
    return (DATA_T)(DATA_T_MAX * tanh(x / 16129.0));
}

static inline DATA_T nl32_exp(SUM_T x)
{
    return (DATA_T)(DATA_T_MAX * exp(x / 16129.0));
}
#endif

static inline char sat8(int x)
{
    return (char)((x > 127)     ? 127 :
                  (x < -128)    ? -128 :
                                  x);
}
/*
static inline unsigned char usat8(int x) {
    return (unsigned char)((x > 255)    ? 255 :
                           (x < 0)      ? 0 :
                                          x);
}
*/
static inline DATA_T sat32(SUM_T x, char rs)
{
#if NB_BITS < 0
    const SUM_T y = x;
#else
    const SUM_T y = (x >> rs);
#endif
    return (DATA_T)((y > DATA_T_MAX) ? DATA_T_MAX :
                    (y < DATA_T_MIN) ? DATA_T_MIN :
                                       y);
}

static inline UDATA_T usat32(SUM_T x, char rs) {
#if NB_BITS < 0
    const SUM_T y = x;
#else
    const SUM_T y = (x >> rs);
#endif
    return (UDATA_T)((y > UDATA_T_MAX) ? UDATA_T_MAX :
                     (y < 0)           ? 0 :
                                         y);
}

static inline char msb32(int32_t x)
{
    int32_t px = (x < 0) ? -x : x;
    char r = 1; // sign bit
    while (px >>= 1)
        ++r;
    return r;
}
/*
static inline char umsb32(uint32_t x) {
    char r = 0;
    while (x >>= 1)
        ++r;
    return r;
}
*/

int compare(void const* a, void const* b);
int sortedFileList(const char* const dirName,
                   char*** fileList,
                   unsigned int nbMax);
void swapEndian(char* str);

void env_read(char* fileName,
              unsigned int nbChannels,
              unsigned int channelsHeight,
              unsigned int channelsWidth,
              DATA_T inputs[nbChannels][channelsHeight][channelsWidth],
              unsigned int outputsHeight,
              unsigned int outputsWidth,
              int32_t outputTargets[outputsHeight][outputsWidth]);

static inline SUM_T sht(SUM_T weightedSum, int shift) {
#if NB_BITS >= 0
    if (shift >= 0)
        return (weightedSum >> shift);
    else
        return (weightedSum << (-shift));
#endif
}

static inline DATA_T
sat(SUM_T weightedSum, ActivationFunction_T func, int shift)
{
#if NB_BITS >= 0
    if (shift > 0)
        weightedSum >>= shift;
    else if (shift < 0)
        weightedSum <<= (-shift);
#endif

    switch (func) {
    case Tanh:
    case TanhLeCun:
#if NB_BITS < 0
        return tanh(weightedSum);
#elif defined(NL)
        return nl32_tanh(weightedSum);
#endif
    case Saturation:
        return sat32(weightedSum, NB_BITS - 1);

    case Logistic:
    case LogisticWithLoss:
#if NB_BITS < 0
        return 1.0 / (1.0 + exp(-weightedSum));
#else
  #if BINARY_THRESHOLD != 0
        weightedSum >>= 2;      // divide by 4
        weightedSum += (DATA_T_MAX << (NB_BITS - 1));
        return MAX((DATA_T)0, sat32(weightedSum, NB_BITS));
  #else
        // Use the full NB_BITS dynamic. Mapping is:
        // for NB_BITS = 8: [-2,2] -> [-128,128]
        return sat32(weightedSum, NB_BITS);
  #endif
#endif

    case Rectifier:
#if NB_BITS < 0
        return MAX((SUM_T)0, weightedSum);
#else
  #if defined(UNSIGNED_DATA) && UNSIGNED_DATA
        // Keep one more bit because the output data is considered unsigned
        return usat32(MAX((SUM_T)0, weightedSum), NB_BITS - 2);
  #else
        return sat32(MAX((SUM_T)0, weightedSum), NB_BITS - 1);
  #endif
#endif

    case Linear:
#if NB_BITS < 0
        return weightedSum;
#else
        // Max value is 2^(NB_BITS-1)*2^(NB_BITS-1) = 2^(2*NB_BITS-2)
        // ex. NB_BITS = 8 ==> -128*-128=16384
        // Output max value is 2^(NB_BITS-1) ==> must be shifted by NB_BITS - 1
        // 16384>>7 = 128
        return sat32(weightedSum, NB_BITS - 1);
#endif

    default:
        fprintf(stderr, "Unsupported activation function in sat()\n");
        return 0;
    }
}

static inline DATA_T
usat(SUM_T weightedSum, ActivationFunction_T func, int shift)
{
#if NB_BITS >= 0
    if (shift > 0)
        weightedSum >>= shift;
    else if (shift < 0)
        weightedSum <<= (-shift);
#endif

    switch (func) {
    case Tanh:
    case TanhLeCun:
#if NB_BITS < 0
        return tanh(weightedSum);
#elif defined(NL)
        weightedSum >>= 1;
        return nl32_tanh(weightedSum);
#endif
    case Saturation:
        return sat32(weightedSum, NB_BITS);

    case Logistic:
    case LogisticWithLoss:
#if NB_BITS < 0
        return 1.0 / (1.0 + exp(-weightedSum));
#else
  #if BINARY_THRESHOLD != 0
        weightedSum >>= 3;      // divide by 4 & divide by 2 (because unsigned)
        weightedSum += (DATA_T_MAX << (NB_BITS - 1));
        return MAX((DATA_T)0, sat32(weightedSum, NB_BITS));
  #else
        return sat32(weightedSum, NB_BITS + 1);
  #endif
#endif

    case Rectifier:
#if NB_BITS < 0
        return MAX((SUM_T)0, weightedSum);
#else
  #if defined(UNSIGNED_DATA) && UNSIGNED_DATA
        // Keep one more bit because the output data is considered unsigned
        return usat32(MAX((SUM_T)0, weightedSum), NB_BITS - 1);
  #else
        return sat32(MAX((SUM_T)0, weightedSum), NB_BITS);
  #endif
#endif

    case Linear:
#if NB_BITS < 0
        return weightedSum;
#else
        // Max value is 2^NB_BITS*2^(NB_BITS-1) = 2^(2*NB_BITS-1)
        // ex. NB_BITS = 8 ==> -256*-128=32768
        // Output max value is 2^(NB_BITS-1) ==> must be shifted by NB_BITS
        // 32768>>8 = 128
        return sat32(weightedSum, NB_BITS);
#endif

    default:
        fprintf(stderr, "Unsupported activation function in usat()\n");
        return 0;
    }
}
void batchnormcell_propagate(
    unsigned int nbChannels,
    unsigned int channelsHeight,
    unsigned int channelsWidth,
    DATA_T inputs[nbChannels][channelsHeight][channelsWidth],
    unsigned int nbOutputs_,
    unsigned int outputOffset,
    DATA_T outputs[nbOutputs_][channelsHeight][channelsWidth],
    float bias[nbChannels],
    float variances[nbChannels],
    float means[nbChannels],
    float scales[nbChannels],
    float epsilon,
    ActivationFunction_T func);

void batchnormcell_upropagate(
    unsigned int nbChannels,
    unsigned int channelsHeight,
    unsigned int channelsWidth,
    UDATA_T inputs[nbChannels][channelsHeight][channelsWidth],
    unsigned int nbOutputs_,
    unsigned int outputOffset,
    DATA_T outputs[nbOutputs_][channelsHeight][channelsWidth],
    float bias[nbChannels],
    float variances[nbChannels],
    float means[nbChannels],
    float scales[nbChannels],
    float epsilon,
    ActivationFunction_T func);

void fmpcell_propagate(unsigned int nbChannels,
                       unsigned int channelsHeight,
                       unsigned int channelsWidth,
                       char overLap,
                       DATA_T inputs[nbChannels][channelsHeight][channelsWidth],
                       unsigned int nbOutputs_,
                       unsigned int outputsHeight,
                       unsigned int outputsWidth,
                       unsigned int nbOutputs,
                       unsigned int outputOffset,
                       DATA_T outputs[nbOutputs_][outputsHeight][outputsWidth],
                       unsigned int gridX[nbOutputs_],
                       unsigned int gridY[nbOutputs_],
                       ActivationFunction_T func);

void convcell_propagate(
    unsigned int nbChannels,
    unsigned int channelsHeight,
    unsigned int channelsWidth,
    int paddingY,
    int paddingX,
    unsigned int strideY,
    unsigned int strideX,
    unsigned int subSampleY,
    unsigned int subSampleX,
    DATA_T inputs[nbChannels][channelsHeight][channelsWidth],
    unsigned int oySize,
    unsigned int oxSize,
    unsigned int nbOutputs_,
    unsigned int outputsHeight,
    unsigned int outputsWidth,
    unsigned int nbOutputs,
    unsigned int outputOffset,
    DATA_T outputs[nbOutputs_][outputsHeight][outputsWidth],
    unsigned int kernelHeight,
    unsigned int kernelWidth,
    const BDATA_T bias[nbOutputs],
    const WDATA_T (*weights[nbOutputs][nbChannels])[kernelHeight][kernelWidth],
    ActivationFunction_T func,
    int shift);

void convcell_upropagate(
    unsigned int nbChannels,
    unsigned int channelsHeight,
    unsigned int channelsWidth,
    int paddingY,
    int paddingX,
    unsigned int strideY,
    unsigned int strideX,
    unsigned int subSampleY,
    unsigned int subSampleX,
    DATA_T inputs[nbChannels][channelsHeight][channelsWidth],
    unsigned int oySize,
    unsigned int oxSize,
    unsigned int nbOutputs_,
    unsigned int outputsHeight,
    unsigned int outputsWidth,
    unsigned int nbOutputs,
    unsigned int outputOffset,
    DATA_T outputs[nbOutputs_][outputsHeight][outputsWidth],
    unsigned int kernelHeight,
    unsigned int kernelWidth,
    const BDATA_T bias[nbOutputs],
    const WDATA_T (*weights[nbOutputs][nbChannels])[kernelHeight][kernelWidth],
    ActivationFunction_T func,
    int shift);

void convcell_propagate_accs_report(
    const char* name,
    unsigned int nbChannels,
    unsigned int channelsHeight,
    unsigned int channelsWidth,
    int paddingY,
    int paddingX,
    unsigned int strideY,
    unsigned int strideX,
    unsigned int subSampleY,
    unsigned int subSampleX,
    DATA_T inputs[nbChannels][channelsHeight][channelsWidth],
    unsigned int oySize,
    unsigned int oxSize,
    unsigned int outputsHeight,
    unsigned int outputsWidth,
    unsigned int nbOutputs,
    SUM_T* accMin,
    SUM_T* accMax,
    SUM_T* preSatMin,
    SUM_T* preSatMax,
    unsigned int kernelHeight,
    unsigned int kernelWidth,
    const BDATA_T bias[nbOutputs],
    const WDATA_T (*weights[nbOutputs][nbChannels])[kernelHeight][kernelWidth],
    ACCS_REPORT_T report);

void lccell_propagate(unsigned int nbChannels,
                      unsigned int channelsHeight,
                      unsigned int channelsWidth,
                      int paddingY,
                      int paddingX,
                      unsigned int strideY,
                      unsigned int strideX,
                      unsigned int subSampleY,
                      unsigned int subSampleX,
                      DATA_T inputs[nbChannels][channelsHeight][channelsWidth],
                      unsigned int oySize,
                      unsigned int oxSize,
                      unsigned int nbOutputs_,
                      unsigned int outputsHeight,
                      unsigned int outputsWidth,
                      unsigned int nbOutputs,
                      unsigned int outputOffset,
                      DATA_T outputs[nbOutputs_][outputsHeight][outputsWidth],
                      unsigned int kernelHeight,
                      unsigned int kernelWidth,
                      const WDATA_T bias[nbOutputs][oySize][oxSize],
                      const WDATA_T (*weights[nbOutputs][nbChannels])
                          [oySize][oxSize][kernelHeight][kernelWidth],
                      ActivationFunction_T func,
                      int shift);

void lccell_upropagate(unsigned int nbChannels,
                       unsigned int channelsHeight,
                       unsigned int channelsWidth,
                       int paddingY,
                       int paddingX,
                       unsigned int strideY,
                       unsigned int strideX,
                       unsigned int subSampleY,
                       unsigned int subSampleX,
                       DATA_T inputs[nbChannels][channelsHeight][channelsWidth],
                       unsigned int oySize,
                       unsigned int oxSize,
                       unsigned int nbOutputs_,
                       unsigned int outputsHeight,
                       unsigned int outputsWidth,
                       unsigned int nbOutputs,
                       unsigned int outputOffset,
                       DATA_T outputs[nbOutputs_][outputsHeight][outputsWidth],
                       unsigned int kernelHeight,
                       unsigned int kernelWidth,
                       const WDATA_T bias[nbOutputs][oySize][oxSize],
                       const WDATA_T (*weights[nbOutputs][nbChannels])
                           [oySize][oxSize][kernelHeight][kernelWidth],
                       ActivationFunction_T func,
                       int shift);

void
poolcell_propagate(unsigned int nbChannels,
                   unsigned int channelsHeight,
                   unsigned int channelsWidth,
                   unsigned int strideY,
                   unsigned int strideX,
                   DATA_T inputs[nbChannels][channelsHeight][channelsWidth],
                   unsigned int nbOutputs_,
                   unsigned int outputsHeight,
                   unsigned int outputsWidth,
                   unsigned int nbOutputs,
                   unsigned int outputOffset,
                   DATA_T outputs[nbOutputs_][outputsHeight][outputsWidth],
                   unsigned int poolHeight,
                   unsigned int poolWidth,
                   const char mapping[nbOutputs][nbChannels],
                   Pooling_T pooling,
                   ActivationFunction_T func,
                   int shift);

void poolcell_propagate_unitmap(
    unsigned int nbChannels,
    unsigned int channelsHeight,
    unsigned int channelsWidth,
    unsigned int strideY,
    unsigned int strideX,
    DATA_T inputs[nbChannels][channelsHeight][channelsWidth],
    unsigned int nbOutputs_,
    unsigned int outputsHeight,
    unsigned int outputsWidth,
    unsigned int nbOutputs,
    unsigned int outputOffset,
    DATA_T outputs[nbOutputs_][outputsHeight][outputsWidth],
    unsigned int poolHeight,
    unsigned int poolWidth,
    Pooling_T pooling,
    ActivationFunction_T func,
    int shift);

void
poolcell_upropagate(unsigned int nbChannels,
                    unsigned int channelsHeight,
                    unsigned int channelsWidth,
                    unsigned int strideY,
                    unsigned int strideX,
                    DATA_T inputs[nbChannels][channelsHeight][channelsWidth],
                    unsigned int nbOutputs_,
                    unsigned int outputsHeight,
                    unsigned int outputsWidth,
                    unsigned int nbOutputs,
                    unsigned int outputOffset,
                    DATA_T outputs[nbOutputs_][outputsHeight][outputsWidth],
                    unsigned int poolHeight,
                    unsigned int poolWidth,
                    const char mapping[nbOutputs][nbChannels],
                    Pooling_T pooling,
                    ActivationFunction_T func,
                    int shift);

void poolcell_upropagate_unitmap(
    unsigned int nbChannels,
    unsigned int channelsHeight,
    unsigned int channelsWidth,
    unsigned int strideY,
    unsigned int strideX,
    DATA_T inputs[nbChannels][channelsHeight][channelsWidth],
    unsigned int nbOutputs_,
    unsigned int outputsHeight,
    unsigned int outputsWidth,
    unsigned int nbOutputs,
    unsigned int outputOffset,
    DATA_T outputs[nbOutputs_][outputsHeight][outputsWidth],
    unsigned int poolHeight,
    unsigned int poolWidth,
    Pooling_T pooling,
    ActivationFunction_T func,
    int shift);
void
rbfcell_propagate_2d(unsigned int nbChannels,
                     unsigned int channelsHeight,
                     unsigned int channelsWidth,
                     DATA_T inputs[nbChannels][channelsHeight][channelsWidth],
                     unsigned int nbOutputs_,
                     unsigned int nbOutputs,
                     unsigned int outputOffset,
                     DATA_T outputs[nbOutputs_],
                     unsigned int nbChannels_,
                     uint8_t scalingPrecision,
                     const unsigned char scaling[nbOutputs],
                     const WDATA_T centers[nbOutputs][nbChannels_]);

void
rbfcell_upropagate_2d(unsigned int nbChannels,
                      unsigned int channelsHeight,
                      unsigned int channelsWidth,
                      DATA_T inputs[nbChannels][channelsHeight][channelsWidth],
                      unsigned int nbOutputs_,
                      unsigned int nbOutputs,
                      unsigned int outputOffset,
                      DATA_T outputs[nbOutputs_],
                      unsigned int nbChannels_,
                      uint8_t scalingPrecision,
                      const unsigned char scaling[nbOutputs],
                      const WDATA_T centers[nbOutputs][nbChannels_]);

void rbfcell_propagate(unsigned int nbChannels,
                       DATA_T inputs[nbChannels],
                       unsigned int nbOutputs_,
                       unsigned int nbOutputs,
                       unsigned int outputOffset,
                       DATA_T outputs[nbOutputs_],
                       uint8_t scalingPrecision,
                       const unsigned char scaling[nbOutputs],
                       const WDATA_T centers[nbOutputs][nbChannels]);
void
fccell_propagate_2d(unsigned int nbChannels,
                    unsigned int channelsHeight,
                    unsigned int channelsWidth,
                    DATA_T inputs[nbChannels][channelsHeight][channelsWidth],
                    unsigned int nbOutputs_,
                    unsigned int nbOutputs,
                    unsigned int outputOffset,
                    DATA_T outputs[nbOutputs_],
                    unsigned int nbChannels_,
                    const BDATA_T bias[nbOutputs],
                    const WDATA_T weights[nbOutputs][nbChannels_],
                    ActivationFunction_T func,
                    int shift);

void
fccell_upropagate_2d(unsigned int nbChannels,
                     unsigned int channelsHeight,
                     unsigned int channelsWidth,
                     DATA_T inputs[nbChannels][channelsHeight][channelsWidth],
                     unsigned int nbOutput_,
                     unsigned int nbOutputs,
                     unsigned int outputOffset,
                     DATA_T outputs[nbOutput_],
                     unsigned int nbChannels_,
                     const BDATA_T bias[nbOutputs],
                     const WDATA_T weights[nbOutputs][nbChannels_],
                     ActivationFunction_T func,
                     int shift);

void fccell_propagate(unsigned int nbChannels,
                      DATA_T inputs[nbChannels],
                      unsigned int nbOutputs_,
                      unsigned int nbOutputs,
                      unsigned int outputOffset,
                      DATA_T outputs[nbOutputs_],
                      const BDATA_T bias[nbOutputs],
                      const WDATA_T weights[nbOutputs][nbChannels],
                      ActivationFunction_T func,
                      int shift);

void fccell_upropagate(unsigned int nbChannels,
                      DATA_T inputs[nbChannels],
                      unsigned int nbOutputs_,
                      unsigned int nbOutputs,
                      unsigned int outputOffset,
                      DATA_T outputs[nbOutputs_],
                      const BDATA_T bias[nbOutputs],
                      const WDATA_T weights[nbOutputs][nbChannels],
                      ActivationFunction_T func,
                      int shift);

void fccell_propagate_2d_sparse(
    unsigned int nbChannels,
    unsigned int channelsHeight,
    unsigned int channelsWidth,
    DATA_T inputs[nbChannels][channelsHeight][channelsWidth],
    unsigned int nbOutputs_,
    unsigned int nbOutputs,
    unsigned int outputOffset,
    DATA_T outputs[nbOutputs_],
    unsigned int nbChannels_,
    const BDATA_T bias[nbOutputs],
    unsigned int nbWeights,
    const WDATA_T weights[nbWeights],
    const unsigned short offsets[nbWeights],
    ActivationFunction_T func,
    int shift);

void fccell_upropagate_2d_sparse(
    unsigned int nbChannels,
    unsigned int channelsHeight,
    unsigned int channelsWidth,
    DATA_T inputs[nbChannels][channelsHeight][channelsWidth],
    unsigned int nbOutput_,
    unsigned int nbOutputs,
    unsigned int outputOffset,
    DATA_T outputs[nbOutput_],
    unsigned int nbChannels_,
    const BDATA_T bias[nbOutputs],
    unsigned int nbWeights,
    const WDATA_T weights[nbWeights],
    const unsigned short offsets[nbWeights],
    ActivationFunction_T func,
    int shift);

void fccell_propagate_sparse(unsigned int nbChannels,
                             DATA_T inputs[nbChannels],
                             unsigned int nbOutputs_,
                             unsigned int nbOutputs,
                             unsigned int outputOffset,
                             DATA_T outputs[nbOutputs_],
                             const BDATA_T bias[nbOutputs],
                             unsigned int nbWeights,
                             const WDATA_T weights[nbWeights],
                             const unsigned short offsets[nbWeights],
                             ActivationFunction_T func,
                             int shift);

void output_max(unsigned int nbOutputs,
                DATA_T outputs[nbOutputs],
                uint32_t outputEstimated[1][1]);

void spatial_output_max(unsigned int nbOutputs,
                        unsigned int outputsHeight,
                        unsigned int outputsWidth,
                        DATA_T outputs[nbOutputs][outputsHeight][outputsWidth],
                        uint32_t outputEstimated[outputsHeight][outputsWidth]);

void
softmaxcell_propagate(unsigned int nbOutputs,
                      unsigned int outputsHeight,
                      unsigned int outputsWidth,
                      DATA_T inputs[nbOutputs][outputsHeight][outputsWidth],
                      DATA_T outputs[nbOutputs][outputsHeight][outputsWidth]);

void resize_bilinear_tf_propagete(unsigned int nbChannels,
                                  unsigned int channelsHeight,
                                  unsigned int channelsWidth,
                                  DATA_T inputs[nbChannels][channelsHeight][channelsWidth],
                                  unsigned int nbOutputs,
                                  unsigned int outputsHeight,
                                  unsigned int outputsWidth,
                                  DATA_T outputs[nbOutputs][outputsHeight][outputsWidth],
                                  const Interpolation interpolationHeight[outputsHeight],
                                  const Interpolation interpolationWidth[outputsWidth]);

void resize_nearest_neighbor_propagete(unsigned int nbChannels,
                                       unsigned int channelsHeight,
                                       unsigned int channelsWidth,
                                       DATA_T inputs[nbChannels][channelsHeight][channelsWidth],
                                       unsigned int nbOutputs,
                                       unsigned int outputsHeight,
                                       unsigned int outputsWidth,
                                       DATA_T outputs[nbOutputs][outputsHeight][outputsWidth]);

void convcell_outputs_print(const char* name,
                            unsigned int nbOutputs,
                            unsigned int outputsHeight,
                            unsigned int outputsWidth,
                            DATA_T outputs[nbOutputs][outputsHeight][outputsWidth]);

void convcell_outputs_save(const char* fileName,
                           unsigned int nbOutputs,
                           unsigned int outputsHeight,
                           unsigned int outputsWidth,
                           DATA_T outputs[nbOutputs][outputsHeight][outputsWidth]);

void convcell_outputs_dynamic_print(
    const char* name,
    unsigned int nbOutputs,
    unsigned int outputsHeight,
    unsigned int outputsWidth,
    DATA_T outputs[nbOutputs][outputsHeight][outputsWidth],
    DATA_T* pMinVal,
    DATA_T* pMaxVal,
    RUNNING_MEAN_T* pMeanVal);

void fccell_outputs_print(const char* name,
                          unsigned int nbOutputs,
                          DATA_T outputs[nbOutputs]);

void fccell_outputs_save(const char* fileName,
                         unsigned int nbOutputs,
                         DATA_T outputs[nbOutputs]);

void fccell_outputs_dynamic_print(const char* name,
                                  unsigned int nbOutputs,
                                  DATA_T outputs[nbOutputs],
                                  DATA_T* pMinVal,
                                  DATA_T* pMaxVal,
                                  RUNNING_MEAN_T* pMeanVal);

void confusion_print(unsigned int nbOutputs,
                     unsigned int confusion[nbOutputs][nbOutputs]);

void time_analysis(const char* name,
                   struct timeval start,
                   struct timeval end,
                   RUNNING_MEAN_T* timing);

#endif // N2D2_EXPORTC_DEEPNET_H
