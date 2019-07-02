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

#include "n2d2.h"

int compare(void const* a, void const* b)
{
    char const* const* pa = a;
    char const* const* pb = b;
    return strcmp(*pa, *pb);
}

int sortedFileList(const char* const dirName,
                   char*** fileList,
                   unsigned int nbMax)
{
    int count = 0;

#ifndef NO_DIRENT
    DIR* pDir = opendir(dirName);

    if (pDir == NULL)
        fprintf(stderr,
                "Couldn't open the directory for input patterns: %s\n",
                dirName);

    struct dirent* pFile;

    // Count the number of files
    while ((pFile = readdir(pDir))) {
        if (pFile->d_name[0] == '.')
            continue;

        ++count;
    }
#else
    char* listName = malloc((strlen(dirName) + strlen(".list")
                                     + 1) * sizeof(char));
    sprintf(listName, "%s.list", dirName);

    FILE *fList = fopen(listName, "r");
    free(listName);

    if (fList == NULL) {
        printf("Error opening .list file!\n");
        exit(1);
    }

    char* line = NULL;
    size_t len = 0;
    int read;

    /* Count the number of files */
    while (((read = getline(&line, &len, fList)) != -1)
        && (nbMax == 0 || count < nbMax))
    {
        ++count;
    }
#endif

    // Allocate enough space
    (*fileList) = malloc(count * sizeof(*(*fileList)));

    if ((*fileList) == NULL) {
#ifndef NO_DIRENT
        closedir(pDir);
#else
        fclose(fList);
#endif
        fprintf(stderr, "sortedFileList(): 'list' memory allocation failed\n");
        return -1;
    }

    count = 0;

#ifndef NO_DIRENT
    rewinddir(pDir); /* reset position */

    while ((pFile = readdir(pDir)) && (nbMax == 0 || count < nbMax)) {
        if (pFile->d_name[0] == '.')
            continue;

        /* + 2 because of the '/' and the terminating 0 */
        (*fileList)[count] = malloc((strlen(dirName) + strlen(pFile->d_name)
                                     + 2) * sizeof(*(*fileList)[count]));

        if ((*fileList)[count] == NULL) {
            closedir(pDir);
            fprintf(stderr,
                    "sortedFileList(): 'fileName' memory allocation failed\n");
            return -1;
        }

        sprintf((*fileList)[count], "%s/%s", dirName, pFile->d_name);
        ++count;
    }

    closedir(pDir);

    if (nbMax > 0) {
        printf("sortedFileList(): warning: nbMax > 0 may lead to globally"
               " unsorted stimuli!\n");
    }
#else
    rewind(fList);

    while (((read = getline(&line, &len, fList)) != -1)
        && (nbMax == 0 || count < nbMax))
    {
        (*fileList)[count] = malloc(read * sizeof(*(*fileList)[count]));
        line[read - 1] = '\0'; // skip \n

        sprintf((*fileList)[count], "%s", line);
        ++count;
    }

    fclose(fList);
    free(line);
#endif

    qsort((*fileList), count, sizeof(*(*fileList)), compare);
    return count;
}

void swapEndian(char* str) {
    char tmp = str[7];
    str[7] = str[1];
    str[1] = tmp;
    tmp = str[6];
    str[6] = str[0];
    str[0] = tmp;
    tmp = str[5];
    str[5] = str[3];
    str[3] = tmp;
    tmp = str[4];
    str[4] = str[2];
    str[2] = tmp;
}

void env_read(char* fileName,
              unsigned int nbChannels,
              unsigned int channelsHeight,
              unsigned int channelsWidth,
              DATA_T inputs[nbChannels][channelsHeight][channelsWidth],
              unsigned int outputsHeight,
              unsigned int outputsWidth,
              int32_t outputTargets[outputsHeight][outputsWidth])
{
    FILE* filePtr = fopen(fileName, "rb");

    if (filePtr == NULL) {
        fprintf(stderr, "Unable to open file \"%s\"\n", fileName);
        perror("The following error occurred");
    }

    char header[2];
    fread(&header[0], sizeof(header[0]), 2, filePtr);

    if (header[0] != 'P' || (header[1] != '5' && header[1] != '6')) {
        fprintf(stderr, "Unknown PGM file format for file %s\n", fileName);
        fclose(filePtr);
        return;
    }

    int pixelWidth;
    int pixelHeight;
    int maxValue;
    fscanf(filePtr, "%d %d %d", &pixelWidth, &pixelHeight, &maxValue);
    fgetc(filePtr);

    if (pixelWidth != (int)channelsWidth || pixelHeight
                                            != (int)channelsHeight) {
        fprintf(stderr,
                "PGM image size does not match array size for file %s\n",
                fileName);
        fclose(filePtr);
        return;
    }

    size_t nbRead = 0;

#if NB_BITS > 0 && NB_BITS != 8 && NB_BITS != 16 && NB_BITS != 32 && NB_BITS   \
                                                                     != 64
#if NB_BITS > 0 && NB_BITS < 8
    char inputs_fixed[nbChannels][channelsHeight][channelsWidth];
#elif NB_BITS > 8 && NB_BITS < 16
    short inputs_fixed[nbChannels][channelsHeight][channelsWidth];
#elif NB_BITS > 16 && NB_BITS < 32
    int inputs_fixed[nbChannels][channelsHeight][channelsWidth];
#elif NB_BITS > 32 && NB_BITS < 64
    long long int inputs_fixed[nbChannels][channelsHeight][channelsWidth];
#endif
    nbRead = fread(inputs_fixed, sizeof(inputs_fixed[0]), nbChannels, filePtr);

    for (unsigned int channel = 0; channel < nbChannels; ++channel) {
        for (unsigned int iy = 0; iy < channelsHeight; ++iy) {
            for (unsigned int ix = 0; ix < channelsWidth; ++ix)
                inputs[channel][iy][ix] = (DATA_T)inputs_fixed[channel][iy][ix];
        }
    }
#else
    nbRead = fread(inputs, sizeof(inputs[0]), nbChannels, filePtr);
#endif

    if (nbRead != nbChannels)
        fprintf(stderr,
                "fread() number of read objects different than expected\n");

    /*
        for (unsigned int channel = 0; channel < nbChannels; ++channel) {
            // DEBUG
            for (unsigned int iy = 0; iy < channelsHeight; ++iy) {
                for (unsigned int ix = 0; ix < channelsWidth; ++ix) {
                    printf("%d", inputs[channel][iy][ix]);
                }

                printf("\n");
            }
        }
    */
    nbRead = fread(
        outputTargets, sizeof(outputTargets[0]), outputsHeight, filePtr);

    if (nbRead != outputsHeight)
        fprintf(stderr,
                "fread() number of read objects different than expected\n");

    if (feof(filePtr))
        fprintf(stderr,
                "End-of-file reached prematurely in data file: %s\n",
                fileName);
    else if (ferror(filePtr))
        fprintf(stderr, "Error while reading data file: %s\n", fileName);
    else if (fgetc(filePtr) != EOF)
        fprintf(stderr, "Data file size larger than expected: %s\n", fileName);

    fclose(filePtr);
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
    ActivationFunction_T func)
{
#if NB_BITS > 0
    fprintf(stderr,
            "batchnormcell_propagate(): only 32-bits floating point is "
            "implemented\n");
    return;
#else
    for (unsigned int output = 0; output < nbChannels; ++output) {
        const float var = sqrt(variances[output] + epsilon);

        for (unsigned int oy = 0; oy < channelsHeight; ++oy) {
            for (unsigned int ox = 0; ox < channelsWidth; ++ox) {
                const float normalized
                    = (inputs[output][oy][ox] - means[output]) / var;
                const float sAs = scales[output] * normalized + bias[output];
                outputs[output + outputOffset][oy][ox] = sat(sAs, func, 0);
            }
        }
    }
#endif
}

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
    ActivationFunction_T func)
{
#if NB_BITS > 0
    fprintf(stderr,
            "batchnormcell_upropagate(): only 32-bits floating point "
            "is implemented\n");
    return;
#else
    for (unsigned int output = 0; output < nbChannels; ++output) {
        const float var = sqrt(variances[output] + epsilon);

        for (unsigned int oy = 0; oy < channelsHeight; ++oy) {
            for (unsigned int ox = 0; ox < channelsWidth; ++ox) {
                const float normalized
                    = (inputs[output][oy][ox] - means[output]) / var;
                const float sAs = scales[output] * normalized + bias[output];
                outputs[output + outputOffset][oy][ox] = usat(sAs, func, 0);
            }
        }
    }
#endif
}
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
                       ActivationFunction_T func)
{
#if NB_BITS < 0
    if (func != Linear) {
        fprintf(stderr, "fmpcell_propagate(): only Linear is implemented\n");
        return;
    }
#endif

    for (unsigned int output = 0; output < nbOutputs; ++output) {
        for (unsigned int oy = 0; oy < outputsHeight; ++oy) {
            for (unsigned int ox = 0; ox < outputsWidth; ++ox) {
                DATA_T poolValue = DATA_T_MIN;

                const unsigned int ixStart = (ox > 0) ? gridX[ox - 1] : 0;
                const unsigned int iyStart = (oy > 0) ? gridY[oy - 1] : 0;
                unsigned int ixStop = gridX[ox];
                unsigned int iyStop = gridY[oy];

                if (overLap == 0) {
                    --ixStop;
                    --iyStop;
                }

                if (ox == outputsWidth - 1)
                    ixStop = channelsWidth - 1;

                if (oy == outputsHeight - 1)
                    iyStop = channelsHeight - 1;

                for (unsigned int sy = iyStart; sy <= iyStop; ++sy) {
                    for (unsigned int sx = ixStart; sx <= ixStop; ++sx) {
                        const unsigned int ix = sx;
                        const unsigned int iy = sy;

                        if (inputs[output][iy][ix] > poolValue)
                            poolValue = inputs[output][iy][ix];
                    }
                }
#if NB_BITS < 0
                outputs[outputOffset + output][oy][ox]
                    = sat(poolValue, func, 0);
#else
                outputs[outputOffset + output][oy][ox] = poolValue;
#endif
            }
        }
    }
}

#if defined(_OPENMP) && _OPENMP >= 200805
#define CONV_COLLAPSE collapse(3)

#define DECLARE_CONVCELL_PROPAGATE_TYPE(PREFIX, TYPE, M_kernelHeight, M_kernelWidth) \
void convcell_##PREFIX##propagate_##M_kernelHeight##x##M_kernelWidth( \
    unsigned int nbChannels, \
    unsigned int channelsHeight, \
    unsigned int channelsWidth, \
    int paddingY, \
    int paddingX, \
    unsigned int strideY, \
    unsigned int strideX, \
    unsigned int subSampleY, \
    unsigned int subSampleX, \
    DATA_T inputs[nbChannels][channelsHeight][channelsWidth], \
    unsigned int oySize, \
    unsigned int oxSize, \
    unsigned int nbOutputs_, \
    unsigned int outputsHeight, \
    unsigned int outputsWidth, \
    unsigned int nbOutputs, \
    unsigned int outputOffset, \
    DATA_T outputs[nbOutputs_][outputsHeight][outputsWidth], \
    const BDATA_T bias[nbOutputs], \
    const WDATA_T (*weights[nbOutputs][nbChannels])[M_kernelHeight][M_kernelWidth], \
    ActivationFunction_T func, \
    int shift) \
{ \
    _Pragma(STR(omp parallel for CONV_COLLAPSE)) \
    for (unsigned int output = 0; output < nbOutputs; ++output) { \
        for (unsigned int oy = 0; oy < oySize; ++oy) { \
            for (unsigned int ox = 0; ox < oxSize; ++ox) { \
                const unsigned int sxMin = (unsigned int)int_max( \
                    (int)paddingX - (int)(ox * strideX), 0); \
                const unsigned int syMin = (unsigned int)int_max( \
                    (int)paddingY - (int)(oy * strideY), 0); \
                const unsigned int sxMax = (unsigned int)int_max( \
                    int_min((int)channelsWidth + paddingX - (int)(ox * strideX), \
                            (int)M_kernelWidth), \
                    0); \
                const unsigned int syMax = (unsigned int)int_max( \
                    int_min((int)channelsHeight + paddingY \
                            - (int)(oy * strideY), \
                            (int)M_kernelHeight), \
                    0); \
 \
                const int ix = (int)(ox * strideX) - (int)paddingX; \
                const int iy = (int)(oy * strideY) - (int)paddingY; \
 \
                SUM_T weightedSum = bias[output]; \
 \
                for (unsigned int channel = 0; channel < nbChannels; \
                     ++channel) { \
                    if (weights[output][channel] == NULL) \
                        continue; \
 \
                    for (unsigned int sy = syMin; sy != syMax; ++sy) { \
                        _Pragma(STR(unroll M_kernelWidth)) \
                        for (unsigned int sx = 0; sx != M_kernelWidth; ++sx) { \
                            if (sx >= sxMin && sx < sxMax) { \
                                weightedSum = ADD_SAT(weightedSum, (SUM_T)( \
                                           *weights[output][channel])[sy][sx] \
                                       * (SUM_T)( \
                                             (TYPE) \
                                             inputs[channel][iy + sy][ix + sx])); \
                            } \
                        } \
                    } \
                } \
 \
                outputs[outputOffset + output][oy][ox] \
                    = PREFIX##sat(weightedSum, func, shift); \
            } \
        } \
    } \
}
#else
#define CONV_COLLAPSE

#define DECLARE_CONVCELL_PROPAGATE_TYPE(PREFIX, TYPE, M_kernelHeight, M_kernelWidth) \
void convcell_##PREFIX##propagate_##M_kernelHeight##x##M_kernelWidth( \
    unsigned int nbChannels, \
    unsigned int channelsHeight, \
    unsigned int channelsWidth, \
    int paddingY, \
    int paddingX, \
    unsigned int strideY, \
    unsigned int strideX, \
    unsigned int subSampleY, \
    unsigned int subSampleX, \
    DATA_T inputs[nbChannels][channelsHeight][channelsWidth], \
    unsigned int oySize, \
    unsigned int oxSize, \
    unsigned int nbOutputs_, \
    unsigned int outputsHeight, \
    unsigned int outputsWidth, \
    unsigned int nbOutputs, \
    unsigned int outputOffset, \
    DATA_T outputs[nbOutputs_][outputsHeight][outputsWidth], \
    const BDATA_T bias[nbOutputs], \
    const WDATA_T (*weights[nbOutputs][nbChannels])[M_kernelHeight][M_kernelWidth], \
    ActivationFunction_T func, \
    int shift) \
{ \
    _Pragma(STR(omp parallel for CONV_COLLAPSE)) \
    for (unsigned int output = 0; output < nbOutputs; ++output) { \
        for (unsigned int oy = 0; oy < oySize; ++oy) { \
            for (unsigned int ox = 0; ox < oxSize; ++ox) { \
                const unsigned int sxMin = (unsigned int)int_max( \
                    (int)paddingX - (int)(ox * strideX), 0); \
                const unsigned int syMin = (unsigned int)int_max( \
                    (int)paddingY - (int)(oy * strideY), 0); \
                const unsigned int sxMax = (unsigned int)int_max( \
                    int_min((int)channelsWidth + paddingX - (int)(ox * strideX), \
                            (int)M_kernelWidth), \
                    0); \
                const unsigned int syMax = (unsigned int)int_max( \
                    int_min((int)channelsHeight + paddingY \
                            - (int)(oy * strideY), \
                            (int)M_kernelHeight), \
                    0); \
 \
                const int ix = (int)(ox * strideX) - (int)paddingX; \
                const int iy = (int)(oy * strideY) - (int)paddingY; \
 \
                SUM_T weightedSum = bias[output]; \
 \
                for (unsigned int channel = 0; channel < nbChannels; \
                     ++channel) { \
                    if (weights[output][channel] == NULL) \
                        continue; \
 \
                    _Pragma(STR(unroll M_kernelHeight)) \
                    for (unsigned int sy = 0; sy < M_kernelHeight; ++sy) { \
                        _Pragma(STR(unroll M_kernelWidth)) \
                        for (unsigned int sx = 0; sx < M_kernelWidth; ++sx) { \
                            if (sx >= sxMin && sx < sxMax \
                                && sy >= syMin && sy < syMax) \
                            { \
                                weightedSum = ADD_SAT(weightedSum, (SUM_T)( \
                                           *weights[output][channel])[sy][sx] \
                                       * (SUM_T)( \
                                             (TYPE) \
                                             inputs[channel][iy + sy][ix + sx])); \
                            } \
                        } \
                    } \
                } \
 \
                outputs[outputOffset + output][oy][ox] \
                    = PREFIX##sat(weightedSum, func, shift); \
            } \
        } \
    } \
}
#endif

#define DECLARE_CONVCELL_PROPAGATE(M_kernelHeight, M_kernelWidth) \
    DECLARE_CONVCELL_PROPAGATE_TYPE(, DATA_T, M_kernelHeight, M_kernelWidth)

#define DECLARE_CONVCELL_UPROPAGATE(M_kernelHeight, M_kernelWidth) \
    DECLARE_CONVCELL_PROPAGATE_TYPE(u, UDATA_T, M_kernelHeight, M_kernelWidth)

#define CONVCELL_PROPAGATE(M_kernelHeight, M_kernelWidth) do { \
    if (kernelHeight == M_kernelHeight && kernelWidth == M_kernelWidth) { \
        convcell_propagate_##M_kernelHeight##x##M_kernelWidth(nbChannels, channelsHeight, channelsWidth, paddingY, paddingX, \
            strideY, strideX, subSampleY, subSampleX, inputs, oySize, oxSize, nbOutputs_, outputsHeight, outputsWidth, nbOutputs, \
            outputOffset, outputs, bias, weights, func, shift); \
        return; \
    } \
} while (0)

#define CONVCELL_UPROPAGATE(M_kernelHeight, M_kernelWidth) do { \
    if (kernelHeight == M_kernelHeight && kernelWidth == M_kernelWidth) { \
        convcell_upropagate_##M_kernelHeight##x##M_kernelWidth(nbChannels, channelsHeight, channelsWidth, paddingY, paddingX, \
            strideY, strideX, subSampleY, subSampleX, inputs, oySize, oxSize, nbOutputs_, outputsHeight, outputsWidth, nbOutputs, \
            outputOffset, outputs, bias, weights, func, shift); \
        return; \
    } \
} while (0)

DECLARE_CONVCELL_PROPAGATE(1, 1)
DECLARE_CONVCELL_PROPAGATE(3, 3)
DECLARE_CONVCELL_PROPAGATE(5, 5)

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
    int shift)
{
    if (subSampleY != 1 || subSampleX != 1) {
        fprintf(stderr, "convcell_propagate(): subsampling not implemented\n");
        return;
    }

    // Specialized functions
    CONVCELL_PROPAGATE(1, 1);
    CONVCELL_PROPAGATE(3, 3);
    CONVCELL_PROPAGATE(5, 5);

    // Fallback
#if defined(_OPENMP) && _OPENMP >= 200805
#pragma omp parallel for collapse(3)
#else
#pragma omp parallel for
#endif
    for (unsigned int output = 0; output < nbOutputs; ++output) {
        for (unsigned int oy = 0; oy < oySize; ++oy) {
            for (unsigned int ox = 0; ox < oxSize; ++ox) {
                const unsigned int sxMin = (unsigned int)int_max(
                    (int)paddingX - (int)(ox * strideX), 0);
                const unsigned int syMin = (unsigned int)int_max(
                    (int)paddingY - (int)(oy * strideY), 0);
                const unsigned int sxMax = (unsigned int)int_max(
                    int_min((int)channelsWidth + paddingX - (int)(ox * strideX),
                            (int)kernelWidth),
                    0);
                const unsigned int syMax = (unsigned int)int_max(
                    int_min((int)channelsHeight + paddingY
                            - (int)(oy * strideY),
                            (int)kernelHeight),
                    0);

                const int ix = (int)(ox * strideX) - (int)paddingX;
                const int iy = (int)(oy * strideY) - (int)paddingY;

                SUM_T weightedSum = bias[output];

                for (unsigned int channel = 0; channel < nbChannels;
                     ++channel) {
                    if (weights[output][channel] == NULL)
                        continue;

                    for (unsigned int sy = syMin; sy < syMax; ++sy) {
                        for (unsigned int sx = sxMin; sx < sxMax; ++sx)
                            weightedSum = ADD_SAT(weightedSum,
                                (*weights[output][channel])[sy][sx]
                                   * inputs[channel][iy + sy][ix + sx]);
                    }
                }

                outputs[outputOffset + output][oy][ox]
                    = sat(weightedSum, func, shift);
            }
        }
    }
}

DECLARE_CONVCELL_UPROPAGATE(1, 1)
DECLARE_CONVCELL_UPROPAGATE(3, 3)
DECLARE_CONVCELL_UPROPAGATE(5, 5)

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
    int shift)
{
    if (subSampleY != 1 || subSampleX != 1) {
        fprintf(stderr, "convcell_upropagate(): subsampling not implemented\n");
        return;
    }

    // Specialized functions
    CONVCELL_UPROPAGATE(1, 1);
    CONVCELL_UPROPAGATE(3, 3);
    CONVCELL_UPROPAGATE(5, 5);

    // Fallback
#if defined(_OPENMP) && _OPENMP >= 200805
#pragma omp parallel for collapse(3)
#else
#pragma omp parallel for
#endif
    for (unsigned int output = 0; output < nbOutputs; ++output) {
        for (unsigned int oy = 0; oy < oySize; ++oy) {
            for (unsigned int ox = 0; ox < oxSize; ++ox) {
                const unsigned int sxMin = (unsigned int)int_max(
                    (int)paddingX - (int)(ox * strideX), 0);
                const unsigned int syMin = (unsigned int)int_max(
                    (int)paddingY - (int)(oy * strideY), 0);
                const unsigned int sxMax = (unsigned int)int_max(
                    int_min((int)channelsWidth + paddingX - (int)(ox * strideX),
                            (int)kernelWidth),
                    0);
                const unsigned int syMax = (unsigned int)int_max(
                    int_min((int)channelsHeight + paddingY
                            - (int)(oy * strideY),
                            (int)kernelHeight),
                    0);

                const int ix = (int)(ox * strideX) - (int)paddingX;
                const int iy = (int)(oy * strideY) - (int)paddingY;

                SUM_T weightedSum = bias[output];

                for (unsigned int channel = 0; channel < nbChannels;
                     ++channel) {
                    if (weights[output][channel] == NULL)
                        continue;

                    for (unsigned int sy = syMin; sy < syMax; ++sy) {
                        for (unsigned int sx = sxMin; sx < sxMax; ++sx) {
                            weightedSum = ADD_SAT(weightedSum, (SUM_T)(
                                       *weights[output][channel])[sy][sx]
                                   * (SUM_T)(
                                         (UDATA_T)
                                         inputs[channel][iy + sy][ix + sx]));
                        }
                    }
                }

                outputs[outputOffset + output][oy][ox]
                    = usat(weightedSum, func, shift);
            }
        }
    }
}

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
    ACCS_REPORT_T report)
{
    if (subSampleY != 1 || subSampleX != 1) {
        fprintf(stderr, "convcell_propagate_accs_report(): subsampling not"
                " implemented\n");
        return;
    }

#if defined(_OPENMP) && _OPENMP >= 200805
#pragma omp parallel for collapse(3)
#else
#pragma omp parallel for
#endif
    for (unsigned int output = 0; output < nbOutputs; ++output) {
        for (unsigned int oy = 0; oy < oySize; ++oy) {
            for (unsigned int ox = 0; ox < oxSize; ++ox) {
                const unsigned int sxMin = (unsigned int)int_max(
                    (int)paddingX - (int)(ox * strideX), 0);
                const unsigned int syMin = (unsigned int)int_max(
                    (int)paddingY - (int)(oy * strideY), 0);
                const unsigned int sxMax = (unsigned int)int_max(
                    int_min((int)channelsWidth + paddingX - (int)(ox * strideX),
                            (int)kernelWidth),
                    0);
                const unsigned int syMax = (unsigned int)int_max(
                    int_min((int)channelsHeight + paddingY
                            - (int)(oy * strideY),
                            (int)kernelHeight),
                    0);

                const int ix = (int)(ox * strideX) - (int)paddingX;
                const int iy = (int)(oy * strideY) - (int)paddingY;

                SUM_T outAccMin = 0;
                SUM_T outAccMax = 0;

                SUM_T weightedSum = bias[output];

                if (weightedSum > outAccMax)
                    outAccMax = weightedSum;

                if (weightedSum < outAccMin)
                    outAccMin = weightedSum;

                if (report == CHW) {
                    for (unsigned int channel = 0; channel < nbChannels;
                         ++channel)
                    {
                        if (weights[output][channel] == NULL)
                            continue;

                        for (unsigned int sy = syMin; sy < syMax; ++sy) {
                            for (unsigned int sx = sxMin; sx < sxMax; ++sx) {
                                weightedSum
                                    += (*weights[output][channel])[sy][sx]
                                       * inputs[channel][iy + sy][ix + sx];

                                if (weightedSum > outAccMax)
                                    outAccMax = weightedSum;

                                if (weightedSum < outAccMin)
                                    outAccMin = weightedSum;
                            }
                        }
                    }
                }
                else if (report == HWC) {
                    for (unsigned int sy = syMin; sy < syMax; ++sy) {
                        for (unsigned int sx = sxMin; sx < sxMax; ++sx) {
                            for (unsigned int channel = 0; channel < nbChannels;
                                 ++channel)
                            {
                                if (weights[output][channel] == NULL)
                                    continue;

                                weightedSum
                                    += (*weights[output][channel])[sy][sx]
                                       * inputs[channel][iy + sy][ix + sx];

                                if (weightedSum > outAccMax)
                                    outAccMax = weightedSum;

                                if (weightedSum < outAccMin)
                                    outAccMin = weightedSum;
                            }
                        }
                    }
                }

                #pragma critical
                {
                    if (weightedSum > (*preSatMax))
                        (*preSatMax) = weightedSum;

                    if (weightedSum < (*preSatMin))
                        (*preSatMin) = weightedSum;

                    if (outAccMax > (*accMax))
                        (*accMax) = outAccMax;

                    if (outAccMin < (*accMin))
                        (*accMin) = outAccMin;
                }
            }
        }
    }

    const SUM_T maxAccVal = MAX(abs(*accMin), (*accMax));
    const unsigned int accNbBits = ceil(log2(maxAccVal)) + 1;
        // +1 for sign bit

    printf("%s acc. dynamic     = [%d, %d] = %d bits\n", name,
           (*accMin), (*accMax), accNbBits);

    const SUM_T maxPreSatVal = MAX(abs(*preSatMin), (*preSatMax));
    const unsigned int preSatNbBits = ceil(log2(maxPreSatVal)) + 1;
        // +1 for sign bit

    printf("%s pre-sat. dynamic = [%d, %d] = %d bits\n", name,
           (*preSatMin), (*preSatMax), preSatNbBits);
}

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
                      int shift)
{
    if (subSampleY != 1 || subSampleX != 1) {
        fprintf(stderr, "lccell_propagate(): subsampling not implemented\n");
        return;
    }

#if defined(_OPENMP) && _OPENMP >= 200805
#pragma omp parallel for collapse(3)
#else
#pragma omp parallel for
#endif
    for (unsigned int output = 0; output < nbOutputs; ++output) {
        for (unsigned int oy = 0; oy < oySize; ++oy) {
            for (unsigned int ox = 0; ox < oxSize; ++ox) {
                const unsigned int sxMin = (unsigned int)int_max(
                    (int)paddingX - (int)(ox * strideX), 0);
                const unsigned int syMin = (unsigned int)int_max(
                    (int)paddingY - (int)(oy * strideY), 0);
                const unsigned int sxMax = (unsigned int)int_max(
                    int_min((int)channelsWidth + paddingX - (int)(ox * strideX),
                            (int)kernelWidth),
                    0);
                const unsigned int syMax = (unsigned int)int_max(
                    int_min((int)channelsHeight + paddingY
                            - (int)(oy * strideY),
                            (int)kernelHeight),
                    0);

                SUM_T weightedSum = bias[output][oy][ox];

                for (unsigned int channel = 0; channel < nbChannels;
                     ++channel) {
                    if (weights[output][channel] == NULL)
                        continue;

                    for (unsigned int sy = syMin; sy < syMax; ++sy) {
                        for (unsigned int sx = sxMin; sx < sxMax; ++sx) {
                            const int ix = (int)(ox * strideX + sx)
                                           - (int)paddingX;
                            const int iy = (int)(oy * strideY + sy)
                                           - (int)paddingY;

                            weightedSum = ADD_SAT(weightedSum,
                                (*weights[output][channel])[oy][ox][sy][sx]
                                   * inputs[channel][iy][ix]);
                        }
                    }
                }

                outputs[outputOffset + output][oy][ox]
                    = sat(weightedSum, func, shift);
            }
        }
    }
}

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
                       int shift)
{
    if (subSampleY != 1 || subSampleX != 1) {
        fprintf(stderr, "lccell_upropagate(): subsampling not implemented\n");
        return;
    }

#if defined(_OPENMP) && _OPENMP >= 200805
#pragma omp parallel for collapse(3)
#else
#pragma omp parallel for
#endif
    for (unsigned int output = 0; output < nbOutputs; ++output) {
        for (unsigned int oy = 0; oy < oySize; ++oy) {
            for (unsigned int ox = 0; ox < oxSize; ++ox) {
                const unsigned int sxMin = (unsigned int)int_max(
                    (int)paddingX - (int)(ox * strideX), 0);
                const unsigned int syMin = (unsigned int)int_max(
                    (int)paddingY - (int)(oy * strideY), 0);
                const unsigned int sxMax = (unsigned int)int_max(
                    int_min((int)channelsWidth + paddingX - (int)(ox * strideX),
                            (int)kernelWidth),
                    0);
                const unsigned int syMax = (unsigned int)int_max(
                    int_min((int)channelsHeight + paddingY
                            - (int)(oy * strideY),
                            (int)kernelHeight),
                    0);

                SUM_T weightedSum = bias[output][oy][ox];

                for (unsigned int channel = 0; channel < nbChannels;
                     ++channel) {
                    if (weights[output][channel] == NULL)
                        continue;

                    for (unsigned int sy = syMin; sy < syMax; ++sy) {
                        for (unsigned int sx = sxMin; sx < sxMax; ++sx) {
                            const int ix = (int)(ox * strideX + sx)
                                           - (int)paddingX;
                            const int iy = (int)(oy * strideY + sy)
                                           - (int)paddingY;

                            weightedSum = ADD_SAT(weightedSum,
                                (SUM_T)(*weights[output][channel])[oy][ox]
                                                                     [sy][sx]
                                   * (SUM_T)((UDATA_T)inputs[channel][iy][ix]));
                        }
                    }
                }

                outputs[outputOffset + output][oy][ox]
                    = usat(weightedSum, func, shift);
            }
        }
    }
}

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
                   int shift)
{
#if NB_BITS < 0
    if (func != Linear) {
        fprintf(stderr, "poolcell_propagate(): only Linear is implemented\n");
        return;
    }
#endif

#if defined(_OPENMP) && _OPENMP >= 200805
#pragma omp parallel for collapse(3)
#else
#pragma omp parallel for
#endif
    for (unsigned int output = 0; output < nbOutputs; ++output) {
        for (unsigned int oy = 0; oy < outputsHeight; ++oy) {
            for (unsigned int ox = 0; ox < outputsWidth; ++ox) {
                const unsigned int sxMax
                    = uint_min(channelsWidth - ox * strideX, poolWidth);
                const unsigned int syMax
                    = uint_min(channelsHeight - oy * strideY, poolHeight);

                if (pooling == Max) {
                    DATA_T poolValue = DATA_T_MIN;

                    for (unsigned int channel = 0; channel < nbChannels;
                         ++channel) {
                        if (!mapping[output][channel])
                            continue;

                        for (unsigned int sy = 0; sy < syMax; ++sy) {
                            for (unsigned int sx = 0; sx < sxMax; ++sx) {
                                const unsigned int ix = ox * strideX + sx;
                                const unsigned int iy = oy * strideY + sy;

                                if (inputs[channel][iy][ix] > poolValue)
                                    poolValue = inputs[channel][iy][ix];
                            }
                        }
                    }
#if NB_BITS < 0
                    outputs[outputOffset + output][oy][ox]
                        = sat(poolValue, func, shift);
#else
                    outputs[outputOffset + output][oy][ox] = poolValue;
#endif
                } else if (pooling == Average) {
                    unsigned int nbMapChan = 0;
                    SUM_T sum = 0;

                    for (unsigned int channel = 0; channel < nbChannels;
                         ++channel) {
                        if (!mapping[output][channel])
                            continue;
                        for (unsigned int sy = 0; sy < syMax; ++sy) {
                            for (unsigned int sx = 0; sx < sxMax; ++sx) {
                                const unsigned int ix = ox * strideX + sx;
                                const unsigned int iy = oy * strideY + sy;
                                sum += inputs[channel][iy][ix];
                            }
                        }
                        ++nbMapChan;
                    }
                    sum /= poolWidth * poolHeight * nbMapChan;
#if NB_BITS < 0
                    outputs[outputOffset + output][oy][ox]
                        = sat(sum, func, shift);
#else
                    outputs[outputOffset + output][oy][ox] = sht(sum, shift);
#endif
                }
            }
        }
    }
}

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
    int shift)
{

#if NB_BITS < 0
    if (func != Linear) {
        fprintf(stderr,
                "poolcell_propagate_unitmap(): only Linear is implemented\n");
        return;
    }
#endif

#if defined(_OPENMP) && _OPENMP >= 200805
#pragma omp parallel for collapse(3)
#else
#pragma omp parallel for
#endif
    for (unsigned int output = 0; output < nbOutputs; ++output) {
        for (unsigned int oy = 0; oy < outputsHeight; ++oy) {
            for (unsigned int ox = 0; ox < outputsWidth; ++ox) {
                const unsigned int sxMax
                    = uint_min(channelsWidth - ox * strideX, poolWidth);
                const unsigned int syMax
                    = uint_min(channelsHeight - oy * strideY, poolHeight);

                if (pooling == Max) {
                    DATA_T poolValue = DATA_T_MIN;

                    for (unsigned int sy = 0; sy < syMax; ++sy) {
                        for (unsigned int sx = 0; sx < sxMax; ++sx) {
                            const unsigned int ix = ox * strideX + sx;
                            const unsigned int iy = oy * strideY + sy;

                            if (inputs[output][iy][ix] > poolValue)
                                poolValue = inputs[output][iy][ix];
                        }
                    }
#if NB_BITS < 0
                    outputs[outputOffset + output][oy][ox]
                        = sat(poolValue, func, shift);
#else
                    outputs[outputOffset + output][oy][ox] = poolValue;
#endif
                } else if (pooling == Average) {
                    SUM_T sum = 0;

                    for (unsigned int sy = 0; sy < syMax; ++sy) {
                        for (unsigned int sx = 0; sx < sxMax; ++sx) {
                            const unsigned int ix = ox * strideX + sx;
                            const unsigned int iy = oy * strideY + sy;
                            sum += inputs[output][iy][ix];
                        }
                    }
                    sum /= poolWidth * poolHeight;
#if NB_BITS < 0
                    outputs[outputOffset + output][oy][ox]
                        = sat(sum, func, shift);
#else
                    outputs[outputOffset + output][oy][ox] = sht(sum, shift);
#endif
                }
            }
        }
    }
}

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
                    int shift)
{
#if NB_BITS < 0
    if (func != Linear) {
        fprintf(stderr, "poolcell_upropagate(): only Linear is implemented\n");
        return;
    }
#endif

#if defined(_OPENMP) && _OPENMP >= 200805
#pragma omp parallel for collapse(3)
#else
#pragma omp parallel for
#endif
    for (unsigned int output = 0; output < nbOutputs; ++output) {
        for (unsigned int oy = 0; oy < outputsHeight; ++oy) {
            for (unsigned int ox = 0; ox < outputsWidth; ++ox) {
                const unsigned int sxMax
                    = uint_min(channelsWidth - ox * strideX, poolWidth);
                const unsigned int syMax
                    = uint_min(channelsHeight - oy * strideY, poolHeight);

                if (pooling == Max) {
                    UDATA_T poolValue = 0;

                    for (unsigned int channel = 0; channel < nbChannels;
                         ++channel) {
                        if (!mapping[output][channel])
                            continue;
                        for (unsigned int sy = 0; sy < syMax; ++sy) {
                            for (unsigned int sx = 0; sx < sxMax; ++sx) {
                                const unsigned int ix = ox * strideX + sx;
                                const unsigned int iy = oy * strideY + sy;

                                if (((UDATA_T)inputs[channel][iy][ix])
                                    > poolValue)
                                    poolValue
                                        = (UDATA_T)inputs[channel][iy][ix];
                            }
                        }
                    }
#if NB_BITS < 0
                    outputs[outputOffset + output][oy][ox]
                        = usat(poolValue, func, shift);
#else
                    outputs[outputOffset + output][oy][ox] = poolValue;
#endif
                } else if (pooling == Average) {
                    unsigned int nbMapChan = 0;
                    SUM_T sum = 0;

                    for (unsigned int channel = 0; channel < nbChannels;
                         ++channel) {
                        if (!mapping[output][channel])
                            continue;

                        for (unsigned int sy = 0; sy < syMax; ++sy) {
                            for (unsigned int sx = 0; sx < sxMax; ++sx) {
                                const unsigned int ix = ox * strideX + sx;
                                const unsigned int iy = oy * strideY + sy;
                                sum += ((UDATA_T)inputs[channel][iy][ix]);
                            }
                        }
                        ++nbMapChan;
                    }
                    sum /= poolWidth * poolHeight * nbMapChan;
#if NB_BITS < 0
                    outputs[outputOffset + output][oy][ox]
                        = usat(sum, func, shift);
#else
                    outputs[outputOffset + output][oy][ox] = sht(sum, shift);
#endif
                }
            }
        }
    }
}

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
    int shift)
{
#if NB_BITS < 0
    if (func != Linear) {
        fprintf(stderr,
                "poolcell_upropagate_unitmap(): only Linear is implemented\n");
        return;
    }
#endif

#if defined(_OPENMP) && _OPENMP >= 200805
#pragma omp parallel for collapse(3)
#else
#pragma omp parallel for
#endif
    for (unsigned int output = 0; output < nbOutputs; ++output) {
        for (unsigned int oy = 0; oy < outputsHeight; ++oy) {
            for (unsigned int ox = 0; ox < outputsWidth; ++ox) {
                const unsigned int sxMax
                    = uint_min(channelsWidth - ox * strideX, poolWidth);
                const unsigned int syMax
                    = uint_min(channelsHeight - oy * strideY, poolHeight);

                if (pooling == Max) {
                    UDATA_T poolValue = 0;

                    for (unsigned int sy = 0; sy < syMax; ++sy) {
                        for (unsigned int sx = 0; sx < sxMax; ++sx) {
                            const unsigned int ix = ox * strideX + sx;
                            const unsigned int iy = oy * strideY + sy;

                            if (((UDATA_T)inputs[output][iy][ix]) > poolValue)
                                poolValue = (UDATA_T)inputs[output][iy][ix];
                        }
                    }
#if NB_BITS < 0
                    outputs[outputOffset + output][oy][ox]
                        = usat(poolValue, func, shift);
#else
                    outputs[outputOffset + output][oy][ox] = poolValue;
#endif
                } else if (pooling == Average) {
                    SUM_T sum = 0;
                    for (unsigned int sy = 0; sy < syMax; ++sy) {
                        for (unsigned int sx = 0; sx < sxMax; ++sx) {
                            const unsigned int ix = ox * strideX + sx;
                            const unsigned int iy = oy * strideY + sy;
                            sum += ((UDATA_T)inputs[output][iy][ix]);
                        }
                    }
                    sum /= poolWidth * poolHeight;
#if NB_BITS < 0
                    outputs[outputOffset + output][oy][ox]
                        = usat(sum, func, shift);
#else
                    outputs[outputOffset + output][oy][ox] = sht(sum, shift);
#endif
                }
            }
        }
    }
}
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
                     const WDATA_T centers[nbOutputs][nbChannels_])
{
#pragma omp parallel for
    for (unsigned int output = 0; output < nbOutputs; ++output) {
        int32_t normSquared = 0;
        unsigned int c = 0;

        for (unsigned int channel = 0; channel < nbChannels; ++channel) {
            for (unsigned int iy = 0; iy < channelsHeight; ++iy) {
                for (unsigned int ix = 0; ix < channelsWidth; ++ix) {
                    // 9 bits dynamic, range = -256 to 255 => saturation
                    const int8_t sub
                        = sat8(centers[output][c++] - inputs[channel][iy][ix]);
                    const int16_t prod = (int16_t)(sub * sub);
                    /*
                                        // With 16 bits ALU & 32 bits MULT:
                                        // 9 bits dynamic, range = -256 to 255
                                        int16_t sub = centers[output][c++] -
                       inputs[channel][iy][ix];
                                        // 17 bits dynamic, range = 0 to 65536
                       (not 65535!!!)
                                        int32_t prod = sub*sub;
                    */
                    normSquared += prod; // normSquared+= (xi-ci)^2
                }
            }
        }

        // normSquared = sum((xi-ci)^2)

        // Example:
        // normSquared = 0b0000,00|01,0111,01|00,0101,1001,0110,1100 (32 bits)
        //                        ^         ^
        //                       msb (=24) lsb (=24-7=17)   (signed integer)
        // msByte = 0101,1101
        //
        // Scaling coding:
        //   scaling_output = 255*(1/(2*sigma_output^2))
        //   maxScaling = max(scaling_output) over all outputs
        //   scalingPrecision = nb bits required to store maxScaling =
        // ceil(log2(maxScaling))
        //   Example: maxScaling = 43 => scalingPrecision = 6 (2^5-1 = 31 < 43 <
        // 2^6-1 = 63)
        //
        // To use the full 8 bit dynamic to store the scaling:
        // scaling[output] = scalingRatio*scaling_output
        //      with scalingRatio = 255/(2^scalingPrecision-1)

        const char msb = msb32(normSquared);
        const char lsb = (char)((msb > 7) ? msb - 7 : 0);
        const char msByte = (char)(normSquared >> lsb);
        normSquared = -msByte * scaling[output];

        const char rs = (char)(8 - lsb + (8 - scalingPrecision));

        normSquared = (rs >= 0) ? (normSquared >> rs) : (normSquared << -rs);

// With 32 (or more!) bits MULT:
// normSquared*= -scaling[output];
// normSquared>>= 8 + (8 - scalingPrecision);

#ifdef NL
        outputs[outputOffset + output] = nl32_exp(normSquared);
#else
        normSquared += 2 * 16129;
        DATA_T val = sat32((SUM_T)normSquared, 9);
        /*
                normSquared+= ((2*16129) >> -rs);

                char rsSat = (rs >= -9) ? rs + 9 : 0;
                char val = sat32(normSquared, rsSat);
        */
        outputs[outputOffset + output] = (val < 0) ? 0 : val;
#endif
    }
}

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
                      const WDATA_T centers[nbOutputs][nbChannels_])
{
#pragma omp parallel for
    for (unsigned int output = 0; output < nbOutputs; ++output) {
        int32_t normSquared = 0;
        unsigned int c = 0;

        for (unsigned int channel = 0; channel < nbChannels; ++channel) {
            for (unsigned int iy = 0; iy < channelsHeight; ++iy) {
                for (unsigned int ix = 0; ix < channelsWidth; ++ix) {
                    char input
                        = (char)(((unsigned char)inputs[channel][iy][ix]) >> 1);
                    int8_t sub = sat8(centers[output][c++] - input);
                    int16_t prod = (int16_t)(sub * sub);

                    normSquared += prod;
                }
            }
        }

        const char msb = msb32(normSquared);
        const char lsb = (char)((msb > 7) ? msb - 7 : 0);
        const char msByte = (char)(normSquared >> lsb);
        normSquared = -msByte * scaling[output];

        const char rs = (char)(8 - lsb + (8 - scalingPrecision));

        normSquared = (rs >= 0) ? (normSquared >> rs) : (normSquared << -rs);

#ifdef NL
        outputs[outputOffset + output] = nl32_exp(normSquared);
#else
        normSquared += 2 * 16129;
        DATA_T val = sat32((SUM_T)normSquared, 9);

        outputs[outputOffset + output] = (val < 0) ? 0 : val;
#endif
    }
}

void rbfcell_propagate(unsigned int nbChannels,
                       DATA_T inputs[nbChannels],
                       unsigned int nbOutputs_,
                       unsigned int nbOutputs,
                       unsigned int outputOffset,
                       DATA_T outputs[nbOutputs_],
                       uint8_t scalingPrecision,
                       const unsigned char scaling[nbOutputs],
                       const WDATA_T centers[nbOutputs][nbChannels])
{
#pragma omp parallel for
    for (unsigned int output = 0; output < nbOutputs; ++output) {
        int32_t normSquared = 0;

        for (unsigned int channel = 0; channel < nbChannels; ++channel) {
            // 9 bits dynamic, range = -256 to 255 => saturation
            const int8_t sub = sat8(centers[output][channel] - inputs[channel]);
            const int16_t prod = (int16_t)(sub * sub);

            normSquared += prod;
        }

        const char msb = msb32(normSquared);
        const char lsb = (char)((msb > 7) ? msb - 7 : 0);
        const char msByte = (char)(normSquared >> lsb);
        normSquared = -msByte * scaling[output];

        const char rs = (char)(8 - lsb + (8 - scalingPrecision));

        normSquared = (rs >= 0) ? (normSquared >> rs) : (normSquared << -rs);

#ifdef NL
        outputs[outputOffset + output] = nl32_exp(normSquared);
#else
        normSquared += 2 * 16129;
        DATA_T val = sat32((SUM_T)normSquared, 9);

        outputs[outputOffset + output] = (val < 0) ? 0 : val;
#endif
    }
}
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
                    int shift)
{
#pragma omp parallel for if (nbOutputs > 32)
    for (unsigned int output = 0; output < nbOutputs; ++output) {
        SUM_T weightedSum = bias[output];
        unsigned int c = 0;

        for (unsigned int channel = 0; channel < nbChannels; ++channel) {
            for (unsigned int iy = 0; iy < channelsHeight; ++iy) {
                for (unsigned int ix = 0; ix < channelsWidth; ++ix)
                    weightedSum = ADD_SAT(weightedSum, weights[output][c++]
                                   * inputs[channel][iy][ix]);
            }
        }

        outputs[outputOffset + output] = sat(weightedSum, func, shift);
    }
}

void
fccell_upropagate_2d(unsigned int nbChannels,
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
                     int shift)
{
#pragma omp parallel for if (nbOutputs > 32)
    for (unsigned int output = 0; output < nbOutputs; ++output) {
        SUM_T weightedSum = bias[output];
        unsigned int c = 0;

        for (unsigned int channel = 0; channel < nbChannels; ++channel) {
            for (unsigned int iy = 0; iy < channelsHeight; ++iy) {
                for (unsigned int ix = 0; ix < channelsWidth; ++ix)
                    weightedSum = ADD_SAT(weightedSum,
                        (SUM_T)weights[output][c++]
                        * (SUM_T)((UDATA_T)inputs[channel][iy][ix]));
            }
        }

        outputs[outputOffset + output] = usat(weightedSum, func, shift);
    }
}

void fccell_propagate(unsigned int nbChannels,
                      DATA_T inputs[nbChannels],
                      unsigned int nbOutputs_,
                      unsigned int nbOutputs,
                      unsigned int outputOffset,
                      DATA_T outputs[nbOutputs_],
                      const BDATA_T bias[nbOutputs],
                      const WDATA_T weights[nbOutputs][nbChannels],
                      ActivationFunction_T func,
                      int shift)
{
#pragma omp parallel for if (nbOutputs > 32)
    for (unsigned int output = 0; output < nbOutputs; ++output) {
        SUM_T weightedSum = bias[output];

        for (unsigned int channel = 0; channel < nbChannels; ++channel)
            weightedSum = ADD_SAT(weightedSum,
                                  weights[output][channel] * inputs[channel]);

        outputs[outputOffset + output] = sat(weightedSum, func, shift);
    }
}

void fccell_upropagate(unsigned int nbChannels,
                      DATA_T inputs[nbChannels],
                      unsigned int nbOutputs_,
                      unsigned int nbOutputs,
                      unsigned int outputOffset,
                      DATA_T outputs[nbOutputs_],
                      const BDATA_T bias[nbOutputs],
                      const WDATA_T weights[nbOutputs][nbChannels],
                      ActivationFunction_T func,
                      int shift)
{
#pragma omp parallel for if (nbOutputs > 32)
    for (unsigned int output = 0; output < nbOutputs; ++output) {
        SUM_T weightedSum = bias[output];

        for (unsigned int channel = 0; channel < nbChannels; ++channel)
            weightedSum = ADD_SAT(weightedSum,
                                  (SUM_T)weights[output][channel]
                                  * (SUM_T)((UDATA_T)inputs[channel]));

        outputs[outputOffset + output] = usat(weightedSum, func, shift);
    }
}

void fccell_propagate_2d_sparse(
    unsigned int nbChannels,
    unsigned int channelsHeight,
    unsigned int channelsWidth,
    DATA_T inputs[nbChannels][channelsHeight][channelsWidth],
    unsigned int nbOutputs_,
    unsigned int nbOutputs,
    unsigned int outputOffset,
    DATA_T outputs[nbOutputs_],
    unsigned int UNUSED(nbChannels_),
    const BDATA_T bias[nbOutputs],
    unsigned int nbWeights,
    const WDATA_T weights[nbWeights],
    const unsigned short offsets[nbWeights],
    ActivationFunction_T func,
    int shift)
{
    const unsigned int channelSize = channelsHeight * channelsWidth;
    const unsigned int channelsSize = nbChannels * channelSize;

    unsigned int w = 0;
    unsigned int c = offsets[0];

    for (unsigned int output = 0; output < nbOutputs; ++output) {
        SUM_T weightedSum = bias[output];

        while (c < channelsSize && w < nbWeights) {
            const unsigned int channel = c / channelSize;
            const unsigned int i = c % channelSize;
            const unsigned int ix = i % channelsWidth;
            const unsigned int iy = i / channelsWidth;

            weightedSum = ADD_SAT(weightedSum,
                                  weights[w] * inputs[channel][iy][ix]);

            ++w;
            c += offsets[w];
        }

        c -= channelsSize;

        outputs[outputOffset + output] = sat(weightedSum, func, shift);
    }
}

void fccell_upropagate_2d_sparse(
    unsigned int nbChannels,
    unsigned int channelsHeight,
    unsigned int channelsWidth,
    DATA_T inputs[nbChannels][channelsHeight][channelsWidth],
    unsigned int nbOutputs_,
    unsigned int nbOutputs,
    unsigned int outputOffset,
    DATA_T outputs[nbOutputs_],
    unsigned int UNUSED(nbChannels_),
    const BDATA_T bias[nbOutputs],
    unsigned int nbWeights,
    const WDATA_T weights[nbWeights],
    const unsigned short offsets[nbWeights],
    ActivationFunction_T func,
    int shift)
{
    const unsigned int channelSize = channelsHeight * channelsWidth;
    const unsigned int channelsSize = nbChannels * channelSize;

    unsigned int w = 0;
    unsigned int c = offsets[0];

    for (unsigned int output = 0; output < nbOutputs; ++output) {
        SUM_T weightedSum = bias[output];

        while (c < channelsSize && w < nbWeights) {
            const unsigned int channel = c / channelSize;
            const unsigned int i = c % channelSize;
            const unsigned int ix = i % channelsWidth;
            const unsigned int iy = i / channelsWidth;

            weightedSum = ADD_SAT(weightedSum, (SUM_T)weights[w]
                           * (SUM_T)((UDATA_T)inputs[channel][iy][ix]));

            ++w;
            c += offsets[w];
        }

        c -= channelsSize;

        outputs[outputOffset + output] = usat(weightedSum, func, shift);
    }
}

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
                             int shift)
{
    unsigned int w = 0;
    unsigned int channel = offsets[0];

    for (unsigned int output = 0; output < nbOutputs; ++output) {
        SUM_T weightedSum = bias[output];

        while (channel < nbChannels && w < nbWeights) {
            weightedSum = ADD_SAT(weightedSum, weights[w] * inputs[channel]);

            ++w;
            channel += offsets[w];
        }

        channel -= nbChannels;

        outputs[outputOffset + output] = sat(weightedSum, func, shift);
    }
}

void output_max(unsigned int nbOutputs,
                DATA_T outputs[nbOutputs],
                uint32_t outputEstimated[1][1])
{
    if (nbOutputs > 1) {
        DATA_T maxVal = outputs[0];
        unsigned int outputMax = 0;

    LOOP_FCCELL_OUTPUT_MAX:
        for (unsigned int output = 1; output < nbOutputs; ++output) {
            if (outputs[output] > maxVal) {
                maxVal = outputs[output];
                outputMax = output;
            }
        }

        outputEstimated[0][0] = outputMax;
    }
    else
        outputEstimated[0][0] = (outputs[0] > (BINARY_THRESHOLD * DATA_T_MAX));
}

void spatial_output_max(unsigned int nbOutputs,
                        unsigned int outputsHeight,
                        unsigned int outputsWidth,
                        DATA_T outputs[nbOutputs][outputsHeight][outputsWidth],
                        uint32_t outputEstimated[outputsHeight][outputsWidth])
{
LOOP_CONVCELL_OY_MAX:
    for (unsigned int oy = 0; oy < outputsHeight; ++oy) {
    LOOP_CONVCELL_OX_MAX:
        for (unsigned int ox = 0; ox < outputsWidth; ++ox) {
            if (nbOutputs > 1) {
                DATA_T maxVal = outputs[0][oy][ox];
                unsigned int outputMax = 0;

            LOOP_CONVCELL_OUTPUT_MAX:
                for (unsigned int output = 1; output < nbOutputs; ++output) {
                    if (outputs[output][oy][ox] > maxVal) {
                        outputMax = output;
                        maxVal = outputs[output][oy][ox];
                    }
                }

                outputEstimated[oy][ox] = outputMax;
            }
            else {
                outputEstimated[oy][ox] = (outputs[0][oy][ox]
                                            > (BINARY_THRESHOLD * DATA_T_MAX));
            }
        }
    }
}
void
softmaxcell_propagate(unsigned int nbOutputs,
                      unsigned int outputsHeight,
                      unsigned int outputsWidth,
                      DATA_T inputs[nbOutputs][outputsHeight][outputsWidth],
                      DATA_T outputs[nbOutputs][outputsHeight][outputsWidth])
{
#if NB_BITS > 0
    if (nbOutputs > 1) {
    LOOP_INT_SOFTMAX_OY_MAX:
        for (unsigned int oy = 0; oy < outputsHeight; ++oy)
        LOOP_INT_SOFTMAX_OX_MAX:
        for (unsigned int ox = 0; ox < outputsWidth; ++ox)
        LOOP_INT_SOFTMAX_OUTPUT:
        for (unsigned int output = 0; output < nbOutputs; ++output)
            outputs[output][oy][ox] = inputs[output][oy][ox];
    } else {
        for (unsigned int oy = 0; oy < outputsHeight; ++oy)
            for (unsigned int ox = 0; ox < outputsWidth; ++ox)
                outputs[0][oy][ox] = inputs[0][oy][ox];
    }
#else
    if (nbOutputs > 1) {
    LOOP_SOFTMAX_OY_MAX:
        for (unsigned int oy = 0; oy < outputsHeight; ++oy) {
        LOOP_SOFTMAX_OX_MAX:
            for (unsigned int ox = 0; ox < outputsWidth; ++ox) {
                DATA_T maxVal = inputs[0][oy][ox];
                double sum = 0.0;

            LOOP_SOFTMAX_GET_MAX_ACCROSS_CHAN:
                for (unsigned int output = 1; output < nbOutputs; ++output) {

                    if (inputs[output][oy][ox] > maxVal)
                        maxVal = inputs[output][oy][ox];
                }

            LOOP_SOFTMAX_SUM_CHAN:
                for (unsigned int output = 0; output < nbOutputs; ++output)
                    sum += exp(inputs[output][oy][ox] - maxVal);

            LOOP_SOFTMAX_OUTPUT:
                for (unsigned int output = 0; output < nbOutputs; ++output)
                    outputs[output][oy][ox]
                        = exp(inputs[output][oy][ox] - maxVal) / sum;
            }
        }
    }
    else {
        for (unsigned int oy = 0; oy < outputsHeight; ++oy) {
            for (unsigned int ox = 0; ox < outputsWidth; ++ox) {
                outputs[0][oy][ox] = (inputs[0][oy][ox]
                                        > (BINARY_THRESHOLD * DATA_T_MAX));
            }
        }
    }
#endif
}

void resize_bilinear_tf_propagete(unsigned int nbChannels,
                                  unsigned int channelsHeight,
                                  unsigned int channelsWidth,
                                  DATA_T inputs[nbChannels][channelsHeight][channelsWidth],
                                  unsigned int nbOutputs,
                                  unsigned int outputsHeight,
                                  unsigned int outputsWidth,
                                  DATA_T outputs[nbOutputs][outputsHeight][outputsWidth],
                                  const Interpolation interpolationHeight[outputsHeight],
                                  const Interpolation interpolationWidth[outputsWidth])
{
    for (unsigned int output = 0; output < nbOutputs; output++) {
        for (unsigned int oy = 0; oy < outputsHeight; oy++) {
            for (unsigned int ox = 0; ox < outputsWidth; ox++) {
                const float top_left = inputs[output][interpolationHeight[oy].lowIndex]
                                                     [interpolationWidth[ox].lowIndex];

                const float top_right = inputs[output][interpolationHeight[oy].lowIndex]
                                                      [interpolationWidth[ox].highIndex];

                const float bottom_left = inputs[output][interpolationHeight[oy].highIndex]
                                                        [interpolationWidth[ox].lowIndex];

                const float bottom_right = inputs[output][interpolationHeight[oy].highIndex]
                                                         [interpolationWidth[ox].highIndex];

                const float top = top_left + 
                                (top_right - top_left) * interpolationWidth[ox].interpolation;
                const float bottom = bottom_left + 
                                (bottom_right - bottom_left) * interpolationWidth[ox].interpolation;


#if NB_BITS < 0
                outputs[output][oy][ox] = 
                        (top + (bottom - top) * interpolationHeight[oy].interpolation);
#else
                outputs[output][oy][ox] = 
                        (DATA_T) lround(top + (bottom - top) * interpolationHeight[oy].interpolation);
#endif
            }
        }
    }
}  

void resize_nearest_neighbor_propagete(unsigned int nbChannels,
                                       unsigned int channelsHeight,
                                       unsigned int channelsWidth,
                                       DATA_T inputs[nbChannels][channelsHeight][channelsWidth],
                                       unsigned int nbOutputs,
                                       unsigned int outputsHeight,
                                       unsigned int outputsWidth,
                                       DATA_T outputs[nbOutputs][outputsHeight][outputsWidth])
{
    if(nbChannels != nbOutputs) {
        fprintf(stderr, "Number of input channels must be equal to the number "
                        "of output channels for nearest neighbor resize.");
        return;
    }

    if(outputsHeight % channelsHeight != 0) {
        fprintf(stderr, "Output height must be a multiple of input height.");
        return;
    }

    if(outputsWidth % channelsWidth != 0) {
        fprintf(stderr, "Output width must be a multiple of input width.");
        return;
    }


    const size_t heightMult = outputsHeight/channelsHeight;
    const size_t widthMult = outputsWidth/channelsWidth;

#if defined(_OPENMP) && _OPENMP >= 200805
#pragma omp parallel for collapse(3)
#else
#pragma omp parallel for
#endif
    for(size_t oy = 0; oy < outputsHeight; oy++) {
        for(size_t ox = 0; ox < outputsWidth; ox++) {
            for(size_t output = 0; output < nbOutputs; output++) {
                outputs[output][oy][ox] = inputs[output][oy/heightMult][ox/widthMult];
            }
        }
    }    
}

void convcell_output_out(FILE* file, unsigned int nbOutputs,
                         unsigned int outputsHeight, unsigned int outputsWidth,
                         DATA_T outputs[nbOutputs][outputsHeight][outputsWidth])
{
    for (unsigned int output = 0; output < nbOutputs; ++output) {
        fprintf(file, "%d:\n", output);

        for (unsigned int oy = 0; oy < outputsHeight; ++oy) {
            for (unsigned int ox = 0; ox < outputsWidth; ++ox) {
#if NB_BITS < 0
                fprintf(file, "%f ", outputs[output][oy][ox]);
#else
                fprintf(file, "%d ", outputs[output][oy][ox]);
#endif
            }

            fprintf(file, "\n");
        }

        fprintf(file, "\n");
    }

}

void convcell_outputs_save(const char* fileName, unsigned int nbOutputs,
                           unsigned int outputsHeight, unsigned int outputsWidth,
                           DATA_T outputs[nbOutputs][outputsHeight][outputsWidth])
{
    FILE* file = fopen(fileName, "w");
    if(!file) {
        fprintf(stderr, "Couldn't open file %s.", fileName);
        return;
    }

    convcell_output_out(file, nbOutputs, outputsHeight, outputsWidth, outputs);
    if(fclose(file) != 0) {
        fprintf(stderr, "Couldn't close file %s.", fileName);
    }
}

void convcell_outputs_print(const char* name, unsigned int nbOutputs,
                            unsigned int outputsHeight, unsigned int outputsWidth,
                            DATA_T outputs[nbOutputs][outputsHeight][outputsWidth])
{
    printf("%s outputs:\n", name);
    convcell_output_out(stdout, nbOutputs, outputsHeight, outputsWidth, outputs);
}

void convcell_outputs_dynamic_print(
    const char* name,
    unsigned int nbOutputs,
    unsigned int outputsHeight,
    unsigned int outputsWidth,
    DATA_T outputs[nbOutputs][outputsHeight][outputsWidth],
    DATA_T* pMinVal,
    DATA_T* pMaxVal,
    RUNNING_MEAN_T* pMeanVal)
{
    DATA_T minVal = (pMinVal != NULL) ? (*pMinVal) : outputs[0][0][0];
    DATA_T maxVal = (pMaxVal != NULL) ? (*pMaxVal) : outputs[0][0][0];
    double meanVal = 0.0;

    for (unsigned int output = 0; output < nbOutputs; ++output) {
        for (unsigned int oy = 0; oy < outputsHeight; ++oy) {
            for (unsigned int ox = 0; ox < outputsWidth; ++ox) {
                minVal = MIN(minVal, outputs[output][oy][ox]);
                maxVal = MAX(maxVal, outputs[output][oy][ox]);
                meanVal+= outputs[output][oy][ox];
            }
        }
    }

    meanVal/= (nbOutputs * outputsHeight * outputsWidth);

    if (pMinVal != NULL)
        (*pMinVal) = minVal;
    if (pMaxVal != NULL)
        (*pMaxVal) = maxVal;
    if (pMeanVal != NULL) {
        meanVal = ((*pMeanVal).mean * (*pMeanVal).count + meanVal)
            / ((*pMeanVal).count + 1.0);

        (*pMeanVal).mean = meanVal;
        ++(*pMeanVal).count;
    }

#if NB_BITS < 0
    printf("%s dynamic = [%f, %f, %f]\n", name, minVal, maxVal, meanVal);
#else
    printf("%s dynamic = [%d, %d, %f]\n", name, minVal, maxVal, meanVal);
#endif
}

void fccell_outputs_out(FILE* file, 
                        unsigned int nbOutputs,
                        DATA_T outputs[nbOutputs])
{
    for (unsigned int output = 0; output < nbOutputs; ++output) {
#if NB_BITS < 0
        fprintf(file, "%d: %f\n", output, outputs[output]);
#else
        fprintf(file, "%d: %d\n", output, outputs[output]);
#endif
    }
}

void fccell_outputs_save(const char* fileName,
                         unsigned int nbOutputs,
                         DATA_T outputs[nbOutputs])
{
    FILE* file = fopen(fileName, "w");
    if(!file) {
        fprintf(stderr, "Couldn't open file %s.", fileName);
        return;
    }

    fccell_outputs_out(file, nbOutputs, outputs);
    if(fclose(file) != 0) {
        fprintf(stderr, "Couldn't close file %s.", fileName);
    }
}

void fccell_outputs_print(const char* name,
                          unsigned int nbOutputs,
                          DATA_T outputs[nbOutputs])
{
    printf("%s outputs:\n", name);
    fccell_outputs_out(stdout, nbOutputs, outputs);
}

void fccell_outputs_dynamic_print(const char* name,
                                  unsigned int nbOutputs,
                                  DATA_T outputs[nbOutputs],
                                  DATA_T* pMinVal,
                                  DATA_T* pMaxVal,
                                  RUNNING_MEAN_T* pMeanVal)
{
    DATA_T minVal = (pMinVal != NULL) ? (*pMinVal) : outputs[0];
    DATA_T maxVal = (pMaxVal != NULL) ? (*pMaxVal) : outputs[0];
    double meanVal = 0.0;

    for (unsigned int output = 0; output < nbOutputs; ++output) {
        minVal = MIN(minVal, outputs[output]);
        maxVal = MAX(maxVal, outputs[output]);
        meanVal+= outputs[output];
    }

    meanVal/= nbOutputs;

    if (pMinVal != NULL)
        (*pMinVal) = minVal;
    if (pMaxVal != NULL)
        (*pMaxVal) = maxVal;
    if (pMeanVal != NULL) {
        meanVal = ((*pMeanVal).mean * (*pMeanVal).count + meanVal)
            / ((*pMeanVal).count + 1.0);

        (*pMeanVal).mean = meanVal;
        ++(*pMeanVal).count;
    }

#if NB_BITS < 0
    printf("%s dynamic = [%f, %f, %f]\n", name, minVal, maxVal, meanVal);
#else
    printf("%s dynamic = [%d, %d, %f]\n", name, minVal, maxVal, meanVal);
#endif
}

void confusion_print(unsigned int nbOutputs,
                     unsigned int confusion[nbOutputs][nbOutputs])
{
    printf("\nConfusion matrix:\n");

    printf("%.*s\n",
           9 + 10 * nbOutputs,
           "----------"
           "----------"
           "----------"
           "----------"
           "----------"
           "----------"
           "----------"
           "----------"
           "----------"
           "----------"
           "----------"
           "----------"
           "----------"
           "----------"
           "----------"
           "----------");

    printf("| T \\ E |");

    for (unsigned int estimated = 0; estimated < nbOutputs; ++estimated)
        printf(" %7d |", estimated);

    printf("\n%.*s\n",
           9 + 10 * nbOutputs,
           "----------"
           "----------"
           "----------"
           "----------"
           "----------"
           "----------"
           "----------"
           "----------"
           "----------"
           "----------"
           "----------"
           "----------"
           "----------"
           "----------"
           "----------"
           "----------");

    unsigned int total = 0;
    unsigned int totalCorrect = 0;

    for (unsigned int target = 0; target < nbOutputs; ++target) {
        unsigned int targetCount = 0;

        for (unsigned int estimated = 0; estimated < nbOutputs; ++estimated)
            targetCount += confusion[target][estimated];

        total += targetCount;
        totalCorrect += confusion[target][target];

        printf("| %5d |", target);

        for (unsigned int estimated = 0; estimated < nbOutputs; ++estimated)
            printf(" %7d |", confusion[target][estimated]);

        printf("\n|       |");

        for (unsigned int estimated = 0; estimated < nbOutputs; ++estimated) {
            printf(" %s%6.2f%%%s |",
                   ESC_BG_LIGHT_YELLOW,
                   100.0 * ((targetCount > 0) ? (confusion[target][estimated]
                                                 / (double)targetCount)
                                              : 0.0),
                   ESC_ALL_OFF);
        }

        printf("\n");
    }

    printf("%.*s\nT: Target    E: Estimated\n",
           9 + 10 * nbOutputs,
           "----------"
           "----------"
           "----------"
           "----------"
           "----------"
           "----------"
           "----------"
           "----------"
           "----------"
           "----------"
           "----------"
           "----------"
           "----------"
           "----------"
           "----------"
           "----------");
}

void time_analysis(const char* name,
                   struct timeval start,
                   struct timeval end,
                   RUNNING_MEAN_T* timing)
{
    const double duration = 1.0e6 * (double)(end.tv_sec - start.tv_sec)
                            + (double)(end.tv_usec - start.tv_usec);

    (*timing).mean = ((*timing).mean * (*timing).count + duration)
        / ((*timing).count + 1.0);
    ++(*timing).count;

    printf("%s timing = %f us\n", name, (*timing).mean);
}
