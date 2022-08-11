/*
    (C) Copyright 2017 CEA LIST. All Rights Reserved.
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

#ifndef CPP_UTILS_H
#define CPP_UTILS_H

#include "typedefs.h"
#include <string>
#include <vector>

#ifdef NO_EXCEPT
#define N2D2_THROW_OR_ABORT(ex, msg) \
do { printf("%s\n", std::string(msg).c_str()); abort(); } while (false)
#else
#include <stdexcept>
#define N2D2_THROW_OR_ABORT(ex, msg) throw ex(msg)
#endif
#define N2D2_ALWAYS_INLINE __attribute__((always_inline))

void getFilesList(const std::string& dir, std::vector<std::string>& files);
std::vector<std::string> getFilesList(const std::string& dir);

template <class Input_T, class Target_T>
void envRead(const std::string& fileName,
             unsigned int size,
             unsigned int channelsHeight,
             unsigned int channelsWidth,
             Input_T* data,
             unsigned int outputsSize,
             Target_T* outputTargets)
{
    FILE* stimuli = fopen(fileName.c_str(), "rb");

    if (stimuli == NULL) {
        N2D2_THROW_OR_ABORT(std::runtime_error, "Could not open file: \""
            + fileName + "\"");
        perror("The following error occurred");
    }

    char header[2];
    if (!fread(&header[0], sizeof(header[0]), 2, stimuli)) {
        N2D2_THROW_OR_ABORT(std::runtime_error,
            "Cannot read header from stimuli");
    }

    if (header[0] != 'P' || (header[1] != '5' && header[1] != '6')) {
        N2D2_THROW_OR_ABORT(std::runtime_error,
            "Unknown PGM file format for file: " + fileName);
    }

    int pixelWidth;
    int pixelHeight;
    int maxValue;
    if (fscanf(stimuli, "%d %d %d", &pixelWidth, &pixelHeight, &maxValue) != 3) {
        N2D2_THROW_OR_ABORT(std::runtime_error,
            "Cannot get pixelWidth, pixelHeight and maxValue from stimuli");
    }
    fgetc(stimuli);

    if (pixelWidth != (int)channelsWidth || pixelHeight != (int)channelsHeight)
    {
        N2D2_THROW_OR_ABORT(std::runtime_error,
            "PGM image size does not match array size for file: " + fileName);
    }

    size_t nbRead = 0;

#if NB_BITS > 0 && NB_BITS != 8 && NB_BITS != 16 && NB_BITS != 32 && NB_BITS   \
                                                                     != 64
#if NB_BITS > 0 && NB_BITS < 8
    char inputsFixed[size];
#elif NB_BITS > 8 && NB_BITS < 16
    short inputsFixed[size];
#elif NB_BITS > 16 && NB_BITS < 32
    int inputsFixed[size];
#elif NB_BITS > 32 && NB_BITS < 64
    long long int inputsFixed[size];
#endif
    nbRead = fread(inputsFixed, sizeof(inputsFixed[0]), size, stimuli);

    for (unsigned int i = 0; i < size; ++i)
        data[i] = (Input_T)inputsFixed[i];
#else
    nbRead = fread(data, sizeof(data[0]), size, stimuli);
#endif
    if (nbRead != size) {
        fprintf(stderr, "fread() number of read objects (%lu) different than"
                        " expected (%u) [data]\n", nbRead, size);
    }

    nbRead = fread(
        outputTargets, sizeof(outputTargets[0]), outputsSize, stimuli);

    if (nbRead != outputsSize) {
        fprintf(stderr, "fread() number of read objects (%lu) different than"
                        " expected (%u) [outputTargets]\n", nbRead, outputsSize);
    }

    if (feof(stimuli)) {
        N2D2_THROW_OR_ABORT(std::runtime_error,
            "End-of-file reached prematurely in data file: " + fileName);
    }
    else if (ferror(stimuli)) {
        N2D2_THROW_OR_ABORT(std::runtime_error,
            "Error while reading data file: " + fileName);
    }
    else if (fgetc(stimuli) != EOF) {
        N2D2_THROW_OR_ABORT(std::runtime_error,
            "Data file size larger than expected: " + fileName);
    }

    fclose(stimuli);
}

/**
 * Read the Netpbm 'file' and store the result in dataOut.
 * 
 * If dataOut is empty, the required size is allocated. Otherwise the dataOut buffer
 * is reused to avoid any allocation. An exception is thrown if dataOut is not empty
 * and doesn't have the same size as the image.
 * 
 * Values are rescaled to [0-255] if rescale is true. Otherwise the values are
 * kept intact ([0-1] for 'pbm' files and [0-maxValue] for 'pgm' and 'ppm' files.
 */
void readNetpbmFile(const std::string& file, std::vector<unsigned char>& dataOut, 
                    bool rescale = true);

void confusion_print(unsigned int nbOutputs, unsigned int* confusion);

// Required for DNeuro V2 emulator
template<class T>
const T& clamp(const T& v, const T& lo, const T& hi) {
    if(v < lo) {
        return lo;
    }

    if(v > hi) {
        return hi;
    }

    return v;
}

#endif
