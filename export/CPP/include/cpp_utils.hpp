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


void getFilesList(const std::string& dir, std::vector<std::string>& files);
std::vector<std::string> getFilesList(const std::string& dir);

void envRead(const std::string& fileName, unsigned int size,
             unsigned int channelsHeight, unsigned int channelsWidth,
             DATA_T* data, unsigned int outputsSize,
             int32_t* outputTargets);

/**
 * Read the Netpbm 'file' and store the result in dataOut.
 * 
 * If dataOut is empty, the required size is allocated. Otherwise the dataOut buffer
 * is reused to avoid any allocation. An exception is thrown if dataOut is not empty
 * and doesn't have the same size as the image.
 * 
 * Values are rescaled to [0-255] in rescale is true. Otherwise the values are
 * kept intact ([0-1] for 'pbm' files and [0-maxValue] for 'pgm' and 'ppm' files.
 */
void readNetpbmFile(const std::string& file, std::vector<unsigned char>& dataOut, 
                    bool rescale = true);

void confusion_print(unsigned int nbOutputs, unsigned int* confusion);

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