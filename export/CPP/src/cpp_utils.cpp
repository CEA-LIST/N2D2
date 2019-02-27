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

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

#include <sys/types.h>
#include <dirent.h>

#include "cpp_utils.hpp"
#include "utils.h"

void getFilesList(const std::string& dir,
                  std::vector<std::string>& files)
{
    struct dirent* pFile;
    DIR* pDir = opendir(dir.c_str());
    if (pDir == NULL)
        throw std::runtime_error(
            "Couldn't open the directory for input patterns: " + dir);

    while ((pFile = readdir(pDir)) != NULL) {
        if (pFile->d_name[0] != '.')
            files.push_back(std::string(dir + "/" + pFile->d_name));
    }
    closedir(pDir);
    std::sort(files.begin(), files.end());
}

std::vector<std::string> getFilesList(const std::string& dir) {
    std::vector<std::string> files;
    getFilesList(dir, files);

    return files;
}

void envRead(const std::string& fileName,
             unsigned int size,
             unsigned int channelsHeight,
             unsigned int channelsWidth,
             DATA_T* data,
             unsigned int outputsSize,
             int32_t* outputTargets)
{
    std::ifstream stimuli(fileName.c_str(), std::fstream::binary);

    if (!stimuli.good())
        throw std::runtime_error("Could not open file: " + fileName);

    char header[2];
    stimuli.read(reinterpret_cast<char*>(&header[0]), sizeof(header));

    if (header[0] != 'P' || (header[1] != '5' && header[1] != '6'))
        throw std::runtime_error("Unknown PGM file format for file: "
                                 + fileName);

    int pixelWidth;
    int pixelHeight;
    int maxValue;

    if (!(stimuli >> pixelWidth) || !(stimuli >> pixelHeight)
        || !(stimuli >> maxValue))
        throw std::runtime_error("Error reading PGM image file: " + fileName);

    stimuli.get();

    if (pixelWidth != (int)channelsWidth || pixelHeight != (int)channelsHeight)
        throw std::runtime_error(
            "PGM image size does not match array size for file: " + fileName);

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
    stimuli.read(reinterpret_cast<char*>(&inputsFixed[0]),
                 size * sizeof(inputsFixed[0]));

    for (unsigned int i = 0; i < size; ++i)
        data[i] = (DATA_T)inputsFixed[i];
#else
    stimuli.read(reinterpret_cast<char*>(&data[0]), size * sizeof(data[0]));
#endif
    stimuli.read(reinterpret_cast<char*>(&outputTargets[0]),
                 outputsSize * sizeof(outputTargets[0]));

    if (stimuli.eof())
        throw std::runtime_error(
            "End-of-file reached prematurely in data file: " + fileName);
    else if (!stimuli.good())
        throw std::runtime_error("Error while reading data file: " + fileName);
    else if (stimuli.get() != std::fstream::traits_type::eof())
        throw std::runtime_error("Data file size larger than expected: "
                                 + fileName);
}

/**** Confusion Matrix ****/
void confusion_print(unsigned int nbOutputs, unsigned int* confusion)
{
    std::cout << "\nConfusion matrix:\n";
    std::cout << std::string(9 + 10 * nbOutputs, '-') << "\n";
    std::cout << "| T \\ E |";

    for (unsigned int estimated = 0; estimated < nbOutputs; ++estimated)
        std::cout << " " << std::setfill(' ') << std::setw(7) << estimated
                  << " |";

    std::cout << "\n" << std::string(9 + 10 * nbOutputs, '-') << "\n";

    unsigned int total = 0;
    unsigned int totalCorrect = 0;

    for (unsigned int target = 0; target < nbOutputs; ++target) {
        unsigned int targetCount = 0;

        for (unsigned int estimated = 0; estimated < nbOutputs; ++estimated)
            targetCount += confusion[estimated + target * nbOutputs];

        total += targetCount;
        totalCorrect += confusion[target + target * nbOutputs];

        std::cout << "| " << std::setfill(' ') << std::setw(5) << target
                  << " |";

        for (unsigned int estimated = 0; estimated < nbOutputs; ++estimated)
            std::cout << " " << std::setfill(' ') << std::setw(7)
                      << confusion[estimated + target * nbOutputs] << " |";

        std::cout << "\n";
        std::cout << "|       |";

        for (unsigned int estimated = 0; estimated < nbOutputs; ++estimated) {
            std::cout << " " << ESC_BG_LIGHT_YELLOW << std::setfill(' ')
                      << std::setw(6) << std::fixed << std::setprecision(2)
                      << 100.0
                         * ((targetCount > 0)
                                ? (confusion[estimated + target * nbOutputs]
                                   / (double)targetCount)
                                : 0.0) << "%" << ESC_ALL_OFF << " |";
        }
        std::cout << "\n";
    }

    std::cout << std::string(9 + 10 * nbOutputs, '-') << "\n"
              << "T: Target    E: Estimated" << std::endl;
}

void readNetpbmFile(const std::string& file, std::vector<unsigned char>& dataOut, bool rescale) {
    enum format_e {
        PBM_ASCII = 1,
        PGM_ASCII = 2,
        PPM_ASCII = 3,
        PBM_BINARY = 4,
        PGM_BINARY = 5,
        PPM_BINARY = 6,
    };
    
    std::ifstream fileStream(file.c_str(), std::fstream::binary);
    if(!fileStream.is_open()) {
        throw std::runtime_error("Couldn't open file '" + file + "'.");
    }

    fileStream.exceptions(std::ifstream::failbit | std::ifstream::badbit | std::ifstream::eofbit);

    char header;
    fileStream >> header;
    if(header != 'P') {
        throw std::runtime_error("The '" + file + "' file is not a valid Netpbm file.");
    }

    std::size_t format; 
    fileStream >> format;
    
    std::size_t width;
    fileStream >> width;
    
    std::size_t height;
    fileStream >> height;

    std::size_t maxValue = 1;
    if(format == PGM_ASCII || format == PPM_ASCII || format == PGM_BINARY || format == PPM_BINARY) {
        fileStream >> maxValue;
    }
    
    std::size_t nbChannels = 1;
    if(format == PPM_ASCII || format == PPM_BINARY) {
        nbChannels = 3;
    }
    
    
    if(dataOut.empty()) {
        dataOut.resize(width*height*nbChannels);
    }
    else if(dataOut.size() != width*height*nbChannels) {
        throw std::runtime_error("dataOut (" + std::to_string(dataOut.size()) + ") should be empty or of size " + std::to_string(width*height*nbChannels) + ".");
    }
    assert(dataOut.size() == width*height*nbChannels);
    
    
    // Read new line character
    fileStream.get();
    
    
    switch(format) {
        case PBM_ASCII:{
            char value;
            for(std::size_t i = 0; i < height*width*nbChannels; i++) {
                fileStream >> value;
                dataOut[i] = (value == '1');
                
            }
            break;
        }
        case PBM_BINARY:{
            std::size_t i = 0;
            for(std::size_t y = 0; y < height; y++) {
                for(std::size_t x = 0; x < width; x += 8) {
                    unsigned char value;
                    fileStream.read((char*) &value, sizeof(value));
                    
                    for(std::size_t xi = 0; xi < std::min(width - x, (std::size_t) 8); xi++) {
                        dataOut[i] = !((value >> (7 - xi)) & 1);
                        i++;
                    }
                }
            }
            break;
        }
        case PGM_ASCII:        
        case PPM_ASCII:
            std::copy_n(std::istream_iterator<std::size_t>(fileStream), dataOut.size(), dataOut.begin());
            break;
        case PGM_BINARY:
        case PPM_BINARY:
            fileStream.read((char*) dataOut.data(), dataOut.size());
            break;
        default:
            throw std::runtime_error("The '" + file + "' file is not a valid Netpbm file.");
    }

    
    // Rescale from [0-maxValue] to [0-255]
    if(rescale && maxValue != 255) {
        std::transform(dataOut.begin(), dataOut.end(),  dataOut.begin(), 
                       [&](unsigned char v) { 
                           return (unsigned char) std::lround(v*255.0/maxValue); 
                        });
    }
}
