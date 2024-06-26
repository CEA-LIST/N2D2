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
#include <string>
#include <vector>
#include <assert.h>

#include <sys/types.h>

#ifndef NO_DIRENT
#include <dirent.h>
#else
#include <getline.hpp>
#endif

#include "cpp_utils.hpp"
#include "utils.h"

void getFilesList(const std::string& dir,
                  std::vector<std::string>& files)
{
#ifndef NO_DIRENT
    struct dirent* pFile;
    DIR* pDir = opendir(dir.c_str());
    if (pDir == NULL) {
        N2D2_THROW_OR_ABORT(std::runtime_error,
            "Couldn't open the directory for input patterns: " + dir);
        perror("The following error occurred");
    }

    while ((pFile = readdir(pDir)) != NULL) {
        if (pFile->d_name[0] != '.')
            files.push_back(std::string(dir + "/" + pFile->d_name));
    }
    closedir(pDir);
#else
    FILE *dirList = fopen((dir + ".list").c_str(), "r");
    if (dirList == NULL) {
        N2D2_THROW_OR_ABORT(std::runtime_error,
            "Couldn't open the directory file list for input patterns: "
            + dir + ".list");
        perror("The following error occurred");
    }

    char* line = NULL;
    size_t len = 0;
    int read;

    while ((read = getLine(&line, &len, dirList)) != -1) {
        if (len > 0) {
            std::string fileName(line);
            fileName = fileName.substr(0, fileName.size() - 1); // skip \n
            files.push_back(fileName);
        }
    }

    fclose(dirList);
#endif
    std::sort(files.begin(), files.end());
}

std::vector<std::string> getFilesList(const std::string& dir) {
    std::vector<std::string> files;
    getFilesList(dir, files);

    return files;
}

/**** Confusion Matrix ****/
void confusion_print(unsigned int nbOutputs, unsigned int* confusion)
{
    printf("\nConfusion matrix:\n");
    printf("%s\n", std::string(9 + 10 * nbOutputs, '-').c_str());
    printf("| T \\ E |");

    for (unsigned int estimated = 0; estimated < nbOutputs; ++estimated)
        printf(" %7d |", estimated);

    printf("\n%s\n", std::string(9 + 10 * nbOutputs, '-').c_str());

    unsigned int total = 0;
    unsigned int totalCorrect = 0;

    for (unsigned int target = 0; target < nbOutputs; ++target) {
        unsigned int targetCount = 0;

        for (unsigned int estimated = 0; estimated < nbOutputs; ++estimated)
            targetCount += confusion[estimated + target * nbOutputs];

        total += targetCount;
        totalCorrect += confusion[target + target * nbOutputs];

        printf("| %5d |", target);

        for (unsigned int estimated = 0; estimated < nbOutputs; ++estimated)
            printf(" %7d |", confusion[estimated + target * nbOutputs]);

        printf("\n|       |");

        for (unsigned int estimated = 0; estimated < nbOutputs; ++estimated) {
            printf(" %s%6.2f%%%s |",
                   ESC_BG_LIGHT_YELLOW,
                   100.0 * ((targetCount > 0) ? (confusion[estimated + target * nbOutputs]
                                                 / (double)targetCount)
                                              : 0.0),
                   ESC_ALL_OFF);
        }
        printf("\n");
    }

    printf("%s\n", std::string(9 + 10 * nbOutputs, '-').c_str());
    printf("T: Target    E: Estimated\n");
}

void readNetpbmFile(const std::string& file, std::vector<unsigned char>& dataOut, bool rescale) {
    enum format_e {
        PBM_ASCII = 1,
        PGM_ASCII = 2,
        PPM_ASCII = 3,
        PBM_BINARY = 4,
        PGM_BINARY = 5,
        PPM_BINARY = 6
    };
    
    FILE* fileStream = fopen(file.c_str(), "rb");
    if(fileStream == NULL) {
        N2D2_THROW_OR_ABORT(std::runtime_error,
            "Couldn't open file '" + file + "'.");
        perror("The following error occurred");
    }

    char header;
    if (!fread(&header, sizeof(header), 1, fileStream)) {
        N2D2_THROW_OR_ABORT(std::runtime_error,
            "Cannot read header from fileStream");
    }

    if(header != 'P') {
        N2D2_THROW_OR_ABORT(std::runtime_error,
            "The '" + file + "' file is not a valid Netpbm file.");
    }

    std::size_t format; 
    if (!fread(&format, sizeof(format), 1, fileStream)) {
        N2D2_THROW_OR_ABORT(std::runtime_error,
            "Cannot read format from fileStream");
    }
    
    std::size_t width;
    if (!fread(&width, sizeof(width), 1, fileStream)) {
        N2D2_THROW_OR_ABORT(std::runtime_error,
            "Cannot read width from fileStream");
    }
    
    std::size_t height;
    if (!fread(&height, sizeof(height), 1, fileStream)) {
        N2D2_THROW_OR_ABORT(std::runtime_error,
            "Cannot read height from fileStream");
    }

    std::size_t maxValue = 1;
    if(format == PGM_ASCII || format == PPM_ASCII || format == PGM_BINARY || format == PPM_BINARY) {

        if (!fread(&maxValue, sizeof(maxValue), 1, fileStream)) {
            N2D2_THROW_OR_ABORT(std::runtime_error,
                "Cannot read maxValue from fileStream");
        }
    }
    
    std::size_t nbChannels = 1;
    if(format == PPM_ASCII || format == PPM_BINARY) {
        nbChannels = 3;
    }
    
    
    if(dataOut.empty()) {
        dataOut.resize(width*height*nbChannels);
    }
    else if(dataOut.size() != width*height*nbChannels) {
        N2D2_THROW_OR_ABORT(std::runtime_error, "Wrong dataOut size!");
    }
    assert(dataOut.size() == width*height*nbChannels);
    
    
    // Read new line character
    fgetc(fileStream);
    
    
    switch(format) {
        case PBM_ASCII:{
            char value;
            for(std::size_t i = 0; i < height*width*nbChannels; i++) {
                if (!fread(&value, sizeof(value), 1, fileStream)) {
                    N2D2_THROW_OR_ABORT(std::runtime_error,
                        "Cannot read value from fileStream at " + std::to_string(i));
                }
                dataOut[i] = (value == '1');
                
            }
            break;
        }
        case PBM_BINARY:{
            std::size_t i = 0;
            for(std::size_t y = 0; y < height; y++) {
                for(std::size_t x = 0; x < width; x += 8) {
                    unsigned char value;
                    if (!fread((char*) &value, sizeof(value), 1, fileStream)) {
                        N2D2_THROW_OR_ABORT(std::runtime_error,
                            "Cannot read value from fileStream at " + std::to_string(y * height + x));
                    }
                    
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
            for (std::size_t i = 0; i < dataOut.size(); ++i)
                if(fscanf(fileStream, "%c", &dataOut[i]) != 1) {
                    N2D2_THROW_OR_ABORT(std::runtime_error,
                        "Cannot get dataOut value from fileStream");
                }
            break;
        case PGM_BINARY:
        case PPM_BINARY:
            if (!fread((char*) dataOut.data(), dataOut.size(), 1, fileStream)) {
                N2D2_THROW_OR_ABORT(std::runtime_error,
                        "Cannot read dataOut values from fileStream");
            }
            break;
        default:
            N2D2_THROW_OR_ABORT(std::runtime_error,
                "The '" + file + "' file is not a valid Netpbm file.");
    }

    fclose(fileStream);
    
    // Rescale from [0-maxValue] to [0-255]
    if(rescale && maxValue != 255) {
        for (std::size_t i = 0; i < dataOut.size(); ++i) {
            dataOut[i] = (unsigned char) lround(dataOut[i]*255.0/maxValue);
        }
    }
}
