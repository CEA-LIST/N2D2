/*
    (C) Copyright 2016 CEA LIST. All Rights Reserved.
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

#include "Database/IDX_Database.hpp"

N2D2::IDX_Database::IDX_Database(bool loadDataInMemory)
    : Database(loadDataInMemory)
{
    // ctor
}

void N2D2::IDX_Database::load(const std::string& dataPath,
                              const std::string& labelPath,
                              bool /*extractROIs*/)
{
    // Images
    std::ifstream images(dataPath.c_str(), std::fstream::binary);

    if (!images.good())
        throw std::runtime_error("Could not open images file: " + dataPath);

    MagicNumber magicNumber;
    unsigned int nbImages;
    unsigned int nbRows;
    unsigned int nbColumns;

    images.read(reinterpret_cast<char*>(&magicNumber.value),
                sizeof(magicNumber));
    images.read(reinterpret_cast<char*>(&nbImages), sizeof(nbImages));
    images.read(reinterpret_cast<char*>(&nbRows), sizeof(nbRows));
    images.read(reinterpret_cast<char*>(&nbColumns), sizeof(nbColumns));

    if (!Utils::isBigEndian()) {
        Utils::swapEndian(magicNumber.value);
        Utils::swapEndian(nbImages);
        Utils::swapEndian(nbRows);
        Utils::swapEndian(nbColumns);
    }

    if (magicNumber.byte[3] != 0 || magicNumber.byte[2] != 0
        || magicNumber.byte[1] != Unsigned || magicNumber.byte[0] != 3) {
        throw std::runtime_error("Wrong file format for images file: "
                                 + dataPath);
    }

    // Labels
    std::ifstream labels(labelPath.c_str(), std::fstream::binary);

    if (!labels.good())
        throw std::runtime_error("Could not open labels file: " + labelPath);

    MagicNumber magicNumberLabels;
    unsigned int nbItemsLabels;

    labels.read(reinterpret_cast<char*>(&magicNumberLabels.value),
                sizeof(magicNumberLabels));
    labels.read(reinterpret_cast<char*>(&nbItemsLabels), sizeof(nbItemsLabels));

    if (!Utils::isBigEndian()) {
        Utils::swapEndian(magicNumberLabels);
        Utils::swapEndian(nbItemsLabels);
    }

    if (magicNumberLabels.byte[3] != 0 || magicNumberLabels.byte[2] != 0
        || magicNumberLabels.byte[1] != Unsigned
        || magicNumberLabels.byte[0] != 1) {
        throw std::runtime_error("Wrong file format for labels file: "
                                 + labelPath);
    }

    if (nbImages != nbItemsLabels)
        throw std::runtime_error(
            "The number of images and the number of labels does not match.");

    // For each image...
    for (unsigned int i = 0; i < nbImages; ++i) {
        unsigned char buff;

        std::ostringstream nameStr;
        nameStr << dataPath << "[" << std::setfill('0') << std::setw(5) << i
                << "].pgm";

        // ... generate the stimuli
        if (!std::ifstream(nameStr.str()).good()) {
            cv::Mat frame(cv::Size(nbColumns, nbRows), CV_8UC1);

            for (unsigned int y = 0; y < nbRows; ++y) {
                for (unsigned int x = 0; x < nbColumns; ++x) {
                    images.read(reinterpret_cast<char*>(&buff), sizeof(buff));
                    frame.at<unsigned char>(y, x) = buff;
                }
            }

            if (!cv::imwrite(nameStr.str(), frame))
                throw std::runtime_error("Unable to write image: "
                                         + nameStr.str());
        } else {
            // Skip image data (to grab labels only)
            images.seekg(nbColumns * nbRows, images.cur);
        }

        // ... attach the corresponding label
        labels.read(reinterpret_cast<char*>(&buff), sizeof(buff));

        std::ostringstream labelStr;
        labelStr << (unsigned int)buff;

        mStimuli.push_back(Stimulus(nameStr.str(), labelID(labelStr.str())));
        mStimuliSets(Unpartitioned).push_back(mStimuli.size() - 1);
    }

    if (images.eof())
        throw std::runtime_error(
            "End-of-file reached prematurely in data file: " + dataPath);
    else if (!images.good())
        throw std::runtime_error("Error while reading data file: " + dataPath);
    else if (images.get() != std::fstream::traits_type::eof())
        throw std::runtime_error("Data file size larger than expected: "
                                 + dataPath);

    if (labels.eof())
        throw std::runtime_error(
            "End-of-file reached prematurely in data file: " + labelPath);
    else if (!labels.good())
        throw std::runtime_error("Error while reading data file: " + labelPath);
    else if (labels.get() != std::fstream::traits_type::eof())
        throw std::runtime_error("Data file size larger than expected: "
                                 + labelPath);
}
