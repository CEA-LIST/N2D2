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

#include "Database/CIFAR_Database.hpp"

N2D2::CIFAR_Database::CIFAR_Database(double validation)
    : Database(true), mValidation(validation)
{
    // ctor
}

void N2D2::CIFAR_Database::loadCIFAR(const std::string& dataFile,
                                     const std::string& labelFile,
                                     bool coarseAndFine,
                                     bool useCoarse)
{
    const unsigned int nbRows = 32;
    const unsigned int nbColumns = 32;

    // Labels
    std::vector<std::string> labelsName;

    std::ifstream labels(labelFile.c_str());

    if (!labels.good())
        throw std::runtime_error("Could not open labels file: " + labelFile);

    std::string line;

    while (std::getline(labels, line))
        labelsName.push_back(line);

    labels.close();

    // Images
    std::ifstream images(dataFile.c_str(), std::fstream::binary);

    if (!images.good())
        throw std::runtime_error("Could not open images file: " + dataFile);

    const std::ifstream::pos_type size
        = images.seekg(0, std::ifstream::end).tellg();
    images.seekg(0, std::ifstream::beg);

    const unsigned int nbImages
        = size / (1 + coarseAndFine + 3 * nbRows * nbColumns);

    // For each image...
    for (unsigned int i = 0; i < nbImages; ++i) {
        // Read label
        unsigned char label;
        images.read(reinterpret_cast<char*>(&label), sizeof(label));

        if (coarseAndFine) {
            unsigned char fineLabel;
            images.read(reinterpret_cast<char*>(&fineLabel), sizeof(fineLabel));

            if (!useCoarse)
                label = fineLabel;
        }

        std::ostringstream nameStr;
        nameStr << dataFile << "[" << std::setfill('0') << std::setw(5) << i
                << "].ppm";

        // ... generate the stimuli
        if (!std::ifstream(nameStr.str()).good()) {
            cv::Mat frame(
                cv::Size(nbColumns, nbRows), CV_8UC3, cv::Scalar(0, 0, 0));

            for (int c = 2; c >= 0; --c) {
                for (unsigned int y = 0; y < nbRows; ++y) {
                    for (unsigned int x = 0; x < nbColumns; ++x) {
                        unsigned char color;
                        images.read(reinterpret_cast<char*>(&color),
                                    sizeof(color));

                        // Vec3b color order: blue, green, red
                        frame.at<cv::Vec3b>(y, x)[c] = color;
                    }
                }
            }

            if (!cv::imwrite(nameStr.str(), frame))
                throw std::runtime_error("Unable to write image: "
                                         + nameStr.str());
        } else {
            // Skip image data (to grab labels only)
            images.seekg(3 * nbColumns * nbRows, images.cur);
        }

        mStimuli.push_back(Stimulus(nameStr.str(), labelID(labelsName[label])));
        mStimuliSets(Unpartitioned).push_back(mStimuli.size() - 1);
    }

    if (images.eof())
        throw std::runtime_error(
            "End-of-file reached prematurely in data file: " + dataFile);
    else if (!images.good())
        throw std::runtime_error("Error while reading data file: " + dataFile);
    else if (images.get() != std::fstream::traits_type::eof())
        throw std::runtime_error("Data file size larger than expected: "
                                 + dataFile);
}

N2D2::CIFAR10_Database::CIFAR10_Database(double validation)
    : CIFAR_Database(validation)
{
    // ctor
}

void N2D2::CIFAR10_Database::load(const std::string& dataPath,
                                  const std::string& labelPath,
                                  bool /*extractROIs*/)
{
    const std::string labelPathDef = (labelPath.empty()) ? dataPath : labelPath;

    // Learn and validation stimuli
    loadCIFAR(dataPath + "/data_batch_1.bin",
              labelPathDef + "/batches.meta.txt");
    loadCIFAR(dataPath + "/data_batch_2.bin",
              labelPathDef + "/batches.meta.txt");
    loadCIFAR(dataPath + "/data_batch_3.bin",
              labelPathDef + "/batches.meta.txt");
    loadCIFAR(dataPath + "/data_batch_4.bin",
              labelPathDef + "/batches.meta.txt");
    loadCIFAR(dataPath + "/data_batch_5.bin",
              labelPathDef + "/batches.meta.txt");
    partitionStimuli(1.0 - mValidation, mValidation, 0.0);

    // Test stimuli
    loadCIFAR(dataPath + "/test_batch.bin", labelPathDef + "/batches.meta.txt");
    partitionStimuli(0.0, 0.0, 1.0);
}

N2D2::CIFAR100_Database::CIFAR100_Database(double validation, bool useCoarse)
    : CIFAR_Database(validation), mUseCoarse(useCoarse)
{
    // ctor
}

void N2D2::CIFAR100_Database::load(const std::string& dataPath,
                                   const std::string& labelPath,
                                   bool /*extractROIs*/)
{
    const std::string labelPathDef = (labelPath.empty()) ? dataPath : labelPath;

    // Learn and validation stimuli
    loadCIFAR(dataPath + "/train.bin",
              labelPathDef + ((mUseCoarse) ? "/coarse_label_names.txt"
                                           : "/fine_label_names.txt"),
              true,
              mUseCoarse);
    partitionStimuli(1.0 - mValidation, mValidation, 0.0);

    // Test stimuli
    loadCIFAR(dataPath + "/test.bin",
              labelPathDef + ((mUseCoarse) ? "/coarse_label_names.txt"
                                           : "/fine_label_names.txt"),
              true,
              mUseCoarse);
    partitionStimuli(0.0, 0.0, 1.0);
}
