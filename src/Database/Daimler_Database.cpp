/*
    (C) Copyright 2016 CEA LIST. All Rights Reserved.
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

#include "Database/Daimler_Database.hpp"

N2D2::Daimler_Database::Daimler_Database(double learn,
                                         double validation,
                                         double test,
                                         bool fully)
    : DIR_Database(),
      mLearn(learn),
      mValidation(validation),
      mTest(test),
      mFully(fully)
{
    // ctor
}

void N2D2::Daimler_Database::load(const std::string& dataPath,
                                  const std::string& labelPath,
                                  bool /*extractROIs*/)
{
    if (!mFully) {
        /**For the learning patch neural network **/

        /** Learn & Validation Daimler Stimuli
            -Pedestrians 48x96: 15660 stimuli
            -Pedestrians 18x36: 15660 stimuli
            -Non-pedestrians 640x480: 6744 stimuli
        **/
        loadDaimlerStimuliPerDir(dataPath + "/TrainingData");
        partitionStimuli(mLearn, mValidation, mTest);

        std::cout
            << Utils::cwarning
            << "The Daimler TestData dataset is disabled in a patch mode. "
               "Using only the Daimler TrainingData dataset." << Utils::cdef
            << std::endl;
    } else {
        /**For the learning fully connected neural networks architecture **/
        std::cout << Utils::cwarning
                  << "The Daimler TrainingData dataset is disabled in "
                     "fully-CNN mode. "
                     "Using only the Daimler TestData dataset." << Utils::cdef
                  << std::endl;

        loadDaimlerTestStimuli(dataPath + "/TestData", labelPath);
        partitionStimuli(mLearn, mValidation, mTest);
    }
}

void N2D2::Daimler_Database::loadDaimlerStimuliPerDir(const std::string
                                                      & dirPath)
{
    loadDir(dirPath + "/Pedestrians/48x96", 0, "Pedestrians", 0);
    loadDir(dirPath + "/Pedestrians/18x36", 0, "Pedestrians", 0);
    loadDir(dirPath + "/NonPedestrians", 0, "NonPedestrians", 0);
}

void N2D2::Daimler_Database::loadDaimlerTestStimuli(const std::string& dirPath,
                                                    const std::string
                                                    & labelPath)
{
    std::ifstream labelTestFile(labelPath.c_str());
    if (!labelTestFile.good())
        throw std::runtime_error("Could not open validation labels file: "
                                 + labelPath);

    std::string stimuliName;

    // Create default label for no ROI
    if (!((std::string)mDefaultLabel).empty())
        labelID(mDefaultLabel);

    unsigned int stimuliNumberOfROI = 0;
    unsigned int datasetSize = 0;
    unsigned int stimuliWidth = 0;
    unsigned int stimuliHeight = 0;
    /**Object Class:
                        0=fully-visible pedestrian
                        1=bicyclist
                        2=motorcyclist
                        10=pedestrian group
                        255=partially visible pedestrian, bicyclist,
    motorcyclist
    **/
    unsigned int objectClass;
    /**Object ID:   Object ID to identify trajectories of the same physical
     * object **/
    unsigned int objectID;
    /**Object Unique ID: additional ID unique to each object entry**/
    unsigned int objectUniqueID;
    /**Confidence: confidence value indicating if this ground truth object is
     * required (1.0) or optional (0.0)**/
    unsigned int confidence;
    /**Bounding Box Parameters**/
    unsigned int bb_left = 0;
    unsigned int bb_top = 0;
    unsigned int bb_right = 0;
    unsigned int bb_bottom = 0;

    unsigned int undefinedField;

    // Parse the Header of the Daimler ROIs File
    if (!(labelTestFile >> stimuliName) || stimuliName != ":"
        || !(labelTestFile >> stimuliName)
        || stimuliName != "PAMI_2009_Detection_Benchmark"
        || !(labelTestFile >> stimuliName) // /PATH/TO/IMAGE/DATA/ or whatever
        || !(labelTestFile >> datasetSize) || !(labelTestFile >> stimuliName)
        || stimuliName != ";") {
        throw std::runtime_error("Daimler_Database::loadDaimlerTestStimuli(): "
                                 "wrong header in label file: " + labelPath);
    }

    std::cout << "Loading directory database \"" << dirPath << "\""
              << std::endl;
    std::cout << "Loading the ROIs datafile \"" << labelPath << "\""
              << std::endl;

    for (unsigned int i = 0; i < datasetSize - 1; ++i) {
        if (!(labelTestFile >> stimuliName) || !(labelTestFile >> stimuliWidth)
            || !(labelTestFile >> stimuliHeight)
            || !(labelTestFile >> undefinedField)
            || !(labelTestFile >> stimuliNumberOfROI)) {
            throw std::runtime_error("Daimler_Database::loadDaimlerTestStimuli("
                                     "): error in label file: " + labelPath);
        }

        mStimuli.push_back(Stimulus(dirPath + "/" + stimuliName, -1));
        mStimuliSets(Unpartitioned).push_back(mStimuli.size() - 1);

        for (unsigned int roi = 0; roi < stimuliNumberOfROI; ++roi) {
            if (!(labelTestFile >> undefinedField)
                || !(labelTestFile >> objectClass)
                || !(labelTestFile >> objectID)
                || !(labelTestFile >> objectUniqueID)
                || !(labelTestFile >> confidence) || !(labelTestFile >> bb_left)
                || !(labelTestFile >> bb_top) || !(labelTestFile >> bb_right)
                || !(labelTestFile >> bb_bottom)) {
                throw std::runtime_error("Daimler_Database::"
                                         "loadDaimlerTestStimuli(): error in "
                                         "ROI in label file: " + labelPath);
            }

            unsigned int width = bb_right - bb_left;
            unsigned int height = bb_bottom - bb_top;

            if ((width + bb_left) > 640) {
                std::cout << Utils::cwarning
                          << "BBV right border > 640 for image: " << stimuliName
                          << Utils::cdef << std::endl;
                width = 640 - bb_left;
            }

            if ((height + bb_top) > 480) {
                std::cout << Utils::cwarning
                          << "BBV bottom border > 480 for image: "
                          << stimuliName << Utils::cdef << std::endl;
                height = 480 - bb_top;
            }

            const std::string label
                = (objectClass == 0)
                      ? "Pedestrians"
                      : (objectClass == 1)
                            ? "Bicyclist"
                            : (objectClass == 2)
                                  ? "Motorcyclist"
                                  : (objectClass == 10) ? "Group" : "";

            if (!label.empty()) {
                mStimuli.back().ROIs.push_back(new RectangularROI<int>(
                    labelID(label),
                    RectangularROI<int>::Point_T(bb_left, bb_top),
                    width,
                    height));
            }

            if (!(labelTestFile >> undefinedField))
                throw std::runtime_error("Daimler_Database::"
                                         "loadDaimlerTestStimuli(): error in "
                                         "label file: " + labelPath);
        }

        if (!(labelTestFile >> stimuliName))
            break;

        if (stimuliName != ";") {
            throw std::runtime_error(
                "Wrong ROIs specification format in file " + labelPath
                + " End of the stimuli section must be ';' character but is '"
                + stimuliName.c_str() + "' ");
        }
    }
}
