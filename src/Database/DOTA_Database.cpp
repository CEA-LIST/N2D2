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

#include "Database/DOTA_Database.hpp"

N2D2::DOTA_Database::DOTA_Database(double learn,
                                   bool useValidationForTest)
    : DIR_Database(),
      mLearn(learn),
      mUseValidationForTest(useValidationForTest)
{
    // ctor
}

void N2D2::DOTA_Database::load(const std::string& dataPath,
                                     const std::string& labelPath,
                                     bool /*extractROIs*/)
{
    // Create default label for no ROI
    if (!((std::string)mDefaultLabel).empty())
        labelID(mDefaultLabel);

    const std::string labelPathDef = (labelPath.empty())
        ? dataPath : labelPath;

    // Learn & Tests Stimuli
    loadDir(dataPath + "/train/images", 1, "", -1);
    loadLabels(labelPathDef + "/train/labelTxt");
    partitionStimuli(mLearn, 0, 1.0 - mLearn);

    // Validation stimuli
    loadDir(dataPath + "/val/images", 1, "", -1);
    loadLabels(labelPathDef + "/val/labelTxt");
    partitionStimuli(0.0, 1.0, 0.0);

    if (mUseValidationForTest) {
        // Test stimuli (using validation database)
        loadDir(dataPath + "/val/images", 1, "", -1);
        loadLabels(labelPathDef + "/val/labelTxt");
        partitionStimuli(0.0, 0.0, 1.0);
    }
}

void N2D2::DOTA_Database::loadLabels(const std::string& labelPath) {
    for (std::vector<StimulusID>::const_iterator it
        = mStimuliSets(Unpartitioned).begin(),
         itEnd = mStimuliSets(Unpartitioned).end(); it != itEnd; ++it)
    {
        const std::string labelName = labelPath + "/"
            + Utils::baseName(Utils::fileBaseName(mStimuli[(*it)].name))
            + ".txt";

        std::ifstream labelData(labelName.c_str());

        if (!labelData.good()) {
            throw std::runtime_error("Could not open TXT label file "
                                     "(missing?): " + labelName);
        }

        std::string line;

        // read imagesource
        if (!std::getline(labelData, line) || line.find("imagesource:") != 0) {
            throw std::runtime_error("First line should start with "
                                     "\"imagesource:\" in: " + labelName);
        }

        // read gsd
        if (!std::getline(labelData, line) || line.find("gsd:") != 0) {
            throw std::runtime_error("Second line should start with "
                                     "\"gsd:\" in: " + labelName);
        }

        // read annotations
        while (std::getline(labelData, line)) {
            if (line.empty())
                continue;

            unsigned int x1, y1, x2, y2, x3, y3, x4, y4;
            std::string category;
            bool difficult;

            std::stringstream values(line);

            if (!(Utils::signChecked<unsigned int>(values) >> x1)
                || !(Utils::signChecked<unsigned int>(values) >> y1)
                || !(Utils::signChecked<unsigned int>(values) >> x2)
                || !(Utils::signChecked<unsigned int>(values) >> y2)
                || !(Utils::signChecked<unsigned int>(values) >> x3)
                || !(Utils::signChecked<unsigned int>(values) >> y3)
                || !(Utils::signChecked<unsigned int>(values) >> x4)
                || !(Utils::signChecked<unsigned int>(values) >> y4)
                || !(values >> category)
                || !(values >> difficult))
            {
                throw std::runtime_error("DOTA_Database: unreadable value in "
                                         "line \"" + line + "\" for file: "
                                         + labelPath);
            }

            std::vector<cv::Point> pts;
            pts.push_back(cv::Point(x1, y1));
            pts.push_back(cv::Point(x2, y2));
            pts.push_back(cv::Point(x3, y3));
            pts.push_back(cv::Point(x4, y4));

            mStimuli[(*it)].ROIs.push_back(new PolygonalROI<int>(
                labelID(category), pts));
        }
    }
}
