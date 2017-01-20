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

#include "Database/FDDB_Database.hpp"

N2D2::FDDB_Database::FDDB_Database(double learn, double validation)
    : DIR_Database(), mLearn(learn), mValidation(validation)
{
    // ctor
}

void N2D2::FDDB_Database::load(const std::string& dataPath,
                               const std::string& labelPath,
                               bool /*extractROIs*/)
{
    const std::string labelPathDef = (labelPath.empty()) ? dataPath : labelPath;

    loadFold(dataPath, labelPathDef + "/FDDB-fold-01-ellipseList.txt");
    loadFold(dataPath, labelPathDef + "/FDDB-fold-02-ellipseList.txt");
    loadFold(dataPath, labelPathDef + "/FDDB-fold-03-ellipseList.txt");
    loadFold(dataPath, labelPathDef + "/FDDB-fold-04-ellipseList.txt");
    loadFold(dataPath, labelPathDef + "/FDDB-fold-05-ellipseList.txt");
    loadFold(dataPath, labelPathDef + "/FDDB-fold-06-ellipseList.txt");
    loadFold(dataPath, labelPathDef + "/FDDB-fold-07-ellipseList.txt");
    loadFold(dataPath, labelPathDef + "/FDDB-fold-08-ellipseList.txt");
    loadFold(dataPath, labelPathDef + "/FDDB-fold-09-ellipseList.txt");
    loadFold(dataPath, labelPathDef + "/FDDB-fold-10-ellipseList.txt");
    partitionStimuli(mLearn, mValidation, 1.0 - mLearn - mValidation);
}

void N2D2::FDDB_Database::loadFold(const std::string& dataPath,
                                   const std::string& labelPath)
{
    std::ifstream dataRoi(labelPath.c_str());

    if (!dataRoi.good())
        throw std::runtime_error(
            "FDDB_Database::loadFold(): could not open ROI data file: "
            + labelPath);

    // Create default label for no ROI
    if (!((std::string)mDefaultLabel).empty())
        labelID(mDefaultLabel);

    std::string line;
    std::string imageName;
    unsigned int nbFaces;
    ROIFileSection section = ImageName;

    while (std::getline(dataRoi, line)) {
        // Left trim & right trim (right trim necessary for extra "!value.eof()"
        // check later)
        line.erase(
            line.begin(),
            std::find_if(line.begin(),
                         line.end(),
                         std::not1(std::ptr_fun<int, int>(std::isspace))));
        line.erase(std::find_if(line.rbegin(),
                                line.rend(),
                                std::not1(std::ptr_fun<int, int>(std::isspace)))
                       .base(),
                   line.end());

        if (line.empty())
            continue;

        std::stringstream values(line);

        if (section == ImageName) {
            if (!(values >> imageName))
                throw std::runtime_error("Unreadable image name in data file: "
                                         + labelPath);

            const std::string stimulusName = dataPath + "/" + imageName
                                             + ".jpg";

            if (!std::ifstream(stimulusName.c_str()).good())
                throw std::runtime_error(
                    "FDDB_Database::loadFold(): stimulus does not exist: "
                    + stimulusName);

            mStimuli.push_back(Stimulus(stimulusName, -1));
            mStimuliSets(Unpartitioned).push_back(mStimuli.size() - 1);

            section = NumberOfFaces;
        } else if (section == NumberOfFaces) {
            if (!(Utils::signChecked<unsigned int>(values) >> nbFaces))
                throw std::runtime_error(
                    "Unreadable number of faces in data file: " + labelPath);

            section = (nbFaces > 0) ? Face : ImageName;
        } else if (section == Face) {
            double majorRadius;
            double minorRadius;
            double angle;
            double centerX;
            double centerY;
            std::string label;

            if (!(values >> majorRadius) || !(values >> minorRadius)
                || !(values >> angle) || !(values >> centerX)
                || !(values >> centerY) || !(values >> label)) {
                throw std::runtime_error("Unreadable value in data file: "
                                         + labelPath);
            }

            const int labelId = labelID(label);
            mStimuli.back().ROIs.push_back(new EllipticROI<int>(
                labelId,
                EllipticROI<int>::Point_T(centerX, centerY),
                majorRadius,
                minorRadius,
                angle));

            --nbFaces;

            if (nbFaces == 0)
                section = ImageName;
        }
    }
}
