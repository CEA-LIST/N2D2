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

#include "Database/CelebA_Database.hpp"
#include "ROI/RectangularROI.hpp"

N2D2::CelebA_Database::CelebA_Database(
    bool inTheWild,
    bool withLandmarks)
    : DIR_Database(),
      mInTheWild(inTheWild),
      mWithLandmarks(withLandmarks)
{
    // ctor
}

void N2D2::CelebA_Database::load(const std::string& dataPath,
                                 const std::string& labelPath,
                                 bool /*extractROIs*/)
{
    std::map<std::string, FaceParameters> facesParam
        = loadFaceParameters(labelPath);

    if (mInTheWild)
        loadStimuli(facesParam, dataPath + "/img_celeba/");
    else
        loadStimuli(facesParam, dataPath + "/img_align_celeba/");

    partitionFace(facesParam,
                  Utils::dirName(labelPath) + "/Eval/list_eval_partition.txt");
}

std::map<std::string, N2D2::CelebA_Database::FaceParameters>
N2D2::CelebA_Database::loadFaceParameters(const std::string& path) const
{
    const std::string identity = path + "/identity_CelebA.txt";

    std::ifstream identityData(identity.c_str());

    if (!identityData.good()) {
        throw std::runtime_error(
            "CelebA_Database::loadFaceParameters(): could not open "
                + identity);
    }

    std::string line;
    std::map<std::string, FaceParameters> data;

    while (std::getline(identityData, line)) {
        // Left trim & right trim
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

        if (line[0] == '#') {
            std::cout << Utils::cnotice << "Ignoring commented line: \"" << line
                      << "\" in file: " << identity << Utils::cdef << std::endl;
            continue;
        }

        std::stringstream values(line);

        std::string image_id;
        FaceParameters fp;

        if (!(values >> image_id)
            || !(Utils::signChecked<unsigned int>(values) >> fp.identity))
        {
            throw std::runtime_error("CelebA_Database::loadStimuli(): "
                                     "unreadable value in line \"" + line
                                     + "\" for file: " + identity);
        }
        else {
            bool newInsert;
            std::tie(std::ignore, newInsert)
                = data.insert(std::make_pair(image_id, fp));

            if (!newInsert) {
                throw std::runtime_error("CelebA_Database::loadStimuli(): "
                                        "duplicate image_id in line \"" + line
                                        + "\" for file: " + identity);
            }
        }
    }

    if (mInTheWild) {
        const std::string bbox = path + "/list_bbox_celeba.txt";

        std::ifstream bboxData(bbox.c_str());

        if (!bboxData.good()) {
            throw std::runtime_error(
                "CelebA_Database::loadFaceParameters(): could not open "
                    + bbox);
        }

        std::string line;

        // Skip first line (count)
        std::getline(bboxData, line);
        // Skip second line (header)
        std::getline(bboxData, line);

        while (std::getline(bboxData, line)) {
            // Left trim & right trim
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

            if (line[0] == '#') {
                std::cout << Utils::cnotice << "Ignoring commented line: \"" << line
                        << "\" in file: " << bbox << Utils::cdef << std::endl;
                continue;
            }

            std::stringstream values(line);
            std::string image_id;

            if (!(values >> image_id)) {
                throw std::runtime_error("CelebA_Database::loadStimuli(): "
                                        "unreadable value in line \"" + line
                                        + "\" for file: " + bbox);
            }

            std::map<std::string, FaceParameters>::iterator
                it = data.find(image_id);

            if (it == data.end()) {
                throw std::runtime_error("CelebA_Database::loadStimuli(): "
                                        "unknown image_id in line \"" + line
                                        + "\" for file: " + bbox);
            }

            FaceParameters& fp = (*it).second;

            if (!(Utils::signChecked<unsigned int>(values) >> fp.x_1)
                || !(Utils::signChecked<unsigned int>(values) >> fp.y_1)
                || !(Utils::signChecked<unsigned int>(values) >> fp.width)
                || !(Utils::signChecked<unsigned int>(values) >> fp.height))
            {
                throw std::runtime_error("CelebA_Database::loadStimuli(): "
                                        "unreadable value in line \"" + line
                                        + "\" for file: " + bbox);
            }
        }
    }

    if (mWithLandmarks) {
        const std::string landmarks = (mInTheWild)
            ? path + "/list_landmarks_celeba.txt"
            : path + "/list_landmarks_align_celeba.txt";

        std::ifstream landmarksData(landmarks.c_str());

        if (!landmarksData.good()) {
            throw std::runtime_error(
                "CelebA_Database::loadFaceParameters(): could not open "
                    + landmarks);
        }

        std::string line;

        // Skip first line (count)
        std::getline(landmarksData, line);
        // Skip second line (header)
        std::getline(landmarksData, line);

        while (std::getline(landmarksData, line)) {
            // Left trim & right trim
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

            if (line[0] == '#') {
                std::cout << Utils::cnotice << "Ignoring commented line: \""
                    << line << "\" in file: " << landmarks << Utils::cdef
                    << std::endl;
                continue;
            }

            std::stringstream values(line);
            std::string image_id;

            if (!(values >> image_id)) {
                throw std::runtime_error("CelebA_Database::loadStimuli(): "
                                        "unreadable value in line \"" + line
                                        + "\" for file: " + landmarks);
            }

            std::map<std::string, FaceParameters>::iterator
                it = data.find(image_id);

            if (it == data.end()) {
                throw std::runtime_error("CelebA_Database::loadStimuli(): "
                                        "unknown image_id in line \"" + line
                                        + "\" for file: " + landmarks);
            }

            FaceParameters& fp = (*it).second;

            if (!(Utils::signChecked<unsigned int>(values) >> fp.lefteye_x)
                || !(Utils::signChecked<unsigned int>(values) >> fp.lefteye_y)
                || !(Utils::signChecked<unsigned int>(values) >> fp.righteye_x)
                || !(Utils::signChecked<unsigned int>(values) >> fp.righteye_y)
                || !(Utils::signChecked<unsigned int>(values) >> fp.nose_x)
                || !(Utils::signChecked<unsigned int>(values) >> fp.nose_y)
                || !(Utils::signChecked<unsigned int>(values) >> fp.leftmouth_x)
                || !(Utils::signChecked<unsigned int>(values) >> fp.leftmouth_y)
                || !(Utils::signChecked<unsigned int>(values) >> fp.rightmouth_x)
                || !(Utils::signChecked<unsigned int>(values) >> fp.rightmouth_y))
            {
                throw std::runtime_error("CelebA_Database::loadStimuli(): "
                                        "unreadable value in line \"" + line
                                        + "\" for file: " + landmarks);
            }
        }
    }

    std::cout << "CelebA_Database: Found: " << data.size()
              << " data labeled." << std::endl;
    return data;
}

void N2D2::CelebA_Database::loadStimuli(
    std::map<std::string, FaceParameters>& facesParam,
    const std::string& dirPath)
{
    // Create default label for no ROI
    if (!((std::string)mDefaultLabel).empty())
        labelID(mDefaultLabel);

    // Match CelebA dataset identity with N2D2 label ID --->
    std::vector<unsigned int> identities;
    identities.reserve(facesParam.size());

    for (std::map<std::string, FaceParameters>::const_iterator it
        = facesParam.begin(), itEnd = facesParam.end(); it != itEnd; ++it)
    {
        identities.push_back((*it).second.identity);
    }

    std::sort(identities.begin(), identities.end());

    for (std::vector<unsigned int>::const_iterator it
        = identities.begin(), itEnd = identities.end(); it != itEnd; ++it)
    {
        std::ostringstream labelStr;
        labelStr << (*it);

        labelID(labelStr.str());
    }
    // <---

    for (std::map<std::string, FaceParameters>::iterator it
        = facesParam.begin(), itEnd = facesParam.end(); it != itEnd; ++it)
    {
        FaceParameters& fp = (*it).second;
        std::ostringstream labelStr;
        labelStr << fp.identity;

        const std::string fileName = dirPath + "/" + (*it).first;

        if (std::ifstream(fileName.c_str()).good()) {
            if (mInTheWild) {
                mStimuli.push_back(Stimulus(fileName, -1));
                fp.id = mStimuli.size() - 1;
                mStimuliSets(Unpartitioned).push_back(fp.id);

                mStimuli.back().ROIs.push_back(new RectangularROI<int>(
                    labelID(labelStr.str()),
                    RectangularROI<int>::Point_T(fp.x_1, fp.y_1),
                    fp.width,
                    fp.height));
            } 
            else
                fp.id = loadFile(fileName, labelID(labelStr.str()));
        }
        else {
            fp.id = -1;

            std::cout << Utils::cwarning << "CelebA_Database::loadStimuli(): "
                "missing file: " << fileName << Utils::cdef << std::endl;
        }
    }
}

void N2D2::CelebA_Database::partitionFace(
    const std::map<std::string, FaceParameters>& facesParam,
    const std::string& partitionFile)
{
    std::ifstream data(partitionFile.c_str());

    if (!data.good()) {
        throw std::runtime_error(
            "CelebA_Database::partitionFace(): could not open "
                + partitionFile);
    }

    std::string line;

    while (std::getline(data, line)) {
        // Left trim & right trim
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

        if (line[0] == '#') {
            std::cout << Utils::cnotice << "Ignoring commented line: \"" << line
                      << "\" in file: " << partitionFile << Utils::cdef << std::endl;
            continue;
        }

        std::stringstream values(line);
        std::string image_id;

        if (!(values >> image_id)) {
            throw std::runtime_error("CelebA_Database::partitionFace(): "
                                    "unreadable value in line \"" + line
                                    + "\" for file: " + partitionFile);
        }

        const std::map<std::string, FaceParameters>::const_iterator
            it = facesParam.find(image_id);

        if (it == facesParam.end()) {
            throw std::runtime_error("CelebA_Database::partitionFace(): "
                                    "unknown image_id in line \"" + line
                                    + "\" for file: " + partitionFile);
        }

        const FaceParameters& fp = (*it).second;
        unsigned int partition;

        if (!(Utils::signChecked<unsigned int>(values) >> partition)) {
            throw std::runtime_error("CelebA_Database::partitionFace(): "
                                    "unreadable value in line \"" + line
                                    + "\" for file: " + partitionFile);
        }

        if (fp.id >= 0) {
            if (partition == 0)
                partitionStimulus(fp.id, Learn);
            else if (partition == 1)
                partitionStimulus(fp.id, Validation);
            else if (partition == 2)
                partitionStimulus(fp.id, Test);
            else {
                throw std::runtime_error("CelebA_Database::partitionFace(): "
                                        "unexpected value in line \"" + line
                                        + "\" for file: " + partitionFile);
            }
        }
    }
}

cv::Mat N2D2::CelebA_Database::getStimulusTargetData(StimulusID /*id*/,
    const cv::Mat& frame,
    const cv::Mat& /*labels*/,
    const std::vector<std::shared_ptr<ROI> >& labelsROI)
{
    assert(labelsROI.size() == 1);

    // Get the bounding box of the ROI after transformations
    const cv::Rect rect = labelsROI[0]->getBoundingRect();
    const cv::Mat channels[4] = {
        cv::Mat(1, 1, CV_32F, rect.x / (float)frame.cols),
        cv::Mat(1, 1, CV_32F, (rect.x + rect.width) / (float)frame.cols),
        cv::Mat(1, 1, CV_32F, rect.y / (float)frame.rows),
        cv::Mat(1, 1, CV_32F, (rect.y + rect.height) / (float)frame.rows)
    };

    cv::Mat faceBox;
    cv::merge(channels, 4, faceBox);
    return faceBox;
}
