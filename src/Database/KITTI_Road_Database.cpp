/*
    (C) Copyright 2016 CEA LIST. All Rights Reserved.
    Contributor(s): Olivier BICHLER (olivier.bichler@cea.fr)
                    Benjamin BERTELONE (benjamin.bertelone@cea.fr)
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

#include "Database/KITTI_Road_Database.hpp"

N2D2::KITTI_Road_Database::KITTI_Road_Database(double learn)
    : DIR_Database(), mLearn(learn)
{
    // ctor
}

void N2D2::KITTI_Road_Database::load(const std::string& dataPath,
                                     const std::string& labelPath,
                                     bool /*extractROIs*/)
{
    /** Learn & Validation KITTI Road Stimuli
    **/
    loadKITTIRoadStimuli(dataPath, labelPath);
    partitionStimuli(mLearn, 1.0 - mLearn, 0.0);

    /** Test KITTI Road Stimuli : The test dataset doesn't provide labels
    **/
    const std::string testPath = N2D2_DATA("KITTI/data_road/testing/image_2");
    loadKITTITestStimuli(testPath);
    partitionStimuli(0.0, 0.0, 1.0);
}

void N2D2::KITTI_Road_Database::loadKITTIRoadStimuli(const std::string& dirPath,
                                                     const std::string
                                                     & labelPath)
{
    std::vector<std::string> files;
    std::vector<std::string> labels;
    struct dirent* pFile;
    struct stat fileStat;
    const std::string label;
    DIR* pDirLabel = opendir(labelPath.c_str());
    if (pDirLabel == NULL)
        throw std::runtime_error("Couldn't open labels directory: "
                                 + labelPath);

    DIR* pDirData = opendir(dirPath.c_str());
    if (pDirData == NULL)
        throw std::runtime_error("Couldn't open database directory: "
                                 + dirPath);

    // Read all data files in the kitti/data_road/training directory
    while ((pFile = readdir(pDirData))) {
        const std::string fileName(pFile->d_name);
        const std::string filePath(dirPath + "/" + fileName);

        // Ignore file in case of stat failure
        if (stat(filePath.c_str(), &fileStat) < 0)
            continue;
        // Exclude current and parent directories
        if (!strcmp(pFile->d_name, ".") || !strcmp(pFile->d_name, ".."))
            continue;

        // Exclude files with wrong extension
        std::string fileExtension = Utils::fileExtension(fileName);
        std::transform(fileExtension.begin(),
                       fileExtension.end(),
                       fileExtension.begin(),
                       ::tolower);

        if (mValidExtensions.empty() || std::find(mValidExtensions.begin(),
                                                  mValidExtensions.end(),
                                                  fileExtension)
                                        != mValidExtensions.end()) {
            if (!Registrar<DataFile>::exists(fileExtension)) {
                std::cout << Utils::cnotice << "Notice: file " << fileName
                          << " does not appear to be a valid stimulus,"
                             " ignoring." << Utils::cdef << std::endl;
                continue;
            }

            files.push_back(filePath);
        }
    }
    // Read all gt files in the kitti/data_road/gt_image_2 directory
    while ((pFile = readdir(pDirLabel))) {
        const std::string fileName(pFile->d_name);
        const std::string filePath(labelPath + "/" + fileName);
        const std::string idFile = fileName.substr(0,7); //um_lane
        if(idFile == "um_lane")
            continue;
        // Ignore file in case of stat failure
        if (stat(filePath.c_str(), &fileStat) < 0)
            continue;
        // Exclude current and parent directories
        if (!strcmp(pFile->d_name, ".") || !strcmp(pFile->d_name, ".."))
            continue;

        // Exclude files with wrong extension
        std::string fileExtension = Utils::fileExtension(fileName);
        std::transform(fileExtension.begin(),
                       fileExtension.end(),
                       fileExtension.begin(),
                       ::tolower);

        if (mValidExtensions.empty() || std::find(mValidExtensions.begin(),
                                                  mValidExtensions.end(),
                                                  fileExtension)
                                        != mValidExtensions.end()) {
            if (!Registrar<DataFile>::exists(fileExtension)) {
                std::cout << Utils::cnotice << "Notice: file " << fileName
                          << " does not appear to be a valid stimulus,"
                             " ignoring." << Utils::cdef << std::endl;
                continue;
            }

            labels.push_back(filePath);
        }
    }

    if (labels.size() != files.size())
        throw std::runtime_error("The number of Ground Truth Labels files are "
                                 "different than the stimuli file");


    // Create default label for no ROI
    if (!((std::string)mDefaultLabel).empty())
        labelID(mDefaultLabel);

    std::sort(files.begin(), files.end());
    std::sort(labels.begin(), labels.end());

    for (unsigned int frame = 0; frame < labels.size(); ++frame) {
        std::cout << "Loading labels ground truth: " << labels[frame]
                  << "\n    (datafile: " << files[frame] << ")" << std::endl;

        cv::Mat gtLabels = cv::imread(labels[frame], CV_LOAD_IMAGE_COLOR)
                               .clone(); // Read the ground truth label file

        if (!gtLabels.data)
            throw std::runtime_error("KITTI_Road_Database::"
                                     "loadKITTIRoadStimuli(): Could not open "
                                     "or find image: " + labels[frame]);

        std::vector<cv::Point> pts;

        mStimuli.push_back(Stimulus(files[frame], -1));
        mStimuliSets(Unpartitioned).push_back(mStimuli.size() - 1);

        for (int r = 0; r < gtLabels.rows; r++) {
            bool borderFlag = false;

            for (int c = 0; c < gtLabels.cols; c++) {
                if ((r + 1 < gtLabels.rows) && (c + 1 < gtLabels.cols)) {
                    if ((int)gtLabels.at<cv::Vec3b>(r, c)[0] == 255
                        && (int)gtLabels.at<cv::Vec3b>(r, c)[2] == 255
                        && borderFlag)
                        pts.back() = cv::Point(c, r);
                }

                if (!borderFlag) {
                    if ((int)gtLabels.at<cv::Vec3b>(r, c)[0] == 255
                        && (int)gtLabels.at<cv::Vec3b>(r, c)[2] == 255) {
                        pts.push_back(cv::Point(c, r));
                        pts.push_back(cv::Point(c, r));
                        borderFlag = true;
                    }
                }
            }
        }

        mStimuli.back().ROIs.push_back(new PolygonalROI
                                       <int>(labelID("Road"), pts));
    }
}

void N2D2::KITTI_Road_Database::loadKITTITestStimuli(const std::string& dirPath)
{
    std::vector<std::string> files;

    struct dirent* pFile;
    struct stat fileStat;

    DIR* pDirData = opendir(dirPath.c_str());
    if (pDirData == NULL)
        throw std::runtime_error("Couldn't open test database directory: "
                                 + dirPath);

    // Read all data files in the imdb directory
    while ((pFile = readdir(pDirData))) {
        const std::string fileName(pFile->d_name);
        const std::string filePath(dirPath + "/" + fileName);

        // Ignore file in case of stat failure
        if (stat(filePath.c_str(), &fileStat) < 0)
            continue;
        // Exclude current and parent directories
        if (!strcmp(pFile->d_name, ".") || !strcmp(pFile->d_name, ".."))
            continue;

        // Exclude files with wrong extension
        std::string fileExtension = Utils::fileExtension(fileName);
        std::transform(fileExtension.begin(),
                       fileExtension.end(),
                       fileExtension.begin(),
                       ::tolower);

        if (mValidExtensions.empty() || std::find(mValidExtensions.begin(),
                                                  mValidExtensions.end(),
                                                  fileExtension)
                                        != mValidExtensions.end()) {
            if (!Registrar<DataFile>::exists(fileExtension)) {
                std::cout << Utils::cnotice << "Notice: file " << fileName
                          << " does not appear to be a valid stimulus,"
                             " ignoring." << Utils::cdef << std::endl;
                continue;
            }

            files.push_back(filePath);
        }
    }

    // Create default label for no ROI
    if (!((std::string)mDefaultLabel).empty())
        labelID(mDefaultLabel);

    std::sort(files.begin(), files.end());

    std::cout << Utils::cnotice << "Notice: The KITTI Road dataset doesn't "
                                   "provide label for the test files "
              << Utils::cdef << std::endl;

    for (unsigned int frame = 0; frame < files.size(); ++frame) {
        std::cout << " Loading test frame " << files[frame] << std::endl;
        mStimuli.push_back(Stimulus(files[frame], -1));
        mStimuliSets(Unpartitioned).push_back(mStimuli.size() - 1);
    }
}
