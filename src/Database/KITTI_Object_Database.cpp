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

#include "Database/KITTI_Object_Database.hpp"

N2D2::KITTI_Object_Database::KITTI_Object_Database(double learn)
    : DIR_Database(), mLearn(learn)
{
    // ctor
}

void N2D2::KITTI_Object_Database::load(const std::string& dataPath,
                                const std::string& labelPath,
                                bool /*extractROIs*/)
{
    /** Learn & Validation KITTI Object Stimuli
        -Composite stimuli (1242x375) : 7481 training stimuli
    **/
    loadKITTIStimuli(dataPath + "/training/image_2", labelPath + "/training/label_2");
    partitionStimuli(mLearn, 1.0 - mLearn, 0.0);

    /** Test KITTI Stimuli
        -Composite stimuli (1242x375) : 7518 tests images
    **/
    loadKITTITestStimuli(dataPath + "/testing/image_2");
    partitionStimuli(0.0, 0.0, 1.0);
}

void N2D2::KITTI_Object_Database::loadKITTIStimuli(const std::string& dirPath,
                                            const std::string& labelPath)
{
    std::vector<std::string> labelfiles;

    struct dirent* lFile;
    DIR* lDir = opendir(labelPath.c_str());
    if (lDir == NULL)
        throw std::runtime_error(
            "Couldn't open the directory for the labeled files database: "
            + labelPath);

    while ((lFile = readdir(lDir)) != NULL) {
        if (lFile->d_name[0] != '.')
            labelfiles.push_back(std::string(labelPath + "/" + lFile->d_name));
    }

    closedir(lDir);
    std::sort(labelfiles.begin(), labelfiles.end());

    // Create default label for no ROI
    if (!((std::string)mDefaultLabel).empty())
        labelID(mDefaultLabel);

    const unsigned int stimuliWidth = 1242;
    const unsigned int stimuliHeight = 375;

    /**Type Parameters:
        Describes the type of object: 'Car', 'Van', 'Truck', 'Pedestrian',
    'Person_sitting', 'Cyclist', 'Tram', 'Misc' or 'DontCare'
    **/
    std::string objectType;
    /**Truncated parameters: Float from 0 (non-truncated) to 1 (truncated),
     where
                             truncated refers to the object leaving image
     boundaries.
     **/
    float truncated = 0;
    /**Occluded Parameters: Integer (0,1,2,3) indicating occlusion state:
     0 = fully visible, 1 = partly occluded
     2 = largely occluded, 3 = unknown
     **/
    float occluded = 0;
    /** Alpha Parameters: Observation angle of object, ranging [-pi..pi] **/
    float alpha = 0.0;
    /**Bounding Box Parameters**/
    float bb_left = 0.0;
    float bb_top = 0.0;
    float bb_right = 0.0;
    float bb_bottom = 0.0;
    /**3-Dimension Bounding Box Parameters:
    2D bounding box of object in the image (0-based index): contains left, top,
    right, bottom pixel coordinates
    **/
    float dim_height = 0.0;
    float dim_width = 0.0;
    float dim_length = 0.0;
    /**3D Bounding Box location Parameters: 3-D object dimensions: height,
     * width, length (in meters) **/
    float loc_x = 0.0;
    float loc_y = 0.0;
    float loc_z = 0.0;
    /**Rotation ry Parameters: Rotation ry around Y-axis in camera coordinates
     * [-pi..pi] **/
    float rotation_x = 0.0;

    std::vector<std::string> files;

    struct dirent* pFile;

    DIR* pDir = opendir(dirPath.c_str());
    if (pDir == NULL)
        throw std::runtime_error(
            "Couldn't open the directory for the directory database: "
            + dirPath);

    while ((pFile = readdir(pDir)) != NULL) {
        if (pFile->d_name[0] != '.')
            files.push_back(std::string(dirPath + "/" + pFile->d_name));
    }

    closedir(pDir);
    std::sort(files.begin(), files.end());

    std::cout << "Loading directory database \"" << dirPath << "\""
              << " size of the directory: " << files.size() << " picture"
              << std::endl;
    std::cout << "Loading the labels datafile \"" << labelPath << "\""
              << std::endl;
    for(unsigned int frameIdx = 0; frameIdx < labelfiles.size(); ++ frameIdx) {
        std::ifstream labelFile(labelfiles[frameIdx].c_str());

        if (!labelFile.good())
            throw std::runtime_error("Could not open validation labels file: "
                                     + labelfiles[frameIdx]);
        mStimuli.push_back(Stimulus(files[frameIdx], -1));
        mStimuliSets(Unpartitioned).push_back(mStimuli.size() - 1);

        while (!labelFile.eof()) {

            if (!(labelFile >> objectType) || !(labelFile >> truncated)
                || !(labelFile >> occluded) || !(labelFile>> alpha)
                || !(labelFile >> bb_left) || !(labelFile >> bb_top)
                || !(labelFile >> bb_right) || !(labelFile >> bb_bottom)
                || !(labelFile >> dim_height) || !(labelFile>> dim_width)
                || !(labelFile >> dim_length) || !(labelFile >> loc_x)
                || !(labelFile >> loc_y) || !(labelFile >> loc_z)
                || !(labelFile >> rotation_x)) {
                throw std::runtime_error("KITTI_Object_Database::loadKITTIStimuli(): "
                                         "unreadable values in file: " + labelfiles[frameIdx]);
            }
            unsigned int width = bb_right - bb_left;
            unsigned int height = bb_bottom - bb_top;

            if ((width + bb_left) > stimuliWidth) {
                std::cout << Utils::cwarning << "BBV right border >" << stimuliWidth
                          << " in picture " << files[frameIdx] << Utils::cdef
                          << std::endl;
                width = stimuliWidth - bb_left;
            }

            if ((height + bb_top) > stimuliHeight) {
                std::cout << Utils::cwarning << "BBV bottom border >"
                          << stimuliHeight << " in picture " << files[frameIdx]
                          << Utils::cdef << std::endl;
                height = stimuliHeight - bb_top;
            }

            mStimuli.back().ROIs.push_back(new RectangularROI<int>(
                labelID(objectType),
                RectangularROI<int>::Point_T(bb_left, bb_top),
                width,
                height));

            labelFile >> std::ws;  // eat up any leading white spaces

            int c = labelFile.peek();  // peek character

            if ( c == EOF )
                break;

        }
    }
}

void N2D2::KITTI_Object_Database::loadKITTITestStimuli(const std::string& dirPath)
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

    std::cout << Utils::cnotice << "Notice: The KITTI Object dataset doesn't "
                                   "provide label for the test files "
              << Utils::cdef << std::endl;

    for (unsigned int frame = 0; frame < files.size(); ++frame) {
        std::cout << " Loading test frame " << files[frame] << std::endl;
        mStimuli.push_back(Stimulus(files[frame], -1));
        mStimuliSets(Unpartitioned).push_back(mStimuli.size() - 1);
    }
}
