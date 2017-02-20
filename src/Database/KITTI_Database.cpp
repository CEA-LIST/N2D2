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

#include "Database/KITTI_Database.hpp"

N2D2::KITTI_Database::KITTI_Database(double learn)
    : DIR_Database(), mLearn(learn)
{
    // ctor
}

void N2D2::KITTI_Database::load(const std::string& dataPath,
                                const std::string& labelPath,
                                bool /*extractROIs*/)
{
    /** Learn & Validation KITTI Stimuli
        -Composite stimuli (1242x375) : Sequence from 0 to 19 of the KITTI
    learning dataset, 7171 stimuli
    **/
    loadKITTIStimuliPerDir(dataPath, labelPath);
    partitionStimuli(mLearn, 1.0 - mLearn, 0.0);

    /** Test KITTI Stimuli
        -Composite stimuli (1242x375) : Sequence 20 of the KITTI learning
    dataset, 837 stimuli
    **/
    loadKITTIStimuli(dataPath + "/0020", labelPath + "/0020.txt");
    partitionStimuli(0.0, 0.0, 1.0);
}

void N2D2::KITTI_Database::loadKITTIStimuliPerDir(const std::string& dirPath,
                                                  const std::string& labelPath)
{
    loadKITTIStimuli(dirPath + "/0000", labelPath + "/0000.txt");
    loadKITTIStimuli(dirPath + "/0001", labelPath + "/0001.txt");
    loadKITTIStimuli(dirPath + "/0002", labelPath + "/0002.txt");
    loadKITTIStimuli(dirPath + "/0003", labelPath + "/0003.txt");
    loadKITTIStimuli(dirPath + "/0004", labelPath + "/0004.txt");
    loadKITTIStimuli(dirPath + "/0005", labelPath + "/0005.txt");
    loadKITTIStimuli(dirPath + "/0006", labelPath + "/0006.txt");
    loadKITTIStimuli(dirPath + "/0007", labelPath + "/0007.txt");
    loadKITTIStimuli(dirPath + "/0008", labelPath + "/0008.txt");
    loadKITTIStimuli(dirPath + "/0009", labelPath + "/0009.txt");
    loadKITTIStimuli(dirPath + "/0010", labelPath + "/0010.txt");
    loadKITTIStimuli(dirPath + "/0011", labelPath + "/0011.txt");
    loadKITTIStimuli(dirPath + "/0012", labelPath + "/0012.txt");
    loadKITTIStimuli(dirPath + "/0013", labelPath + "/0013.txt");
    loadKITTIStimuli(dirPath + "/0014", labelPath + "/0014.txt");
    loadKITTIStimuli(dirPath + "/0015", labelPath + "/0015.txt");
    loadKITTIStimuli(dirPath + "/0016", labelPath + "/0016.txt");
    loadKITTIStimuli(dirPath + "/0017", labelPath + "/0017.txt");
    loadKITTIStimuli(dirPath + "/0018", labelPath + "/0018.txt");
    loadKITTIStimuli(dirPath + "/0019", labelPath + "/0019.txt");
}

void N2D2::KITTI_Database::loadKITTIStimuli(const std::string& dirPath,
                                            const std::string& labelPath)
{
    std::ifstream labelFile(labelPath.c_str());

    if (!labelFile.good())
        throw std::runtime_error("Could not open validation labels file: "
                                 + labelPath);

    // Create default label for no ROI
    if (!((std::string)mDefaultLabel).empty())
        labelID(mDefaultLabel);

    const unsigned int stimuliWidth = 1242;
    const unsigned int stimuliHeight = 375;

    /**frame Parameters: Frame within the sequence where the object appearers**/
    int frame = 0;
    /**Track_ID Parameters: Unique tracking id of this object within this
     * sequence**/
    int track_ID = 0;
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
    int truncated = 0;
    /**Occluded Parameters: Integer (0,1,2,3) indicating occlusion state:
     0 = fully visible, 1 = partly occluded
     2 = largely occluded, 3 = unknown
     **/
    int occluded = 0;
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

    int previous_frame = -1;
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

    while (!labelFile.eof()) {
        if (!(labelFile >> frame) || !(labelFile >> track_ID)
            || !(labelFile >> objectType) || !(labelFile >> truncated)
            || !(labelFile >> occluded) || !(labelFile >> alpha)
            || !(labelFile >> bb_left) || !(labelFile >> bb_top)
            || !(labelFile >> bb_right) || !(labelFile >> bb_bottom)
            || !(labelFile >> dim_height) || !(labelFile >> dim_width)
            || !(labelFile >> dim_length) || !(labelFile >> loc_x)
            || !(labelFile >> loc_y) || !(labelFile >> loc_z)
            || !(labelFile >> rotation_x)) {
            throw std::runtime_error("KITTI_Database::loadKITTIStimuli(): "
                                     "unreadable values in file: " + labelPath);
        }

        if ((previous_frame == -1) || (previous_frame != frame)) {
            mStimuli.push_back(Stimulus(files[frame], -1));
            mStimuliSets(Unpartitioned).push_back(mStimuli.size() - 1);
        }

        previous_frame = frame;

        unsigned int width = bb_right - bb_left;
        unsigned int height = bb_bottom - bb_top;

        if ((width + bb_left) > stimuliWidth) {
            std::cout << Utils::cwarning << "BBV right border >" << stimuliWidth
                      << " in picture " << files[frame] << Utils::cdef
                      << std::endl;
            width = stimuliWidth - bb_left;
        }

        if ((height + bb_top) > stimuliHeight) {
            std::cout << Utils::cwarning << "BBV bottom border >"
                      << stimuliHeight << " in picture " << files[frame]
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
