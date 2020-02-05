

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

#ifndef N2D2_CELEBA_DATABASE_H
#define N2D2_CELEBA_DATABASE_H

#include "Database.hpp"
#include "Database/DIR_Database.hpp"
#include "N2D2.hpp"
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

namespace N2D2 {
class CelebA_Database : public DIR_Database {
public:
    struct FaceParameters {
        // N2D2
        int id;
        // identity_CelebA.txt
        unsigned int identity;
        // list_bbox_celeba.txt
        unsigned int x_1;
        unsigned int y_1;
        unsigned int width;
        unsigned int height;
        // list_landmarks_celeba.txt
        unsigned int lefteye_x;
        unsigned int lefteye_y;
        unsigned int righteye_x;
        unsigned int righteye_y;
        unsigned int nose_x;
        unsigned int nose_y;
        unsigned int leftmouth_x;
        unsigned int leftmouth_y;
        unsigned int rightmouth_x;
        unsigned int rightmouth_y;
    };

    CelebA_Database(bool inTheWild,
                    bool withLandmarks);
    virtual void load(const std::string& dataPath,
                      const std::string& labelPath = "",
                      bool /*extractROIs*/ = false);
    virtual ~CelebA_Database() {};

protected:
    cv::Mat getStimulusTargetData(StimulusID id,
                        const cv::Mat& frame = cv::Mat(),
                        const cv::Mat& labels = cv::Mat(),
                        const std::vector<std::shared_ptr<ROI> >& labelsROI
                            = std::vector<std::shared_ptr<ROI> >());
    std::map<std::string, FaceParameters> loadFaceParameters(const std::string
                                                   & path) const;
    void loadStimuli(std::map<std::string, FaceParameters>& faceParams,
                     const std::string& dirPath);
    void partitionFace(const std::map<std::string, FaceParameters>& faceParams,
                       const std::string& partitionFile);

    bool mInTheWild;
    bool mWithLandmarks;

};
}

#endif // N2D2_CELEBA_DATABASE_H
