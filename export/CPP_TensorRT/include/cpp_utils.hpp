/*
    (C) Copyright 2017 CEA LIST. All Rights Reserved.
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

#include "typedefs.h"
#include "utils.h"
#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdlib.h>
#include <iterator>
#include <vector>
#include <string>
#include <algorithm> // std::sort
#include <cmath>
#include <cstdlib>
#include <dirent.h>
#include <iomanip>
#include <stdexcept>

#ifdef CV_SUPPORT
#ifdef OPENCV_USE_OLD_HEADERS       //  before OpenCV 2.2.0
    #include "cv.h"
    #include "highgui.h"
#else
    #include "opencv2/core/version.hpp"
    #if CV_MAJOR_VERSION == 2
        #include "opencv2/core/core.hpp"
        #include "opencv2/imgproc/imgproc.hpp"
        #include "opencv2/highgui/highgui.hpp"
    #elif CV_MAJOR_VERSION >= 3
        #include "opencv2/core.hpp"
        #include "opencv2/imgproc.hpp"
        #include "opencv2/highgui.hpp"
    #endif
#endif

#endif

void getFilesList(const std::string dir, std::vector<std::string>& files);

void envRead(const std::string& fileName, unsigned int size,
             unsigned int channelsHeight, unsigned int channelsWidth,
             DATA_T* data, bool noLabels,
             unsigned int outputsSize,
             int32_t* outputTargets);

void confusion_print(unsigned int nbOutputs, unsigned int* confusion);

#ifdef CV_SUPPORT
enum CV_ColorSpace {
    BGR,
    RGB,
    HSV,
    HLS,
    YCrCb,
    CIELab,
    CIELuv
};

cv::Mat cv_read(const std::string& fileName);
void cv_ColorSpaceTransformation(cv::Mat& frame, CV_ColorSpace colorspace);
void cv_DrawBBOX(cv::Mat& frame, DATA_T* ROIs, unsigned int nbBBox, unsigned int stepIdx = 4);
void cv_Display(cv::Mat frame, unsigned int displayW, unsigned int displayH);
void cv_toBuffer(cv::Mat& frame, DATA_T * &input);

#endif
