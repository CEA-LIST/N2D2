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

#ifndef N2D2_DATAFILE_H
#define N2D2_DATAFILE_H

#include <memory>

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

#include "utils/Registrar.hpp"

namespace N2D2 {
class DataFile {
public:
    typedef std::function<std::shared_ptr<DataFile>()> RegistryCreate_T;

    static RegistryMap_T& registry()
    {
        static RegistryMap_T rMap;
        return rMap;
    }

    virtual cv::Mat read(const std::string& fileName) = 0;
    virtual cv::Mat readLabel(const std::string& /*fileName*/)
        { return cv::Mat(); }
    virtual void write(const std::string& fileName, const cv::Mat& data) = 0;
    virtual ~DataFile() {};
};
}

#endif // N2D2_DATAFILE_H
