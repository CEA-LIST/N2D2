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

#include "ROI/RectangularROI.hpp"
#include "Transformation/SliceExtractionTransformation.hpp"
#include "utils/UnitTest.hpp"
#include "utils/Utils.hpp"

using namespace N2D2;

TEST_DATASET(SliceExtractionTransformation,
             apply,
             (unsigned int width,
              unsigned int height,
              unsigned int offsetX,
              unsigned int offsetY),
             std::make_tuple(256, 128, 0, 0),
             std::make_tuple(256, 128, 256, 0),
             std::make_tuple(256, 128, 0, 128),
             std::make_tuple(256, 128, 256, 128),
             std::make_tuple(128, 256, 0, 0),
             std::make_tuple(128, 256, 256, 0),
             std::make_tuple(128, 256, 0, 128),
             std::make_tuple(128, 256, 256, 128))
{
    RectangularROI<int> roi1(64, cv::Point(0, 0), 256, 256);
    RectangularROI<int> roi2(128, cv::Point(256, 0), 256, 256);
    RectangularROI<int> roi3(255, cv::Point(256, 256), 256, 256);

    cv::Mat labels(512, 512, CV_32SC1, cv::Scalar(0));
    roi1.append(labels);
    roi2.append(labels);
    roi3.append(labels);

    SliceExtractionTransformation trans(width, height, offsetX, offsetY);

    cv::Mat img = cv::imread("tests_data/Lenna.png",
#if CV_MAJOR_VERSION >= 3
        cv::IMREAD_COLOR);
#else
        CV_LOAD_IMAGE_COLOR);
#endif

    if (!img.data)
        throw std::runtime_error(
            "Could not open or find image: tests_data/Lenna.png");

    trans.apply(img, labels);

    std::ostringstream fileName;
    fileName << "SliceExtractionTransformation_apply(W" << width << "_H"
             << height << "_OX" << offsetX << "_OY" << offsetY
             << ")[frame].png";

    Utils::createDirectories("Transformation");
    if (!cv::imwrite("Transformation/" + fileName.str(), img))
        throw std::runtime_error("Unable to write image: Transformation/"
                                 + fileName.str());

    ASSERT_EQUALS(img.cols, (int)width);
    ASSERT_EQUALS(img.rows, (int)height);

    fileName.str(std::string());
    fileName << "SliceExtractionTransformation_apply(W" << width << "_H"
             << height << "_OX" << offsetX << "_OY" << offsetY
             << ")[labels].png";

    if (!cv::imwrite("Transformation/" + fileName.str(), labels))
        throw std::runtime_error("Unable to write image: Transformation/"
                                 + fileName.str());

    ASSERT_EQUALS(labels.cols, (int)width);
    ASSERT_EQUALS(labels.rows, (int)height);
}

RUN_TESTS()
