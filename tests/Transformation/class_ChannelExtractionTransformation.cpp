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
#include "Transformation/ChannelExtractionTransformation.hpp"
#include "utils/UnitTest.hpp"

using namespace N2D2;

TEST_DATASET(ChannelExtractionTransformation,
             apply,
             (ChannelExtractionTransformation::Channel channel),
             std::make_tuple(ChannelExtractionTransformation::Red),
             std::make_tuple(ChannelExtractionTransformation::Green),
             std::make_tuple(ChannelExtractionTransformation::Blue),
             std::make_tuple(ChannelExtractionTransformation::Hue),
             std::make_tuple(ChannelExtractionTransformation::Saturation),
             std::make_tuple(ChannelExtractionTransformation::Value),
             std::make_tuple(ChannelExtractionTransformation::Gray),
             std::make_tuple(ChannelExtractionTransformation::Y),
             std::make_tuple(ChannelExtractionTransformation::Cb),
             std::make_tuple(ChannelExtractionTransformation::Cr))
{
    RectangularROI<int> roi1(64, cv::Point(0, 0), 255, 255);
    RectangularROI<int> roi2(128, cv::Point(256, 0), 255, 255);
    RectangularROI<int> roi3(255, cv::Point(256, 256), 255, 255);

    cv::Mat labels(512, 512, CV_32SC1, cv::Scalar(0));
    roi1.append(labels);
    roi2.append(labels);
    roi3.append(labels);

    ChannelExtractionTransformation trans(channel);

    cv::Mat img = cv::imread("tests_data/Lenna.png", CV_LOAD_IMAGE_COLOR);

    if (!img.data)
        throw std::runtime_error(
            "Could not open or find image: tests_data/Lenna.png");

    const int cols = img.cols;
    const int rows = img.rows;

    trans.apply(img, labels);

    std::ostringstream fileName;
    fileName << "ChannelExtractionTransformation_apply(C" << channel
             << ")[frame].png";

    if (!cv::imwrite("Transformation/" + fileName.str(), img))
        throw std::runtime_error("Unable to write image: Transformation/"
                                 + fileName.str());

    ASSERT_EQUALS(img.cols, cols);
    ASSERT_EQUALS(img.rows, rows);
    ASSERT_EQUALS(img.channels(), 1);

    fileName.str(std::string());
    fileName << "ChannelExtractionTransformation_apply(C" << channel
             << ")[labels].png";

    if (!cv::imwrite("Transformation/" + fileName.str(), labels))
        throw std::runtime_error("Unable to write image: Transformation/"
                                 + fileName.str());
}

RUN_TESTS()
