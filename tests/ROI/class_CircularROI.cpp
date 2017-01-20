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

#include "ROI/CircularROI.hpp"
#include "utils/UnitTest.hpp"

using namespace N2D2;

TEST(CircularROI, CircularROI)
{
    CircularROI<int> roi(1, cv::Point(100, 120), 3);

    ASSERT_EQUALS(roi.getLabel(), 1);
    ASSERT_EQUALS(roi.center.x, 100);
    ASSERT_EQUALS(roi.center.y, 120);
    ASSERT_EQUALS(roi.radius, 3);
}

TEST(CircularROI, draw)
{
    CircularROI<int> roi(1, cv::Point(256, 128), 128);

    cv::Mat img = cv::imread("tests_data/Lenna.png", CV_LOAD_IMAGE_COLOR);

    if (!img.data)
        throw std::runtime_error(
            "Could not open or find image: tests_data/Lenna.png");

    roi.draw(img);

    if (!cv::imwrite("ROI/CircularROI_draw.png", img))
        throw std::runtime_error(
            "Unable to write image: ROI/CircularROI_draw.png");
}

TEST(CircularROI, extract)
{
    CircularROI<int> roi(1, cv::Point(256, 128), 128);

    cv::Mat img = cv::imread("tests_data/Lenna.png", CV_LOAD_IMAGE_COLOR);

    if (!img.data)
        throw std::runtime_error(
            "Could not open or find image: tests_data/Lenna.png");

    cv::Mat extracted = roi.extract(img);

    if (!cv::imwrite("ROI/CircularROI_extract.png", extracted))
        throw std::runtime_error(
            "Unable to write image: ROI/CircularROI_extract.png");

    ASSERT_EQUALS(extracted.rows, 256);
    ASSERT_EQUALS(extracted.cols, 256);
}

TEST(CircularROI, append)
{
    CircularROI<int> roi(255, cv::Point(256, 128), 128);

    cv::Mat labels(512, 512, CV_32SC1, cv::Scalar(0));
    roi.append(labels);

    if (!cv::imwrite("ROI/CircularROI_append.png", labels))
        throw std::runtime_error(
            "Unable to write image: ROI/CircularROI_append.png");

    ASSERT_EQUALS(labels.at<int>(0, 0), 0);
    ASSERT_EQUALS(labels.at<int>(256, 256), 255);
}

RUN_TESTS()
