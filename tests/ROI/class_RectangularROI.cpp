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
#include "utils/UnitTest.hpp"

using namespace N2D2;

TEST(RectangularROI, RectangularROI)
{
    RectangularROI<int> roi(1, cv::Point(100, 120), cv::Point(150, 180));

    ASSERT_EQUALS(roi.getLabel(), 1);
    ASSERT_EQUALS(roi.points.size(), 4U);
    ASSERT_EQUALS(roi.points[0].x, 100);
    ASSERT_EQUALS(roi.points[0].y, 120);
    ASSERT_EQUALS(roi.points[1].x, 150);
    ASSERT_EQUALS(roi.points[1].y, 120);
    ASSERT_EQUALS(roi.points[2].x, 150);
    ASSERT_EQUALS(roi.points[2].y, 180);
    ASSERT_EQUALS(roi.points[3].x, 100);
    ASSERT_EQUALS(roi.points[3].y, 180);
}

TEST(RectangularROI, RectangularROI__width_height)
{
    RectangularROI<int> roi(1, cv::Point(100, 120), 50, 60);

    ASSERT_EQUALS(roi.getLabel(), 1);
    ASSERT_EQUALS(roi.points.size(), 4U);
    ASSERT_EQUALS(roi.points[0].x, 100);
    ASSERT_EQUALS(roi.points[0].y, 120);
    ASSERT_EQUALS(roi.points[1].x, 150);
    ASSERT_EQUALS(roi.points[1].y, 120);
    ASSERT_EQUALS(roi.points[2].x, 150);
    ASSERT_EQUALS(roi.points[2].y, 180);
    ASSERT_EQUALS(roi.points[3].x, 100);
    ASSERT_EQUALS(roi.points[3].y, 180);
}

TEST(RectangularROI, draw)
{
    RectangularROI<int> roi(1, cv::Point(128, 0), 256, 256);

    cv::Mat img = cv::imread("tests_data/Lenna.png", CV_LOAD_IMAGE_COLOR);

    if (!img.data)
        throw std::runtime_error(
            "Could not open or find image: tests_data/Lenna.png");

    roi.draw(img);

    if (!cv::imwrite("ROI/RectangularROI_draw.png", img))
        throw std::runtime_error(
            "Unable to write image: ROI/RectangularROI_draw.png");
}

TEST(RectangularROI, extract)
{
    RectangularROI<int> roi(1, cv::Point(128, 0), 256, 256);

    cv::Mat img = cv::imread("tests_data/Lenna.png", CV_LOAD_IMAGE_COLOR);

    if (!img.data)
        throw std::runtime_error(
            "Could not open or find image: tests_data/Lenna.png");

    cv::Mat extracted = roi.extract(img);

    if (!cv::imwrite("ROI/RectangularROI_extract.png", extracted))
        throw std::runtime_error(
            "Unable to write image: ROI/RectangularROI_extract.png");

    ASSERT_EQUALS(extracted.rows, 256);
    ASSERT_EQUALS(extracted.cols, 256);
}

TEST(RectangularROI, append)
{
    RectangularROI<int> roi(255, cv::Point(128, 0), 256, 256);

    cv::Mat labels(512, 512, CV_32SC1, cv::Scalar(0));
    roi.append(labels);

    if (!cv::imwrite("ROI/RectangularROI_append.png", labels))
        throw std::runtime_error(
            "Unable to write image: ROI/RectangularROI_append.png");

    ASSERT_EQUALS(labels.at<int>(0, 0), 0);
    ASSERT_EQUALS(labels.at<int>(255, 255), 255);
}

TEST(RectangularROI, append__margin)
{
    const int backgroundLabel = 50;
    RectangularROI<int> roi(255, cv::Point(128, 0), 256, 256);

    cv::Mat labels(512, 512, CV_32SC1, cv::Scalar(backgroundLabel));
    roi.append(labels, 10, backgroundLabel);

    if (!cv::imwrite("ROI/RectangularROI_append__margin.png", labels))
        throw std::runtime_error(
            "Unable to write image: ROI/RectangularROI_append__margin.png");

    ASSERT_EQUALS(labels.at<int>(0, 0), backgroundLabel);
    ASSERT_EQUALS(labels.at<int>(255, 255), 255);
}

TEST(RectangularROI, append__overlap)
{
    const int backgroundLabel = 50;
    RectangularROI<int> roi1(150, cv::Point(128, 0), 256, 256);
    RectangularROI<int> roi2(255, cv::Point(256, 128), 256, 256);

    cv::Mat labels(512, 512, CV_32SC1, cv::Scalar(backgroundLabel));
    roi1.append(labels, 10, backgroundLabel);
    roi2.append(labels, 10, backgroundLabel);

    if (!cv::imwrite("ROI/RectangularROI_append__overlap.png", labels))
        throw std::runtime_error(
            "Unable to write image: ROI/RectangularROI_append__overlap.png");

    ASSERT_EQUALS(labels.at<int>(0, 0), backgroundLabel);
    ASSERT_EQUALS(labels.at<int>(255, 255), 150);
    ASSERT_EQUALS(labels.at<int>(0, 117), backgroundLabel);
    ASSERT_EQUALS(labels.at<int>(0, 118), -1);
    ASSERT_EQUALS(labels.at<int>(0, 127), -1);
    ASSERT_EQUALS(labels.at<int>(0, 128), 150);
}

RUN_TESTS()
