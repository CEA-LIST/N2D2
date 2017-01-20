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

#include "ROI/PolygonalROI.hpp"
#include "utils/UnitTest.hpp"

using namespace N2D2;

TEST(PolygonalROI, PolygonalROI)
{
    std::vector<cv::Point> pts;
    pts.push_back(cv::Point(100, 120));
    pts.push_back(cv::Point(101, 121));
    pts.push_back(cv::Point(102, 122));

    PolygonalROI<int> roi(1, pts);

    ASSERT_EQUALS(roi.getLabel(), 1);
    ASSERT_EQUALS(roi.points.size(), 3U);
    ASSERT_EQUALS(roi.points[0].x, 100);
    ASSERT_EQUALS(roi.points[0].y, 120);
    ASSERT_EQUALS(roi.points[1].x, 101);
    ASSERT_EQUALS(roi.points[1].y, 121);
    ASSERT_EQUALS(roi.points[2].x, 102);
    ASSERT_EQUALS(roi.points[2].y, 122);
}

TEST(PolygonalROI, draw)
{
    std::vector<cv::Point> pts;
    pts.push_back(cv::Point(256, 0));
    pts.push_back(cv::Point(383, 255));
    pts.push_back(cv::Point(128, 255));

    PolygonalROI<int> roi(1, pts);

    cv::Mat img = cv::imread("tests_data/Lenna.png", CV_LOAD_IMAGE_COLOR);

    if (!img.data)
        throw std::runtime_error(
            "Could not open or find image: tests_data/Lenna.png");

    roi.draw(img);

    if (!cv::imwrite("ROI/PolygonalROI_draw.png", img))
        throw std::runtime_error(
            "Unable to write image: ROI/PolygonalROI_draw.png");
}

TEST(PolygonalROI, extract)
{
    std::vector<cv::Point> pts;
    pts.push_back(cv::Point(256, 0));
    pts.push_back(cv::Point(383, 255));
    pts.push_back(cv::Point(128, 255));

    PolygonalROI<int> roi(1, pts);

    cv::Mat img = cv::imread("tests_data/Lenna.png", CV_LOAD_IMAGE_COLOR);

    if (!img.data)
        throw std::runtime_error(
            "Could not open or find image: tests_data/Lenna.png");

    cv::Mat extracted = roi.extract(img);

    if (!cv::imwrite("ROI/PolygonalROI_extract.png", extracted))
        throw std::runtime_error(
            "Unable to write image: ROI/PolygonalROI_extract.png");

    ASSERT_EQUALS(extracted.rows, 255);
    ASSERT_EQUALS(extracted.cols, 255);
}

TEST(PolygonalROI, append)
{
    std::vector<cv::Point> pts;
    pts.push_back(cv::Point(256, 0));
    pts.push_back(cv::Point(383, 255));
    pts.push_back(cv::Point(128, 255));

    PolygonalROI<int> roi(255, pts);

    cv::Mat labels(512, 512, CV_32SC1, cv::Scalar(0));
    roi.append(labels);

    if (!cv::imwrite("ROI/PolygonalROI_append.png", labels))
        throw std::runtime_error(
            "Unable to write image: ROI/PolygonalROI_append.png");

    ASSERT_EQUALS(labels.at<int>(0, 0), 0);
    ASSERT_EQUALS(labels.at<int>(255, 255), 255);
}

RUN_TESTS()
