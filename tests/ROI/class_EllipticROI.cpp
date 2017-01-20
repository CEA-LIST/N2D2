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

#include "ROI/EllipticROI.hpp"
#include "utils/UnitTest.hpp"

using namespace N2D2;

TEST(EllipticROI, EllipticROI)
{
    EllipticROI<int> roi(1, cv::Point(100, 120), 3.0, 2.0, 0.1);

    ASSERT_EQUALS(roi.getLabel(), 1);
    ASSERT_EQUALS(roi.center.x, 100);
    ASSERT_EQUALS(roi.center.y, 120);
    ASSERT_EQUALS(roi.majorRadius, 3.0);
    ASSERT_EQUALS(roi.minorRadius, 2.0);
    ASSERT_EQUALS(roi.angle, 0.1);
}

TEST(EllipticROI, draw)
{
    EllipticROI<int> roi(1, cv::Point(256, 128), 128, 64, M_PI / 4.0);

    cv::Mat img = cv::imread("tests_data/Lenna.png", CV_LOAD_IMAGE_COLOR);

    if (!img.data)
        throw std::runtime_error(
            "Could not open or find image: tests_data/Lenna.png");

    roi.draw(img);

    if (!cv::imwrite("ROI/EllipticROI_draw.png", img))
        throw std::runtime_error(
            "Unable to write image: ROI/EllipticROI_draw.png");
}

TEST(EllipticROI, extract)
{
    EllipticROI<int> roi(1, cv::Point(256, 128), 128, 64, M_PI / 4.0);

    cv::Mat img = cv::imread("tests_data/Lenna.png", CV_LOAD_IMAGE_COLOR);

    if (!img.data)
        throw std::runtime_error(
            "Could not open or find image: tests_data/Lenna.png");

    cv::Mat extracted = roi.extract(img);

    if (!cv::imwrite("ROI/EllipticROI_extract.png", extracted))
        throw std::runtime_error(
            "Unable to write image: ROI/EllipticROI_extract.png");

    ASSERT_EQUALS(extracted.rows, 203);
    ASSERT_EQUALS(extracted.cols, 203);
}

TEST(EllipticROI, append)
{
    EllipticROI<int> roi(255, cv::Point(256, 128), 128, 64, M_PI / 4.0);

    cv::Mat labels(512, 512, CV_32SC1, cv::Scalar(0));
    roi.append(labels);

    if (!cv::imwrite("ROI/EllipticROI_append.png", labels))
        throw std::runtime_error(
            "Unable to write image: ROI/EllipticROI_append.png");

    ASSERT_EQUALS(labels.at<int>(0, 0), 0);
    ASSERT_EQUALS(labels.at<int>(128, 256), 255);
}

TEST_DATASET(EllipticROI,
             rescale,
             (double xRatio,
              double yRatio,
              double angle,
              double majorRadiusRatio,
              double minorRadiusRatio),
             std::make_tuple(0.5, 0.5, 0.0, 0.5, 0.5),
             std::make_tuple(1.0, 0.5, 0.0, 1.0, 0.5),
             std::make_tuple(0.5, 1.0, 0.0, 0.5, 1.0),
             std::make_tuple(0.5, 0.5, M_PI / 2.0, 0.5, 0.5),
             std::make_tuple(1.0, 0.5, M_PI / 2.0, 0.5, 1.0),
             std::make_tuple(0.5, 1.0, M_PI / 2.0, 1.0, 0.5))
{
    EllipticROI<int> roi(255, cv::Point(256, 128), 128, 64, angle);
    roi.rescale(xRatio, yRatio);

    ASSERT_EQUALS(roi.center.x, 256 * xRatio);
    ASSERT_EQUALS(roi.center.y, 128 * yRatio);
    ASSERT_EQUALS(roi.majorRadius, 128 * majorRadiusRatio);
    ASSERT_EQUALS(roi.minorRadius, 64 * minorRadiusRatio);
    ASSERT_EQUALS(roi.angle, std::atan((yRatio / xRatio) * std::tan(angle)));
}

RUN_TESTS()
