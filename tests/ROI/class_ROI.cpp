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

#include "ROI/BitmapROI.hpp"
#include "ROI/CircularROI.hpp"
#include "ROI/EllipticROI.hpp"
#include "ROI/PolygonalROI.hpp"
#include "ROI/RectangularROI.hpp"
#include "utils/UnitTest.hpp"
#include "utils/Utils.hpp"

using namespace N2D2;

////////////////////////////////////////////////////////////////////////////////
// CircularROI
////////////////////////////////////////////////////////////////////////////////
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

    cv::Mat img = cv::imread("tests_data/Lenna.png",
#if CV_MAJOR_VERSION >= 3
        cv::IMREAD_COLOR);
#else
        CV_LOAD_IMAGE_COLOR);
#endif

    if (!img.data)
        throw std::runtime_error(
            "Could not open or find image: tests_data/Lenna.png");

    roi.draw(img);

    Utils::createDirectories("ROI");
    if (!cv::imwrite("ROI/CircularROI_draw.png", img))
        throw std::runtime_error(
            "Unable to write image: ROI/CircularROI_draw.png");
}

TEST(CircularROI, extract)
{
    CircularROI<int> roi(1, cv::Point(256, 128), 128);

    cv::Mat img = cv::imread("tests_data/Lenna.png",
#if CV_MAJOR_VERSION >= 3
        cv::IMREAD_COLOR);
#else
        CV_LOAD_IMAGE_COLOR);
#endif

    if (!img.data)
        throw std::runtime_error(
            "Could not open or find image: tests_data/Lenna.png");

    cv::Mat extracted = roi.extract(img);

    Utils::createDirectories("ROI");
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

    Utils::createDirectories("ROI");
    if (!cv::imwrite("ROI/CircularROI_append.png", labels))
        throw std::runtime_error(
            "Unable to write image: ROI/CircularROI_append.png");

    ASSERT_EQUALS(labels.at<int>(0, 0), 0);
    ASSERT_EQUALS(labels.at<int>(256, 256), 255);
}

////////////////////////////////////////////////////////////////////////////////
// EllipticROI
////////////////////////////////////////////////////////////////////////////////
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

    cv::Mat img = cv::imread("tests_data/Lenna.png",
#if CV_MAJOR_VERSION >= 3
        cv::IMREAD_COLOR);
#else
        CV_LOAD_IMAGE_COLOR);
#endif

    if (!img.data)
        throw std::runtime_error(
            "Could not open or find image: tests_data/Lenna.png");

    roi.draw(img);

    Utils::createDirectories("ROI");
    if (!cv::imwrite("ROI/EllipticROI_draw.png", img))
        throw std::runtime_error(
            "Unable to write image: ROI/EllipticROI_draw.png");
}

TEST(EllipticROI, extract)
{
    EllipticROI<int> roi(1, cv::Point(256, 128), 128, 64, M_PI / 4.0);

    cv::Mat img = cv::imread("tests_data/Lenna.png",
#if CV_MAJOR_VERSION >= 3
        cv::IMREAD_COLOR);
#else
        CV_LOAD_IMAGE_COLOR);
#endif

    if (!img.data)
        throw std::runtime_error(
            "Could not open or find image: tests_data/Lenna.png");

    cv::Mat extracted = roi.extract(img);

    Utils::createDirectories("ROI");
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

    Utils::createDirectories("ROI");
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

////////////////////////////////////////////////////////////////////////////////
// PolygonalROI
////////////////////////////////////////////////////////////////////////////////
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

    cv::Mat img = cv::imread("tests_data/Lenna.png",
#if CV_MAJOR_VERSION >= 3
        cv::IMREAD_COLOR);
#else
        CV_LOAD_IMAGE_COLOR);
#endif

    if (!img.data)
        throw std::runtime_error(
            "Could not open or find image: tests_data/Lenna.png");

    roi.draw(img);

    Utils::createDirectories("ROI");
    if (!cv::imwrite("ROI/PolygonalROI_draw.png", img))
        throw std::runtime_error(
            "Unable to write image: ROI/PolygonalROI_draw.png");
}

TEST(PolygonalROI, extract)
{
    // Polygonal points are inclusive
    std::vector<cv::Point> pts;
    pts.push_back(cv::Point(256, 0));
    pts.push_back(cv::Point(383, 255));
    pts.push_back(cv::Point(128, 255));

    PolygonalROI<int> roi(1, pts);

    cv::Mat img = cv::imread("tests_data/Lenna.png",
#if CV_MAJOR_VERSION >= 3
        cv::IMREAD_COLOR);
#else
        CV_LOAD_IMAGE_COLOR);
#endif

    if (!img.data)
        throw std::runtime_error(
            "Could not open or find image: tests_data/Lenna.png");

    cv::Mat extracted = roi.extract(img);

    Utils::createDirectories("ROI");
    if (!cv::imwrite("ROI/PolygonalROI_extract.png", extracted))
        throw std::runtime_error(
            "Unable to write image: ROI/PolygonalROI_extract.png");

    ASSERT_EQUALS(extracted.rows, 256);
    ASSERT_EQUALS(extracted.cols, 256);
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

    Utils::createDirectories("ROI");
    if (!cv::imwrite("ROI/PolygonalROI_append.png", labels))
        throw std::runtime_error(
            "Unable to write image: ROI/PolygonalROI_append.png");

    ASSERT_EQUALS(labels.at<int>(0, 0), 0);
    ASSERT_EQUALS(labels.at<int>(255, 255), 255);
}

////////////////////////////////////////////////////////////////////////////////
// RectangularROI
////////////////////////////////////////////////////////////////////////////////
TEST(RectangularROI, RectangularROI)
{
    // Constructor BR point is exclusive
    RectangularROI<int> roi(1, cv::Point(100, 120), cv::Point(150, 180));

    ASSERT_EQUALS(roi.getLabel(), 1);
    ASSERT_EQUALS(roi.points.size(), 4U);
    ASSERT_EQUALS(roi.points[0].x, 100);
    ASSERT_EQUALS(roi.points[0].y, 120);
    ASSERT_EQUALS(roi.points[1].x, 149);  // internal point is inclusive
    ASSERT_EQUALS(roi.points[1].y, 120);
    ASSERT_EQUALS(roi.points[2].x, 149);  // internal point is inclusive
    ASSERT_EQUALS(roi.points[2].y, 179);  // internal point is inclusive
    ASSERT_EQUALS(roi.points[3].x, 100);
    ASSERT_EQUALS(roi.points[3].y, 179);  // internal point is inclusive

    const cv::Rect rect = roi.getBoundingRect();

    ASSERT_EQUALS(rect.x, 100);
    ASSERT_EQUALS(rect.y, 120);
    ASSERT_EQUALS(rect.width, 50);
    ASSERT_EQUALS(rect.height, 60);
}

TEST(RectangularROI, RectangularROI__width_height)
{
    RectangularROI<int> roi(1, cv::Point(100, 120), 50, 60);

    ASSERT_EQUALS(roi.getLabel(), 1);
    ASSERT_EQUALS(roi.points.size(), 4U);
    ASSERT_EQUALS(roi.points[0].x, 100);
    ASSERT_EQUALS(roi.points[0].y, 120);
    ASSERT_EQUALS(roi.points[1].x, 149);  // internal point is inclusive
    ASSERT_EQUALS(roi.points[1].y, 120);
    ASSERT_EQUALS(roi.points[2].x, 149);  // internal point is inclusive
    ASSERT_EQUALS(roi.points[2].y, 179);  // internal point is inclusive
    ASSERT_EQUALS(roi.points[3].x, 100);
    ASSERT_EQUALS(roi.points[3].y, 179);  // internal point is inclusive

    const cv::Rect rect = roi.getBoundingRect();

    ASSERT_EQUALS(rect.x, 100);
    ASSERT_EQUALS(rect.y, 120);
    ASSERT_EQUALS(rect.width, 50);
    ASSERT_EQUALS(rect.height, 60);
}

TEST(RectangularROI, draw)
{
    RectangularROI<int> roi(1, cv::Point(128, 0), 256, 256);

    cv::Mat img = cv::imread("tests_data/Lenna.png",
#if CV_MAJOR_VERSION >= 3
        cv::IMREAD_COLOR);
#else
        CV_LOAD_IMAGE_COLOR);
#endif

    if (!img.data)
        throw std::runtime_error(
            "Could not open or find image: tests_data/Lenna.png");

    roi.draw(img);

    Utils::createDirectories("ROI");
    if (!cv::imwrite("ROI/RectangularROI_draw.png", img))
        throw std::runtime_error(
            "Unable to write image: ROI/RectangularROI_draw.png");
}

TEST(RectangularROI, extract)
{
    RectangularROI<int> roi(1, cv::Point(128, 0), 256, 256);

    cv::Mat img = cv::imread("tests_data/Lenna.png",
#if CV_MAJOR_VERSION >= 3
        cv::IMREAD_COLOR);
#else
        CV_LOAD_IMAGE_COLOR);
#endif

    if (!img.data)
        throw std::runtime_error(
            "Could not open or find image: tests_data/Lenna.png");

    cv::Mat extracted = roi.extract(img);

    Utils::createDirectories("ROI");
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

    Utils::createDirectories("ROI");
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

    Utils::createDirectories("ROI");
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

    Utils::createDirectories("ROI");
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

////////////////////////////////////////////////////////////////////////////////
// BitmapROI
////////////////////////////////////////////////////////////////////////////////
TEST_DATASET(BitmapROI,
             BitmapROI,
             (cv::Point origin, int scale, cv::Size size),
             std::make_tuple(cv::Point(0, 0), 1, cv::Size(40, 40)),
             std::make_tuple(cv::Point(0, 0), 4, cv::Size(40, 40)),
             std::make_tuple(cv::Point(50, 100), 1, cv::Size(40, 40)),
             std::make_tuple(cv::Point(50, 100), 4, cv::Size(40, 40)),
             std::make_tuple(cv::Point(472, 472), 1, cv::Size(40, 40)),
             std::make_tuple(cv::Point(352, 352), 4, cv::Size(40, 40)),
             std::make_tuple(cv::Point(0, 0), 1, cv::Size(40, 80)),
             std::make_tuple(cv::Point(0, 0), 4, cv::Size(40, 80)),
             std::make_tuple(cv::Point(50, 100), 1, cv::Size(40, 80)),
             std::make_tuple(cv::Point(50, 100), 4, cv::Size(40, 80)),
             std::make_tuple(cv::Point(472, 432), 1, cv::Size(40, 80)),
             std::make_tuple(cv::Point(352, 192), 4, cv::Size(40, 80)))
{
    cv::Mat data(size, CV_8UC1);
    data(cv::Rect(0, 0, size.width / 2, size.height / 4)) = 1;
    data(cv::Rect(size.width / 3, size.height / 4, size.width / 3, 3 * size.height / 4)) = 1;
    data(cv::Rect(2 * size.width / 3, 7 * size.height / 8, size.width / 3, size.height / 8)) = 1;

    BitmapROI<int> roi(1, origin, scale, data);
    ASSERT_EQUALS(roi.getLabel(), 1);

    const cv::Rect rect = roi.getBoundingRect();
    ASSERT_EQUALS(rect.x, origin.x);
    ASSERT_EQUALS(rect.y, origin.y);
    ASSERT_EQUALS(rect.width, size.width * scale);
    ASSERT_EQUALS(rect.height, size.height * scale);
}

TEST_DATASET(BitmapROI,
             draw,
             (cv::Point origin, int scale, cv::Size size),
             std::make_tuple(cv::Point(0, 0), 1, cv::Size(40, 40)),
             std::make_tuple(cv::Point(0, 0), 4, cv::Size(40, 40)),
             std::make_tuple(cv::Point(50, 100), 1, cv::Size(40, 40)),
             std::make_tuple(cv::Point(50, 100), 4, cv::Size(40, 40)),
             std::make_tuple(cv::Point(472, 472), 1, cv::Size(40, 40)),
             std::make_tuple(cv::Point(352, 352), 4, cv::Size(40, 40)),
             std::make_tuple(cv::Point(0, 0), 1, cv::Size(40, 80)),
             std::make_tuple(cv::Point(0, 0), 4, cv::Size(40, 80)),
             std::make_tuple(cv::Point(50, 100), 1, cv::Size(40, 80)),
             std::make_tuple(cv::Point(50, 100), 4, cv::Size(40, 80)),
             std::make_tuple(cv::Point(472, 432), 1, cv::Size(40, 80)),
             std::make_tuple(cv::Point(352, 192), 4, cv::Size(40, 80)))
{
    cv::Mat data = cv::Mat::zeros(size, CV_8UC1);
    data(cv::Rect(0, 0, size.width / 2, size.height / 4)) = 1;
    data(cv::Rect(size.width / 3, size.height / 4, size.width / 3, 3 * size.height / 4)) = 1;
    data(cv::Rect(2 * size.width / 3, 7 * size.height / 8, size.width / 3, size.height / 8)) = 1;

    Utils::createDirectories("ROI");

    if (!cv::imwrite("ROI/BitmapROI_pattern.png", 255 * data))
        throw std::runtime_error(
            "Unable to write image: ROI/BitmapROI_pattern.png");

    BitmapROI<int> roi(1, origin, scale, data);

    cv::Mat img = cv::imread("tests_data/Lenna.png",
#if CV_MAJOR_VERSION >= 3
        cv::IMREAD_COLOR);
#else
        CV_LOAD_IMAGE_COLOR);
#endif

    if (!img.data)
        throw std::runtime_error(
            "Could not open or find image: tests_data/Lenna.png");

    roi.draw(img);

    std::ostringstream fileName;
    fileName << "ROI/BitmapROI_draw"
        << "_O" << origin.x << "x" << origin.y
        << "_S" << scale
        << "_" << size.width << "x" << size.height
        << ".png";

    if (!cv::imwrite(fileName.str().c_str(), img))
        throw std::runtime_error(
            "Unable to write image: " + fileName.str());
}

TEST_DATASET(BitmapROI,
             append,
             (cv::Point origin, int scale, cv::Size size),
             std::make_tuple(cv::Point(0, 0), 1, cv::Size(40, 40)),
             std::make_tuple(cv::Point(0, 0), 4, cv::Size(40, 40)),
             std::make_tuple(cv::Point(50, 100), 1, cv::Size(40, 40)),
             std::make_tuple(cv::Point(50, 100), 4, cv::Size(40, 40)),
             std::make_tuple(cv::Point(472, 472), 1, cv::Size(40, 40)),
             std::make_tuple(cv::Point(352, 352), 4, cv::Size(40, 40)),
             std::make_tuple(cv::Point(0, 0), 1, cv::Size(40, 80)),
             std::make_tuple(cv::Point(0, 0), 4, cv::Size(40, 80)),
             std::make_tuple(cv::Point(50, 100), 1, cv::Size(40, 80)),
             std::make_tuple(cv::Point(50, 100), 4, cv::Size(40, 80)),
             std::make_tuple(cv::Point(472, 432), 1, cv::Size(40, 80)),
             std::make_tuple(cv::Point(352, 192), 4, cv::Size(40, 80)))
{
    cv::Mat data = cv::Mat::zeros(size, CV_8UC1);
    data(cv::Rect(0, 0, size.width / 2, size.height / 4)) = 1;
    data(cv::Rect(size.width / 3, size.height / 4, size.width / 3, 3 * size.height / 4)) = 1;
    data(cv::Rect(2 * size.width / 3, 7 * size.height / 8, size.width / 3, size.height / 8)) = 1;

    // No transform.
    BitmapROI<int> roi(255, origin, scale, data);

    cv::Mat labels(512, 512, CV_32SC1, cv::Scalar(0));
    roi.append(labels);

    Utils::createDirectories("ROI");

    std::ostringstream fileName;
    fileName << "ROI/BitmapROI_append"
        << "_O" << origin.x << "x" << origin.y
        << "_S" << scale
        << "_" << size.width << "x" << size.height
        << ".png";

    if (!cv::imwrite(fileName.str().c_str(), labels))
        throw std::runtime_error(
            "Unable to write image: " + fileName.str());

    // Rescale
    BitmapROI<int> roiRescale(255, origin, scale, data);
    roiRescale.rescale(0.5, 2.0);

    cv::Mat labelsRescale(512, 512, CV_32SC1, cv::Scalar(0));
    roiRescale.append(labelsRescale);

    fileName.str(std::string());
    fileName << "ROI/BitmapROI_append"
        << "_O" << origin.x << "x" << origin.y
        << "_S" << scale
        << "_" << size.width << "x" << size.height
        << "_rescale.png";

    if (!cv::imwrite(fileName.str().c_str(), labelsRescale))
        throw std::runtime_error(
            "Unable to write image: " + fileName.str());

    // PadCrop
    BitmapROI<int> roiPadCrop(255, origin, scale, data);
    roiPadCrop.padCrop(50, 50, 100, 100);
    roiPadCrop.padCrop(-50, -50, 512, 512);

    cv::Mat labelsPadCrop(512, 512, CV_32SC1, cv::Scalar(0));
    roiPadCrop.append(labelsPadCrop);

    // Debug rectangle
    cv::rectangle(labelsPadCrop,
                  cv::Rect(50, 50, 100, 100),
                  cv::Scalar(128));

    fileName.str(std::string());
    fileName << "ROI/BitmapROI_append"
        << "_O" << origin.x << "x" << origin.y
        << "_S" << scale
        << "_" << size.width << "x" << size.height
        << "_padcrop.png";

    if (!cv::imwrite(fileName.str().c_str(), labelsPadCrop))
        throw std::runtime_error(
            "Unable to write image: " + fileName.str());

    // Flip
    BitmapROI<int> roiFlip(255, origin, scale, data);
    roiFlip.flip(512, 512, true, true);

    cv::Mat labelsFlip(512, 512, CV_32SC1, cv::Scalar(0));
    roiFlip.append(labelsFlip);

    fileName.str(std::string());
    fileName << "ROI/BitmapROI_append"
        << "_O" << origin.x << "x" << origin.y
        << "_S" << scale
        << "_" << size.width << "x" << size.height
        << "_flip.png";

    if (!cv::imwrite(fileName.str().c_str(), labelsFlip))
        throw std::runtime_error(
            "Unable to write image: " + fileName.str());
}

TEST_DATASET(BitmapROI,
             append__margin,
             (cv::Point origin, int scale, cv::Size size),
             std::make_tuple(cv::Point(0, 0), 1, cv::Size(40, 40)),
             std::make_tuple(cv::Point(0, 0), 4, cv::Size(40, 40)),
             std::make_tuple(cv::Point(50, 100), 1, cv::Size(40, 40)),
             std::make_tuple(cv::Point(50, 100), 4, cv::Size(40, 40)),
             std::make_tuple(cv::Point(472, 472), 1, cv::Size(40, 40)),
             std::make_tuple(cv::Point(352, 352), 4, cv::Size(40, 40)),
             std::make_tuple(cv::Point(0, 0), 1, cv::Size(40, 80)),
             std::make_tuple(cv::Point(0, 0), 4, cv::Size(40, 80)),
             std::make_tuple(cv::Point(50, 100), 1, cv::Size(40, 80)),
             std::make_tuple(cv::Point(50, 100), 4, cv::Size(40, 80)),
             std::make_tuple(cv::Point(472, 432), 1, cv::Size(40, 80)),
             std::make_tuple(cv::Point(352, 192), 4, cv::Size(40, 80)))
{
    const int backgroundLabel = 50;

    cv::Mat data = cv::Mat::zeros(size, CV_8UC1);
    data(cv::Rect(0, 0, size.width / 2, size.height / 4)) = 1;
    data(cv::Rect(size.width / 3, size.height / 4, size.width / 3, 3 * size.height / 4)) = 1;
    data(cv::Rect(2 * size.width / 3, 7 * size.height / 8, size.width / 3, size.height / 8)) = 1;

    BitmapROI<int> roi(255, origin, scale, data);

    cv::Mat labels(512, 512, CV_32SC1, cv::Scalar(backgroundLabel));
    roi.append(labels, 10, backgroundLabel);

    Utils::createDirectories("ROI");

    std::ostringstream fileName;
    fileName << "ROI/BitmapROI_append__margin"
        << "_O" << origin.x << "x" << origin.y
        << "_S" << scale
        << "_" << size.width << "x" << size.height
        << ".png";

    if (!cv::imwrite(fileName.str().c_str(), labels))
        throw std::runtime_error(
            "Unable to write image: " + fileName.str());
}

TEST_DATASET(BitmapROI,
             append__overlap,
             (cv::Point origin1, int scale1, cv::Size size1,
              cv::Point origin2, int scale2, cv::Size size2),
             std::make_tuple(cv::Point(0, 0), 1, cv::Size(40, 40),
                             cv::Point(352, 192), 4, cv::Size(40, 80)),
             std::make_tuple(cv::Point(0, 0), 4, cv::Size(40, 40),
                             cv::Point(352, 192), 4, cv::Size(40, 80)),
             std::make_tuple(cv::Point(50, 100), 1, cv::Size(40, 40),
                             cv::Point(352, 192), 4, cv::Size(40, 80)),
             std::make_tuple(cv::Point(50, 100), 4, cv::Size(40, 40),
                             cv::Point(352, 192), 4, cv::Size(40, 80)),
             std::make_tuple(cv::Point(472, 472), 1, cv::Size(40, 40),
                             cv::Point(352, 192), 4, cv::Size(40, 80)),
             std::make_tuple(cv::Point(352, 352), 4, cv::Size(40, 40),
                             cv::Point(352, 192), 4, cv::Size(40, 80)),
             std::make_tuple(cv::Point(0, 0), 1, cv::Size(40, 80),
                             cv::Point(352, 192), 4, cv::Size(40, 80)),
             std::make_tuple(cv::Point(0, 0), 4, cv::Size(40, 80),
                             cv::Point(352, 192), 4, cv::Size(40, 80)),
             std::make_tuple(cv::Point(50, 100), 1, cv::Size(40, 80),
                             cv::Point(352, 192), 4, cv::Size(40, 80)),
             std::make_tuple(cv::Point(50, 100), 4, cv::Size(40, 80),
                             cv::Point(352, 192), 4, cv::Size(40, 80)),
             std::make_tuple(cv::Point(472, 432), 1, cv::Size(40, 80),
                             cv::Point(352, 192), 4, cv::Size(40, 80)),
             std::make_tuple(cv::Point(352, 192), 4, cv::Size(40, 80),
                             cv::Point(352, 192), 4, cv::Size(40, 80)))
{
    const int backgroundLabel = 50;

    cv::Mat data1 = cv::Mat::zeros(size1, CV_8UC1);
    data1(cv::Rect(0, 0, size1.width / 2, size1.height / 4)) = 1;
    data1(cv::Rect(size1.width / 3, size1.height / 4, size1.width / 3, 3 * size1.height / 4)) = 1;
    data1(cv::Rect(2 * size1.width / 3, 7 * size1.height / 8, size1.width / 3, size1.height / 8)) = 1;

    cv::Mat data2 = cv::Mat::zeros(size2, CV_8UC1);
    data2(cv::Rect(0, 0, size2.width / 2, size2.height / 4)) = 1;
    data2(cv::Rect(size2.width / 3, size2.height / 4, size2.width / 3, 3 * size2.height / 4)) = 1;
    data2(cv::Rect(2 * size2.width / 3, 7 * size2.height / 8, size2.width / 3, size2.height / 8)) = 1;

    BitmapROI<int> roi1(255, origin1, scale1, data1);
    BitmapROI<int> roi2(150, origin2, scale2, data2);

    cv::Mat labels(512, 512, CV_32SC1, cv::Scalar(backgroundLabel));
    roi2.append(labels, 10, backgroundLabel);
    roi1.append(labels, 10, backgroundLabel);

    Utils::createDirectories("ROI");

    std::ostringstream fileName;
    fileName << "ROI/BitmapROI_append__overlap"
        << "_O" << origin1.x << "x" << origin1.y
        << "_S" << scale1
        << "_" << size1.width << "x" << size1.height
        << ".png";

    if (!cv::imwrite(fileName.str().c_str(), labels))
        throw std::runtime_error(
            "Unable to write image: " + fileName.str());
}

RUN_TESTS()
