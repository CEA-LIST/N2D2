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
#include "Transformation/LabelSliceExtractionTransformation.hpp"
#include "utils/UnitTest.hpp"
#include "utils/Utils.hpp"

using namespace N2D2;

TEST_DATASET(LabelSliceExtractionTransformation,
             apply,
             (unsigned int width, unsigned int height),
             std::make_tuple(128, 64),
             std::make_tuple(64, 128),
             std::make_tuple(64, 64),
             std::make_tuple(127, 63),
             std::make_tuple(63, 127),
             std::make_tuple(63, 63))
{
    Random::mtSeed(0);

    RectangularROI<int> roi1(64, cv::Point(0, 0), 256, 256);
    RectangularROI<int> roi2(128, cv::Point(256, 0), 256, 256);
    RectangularROI<int> roi3(255, cv::Point(256, 256), 256, 256);

    cv::Mat labels(512, 512, CV_32SC1, cv::Scalar(0));
    roi1.append(labels);
    roi2.append(labels);
    roi3.append(labels);

    LabelSliceExtractionTransformation trans(width, height);

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
    fileName << "LabelSliceExtractionTransformation_apply(W" << width << "_H"
             << height << ")[frame].png";

    Utils::createDirectories("Transformation");
    if (!cv::imwrite("Transformation/" + fileName.str(), img))
        throw std::runtime_error("Unable to write image: Transformation/"
                                 + fileName.str());

    fileName.str(std::string());
    fileName << "LabelSliceExtractionTransformation_apply(W" << width << "_H"
             << height << ")[labels].png";

    ASSERT_EQUALS(labels.rows, 1);
    ASSERT_EQUALS(labels.cols, 1);
    ASSERT_EQUALS(labels.at<int>(0, 0), trans.getLastLabel());

    if (!cv::imwrite("Transformation/" + fileName.str(), labels))
        throw std::runtime_error("Unable to write image: Transformation/"
                                 + fileName.str());
}

TEST_DATASET(LabelSliceExtractionTransformation,
             apply__bis,
             (unsigned int width, unsigned int height, int slicesMargin),
             std::make_tuple(128, 64, 0),
             std::make_tuple(64, 128, 0),
             std::make_tuple(64, 64, 0),
             std::make_tuple(127, 63, 0),
             std::make_tuple(63, 127, 0),
             std::make_tuple(63, 63, 0),
             std::make_tuple(128, 64, 5),
             std::make_tuple(64, 128, 5),
             std::make_tuple(64, 64, 5),
             std::make_tuple(127, 63, 10),
             std::make_tuple(63, 127, 10),
             std::make_tuple(63, 63, 10),
             /*std::make_tuple(127, 63, -60),
             std::make_tuple(63, 127, -60),
             std::make_tuple(63, 63, -60)*/)
{
    Random::mtSeed(0);

    RectangularROI<int> roi1(255, cv::Point(150, 150), 150, 150);

    cv::Mat labels(512, 512, CV_32SC1, cv::Scalar(0));
    roi1.append(labels);

    LabelSliceExtractionTransformation trans(width, height, 0);
    trans.setParameter("SlicesMargin", slicesMargin);

    const cv::Mat img = cv::imread("tests_data/Lenna.png",
#if CV_MAJOR_VERSION >= 3
        cv::IMREAD_COLOR);
#else
        CV_LOAD_IMAGE_COLOR);
#endif

    if (!img.data)
        throw std::runtime_error(
            "Could not open or find image: tests_data/Lenna.png");

    cv::Mat slices = ((cv::Mat)labels).clone();

    for (unsigned int i = 0; i < 100; ++i) {
        cv::Mat imgCopy = img.clone();
        cv::Mat labelsCopy = labels.clone();
        trans.apply(imgCopy, labelsCopy);

        ASSERT_EQUALS(imgCopy.rows, (int)height);
        ASSERT_EQUALS(imgCopy.cols, (int)width);
        ASSERT_EQUALS(labelsCopy.rows, 1);
        ASSERT_EQUALS(labelsCopy.cols, 1);

        cv::rectangle(slices,
                      trans.getLastSlice().tl(),
                      trans.getLastSlice().br(),
                      cv::Scalar(128));
    }

    std::ostringstream fileName;
    fileName << "LabelSliceExtractionTransformation_apply__bis(W" << width
             << "_H" << height << "_S" << slicesMargin << ").png";

    Utils::createDirectories("Transformation");
    if (!cv::imwrite("Transformation/" + fileName.str(), slices))
        throw std::runtime_error("Unable to write image: Transformation/"
                                 + fileName.str());
}

TEST_DATASET(LabelSliceExtractionTransformation,
             apply__ter,
             (unsigned int width, unsigned int height, int slicesMargin),
             std::make_tuple(128, 64, 0),
             std::make_tuple(64, 128, 0),
             std::make_tuple(64, 64, 0),
             std::make_tuple(127, 63, 0),
             std::make_tuple(63, 127, 0),
             std::make_tuple(63, 63, 0),
             std::make_tuple(128, 64, 5),
             std::make_tuple(64, 128, 5),
             std::make_tuple(64, 64, 5),
             std::make_tuple(127, 63, 10),
             std::make_tuple(63, 127, 10),
             std::make_tuple(63, 63, 10))
{
    Random::mtSeed(0);

    RectangularROI<int> roi1(255, cv::Point(150, 150), 150, 150);
    RectangularROI<int> roi2(255, cv::Point(350, 350), 20, 20);

    cv::Mat labels(512, 512, CV_32SC1, cv::Scalar(0));
    roi1.append(labels);
    roi2.append(labels);

    LabelSliceExtractionTransformation trans(width, height, 0);
    trans.setParameter("SlicesMargin", slicesMargin);

    const cv::Mat img = cv::imread("tests_data/Lenna.png",
#if CV_MAJOR_VERSION >= 3
        cv::IMREAD_COLOR);
#else
        CV_LOAD_IMAGE_COLOR);
#endif

    if (!img.data)
        throw std::runtime_error(
            "Could not open or find image: tests_data/Lenna.png");

    cv::Mat slices = ((cv::Mat)labels).clone();

    for (unsigned int i = 0; i < 100; ++i) {
        cv::Mat imgCopy = img.clone();
        cv::Mat labelsCopy = labels.clone();
        trans.apply(imgCopy, labelsCopy);

        ASSERT_EQUALS(imgCopy.rows, (int)height);
        ASSERT_EQUALS(imgCopy.cols, (int)width);
        ASSERT_EQUALS(labelsCopy.rows, 1);
        ASSERT_EQUALS(labelsCopy.cols, 1);

        cv::rectangle(slices,
                      trans.getLastSlice().tl(),
                      trans.getLastSlice().br(),
                      cv::Scalar(128));
    }

    std::ostringstream fileName;
    fileName << "LabelSliceExtractionTransformation_apply__ter(W" << width
             << "_H" << height << "_S" << slicesMargin << ").png";

    Utils::createDirectories("Transformation");
    if (!cv::imwrite("Transformation/" + fileName.str(), slices))
        throw std::runtime_error("Unable to write image: Transformation/"
                                 + fileName.str());
}

TEST_DATASET(LabelSliceExtractionTransformation,
             apply__quater,
             (unsigned int width, unsigned int height, int slicesMargin),
             std::make_tuple(32, 64, 0),
             std::make_tuple(64, 32, 0),
             std::make_tuple(64, 64, 0),
             std::make_tuple(31, 63, 0),
             std::make_tuple(63, 31, 0),
             std::make_tuple(63, 63, 0),
             std::make_tuple(32, 64, 5),
             std::make_tuple(64, 32, 5),
             std::make_tuple(64, 64, 5),
             std::make_tuple(31, 63, 10),
             std::make_tuple(63, 31, 10),
             std::make_tuple(63, 63, 10),
             /*std::make_tuple(31, 63, -60),
             std::make_tuple(63, 31, -60),
             std::make_tuple(63, 63, -60)*/)
{
    Random::mtSeed(0);

    RectangularROI<int> roi1(255, cv::Point(150, 150), 150, 150);
    RectangularROI<int> roi2(255, cv::Point(350, 350), 100, 100);

    cv::Mat labels(512, 512, CV_32SC1, cv::Scalar(0));
    roi1.append(labels);
    roi2.append(labels);

    LabelSliceExtractionTransformation trans(width, height, 255);
    trans.setParameter("SlicesMargin", slicesMargin);

    const cv::Mat img = cv::imread("tests_data/Lenna.png",
#if CV_MAJOR_VERSION >= 3
        cv::IMREAD_COLOR);
#else
        CV_LOAD_IMAGE_COLOR);
#endif

    if (!img.data)
        throw std::runtime_error(
            "Could not open or find image: tests_data/Lenna.png");

    cv::Mat slices = ((cv::Mat)labels).clone();

    for (unsigned int i = 0; i < 100; ++i) {
        cv::Mat imgCopy = img.clone();
        cv::Mat labelsCopy = labels.clone();
        trans.apply(imgCopy, labelsCopy);

        ASSERT_EQUALS(imgCopy.rows, (int)height);
        ASSERT_EQUALS(imgCopy.cols, (int)width);
        ASSERT_EQUALS(labelsCopy.rows, 1);
        ASSERT_EQUALS(labelsCopy.cols, 1);

        cv::rectangle(slices,
                      trans.getLastSlice().tl(),
                      trans.getLastSlice().br(),
                      cv::Scalar(128));
    }

    std::ostringstream fileName;
    fileName << "LabelSliceExtractionTransformation_apply__quater(W" << width
             << "_H" << height << "_S" << slicesMargin << ").png";

    Utils::createDirectories("Transformation");
    if (!cv::imwrite("Transformation/" + fileName.str(), slices))
        throw std::runtime_error("Unable to write image: Transformation/"
                                 + fileName.str());
}

RUN_TESTS()
