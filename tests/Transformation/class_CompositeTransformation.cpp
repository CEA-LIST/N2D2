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
#include "Transformation/CompositeTransformation.hpp"
#include "Transformation/FlipTransformation.hpp"
#include "Transformation/RescaleTransformation.hpp"
#include "utils/UnitTest.hpp"

using namespace N2D2;

TEST(CompositeTransformation, apply)
{
    RectangularROI<int> roi1(64, cv::Point(0, 0), 255, 255);
    RectangularROI<int> roi2(128, cv::Point(256, 0), 255, 255);
    RectangularROI<int> roi3(255, cv::Point(256, 256), 255, 255);

    cv::Mat labels(512, 512, CV_32SC1, cv::Scalar(0));
    roi1.append(labels);
    roi2.append(labels);
    roi3.append(labels);

    RescaleTransformation rescale(256, 256);

    CompositeTransformation trans;
    trans.push_back(rescale);
    trans.push_back(FlipTransformation(true, false));
    trans.push_back(BlueChannelExtractionTransformation());

    cv::Mat img = cv::imread("tests_data/Lenna.png", CV_LOAD_IMAGE_COLOR);

    if (!img.data)
        throw std::runtime_error(
            "Could not open or find image: tests_data/Lenna.png");

    trans.apply(img, labels);

    std::ostringstream fileName;
    fileName << "CompositeTransformation_apply[frame].png";

    if (!cv::imwrite("Transformation/" + fileName.str(), img))
        throw std::runtime_error("Unable to write image: Transformation/"
                                 + fileName.str());

    fileName.str(std::string());
    fileName << "CompositeTransformation_apply[labels].png";

    if (!cv::imwrite("Transformation/" + fileName.str(), labels))
        throw std::runtime_error("Unable to write image: Transformation/"
                                 + fileName.str());
}

TEST(CompositeTransformation, apply__random)
{
    Random::mtSeed(0);

    RectangularROI<int> roi1(64, cv::Point(0, 0), 255, 255);
    RectangularROI<int> roi2(128, cv::Point(256, 0), 255, 255);
    RectangularROI<int> roi3(255, cv::Point(256, 256), 255, 255);

    cv::Mat labels(512, 512, CV_32SC1, cv::Scalar(0));
    roi1.append(labels);
    roi2.append(labels);
    roi3.append(labels);

    FlipTransformation flip;
    flip.setParameter("RandomHorizontalFlip", true);
    flip.setParameter("RandomVerticalFlip", true);

    CompositeTransformation trans;
    trans.push_back(RescaleTransformation(256, 256));
    trans.push_back(flip);
    trans.push_back(BlueChannelExtractionTransformation());

    cv::Mat img = cv::imread("tests_data/Lenna.png", CV_LOAD_IMAGE_COLOR);

    if (!img.data)
        throw std::runtime_error(
            "Could not open or find image: tests_data/Lenna.png");

    trans.apply(img, labels);

    std::ostringstream fileName;
    fileName << "CompositeTransformation_apply__random[frame].png";

    if (!cv::imwrite("Transformation/" + fileName.str(), img))
        throw std::runtime_error("Unable to write image: Transformation/"
                                 + fileName.str());

    fileName.str(std::string());
    fileName << "CompositeTransformation_apply__random[labels].png";

    if (!cv::imwrite("Transformation/" + fileName.str(), labels))
        throw std::runtime_error("Unable to write image: Transformation/"
                                 + fileName.str());
}

RUN_TESTS()
