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
#include "Transformation/PadCropTransformation.hpp"
#include "utils/UnitTest.hpp"

using namespace N2D2;

TEST_DATASET(PadCropTransformation,
             apply,
             (unsigned int width,
              unsigned int height,
              PadCropTransformation::PaddingBackground paddingBackground),
             std::make_tuple(200, 160, PadCropTransformation::BlackColor),
             std::make_tuple(160, 200, PadCropTransformation::BlackColor),
             std::make_tuple(512, 512, PadCropTransformation::BlackColor),
             std::make_tuple(700, 512, PadCropTransformation::BlackColor),
             std::make_tuple(512, 700, PadCropTransformation::BlackColor),
             std::make_tuple(700, 300, PadCropTransformation::BlackColor),
             std::make_tuple(300, 700, PadCropTransformation::BlackColor),
             std::make_tuple(700, 800, PadCropTransformation::BlackColor),
             std::make_tuple(800, 700, PadCropTransformation::BlackColor),
             std::make_tuple(700, 512, PadCropTransformation::MeanColor),
             std::make_tuple(512, 700, PadCropTransformation::MeanColor),
             std::make_tuple(700, 300, PadCropTransformation::MeanColor),
             std::make_tuple(300, 700, PadCropTransformation::MeanColor),
             std::make_tuple(700, 800, PadCropTransformation::MeanColor),
             std::make_tuple(800, 700, PadCropTransformation::MeanColor))
{
    RectangularROI<int> roi1(64, cv::Point(0, 0), 255, 255);
    RectangularROI<int> roi2(128, cv::Point(256, 0), 255, 255);
    RectangularROI<int> roi3(255, cv::Point(256, 256), 255, 255);

    cv::Mat labels(512, 512, CV_32SC1, cv::Scalar(0));
    roi1.append(labels);
    roi2.append(labels);
    roi3.append(labels);

    PadCropTransformation trans(width, height);
    trans.setParameter("PaddingBackground", paddingBackground);

    cv::Mat img = cv::imread("tests_data/Lenna.png", CV_LOAD_IMAGE_COLOR);

    if (!img.data)
        throw std::runtime_error(
            "Could not open or find image: tests_data/Lenna.png");

    const cv::Scalar mean = cv::mean(img);
    trans.apply(img, labels);

    std::ostringstream fileName;
    fileName << "PadCropTransformation_apply(W" << width << "_H" << height
             << "_P" << paddingBackground << ")[frame].png";

    if (!cv::imwrite("Transformation/" + fileName.str(), img))
        throw std::runtime_error("Unable to write image: Transformation/"
                                 + fileName.str());

    ASSERT_EQUALS(img.cols, (int)width);
    ASSERT_EQUALS(img.rows, (int)height);

    if (width > 512 || height > 512) {
        const cv::Vec3b value = img.at<cv::Vec3b>(0, 0);

        if (paddingBackground == PadCropTransformation::BlackColor) {
            ASSERT_EQUALS(value[0], 0);
            ASSERT_EQUALS(value[1], 0);
            ASSERT_EQUALS(value[2], 0);
        } else {
            ASSERT_EQUALS(value[0], (unsigned char)mean.val[0]);
            ASSERT_EQUALS(value[1], (unsigned char)mean.val[1]);
            ASSERT_EQUALS(value[2], (unsigned char)mean.val[2]);
        }
    }

    fileName.str(std::string());
    fileName << "PadCropTransformation_apply(W" << width << "_H" << height
             << "_P" << paddingBackground << ")[labels].png";

    if (!cv::imwrite("Transformation/" + fileName.str(), labels))
        throw std::runtime_error("Unable to write image: Transformation/"
                                 + fileName.str());

    ASSERT_EQUALS(labels.cols, (int)width);
    ASSERT_EQUALS(labels.rows, (int)height);

    if (width > 512 || height > 512)
        ASSERT_EQUALS(labels.at<int>(0, 0), -1);
}

RUN_TESTS()
