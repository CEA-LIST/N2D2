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
#include "Transformation/MorphologyTransformation.hpp"
#include "utils/UnitTest.hpp"
#include "utils/Utils.hpp"

using namespace N2D2;

TEST_DATASET(MorphologyTransformation,
             apply,
             (MorphologyTransformation::Operation operation,
              unsigned int size,
              bool applyToLabels),
             // 1.
             std::make_tuple(MorphologyTransformation::Erode, 5, false),
             std::make_tuple(MorphologyTransformation::Dilate, 5, false),
             std::make_tuple(MorphologyTransformation::Opening, 5, false),
             std::make_tuple(MorphologyTransformation::Closing, 5, false),
             std::make_tuple(MorphologyTransformation::Gradient, 5, false),
             std::make_tuple(MorphologyTransformation::TopHat, 5, false),
             std::make_tuple(MorphologyTransformation::BlackHat, 5, false),
             // 2.
             std::make_tuple(MorphologyTransformation::Erode, 5, true),
             std::make_tuple(MorphologyTransformation::Dilate, 5, true),
             std::make_tuple(MorphologyTransformation::Opening, 5, true),
             std::make_tuple(MorphologyTransformation::Closing, 5, true),
             std::make_tuple(MorphologyTransformation::Gradient, 5, true),
             std::make_tuple(MorphologyTransformation::TopHat, 5, true),
             std::make_tuple(MorphologyTransformation::BlackHat, 5, true))
{
    RectangularROI<int> roi1(255, cv::Point(150, 150), 150, 150);
    RectangularROI<int> roi2(196, cv::Point(350, 350), 100, 100);
    RectangularROI<int> roi3(128, cv::Point(100, 400), 3, 3);
    RectangularROI<int> roi4(100, cv::Point(150, 400), 4, 4);
    RectangularROI<int> roi5(64, cv::Point(200, 400), 5, 5);

    cv::Mat labels(512, 512, CV_32SC1, cv::Scalar(0));
    roi1.append(labels);
    roi2.append(labels);
    roi3.append(labels);
    roi4.append(labels);
    roi5.append(labels);

    MorphologyTransformation trans(operation, size, applyToLabels);

    cv::Mat img = cv::imread("tests_data/Lenna.png",
#if CV_MAJOR_VERSION >= 3
        cv::IMREAD_GRAYSCALE);
#else
        CV_LOAD_IMAGE_GRAYSCALE);
#endif

    if (!img.data)
        throw std::runtime_error(
            "Could not open or find image: tests_data/Lenna.png");

    trans.apply(img, labels);

    Utils::createDirectories("Transformation");
    if (applyToLabels) {
        std::ostringstream fileName;
        fileName << "MorphologyTransformation_apply(O" << operation << "_S"
                 << size << ")[labels].png";

        if (!cv::imwrite("Transformation/" + fileName.str(), labels))
            throw std::runtime_error("Unable to write image: Transformation/"
                                     + fileName.str());
    } else {
        std::ostringstream fileName;
        fileName << "MorphologyTransformation_apply(O" << operation << "_S"
                 << size << ")[frame].png";

        if (!cv::imwrite("Transformation/" + fileName.str(), img))
            throw std::runtime_error("Unable to write image: Transformation/"
                                     + fileName.str());
    }
}

RUN_TESTS()
