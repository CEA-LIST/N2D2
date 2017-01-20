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

#include "Transformation/ThresholdTransformation.hpp"
#include "utils/UnitTest.hpp"

using namespace N2D2;

TEST_DATASET(ThresholdTransformation,
             apply,
             (double threshold,
              ThresholdTransformation::Operation operation,
              double maxValue),
             // 1.
             std::make_tuple(127, ThresholdTransformation::Binary, 255),
             std::make_tuple(127, ThresholdTransformation::BinaryInverted, 255),
             std::make_tuple(127, ThresholdTransformation::Truncate, 255),
             std::make_tuple(127, ThresholdTransformation::ToZero, 255),
             std::make_tuple(127, ThresholdTransformation::ToZeroInverted, 255),
             // 2.
             std::make_tuple(127, ThresholdTransformation::Binary, 64),
             std::make_tuple(127, ThresholdTransformation::BinaryInverted, 64),
             std::make_tuple(127, ThresholdTransformation::Truncate, 64),
             std::make_tuple(127, ThresholdTransformation::ToZero, 64),
             std::make_tuple(127, ThresholdTransformation::ToZeroInverted, 64))
{
    ThresholdTransformation trans(threshold);
    trans.setParameter("Operation", operation);
    trans.setParameter("MaxValue", maxValue);

    cv::Mat img = cv::imread("tests_data/Lenna.png", CV_LOAD_IMAGE_GRAYSCALE);

    if (!img.data)
        throw std::runtime_error(
            "Could not open or find image: tests_data/Lenna.png");

    trans.apply(img);

    std::ostringstream fileName;
    fileName << "ThresholdTransformation_apply(T" << threshold << "_O"
             << operation << "_M" << maxValue << ").png";

    if (!cv::imwrite("Transformation/" + fileName.str(), img))
        throw std::runtime_error("Unable to write image: Transformation/"
                                 + fileName.str());
}

RUN_TESTS()
