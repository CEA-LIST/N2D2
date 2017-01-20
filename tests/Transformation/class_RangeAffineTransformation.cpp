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

#include "Transformation/RangeAffineTransformation.hpp"
#include "utils/UnitTest.hpp"

using namespace N2D2;

TEST_DATASET(RangeAffineTransformation,
             apply,
             (RangeAffineTransformation::Operator firstOperator,
              double firstValue,
              RangeAffineTransformation::Operator secondOperator,
              double secondValue),
             std::make_tuple(RangeAffineTransformation::Plus,
                             1.0,
                             RangeAffineTransformation::Multiplies,
                             2.0),
             std::make_tuple(RangeAffineTransformation::Multiplies,
                             2.0,
                             RangeAffineTransformation::Plus,
                             1.0),
             std::make_tuple(RangeAffineTransformation::Plus,
                             1.0,
                             RangeAffineTransformation::Divides,
                             2.0),
             std::make_tuple(RangeAffineTransformation::Divides,
                             2.0,
                             RangeAffineTransformation::Plus,
                             1.0),
             std::make_tuple(RangeAffineTransformation::Minus,
                             1.0,
                             RangeAffineTransformation::Multiplies,
                             2.0),
             std::make_tuple(RangeAffineTransformation::Multiplies,
                             3.0,
                             RangeAffineTransformation::Minus,
                             3.0),
             std::make_tuple(RangeAffineTransformation::Minus,
                             3.0,
                             RangeAffineTransformation::Divides,
                             3.0),
             std::make_tuple(RangeAffineTransformation::Divides,
                             3.0,
                             RangeAffineTransformation::Minus,
                             3.0),
             std::make_tuple(RangeAffineTransformation::Plus,
                             4.0,
                             RangeAffineTransformation::Plus,
                             3.0),
             std::make_tuple(RangeAffineTransformation::Minus,
                             4.0,
                             RangeAffineTransformation::Minus,
                             1.0),
             std::make_tuple(RangeAffineTransformation::Divides,
                             5.0,
                             RangeAffineTransformation::Multiplies,
                             2.0),
             std::make_tuple(RangeAffineTransformation::Multiplies,
                             7.0,
                             RangeAffineTransformation::Divides,
                             8.0))
{
    RangeAffineTransformation trans(
        firstOperator, firstValue, secondOperator, secondValue);

    cv::Mat img = cv::imread("tests_data/Lenna.png", CV_LOAD_IMAGE_GRAYSCALE);

    if (!img.data)
        throw std::runtime_error(
            "Could not open or find image: tests_data/Lenna.png");

    cv::Mat img2 = img.clone();
    trans.apply(img2);

    cv::Mat img64F;
    img.convertTo(img64F, CV_64F);

    switch (firstOperator) {
    case RangeAffineTransformation::Plus:
        img64F += firstValue;
        break;
    case RangeAffineTransformation::Minus:
        img64F -= firstValue;
        break;
    case RangeAffineTransformation::Multiplies:
        img64F *= firstValue;
        break;
    case RangeAffineTransformation::Divides:
        img64F /= firstValue;
        break;
    }

    switch (secondOperator) {
    case RangeAffineTransformation::Plus:
        img64F += secondValue;
        break;
    case RangeAffineTransformation::Minus:
        img64F -= secondValue;
        break;
    case RangeAffineTransformation::Multiplies:
        img64F *= secondValue;
        break;
    case RangeAffineTransformation::Divides:
        img64F /= secondValue;
        break;
    }

    cv::Mat diff = img64F != img2;
    ASSERT_EQUALS(cv::countNonZero(diff), 0);
}

TEST_DATASET(RangeAffineTransformation,
             apply__nochange,
             (RangeAffineTransformation::Operator firstOperator,
              double firstValue,
              RangeAffineTransformation::Operator secondOperator,
              double secondValue),
             std::make_tuple(RangeAffineTransformation::Plus,
                             0.0,
                             RangeAffineTransformation::Multiplies,
                             1.0),
             std::make_tuple(RangeAffineTransformation::Multiplies,
                             1.0,
                             RangeAffineTransformation::Plus,
                             0.0),
             std::make_tuple(RangeAffineTransformation::Plus,
                             0.0,
                             RangeAffineTransformation::Divides,
                             1.0),
             std::make_tuple(RangeAffineTransformation::Divides,
                             1.0,
                             RangeAffineTransformation::Plus,
                             0.0),
             std::make_tuple(RangeAffineTransformation::Minus,
                             0.0,
                             RangeAffineTransformation::Multiplies,
                             1.0),
             std::make_tuple(RangeAffineTransformation::Multiplies,
                             1.0,
                             RangeAffineTransformation::Minus,
                             0.0),
             std::make_tuple(RangeAffineTransformation::Minus,
                             0.0,
                             RangeAffineTransformation::Divides,
                             1.0),
             std::make_tuple(RangeAffineTransformation::Divides,
                             1.0,
                             RangeAffineTransformation::Minus,
                             0.0),
             std::make_tuple(RangeAffineTransformation::Plus,
                             0.0,
                             RangeAffineTransformation::Plus,
                             0.0),
             std::make_tuple(RangeAffineTransformation::Minus,
                             0.0,
                             RangeAffineTransformation::Minus,
                             0.0),
             std::make_tuple(RangeAffineTransformation::Divides,
                             1.0,
                             RangeAffineTransformation::Multiplies,
                             1.0),
             std::make_tuple(RangeAffineTransformation::Multiplies,
                             1.0,
                             RangeAffineTransformation::Divides,
                             1.0))
{
    RangeAffineTransformation trans(
        firstOperator, firstValue, secondOperator, secondValue);

    const cv::Mat img
        = cv::imread("tests_data/Lenna.png", CV_LOAD_IMAGE_GRAYSCALE);

    if (!img.data)
        throw std::runtime_error(
            "Could not open or find image: tests_data/Lenna.png");

    cv::Mat img2 = img.clone();
    trans.apply(img2);

    cv::Mat img64F;
    img.convertTo(img64F, CV_64F);

    cv::Mat diff = img64F != img2;
    ASSERT_EQUALS(cv::countNonZero(diff), 0);
}

RUN_TESTS()
