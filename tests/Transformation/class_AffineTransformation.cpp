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

#include "Transformation/AffineTransformation.hpp"
#include "utils/UnitTest.hpp"

using namespace N2D2;

TEST_DATASET(
    AffineTransformation,
    apply,
    (AffineTransformation::Operator firstOperator,
     double firstValue,
     AffineTransformation::Operator secondOperator,
     double secondValue),
    std::make_tuple(
        AffineTransformation::Plus, 1.0, AffineTransformation::Multiplies, 2.0),
    std::make_tuple(
        AffineTransformation::Multiplies, 2.0, AffineTransformation::Plus, 1.0),
    std::make_tuple(
        AffineTransformation::Plus, 1.0, AffineTransformation::Divides, 2.0),
    std::make_tuple(
        AffineTransformation::Divides, 2.0, AffineTransformation::Plus, 1.0),
    std::make_tuple(AffineTransformation::Minus,
                    1.0,
                    AffineTransformation::Multiplies,
                    2.0),
    std::make_tuple(AffineTransformation::Multiplies,
                    3.0,
                    AffineTransformation::Minus,
                    3.0),
    std::make_tuple(
        AffineTransformation::Minus, 3.0, AffineTransformation::Divides, 3.0),
    std::make_tuple(
        AffineTransformation::Divides, 3.0, AffineTransformation::Minus, 3.0),
    std::make_tuple(
        AffineTransformation::Plus, 4.0, AffineTransformation::Plus, 3.0),
    std::make_tuple(
        AffineTransformation::Minus, 4.0, AffineTransformation::Minus, 1.0),
    std::make_tuple(AffineTransformation::Divides,
                    5.0,
                    AffineTransformation::Multiplies,
                    2.0),
    std::make_tuple(AffineTransformation::Multiplies,
                    7.0,
                    AffineTransformation::Divides,
                    8.0))
{
    cv::Mat img = cv::imread("tests_data/Lenna.png", CV_LOAD_IMAGE_COLOR);

    if (!img.data)
        throw std::runtime_error(
            "Could not open or find image: tests_data/Lenna.png");

    cv::Mat img2 = img.clone();

    cv::Mat mat1(
        cv::Size(img.cols, img.rows), CV_64FC3,
        cv::Scalar(firstValue, firstValue, firstValue));
    cv::Mat mat2(
        cv::Size(img.cols, img.rows), CV_64FC3,
        cv::Scalar(secondValue, secondValue, secondValue));
    AffineTransformation trans(firstOperator, mat1, secondOperator, mat2);

    trans.apply(img2);

    cv::Mat imgMat;
    img.convertTo(imgMat, mat1.type());

    switch (firstOperator) {
    case AffineTransformation::Plus:
        imgMat += mat1;
        break;
    case AffineTransformation::Minus:
        imgMat -= mat1;
        break;
    case AffineTransformation::Multiplies:
        cv::multiply(imgMat, mat1, imgMat);
        break;
    case AffineTransformation::Divides:
        cv::divide(imgMat, mat1, imgMat);
        break;
    }

    switch (secondOperator) {
    case AffineTransformation::Plus:
        imgMat += mat2;
        break;
    case AffineTransformation::Minus:
        imgMat -= mat2;
        break;
    case AffineTransformation::Multiplies:
        cv::multiply(imgMat, mat2, imgMat);
        break;
    case AffineTransformation::Divides:
        cv::divide(imgMat, mat2, imgMat);
        break;
    }

    ASSERT_EQUALS(img2.type(), CV_64FC3);

    cv::Mat imgMat2;
    img2.convertTo(imgMat2, mat1.type());

    cv::Mat diff = imgMat.reshape(1) != imgMat2.reshape(1);
    ASSERT_EQUALS(cv::countNonZero(diff), 0);
}

RUN_TESTS()
