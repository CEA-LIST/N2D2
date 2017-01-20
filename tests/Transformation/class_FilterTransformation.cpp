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

#include "Transformation/FilterTransformation.hpp"
#include "utils/UnitTest.hpp"

using namespace N2D2;

TEST_DATASET(FilterTransformation,
             apply__GaborKernel,
             (bool color, double theta),
             std::make_tuple(true, 0.0),
             std::make_tuple(true, M_PI / 4.0),
             std::make_tuple(false, M_PI / 2.0),
             std::make_tuple(false, 3.0 * M_PI / 4.0))
{
    FilterTransformation trans(GaborKernel<double>(9, 9, theta));

    cv::Mat img
        = cv::imread("tests_data/Lenna.png",
                     (color) ? CV_LOAD_IMAGE_COLOR : CV_LOAD_IMAGE_GRAYSCALE);

    if (!img.data)
        throw std::runtime_error(
            "Could not open or find image: tests_data/Lenna.png");

    const int cols = img.cols;
    const int rows = img.rows;

    trans.apply(img);

    std::ostringstream fileName;
    fileName << "FilterTransformation_apply__GaborKernel(C" << color << "_T"
             << theta << ").png";

    if (!cv::imwrite("Transformation/" + fileName.str(), img))
        throw std::runtime_error("Unable to write image: Transformation/"
                                 + fileName.str());

    ASSERT_EQUALS(img.cols, cols);
    ASSERT_EQUALS(img.rows, rows);
}

TEST_DATASET(FilterTransformation,
             apply__GaussianKernel,
             (bool color, double sigma),
             std::make_tuple(true, 0.5),
             std::make_tuple(true, 1.0),
             std::make_tuple(false, 2.0),
             std::make_tuple(false, 5.0))
{
    FilterTransformation trans(GaussianKernel<double>(5, 5, sigma));

    cv::Mat img
        = cv::imread("tests_data/Lenna.png",
                     (color) ? CV_LOAD_IMAGE_COLOR : CV_LOAD_IMAGE_GRAYSCALE);

    if (!img.data)
        throw std::runtime_error(
            "Could not open or find image: tests_data/Lenna.png");

    const int cols = img.cols;
    const int rows = img.rows;

    trans.apply(img);

    std::ostringstream fileName;
    fileName << "FilterTransformation_apply__GaussianKernel(C" << color << "_S"
             << sigma << ").png";

    if (!cv::imwrite("Transformation/" + fileName.str(), img))
        throw std::runtime_error("Unable to write image: Transformation/"
                                 + fileName.str());

    ASSERT_EQUALS(img.cols, cols);
    ASSERT_EQUALS(img.rows, rows);
}

TEST_DATASET(FilterTransformation,
             reconstructFilter__GaborKernel,
             (double theta),
             std::make_tuple(0.0),
             std::make_tuple(M_PI / 4.0),
             std::make_tuple(M_PI / 2.0),
             std::make_tuple(3.0 * M_PI / 4.0))
{
    FilterTransformation trans(GaborKernel<double>(9, 9, theta), theta);

    std::ostringstream fileName;
    fileName << "FilterTransformation_reconstructFilter__GaborKernel(T" << theta
             << ").dat";
    trans.getKernel().log("Transformation/" + fileName.str());
}

RUN_TESTS()
