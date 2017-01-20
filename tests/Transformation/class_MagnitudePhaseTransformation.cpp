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

#include "Transformation/DFTTransformation.hpp"
#include "Transformation/MagnitudePhaseTransformation.hpp"
#include "containers/Tensor2d.hpp"
#include "utils/UnitTest.hpp"

using namespace N2D2;

TEST_DATASET(MagnitudePhaseTransformation,
             apply__1D,
             (std::string xStr, std::string yStr),
             std::make_tuple(
                 "1.0 1.0 1.0 1.0 0.0 0.0 0.0 0.0",
                 "(4,0) (1,-2.414214) (0,0) (1,-0.4142136) (0,0) (1,0.4142136) "
                 "(0,0) (1,2.414214)"),
             std::make_tuple(
                 "0 1 2 3\n4 5 6 7",
                 "(6,0) (-2,2) (-2,0) (-2,-2)\n(22,0) (-2,2) (-2,0) (-2,-2)"))
{
    Tensor2d<double> x;
    x << xStr;

    cv::Mat data = (cv::Mat)x;

    DFTTransformation trans(false);
    trans.apply(data);

    MagnitudePhaseTransformation trans2;
    trans2.apply(data);

    ASSERT_EQUALS(data.channels(), 2);
    ASSERT_EQUALS(data.cols, (int)x.dimX());
    ASSERT_EQUALS(data.rows, (int)x.dimY());

    Tensor2d<std::complex<double> > y;
    y << yStr;

    for (int i = 0; i < data.rows; ++i) {
        for (int j = 0; j < data.cols; ++j) {
            ASSERT_EQUALS_DELTA(
                data.at<cv::Vec2d>(i, j)[0], std::abs(y(j, i)), 1.0e-6);
            ASSERT_EQUALS_DELTA(
                data.at<cv::Vec2d>(i, j)[1], std::arg(y(j, i)), 1.0e-6);
        }
    }
}

TEST(MagnitudePhaseTransformation, apply__2D)
{
    DFTTransformation trans(true);

    cv::Mat img = cv::imread("tests_data/Lenna.png", CV_LOAD_IMAGE_GRAYSCALE);

    if (!img.data)
        throw std::runtime_error(
            "Could not open or find image: tests_data/Lenna.png");

    trans.apply(img);

    MagnitudePhaseTransformation trans2(true);
    trans2.apply(img);

    std::vector<cv::Mat> planes;
    cv::split(img, planes);

    cv::Mat magNorm, phaseNorm;
    cv::normalize(planes[0], magNorm, 0.0, 255.0, cv::NORM_MINMAX);
    cv::normalize(planes[1], phaseNorm, 0.0, 255.0, cv::NORM_MINMAX);

    std::ostringstream fileName;
    fileName << "MagnitudePhaseTransformation_apply__2D[mag].png";

    if (!cv::imwrite("Transformation/" + fileName.str(), magNorm))
        throw std::runtime_error("Unable to write image: Transformation/"
                                 + fileName.str());

    fileName.str(std::string());
    fileName << "MagnitudePhaseTransformation_apply__2D[ang].png";

    if (!cv::imwrite("Transformation/" + fileName.str(), phaseNorm))
        throw std::runtime_error("Unable to write image: Transformation/"
                                 + fileName.str());
}

RUN_TESTS()
