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

#include "Transformation/NormalizeTransformation.hpp"
#include "utils/UnitTest.hpp"

using namespace N2D2;

TEST_DATASET(NormalizeTransformation,
             apply,
             (bool color, NormalizeTransformation::Norm norm, double normValue),
             std::make_tuple(true, NormalizeTransformation::L1, 1.0),
             std::make_tuple(true, NormalizeTransformation::L1, 0.5),
             std::make_tuple(true, NormalizeTransformation::L2, 1.0),
             std::make_tuple(true, NormalizeTransformation::L2, 0.5),
             std::make_tuple(true, NormalizeTransformation::Linf, 1.0),
             std::make_tuple(true, NormalizeTransformation::Linf, 0.5),
             std::make_tuple(false, NormalizeTransformation::L1, 1.0),
             std::make_tuple(false, NormalizeTransformation::L1, 0.5),
             std::make_tuple(false, NormalizeTransformation::L2, 1.0),
             std::make_tuple(false, NormalizeTransformation::L2, 0.5),
             std::make_tuple(false, NormalizeTransformation::Linf, 1.0),
             std::make_tuple(false, NormalizeTransformation::Linf, 0.5))
{
    NormalizeTransformation trans;
    trans.setParameter("Norm", norm);
    trans.setParameter("NormValue", normValue);

    cv::Mat img
        = cv::imread("tests_data/Lenna.png",
                     (color) ? CV_LOAD_IMAGE_COLOR : CV_LOAD_IMAGE_GRAYSCALE);

    if (!img.data)
        throw std::runtime_error(
            "Could not open or find image: tests_data/Lenna.png");

    trans.apply(img);

    std::ostringstream fileName;
    fileName << "NormalizeTransformation_apply(C" << color << "_N" << norm
             << "_NV" << normValue << ").png";

    if (!cv::imwrite("Transformation/" + fileName.str(), img))
        throw std::runtime_error("Unable to write image: Transformation/"
                                 + fileName.str());
}

TEST_DATASET(NormalizeTransformation,
             apply__MinMax,
             (bool color, double normMin, double normMax),
             std::make_tuple(true, 0.0, 1.0),
             std::make_tuple(true, 0.0, 0.5),
             std::make_tuple(true, 0.5, 1.0),
             std::make_tuple(true, 0.25, 0.75),
             std::make_tuple(false, 0.0, 1.0),
             std::make_tuple(false, 0.0, 0.5),
             std::make_tuple(false, 0.5, 1.0),
             std::make_tuple(false, 0.25, 0.75))
{
    NormalizeTransformation trans;
    trans.setParameter("Norm", NormalizeTransformation::MinMax);
    trans.setParameter("NormMin", normMin);
    trans.setParameter("NormMax", normMax);

    cv::Mat img
        = cv::imread("tests_data/Lenna.png",
                     (color) ? CV_LOAD_IMAGE_COLOR : CV_LOAD_IMAGE_GRAYSCALE);

    if (!img.data)
        throw std::runtime_error(
            "Could not open or find image: tests_data/Lenna.png");

    trans.apply(img);

    std::ostringstream fileName;
    fileName << "NormalizeTransformation_apply__MinMax(C" << color << "_NMIN"
             << normMin << "_NMAX" << normMax << ").png";

    if (!cv::imwrite("Transformation/" + fileName.str(), img))
        throw std::runtime_error("Unable to write image: Transformation/"
                                 + fileName.str());
}

RUN_TESTS()
