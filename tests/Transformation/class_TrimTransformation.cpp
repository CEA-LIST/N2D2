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
#include "Transformation/TrimTransformation.hpp"
#include "utils/UnitTest.hpp"

using namespace N2D2;

TEST_DATASET(TrimTransformation,
             apply,
             (bool color,
              unsigned int nbLevels,
              TrimTransformation::Method method),
             std::make_tuple(true, 0, TrimTransformation::Discretize),
             std::make_tuple(true, 2, TrimTransformation::Discretize),
             std::make_tuple(true, 10, TrimTransformation::Discretize),
             std::make_tuple(true, 255, TrimTransformation::Discretize),
             std::make_tuple(true, 0, TrimTransformation::Reduce),
             std::make_tuple(true, 2, TrimTransformation::Reduce),
             std::make_tuple(true, 10, TrimTransformation::Reduce),
             std::make_tuple(true, 255, TrimTransformation::Reduce),
             std::make_tuple(false, 0, TrimTransformation::Discretize),
             std::make_tuple(false, 2, TrimTransformation::Discretize),
             std::make_tuple(false, 10, TrimTransformation::Discretize),
             std::make_tuple(false, 255, TrimTransformation::Discretize),
             std::make_tuple(false, 0, TrimTransformation::Reduce),
             std::make_tuple(false, 2, TrimTransformation::Reduce),
             std::make_tuple(false, 10, TrimTransformation::Reduce),
             std::make_tuple(false, 255, TrimTransformation::Reduce))
{
    RectangularROI<int> roi1(64, cv::Point(0, 0), 127, 127);
    RectangularROI<int> roi2(128, cv::Point(128, 0), 127, 127);
    RectangularROI<int> roi3(255, cv::Point(128, 128), 127, 127);

    cv::Mat labels(256, 256, CV_32SC1, cv::Scalar(0));
    roi1.append(labels);
    roi2.append(labels);
    roi3.append(labels);

    TrimTransformation trans(nbLevels);
    trans.setParameter("Method", method);

    cv::Mat img
        = cv::imread("tests_data/SIPI_Jelly_Beans_4.1.07.tiff",
                     (color) ? CV_LOAD_IMAGE_COLOR : CV_LOAD_IMAGE_GRAYSCALE);

    if (!img.data)
        throw std::runtime_error("Could not open or find image: "
                                 "tests_data/SIPI_Jelly_Beans_4.1.07.tiff");

    trans.apply(img, labels);

    std::ostringstream fileName;
    fileName << "TrimTransformation_apply(C" << color << "_N" << nbLevels
             << "_M" << method << ")[frame].png";

    if (!cv::imwrite("Transformation/" + fileName.str(), img))
        throw std::runtime_error("Unable to write image: Transformation/"
                                 + fileName.str());

    fileName.str(std::string());
    fileName << "TrimTransformation_apply(C" << color << "_N" << nbLevels
             << "_M" << method << ")[labels].png";

    if (!cv::imwrite("Transformation/" + fileName.str(), labels))
        throw std::runtime_error("Unable to write image: Transformation/"
                                 + fileName.str());

    ASSERT_EQUALS(img.cols, labels.cols);
    ASSERT_EQUALS(img.rows, labels.rows);
}

RUN_TESTS()
