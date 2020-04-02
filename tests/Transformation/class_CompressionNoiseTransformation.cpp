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
#include "Transformation/CompressionNoiseTransformation.hpp"
#include "utils/UnitTest.hpp"
#include "utils/Utils.hpp"

using namespace N2D2;

TEST_DATASET(CompressionNoiseTransformation,
             apply,
             (int compressionMin, int compressionMax),
             std::make_tuple(0, 0),
             std::make_tuple(50, 50),
             std::make_tuple(80, 80),
             std::make_tuple(85, 85),
             std::make_tuple(90, 90),
             std::make_tuple(95, 95),
             std::make_tuple(100, 100))
{
    Random::mtSeed(0);

    CompressionNoiseTransformation trans;
    trans.setParameter<std::vector<int> >("CompressionRange",
        std::vector<int>({compressionMin, compressionMax}));

    cv::Mat img
        = cv::imread("tests_data/Lenna.png",
#if CV_MAJOR_VERSION >= 3
                     cv::IMREAD_COLOR);
#else
                     CV_LOAD_IMAGE_COLOR);
#endif

    if (!img.data)
        throw std::runtime_error("Could not open or find image: "
                                 "tests_data/Lenna.png");

    trans.apply(img);

    std::ostringstream fileName;
    fileName << "CompressionNoiseTransformation_apply"
        << compressionMin << ".png";

    Utils::createDirectories("Transformation");
    if (!cv::imwrite("Transformation/" + fileName.str(), img))
        throw std::runtime_error("Unable to write image: Transformation/"
                                 + fileName.str());
}

RUN_TESTS()
