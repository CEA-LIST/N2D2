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

#include "Generator/ChannelExtractionTransformationGenerator.hpp"
#include "Generator/TransformationGenerator.hpp"
#include "utils/UnitTest.hpp"

using namespace N2D2;

TEST(TransformationGenerator,
     TransformationGenerator_ChannelExtractionTransformation)
{
    const std::string data = "[test]\n"
                             "Type=ChannelExtractionTransformation\n"
                             "CSChannel=Gray\n";

    UnitTest::FileWriteContent(
        "TransformationGenerator_ChannelExtractionTransformation.in", data);

    IniParser iniConfig;
    iniConfig.load(
        "TransformationGenerator_ChannelExtractionTransformation.in");

    std::shared_ptr<Transformation> trans
        = TransformationGenerator::generate(iniConfig, "test");

    std::shared_ptr<ChannelExtractionTransformation> actualTrans
        = std::dynamic_pointer_cast<ChannelExtractionTransformation>(trans);

    ASSERT_TRUE(actualTrans != NULL);
    ASSERT_EQUALS(actualTrans->getChannel(),
                  ChannelExtractionTransformation::Gray);

    cv::Mat img = cv::imread("tests_data/Lenna.png", CV_LOAD_IMAGE_COLOR);

    if (!img.data)
        throw std::runtime_error(
            "Could not open or find image: tests_data/Lenna.png");

    trans->apply(img);

    std::ostringstream fileName;
    fileName << "TransformationGenerator_ChannelExtractionTransformation.png";

    if (!cv::imwrite("Transformation/" + fileName.str(), img))
        throw std::runtime_error("Unable to write image: Transformation/"
                                 + fileName.str());
}

RUN_TESTS()
