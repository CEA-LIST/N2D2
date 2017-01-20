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

#include "Environment.hpp"
#include "Generator/ConvCellGenerator.hpp"
#include "utils/UnitTest.hpp"

using namespace N2D2;

TEST(ConvCellGenerator, ConvCellGenerator)
{
    const std::string data = "[conv1]\n"
                             "KernelWidth=4\n"
                             "KernelHeight=4\n"
                             "NbChannels=16\n"
                             "Stride=2\n";

    UnitTest::FileWriteContent("ConvCellGenerator.in", data);

    IniParser iniConfig;
    iniConfig.load("ConvCellGenerator.in");

    Network net;
    Environment env(net, EmptyDatabase, 24, 24);

    std::shared_ptr<ConvCell> convCell = ConvCellGenerator::generate(
        net, env, std::vector<std::shared_ptr<Cell> >(1), iniConfig, "conv1");

    for (unsigned int output = 0; output < 16; ++output) {
        ASSERT_EQUALS(convCell->isConnection(0, output), true);
    }

    ASSERT_EQUALS(convCell->getNbChannels(), 1U);
    ASSERT_EQUALS(convCell->getKernelWidth(), 4U);
    ASSERT_EQUALS(convCell->getKernelHeight(), 4U);
    ASSERT_EQUALS(convCell->getNbOutputs(), 16U);
    ASSERT_EQUALS(convCell->getOutputsWidth(), 11U);
    ASSERT_EQUALS(convCell->getOutputsHeight(), 11U);
    ASSERT_EQUALS(convCell->getPaddingX(), 0U);
    ASSERT_EQUALS(convCell->getPaddingY(), 0U);
    ASSERT_EQUALS(convCell->getStrideX(), 2U);
    ASSERT_EQUALS(convCell->getStrideY(), 2U);
}

TEST_DATASET(ConvCellGenerator,
             ConvCellGenerator_mapping,
             (unsigned int unknownProperty),
             std::make_tuple(false),
             std::make_tuple(true))
{
    std::string data = "[conv1]\n"
                       "KernelWidth=4\n"
                       "KernelHeight=4\n"
                       "NbChannels=16\n"
                       "Stride=2\n"
                       "\n"
                       "[conv2]\n"
                       "KernelWidth=5\n"
                       "KernelHeight=5\n"
                       "NbChannels=24\n"
                       "Stride=2\n"
                       "Map(conv1)=\\\n"
                       "1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 1 1 \\\n"
                       "1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 1 1 \\\n"
                       "0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 1 1 \\\n"
                       "0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 1 1 \\\n"
                       "0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 1 1 \\\n"
                       "0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 1 1 \\\n"
                       "0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 1 1 \\\n"
                       "0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 1 1 0 0 0 1 1 \\\n"
                       "0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 1 1 0 0 1 1 \\\n"
                       "0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 1 1 0 0 1 1 \\\n"
                       "0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 1 1 0 1 1 \\\n"
                       "0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 1 1 0 1 1 \\\n"
                       "0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 1 1 1 1 \\\n"
                       "0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 1 1 1 1 \\\n"
                       "0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 1 1 1 \\\n"
                       "0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 1 1 1\n";

    if (unknownProperty)
        data += "Bidon=Blabla\n";

    UnitTest::FileWriteContent("ConvCellGenerator_mapping.in", data);

    IniParser iniConfig;
    iniConfig.load("ConvCellGenerator_mapping.in");

    Network net;
    Environment env(net, EmptyDatabase, 24, 24);

    std::shared_ptr<ConvCell> convCell1 = ConvCellGenerator::generate(
        net, env, std::vector<std::shared_ptr<Cell> >(1), iniConfig, "conv1");

    if (unknownProperty) {
        ASSERT_THROW_ANY(ConvCellGenerator::generate(
            net,
            env,
            std::vector<std::shared_ptr<Cell> >(1, convCell1),
            iniConfig,
            "conv2"));
    } else {
        std::shared_ptr<ConvCell> convCell2 = ConvCellGenerator::generate(
            net,
            env,
            std::vector<std::shared_ptr<Cell> >(1, convCell1),
            iniConfig,
            "conv2");

        ASSERT_EQUALS(convCell1->getNbChannels(), 1U);
        ASSERT_EQUALS(convCell1->getKernelWidth(), 4U);
        ASSERT_EQUALS(convCell1->getKernelHeight(), 4U);
        ASSERT_EQUALS(convCell1->getNbOutputs(), 16U);
        ASSERT_EQUALS(convCell1->getOutputsWidth(), 11U);
        ASSERT_EQUALS(convCell1->getOutputsHeight(), 11U);
        ASSERT_EQUALS(convCell1->getPaddingX(), 0U);
        ASSERT_EQUALS(convCell1->getPaddingY(), 0U);
        ASSERT_EQUALS(convCell1->getStrideX(), 2U);
        ASSERT_EQUALS(convCell1->getStrideY(), 2U);

        ASSERT_EQUALS(convCell2->getNbChannels(), 16U);
        ASSERT_EQUALS(convCell2->getKernelWidth(), 5U);
        ASSERT_EQUALS(convCell2->getKernelHeight(), 5U);
        ASSERT_EQUALS(convCell2->getNbOutputs(), 24U);
        ASSERT_EQUALS(convCell2->getOutputsWidth(), 4U);
        ASSERT_EQUALS(convCell2->getOutputsHeight(), 4U);
        ASSERT_EQUALS(convCell2->getPaddingX(), 0U);
        ASSERT_EQUALS(convCell2->getPaddingY(), 0U);
        ASSERT_EQUALS(convCell2->getStrideX(), 2U);
        ASSERT_EQUALS(convCell2->getStrideY(), 2U);

        Matrix<bool> mapping(16, 24);
        mapping << "1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 1 1 "
                   "1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 1 1 "
                   "0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 1 1 "
                   "0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 1 1 "
                   "0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 1 1 "
                   "0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 1 1 "
                   "0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 1 1 "
                   "0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 1 1 0 0 0 1 1 "
                   "0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 1 1 0 0 1 1 "
                   "0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 1 1 0 0 1 1 "
                   "0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 1 1 0 1 1 "
                   "0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 1 1 0 1 1 "
                   "0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 1 1 1 1 "
                   "0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 1 1 1 1 "
                   "0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 1 1 1 "
                   "0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 1 1 1";

        for (unsigned int output = 0; output < 24; ++output) {
            for (unsigned int channel = 0; channel < 16; ++channel) {
                ASSERT_EQUALS(convCell2->isConnection(channel, output),
                              mapping(channel, output));
            }
        }
    }
}

RUN_TESTS()
