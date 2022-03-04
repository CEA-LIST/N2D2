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

#include "Cell/ConvCell_Frame.hpp"
#include "DeepNet.hpp"
#include "Xnet/Environment.hpp"
#include "Xnet/Network.hpp"
#include "Generator/MappingGenerator.hpp"
#include "containers/Matrix.hpp"
#include "utils/UnitTest.hpp"

using namespace N2D2;

TEST(MappingGenerator, MappingGenerator)
{
    const std::string data = "[conv2]\n"
                             "Mapping(conv1).SizeX=1\n"
                             "Mapping(conv1).SizeY=2\n"
                             "Mapping(conv1).StrideX=2\n"
                             "Mapping(conv1).StrideY=1\n"
                             "Mapping(conv1).OffsetX=2\n"
                             "Mapping(conv1).OffsetY=1\n"
                             "Mapping(conv1).NbIterations=3\n";

    UnitTest::FileWriteContent("MappingGenerator.in", data);

    IniParser iniConfig;
    iniConfig.load("MappingGenerator.in");

    Network net(0U,false);
    DeepNet dn(net);
    Environment env(net, EmptyDatabase, {24, 24, 1});
    const std::shared_ptr<ConvCell_Frame<Float_T> > conv1(
        new ConvCell_Frame<Float_T>(dn, "conv1", std::vector<unsigned int>({4, 4}), 10));

    Tensor<bool> mapping
        = MappingGenerator::generate(env, conv1, 10, iniConfig, "conv2",
                                     "Mapping");

    Tensor<bool> mappingCheck;
    mappingCheck << "0 0 0 0 0 0 0 0 0 0 "
                    "0 0 1 0 0 0 0 0 0 0 "
                    "0 0 1 0 1 0 0 0 0 0 "
                    "0 0 0 0 1 0 1 0 0 0 "
                    "0 0 0 0 0 0 1 0 0 0 "
                    "0 0 0 0 0 0 0 0 0 0 "
                    "0 0 0 0 0 0 0 0 0 0 "
                    "0 0 0 0 0 0 0 0 0 0 "
                    "0 0 0 0 0 0 0 0 0 0 "
                    "0 0 0 0 0 0 0 0 0 0";
    mappingCheck.reshape({10, 10});

    ASSERT_EQUALS(mapping.dimY(), mappingCheck.dimY());
    ASSERT_EQUALS(mapping.dimX(), mappingCheck.dimX());

    for (unsigned int i = 0; i < mapping.size(); ++i) {
        ASSERT_EQUALS(mapping(i), mappingCheck(i));
    }
}

TEST(MappingGenerator, MappingGenerator_bis)
{
    const std::string data = "[conv2]\n"
                             "Mapping(conv1)=\\\n"
                             "0 0 0 0 0 0 0 0 0 0 \\\n"
                             "0 0 1 0 0 0 0 0 0 0 \\\n"
                             "0 0 1 0 1 0 0 0 0 0 \\\n"
                             "0 0 0 0 1 0 1 0 0 0 \\\n"
                             "0 0 0 0 0 0 1 0 0 0 \\\n"
                             "0 0 0 0 0 0 0 0 0 0 \\\n"
                             "0 0 0 0 0 0 0 0 0 0 \\\n"
                             "0 0 0 0 0 0 0 0 0 0 \\\n"
                             "0 0 0 0 0 0 0 0 0 0 \\\n"
                             "0 0 0 0 0 0 0 0 0 0\n";

    UnitTest::FileWriteContent("MappingGenerator_bis.in", data);

    IniParser iniConfig;
    iniConfig.load("MappingGenerator_bis.in");

    Network net(0U,false);
    DeepNet dn(net);
    Environment env(net, EmptyDatabase, {24, 24, 1});
    const std::shared_ptr<ConvCell_Frame<Float_T> > conv1(
        new ConvCell_Frame<Float_T>(dn, "conv1", std::vector<unsigned int>({4, 4}), 10));

    Tensor<bool> mapping
        = MappingGenerator::generate(env, conv1, 10, iniConfig, "conv2",
                                     "Mapping");

    Tensor<bool> mappingCheck;
    mappingCheck << "0 0 0 0 0 0 0 0 0 0 "
                    "0 0 1 0 0 0 0 0 0 0 "
                    "0 0 1 0 1 0 0 0 0 0 "
                    "0 0 0 0 1 0 1 0 0 0 "
                    "0 0 0 0 0 0 1 0 0 0 "
                    "0 0 0 0 0 0 0 0 0 0 "
                    "0 0 0 0 0 0 0 0 0 0 "
                    "0 0 0 0 0 0 0 0 0 0 "
                    "0 0 0 0 0 0 0 0 0 0 "
                    "0 0 0 0 0 0 0 0 0 0";
    mappingCheck.reshape({10, 10});

    ASSERT_EQUALS(mapping.dimY(), mappingCheck.dimY());
    ASSERT_EQUALS(mapping.dimX(), mappingCheck.dimX());

    for (unsigned int i = 0; i < mapping.size(); ++i) {
        ASSERT_EQUALS(mapping(i), mappingCheck(i));
    }
}

TEST(MappingGenerator, MappingGenerator_ter)
{
    const std::string data = "[conv2]\n";

    UnitTest::FileWriteContent("MappingGenerator_ter.in", data);

    IniParser iniConfig;
    iniConfig.load("MappingGenerator_ter.in");

    Network net(0U,false);
    DeepNet dn(net);
    Environment env(net, EmptyDatabase, {24, 24, 1});
    const std::shared_ptr<ConvCell_Frame<Float_T> > conv1(
        new ConvCell_Frame<Float_T>(dn, "conv1", std::vector<unsigned int>({4, 4}), 10));

    Tensor<bool> mapping
        = MappingGenerator::generate(env, conv1, 10, iniConfig, "conv2",
                                     "Mapping");

    Tensor<bool> mappingCheck({10, 10}, true);
    ASSERT_EQUALS(mapping.dimY(), mappingCheck.dimY());
    ASSERT_EQUALS(mapping.dimX(), mappingCheck.dimX());

    for (unsigned int i = 0; i < mapping.size(); ++i) {
        ASSERT_EQUALS(mapping(i), mappingCheck(i));
    }
}

TEST(MappingGenerator, MappingGenerator_env_ter)
{
    const std::string data = "[conv1]\n";

    UnitTest::FileWriteContent("MappingGenerator_env_ter.in", data);

    IniParser iniConfig;
    iniConfig.load("MappingGenerator_env_ter.in");

    Network net(0U,false);
    Environment env(net, EmptyDatabase, {24, 24, 1});
    Tensor<bool> mapping = MappingGenerator::generate(
        env, std::shared_ptr<Cell>(), 10, iniConfig, "conv1", "Mapping");

    Tensor<bool> mappingCheck({10, 1}, true);
    ASSERT_EQUALS(mapping.dimY(), mappingCheck.dimY());
    ASSERT_EQUALS(mapping.dimX(), mappingCheck.dimX());

    for (unsigned int i = 0; i < mapping.size(); ++i) {
        ASSERT_EQUALS(mapping(i), mappingCheck(i));
    }
}

RUN_TESTS()
