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

#include "utils/IniParser.hpp"
#include "utils/UnitTest.hpp"

using namespace N2D2;

TEST(IniParser, load)
{
    const std::string data = "[conv1]\n"
                             "KernelWidth=4\n"
                             "KernelHeight=4\n"
                             "NbChannels=16\n"
                             "Stride=2\n"
                             "Test=1 2 3 4\n";

    UnitTest::FileWriteContent("IniParser.in", data);

    IniParser iniConfig;
    iniConfig.load("IniParser.in");

    iniConfig.currentSection("conv1");

    ASSERT_EQUALS(iniConfig.getProperty<int>("KernelWidth"), 4);
    ASSERT_EQUALS(iniConfig.getProperty<int>("NbChannels"), 16);
    ASSERT_EQUALS(iniConfig.getProperty<int>("Stride"), 2);
    ASSERT_EQUALS(iniConfig.getProperty<int>("StrideX", 3), 3);

    std::vector<int> test = iniConfig.getProperty<std::vector<int> >("Test");
    ASSERT_EQUALS(test.size(), 4U);
    ASSERT_EQUALS(test[0], 1);
    ASSERT_EQUALS(test[1], 2);
    ASSERT_EQUALS(test[2], 3);
    ASSERT_EQUALS(test[3], 4);

    ASSERT_THROW(iniConfig.getProperty<int>("NotExist"), std::runtime_error);
    ASSERT_THROW(iniConfig.currentSection(), std::runtime_error);
}

RUN_TESTS()
