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

#include "Database/DIR_Database.hpp"
#include "Generator/StimuliProviderGenerator.hpp"
#include "Transformation/ChannelExtractionTransformation.hpp"
#include "Transformation/RescaleTransformation.hpp"
#include "Transformation/PadCropTransformation.hpp"
#include "Transformation/FlipTransformation.hpp"
#include "utils/UnitTest.hpp"

using namespace N2D2;

TEST(StimuliProviderGenerator, StimuliProviderGenerator)
{
    const std::string data = "[env]\n"
                             "SizeX=256\n"
                             "SizeY=256\n"
                             "BatchSize=3\n"
                             "\n"
                             "[env.Transformation]\n"
                             "Type=ChannelExtractionTransformation\n"
                             "CSChannel=Red\n"
                             "\n"
                             "[env.OnTheFlyTransformation]\n"
                             "Type=FilterTransformation\n"
                             "Kernel=Gabor\n"
                             "Kernel.SizeX=5\n"
                             "Kernel.SizeY=5\n"
                             "Kernel.Theta=0.785\n";

    UnitTest::FileWriteContent("StimuliProviderGenerator.in", data);

    IniParser iniConfig;
    iniConfig.load("StimuliProviderGenerator.in");

    DIR_Database database;
    std::shared_ptr<StimuliProvider> sp
        = StimuliProviderGenerator::generate(database, iniConfig, "env");

    ASSERT_EQUALS(sp->getNbChannels(), 1U);
    ASSERT_EQUALS(sp->getTransformation(Database::Learn).size(), 1U);
    ASSERT_EQUALS(sp->getTransformation(Database::Validation).size(), 1U);
    ASSERT_EQUALS(sp->getTransformation(Database::Test).size(), 1U);
    ASSERT_EQUALS(sp->getTransformation(Database::Learn).empty(), false);
    ASSERT_EQUALS(sp->getTransformation(Database::Validation).empty(), false);
    ASSERT_EQUALS(sp->getTransformation(Database::Test).empty(), false);
    ASSERT_EQUALS(sp->getOnTheFlyTransformation(Database::Learn).size(), 1U);
    ASSERT_EQUALS(sp->getOnTheFlyTransformation(Database::Validation).size(),
                  1U);
    ASSERT_EQUALS(sp->getOnTheFlyTransformation(Database::Test).size(), 1U);
    ASSERT_EQUALS(sp->getOnTheFlyTransformation(Database::Learn).empty(),
                  false);
    ASSERT_EQUALS(sp->getOnTheFlyTransformation(Database::Validation).empty(),
                  false);
    ASSERT_EQUALS(sp->getOnTheFlyTransformation(Database::Test).empty(), false);
}

TEST(StimuliProviderGenerator, StimuliProviderGenerator_bis)
{
    const std::string data = "[env]\n"
                             "SizeX=256\n"
                             "SizeY=256\n"
                             "BatchSize=3\n"
                             "\n"
                             "[env.Transformation-1]\n"
                             "Type=ChannelExtractionTransformation\n"
                             "CSChannel=Red\n"
                             "\n"
                             "[env.Transformation-2]\n"
                             "Type=FilterTransformation\n"
                             "Kernel=Gabor\n"
                             "Kernel.SizeX=5\n"
                             "Kernel.SizeY=5\n"
                             "Kernel.Theta=0.785\n";

    UnitTest::FileWriteContent("StimuliProviderGenerator_bis.in", data);

    IniParser iniConfig;
    iniConfig.load("StimuliProviderGenerator_bis.in");

    DIR_Database database;
    std::shared_ptr<StimuliProvider> sp
        = StimuliProviderGenerator::generate(database, iniConfig, "env");

    ASSERT_EQUALS(sp->getNbChannels(), 1U);
    ASSERT_EQUALS(sp->getTransformation(Database::Learn).size(), 2U);
    ASSERT_EQUALS(sp->getTransformation(Database::Validation).size(), 2U);
    ASSERT_EQUALS(sp->getTransformation(Database::Test).size(), 2U);
    ASSERT_EQUALS(sp->getTransformation(Database::Learn).empty(), false);
    ASSERT_EQUALS(sp->getTransformation(Database::Validation).empty(), false);
    ASSERT_EQUALS(sp->getTransformation(Database::Test).empty(), false);
    ASSERT_EQUALS(sp->getOnTheFlyTransformation(Database::Learn).size(), 0U);
    ASSERT_EQUALS(sp->getOnTheFlyTransformation(Database::Validation).size(),
                  0U);
    ASSERT_EQUALS(sp->getOnTheFlyTransformation(Database::Test).size(), 0U);
    ASSERT_EQUALS(sp->getOnTheFlyTransformation(Database::Learn).empty(), true);
    ASSERT_EQUALS(sp->getOnTheFlyTransformation(Database::Validation).empty(),
                  true);
    ASSERT_EQUALS(sp->getOnTheFlyTransformation(Database::Test).empty(), true);
}

TEST(StimuliProviderGenerator, StimuliProviderGenerator_applyTo)
{
    const std::string data = "[env]\n"
                             "SizeX=48\n"
                             "SizeY=48\n"
                             "BatchSize=12\n"
                             "\n"
                             "[env.Transformation-1]\n"
                             "Type=ChannelExtractionTransformation\n"
                             "CSChannel=Gray\n"
                             "\n"
                             "[env.Transformation-2]\n"
                             "Type=RescaleTransformation\n"
                             "Width=56\n"
                             "Height=56\n"
                             "\n"
                             "[env.Transformation-3]\n"
                             "Type=PadCropTransformation\n"
                             "ApplyTo=NoLearn\n"
                             "Width=48\n"
                             "Height=48\n"
                             "\n"
                             "[env.OnTheFlyTransformation-1]\n"
                             "Type=PadCropTransformation\n"
                             "ApplyTo=LearnOnly\n"
                             "Width=48\n"
                             "Height=48\n"
                             "\n"
                             "[env.OnTheFlyTransformation-2]\n"
                             "Type=FlipTransformation\n"
                             "ApplyTo=LearnOnly\n"
                             "HorizontalFlip=1\n"
                             "\n"
                             "[env.OnTheFlyTransformation-3]\n"
                             "Type=FlipTransformation\n"
                             "ApplyTo=LearnOnly\n"
                             "VerticalFlip=1\n";

    UnitTest::FileWriteContent("StimuliProviderGenerator_applyTo.in", data);

    IniParser iniConfig;
    iniConfig.load("StimuliProviderGenerator_applyTo.in");

    DIR_Database database;
    std::shared_ptr<StimuliProvider> sp
        = StimuliProviderGenerator::generate(database, iniConfig, "env");

    ASSERT_EQUALS(sp->getNbChannels(), 1U);
    ASSERT_EQUALS(sp->getTransformation(Database::Learn).size(), 2U);
    ASSERT_EQUALS((bool)std::dynamic_pointer_cast
                  <ChannelExtractionTransformation>(
                      sp->getTransformation(Database::Learn)[0]),
                  true);
    ASSERT_EQUALS((bool)std::dynamic_pointer_cast<RescaleTransformation>(
                      sp->getTransformation(Database::Learn)[1]),
                  true);
    ASSERT_EQUALS(sp->getTransformation(Database::Validation).size(), 3U);
    ASSERT_EQUALS(sp->getTransformation(Database::Test).size(), 3U);
    ASSERT_EQUALS((bool)std::dynamic_pointer_cast
                  <ChannelExtractionTransformation>(
                      sp->getTransformation(Database::Test)[0]),
                  true);
    ASSERT_EQUALS((bool)std::dynamic_pointer_cast<RescaleTransformation>(
                      sp->getTransformation(Database::Test)[1]),
                  true);
    ASSERT_EQUALS((bool)std::dynamic_pointer_cast
                  <PadCropTransformation>(
                      sp->getTransformation(Database::Test)[2]),
                  true);
    ASSERT_EQUALS(sp->getOnTheFlyTransformation(Database::Learn).size(), 3U);
    ASSERT_EQUALS(sp->getOnTheFlyTransformation(Database::Validation).size(),
                  0U);
    ASSERT_EQUALS(sp->getOnTheFlyTransformation(Database::Test).size(), 0U);
}

RUN_TESTS()
