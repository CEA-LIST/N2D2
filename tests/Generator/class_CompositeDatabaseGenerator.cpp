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

#include "N2D2.hpp"

#include "Generator/DatabaseGenerator.hpp"
#include "utils/UnitTest.hpp"

using namespace N2D2;

TEST(CompositeDatabaseGenerator, generate)
{
    REQUIRED(UnitTest::DirExists(N2D2_DATA("CelebA")));
    REQUIRED(UnitTest::DirExists(N2D2_DATA("CelebA/Anno")));
    REQUIRED(UnitTest::DirExists(N2D2_DATA("CelebA/Eval")));
    REQUIRED(UnitTest::DirExists(N2D2_DATA("CelebA/Img")));
    REQUIRED(UnitTest::DirExists(N2D2_DATA("lfw")));

    Random::mtSeed(0);

    const std::string data = "[database]\n"
        "Type=CompositeDatabase\n"
        "Databases=database.celebA database.lfw\n"
        "\n"
        "[database.celebA]\n"
        "Type=CelebA_Database\n"
        "InTheWild=0\n"
        "WithPartitioning=0\n"
        "Learn=1.0\n"
        "\n"
        "[database.lfw]\n"
        "Type=DIR_Database\n"
        "DataPath=${N2D2_DATA}/lfw\n"
        "EquivLabelPartitioning=0\n"
        "Learn=0.0\n"
        "Validation=1.0\n";

    UnitTest::FileWriteContent("CompositeDatabaseGenerator.in", data);

    IniParser iniConfig;
    iniConfig.load("CompositeDatabaseGenerator.in");

    std::shared_ptr<Database> db = DatabaseGenerator::generate(iniConfig,
        "database");

    const unsigned int nbStimuli_CelebA = 202599;
    const unsigned int nbIdentities_CelebA = 10177;

    const unsigned int nbStimuli_LFW = 13233;
    const unsigned int nbIdentities_LFW = 5749;

    ASSERT_EQUALS(db->getNbStimuli(), nbStimuli_CelebA + nbStimuli_LFW);
    ASSERT_EQUALS(db->getNbROIs(), 0U);
    ASSERT_EQUALS(db->getNbLabels(), nbIdentities_CelebA + nbIdentities_LFW);

    ASSERT_EQUALS(db->getNbStimuli(Database::Learn), nbStimuli_CelebA);
    ASSERT_EQUALS(db->getNbStimuli(Database::Validation), nbStimuli_LFW);
    ASSERT_EQUALS(db->getNbStimuli(Database::Unpartitioned), 0U);

    // CelebA
    ASSERT_EQUALS(db->getLabelID("1"), 0);
    ASSERT_EQUALS(db->getLabelID("2"), 1);
    ASSERT_EQUALS(db->getLabelID("3"), 2);
    ASSERT_EQUALS(db->getLabelID(std::to_string(nbIdentities_CelebA)), nbIdentities_CelebA - 1);

    ASSERT_EQUALS(Utils::baseName(db->getStimulusName(0)), "000001.jpg");
    ASSERT_EQUALS(db->getStimulusLabel(0), db->getLabelID("2880"));
    ASSERT_EQUALS(Utils::baseName(db->getStimulusName(1)), "000002.jpg");
    ASSERT_EQUALS(db->getStimulusLabel(1), db->getLabelID("2937"));
    ASSERT_EQUALS(Utils::baseName(db->getStimulusName(2)), "000003.jpg");
    ASSERT_EQUALS(db->getStimulusLabel(2), db->getLabelID("8692"));
    ASSERT_EQUALS(Utils::baseName(db->getStimulusName(nbStimuli_CelebA - 1)), "202599.jpg");
    ASSERT_EQUALS(db->getStimulusLabel(nbStimuli_CelebA - 1), db->getLabelID("10101"));

    // LFW
    ASSERT_EQUALS(db->getLabelID("/AJ_Cook"), nbIdentities_CelebA + 0);
    ASSERT_EQUALS(db->getLabelID("/AJ_Lamas"), nbIdentities_CelebA + 1);
    ASSERT_EQUALS(db->getLabelID("/Aaron_Eckhart"), nbIdentities_CelebA + 2);
    ASSERT_EQUALS(db->getLabelID("/Zydrunas_Ilgauskas"), nbIdentities_CelebA + nbIdentities_LFW - 1);

    ASSERT_EQUALS(Utils::baseName(db->getStimulusName(nbStimuli_CelebA + 0)), "AJ_Cook_0001.jpg");
    ASSERT_EQUALS(db->getStimulusLabel(nbStimuli_CelebA + 0), db->getLabelID("/AJ_Cook"));
    ASSERT_EQUALS(Utils::baseName(db->getStimulusName(nbStimuli_CelebA + 1)), "AJ_Lamas_0001.jpg");
    ASSERT_EQUALS(db->getStimulusLabel(nbStimuli_CelebA + 1), db->getLabelID("/AJ_Lamas"));
    ASSERT_EQUALS(Utils::baseName(db->getStimulusName(nbStimuli_CelebA + 2)), "Aaron_Eckhart_0001.jpg");
    ASSERT_EQUALS(db->getStimulusLabel(nbStimuli_CelebA + 2), db->getLabelID("/Aaron_Eckhart"));
    ASSERT_EQUALS(Utils::baseName(db->getStimulusName(nbStimuli_CelebA + nbStimuli_LFW - 1)), "Zydrunas_Ilgauskas_0001.jpg");
    ASSERT_EQUALS(db->getStimulusLabel(nbStimuli_CelebA + nbStimuli_LFW - 1), db->getLabelID("/Zydrunas_Ilgauskas"));
}

RUN_TESTS()
