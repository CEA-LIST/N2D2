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

#include "Database/DIR_Database.hpp"
#include "utils/UnitTest.hpp"

using namespace N2D2;

TEST(DIR_Database, load)
{
    REQUIRED(UnitTest::DirExists(N2D2_DATA("lfw")));

    Random::mtSeed(0);

    DIR_Database db;
    db.load(N2D2_DATA("lfw"));

    const unsigned int nbStimuli = 13233;
    const unsigned int nbIdentities = 5749;

    ASSERT_EQUALS(db.getNbStimuli(), nbStimuli);
    ASSERT_EQUALS(db.getNbROIs(), 0U);
    ASSERT_EQUALS(db.getNbLabels(), nbIdentities);

    ASSERT_EQUALS(db.getNbStimuli(Database::Unpartitioned), nbStimuli);

    ASSERT_EQUALS(db.getLabelID("/AJ_Cook"), 0);
    ASSERT_EQUALS(db.getLabelID("/AJ_Lamas"), 1);
    ASSERT_EQUALS(db.getLabelID("/Aaron_Eckhart"), 2);
    ASSERT_EQUALS(db.getLabelID("/Zydrunas_Ilgauskas"), nbIdentities - 1);

    ASSERT_EQUALS(Utils::baseName(db.getStimulusName(0)), "AJ_Cook_0001.jpg");
    ASSERT_EQUALS(db.getStimulusLabel(0), db.getLabelID("/AJ_Cook"));
    ASSERT_EQUALS(Utils::baseName(db.getStimulusName(1)), "AJ_Lamas_0001.jpg");
    ASSERT_EQUALS(db.getStimulusLabel(1), db.getLabelID("/AJ_Lamas"));
    ASSERT_EQUALS(Utils::baseName(db.getStimulusName(2)), "Aaron_Eckhart_0001.jpg");
    ASSERT_EQUALS(db.getStimulusLabel(2), db.getLabelID("/Aaron_Eckhart"));
    ASSERT_EQUALS(Utils::baseName(db.getStimulusName(nbStimuli - 1)), "Zydrunas_Ilgauskas_0001.jpg");
    ASSERT_EQUALS(db.getStimulusLabel(nbStimuli - 1), db.getLabelID("/Zydrunas_Ilgauskas"));
}

RUN_TESTS()
