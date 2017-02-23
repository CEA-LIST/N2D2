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

#include "Database/ILSVRC2012_Database.hpp"
#include "utils/UnitTest.hpp"

using namespace N2D2;

TEST(ILSVRC2012_Database, loadAllStimuli)
{
    REQUIRED(UnitTest::DirExists(N2D2_DATA("ILSVRC2012/train")));

    ILSVRC2012_Database db(0.9, false);
    db.load(N2D2_DATA("ILSVRC2012"), N2D2_DATA("ILSVRC2012/synsets.txt"));

    const unsigned int nbStimuli = 1281167;
    const unsigned int nbStimuliVal = 50000;
    const unsigned int nbLabels = 1000;

    ASSERT_EQUALS(db.getNbStimuli(), nbStimuli + nbStimuliVal);
    ASSERT_EQUALS(db.getNbStimuli(Database::Learn),
                  (unsigned int)Utils::round(0.9 * nbStimuli, Utils::HalfUp));
    ASSERT_EQUALS(db.getNbStimuli(Database::Test),
                  (unsigned int)Utils::round(0.1 * nbStimuli, Utils::HalfDown));
    ASSERT_EQUALS(db.getNbStimuli(Database::Validation), nbStimuliVal);
    ASSERT_EQUALS(db.getNbStimuli(Database::Unpartitioned), 0U);
    ASSERT_EQUALS(db.getNbLabels(), nbLabels);

    ASSERT_EQUALS(db.getLabelName(0), "n01440764");
    ASSERT_EQUALS(db.getNbStimuliWithLabel(0), 1350U);
    ASSERT_EQUALS(db.getLabelName(1), "n01443537");
    ASSERT_EQUALS(db.getNbStimuliWithLabel(1), 1350U);
    ASSERT_EQUALS(db.getLabelName(811), "n04265275");
    ASSERT_EQUALS(db.getNbStimuliWithLabel(811), 1054U);
    ASSERT_EQUALS(db.getLabelName(998), "n13133613");
    ASSERT_EQUALS(db.getNbStimuliWithLabel(998), 1350U);
    ASSERT_EQUALS(db.getLabelName(999), "n15075141");
    ASSERT_EQUALS(db.getNbStimuliWithLabel(999), 1350U);
}


RUN_TESTS()
