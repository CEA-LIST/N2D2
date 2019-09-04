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

#include "Database/Actitracker_Database.hpp"
#include "utils/UnitTest.hpp"

using namespace N2D2;

TEST(Actitracker_Database, load)
{
    REQUIRED(UnitTest::DirExists(N2D2_DATA("WISDM_at_v2.0")));

    Random::mtSeed(0);

    Actitracker_Database db(0.6, 0.2, false);
    db.load(N2D2_DATA("WISDM_at_v2.0"));

    // Check default parameter values
    ASSERT_EQUALS(db.getParameter<unsigned int>("WindowSize"), 90);
    ASSERT_EQUALS(db.getParameter<double>("Overlapping"), 0.5);

    const unsigned int nbLines = 2980765;
    const unsigned int overlap
        = Utils::round(db.getParameter<unsigned int>("WindowSize")
                        * db.getParameter<double>("Overlapping"));
    const unsigned int nbSegments
        = (nbLines - db.getParameter<unsigned int>("WindowSize")) / overlap;

    ASSERT_EQUALS(db.getNbStimuli(), nbSegments);
    ASSERT_EQUALS(db.getNbLabels(), 6U);

    ASSERT_EQUALS(db.getLabelName(0), "Jogging");
    ASSERT_EQUALS(db.getNbStimuliWithLabel(0), 9752U);
    ASSERT_EQUALS(db.getLabelName(1), "LyingDown");
    ASSERT_EQUALS(db.getNbStimuliWithLabel(1), 6135U);
    ASSERT_EQUALS(db.getLabelName(2), "Sitting");
    ASSERT_EQUALS(db.getNbStimuliWithLabel(2), 14751U);
    ASSERT_EQUALS(db.getLabelName(3), "Stairs");
    ASSERT_EQUALS(db.getNbStimuliWithLabel(3), 1280U);
    ASSERT_EQUALS(db.getLabelName(4), "Standing");
    ASSERT_EQUALS(db.getNbStimuliWithLabel(4), 6416U);
    ASSERT_EQUALS(db.getLabelName(5), "Walking");
    ASSERT_EQUALS(db.getNbStimuliWithLabel(5), 27903U);
}

RUN_TESTS()
