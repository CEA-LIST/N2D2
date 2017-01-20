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

#include "Database/FDDB_Database.hpp"
#include "utils/UnitTest.hpp"

using namespace N2D2;

TEST_DATASET(FDDB_Database,
             load,
             (double learn, double validation),
             std::make_tuple(0.0, 0.0),
             std::make_tuple(1.0, 0.0),
             std::make_tuple(0.0, 1.0),
             std::make_tuple(0.6, 0.1),
             std::make_tuple(0.1, 0.4))
{
    REQUIRED(UnitTest::DirExists(N2D2_DATA("FDDB")));

    FDDB_Database db(learn, validation);
    db.load(N2D2_DATA("FDDB"));

    const unsigned int nbStimuli = 2845;

    ASSERT_EQUALS(db.getNbStimuli(), nbStimuli);

    if (learn == 0.0 && validation == 0.0) {
        ASSERT_EQUALS(db.getNbStimuli(Database::Test), nbStimuli);
    }

    ASSERT_EQUALS(db.getNbStimuli(Database::Unpartitioned), 0U);
    ASSERT_EQUALS(db.getNbStimuli(Database::Learn)
                  + db.getNbStimuli(Database::Validation)
                  + db.getNbStimuli(Database::Test),
                  nbStimuli);
    ASSERT_EQUALS(db.getNbLabels(), 1U);
    ASSERT_EQUALS(db.getLabelName(0), "1");
    ASSERT_EQUALS(db.getNbROIs(), 5171U);
    ASSERT_EQUALS(db.getNbROIsWithLabel(0), 5171U);
}

RUN_TESTS()
