/*
    (C) Copyright 2016 CEA LIST. All Rights Reserved.
    Contributor(s): Olivier BICHLER (olivier.bichler@cea.fr)
                    David BRIAND (david.briand@cea.fr)

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

#include "Database/KITTI_Database.hpp"
#include "utils/UnitTest.hpp"

using namespace N2D2;

TEST(KITTI_Database, load)
{

    REQUIRED(UnitTest::DirExists(N2D2_DATA("KITTI")));

    REQUIRED(UnitTest::DirExists(N2D2_DATA("training/")));
    REQUIRED(UnitTest::DirExists(N2D2_DATA("training/label_02")));
    REQUIRED(UnitTest::DirExists(N2D2_DATA("training/image_02")));

    KITTI_Database db(0.9);

    db.load(N2D2_DATA("KITTI/training/image_02"),
            N2D2_DATA("KITTI/training/label_02"));

    const unsigned int nbLearnStimuli = 7171;
    const unsigned int nbTestStimuli = 8375;
    const unsigned int nbLabels = 9;

    ASSERT_EQUALS(db.getNbStimuli(), nbLearnStimuli + nbTestStimuli);
    ASSERT_EQUALS(
        db.getNbStimuli(Database::Learn),
        (unsigned int)Utils::round(0.9 * nbLearnStimuli, Utils::HalfDown));
    ASSERT_EQUALS(db.getNbStimuli(Database::Test), nbTestStimuli);
    ASSERT_EQUALS(
        db.getNbStimuli(Database::Validation),
        (unsigned int)Utils::round(0.1 * nbLearnStimuli, Utils::HalfDown));
    ASSERT_EQUALS(db.getNbStimuli(Database::Unpartitioned), 0U);
    ASSERT_EQUALS(db.getNbLabels(), nbLabels);
}

RUN_TESTS()
