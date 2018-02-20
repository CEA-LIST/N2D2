/*
    (C) Copyright 2018 CEA LIST. All Rights Reserved.
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

#ifdef JSONCPP

#include "N2D2.hpp"

#include "Database/Cityscapes_Database.hpp"
#include "utils/Utils.hpp"
#include "utils/UnitTest.hpp"

using namespace N2D2;

TEST(Cityscapes_Database, load)
{
    REQUIRED(UnitTest::DirExists(N2D2_DATA("Cityscapes")));
    REQUIRED(UnitTest::DirExists(N2D2_DATA("Cityscapes/gtFine")));
    REQUIRED(UnitTest::DirExists(N2D2_DATA("Cityscapes/leftImg8bit")));

    Cityscapes_Database db;
    db.setParameter("RandomPartitioning", false);
    db.load(N2D2_DATA("Cityscapes/leftImg8bit"));

    ASSERT_EQUALS(db.getNbStimuli(), 5000U);
    ASSERT_EQUALS(db.getNbStimuli(Database::Learn), 2975U);
    ASSERT_EQUALS(db.getNbStimuli(Database::Validation), 500U);
    ASSERT_EQUALS(db.getNbStimuli(Database::Test), 1525U);
    ASSERT_EQUALS(db.getNbStimuli(Database::Unpartitioned), 0U);

    ASSERT_EQUALS(db.getNbLabels(), 35U);
    ASSERT_EQUALS(db.getLabelName(0), "unlabeled");
    ASSERT_EQUALS(db.getLabelName(34), "license plate");
    ASSERT_EQUALS(Utils::baseName(db.getStimulusName(Database::Learn, 0)),
                  "aachen_000000_000019_leftImg8bit.png");
    ASSERT_EQUALS(db.getStimulusROIs(Database::Learn, 0).size(), 80U);
    ASSERT_EQUALS(db.getStimulusROIs(Database::Learn, 0)[0]->getLabel(),
                  db.getLabelID("road"));
    ASSERT_EQUALS(db.getStimulusROIs(Database::Learn, 0)[79]->getLabel(),
                  db.getLabelID("out of roi"));
}

RUN_TESTS()

#else

int main()
{
    return 0;
}

#endif
