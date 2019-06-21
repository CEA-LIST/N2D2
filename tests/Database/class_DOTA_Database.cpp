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

#include "Database/DOTA_Database.hpp"
#include "utils/UnitTest.hpp"

using namespace N2D2;

TEST(DOTA_Database, loadAllStimuli)
{
    REQUIRED(UnitTest::DirExists(N2D2_DATA("DOTA/train")));

    Random::mtSeed(0);

    DOTA_Database db(0.9, false);
    db.load(N2D2_DATA("DOTA"));

    const unsigned int nbStimuli = 1411;
    const unsigned int nbStimuliVal = 458;
    const unsigned int nbLabels = 15;

    ASSERT_EQUALS(db.getNbStimuli(), nbStimuli + nbStimuliVal);
    ASSERT_EQUALS(db.getNbStimuli(Database::Learn),
                  (unsigned int)Utils::round(0.9 * nbStimuli, Utils::HalfUp));
    ASSERT_EQUALS(db.getNbStimuli(Database::Test),
                  (unsigned int)Utils::round(0.1 * nbStimuli, Utils::HalfDown));
    ASSERT_EQUALS(db.getNbStimuli(Database::Validation), nbStimuliVal);
    ASSERT_EQUALS(db.getNbStimuli(Database::Unpartitioned), 0U);
    ASSERT_EQUALS(db.getNbLabels(), nbLabels);

    ASSERT_EQUALS(db.getNbROIs(), 127843U);

    for (unsigned int label = 0; label < nbLabels; ++label) {
        std::cout << label << ": "
            << db.getLabelName(label) << " ("
            << db.getNbROIsWithLabel(label) << ")" << std::endl;
    }

    ASSERT_EQUALS(db.getLabelName(0), "plane");
    ASSERT_EQUALS(db.getNbROIsWithLabel(0), 10586U);
    ASSERT_EQUALS(db.getLabelName(1), "large-vehicle");
    ASSERT_EQUALS(db.getNbROIsWithLabel(1), 21356U);
    ASSERT_EQUALS(db.getLabelName(14), "helicopter");
    ASSERT_EQUALS(db.getNbROIsWithLabel(14), 703U);

    db.logROIsStats("DOTA_Database_ROIs_size.dat",
                    "DOTA_Database_ROIs_label.dat");
}


RUN_TESTS()
