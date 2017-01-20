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

#include "Database/CaltechPedestrian_Database.hpp"
#include "utils/UnitTest.hpp"

using namespace N2D2;

TEST_DATASET(CaltechPedestrian_Database,
             load,
             (double validation, bool singleLabel, bool incAmbiguous),
             std::make_tuple(0.0, true, false),
             std::make_tuple(0.5, true, false),
             std::make_tuple(1.0, true, false),
             std::make_tuple(0.0, false, false),
             std::make_tuple(0.0, true, true),
             std::make_tuple(0.0, false, true))
{
    REQUIRED(
        UnitTest::DirExists(N2D2_DATA("CaltechPedestrians/data-USA/images")));

    CaltechPedestrian_Database db(validation, singleLabel, incAmbiguous);
    db.load(N2D2_DATA("CaltechPedestrians/data-USA/images"),
            N2D2_DATA("CaltechPedestrians/data-USA/annotations"));

    const unsigned int nbStimuli = 25693U + 10864U + 22239U + 23944U + 21995U
                                   + 23684U;
    const unsigned int nbStimuliTest = 34855U + 22525U + 19822U + 22274U
                                       + 21989U;

    ASSERT_EQUALS(db.getNbStimuli(), nbStimuli + nbStimuliTest);
    ASSERT_EQUALS(db.getNbStimuli(Database::Learn),
                  (unsigned int)Utils::round((1.0 - validation) * nbStimuli,
                                             Utils::HalfUp));
    ASSERT_EQUALS(
        db.getNbStimuli(Database::Validation),
        (unsigned int)Utils::round(validation * nbStimuli, Utils::HalfDown));
    ASSERT_EQUALS(db.getNbStimuli(Database::Test), nbStimuliTest);
    ASSERT_EQUALS(db.getNbStimuli(Database::Unpartitioned), 0U);

    const unsigned int nbAmbiguousROIs = 7793;
    const unsigned int nbPersonROIs = 285558;
    const unsigned int nbPeopleROIs = 49433;

    if (!incAmbiguous)
        ASSERT_EQUALS(db.getNbROIsWithLabel(-1), nbAmbiguousROIs);

    if (singleLabel) {
        ASSERT_EQUALS(db.getNbLabels(), 1U);
        ASSERT_EQUALS(db.getNbROIsWithLabel("person"),
                      nbPersonROIs + nbPeopleROIs
                      + ((incAmbiguous) ? nbAmbiguousROIs : 0));
    } else {
        ASSERT_EQUALS(db.getNbLabels(), 2U);
        ASSERT_EQUALS(db.getNbROIsWithLabel("person"),
                      nbPersonROIs + ((incAmbiguous) ? nbAmbiguousROIs : 0));
        ASSERT_EQUALS(db.getNbROIsWithLabel("people"), nbPeopleROIs);
    }

    ASSERT_EQUALS(db.getLabelName(0), "person");

    if (!singleLabel) {
        ASSERT_EQUALS(db.getLabelName(1), "people");
    }
}

RUN_TESTS()
