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

#ifdef PUGIXML

#include "N2D2.hpp"

#include "Database/IMDBWIKI_Database.hpp"
#include "utils/UnitTest.hpp"

using namespace N2D2;

TEST(IMDBWIKI_Database, load)
{
    REQUIRED(UnitTest::DirExists(N2D2_DATA("IMDB-WIKI")));
    REQUIRED(UnitTest::DirExists(N2D2_DATA("IMDB-WIKI/imdb_metadata")));
    REQUIRED(UnitTest::DirExists(N2D2_DATA("IMDB-WIKI/wiki")));
    REQUIRED(UnitTest::DirExists(N2D2_DATA("IMDB-WIKI/imdb")));

    IMDBWIKI_Database db(1, 0, 0, 1.0, 0.0);
    db.load(N2D2_DATA("IMDB-WIKI"), N2D2_DATA("IMDB-WIKI/imdb_metadata"));

    const unsigned int nbStimuli = 44269;

    ASSERT_EQUALS(db.getNbStimuli(), nbStimuli);
    ASSERT_EQUALS(db.getNbStimuli(Database::Learn), nbStimuli);
    ASSERT_EQUALS(db.getNbStimuli(Database::Unpartitioned), 0U);
}

RUN_TESTS()

#else

int main()
{
    return 0;
}

#endif
