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

#include "Database/MNIST_IDX_Database.hpp"
#include "Environment.hpp"
#include "N2D2.hpp"
#include "Network.hpp"
#include "Transformation/RescaleTransformation.hpp"
#include "utils/UnitTest.hpp"

using namespace N2D2;

TEST_DATASET(Environment,
             Environment,
             (unsigned int x, unsigned int y),
             std::make_tuple(1U, 3U),
             std::make_tuple(3U, 1U),
             std::make_tuple(3U, 3U))
{
    Network net;
    Environment env(net, EmptyDatabase, x, y);

    ASSERT_EQUALS(env.getSizeX(), x);
    ASSERT_EQUALS(env.getSizeY(), y);
    ASSERT_EQUALS(env.getBatchSize(), 1U);
    ASSERT_EQUALS(env.getNodes().size(), x * y);
}

TEST_DATASET(Environment,
             readRandomBatch,
             (unsigned int channelsWidth, unsigned int channelsHeight),
             std::make_tuple(1U, 1U),
             std::make_tuple(1U, 2U),
             std::make_tuple(2U, 1U),
             std::make_tuple(3U, 3U),
             std::make_tuple(10U, 10U),
             std::make_tuple(25U, 25U),
             std::make_tuple(25U, 30U),
             std::make_tuple(30U, 25U),
             std::make_tuple(30U, 30U))
{
    REQUIRED(UnitTest::DirExists(N2D2_DATA("mnist")));

    Network net;

    MNIST_IDX_Database database;
    database.load(N2D2_DATA("mnist"));

    Environment env(net, database, channelsWidth, channelsHeight, 1, 2, false);
    env.addTransformation(RescaleTransformation(channelsWidth, channelsHeight));
    env.setCachePath();

    env.readRandomBatch(Database::Test);
}

RUN_TESTS()
