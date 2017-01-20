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

#include "Database/MNIST_IDX_Database.hpp"
#include "utils/UnitTest.hpp"

using namespace N2D2;

TEST_DATASET(MNIST_IDX_Database,
             load,
             (double validation),
             std::make_tuple(0.0),
             std::make_tuple(0.5),
             std::make_tuple(1.0))
{
    REQUIRED(UnitTest::DirExists(N2D2_DATA("mnist")));

    MNIST_IDX_Database db(validation);
    db.load(N2D2_DATA("mnist"));

    ASSERT_EQUALS(db.getNbStimuli(), 60000U + 10000U);
    ASSERT_EQUALS(db.getNbStimuli(Database::Learn),
                  (unsigned int)((1.0 - validation) * 60000));
    ASSERT_EQUALS(db.getNbStimuli(Database::Validation),
                  (unsigned int)(validation * 60000));
    ASSERT_EQUALS(db.getNbStimuli(Database::Test), 10000U);
    ASSERT_EQUALS(db.getNbStimuli(Database::Unpartitioned), 0U);
    ASSERT_EQUALS(db.getNbLabels(), 10U);
}

TEST(MNIST_IDX_Database, getStimulusData)
{
    REQUIRED(UnitTest::DirExists(N2D2_DATA("mnist")));

    MNIST_IDX_Database db;
    db.load(N2D2_DATA("mnist"));

    cv::Mat data = db.getStimulusData(Database::Learn, 0);

    ASSERT_EQUALS(data.channels(), 1);
    ASSERT_EQUALS(data.rows, 28);
    ASSERT_EQUALS(data.cols, 28);
}

TEST(MNIST_IDX_Database, getStimulusLabelsData)
{
    REQUIRED(UnitTest::DirExists(N2D2_DATA("mnist")));

    MNIST_IDX_Database db;
    db.load(N2D2_DATA("mnist"));

    cv::Mat labels = db.getStimulusLabelsData(Database::Learn, 0);

    ASSERT_EQUALS(labels.channels(), 1);
    ASSERT_EQUALS(labels.rows, 1);
    ASSERT_EQUALS(labels.cols, 1);
    ASSERT_EQUALS(labels.at<int>(0, 0), 0);
}

RUN_TESTS()
