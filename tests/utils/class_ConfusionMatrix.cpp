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

#include "utils/ConfusionMatrix.hpp"
#include "utils/UnitTest.hpp"
#include "utils/Utils.hpp"

using namespace N2D2;

TEST(ConfusionMatrix, ConfusionTable)
{
    ConfusionMatrix<unsigned int> conf(3, 3);
    conf << "5 3 0 "
            "2 3 1 "
            "0 2 11";

    ConfusionTable<unsigned int> confTable0 = conf.getConfusionTable(0);

    ASSERT_EQUALS(confTable0.tp(), 5);
    ASSERT_EQUALS(confTable0.fp(), 2);
    ASSERT_EQUALS(confTable0.fn(), 3);
    ASSERT_EQUALS(confTable0.tn(), 17);
    ASSERT_EQUALS_DELTA(confTable0.sensitivity(),
        confTable0.tp() / (double)(confTable0.tp() + confTable0.fn()), 1e-12);
    ASSERT_EQUALS_DELTA(confTable0.specificity(),
        confTable0.tn() / (double)(confTable0.tn() + confTable0.fp()), 1e-12);
    ASSERT_EQUALS_DELTA(confTable0.precision(),
        confTable0.tp() / (double)(confTable0.tp() + confTable0.fp()), 1e-12);
    ASSERT_EQUALS_DELTA(confTable0.negativePredictiveValue(),
        confTable0.tn() / (double)(confTable0.tn() + confTable0.fn()), 1e-12);

    ConfusionTable<unsigned int> confTable1 = conf.getConfusionTable(1);

    ASSERT_EQUALS(confTable1.tp(), 3);
    ASSERT_EQUALS(confTable1.fp(), 5);
    ASSERT_EQUALS(confTable1.fn(), 3);
    ASSERT_EQUALS(confTable1.tn(), 16);
}

RUN_TESTS()
