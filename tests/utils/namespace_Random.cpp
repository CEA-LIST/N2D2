/*
    (C) Copyright 2014 CEA LIST. All Rights Reserved.
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

#include "utils/Random.hpp"
#include "utils/UnitTest.hpp"

using namespace N2D2;

TEST(Random, mtRand)
{
    Random::mtSeed(1);
    const unsigned int mtRand_1[]
        = {1791095845, 4282876139, 3093770124, 4005303368, 491263,
           550290313,  1298508491, 4290846341, 630311759,  1013994432};

    for (unsigned int i = 0, size = sizeof(mtRand_1) / sizeof(mtRand_1[0]);
         i < size;
         ++i)
        ASSERT_EQUALS(Random::mtRand(), mtRand_1[i]);

    Random::mtSeed(42);
    const unsigned int mtRand_42[]
        = {1608637542, 3421126067, 4083286876, 787846414, 3143890026,
           3348747335, 2571218620, 2563451924, 670094950, 1914837113};

    for (unsigned int i = 0, size = sizeof(mtRand_42) / sizeof(mtRand_42[0]);
         i < size;
         ++i)
        ASSERT_EQUALS(Random::mtRand(), mtRand_42[i]);

    Random::mtSeed(2147483647);
    const unsigned int mtRand_2147483647[]
        = {1689602031, 3831148394, 2820341149, 2744746572, 370616153,
           3004629480, 4141996784, 3942456616, 2667712047, 1179284407};

    for (unsigned int i = 0,
                      size = sizeof(mtRand_2147483647)
                             / sizeof(mtRand_2147483647[0]);
         i < size;
         ++i)
        ASSERT_EQUALS(Random::mtRand(), mtRand_2147483647[i]);

    Random::mtSeed(0xFFFFFFFF);
    const unsigned int mtRand_0xFFFFFFFF[]
        = {419326371,  479346978,  3918654476, 2416749639, 3388880820,
           2260532800, 3350089942, 3309765114, 77050329,   1217888032};

    for (unsigned int i = 0,
                      size = sizeof(mtRand_0xFFFFFFFF)
                             / sizeof(mtRand_0xFFFFFFFF[0]);
         i < size;
         ++i)
        ASSERT_EQUALS(Random::mtRand(), mtRand_0xFFFFFFFF[i]);
}

RUN_TESTS()
