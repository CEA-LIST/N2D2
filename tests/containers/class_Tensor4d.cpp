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

#include "containers/Tensor3d.hpp"
#include "containers/Tensor4d.hpp"
#include "utils/UnitTest.hpp"

using namespace N2D2;

TEST(Tensor4d, Tensor4d)
{
    const Tensor4d<double> A;

    ASSERT_EQUALS(A.dimX(), 0U);
    ASSERT_EQUALS(A.dimY(), 0U);
    ASSERT_EQUALS(A.dimZ(), 0U);
    ASSERT_EQUALS(A.dimB(), 0U);
    ASSERT_EQUALS(A.size(), 0U);
    ASSERT_TRUE(A.empty());
}

TEST_DATASET(Tensor4d,
             Tensor4d__dimX_dimY_dimZ_dimB,
             (unsigned int dimX,
              unsigned int dimY,
              unsigned int dimZ,
              unsigned int dimB),
             std::make_tuple(0U, 3U, 1U, 1U),
             std::make_tuple(3U, 0U, 1U, 2U),
             std::make_tuple(1U, 3U, 2U, 2U),
             std::make_tuple(3U, 1U, 1U, 3U),
             std::make_tuple(3U, 3U, 4U, 10U),
             std::make_tuple(12U, 34U, 1U, 1U),
             std::make_tuple(34U, 12U, 5U, 21U))
{
    Tensor4d<int> tensor(dimX, dimY, dimZ, dimB);

    ASSERT_EQUALS(tensor.dimX(), dimX);
    ASSERT_EQUALS(tensor.dimY(), dimY);
    ASSERT_EQUALS(tensor.dimZ(), dimZ);
    ASSERT_EQUALS(tensor.dimB(), dimB);
    ASSERT_EQUALS(tensor.size(), dimX * dimY * dimZ * dimB);
}

TEST(Tensor4d, push_back)
{
    Tensor3d<int> A;
    Tensor4d<int> tensor;

    ASSERT_EQUALS(tensor.dimB(), 0U);
    ASSERT_EQUALS(tensor.size(), 0U);
    ASSERT_TRUE(tensor.empty());

    tensor.push_back(A);

    ASSERT_EQUALS(tensor.dimB(), 1U);
    ASSERT_EQUALS(tensor.size(), 0U);
    ASSERT_TRUE(tensor.empty());
}

TEST_DATASET(Tensor4d,
             dimB,
             (unsigned int dimX,
              unsigned int dimY,
              unsigned int dimZ,
              unsigned int dimB),
             std::make_tuple(0U, 3U, 1U, 1U),
             std::make_tuple(3U, 0U, 1U, 2U),
             std::make_tuple(1U, 3U, 2U, 2U),
             std::make_tuple(3U, 1U, 1U, 3U),
             std::make_tuple(3U, 3U, 4U, 10U),
             std::make_tuple(12U, 34U, 1U, 1U),
             std::make_tuple(34U, 12U, 5U, 21U))
{
    Tensor3d<int> A(dimX, dimY, dimZ);
    Tensor4d<int> tensor;

    for (unsigned int i = 0; i < dimB; ++i)
        tensor.push_back(A);

    ASSERT_EQUALS(tensor.dimB(), dimB);
    ASSERT_EQUALS(tensor.size(), dimX * dimY * dimZ * dimB);
}

TEST_DATASET(Tensor4d,
             fill,
             (int value),
             std::make_tuple(0),
             std::make_tuple(-10),
             std::make_tuple(125))
{
    Tensor4d<int> A1(8, 8, 3, 4);
    A1.fill(value);

    for (unsigned int b = 0; b < 4; ++b) {
        for (unsigned int k = 0; k < 3; ++k) {
            for (unsigned int i = 0; i < 8; ++i) {
                for (unsigned int j = 0; j < 8; ++j) {
                    ASSERT_EQUALS(A1(i, j, k, b), value);
                }
            }
        }
    }
}

TEST(Tensor4d, clear)
{
    Tensor4d<double> A(2, 3, 4, 5, 1.0);

    ASSERT_EQUALS(A(1, 1, 1, 1), 1.0);

    A.clear();

    ASSERT_EQUALS(A.dimX(), 0U);
    ASSERT_EQUALS(A.dimY(), 0U);
    ASSERT_EQUALS(A.dimZ(), 0U);
    ASSERT_EQUALS(A.dimB(), 0U);
    ASSERT_EQUALS(A.size(), 0U);
    ASSERT_TRUE(A.empty());
}

RUN_TESTS()
