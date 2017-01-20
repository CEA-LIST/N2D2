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

#include "N2D2.hpp"

#include "controler/Interface.hpp"
#include "utils/UnitTest.hpp"

using namespace N2D2;

TEST(Interface, Interface)
{
    const Interface<double> interface;

    ASSERT_EQUALS(interface.dimZ(), 0U);
    ASSERT_EQUALS(interface.dimB(), 0U);
}

TEST_DATASET(Interface,
             push_back,
             (unsigned int dimX, unsigned int dimY),
             std::make_tuple(0U, 0U),
             std::make_tuple(1U, 0U),
             std::make_tuple(0U, 1U),
             std::make_tuple(3U, 1U),
             std::make_tuple(3U, 3U),
             std::make_tuple(12U, 34U),
             std::make_tuple(34U, 12U))
{
    Tensor4d<int> A(dimX, dimY, 1, 1);
    Interface<int> interface;

    ASSERT_EQUALS(interface.dimZ(), 0U);
    ASSERT_EQUALS(interface.dimB(), 0U);

    interface.push_back(&A);

    ASSERT_EQUALS(interface[0].dimX(), dimX);
    ASSERT_EQUALS(interface[0].dimY(), dimY);
    ASSERT_EQUALS(interface.dimZ(), 1U);
    ASSERT_EQUALS(interface.dimB(), 1U);
}

TEST_DATASET(Interface,
             push_back_bis,
             (unsigned int dimX,
              unsigned int dimY,
              unsigned int dimZ1,
              unsigned int dimZ2,
              unsigned int dimB),
             std::make_tuple(1U, 1U, 1U, 1U, 1U),
             std::make_tuple(2U, 2U, 1U, 1U, 1U),
             std::make_tuple(4U, 3U, 1U, 1U, 1U),
             std::make_tuple(3U, 4U, 1U, 1U, 1U),
             std::make_tuple(4U, 3U, 1U, 1U, 2U),
             std::make_tuple(3U, 4U, 1U, 1U, 2U),
             std::make_tuple(4U, 3U, 2U, 1U, 1U),
             std::make_tuple(4U, 3U, 1U, 2U, 1U),
             std::make_tuple(4U, 3U, 3U, 5U, 10U))
{
    Interface<int> interface;
    Tensor4d<int> A1(dimX, dimY, dimZ1, dimB);
    Tensor4d<int> A2(dimX, dimY, dimZ2, dimB);

    interface.push_back(&A1);
    interface.push_back(&A2);

    ASSERT_EQUALS(interface[0].dimX(), dimX);
    ASSERT_EQUALS(interface[0].dimY(), dimY);
    ASSERT_EQUALS(interface[1].dimX(), dimX);
    ASSERT_EQUALS(interface[1].dimY(), dimY);
    ASSERT_EQUALS(interface.dimZ(), dimZ1 + dimZ2);
    ASSERT_EQUALS(interface.dimB(), dimB);
}

TEST_DATASET(Interface,
             push_back__throw,
             (unsigned int dimX1,
              unsigned int dimY1,
              unsigned int dimZ1,
              unsigned int dimB1,
              unsigned int dimX2,
              unsigned int dimY2,
              unsigned int dimZ2,
              unsigned int dimB2),
             std::make_tuple(2U, 2U, 1U, 2U, 2U, 1U, 1U, 1U),
             std::make_tuple(3U, 4U, 1U, 1U, 3U, 4U, 1U, 5U))
{
    Interface<int> interface;
    Tensor4d<int> A1(dimX1, dimY1, dimZ1, dimB1);
    Tensor4d<int> A2(dimX2, dimY2, dimZ2, dimB2);

    interface.push_back(&A1);
    ASSERT_THROW_ANY(interface.push_back(&A2));
}

TEST_DATASET(Interface,
             function_call_operator,
             (unsigned int dimX,
              unsigned int dimY,
              unsigned int dimZ1,
              unsigned int dimZ2,
              unsigned int dimB),
             std::make_tuple(1U, 1U, 1U, 1U, 1U),
             std::make_tuple(2U, 2U, 1U, 1U, 1U),
             std::make_tuple(4U, 3U, 1U, 1U, 1U),
             std::make_tuple(3U, 4U, 1U, 1U, 1U),
             std::make_tuple(4U, 3U, 1U, 1U, 2U),
             std::make_tuple(3U, 4U, 1U, 1U, 2U),
             std::make_tuple(4U, 3U, 2U, 1U, 1U),
             std::make_tuple(4U, 3U, 1U, 2U, 1U),
             std::make_tuple(4U, 3U, 3U, 5U, 10U))
{
    Random::mtSeed(0);

    Interface<int> interface;
    Tensor4d<int> A1(dimX, dimY, dimZ1, dimB);
    Tensor4d<int> A2(dimX, dimY, dimZ2, dimB);

    for (unsigned int b = 0; b < dimB; ++b) {
        for (unsigned int k = 0; k < dimZ1 + dimZ2; ++k) {
            for (unsigned int i = 0; i < dimX; ++i) {
                for (unsigned int j = 0; j < dimY; ++j) {
                    const int value = Random::randUniform(-10000, 10000);

                    if (k < dimZ1)
                        A1(i, j, k, b) = value;
                    else
                        A2(i, j, k - dimZ1, b) = value;
                }
            }
        }
    }

    interface.push_back(&A1);
    interface.push_back(&A2);

    ASSERT_EQUALS(interface[0].dimX(), dimX);
    ASSERT_EQUALS(interface[0].dimY(), dimY);
    ASSERT_EQUALS(interface[1].dimX(), dimX);
    ASSERT_EQUALS(interface[1].dimY(), dimY);
    ASSERT_EQUALS(interface.dimZ(), dimZ1 + dimZ2);
    ASSERT_EQUALS(interface.dimB(), dimB);

    for (unsigned int b = 0; b < dimB; ++b) {
        for (unsigned int k = 0; k < dimZ1 + dimZ2; ++k) {
            for (unsigned int i = 0; i < dimX; ++i) {
                for (unsigned int j = 0; j < dimY; ++j) {
                    const int value = (k < dimZ1) ? A1(i, j, k, b)
                                                  : A2(i, j, k - dimZ1, b);

                    ASSERT_EQUALS(interface(i, j, k, b), value);
                }
            }
        }
    }
}

TEST_DATASET(Interface,
             function_call_operator_bis,
             (unsigned int dimX,
              unsigned int dimY,
              unsigned int dimZ1,
              unsigned int dimZ2,
              unsigned int dimB),
             std::make_tuple(1U, 1U, 1U, 1U, 1U),
             std::make_tuple(2U, 2U, 1U, 1U, 1U),
             std::make_tuple(4U, 3U, 1U, 1U, 1U),
             std::make_tuple(3U, 4U, 1U, 1U, 1U),
             std::make_tuple(4U, 3U, 1U, 1U, 2U),
             std::make_tuple(3U, 4U, 1U, 1U, 2U),
             std::make_tuple(4U, 3U, 2U, 1U, 1U),
             std::make_tuple(4U, 3U, 1U, 2U, 1U),
             std::make_tuple(4U, 3U, 3U, 5U, 10U))
{
    Random::mtSeed(0);

    Interface<int> interface;
    Tensor4d<int> A1(dimX, dimY, dimZ1, dimB, 0);
    Tensor4d<int> A2(dimX, dimY, dimZ2, dimB, 23);

    interface.push_back(&A1);
    interface.push_back(&A2);

    ASSERT_EQUALS(interface[0].dimX(), dimX);
    ASSERT_EQUALS(interface[0].dimY(), dimY);
    ASSERT_EQUALS(interface[1].dimX(), dimX);
    ASSERT_EQUALS(interface[1].dimY(), dimY);
    ASSERT_EQUALS(interface.dimZ(), dimZ1 + dimZ2);
    ASSERT_EQUALS(interface.dimB(), dimB);

    for (unsigned int b = 0; b < dimB; ++b) {
        for (unsigned int k = 0; k < dimZ1 + dimZ2; ++k) {
            for (unsigned int i = 0; i < dimX; ++i) {
                for (unsigned int j = 0; j < dimY; ++j)
                    interface(i, j, k, b) = Random::randUniform(-10000, 10000);
            }
        }
    }

    for (unsigned int b = 0; b < dimB; ++b) {
        for (unsigned int k = 0; k < dimZ1 + dimZ2; ++k) {
            for (unsigned int i = 0; i < dimX; ++i) {
                for (unsigned int j = 0; j < dimY; ++j) {
                    if (k < dimZ1) {
                        ASSERT_EQUALS(A1(i, j, k, b), interface(i, j, k, b));
                    } else {
                        ASSERT_EQUALS(A2(i, j, k - dimZ1, b),
                                      interface(i, j, k, b));
                    }
                }
            }
        }
    }
}

TEST_DATASET(Interface,
             fill,
             (int value),
             std::make_tuple(0),
             std::make_tuple(-10),
             std::make_tuple(125))
{
    Interface<int> interface;
    Tensor4d<int> A1(8, 8, 3, 4);
    Tensor4d<int> A2(8, 8, 5, 4);

    interface.push_back(&A1);
    interface.push_back(&A2);

    interface.fill(value);

    for (unsigned int b = 0; b < 4; ++b) {
        for (unsigned int k = 0; k < 3 + 5; ++k) {
            for (unsigned int i = 0; i < 8; ++i) {
                for (unsigned int j = 0; j < 8; ++j) {
                    ASSERT_EQUALS(interface(i, j, k, b), value);

                    if (k < 3) {
                        ASSERT_EQUALS(A1(i, j, k, b), value);
                    } else {
                        ASSERT_EQUALS(A2(i, j, k - 3, b), value);
                    }
                }
            }
        }
    }
}

TEST(Interface, clear)
{
    Interface<int> interface;
    Tensor4d<int> A1(1, 2, 3, 4);
    Tensor4d<int> A2(1, 2, 5, 4);

    interface.push_back(&A1);
    interface.push_back(&A2);

    interface.clear();

    ASSERT_EQUALS(interface.dimZ(), 0U);
    ASSERT_EQUALS(interface.dimB(), 0U);
    ASSERT_EQUALS(interface.dataSize(), 0U);
    ASSERT_TRUE(interface.empty());
}

RUN_TESTS()
