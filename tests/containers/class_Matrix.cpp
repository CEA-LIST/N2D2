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

#include "containers/Matrix.hpp"
#include "utils/UnitTest.hpp"

using namespace N2D2;

TEST(Matrix, Matrix)
{
    const Matrix<double> A;

    ASSERT_EQUALS(A.rows(), 0U);
    ASSERT_EQUALS(A.cols(), 0U);
    ASSERT_EQUALS(A.size(), 0U);
    ASSERT_TRUE(A.empty());
}

TEST_DATASET(Matrix,
             Matrix__rows_cols,
             (unsigned int rows, unsigned int cols),
             std::make_tuple(0U, 3U),
             std::make_tuple(3U, 0U),
             std::make_tuple(1U, 3U),
             std::make_tuple(3U, 1U),
             std::make_tuple(3U, 3U),
             std::make_tuple(12U, 34U),
             std::make_tuple(34U, 12U))
{
    const Matrix<double> A(rows, cols);

    ASSERT_EQUALS(A.rows(), rows);
    ASSERT_EQUALS(A.cols(), cols);
    ASSERT_EQUALS(A.size(), rows * cols);
    ASSERT_TRUE(A.empty() == (rows * cols == 0));
}

TEST_DATASET(Matrix,
             left_shift_operator,
             (unsigned int rows, unsigned int cols, std::string values),
             std::make_tuple(1U, 3U, "1 2 3"),
             std::make_tuple(3U, 1U, "1 2 3"),
             std::make_tuple(3U, 3U, "1 2 3 4 5 6 7 8 9"),
             std::make_tuple(2U, 3U, "1 2 3 4 5 6"),
             std::make_tuple(3U, 2U, "1 2 3 4 5 6"))
{
    Matrix<double> A(rows, cols);
    A << values;

    ASSERT_EQUALS(A(0, 0), 1);
    ASSERT_EQUALS(A(0), 1);
    ASSERT_EQUALS(A(rows - 1, cols - 1), rows * cols);
    ASSERT_EQUALS(A(rows * cols - 1), rows * cols);
}

TEST(Matrix, left_shift_operator__throw)
{
    Matrix<double> A(2, 2);
    A << "1 2 3 4 ";
    A << " 1 2 3 4";
    A << " 1   2   3   4";

    ASSERT_THROW(A << "1 2 3", std::runtime_error);
    ASSERT_THROW(A << "1 2 3 ", std::runtime_error);
    ASSERT_THROW(A << "1 2 3 4 5", std::runtime_error);
    ASSERT_THROW(A << "1 a 3 4", std::runtime_error);
}

TEST(Matrix, clear)
{
    Matrix<double> A(2, 2, 1.0);

    ASSERT_EQUALS(A(1, 1), 1.0);

    A.clear();

    ASSERT_EQUALS(A.rows(), 0U);
    ASSERT_EQUALS(A.cols(), 0U);
    ASSERT_EQUALS(A.size(), 0U);
    ASSERT_TRUE(A.empty());
}

TEST_DATASET(Matrix,
             block,
             (unsigned int row,
              unsigned int col,
              unsigned int nbRows,
              unsigned int nbCols),
             std::make_tuple(1U, 1U, 1U, 1U),
             std::make_tuple(2U, 1U, 1U, 1U),
             std::make_tuple(1U, 2U, 1U, 1U),
             std::make_tuple(2U, 1U, 2U, 1U),
             std::make_tuple(1U, 2U, 2U, 1U),
             std::make_tuple(2U, 1U, 1U, 2U),
             std::make_tuple(1U, 2U, 1U, 2U),
             std::make_tuple(0U, 0U, 3U, 3U))
{
    Matrix<double> A(4, 4);
    A << "1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16";

    const Matrix<double> B = A.block(row, col, nbRows, nbCols);

    ASSERT_EQUALS(B.rows(), nbRows);
    ASSERT_EQUALS(B.cols(), nbCols);
    ASSERT_EQUALS(B(0, 0), A(row, col));
    ASSERT_EQUALS(B(nbRows - 1, nbCols - 1),
                  A(row + nbRows - 1, col + nbCols - 1));
}

RUN_TESTS()
