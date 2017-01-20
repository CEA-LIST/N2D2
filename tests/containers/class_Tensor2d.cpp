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

#include "containers/Tensor2d.hpp"
#include "utils/UnitTest.hpp"

using namespace N2D2;

TEST(Tensor2d, Tensor2d)
{
    const Tensor2d<double> A;

    ASSERT_EQUALS(A.dimX(), 0U);
    ASSERT_EQUALS(A.dimY(), 0U);
    ASSERT_EQUALS(A.size(), 0U);
    ASSERT_TRUE(A.empty());
}

TEST_DATASET(Tensor2d,
             Tensor2d__dimX_dimY,
             (unsigned int dimX, unsigned int dimY),
             std::make_tuple(0U, 3U),
             std::make_tuple(3U, 0U),
             std::make_tuple(1U, 3U),
             std::make_tuple(3U, 1U),
             std::make_tuple(3U, 3U),
             std::make_tuple(12U, 34U),
             std::make_tuple(34U, 12U))
{
    const Tensor2d<double> A(dimX, dimY);

    ASSERT_EQUALS(A.dimX(), dimX);
    ASSERT_EQUALS(A.dimY(), dimY);
    ASSERT_EQUALS(A.size(), dimX * dimY);
    ASSERT_TRUE(A.empty() == (dimX * dimY == 0));
}

TEST_DATASET(Tensor2d,
             Tensor2d_size,
             (unsigned int dimX, unsigned int dimY),
             std::make_tuple(1U, 1U),
             std::make_tuple(1U, 3U),
             std::make_tuple(3U, 1U),
             std::make_tuple(3U, 3U),
             std::make_tuple(12U, 34U),
             std::make_tuple(34U, 12U))
{
    Tensor2d<double> A(dimX, dimY);

    ASSERT_EQUALS(A.dimX(), dimX);
    ASSERT_EQUALS(A.dimY(), dimY);
    ASSERT_EQUALS(A.size(), dimX * dimY);
}

TEST_DATASET(Tensor2d,
             Tensor2d_resize,
             (unsigned int dimX, unsigned int dimY),
             std::make_tuple(1U, 1U),
             std::make_tuple(1U, 3U),
             std::make_tuple(3U, 1U),
             std::make_tuple(3U, 3U),
             std::make_tuple(12U, 34U),
             std::make_tuple(34U, 12U))
{
    Tensor2d<double> A;
    A.resize(dimX, dimY);

    ASSERT_EQUALS(A.dimX(), dimX);
    ASSERT_EQUALS(A.dimY(), dimY);
    ASSERT_EQUALS(A.size(), dimX * dimY);
}

TEST_DATASET(Tensor2d,
             Tensor2d__fromCV,
             (unsigned int dimX, unsigned int dimY),
             std::make_tuple(1U, 1U),
             std::make_tuple(1U, 3U),
             std::make_tuple(3U, 1U),
             std::make_tuple(3U, 3U),
             std::make_tuple(12U, 34U),
             std::make_tuple(34U, 12U))
{
    cv::Mat mat(cv::Size(dimX, dimY), CV_32SC1);

    for (unsigned int i = 0; i < dimX; ++i) {
        for (unsigned int j = 0; j < dimY; ++j)
            mat.at<int>(j, i) = i + j * dimX;
    }

    ASSERT_EQUALS(mat.at<int>(0, 0), 0);
    ASSERT_EQUALS(mat.at<int>(dimY - 1, dimX - 1), (int)(dimX * dimY) - 1);

    const Tensor2d<int> A(mat);

    ASSERT_EQUALS(mat.cols, (int)dimX);
    ASSERT_EQUALS(mat.rows, (int)dimY);
    ASSERT_EQUALS(A.dimX(), dimX);
    ASSERT_EQUALS(A.dimY(), dimY);
    ASSERT_EQUALS(A.size(), dimX * dimY);
    ASSERT_TRUE(A.empty() == (dimX * dimY == 0));

    for (unsigned int i = 0; i < dimX; ++i) {
        for (unsigned int j = 0; j < dimY; ++j)
            ASSERT_EQUALS(A(i, j), mat.at<int>(j, i));
    }
}

TEST_DATASET(Tensor2d,
             Tensor2d__fromCV_uchar_to_float,
             (unsigned int dimX, unsigned int dimY),
             std::make_tuple(1U, 1U),
             std::make_tuple(1U, 3U),
             std::make_tuple(3U, 1U),
             std::make_tuple(3U, 3U),
             std::make_tuple(12U, 34U),
             std::make_tuple(34U, 12U))
{
    cv::Mat mat(cv::Size(dimX, dimY), CV_8UC1);

    for (unsigned int i = 0; i < dimX; ++i) {
        for (unsigned int j = 0; j < dimY; ++j)
            mat.at<unsigned char>(j, i) = i + j * dimX;
    }

    ASSERT_EQUALS(mat.at<unsigned char>(0, 0), 0);
    ASSERT_EQUALS(mat.at<unsigned char>(dimY - 1, dimX - 1),
                  (unsigned char)(dimX * dimY) - 1);

    const Tensor2d<float> A(mat);

    ASSERT_EQUALS(mat.cols, (int)dimX);
    ASSERT_EQUALS(mat.rows, (int)dimY);
    ASSERT_EQUALS(A.dimX(), dimX);
    ASSERT_EQUALS(A.dimY(), dimY);
    ASSERT_EQUALS(A.size(), dimX * dimY);
    ASSERT_TRUE(A.empty() == (dimX * dimY == 0));

    for (unsigned int i = 0; i < dimX; ++i) {
        for (unsigned int j = 0; j < dimY; ++j)
            ASSERT_EQUALS_DELTA(
                A(i, j), mat.at<unsigned char>(j, i) / 255.0, 1e-6);
    }
}

TEST_DATASET(Tensor2d,
             Tensor2d__toCV,
             (unsigned int dimX, unsigned int dimY),
             std::make_tuple(1U, 1U),
             std::make_tuple(1U, 3U),
             std::make_tuple(3U, 1U),
             std::make_tuple(3U, 3U),
             std::make_tuple(12U, 34U),
             std::make_tuple(34U, 12U))
{
    Tensor2d<double> A(dimX, dimY);

    for (unsigned int i = 0; i < A.size(); ++i)
        A(i) = i;

    ASSERT_EQUALS(A(0, 0), 0);
    ASSERT_EQUALS(A(dimX - 1, dimY - 1), (int)(dimX * dimY) - 1);

    const cv::Mat mat = (cv::Mat)A;

    ASSERT_EQUALS(mat.rows, (int)dimY);
    ASSERT_EQUALS(mat.cols, (int)dimX);

    for (unsigned int i = 0; i < dimX; ++i) {
        for (unsigned int j = 0; j < dimY; ++j)
            ASSERT_EQUALS(A(i, j), mat.at<double>(j, i));
    }
}

TEST(Tensor2d, clear)
{
    Tensor2d<double> A(2, 2, 1.0);

    ASSERT_EQUALS(A(1, 1), 1.0);

    A.clear();

    ASSERT_EQUALS(A.dimX(), 0U);
    ASSERT_EQUALS(A.dimY(), 0U);
    ASSERT_EQUALS(A.size(), 0U);
    ASSERT_TRUE(A.empty());
}

RUN_TESTS()
