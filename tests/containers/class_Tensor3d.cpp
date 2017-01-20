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
#include "containers/Tensor3d.hpp"
#include "utils/Random.hpp"
#include "utils/UnitTest.hpp"
#include "utils/Utils.hpp"

using namespace N2D2;

TEST(Tensor3d, Tensor3d)
{
    const Tensor3d<double> A;

    ASSERT_EQUALS(A.dimX(), 0U);
    ASSERT_EQUALS(A.dimY(), 0U);
    ASSERT_EQUALS(A.dimZ(), 0U);
    ASSERT_EQUALS(A.size(), 0U);
    ASSERT_TRUE(A.empty());
}

TEST_DATASET(Tensor3d,
             Tensor3d__dimX_dimY_dimZ,
             (unsigned int dimX, unsigned int dimY, unsigned int dimZ),
             std::make_tuple(0U, 3U, 1U),
             std::make_tuple(3U, 0U, 1U),
             std::make_tuple(1U, 3U, 1U),
             std::make_tuple(3U, 1U, 1U),
             std::make_tuple(3U, 3U, 1U),
             std::make_tuple(12U, 34U, 1U),
             std::make_tuple(34U, 12U, 1U),
             std::make_tuple(34U, 12U, 10U))
{
    Tensor3d<int> tensor(dimX, dimY, dimZ);

    ASSERT_EQUALS(tensor.dimX(), dimX);
    ASSERT_EQUALS(tensor.dimY(), dimY);
    ASSERT_EQUALS(tensor.dimZ(), dimZ);
    ASSERT_EQUALS(tensor.size(), dimX * dimY * dimZ);
}

TEST(Tensor3d, push_back)
{
    Tensor2d<int> A;
    Tensor3d<int> tensor;

    ASSERT_EQUALS(tensor.dimZ(), 0U);
    ASSERT_EQUALS(tensor.size(), 0U);
    ASSERT_TRUE(tensor.empty());

    tensor.push_back(A);

    ASSERT_EQUALS(tensor.dimZ(), 1U);
    ASSERT_EQUALS(tensor.size(), 0U);
    ASSERT_TRUE(tensor.empty());
}

TEST_DATASET(Tensor3d,
             subscript_operator,
             (unsigned int dimX, unsigned int dimY, unsigned int dimZ),
             std::make_tuple(0U, 3U, 3U),
             std::make_tuple(3U, 0U, 3U),
             std::make_tuple(1U, 3U, 3U),
             std::make_tuple(3U, 1U, 3U),
             std::make_tuple(3U, 3U, 3U),
             std::make_tuple(12U, 34U, 3U),
             std::make_tuple(34U, 12U, 3U),
             std::make_tuple(34U, 12U, 10U))
{
    Random::mtSeed(0);

    Tensor3d<int> tensor(dimX, dimY, dimZ);

    for (unsigned int i = 0; i < tensor.size(); ++i)
        tensor(i) = Random::randUniform(-100, 100);

    Tensor2d<int> subTensor = tensor[1];

    ASSERT_EQUALS(subTensor.dimX(), dimX);
    ASSERT_EQUALS(subTensor.dimY(), dimY);

    for (unsigned int i = 0; i < subTensor.size(); ++i) {
        ASSERT_EQUALS(subTensor(i), tensor(i + dimX * dimY));
    }

    for (unsigned int i = 0; i < dimX; ++i) {
        for (unsigned int j = 0; j < dimY; ++j) {
            ASSERT_EQUALS(subTensor(i, j), tensor(i, j, 1));
        }
    }
}

TEST_DATASET(Tensor3d,
             subscript_operator_bis,
             (unsigned int dimX, unsigned int dimY, unsigned int dimZ),
             std::make_tuple(0U, 3U, 3U),
             std::make_tuple(3U, 0U, 3U),
             std::make_tuple(1U, 3U, 3U),
             std::make_tuple(3U, 1U, 3U),
             std::make_tuple(3U, 3U, 3U),
             std::make_tuple(12U, 34U, 3U),
             std::make_tuple(34U, 12U, 3U),
             std::make_tuple(34U, 12U, 10U))
{
    Random::mtSeed(0);

    Tensor3d<int> tensor(dimX, dimY, dimZ, 1);
    Tensor2d<int> subTensor = tensor[1];

    for (unsigned int i = 0; i < subTensor.size(); ++i)
        subTensor(i) = Random::randUniform(-100, 100);

    for (unsigned int i = 0; i < dimX; ++i) {
        for (unsigned int j = 0; j < dimY; ++j) {
            ASSERT_EQUALS(tensor(i, j, 0), 1);
            ASSERT_EQUALS(subTensor(i, j), tensor(i, j, 1));
            ASSERT_EQUALS(tensor(i, j, 2), 1);
        }
    }
}

TEST_DATASET(Tensor3d,
             subscript_operator_ter,
             (unsigned int dimX, unsigned int dimY, unsigned int dimZ),
             std::make_tuple(0U, 3U, 3U),
             std::make_tuple(3U, 0U, 3U),
             std::make_tuple(1U, 3U, 3U),
             std::make_tuple(3U, 1U, 3U),
             std::make_tuple(3U, 3U, 3U),
             std::make_tuple(12U, 34U, 3U),
             std::make_tuple(34U, 12U, 3U),
             std::make_tuple(34U, 12U, 10U))
{
    Random::mtSeed(0);

    Tensor3d<int> tensor(dimX, dimY, dimZ, 1);
    Tensor2d<int> subTensor(dimX, dimY);

    for (unsigned int i = 0; i < subTensor.size(); ++i)
        subTensor(i) = Random::randUniform(-100, 100);

    tensor[1] = subTensor;

    for (unsigned int i = 0; i < dimX; ++i) {
        for (unsigned int j = 0; j < dimY; ++j) {
            ASSERT_EQUALS(tensor(i, j, 0), 1);
            ASSERT_EQUALS(subTensor(i, j), tensor(i, j, 1));
            ASSERT_EQUALS(tensor(i, j, 2), 1);
        }
    }
}

TEST_DATASET(Tensor3d,
             dimZ,
             (unsigned int dimX, unsigned int dimY, unsigned int dimZ),
             std::make_tuple(0U, 3U, 1U),
             std::make_tuple(3U, 0U, 1U),
             std::make_tuple(1U, 3U, 1U),
             std::make_tuple(3U, 1U, 1U),
             std::make_tuple(3U, 3U, 1U),
             std::make_tuple(12U, 34U, 1U),
             std::make_tuple(34U, 12U, 1U),
             std::make_tuple(34U, 12U, 10U))
{
    Tensor2d<int> A(dimX, dimY);
    Tensor3d<int> tensor;

    for (unsigned int i = 0; i < dimZ; ++i)
        tensor.push_back(A);

    ASSERT_EQUALS(tensor.dimZ(), dimZ);
    ASSERT_EQUALS(tensor.size(), dimX * dimY * dimZ);
}

TEST_DATASET(Tensor3d,
             Tensor3d__fromCV,
             (unsigned int dimX, unsigned int dimY, unsigned int dimZ),
             std::make_tuple(1U, 1U, 1U),
             std::make_tuple(1U, 3U, 1U),
             std::make_tuple(3U, 1U, 1U),
             std::make_tuple(3U, 3U, 1U),
             std::make_tuple(12U, 34U, 1U),
             std::make_tuple(34U, 12U, 1U),
             std::make_tuple(1U, 1U, 2U),
             std::make_tuple(1U, 3U, 2U),
             std::make_tuple(3U, 1U, 2U),
             std::make_tuple(3U, 3U, 2U),
             std::make_tuple(12U, 34U, 2U),
             std::make_tuple(34U, 12U, 2U),
             std::make_tuple(1U, 1U, 4U),
             std::make_tuple(1U, 3U, 4U),
             std::make_tuple(3U, 1U, 4U),
             std::make_tuple(3U, 3U, 4U),
             std::make_tuple(12U, 34U, 4U),
             std::make_tuple(34U, 12U, 4U))
{
    cv::Mat channel(cv::Size(dimX, dimY), CV_32SC1);

    for (unsigned int i = 0; i < dimX; ++i) {
        for (unsigned int j = 0; j < dimY; ++j)
            channel.at<int>(j, i) = i + j * dimX;
    }

    std::vector<cv::Mat> channels;

    for (unsigned int k = 0; k < dimZ; ++k) {
        channels.push_back(channel.clone());
        channel += dimX * dimY;
    }

    cv::Mat mat;
    cv::merge(channels, mat);

    if (dimZ == 1) {
        ASSERT_EQUALS(mat.type(), CV_32SC1);
        ASSERT_EQUALS(mat.at<int>(0, 0), 0);
        ASSERT_EQUALS(mat.at<int>(dimY - 1, dimX - 1),
                      (int)(dimX * dimY * dimZ) - 1);
    } else if (dimZ == 2) {
        ASSERT_EQUALS(mat.type(), CV_32SC2);
        ASSERT_EQUALS(mat.at<cv::Vec2i>(0, 0)[0], 0);
        ASSERT_EQUALS(mat.at<cv::Vec2i>(dimY - 1, dimX - 1)[dimZ - 1],
                      (int)(dimX * dimY * dimZ) - 1);
    } else if (dimZ == 3) {
        ASSERT_EQUALS(mat.type(), CV_32SC3);
        ASSERT_EQUALS(mat.at<cv::Vec3i>(0, 0)[0], 0);
        ASSERT_EQUALS(mat.at<cv::Vec3i>(dimY - 1, dimX - 1)[dimZ - 1],
                      (int)(dimX * dimY * dimZ) - 1);
    } else if (dimZ == 4) {
        ASSERT_EQUALS(mat.type(), CV_32SC4);
        ASSERT_EQUALS(mat.at<cv::Vec4i>(0, 0)[0], 0);
        ASSERT_EQUALS(mat.at<cv::Vec4i>(dimY - 1, dimX - 1)[dimZ - 1],
                      (int)(dimX * dimY * dimZ) - 1);
    }

    const Tensor3d<int> A(mat);

    ASSERT_EQUALS(mat.cols, (int)dimX);
    ASSERT_EQUALS(mat.rows, (int)dimY);
    ASSERT_EQUALS(mat.channels(), (int)dimZ);
    ASSERT_EQUALS(A.dimX(), dimX);
    ASSERT_EQUALS(A.dimY(), dimY);
    ASSERT_EQUALS(A.dimZ(), dimZ);
    ASSERT_EQUALS(A.size(), dimX * dimY * dimZ);
    ASSERT_TRUE(A.empty() == (dimX * dimY * dimZ == 0));

    for (unsigned int i = 0; i < dimX; ++i) {
        for (unsigned int j = 0; j < dimY; ++j) {
            for (unsigned int k = 0; k < dimZ; ++k) {
                if (dimZ == 1) {
                    ASSERT_EQUALS(A(i, j, k), mat.at<int>(j, i));
                } else if (dimZ == 2) {
                    ASSERT_EQUALS(A(i, j, k), mat.at<cv::Vec2i>(j, i)[k]);
                } else if (dimZ == 3) {
                    ASSERT_EQUALS(A(i, j, k), mat.at<cv::Vec3i>(j, i)[k]);
                } else if (dimZ == 4) {
                    ASSERT_EQUALS(A(i, j, k), mat.at<cv::Vec4i>(j, i)[k]);
                }
            }
        }
    }
}

TEST_DATASET(Tensor3d,
             Tensor3d__toCV,
             (unsigned int dimX, unsigned int dimY, unsigned int dimZ),
             std::make_tuple(1U, 1U, 1U),
             std::make_tuple(1U, 3U, 1U),
             std::make_tuple(3U, 1U, 1U),
             std::make_tuple(3U, 3U, 1U),
             std::make_tuple(12U, 34U, 1U),
             std::make_tuple(34U, 12U, 1U),
             std::make_tuple(1U, 1U, 2U),
             std::make_tuple(1U, 3U, 2U),
             std::make_tuple(3U, 1U, 2U),
             std::make_tuple(3U, 3U, 2U),
             std::make_tuple(12U, 34U, 2U),
             std::make_tuple(34U, 12U, 2U),
             std::make_tuple(1U, 1U, 4U),
             std::make_tuple(1U, 3U, 4U),
             std::make_tuple(3U, 1U, 4U),
             std::make_tuple(3U, 3U, 4U),
             std::make_tuple(12U, 34U, 4U),
             std::make_tuple(34U, 12U, 4U))
{
    Tensor3d<double> A(dimX, dimY, dimZ);

    for (unsigned int i = 0; i < A.size(); ++i)
        A(i) = i;

    ASSERT_EQUALS(A(0, 0, 0), 0);
    ASSERT_EQUALS(A(dimX - 1, dimY - 1, dimZ - 1),
                  (int)(dimX * dimY * dimZ) - 1);

    const cv::Mat mat = (cv::Mat)A;

    ASSERT_EQUALS(mat.rows, (int)dimY);
    ASSERT_EQUALS(mat.cols, (int)dimX);
    ASSERT_EQUALS(mat.channels(), (int)dimZ);

    if (dimZ == 1) {
        ASSERT_EQUALS(mat.type(), CV_64FC1);
    } else if (dimZ == 2) {
        ASSERT_EQUALS(mat.type(), CV_64FC2);
    } else if (dimZ == 3) {
        ASSERT_EQUALS(mat.type(), CV_64FC3);
    } else if (dimZ == 4) {
        ASSERT_EQUALS(mat.type(), CV_64FC4);
    }

    for (unsigned int i = 0; i < dimX; ++i) {
        for (unsigned int j = 0; j < dimY; ++j) {
            for (unsigned int k = 0; k < dimZ; ++k) {
                if (dimZ == 1) {
                    ASSERT_EQUALS(A(i, j, k), mat.at<double>(j, i));
                } else if (dimZ == 2) {
                    ASSERT_EQUALS(A(i, j, k), mat.at<cv::Vec2d>(j, i)[k]);
                } else if (dimZ == 3) {
                    ASSERT_EQUALS(A(i, j, k), mat.at<cv::Vec3d>(j, i)[k]);
                } else if (dimZ == 4) {
                    ASSERT_EQUALS(A(i, j, k), mat.at<cv::Vec4d>(j, i)[k]);
                }
            }
        }
    }
}

TEST(Tensor3d, clear)
{
    Tensor3d<double> A(2, 3, 4, 1.0);

    ASSERT_EQUALS(A(1, 1, 1), 1.0);

    A.clear();

    ASSERT_EQUALS(A.dimX(), 0U);
    ASSERT_EQUALS(A.dimY(), 0U);
    ASSERT_EQUALS(A.dimZ(), 0U);
    ASSERT_EQUALS(A.size(), 0U);
    ASSERT_TRUE(A.empty());
}

RUN_TESTS()
