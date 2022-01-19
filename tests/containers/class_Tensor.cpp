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

#include "containers/Tensor.hpp"
#include "utils/Random.hpp"
#include "utils/UnitTest.hpp"
#include "utils/Utils.hpp"

using namespace N2D2;

TEST(Tensor2d, Tensor2d)
{
    const Tensor<double> A({0, 0});

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
    const Tensor<double> A({dimX, dimY});

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
    Tensor<double> A({dimX, dimY});

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
    Tensor<double> A;
    A.resize({dimX, dimY});

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

    const Tensor<int> A(mat);

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

    const Tensor<float> A(mat);

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
    Tensor<double> A({dimX, dimY});

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
    Tensor<double> A({2, 2}, 1.0);

    ASSERT_EQUALS(A(1, 1), 1.0);

    A.clear();

    ASSERT_EQUALS(A.dimX(), 0U);
    ASSERT_EQUALS(A.dimY(), 0U);
    ASSERT_EQUALS(A.size(), 0U);
    ASSERT_TRUE(A.empty());
}

TEST(Tensor3d, Tensor3d)
{
    const Tensor<double> A({0, 0, 0});

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
    Tensor<int> tensor({dimX, dimY, dimZ});

    ASSERT_EQUALS(tensor.dimX(), dimX);
    ASSERT_EQUALS(tensor.dimY(), dimY);
    ASSERT_EQUALS(tensor.dimZ(), dimZ);
    ASSERT_EQUALS(tensor.size(), dimX * dimY * dimZ);
}

TEST(Tensor3d, push_back)
{
    Tensor<int> A({0, 0});
    Tensor<int> tensor({0, 0, 0});

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

    Tensor<int> tensor({dimX, dimY, dimZ});

    for (unsigned int i = 0; i < tensor.size(); ++i)
        tensor(i) = Random::randUniform(-100, 100);

    Tensor<int> subTensor = tensor[1];

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

    Tensor<int> tensor({dimX, dimY, dimZ}, 1);
    Tensor<int> subTensor = tensor[1];

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

    Tensor<int> tensor({dimX, dimY, dimZ}, 1);
    Tensor<int> subTensor({dimX, dimY});

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
    Tensor<int> A({dimX, dimY});
    Tensor<int> tensor;

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

    const Tensor<int> A(mat);

    ASSERT_EQUALS(mat.cols, (int)dimX);
    ASSERT_EQUALS(mat.rows, (int)dimY);
    ASSERT_EQUALS(mat.channels(), (int)dimZ);
    ASSERT_EQUALS(A.dimX(), dimX);
    ASSERT_EQUALS(A.dimY(), dimY);
    ASSERT_EQUALS(A.dimZ(), (dimZ > 1) ? dimZ : dimX);
    ASSERT_EQUALS(A.size(), dimX * dimY * dimZ);
    ASSERT_TRUE(A.empty() == (dimX * dimY * dimZ == 0));

    for (unsigned int i = 0; i < dimX; ++i) {
        for (unsigned int j = 0; j < dimY; ++j) {
            for (unsigned int k = 0; k < dimZ; ++k) {
                if (dimZ == 1) {
                    ASSERT_EQUALS(A(i, j), mat.at<int>(j, i));
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
    Tensor<double> A({dimX, dimY, dimZ});

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
    Tensor<double> A({2, 3, 4}, 1.0);

    ASSERT_EQUALS(A(1, 1, 1), 1.0);

    A.clear();

    ASSERT_EQUALS(A.dimX(), 0U);
    ASSERT_EQUALS(A.dimY(), 0U);
    ASSERT_EQUALS(A.dimZ(), 0U);
    ASSERT_EQUALS(A.size(), 0U);
    ASSERT_TRUE(A.empty());
}

TEST(Tensor4d, Tensor4d)
{
    const Tensor<double> A({0, 0, 0, 0});

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
    Tensor<int> tensor({dimX, dimY, dimZ, dimB});

    ASSERT_EQUALS(tensor.dimX(), dimX);
    ASSERT_EQUALS(tensor.dimY(), dimY);
    ASSERT_EQUALS(tensor.dimZ(), dimZ);
    ASSERT_EQUALS(tensor.dimB(), dimB);
    ASSERT_EQUALS(tensor.size(), dimX * dimY * dimZ * dimB);
}

TEST(Tensor4d, push_back)
{
    Tensor<int> A({0, 0, 0});
    Tensor<int> tensor;

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
    Tensor<int> A({dimX, dimY, dimZ});
    Tensor<int> tensor;

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
    Tensor<int> A1({8, 8, 3, 4});
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

TEST(Tensor4d, append)
{
    Tensor<int> A({2, 3, 4, 5});
    Tensor<int> B({2, 3, 4, 3});

    int idxA = 0;
    int idxB = -100000;
    std::iota(A.begin(), A.end(), idxA);
    std::iota(B.begin(), B.end(), idxB);

    A.append(B);

    ASSERT_EQUALS(A.dimX(), 2U);
    ASSERT_EQUALS(A.dimY(), 3U);
    ASSERT_EQUALS(A.dimZ(), 4U);
    ASSERT_EQUALS(A.dimB(), 8U);

    for (unsigned int b = 0; b < 5; ++b) {
        for (unsigned int k = 0; k < 4; ++k) {
            for (unsigned int j = 0; j < 3; ++j) {
                for (unsigned int i = 0; i < 2; ++i) {
                    ASSERT_EQUALS(A(i, j, k, b), idxA);
                    ++idxA;
                }
            }
        }
    }
    for (unsigned int b = 5; b < 8; ++b) {
        for (unsigned int k = 0; k < 4; ++k) {
            for (unsigned int j = 0; j < 3; ++j) {
                for (unsigned int i = 0; i < 2; ++i) {
                    ASSERT_EQUALS(A(i, j, k, b), idxB);
                    ++idxB;
                }
            }
        }
    }
}

TEST(Tensor4d, append_0)
{
    Tensor<int> A({2, 3, 4, 5});
    Tensor<int> B({10, 3, 4, 5});

    int idxA = 0;
    int idxB = -100000;
    std::iota(A.begin(), A.end(), idxA);
    std::iota(B.begin(), B.end(), idxB);

    A.append(B, 0);

    ASSERT_EQUALS(A.dimX(), 12U);
    ASSERT_EQUALS(A.dimY(), 3U);
    ASSERT_EQUALS(A.dimZ(), 4U);
    ASSERT_EQUALS(A.dimB(), 5U);

    for (unsigned int b = 0; b < 5; ++b) {
        for (unsigned int k = 0; k < 4; ++k) {
            for (unsigned int j = 0; j < 3; ++j) {
                for (unsigned int i = 0; i < 2; ++i) {
                    ASSERT_EQUALS(A(i, j, k, b), idxA);
                    ++idxA;
                }
                for (unsigned int i = 2; i < 12; ++i) {
                    ASSERT_EQUALS(A(i, j, k, b), idxB);
                    ++idxB;
                }
            }
        }
    }
}

TEST(Tensor4d, append_1)
{
    Tensor<int> A({2, 3, 4, 5});
    Tensor<int> B({2, 1, 4, 5});

    int idxA = 0;
    int idxB = -100000;
    std::iota(A.begin(), A.end(), idxA);
    std::iota(B.begin(), B.end(), idxB);

    A.append(B, 1);

    ASSERT_EQUALS(A.dimX(), 2U);
    ASSERT_EQUALS(A.dimY(), 4U);
    ASSERT_EQUALS(A.dimZ(), 4U);
    ASSERT_EQUALS(A.dimB(), 5U);

    for (unsigned int b = 0; b < 5; ++b) {
        for (unsigned int k = 0; k < 4; ++k) {
            for (unsigned int j = 0; j < 3; ++j) {
                for (unsigned int i = 0; i < 2; ++i) {
                    ASSERT_EQUALS(A(i, j, k, b), idxA);
                    ++idxA;
                }
            }
            for (unsigned int j = 3; j < 4; ++j) {
                for (unsigned int i = 0; i < 2; ++i) {
                    ASSERT_EQUALS(A(i, j, k, b), idxB);
                    ++idxB;
                }
            }
        }
    }
}

TEST(Tensor4d, rows)
{
    Tensor<int> A({2, 3, 4, 5});
    int idxA = 0;
    std::iota(A.begin(), A.end(), idxA);

    unsigned int offset = 1;
    unsigned int size = 2;
    Tensor<int> B = A.rows(offset, size, 2);

    ASSERT_EQUALS(B.dimX(), 2U);
    ASSERT_EQUALS(B.dimY(), 3U);
    ASSERT_EQUALS(B.dimZ(), size);
    ASSERT_EQUALS(B.dimB(), 5U);

    for (unsigned int b = 0; b < 5; ++b) {
        for (unsigned int k = 0; k < 4; ++k) {
            for (unsigned int j = 0; j < 3; ++j) {
                for (unsigned int i = 0; i < 2; ++i) {
                    ASSERT_EQUALS(A(i, j, k, b), idxA);

                    if (k >= offset && k < offset + size) {
                        ASSERT_EQUALS(B(i, j, k - offset, b), idxA);
                    }

                    ++idxA;
                }
            }
        }
    }
}

TEST(Tensor4d, rows_0)
{
    Tensor<int> A({10, 3, 4, 5});
    int idxA = 0;
    std::iota(A.begin(), A.end(), idxA);

    unsigned int offset = 5;
    unsigned int size = 5;
    Tensor<int> B = A.rows(offset, size, 0);

    ASSERT_EQUALS(B.dimX(), size);
    ASSERT_EQUALS(B.dimY(), 3U);
    ASSERT_EQUALS(B.dimZ(), 4U);
    ASSERT_EQUALS(B.dimB(), 5U);

    for (unsigned int b = 0; b < 5; ++b) {
        for (unsigned int k = 0; k < 4; ++k) {
            for (unsigned int j = 0; j < 3; ++j) {
                for (unsigned int i = 0; i < 10; ++i) {
                    ASSERT_EQUALS(A(i, j, k, b), idxA);

                    if (i >= offset && i < offset + size) {
                        ASSERT_EQUALS(B(i - offset, j, k, b), idxA);
                    }

                    ++idxA;
                }
            }
        }
    }
}

TEST(Tensor4d, clear)
{
    Tensor<double> A({2, 3, 4, 5}, 1.0);

    ASSERT_EQUALS(A(1, 1, 1, 1), 1.0);

    A.clear();

    ASSERT_EQUALS(A.dimX(), 0U);
    ASSERT_EQUALS(A.dimY(), 0U);
    ASSERT_EQUALS(A.dimZ(), 0U);
    ASSERT_EQUALS(A.dimB(), 0U);
    ASSERT_EQUALS(A.size(), 0U);
    ASSERT_TRUE(A.empty());
}

TEST(Tensor4d, ctor_copy)
{
    const Tensor<double> A({2, 3, 4, 5}, 1.0);

    Tensor<double> B(A);
    // Data from A was converted to B
    ASSERT_EQUALS(B.dims(), A.dims());
    ASSERT_EQUALS(B(1, 1, 1, 1), 1.0);
    // Changes in B will affect A
    B(1, 1, 1, 1) = 2.0;
    ASSERT_EQUALS(B(1, 1, 1, 1), 2.0);
    ASSERT_EQUALS(A(1, 1, 1, 1), 2.0);
}

TEST(Tensor4d, clone)
{
    const Tensor<double> A({2, 3, 4, 5}, 1.0);

    Tensor<double> B = A.clone();
    // Data from A was converted to B
    ASSERT_EQUALS(B.dims(), A.dims());
    ASSERT_EQUALS(B(1, 1, 1, 1), 1.0);
    // Changes in B won't affect A
    B(1, 1, 1, 1) = 2.0;
    ASSERT_EQUALS(B(1, 1, 1, 1), 2.0);
    ASSERT_EQUALS(A(1, 1, 1, 1), 1.0);
}

TEST(Tensor4d, tensor_cast_double_to_float)
{
    Tensor<double> A({2, 3, 4, 5}, 1.0);

    ASSERT_EQUALS(A(1, 1, 1, 1), 1.0);

    // 1. First cast double to float
    Tensor<float> B = tensor_cast<float>(A);
    // Data from A was converted to B
    ASSERT_EQUALS(B.dims(), A.dims());
    ASSERT_EQUALS(B(1, 1, 1, 1), 1.0);
    // Changes in B won't affect A
    B(1, 1, 1, 1) = 2.0;
    ASSERT_EQUALS(B(1, 1, 1, 1), 2.0);
    ASSERT_EQUALS(A(1, 1, 1, 1), 1.0);

    // 2. Cast in the same type (double to double)
    Tensor<double> C = tensor_cast<double>(A);
    // C shares the same data as A
    ASSERT_EQUALS(C.dims(), A.dims());
    C(1, 1, 1, 1) = 4.0;
    ASSERT_EQUALS(A(1, 1, 1, 1), 4.0);

    // 3. Second cast double to float
    // Since A was already casted to double with B, B2 shares the same data as B
    // In this case, no new allocation is made
    Tensor<float> B2 = tensor_cast_nocopy<float>(A);
    ASSERT_EQUALS(B2(0, 0, 0, 0), 1.0);
    ASSERT_EQUALS(B2(1, 1, 1, 1), 2.0);

    // 4. Third cast double to float with copy
    Tensor<float> B3 = tensor_cast<float>(A);
    // Since B, B2 and B3 share the same data, they are now all in sync. with A
    ASSERT_EQUALS(B3(0, 0, 0, 0), 1.0);
    ASSERT_EQUALS(B3(1, 1, 1, 1), 4.0);
    ASSERT_EQUALS(B2(0, 0, 0, 0), 1.0);
    ASSERT_EQUALS(B2(1, 1, 1, 1), 4.0);
    ASSERT_EQUALS(B(0, 0, 0, 0), 1.0);
    ASSERT_EQUALS(B(1, 1, 1, 1), 4.0);

    // 5. Copy B back to A with automatic conversion
    B(1, 1, 1, 1) = 8.0;
    A = B;
    ASSERT_EQUALS(B(1, 1, 1, 1), 8.0);
    ASSERT_EQUALS(A(1, 1, 1, 1), 8.0);
}

TEST(Tensor4d, tensor_cast_float_to_double)
{
    Tensor<float> A({2, 3, 4, 5}, 1.0);

    ASSERT_EQUALS(A(1, 1, 1, 1), 1.0);

    // 1. First cast float to double
    Tensor<double> B = tensor_cast<double>(A);
    // Data from A was converted to B
    ASSERT_EQUALS(B.dims(), A.dims());
    ASSERT_EQUALS(B(1, 1, 1, 1), 1.0);
    // Changes in B won't affect A
    B(1, 1, 1, 1) = 2.0;
    ASSERT_EQUALS(B(1, 1, 1, 1), 2.0);
    ASSERT_EQUALS(A(1, 1, 1, 1), 1.0);

    // 2. Cast in the same type (float to float)
    Tensor<float> C = tensor_cast<float>(A);
    // C shares the same data as A
    ASSERT_EQUALS(C.dims(), A.dims());
    C(1, 1, 1, 1) = 4.0;
    ASSERT_EQUALS(A(1, 1, 1, 1), 4.0);

    // 3. Second cast float to double
    // Since A was already casted to float with B, B2 shares the same data as B
    // In this case, no new allocation is made
    Tensor<double> B2 = tensor_cast_nocopy<double>(A);
    ASSERT_EQUALS(B2(0, 0, 0, 0), 1.0);
    ASSERT_EQUALS(B2(1, 1, 1, 1), 2.0);

    // 4. Third cast float to double with copy
    Tensor<double> B3 = tensor_cast<double>(A);
    // Since B, B2 and B3 share the same data, they are now all in sync. with A
    ASSERT_EQUALS(B3(0, 0, 0, 0), 1.0);
    ASSERT_EQUALS(B3(1, 1, 1, 1), 4.0);
    ASSERT_EQUALS(B2(0, 0, 0, 0), 1.0);
    ASSERT_EQUALS(B2(1, 1, 1, 1), 4.0);
    ASSERT_EQUALS(B(0, 0, 0, 0), 1.0);
    ASSERT_EQUALS(B(1, 1, 1, 1), 4.0);
}

TEST(Tensor4d, tensor_cast_float_to_int)
{
    Tensor<float> A({2, 3, 4, 5}, 1.0);

    ASSERT_EQUALS(A(1, 1, 1, 1), 1.0);

    // 1. First cast float to int
    Tensor<int> B = tensor_cast<int>(A);
    // Data from A was converted to B
    ASSERT_EQUALS(B.dims(), A.dims());
    ASSERT_EQUALS(B(1, 1, 1, 1), 1);
    // Changes in B won't affect A
    B(1, 1, 1, 1) = 2;
    ASSERT_EQUALS(B(1, 1, 1, 1), 2);
    ASSERT_EQUALS(A(1, 1, 1, 1), 1.0);

    // 2. Cast in the same type (float to float)
    Tensor<float> C = tensor_cast<float>(A);
    // C shares the same data as A
    ASSERT_EQUALS(C.dims(), A.dims());
    C(1, 1, 1, 1) = 4.0;
    ASSERT_EQUALS(A(1, 1, 1, 1), 4.0);

    // 3. Second cast float to int
    // Since A was already casted to float with B, B2 shares the same data as B
    // In this case, no new allocation is made
    Tensor<int> B2 = tensor_cast_nocopy<int>(A);
    ASSERT_EQUALS(B2(0, 0, 0, 0), 1);
    ASSERT_EQUALS(B2(1, 1, 1, 1), 2);

    // 4. Third cast float to int with copy
    Tensor<int> B3 = tensor_cast<int>(A);
    // Since B, B2 and B3 share the same data, they are now all in sync. with A
    ASSERT_EQUALS(B3(0, 0, 0, 0), 1);
    ASSERT_EQUALS(B3(1, 1, 1, 1), 4);
    ASSERT_EQUALS(B2(0, 0, 0, 0), 1);
    ASSERT_EQUALS(B2(1, 1, 1, 1), 4);
    ASSERT_EQUALS(B(0, 0, 0, 0), 1);
    ASSERT_EQUALS(B(1, 1, 1, 1), 4);
}

RUN_TESTS()
