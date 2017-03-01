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

#ifdef CUDA

#include "containers/CudaTensor4d.hpp"
#include "utils/UnitTest.hpp"

using namespace N2D2;

TEST(CudaTensor4d, CudaTensor4d)
{
    REQUIRED(UnitTest::CudaDeviceExists());

    CudaTensor4d<double> A;

    ASSERT_EQUALS(A.dimB(), 0U);
    ASSERT_EQUALS(A.size(), 0U);
    ASSERT_TRUE(A.empty());
}

TEST(CudaTensor4d, push_back)
{
    REQUIRED(UnitTest::CudaDeviceExists());

    Tensor3d<int> A;
    CudaTensor4d<int> tensor;

    ASSERT_EQUALS(tensor.dimB(), 0U);
    ASSERT_EQUALS(tensor.size(), 0U);
    ASSERT_TRUE(tensor.empty());

    tensor.push_back(A);

    ASSERT_EQUALS(tensor.dimB(), 1U);
    ASSERT_EQUALS(tensor.size(), 0U);
    ASSERT_TRUE(tensor.empty());
}

TEST_DATASET(CudaTensor4d,
             dims,
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
    REQUIRED(UnitTest::CudaDeviceExists());

    Tensor3d<int> A(dimX, dimY, dimZ);
    CudaTensor4d<int> tensor;

    for (unsigned int b = 0; b < dimB; ++b)
        tensor.push_back(A);

    ASSERT_EQUALS(tensor.dimB(), dimB);
    ASSERT_EQUALS(tensor.size(), dimX * dimY * dimZ * dimB);
}

TEST_DATASET(CudaTensor4d,
             dims2,
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
    REQUIRED(UnitTest::CudaDeviceExists());

    CudaTensor4d<int> tensor(dimX, dimY, dimZ, dimB);

    ASSERT_EQUALS(tensor.dimB(), dimB);
    ASSERT_EQUALS(tensor.size(), dimX * dimY * dimZ * dimB);
}

TEST_DATASET(CudaTensor4d,
             resize_sync_DToH,
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
    REQUIRED(UnitTest::CudaDeviceExists());

    CudaTensor4d<float> tensor;
    tensor.resize(dimX, dimY, dimZ, dimB, 1.0);

    std::vector<float> hostDest(dimX * dimY * dimZ * dimB, 0.0);
    CHECK_CUDA_STATUS(cudaMemcpy(&hostDest[0],
                                 tensor.getDevicePtr(),
                                 dimX * dimY * dimZ * dimB * sizeof(float),
                                 cudaMemcpyDeviceToHost));

    for (std::vector<float>::iterator it = hostDest.begin(),
                                      itEnd = hostDest.end();
         it != itEnd;
         ++it) {
        ASSERT_EQUALS((*it), 1.0);
    }
}

TEST_DATASET(CudaTensor4d,
             resize_partial_sync_DToH,
             (unsigned int dimX,
              unsigned int dimY,
              unsigned int dimZ,
              unsigned int dimB),
             std::make_tuple(1U, 3U, 1U, 1U),
             std::make_tuple(3U, 1U, 1U, 2U),
             std::make_tuple(1U, 3U, 2U, 2U),
             std::make_tuple(3U, 1U, 1U, 3U),
             std::make_tuple(3U, 3U, 4U, 10U),
             std::make_tuple(12U, 34U, 1U, 1U),
             std::make_tuple(34U, 12U, 5U, 21U))
{
    REQUIRED(UnitTest::CudaDeviceExists());

    CudaTensor4d<float> tensor;

    tensor.resize(dimX, dimY, dimZ, dimB, 0.0);

    for (unsigned int i = 0; i < dimB; ++i) {
        tensor(dimX - 1, dimY - 1, dimZ - 1, i) = 1.0;
        tensor.synchronizeHToD(i * dimZ * dimX * dimY + (dimZ - 1) * dimX * dimY
                               + (dimY - 1) * dimX + dimX - 1,
                               1);
        tensor(dimX - 1, dimY - 1, dimZ - 1, i) = 0.0;
        tensor.synchronizeDToH(i * dimZ * dimX * dimY + (dimZ - 1) * dimX * dimY
                               + (dimY - 1) * dimX + dimX - 1,
                               1);

        ASSERT_EQUALS(tensor(dimX - 1, dimY - 1, dimZ - 1, i), 1.0);
    }

    for (unsigned int i = 0; i < dimZ; ++i) {
        tensor(dimX - 1, dimY - 1, i, dimB - 1) = 2.0;
        tensor.synchronizeHToD((dimB - 1) * dimZ * dimX * dimY + i * dimX * dimY
                               + (dimY - 1) * dimX + dimX - 1,
                               1);
        tensor(dimX - 1, dimY - 1, i, dimB - 1) = 0.0;
        tensor.synchronizeDToH((dimB - 1) * dimZ * dimX * dimY + i * dimX * dimY
                               + (dimY - 1) * dimX + dimX - 1,
                               1);

        ASSERT_EQUALS(tensor(dimX - 1, dimY - 1, i, dimB - 1), 2.0);
    }

    for (unsigned int i = 0; i < dimY; ++i) {
        tensor(dimX - 1, i, dimZ - 1, dimB - 1) = 3.0;
        tensor.synchronizeHToD((dimB - 1) * dimZ * dimX * dimY
                               + (dimZ - 1) * dimX * dimY + i * dimX + dimX - 1,
                               1);
        tensor(dimX - 1, i, dimZ - 1, dimB - 1) = 0.0;
        tensor.synchronizeDToH((dimB - 1) * dimZ * dimX * dimY
                               + (dimZ - 1) * dimX * dimY + i * dimX + dimX - 1,
                               1);

        ASSERT_EQUALS(tensor(dimX - 1, i, dimZ - 1, dimB - 1), 3.0);
    }

    for (unsigned int i = 0; i < dimX; ++i) {
        tensor(i, dimY - 1, dimZ - 1, dimB - 1) = 4.0;
        tensor.synchronizeHToD((dimB - 1) * dimZ * dimX * dimY
                               + (dimZ - 1) * dimX * dimY + (dimY - 1) * dimX
                               + i,
                               1);
        tensor(i, dimY - 1, dimZ - 1, dimB - 1) = 0.0;
        tensor.synchronizeDToH((dimB - 1) * dimZ * dimX * dimY
                               + (dimZ - 1) * dimX * dimY + (dimY - 1) * dimX
                               + i,
                               1);

        ASSERT_EQUALS(tensor(i, dimY - 1, dimZ - 1, dimB - 1), 4.0);
    }
}

RUN_TESTS()

#else

int main()
{
    return 0;
}

#endif
