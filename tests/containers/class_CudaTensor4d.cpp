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

#include "containers/CudaTensor.hpp"
#include "utils/UnitTest.hpp"
#include "utils/Utils.hpp"

using namespace N2D2;

TEST(CudaTensor4d, CudaTensor4d)
{
    REQUIRED(UnitTest::CudaDeviceExists());

    CudaTensor<double> A({0, 0, 0, 0});

    ASSERT_EQUALS(A.dimB(), 0U);
    ASSERT_EQUALS(A.size(), 0U);
    ASSERT_TRUE(A.empty());
}

TEST(CudaTensor4d, push_back)
{
    REQUIRED(UnitTest::CudaDeviceExists());

    Tensor<int> A({0, 0, 0, 0});
    CudaTensor<int> tensor;

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

    Tensor<int> A({dimX, dimY, dimZ});
    CudaTensor<int> tensor;

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

    CudaTensor<int> tensor({dimX, dimY, dimZ, dimB});

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

    CudaTensor<float> tensor;
    tensor.resize({dimX, dimY, dimZ, dimB}, 1.0);

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

    CudaTensor<float> tensor;

    tensor.resize({dimX, dimY, dimZ, dimB}, 0.0);

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

TEST(CudaTensor4d, cuda_tensor_cast_double_to_float)
{
    REQUIRED(UnitTest::CudaDeviceExists());

    CudaTensor<double> A({2, 3, 4, 5});
    A.fill(1.0);

    ASSERT_EQUALS(A(1, 1, 1, 1), 1.0);

    // 1. First cast double to float
    CudaTensor<float> B = cuda_tensor_cast<float>(A);
    // Data from A was converted to B
    ASSERT_EQUALS(B.dims(), A.dims());
    ASSERT_EQUALS(B(1, 1, 1, 1), 1.0);
    // Changes in B won't affect A
    B(1, 1, 1, 1) = 2.0;
    ASSERT_EQUALS(B(1, 1, 1, 1), 2.0);
    ASSERT_EQUALS(A(1, 1, 1, 1), 1.0);

    // 2. Cast in the same type (double to double)
    CudaTensor<double> C = cuda_tensor_cast<double>(A);
    // C shares the same data as A
    ASSERT_EQUALS(C.dims(), A.dims());
    C(1, 1, 1, 1) = 4.0;
    ASSERT_EQUALS(A(1, 1, 1, 1), 4.0);

    // 3. Second cast double to float
    // Since A was already casted to double with B, B2 shares the same data as B
    // In this case, no new allocation is made
    CudaTensor<float> B2 = cuda_tensor_cast_nocopy<float>(A);
    ASSERT_EQUALS(B2(0, 0, 0, 0), 1.0);
    ASSERT_EQUALS(B2(1, 1, 1, 1), 2.0);

    // 4. Third cast double to float with copy
    CudaTensor<float> B3 = cuda_tensor_cast<float>(A);
    // Since B, B2 and B3 share the same data, they are now all in sync. with A
    ASSERT_EQUALS(B3(0, 0, 0, 0), 1.0);
    ASSERT_EQUALS(B3(1, 1, 1, 1), 4.0);
    ASSERT_EQUALS(B2(0, 0, 0, 0), 1.0);
    ASSERT_EQUALS(B2(1, 1, 1, 1), 4.0);
    ASSERT_EQUALS(B(0, 0, 0, 0), 1.0);
    ASSERT_EQUALS(B(1, 1, 1, 1), 4.0);
}

TEST(CudaTensor4d, cuda_device_tensor_cast_double_to_float)
{
    REQUIRED(UnitTest::CudaDeviceExists());

    CudaTensor<double> A({2, 3, 4, 5});
    A.fill(1.0);

    ASSERT_EQUALS(A(1, 1, 1, 1), 1.0);
    A.synchronizeHToD();

    std::shared_ptr<CudaDeviceTensor<float> > devB
        = cuda_device_tensor_cast<float>(A);

    Tensor<float> B(A.dims(), 0.0);
    CHECK_CUDA_STATUS(cudaMemcpy(&(B.data())[0],
                                 devB->getDevicePtr(),
                                 A.size() * sizeof(float),
                                 cudaMemcpyDeviceToHost));

    ASSERT_EQUALS(B(1, 1, 1, 1), 1.0);
    B(1, 1, 1, 1) = 2.0;

    CHECK_CUDA_STATUS(cudaMemcpy(devB->getDevicePtr(),
                                 &(B.data())[0],
                                 A.size() * sizeof(float),
                                 cudaMemcpyHostToDevice));

    A.synchronizeDToH();
    ASSERT_EQUALS(A(1, 1, 1, 1), 1.0);

    std::shared_ptr<CudaDeviceTensor<float> > devC
        = cuda_device_tensor_cast_nocopy<float>(A);

    CHECK_CUDA_STATUS(cudaMemcpy(&(B.data())[0],
                                 devB->getDevicePtr(),
                                 A.size() * sizeof(float),
                                 cudaMemcpyDeviceToHost));

    ASSERT_EQUALS(B(0, 0, 0, 0), 1.0);
    ASSERT_EQUALS(B(1, 1, 1, 1), 2.0);

    A.deviceTensor() = *devB;
    A.synchronizeDToH();

    ASSERT_EQUALS(A(0, 0, 0, 0), 1.0);
    ASSERT_EQUALS(A(1, 1, 1, 1), 2.0);
}

TEST(CudaTensor4d, cuda_device_tensor_cast_float_to_double)
{
    REQUIRED(UnitTest::CudaDeviceExists());

    CudaTensor<float> A({2, 3, 4, 5});
    A.fill(1.0);

    ASSERT_EQUALS(A(1, 1, 1, 1), 1.0);
    A.synchronizeHToD();

    std::shared_ptr<CudaDeviceTensor<double> > devB
        = cuda_device_tensor_cast<double>(A);

    Tensor<double> B(A.dims(), 0.0);
    CHECK_CUDA_STATUS(cudaMemcpy(&(B.data())[0],
                                 devB->getDevicePtr(),
                                 A.size() * sizeof(double),
                                 cudaMemcpyDeviceToHost));

    ASSERT_EQUALS(B(1, 1, 1, 1), 1.0);
    B(1, 1, 1, 1) = 2.0;

    CHECK_CUDA_STATUS(cudaMemcpy(devB->getDevicePtr(),
                                 &(B.data())[0],
                                 A.size() * sizeof(double),
                                 cudaMemcpyHostToDevice));

    A.synchronizeDToH();
    ASSERT_EQUALS(A(1, 1, 1, 1), 1.0);

    std::shared_ptr<CudaDeviceTensor<double> > devC
        = cuda_device_tensor_cast_nocopy<double>(A);

    CHECK_CUDA_STATUS(cudaMemcpy(&(B.data())[0],
                                 devB->getDevicePtr(),
                                 A.size() * sizeof(double),
                                 cudaMemcpyDeviceToHost));

    ASSERT_EQUALS(B(0, 0, 0, 0), 1.0);
    ASSERT_EQUALS(B(1, 1, 1, 1), 2.0);

    A.deviceTensor() = *devB;
    A.synchronizeDToH();

    ASSERT_EQUALS(A(0, 0, 0, 0), 1.0);
    ASSERT_EQUALS(A(1, 1, 1, 1), 2.0);
}

RUN_TESTS()

#else

int main()
{
    return 0;
}

#endif
