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

#include "utils/Kernel.hpp"
#include "utils/UnitTest.hpp"
#include "utils/Utils.hpp"

using namespace N2D2;

TEST_DATASET(Kernel,
             Kernel__rows_cols,
             (std::string kernelStr, unsigned int sizeX, unsigned int sizeY),
             std::make_tuple("", 0U, 3U),
             std::make_tuple("1.0 1.0 0.0", 1U, 3U),
             std::make_tuple("1.0 1.0 0.0", 3U, 1U),
             std::make_tuple("1.0 1.0 0.0 1.0", 2U, 2U))
{
    const Kernel<double> kernel(kernelStr, sizeX, sizeY);

    ASSERT_EQUALS(kernel.cols(), sizeX);
    ASSERT_EQUALS(kernel.rows(), sizeY);
    ASSERT_EQUALS(kernel.size(), sizeX * sizeY);
    ASSERT_TRUE(kernel.empty() == (sizeX * sizeY == 0));
}

TEST_DATASET(Kernel,
             Kernel__rows_cols_throw,
             (std::string kernelStr, unsigned int sizeX, unsigned int sizeY),
             std::make_tuple("1", 0U, 3U),
             std::make_tuple("1.0 1.0 0.0 1.0", 1U, 3U),
             std::make_tuple("1.0 1.0", 3U, 1U))
{
    ASSERT_THROW_ANY(const Kernel<double> kernel(kernelStr, sizeX, sizeY));
}

TEST_DATASET(Kernel,
             Kernel__no_rows_cols,
             (std::string kernelStr, unsigned int sizeX, unsigned int sizeY),
             std::make_tuple("", 0U, 0U),
             std::make_tuple("1.0 1.0 0.0 1.0", 2U, 2U))
{
    const Kernel<double> kernel(kernelStr);

    ASSERT_EQUALS(kernel.cols(), sizeX);
    ASSERT_EQUALS(kernel.rows(), sizeY);
    ASSERT_EQUALS(kernel.size(), sizeX * sizeY);
    ASSERT_TRUE(kernel.empty() == (sizeX * sizeY == 0));
}

TEST_DATASET(Kernel,
             Kernel__no_rows_cols_throw,
             (std::string kernelStr),
             std::make_tuple("1.0 0.0"),
             std::make_tuple("1.0 1.0 0.0 1.0 0.0"))
{
    ASSERT_THROW_ANY(const Kernel<double> kernel(kernelStr));
}

#if CV_MAJOR_VERSION >= 2 && CV_MINOR_VERSION >= 4
TEST(GaborKernel, GaborKernel)
{
    const unsigned int sizeX = 5;
    const unsigned int sizeY = 5;
    const double theta = M_PI / 4.0;
    const double sigma = std::sqrt(2.0);
    const double lambda = 10.0;
    const double psi = M_PI / 2.0;
    const double gamma = 0.5;

    const Kernel<double> A = GaborKernel
        <double>(sizeX, sizeY, theta, sigma, lambda, psi, gamma, false);

    // cv::getGaborKernel only works with square and odd-size kernels
    const cv::Mat unflippedB = cv::getGaborKernel(
        cv::Size(sizeX, sizeY), sigma, theta, lambda, gamma, psi, CV_64F);
    cv::Mat B;
    cv::flip(unflippedB, B, -1);

    ASSERT_EQUALS((int)A.rows(), B.rows);
    ASSERT_EQUALS((int)A.cols(), B.cols);

    const double sigmaX = sigma;
    const double sigmaY = sigma / gamma;
    const double factor = 1.0 / (2.0 * M_PI * sigmaX * sigmaY);

    for (unsigned int row = 0; row < sizeY; ++row) {
        for (unsigned int col = 0; col < sizeX; ++col) {
            // std::cout << "(" << row << "," << col << ") = " << A(row,col) <<
            // "   " << factor*B.at<double>(row,col) << std::endl;
            ASSERT_EQUALS_DELTA(
                A(row, col), factor * B.at<double>(row, col), 1.0e-16);
        }
    }
}
#endif

RUN_TESTS()
