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

#include "DataFile/CsvDataFile.hpp"
#include "Transformation/ApodizationTransformation.hpp"
#include "utils/UnitTest.hpp"

using namespace N2D2;

TEST(ApodizationTransformation, apply)
{
    const std::string data = "1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0\n"
                             "0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4\n";

    UnitTest::FileWriteContent("ApodizationTransformation_read.csv", data);

    CsvDataFile dataFile;
    cv::Mat mat = dataFile.read("ApodizationTransformation_read.csv");

    const Hann<double> windowFunction = Hann<double>();
    ApodizationTransformation trans(windowFunction, 10);

    trans.apply(mat);

    ASSERT_EQUALS(mat.cols, 10);
    ASSERT_EQUALS(mat.rows, 2);
    ASSERT_EQUALS(mat.channels(), 1);

    ASSERT_EQUALS_DELTA(mat.at<double>(0, 0), 0.0, 1.0e-5);
    ASSERT_EQUALS_DELTA(mat.at<double>(0, 1), 0.11698, 1.0e-5);
    ASSERT_EQUALS_DELTA(mat.at<double>(0, 2), 0.41318, 1.0e-5);
    ASSERT_EQUALS_DELTA(mat.at<double>(0, 3), 0.75000, 1.0e-5);
    ASSERT_EQUALS_DELTA(mat.at<double>(0, 4), 0.96985, 1.0e-5);
    ASSERT_EQUALS_DELTA(mat.at<double>(0, 5), 0.96985, 1.0e-5);
    ASSERT_EQUALS_DELTA(mat.at<double>(0, 6), 0.75000, 1.0e-5);
    ASSERT_EQUALS_DELTA(mat.at<double>(0, 7), 0.41318, 1.0e-5);
    ASSERT_EQUALS_DELTA(mat.at<double>(0, 8), 0.11698, 1.0e-5);
    ASSERT_EQUALS_DELTA(mat.at<double>(0, 9), 0.0, 1.0e-5);

    ASSERT_EQUALS_DELTA(mat.at<double>(1, 0), 0.5 * 0.0, 1.0e-5);
    ASSERT_EQUALS_DELTA(mat.at<double>(1, 1), 0.6 * 0.11698, 1.0e-5);
    ASSERT_EQUALS_DELTA(mat.at<double>(1, 2), 0.7 * 0.41318, 1.0e-5);
    ASSERT_EQUALS_DELTA(mat.at<double>(1, 3), 0.8 * 0.75000, 1.0e-5);
    ASSERT_EQUALS_DELTA(mat.at<double>(1, 4), 0.9 * 0.96985, 1.0e-5);
    ASSERT_EQUALS_DELTA(mat.at<double>(1, 5), 1.0 * 0.96985, 1.0e-5);
    ASSERT_EQUALS_DELTA(mat.at<double>(1, 6), 1.1 * 0.75000, 1.0e-5);
    ASSERT_EQUALS_DELTA(mat.at<double>(1, 7), 1.2 * 0.41318, 1.0e-5);
    ASSERT_EQUALS_DELTA(mat.at<double>(1, 8), 1.3 * 0.11698, 1.0e-5);
    ASSERT_EQUALS_DELTA(mat.at<double>(1, 9), 1.4 * 0.0, 1.0e-5);
}

RUN_TESTS()
