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

#include "N2D2.hpp"

#include "DataFile/CsvDataFile.hpp"
#include "utils/UnitTest.hpp"

using namespace N2D2;

TEST(CsvDataFile, read)
{
    const std::string data = "1.0 0.5\n"
                             "2.0 0.6\n"
                             "3.0 0.7\n"
                             "4.0 0.8\n"
                             "5.0 0.9\n";

    UnitTest::FileWriteContent("CsvDataFile_read.csv", data);

    CsvDataFile dataFile;
    cv::Mat mat = dataFile.read("CsvDataFile_read.csv");

    ASSERT_EQUALS(mat.rows, 5);
    ASSERT_EQUALS(mat.cols, 2);

    for (unsigned int i = 0; i < 5; ++i) {
        ASSERT_EQUALS_DELTA(mat.at<double>(i, 0), i + 1.0, 1.0e-15);
        ASSERT_EQUALS_DELTA(mat.at<double>(i, 1), 0.5 + i * 0.1, 1.0e-15);
    }
}

TEST(CsvDataFile, write)
{
    const std::string data = "1.0 0.5\n"
                             "2.0 0.6\n"
                             "3.0 0.7\n"
                             "4.0 0.8\n"
                             "5.0 0.9\n";

    UnitTest::FileWriteContent("CsvDataFile_write.csv", data);

    CsvDataFile dataFile;
    dataFile.write("CsvDataFile_write2.csv",
                   dataFile.read("CsvDataFile_write.csv"));
    cv::Mat mat = dataFile.read("CsvDataFile_write2.csv");

    ASSERT_EQUALS(mat.rows, 5);
    ASSERT_EQUALS(mat.cols, 2);

    for (unsigned int i = 0; i < 5; ++i) {
        ASSERT_EQUALS_DELTA(mat.at<double>(i, 0), i + 1.0, 1.0e-15);
        ASSERT_EQUALS_DELTA(mat.at<double>(i, 1), 0.5 + i * 0.1, 1.0e-15);
    }
}

RUN_TESTS()
