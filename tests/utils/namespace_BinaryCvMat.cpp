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

#include "utils/BinaryCvMat.hpp"
#include "utils/UnitTest.hpp"

using namespace N2D2;

TEST(BinaryCvMat, read_write)
{
    std::vector<cv::Mat> data;
    data.push_back(cv::imread("tests_data/Lenna.png", CV_LOAD_IMAGE_UNCHANGED));
    data.push_back(cv::imread("tests_data/SIPI_Jelly_Beans_4.1.07.tiff",
                              CV_LOAD_IMAGE_UNCHANGED));

    // Write
    const std::string fileName = "BinaryCvMat_read_write.bin";

    std::ofstream os(fileName.c_str(), std::ios::binary);

    if (!os.good())
        throw std::runtime_error("Could not create file: " + fileName);

    for (std::vector<cv::Mat>::const_iterator it = data.begin(),
                                              itEnd = data.end();
         it != itEnd;
         ++it)
        BinaryCvMat::write(os, *it);

    os.close();

    // Read
    std::ifstream is(fileName.c_str(), std::ios::binary);

    if (!is.good())
        throw std::runtime_error("Could not read file: " + fileName);

    std::vector<cv::Mat> data2;

    do {
        data2.push_back(cv::Mat());
        BinaryCvMat::read(is, data2.back());
    } while (is.peek() != EOF);

    // Check
    ASSERT_EQUALS(data.size(), data2.size());

    for (unsigned int k = 0; k < data.size(); ++k) {
        ASSERT_EQUALS(data[k].rows, data2[k].rows);
        ASSERT_EQUALS(data[k].cols, data2[k].cols);
        ASSERT_EQUALS(data[k].type(), data2[k].type());

        for (int i = 0; i < data[k].rows; ++i) {
            for (int j = 0; j < data[k].cols; ++j) {
                ASSERT_EQUALS(data[k].at<unsigned char>(i, j),
                              data2[k].at<unsigned char>(i, j));
            }
        }
    }
}

RUN_TESTS()
