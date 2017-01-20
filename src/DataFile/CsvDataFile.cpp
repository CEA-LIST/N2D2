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

N2D2::Registrar<N2D2::DataFile> N2D2::CsvDataFile::mRegistrar(
    N2D2::CsvDataFile::create, "csv", "dat", "txt", NULL);

N2D2::CsvDataFile::CsvDataFile()
{
    // ctor
    setReadDelimiters();
}

void N2D2::CsvDataFile::setReadDelimiters(const std::string& delimiters)
{
    mCsvLocale = (delimiters.empty())
                     ? std::locale(std::locale(),
                                   new N2D2::Utils::streamIgnore(",; \t"))
                     : std::locale(std::locale(),
                                   new N2D2::Utils::streamIgnore(delimiters));
}

cv::Mat N2D2::CsvDataFile::read(const std::string& fileName)
{
    return ((cv::Mat)read<double>(fileName)).clone();
}

void N2D2::CsvDataFile::write(const std::string& fileName, const cv::Mat& data)
{
    switch (data.depth()) {
    case CV_8U:
        write(fileName, Tensor2d<unsigned char>(data));
        break;
    case CV_8S:
        write(fileName, Tensor2d<char>(data));
        break;
    case CV_16U:
        write(fileName, Tensor2d<unsigned short>(data));
        break;
    case CV_16S:
        write(fileName, Tensor2d<short>(data));
        break;
    case CV_32S:
        write(fileName, Tensor2d<int>(data));
        break;
    case CV_32F:
        write(fileName, Tensor2d<float>(data));
        break;
    case CV_64F:
        write(fileName, Tensor2d<double>(data));
        break;
    default:
        throw std::runtime_error(
            "Cannot convert cv::Mat to Tensor2d: incompatible types.");
    }
}
