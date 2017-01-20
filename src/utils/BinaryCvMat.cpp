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

#include "utils/BinaryCvMat.hpp"

void N2D2::BinaryCvMat::write(std::ostream& os, const cv::Mat& mat)
{
    cv::Mat matCont = (mat.isContinuous()) ? mat : mat.clone();

    const int type = matCont.type();
    os.write(reinterpret_cast<const char*>(&matCont.rows),
             sizeof(matCont.rows));
    os.write(reinterpret_cast<const char*>(&matCont.cols),
             sizeof(matCont.cols));
    os.write(reinterpret_cast<const char*>(&type), sizeof(type));
    os.write(reinterpret_cast<const char*>(matCont.data),
             matCont.elemSize() * matCont.rows * matCont.cols);

    if (!os.good())
        throw std::runtime_error("Error writing cvMat binary file");
}

void N2D2::BinaryCvMat::write(const std::string& fileName, const cv::Mat& mat)
{
    std::ofstream os(fileName.c_str(), std::ios::binary);

    if (!os.good())
        throw std::runtime_error("Could not create cvMat binary file: "
                                 + fileName);

    write(os, mat);
}

void N2D2::BinaryCvMat::read(std::istream& is, cv::Mat& mat)
{
    int rows, cols, type;
    is.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    is.read(reinterpret_cast<char*>(&cols), sizeof(cols));
    is.read(reinterpret_cast<char*>(&type), sizeof(type));

    if (!is.good())
        throw std::runtime_error("Error reading cvMat binary file");

    mat.release();
    mat.create(rows, cols, type);
    is.read(reinterpret_cast<char*>(mat.data),
            mat.elemSize() * mat.rows * mat.cols);

    if (!is.good())
        throw std::runtime_error("Error reading cvMat binary file");
}

void N2D2::BinaryCvMat::read(const std::string& fileName, cv::Mat& mat)
{
    std::ifstream is(fileName.c_str(), std::ios::binary);

    if (!is.good())
        throw std::runtime_error("Could not read cvMat binary file: "
                                 + fileName);

    read(is, mat);
}
