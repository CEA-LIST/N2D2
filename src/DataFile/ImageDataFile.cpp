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

#include "DataFile/ImageDataFile.hpp"
#include "utils/Utils.hpp"

N2D2::Registrar<N2D2::DataFile>
N2D2::ImageDataFile::mRegistrar({// Windows bitmaps
                                "bmp",
                                "dib",
                                // JPEG files
                                "jpeg",
                                "jpg",
                                "jpe",
                                // JPEG 2000 files
                                "jp2",
                                // Portable Network Graphics
                                "png",
                                // Portable image format
                                "pbm",
                                "pgm",
                                "ppm",
                                // Sun rasters
                                "sr",
                                "ras",
                                // TIFF files
                                "tiff",
                                "tif"},
                                N2D2::ImageDataFile::create);

cv::Mat N2D2::ImageDataFile::read(const std::string& fileName)
{
    cv::Mat data;

    try {
#if CV_MAJOR_VERSION >= 3
        data = cv::imread(fileName, cv::IMREAD_UNCHANGED);
#else
        data = cv::imread(fileName, CV_LOAD_IMAGE_UNCHANGED);
#endif
    }
    catch (...) {
        std::cout << Utils::cwarning
            << "ImageDataFile::read(): unable to read possibly corrupted file: "
            << fileName << std::endl;
        std::cout << "The following exception occurred in OpenCV:"
            << Utils::cdef << std::endl;
        throw;
    }

    if (!data.data)
        throw std::runtime_error("ImageDataFile::read(): unable to read image: "
                                 + fileName);

    return data;
}

void N2D2::ImageDataFile::write(const std::string& fileName,
                                const cv::Mat& data)
{
    if (!cv::imwrite(fileName, data))
        throw std::runtime_error(
            "ImageDataFile::write(): unable to write image: " + fileName);
}
