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

#include "ROI/RectangularROI.hpp"
#include "Transformation/DistortionTransformation.hpp"
#include "Transformation/RescaleTransformation.hpp"
#include "utils/Gnuplot.hpp"
#include "utils/UnitTest.hpp"
#include "utils/Utils.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace N2D2;

TEST_DATASET(DistortionTransformation,
             apply,
             (bool color),
             std::make_tuple(true),
             std::make_tuple(false))
{
    Random::mtSeed(0);

    RectangularROI<int> roi1(64, cv::Point(0, 0), 256, 256);
    RectangularROI<int> roi2(128, cv::Point(256, 0), 256, 256);
    RectangularROI<int> roi3(255, cv::Point(256, 256), 256, 256);

    cv::Mat labels(512, 512, CV_32SC1, cv::Scalar(0));
    roi1.append(labels);
    roi2.append(labels);
    roi3.append(labels);

    DistortionTransformation trans;
    trans.setParameter("ElasticGaussianSize", 15U);
    trans.setParameter("ElasticSigma", 6.0);
    trans.setParameter("ElasticScaling", 36.0);
    trans.setParameter("Scaling", 10.0);
    trans.setParameter("Rotation", 10.0);

    cv::Mat img
        = cv::imread("tests_data/Lenna.png",
#if CV_MAJOR_VERSION >= 3
                     (color) ? cv::IMREAD_COLOR : cv::IMREAD_GRAYSCALE);
#else
                     (color) ? CV_LOAD_IMAGE_COLOR : CV_LOAD_IMAGE_GRAYSCALE);
#endif

    if (!img.data)
        throw std::runtime_error(
            "Could not open or find image: tests_data/Lenna.png");

    const int cols = img.cols;
    const int rows = img.rows;

    trans.apply(img, labels);

    std::ostringstream fileName;
    fileName << "DistortionTransformation_apply(C" << color << ")[frame].png";

    Utils::createDirectories("Transformation");
    if (!cv::imwrite("Transformation/" + fileName.str(), img))
        throw std::runtime_error("Unable to write image: Transformation/"
                                 + fileName.str());

    ASSERT_EQUALS(img.cols, cols);
    ASSERT_EQUALS(img.rows, rows);

    fileName.str(std::string());
    fileName << "DistortionTransformation_apply(C" << color << ")[labels].png";

    if (!cv::imwrite("Transformation/" + fileName.str(), labels))
        throw std::runtime_error("Unable to write image: Transformation/"
                                 + fileName.str());

    ASSERT_EQUALS(img.cols, labels.cols);
    ASSERT_EQUALS(img.rows, labels.rows);
}

TEST(DistortionTransformation, benchmark)
{
    Random::mtSeed(0);

    DistortionTransformation trans;
    trans.setParameter("ElasticGaussianSize", 15U);
    trans.setParameter("ElasticSigma", 6.0);
    trans.setParameter("ElasticScaling", 36.0);
    trans.setParameter("Scaling", 10.0);
    trans.setParameter("Rotation", 10.0);

    Utils::createDirectories("Transformation");
    const std::string fileName
        = "Transformation/DistortionTransformation_benchmark.dat";
    std::ofstream data(fileName);

    if (!data.good())
        throw std::runtime_error("Unable to write file: " + fileName);

    for (unsigned int i = 0; i < 3; ++i) {
#ifdef _OPENMP
        if (i == 2)
            omp_set_num_threads(1);
#endif

        for (unsigned int size = 2; size < 1024; size *= 2) {
            cv::Mat img
                = cv::imread("tests_data/Lenna.png",
#if CV_MAJOR_VERSION >= 3
                    cv::IMREAD_GRAYSCALE);
#else
                    CV_LOAD_IMAGE_GRAYSCALE);
#endif

            if (!img.data)
                throw std::runtime_error(
                    "Could not open or find image: tests_data/Lenna.png");

            RescaleTransformation resTrans(size, size);
            resTrans.apply(img);

            std::chrono::high_resolution_clock::time_point startTime
                = std::chrono::high_resolution_clock::now();
            trans.apply(img);
            std::chrono::high_resolution_clock::time_point curTime
                = std::chrono::high_resolution_clock::now();
            const double timeElapsed
                = std::chrono::duration_cast
                  <std::chrono::duration<double> >(curTime - startTime).count();

            if (i > 0)
                data << size << " " << timeElapsed << "\n";
        }

        if (i > 0)
            data << "\n\n";
    }

    data.close();

    Gnuplot gnuplot;
    gnuplot.set("grid");
    gnuplot.set("logscale xy");
    gnuplot.setXlabel("Size");
    gnuplot.setYlabel("Time (s)");
    gnuplot.saveToFile(fileName);
    gnuplot.plot(
        fileName,
        "index 0 using 1:2 with linespoints lt 1 title \"with OpenMP\", "
        "'' index 1 using 1:2 with linespoints lt 2 title \"w/o OpenMP\"");
}

RUN_TESTS()
