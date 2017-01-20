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

#include "Database/DIR_Database.hpp"
#include "Database/MNIST_IDX_Database.hpp"
#include "N2D2.hpp"
#include "StimuliProvider.hpp"
#include "Transformation/ChannelExtractionTransformation.hpp"
#include "Transformation/FlipTransformation.hpp"
#include "Transformation/FilterTransformation.hpp"
#include "Transformation/RescaleTransformation.hpp"
#include "utils/UnitTest.hpp"

using namespace N2D2;

TEST_DATASET(StimuliProvider,
             StimuliProvider,
             (unsigned int sizeX, unsigned int sizeY, unsigned int batchSize),
             std::make_tuple(10U, 30U, 1U),
             std::make_tuple(30U, 10U, 1U),
             std::make_tuple(30U, 30U, 1U),
             std::make_tuple(10U, 30U, 10U),
             std::make_tuple(30U, 10U, 10U),
             std::make_tuple(30U, 30U, 10U))
{
    Database database;
    StimuliProvider sp(database, sizeX, sizeY, 1, batchSize);

    ASSERT_EQUALS(sp.getSizeX(), sizeX);
    ASSERT_EQUALS(sp.getSizeY(), sizeY);
    ASSERT_EQUALS(sp.getBatchSize(), batchSize);
    ASSERT_EQUALS(sp.getNbChannels(), 1U);
    ASSERT_EQUALS(sp.isCompositeStimuli(), false);
    ASSERT_EQUALS(sp.getTransformation(Database::Learn).size(), 0U);
    ASSERT_EQUALS(sp.getTransformation(Database::Test).size(), 0U);
    ASSERT_EQUALS(sp.getTransformation(Database::Validation).size(), 0U);
    ASSERT_EQUALS(sp.getTransformation(Database::Learn).empty(), true);
    ASSERT_EQUALS(sp.getTransformation(Database::Test).empty(), true);
    ASSERT_EQUALS(sp.getTransformation(Database::Validation).empty(), true);
    ASSERT_EQUALS(sp.getOnTheFlyTransformation(Database::Learn).size(), 0U);
    ASSERT_EQUALS(sp.getOnTheFlyTransformation(Database::Test).size(), 0U);
    ASSERT_EQUALS(sp.getOnTheFlyTransformation(Database::Validation).size(),
                  0U);
    ASSERT_EQUALS(sp.getOnTheFlyTransformation(Database::Learn).empty(), true);
    ASSERT_EQUALS(sp.getOnTheFlyTransformation(Database::Test).empty(), true);
    ASSERT_EQUALS(sp.getOnTheFlyTransformation(Database::Validation).empty(),
                  true);
}

TEST_DATASET(StimuliProvider,
             addTransformation,
             (bool cache),
             std::make_tuple(false),
             std::make_tuple(true))
{
    DIR_Database database;
    database.loadFile("tests_data/Lenna.png", "Lenna");
    database.loadFile("tests_data/SIPI_Jelly_Beans_4.1.07.tiff", "Jelly_Beans");

    StimuliProvider sp(database, 256, 256);
    sp.addTransformation(GrayChannelExtractionTransformation());
    sp.addTransformation(RescaleTransformation(256, 256));

    ASSERT_EQUALS(sp.getNbChannels(), 1U);
    ASSERT_EQUALS(sp.getTransformation(Database::Learn).size(), 2U);
    ASSERT_EQUALS(sp.getTransformation(Database::Learn).empty(), false);
    ASSERT_EQUALS(sp.getOnTheFlyTransformation(Database::Learn).size(), 0U);
    ASSERT_EQUALS(sp.getOnTheFlyTransformation(Database::Learn).empty(), true);

    if (cache)
        sp.setCachePath("_cache_1");

    for (int i = 0; i < 4; ++i) {
        sp.readStimulus(i % 2, Database::Learn);

        Tensor2d<Float_T> data = sp.getData(0);

        ASSERT_EQUALS(data.dimX(), 256U);
        ASSERT_EQUALS(data.dimY(), 256U);

        std::ostringstream fileName;
        fileName << "StimuliProvider_addTransformation(C" << cache << "-" << i
                 << ").png";

        cv::Mat mat(data), matNorm;
        mat.convertTo(matNorm, CV_8UC1, 255.0);

        if (!cv::imwrite(fileName.str(), matNorm))
            throw std::runtime_error("Unable to write image: "
                                     + fileName.str());

        const Tensor4d<Float_T>& fullData = sp.getData();

        ASSERT_EQUALS(fullData.dimX(), 256U);
        ASSERT_EQUALS(fullData.dimY(), 256U);
        ASSERT_EQUALS(fullData.dimZ(), 1U);
        ASSERT_EQUALS(fullData.dimB(), 1U);
    }
}

TEST_DATASET(StimuliProvider,
             addTransformation_bis,
             (Database::StimuliSetMask setMask),
             std::make_tuple(Database::LearnOnly),
             std::make_tuple(Database::ValidationOnly),
             std::make_tuple(Database::TestOnly),
             std::make_tuple(Database::NoLearn),
             std::make_tuple(Database::NoValidation),
             std::make_tuple(Database::NoTest),
             std::make_tuple(Database::All))
{
    DIR_Database database;
    database.loadFile("tests_data/Lenna.png", "Lenna");
    database.loadFile("tests_data/SIPI_Jelly_Beans_4.1.07.tiff", "Jelly_Beans");

    StimuliProvider sp(database, 256, 256);
    sp.addTransformation(GrayChannelExtractionTransformation(), setMask);
    sp.addTransformation(RescaleTransformation(256, 256), setMask);

    ASSERT_EQUALS(sp.getNbChannels(), 1U);
    ASSERT_EQUALS(sp.getTransformation(Database::Learn).size(),
                  (setMask == Database::LearnOnly
                   || setMask == Database::NoValidation
                   || setMask == Database::NoTest || setMask == Database::All)
                      ? 2U
                      : 0U);
    ASSERT_EQUALS(sp.getTransformation(Database::Validation).size(),
                  (setMask == Database::ValidationOnly
                   || setMask == Database::NoLearn
                   || setMask == Database::NoTest || setMask == Database::All)
                      ? 2U
                      : 0U);
    ASSERT_EQUALS(sp.getTransformation(Database::Test).size(),
                  (setMask == Database::TestOnly || setMask == Database::NoLearn
                   || setMask == Database::NoValidation
                   || setMask == Database::All)
                      ? 2U
                      : 0U);
}

TEST_DATASET(StimuliProvider,
             addOnTheFlyTransformation,
             (bool cache),
             std::make_tuple(false),
             std::make_tuple(true))
{
    Random::mtSeed(0);

    DIR_Database database;
    database.loadFile("tests_data/Lenna.png", "Lenna");
    database.loadFile("tests_data/SIPI_Jelly_Beans_4.1.07.tiff", "Jelly_Beans");

    StimuliProvider sp(database, 256, 256);
    sp.addTransformation(GrayChannelExtractionTransformation());
    sp.addTransformation(RescaleTransformation(256, 256));

    FlipTransformation trans(true, true);

    sp.addOnTheFlyTransformation(trans);

    ASSERT_EQUALS(sp.getNbChannels(), 1U);
    ASSERT_EQUALS(sp.getTransformation(Database::Learn).size(), 2U);
    ASSERT_EQUALS(sp.getTransformation(Database::Learn).empty(), false);
    ASSERT_EQUALS(sp.getOnTheFlyTransformation(Database::Learn).size(), 1U);
    ASSERT_EQUALS(sp.getOnTheFlyTransformation(Database::Learn).empty(), false);

    if (cache)
        sp.setCachePath("_cache_2");

    for (int i = 0; i < 4; ++i) {
        sp.readStimulus(i % 2, Database::Learn);

        Tensor2d<Float_T> data = sp.getData(0);

        ASSERT_EQUALS(data.dimX(), 256U);
        ASSERT_EQUALS(data.dimY(), 256U);

        std::ostringstream fileName;
        fileName << "StimuliProvider_addOnTheFlyTransformation(C" << cache
                 << "-" << i << ").png";

        cv::Mat mat(data), matNorm;
        mat.convertTo(matNorm, CV_8UC1, 255.0);

        if (!cv::imwrite(fileName.str(), matNorm))
            throw std::runtime_error("Unable to write image: "
                                     + fileName.str());

        const Tensor4d<Float_T>& fullData = sp.getData();

        ASSERT_EQUALS(fullData.dimX(), 256U);
        ASSERT_EQUALS(fullData.dimY(), 256U);
        ASSERT_EQUALS(fullData.dimZ(), 1U);
        ASSERT_EQUALS(fullData.dimB(), 1U);
    }
}

TEST_DATASET(StimuliProvider,
             addChannelTransformation,
             (bool cache),
             std::make_tuple(false),
             std::make_tuple(true))
{
    DIR_Database database;
    database.loadFile("tests_data/Lenna.png", "Lenna");
    database.loadFile("tests_data/SIPI_Jelly_Beans_4.1.07.tiff", "Jelly_Beans");

    StimuliProvider sp(database, 256, 256, 1, 1, false);
    sp.addTransformation(RescaleTransformation(256, 256));
    sp.addChannelTransformation(HueChannelExtractionTransformation());
    sp.addChannelTransformation(SaturationChannelExtractionTransformation());
    sp.addChannelTransformation(ValueChannelExtractionTransformation());

    ASSERT_EQUALS(sp.getNbChannels(), 3U);
    ASSERT_EQUALS(sp.getTransformation(Database::Learn).size(), 1U);
    ASSERT_EQUALS(sp.getTransformation(Database::Learn).empty(), false);
    ASSERT_EQUALS(sp.getOnTheFlyTransformation(Database::Learn).size(), 0U);
    ASSERT_EQUALS(sp.getOnTheFlyTransformation(Database::Learn).empty(), true);
    ASSERT_EQUALS(sp.getChannelTransformation(0, Database::Learn).size(), 1U);
    ASSERT_EQUALS(sp.getChannelTransformation(1, Database::Learn).size(), 1U);
    ASSERT_EQUALS(sp.getChannelTransformation(2, Database::Learn).size(), 1U);
    ASSERT_EQUALS(sp.getChannelTransformation(0, Database::Learn).empty(),
                  false);
    ASSERT_EQUALS(sp.getChannelTransformation(1, Database::Learn).empty(),
                  false);
    ASSERT_EQUALS(sp.getChannelTransformation(2, Database::Learn).empty(),
                  false);
    ASSERT_EQUALS(
        sp.getChannelOnTheFlyTransformation(0, Database::Learn).size(), 0U);
    ASSERT_EQUALS(
        sp.getChannelOnTheFlyTransformation(1, Database::Learn).size(), 0U);
    ASSERT_EQUALS(
        sp.getChannelOnTheFlyTransformation(2, Database::Learn).size(), 0U);
    ASSERT_EQUALS(
        sp.getChannelOnTheFlyTransformation(0, Database::Learn).empty(), true);
    ASSERT_EQUALS(
        sp.getChannelOnTheFlyTransformation(1, Database::Learn).empty(), true);
    ASSERT_EQUALS(
        sp.getChannelOnTheFlyTransformation(2, Database::Learn).empty(), true);

    if (cache)
        sp.setCachePath("_cache_3");

    for (int i = 0; i < 4; ++i) {
        sp.readStimulus(i % 2, Database::Learn);

        for (int channel = 0; channel < 3; ++channel) {
            Tensor2d<Float_T> data = sp.getData(channel);

            ASSERT_EQUALS(data.dimX(), 256U);
            ASSERT_EQUALS(data.dimY(), 256U);

            std::ostringstream fileName;
            fileName << "StimuliProvider_addChannelTransformation(C" << cache
                     << "-" << i << ")[" << channel << "].png";

            cv::Mat mat(data), matNorm;
            mat.convertTo(matNorm, CV_8UC1, 255.0);

            if (!cv::imwrite(fileName.str(), matNorm))
                throw std::runtime_error("Unable to write image: "
                                         + fileName.str());
        }

        const Tensor4d<Float_T>& fullData = sp.getData();

        ASSERT_EQUALS(fullData.dimX(), 256U);
        ASSERT_EQUALS(fullData.dimY(), 256U);
        ASSERT_EQUALS(fullData.dimZ(), 3U);
        ASSERT_EQUALS(fullData.dimB(), 1U);
    }
}

TEST(StimuliProvider, readRandomBatch)
{
    REQUIRED(UnitTest::DirExists(N2D2_DATA("mnist")));

    Random::mtSeed(0);

    MNIST_IDX_Database database;
    database.load(N2D2_DATA("mnist"));

    StimuliProvider sp(database, 28, 28, 1, 1, false);
    sp.setCachePath();

    sp.readRandomBatch(Database::Test);
}

TEST(StimuliProvider, readRandomBatch_bis)
{
    REQUIRED(UnitTest::DirExists(N2D2_DATA("mnist")));

    Random::mtSeed(0);

    MNIST_IDX_Database database;
    database.load(N2D2_DATA("mnist"));

    StimuliProvider sp(database, 32, 32, 1, 1, false);
    sp.addTransformation(RescaleTransformation(32, 32));
    sp.setCachePath();

    sp.readRandomBatch(Database::Test);
}

TEST(StimuliProvider, streamStimulus)
{
    StimuliProvider sp(EmptyDatabase, 28, 28, 1, 2, false);

    const cv::Mat img0(28, 28, CV_8UC1, cv::Scalar(1));
    const cv::Mat img1(28, 28, CV_8UC1, cv::Scalar(2));

    sp.streamStimulus(img0, Database::Learn, 0);
    sp.streamStimulus(img1, Database::Learn, 1);

    Tensor2d<Float_T> data0 = sp.getData(0, 0);
    Tensor2d<Float_T> data1 = sp.getData(0, 1);

    ASSERT_EQUALS(data0.dimX(), 28U);
    ASSERT_EQUALS(data0.dimY(), 28U);
    ASSERT_EQUALS(data1.dimX(), 28U);
    ASSERT_EQUALS(data1.dimY(), 28U);

    for (unsigned int index = 0; index < data0.size(); ++index) {
        ASSERT_EQUALS_DELTA(data0(index), 1.0 / 255.0, 1e-6);
        ASSERT_EQUALS_DELTA(data1(index), 2.0 / 255.0, 1e-6);
    }
}

RUN_TESTS()
