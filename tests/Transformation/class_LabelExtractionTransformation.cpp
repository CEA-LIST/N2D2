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
#include "ROI/RectangularROI.hpp"
#include "Transformation/LabelExtractionTransformation.hpp"
#include "utils/UnitTest.hpp"
#include "utils/Utils.hpp"

using namespace N2D2;

class Database_Test : public Database {
public:
    Database_Test()
    {
    }

    void load(const std::string& /*dataPath*/,
              const std::string& /*labelPath*/ = "",
              bool /*extractROIs*/ = false)
    {
        std::vector<ROI*> rois1;

        rois1.push_back(new RectangularROI
                        <int>(255,
                              cv::Point(54, 92),
                              157,
                              311)); // person 54   92 157 311 0 0 0 0 0 0 0
        rois1.push_back(new RectangularROI
                        <int>(255,
                              cv::Point(370, 166),
                              30,
                              62)); // person 370 166  30  62 0 0 0 0 0 0 0
        rois1.push_back(new RectangularROI
                        <int>(255,
                              cv::Point(549, 160),
                              25,
                              49)); // person 554 161  20  44 0 0 0 0 0 0 0
        rois1.push_back(new RectangularROI
                        <int>(255,
                              cv::Point(294, 147),
                              49,
                              97)); // person 300 158  36  74 0 0 0 0 0 0 0
        rois1.push_back(new RectangularROI
                        <int>(255,
                              cv::Point(235, 168),
                              33,
                              59)); // person 235 168  33  59 0 0 0 0 0 0 0

        mStimuli.push_back(Stimulus("I01444.jpg", -1, rois1));

        std::vector<ROI*> rois2;

        rois2.push_back(new RectangularROI
                        <int>(255,
                              cv::Point(561, 128),
                              48,
                              157)); // person 561 128 42 157 0 0 0 0 0 0 0
        rois2.push_back(new RectangularROI
                        <int>(255,
                              cv::Point(413, 177),
                              25,
                              46)); // person 413 177 25 46  0 0 0 0 0 0 0
        rois2.push_back(new RectangularROI
                        <int>(255,
                              cv::Point(35, 190),
                              1,
                              21)); // person 24  190 17 42  1 35  190 1 21 0 0
        rois2.push_back(new RectangularROI
                        <int>(255,
                              cv::Point(133, 188),
                              8,
                              28)); // person 128 188 17 32  1 133 190 8 28 0 0

        mStimuli.push_back(Stimulus("I00015.jpg", -1, rois2));
    }
};

TEST_DATASET(LabelExtractionTransformation,
             applyToFrame,
             (int seed,
              int id,
              int stimuliId,
              int iteration,
              std::string widths,
              std::string heights),
             std::make_tuple(0, -1, 0, 5, "24", "48"),
             std::make_tuple(1, 0, 0, 5, "24", "48"))
{
    REQUIRED(UnitTest::FileExists("I01444.jpg"));
    REQUIRED(UnitTest::FileExists("I00015.jpg"));

    Random::mtSeed(seed);

    Database_Test database;
    database.load("");

    cv::Mat labels(480, 640, CV_32SC1, cv::Scalar(0));
    cv::Mat img
        = cv::imread(database.getStimulusName(stimuliId),
#if CV_MAJOR_VERSION >= 3
            cv::IMREAD_COLOR);
#else
            CV_LOAD_IMAGE_COLOR);
#endif

    if (!img.data)
        throw std::runtime_error("Could not open or find image: "
                                 + database.getStimulusName(stimuliId));

    std::vector<std::shared_ptr<ROI> > stimulusROIs
        = database.getStimulusROIs(stimuliId);

    for (std::vector<std::shared_ptr<ROI> >::const_iterator it
        = stimulusROIs.begin(); it != stimulusROIs.end(); it++)
    {
        (*it)->append(labels);
    }

    LabelExtractionTransformation trans(widths, heights);

    Utils::createDirectories("Transformation/LabelExtraction/");

    for (int i = 0; i < iteration; ++i) {
        cv::Mat imgCopy = img.clone();
        cv::Mat labelsCopy = labels.clone();

        trans.apply(imgCopy, labelsCopy, stimulusROIs, id);

        std::ostringstream filename;
        filename << "LabelExtractionTransformation_applyToFrame(W"
                 << trans.getLastSlice().width << "_H"
                 << trans.getLastSlice().height << "_id" << id << "_i" << i
                 << ")[frame].png";
        cv::imwrite("Transformation/LabelExtraction/" + filename.str(),
                    imgCopy);

        filename.str(std::string());
        filename << "LabelExtractionTransformation_applyToFrame(W"
                 << trans.getLastSlice().width << "_H"
                 << trans.getLastSlice().height << "_id" << id << "_i" << i
                 << ")[labels].png";
        cv::imwrite("Transformation/LabelExtraction/" + filename.str(),
                    labelsCopy);

        ASSERT_EQUALS(labelsCopy.rows, 1);
        ASSERT_EQUALS(labelsCopy.cols, 1);
        ASSERT_EQUALS(labelsCopy.at<int>(0, 0), trans.getLastLabel());
    }
}

TEST_DATASET(LabelExtractionTransformation,
             applyToFrame_bis,
             (int seed,
              int id,
              int stimuliId,
              int iteration,
              std::string widths,
              std::string heights),
             // std::make_tuple(0, -1, 0, 2, "18,24,48,96", "36,48,96,192"),
             std::make_tuple(1, 0, 0, 40, "18,24,48,96", "36,48,96,192"),
             std::make_tuple(1, 1, 1, 20, "18,24,48,96", "36,48,96,192"))
{
    REQUIRED(UnitTest::FileExists("I01444.jpg"));
    REQUIRED(UnitTest::FileExists("I00015.jpg"));

    Random::mtSeed(seed);

    Database_Test database;
    database.load("");

    cv::Mat img
        = cv::imread(database.getStimulusName(stimuliId),
#if CV_MAJOR_VERSION >= 3
            cv::IMREAD_COLOR);
#else
            CV_LOAD_IMAGE_COLOR);
#endif

    if (!img.data)
        throw std::runtime_error("Could not open or find image: "
                                 + database.getStimulusName(stimuliId));

    cv::Mat labels(480, 640, CV_32SC1, cv::Scalar(0));

    std::vector<std::shared_ptr<ROI> > stimulusROIs
        = database.getStimulusROIs(stimuliId);
    std::vector<std::shared_ptr<ROI> >::const_iterator it
        = stimulusROIs.begin();

    for (; it != stimulusROIs.end(); it++)
        (*it)->append(labels);

    LabelExtractionTransformation trans(widths, heights);

    cv::Mat slicesLabels = labels.clone();
    cv::Mat slicesImg = img.clone();

    std::ostringstream filename;

    for (int i = 0; i < iteration; ++i) {
        cv::Mat imgCopy = img.clone();
        cv::Mat labelsCopy = labels.clone();

        trans.apply(imgCopy, labelsCopy, stimulusROIs, id);

        cv::rectangle(slicesLabels,
                      trans.getLastSlice().tl(),
                      trans.getLastSlice().br(),
                      cv::Scalar(128));

        cv::rectangle(slicesImg,
                      trans.getLastSlice().tl(),
                      trans.getLastSlice().br(),
                      cv::Scalar(0, 0, trans.getLastLabel()));

        /*filename.str(std::string());
        filename << "LabelExtractionTransformation_applyToFrame_bis(file_" <<
        database.getStimulusName(stimuliId)
            << "W_" << trans.getLastSlice().width << "_H" <<
        trans.getLastSlice().height << " " << i << ")[patchs].png";
        cv::imwrite("Transformation/LabelExtraction/" + filename.str(),
        labelsCopy);*/

        ASSERT_EQUALS(labelsCopy.rows, 1);
        ASSERT_EQUALS(labelsCopy.cols, 1);
        ASSERT_EQUALS(labelsCopy.at<int>(0, 0), trans.getLastLabel());
    }

    for (it = stimulusROIs.begin(); it != stimulusROIs.end(); it++) {
        const cv::Rect rect = (*it)->getBoundingRect();
        cv::rectangle(slicesImg, rect.tl(), rect.br(), cv::Scalar(0, 255, 0));
    }

    Utils::createDirectories("Transformation/LabelExtraction/");
    
    filename.str(std::string());
    filename << "LabelExtractionTransformation_applyToFrame_bis(W" << widths
             << "_H" << heights << "_id" << id << "_file_"
             << database.getStimulusName(stimuliId) << ")[frame].png";
    cv::imwrite("Transformation/LabelExtraction/" + filename.str(), slicesImg);

    filename.str(std::string());
    filename << "LabelExtractionTransformation_applyToFrame_bis(W" << widths
             << "_H" << heights << "_id" << id << "_file_"
             << database.getStimulusName(stimuliId) << ")[labels].png";
    cv::imwrite("Transformation/LabelExtraction/" + filename.str(),
                slicesLabels);
}

RUN_TESTS()
