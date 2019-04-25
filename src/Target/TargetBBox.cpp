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

#include "StimuliProvider.hpp"
#include "Cell/Cell.hpp"
#include "Cell/Cell_Frame_Top.hpp"
#include "Cell/AnchorCell.hpp"
#include "Target/TargetBBox.hpp"
#include "ROI/RectangularROI.hpp"

N2D2::Registrar<N2D2::Target>
N2D2::TargetBBox::mRegistrar("TargetBBox", N2D2::TargetBBox::create);

const char* N2D2::TargetBBox::Type = "TargetBBox";

N2D2::TargetBBox::TargetBBox(const std::string& name,
                            const std::shared_ptr<Cell>& cell,
                            const std::shared_ptr<StimuliProvider>& sp,
                            double targetValue,
                            double defaultValue,
                            unsigned int targetTopN,
                            const std::string& labelsMapping)
    : Target(
          name, cell, sp, targetValue, defaultValue, targetTopN, labelsMapping)
{
    // ctor
}

void N2D2::TargetBBox::logConfusionMatrix(const std::string& fileName,
                                          Database::StimuliSet set) const
{
    if(set == Database::Learn || set == Database::Validation || set == Database::Test)
    {
        (*mScoreSet.find(set)).second.confusionMatrix.log(
            mName + "/ConfusionMatrix_" + fileName + ".dat", mLabelsBBoxName);
    }
}

void N2D2::TargetBBox::clearConfusionMatrix(Database::StimuliSet set)
{
    if(set == Database::Learn || set == Database::Validation || set == Database::Test)
    {

        mScoreSet[set].confusionMatrix.clear();
    }
}

void N2D2::TargetBBox::clearSuccess(Database::StimuliSet set)
{
    if(set == Database::Learn || set == Database::Validation || set == Database::Test)
    {
        mScoreSet[set].success.clear();
        //mScoreTopNSet[set].success.clear();
    }
}

double N2D2::TargetBBox::getAverageSuccess(Database::StimuliSet set,
                                            unsigned int /*avgWindow*/) const
{
    assert(mScoreSet.find(set) != mScoreSet.end());
    const std::vector<std::deque<double>>& success = (*mScoreSet.find(set)).second.success;

    if (success[0].empty())
        return 0.0;

    //double success = (*mScoreSet.find(set)).second.recall[0].back();

    return success[0].back();
}

double N2D2::TargetBBox::getBBoxSucess(Database::StimuliSet set)
{
    assert(mScoreSet.find(set) != mScoreSet.end());
    double avgScore = 0.0;

    if(set == Database::Learn || set == Database::Validation || set == Database::Test)
    {
        const ConfusionMatrix<unsigned long long int>& matrix = (*mScoreSet.find(set)).second.confusionMatrix;
        std::vector<std::deque<double>>& success = (*mScoreSet.find(set)).second.success;

        if (success.empty())
            success.resize(mLabelsBBoxName.size());

        for(unsigned int target = 1;  target < mLabelsBBoxName.size(); ++target)
        {
            avgScore += matrix.getConfusionTable(target).fScore();
        }
        avgScore /= mLabelsBBoxName.size() > 1 ? (mLabelsBBoxName.size() - 1.0) : 1;
        success[0].push_back(avgScore);
    }

    return avgScore;
}

bool N2D2::TargetBBox::newValidationScore(double validationScore)
{
    //mValidationScore.push_back(validationScore);
    mValidationScore.push_back(std::make_pair((*mScoreSet.find(Database::Learn)).second.success[0].size(), validationScore));

    if (validationScore > mMaxValidationScore) {
        mMaxValidationScore = validationScore;
        return true;
    } else
        return false;
}

void N2D2::TargetBBox::logSuccess(const std::string& fileName,
                                   Database::StimuliSet set,
                                   unsigned int avgWindow) const
{
    assert(mScoreSet.find(set) != mScoreSet.end());


    if (set == Database::Validation) {
        const std::string dataFileName = mName + "/AverageSuccess_" + fileName + ".dat";
        // Save validation scores file
        std::ofstream dataFile(dataFileName);

        if (!dataFile.good())
            throw std::runtime_error("Could not create data rate log file: "
                                     + dataFileName);

        for (std::vector<std::pair<unsigned int, double> >::const_iterator it
             = mValidationScore.begin(),
             itEnd = mValidationScore.end();
             it != itEnd;
             ++it) {
            dataFile << (*it).first << " " << (*it).second << "\n";
        }


        dataFile.close();

        const std::vector<std::deque<double>>& successLearn = (*mScoreSet.find(Database::Learn)).second.success;

        // Plot validation
        const double lastValidation = (!mValidationScore.empty())
            ? mValidationScore.back().second : 0.0;
        const double lastLearn = successLearn[0].empty()
            ? successLearn[0].back() : 0.0;

        const double minFinalRate = std::min(lastValidation, lastLearn);
        const double maxFinalRate = std::max(lastValidation, lastLearn);

        std::ostringstream label;
        label << "\"Best validation: " << 100.0 * mMaxValidationScore
              << "%\" at graph 0.5, graph 0.15 front";

        Gnuplot multiplot;
        multiplot.saveToFile(dataFileName);
        multiplot.setMultiplot();


        Monitor::logDataRate(successLearn[0],
                             mName + "/" + fileName + "_LearningSuccess.dat",
                             avgWindow,
                             true);

        multiplot << "clear";
        multiplot.setYrange(
            Utils::clamp(minFinalRate - (1.0 - minFinalRate), 0.0, 0.99),
            Utils::clamp(2.0 * maxFinalRate, 0.01, 1.0));
        multiplot.set("label", label.str());
        multiplot
            << ("replot \"" + dataFileName
                + "\" using 1:2 with linespoints lt 7 title \"Validation\"");


/*
        Monitor::logDataRate(mValidationScore,
                             dataFile,
                             avgWindow,
                             true);*/
    }

}

void N2D2::TargetBBox::initialize(bool genAnchors, unsigned int nbAnchors, long unsigned int nbIterMax)
{
    mGenerateAnchors = genAnchors;
    mNbAnchors = nbAnchors;
    mIterMax = nbIterMax;

    std::shared_ptr<Cell_Frame_Top> targetCell = std::dynamic_pointer_cast
        <Cell_Frame_Top>(mCell);
    const BaseTensor& outputsShape = targetCell->getOutputs();
    
    if (outputsShape.dimZ() != 4 && outputsShape.dimZ() != 5) {
        //throw std::runtime_error("TargetBBox::initialize(): cell must have 4 or 5"
        //                        " output channels for BBox TargetBBox " + mName);

        std::cout << "Target BBOX" << std::endl;
    }

    mTargets.resize({mCell->getOutputsWidth(),
                    mCell->getOutputsHeight(),
                    outputsShape.dimZ(),
                    outputsShape.dimB()});
    
    //if(mGenerateAnchors)
    //    computeKmeansClustering(Database::Learn);

    const std::vector<std::string> labelsName = getTargetLabelsName();
    mLabelsBBoxName.push_back("Background");
    for(unsigned int i = 0; i < labelsName.size(); ++i)
    {
        if(!labelsName[i].empty())
        {
            mLabelsBBoxName.push_back(labelsName[i]);
        }
    }
}

void N2D2::TargetBBox::process(Database::StimuliSet set)
{

    std::shared_ptr<Cell_Frame_Top> targetCell = std::dynamic_pointer_cast
        <Cell_Frame_Top>(mCell);

    const std::vector<int>& batch = mStimuliProvider->getBatch();
    const int nbBBoxMax = (int)mTargets.dimB()/batch.size();

    targetCell->getOutputs().synchronizeDToH();
    const Tensor<Float_T>& values
        = tensor_cast<Float_T>(targetCell->getOutputs());
    const std::vector<int> labelsCls = getTargetLabels(0);
    const unsigned int nbTargets = mLabelsBBoxName.size() ; //labelsCls.size();

    ConfusionMatrix<unsigned long long int>& confusionMatrix = mScoreSet[set].confusionMatrix;
    
    if (confusionMatrix.empty())
        confusionMatrix.resize(nbTargets, nbTargets, 0);

    mBatchDetectedBBox.assign(mTargets.dimB(), std::vector<DetectedBB>());

    //#pragma omp parallel for if (mTargets.dimB() > 4)
    for (int batchPos = 0; batchPos < (int)batch.size(); ++batchPos) {

        // Extract ground true ROIs
        std::vector<std::shared_ptr<ROI> > labelROIs = mStimuliProvider->getLabelsROIs(batchPos);

        std::vector<DetectedBB> bbox;

        for(int bboxIdx = 0; bboxIdx < nbBBoxMax; ++ bboxIdx)
        {
            const Tensor<Float_T>& value = values[bboxIdx + batchPos*nbBBoxMax];
            Float_T x = value(0);
            Float_T y = value(1);
            Float_T w = value(2);
            Float_T h = value(3);
            Float_T conf = 0.0;
            int cls = 0;
            if(values.dimZ() > 4)
            {
                conf = value(4);
                cls = (int) value(5);
            }
            else
                cls = (int) value(4);

            if(w > 0.0 && h > 0.0 && x >= 0.0 
                && x < mStimuliProvider->getSizeX() 
                && y >= 0.0 && y < mStimuliProvider->getSizeY())
            {
                DetectedBB dbb (std::make_shared<RectangularROI<int> >(
                                cls,
                                cv::Point(int (x), int(y)),
                                cv::Point(int(w + x), int(h + y))),
                                conf,
                                std::shared_ptr<ROI>(),
                                0.0,
                                false);  

                bbox.push_back(dbb);        

            }
        }

        if(set == Database::Learn || set == Database::Validation || set == Database::Test)
        {

            std::sort(bbox.begin(), bbox.end(), scoreCompare);

            // ROI and BB association
            for (std::vector<DetectedBB>::iterator itBB = bbox.begin(),
                                                    itBBEnd = bbox.end();
                    itBB != itBBEnd;
                    ++itBB) {
                for (std::vector<std::shared_ptr<ROI> >::const_iterator itLabel
                        = labelROIs.begin(),
                        itLabelEnd = labelROIs.end();
                        itLabel != itLabelEnd;
                        ++itLabel) {
                    const cv::Rect bbRect = (*itBB).bb->getBoundingRect();
                    cv::Rect labelRect = (*itLabel)->getBoundingRect();

                    const int interLeft = std::max(labelRect.tl().x,
                                                    bbRect.tl().x);
                    const int interRight
                        = std::min(labelRect.br().x, bbRect.br().x);
                    const int interTop = std::max(labelRect.tl().y,
                                                    bbRect.tl().y);
                    const int interBottom
                        = std::min(labelRect.br().y, bbRect.br().y);
                    const cv::Rect interRect
                        = cv::Rect(cv::Point(interLeft, interTop),
                                    cv::Point(interRight, interBottom));

                    if (interLeft < interRight && interTop < interBottom) {
                        const int interArea = interRect.area();
                        const int unionArea = labelRect.area() + bbRect.area()
                                                - interArea;
                        const double overlapFraction = interArea
                            / (double)unionArea;

                        if (overlapFraction > 0.5) {
                            if (!(*itBB).roi
                                || overlapFraction > (*itBB).matching)
                            {
                                (*itBB).roi = (*itLabel);
                                (*itBB).matching = overlapFraction;
                            }
                        }
                    }
                }
            }

            // Confusion computation
            for (std::vector<DetectedBB>::iterator
                 itBB = bbox.begin(), itBBEnd = bbox.end();
                 itBB != itBBEnd;
                 ++itBB) {
                const int bbLabel = (*itBB).bb->getLabel();

                if ((*itBB).roi) {
                    // Found a matching ROI
                    const std::vector<std::shared_ptr<ROI> >::iterator
                        itLabel = std::find(labelROIs.begin(),
                                            labelROIs.end(),
                                            (*itBB).roi);

                    (*itBB).duplicate = (itLabel == labelROIs.end());

                    if (!(*itBB).duplicate) {
                        // If this is the first match, remove this label
                        // from the list and count it for the confusion
                        // matrix
                        labelROIs.erase(itLabel);

                        const int targetLabel = getLabelTarget((*itBB).roi ->getLabel());

                        if (targetLabel >= 0) {
//#pragma omp atomic
                            confusionMatrix(targetLabel + 1, bbLabel + 1) += 1ULL;
                        }
                    }
                } else {
                    // False positive
//#pragma omp atomic
                    confusionMatrix(0, bbLabel + 1) += 1ULL;
                }
            }

            // False negative (miss) for remaining unmatched label ROIs
            for (std::vector<std::shared_ptr<ROI> >::const_iterator itLabel
                 = labelROIs.begin(),
                 itLabelEnd = labelROIs.end();
                 itLabel != itLabelEnd;
                 ++itLabel) {
                const int targetLabel = getLabelTarget((*itLabel)->getLabel());

                if (targetLabel >= 0) {
//#pragma omp atomic
                    confusionMatrix(targetLabel + 1, 0) += 1ULL;
                }
            }
            double success = getBBoxSucess(set);
            if(success < 0.0)
                std::cout << "Warning, cannot compute success rate: " << success << std::endl;
        }

        //std::cout << "TargetBBox: Number of BBox detected on batch " << batchPos << ": " << bbox.size() << std::endl;

        mBatchDetectedBBox[batchPos].swap(bbox);

        if(set == Database::Validation )
            logEstimatedLabels("Validation/");

    }

}




cv::Mat N2D2::TargetBBox::drawEstimatedBBox(unsigned int batchPos) const
{
    const std::vector<DetectedBB>& detectedBB = mBatchDetectedBBox[batchPos];
    const std::vector<std::string> labelsName = getTargetLabelsName();

    //const int defaultLabel = getLabelTarget(mStimuliProvider->getDatabase()
    //                                            .getDefaultLabelID());

    // Input image
    cv::Mat img = (cv::Mat)mStimuliProvider->getData(0, batchPos);
    cv::Mat img8U;
    // img.convertTo(img8U, CV_8U, 255.0);

    // Normalize image
    cv::Mat imgNorm;
    cv::normalize(img.reshape(1), imgNorm, 0, 255, cv::NORM_MINMAX);
    img = imgNorm.reshape(img.channels());
    img.convertTo(img8U, CV_8U);

    cv::Mat imgBB;
#if CV_MAJOR_VERSION >= 3
    cv::cvtColor(img8U, imgBB, cv::COLOR_GRAY2BGR);
#else
    cv::cvtColor(img8U, imgBB, CV_GRAY2BGR);
#endif

    // Draw targets ROIs
    const std::vector<std::shared_ptr<ROI> >& labelROIs
        = mStimuliProvider->getLabelsROIs(batchPos);

    for (std::vector<std::shared_ptr<ROI> >::const_iterator itLabel
         = labelROIs.begin(),
         itLabelEnd = labelROIs.end();
         itLabel != itLabelEnd;
         ++itLabel)
    {
        const int target = getLabelTarget((*itLabel)->getLabel());

        if (target >= 0) {
            (*itLabel)->draw(imgBB, cv::Scalar(255, 0, 0));

            // Draw legend
            const cv::Rect rect = (*itLabel)->getBoundingRect();
            std::stringstream legend;
            legend << target << " " << labelsName[target];

            int baseline = 0;
            cv::Size textSize = cv::getTextSize(
                legend.str(), cv::FONT_HERSHEY_SIMPLEX, 0.25, 1, &baseline);
            cv::putText(imgBB,
                        legend.str(),
                        cv::Point(std::max(0, rect.x) + 2,
                                  std::max(0, rect.y) + textSize.height + 2),
                        cv::FONT_HERSHEY_SIMPLEX,
                        0.25,
                        cv::Scalar(255, 0, 0),
                        1,
#if CV_MAJOR_VERSION >= 3
                        cv::LINE_AA);
#else
                        CV_AA);
#endif
        }
    }

    // Draw detected BB


    for (std::vector<DetectedBB>::const_iterator it = detectedBB.begin(),
                                                 itEnd = detectedBB.end();
                                                 it != itEnd; ++it)
    {

        //cv::Scalar color = cv::Scalar(0, 255 , 0);

        //RectangularROI<int> bbox( (*it).cls, cv::Point((*it).x, (*it).y), (*it).w, (*it).h);
        //bbox.draw(imgBB, color);


        cv::Scalar color = cv::Scalar(0, 0, 255);   // red = miss

        if ((*it).roi) {
            // Found a matching ROI
            const bool match = (getLabelTarget((*it).roi->getLabel())
                                    == (*it).bb->getLabel());

            if (match) {
                // True hit
                color = (!(*it).duplicate)
                           ? cv::Scalar(0, 255, 0)  // green = true hit
                           : cv::Scalar(0, 127, 0); // dark green = duplicate
            }
            else {
                color = (!(*it).duplicate)
                           ? cv::Scalar(0, 255, 255)  // yellow = wrong label
                           : cv::Scalar(0, 127, 127); // dark yellow = duplicate
            }
        }

        (*it).bb->draw(imgBB, color);

        // Draw legend
        const cv::Rect rect = (*it).bb->getBoundingRect();
        std::stringstream legend;
        legend << (*it).bb->getLabel() << " "
               << labelsName[(*it).bb->getLabel()];

        // Draw legend

        //const cv::Rect rect = bbox.getBoundingRect();
        //std::stringstream legend;
        ////legend << bbox.getLabel();
        //legend << std::setprecision(2) << (*it).cls << "(" << (*it).conf << ")";


        int baseline = 0;
        cv::Size textSize = cv::getTextSize(
            legend.str(), cv::FONT_HERSHEY_SIMPLEX, 0.25, 1, &baseline);
        cv::putText(imgBB,
                    legend.str(),
                    cv::Point(rect.x + 2, rect.y + textSize.height + 2),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.25,
                    color,
                    1,
#if CV_MAJOR_VERSION >= 3
                    cv::LINE_AA);
#else
                    CV_AA);
#endif



    }

    return imgBB;
}

void N2D2::TargetBBox::logEstimatedLabels(const std::string& dirName) const
{
    const std::string dirPath = mName + "/" + dirName;
    Utils::createDirectories(dirPath);

    const std::vector<int>& batch = mStimuliProvider->getBatch();

#pragma omp parallel for if (batch.size() > 4)
    for (int batchPos = 0; batchPos < (int)batch.size(); ++batchPos) {
        const int id = batch[batchPos];

        if (id < 0) {
            // Invalid stimulus in batch (can occur for the last batch of the
            // set)
            continue;
        }

        const std::string imgFile
            = mStimuliProvider->getDatabase().getStimulusName(id);
        const std::string baseName = Utils::baseName(imgFile);
        const std::string fileBaseName = Utils::fileBaseName(baseName);
        std::string fileExtension = Utils::fileExtension(baseName);

        if (!((std::string)mImageLogFormat).empty()) {
            // Keep "[x,y]" after file extension, appended by
            // getStimulusName() in case of slicing
            fileExtension.replace(0, fileExtension.find_first_of('['),
                                  mImageLogFormat);
        }

        const std::string fileName = dirPath + "/" + fileBaseName
                                        + "." + fileExtension;

        // Draw image

        if (!cv::imwrite(fileName, drawEstimatedBBox(batchPos)))
            throw std::runtime_error("Unable to write image: " + fileName);

    }
}

void N2D2::TargetBBox::log(const std::string& fileName,
                           Database::StimuliSet set)
{
    //if(set == Database::Validation || set == Database::Test)
    logConfusionMatrix(fileName, set);
}

void N2D2::TargetBBox::clear(Database::StimuliSet set)
{
    Target::clear(set);
    clearConfusionMatrix(set);
}

N2D2::TargetBBox::~TargetBBox()
{

}

