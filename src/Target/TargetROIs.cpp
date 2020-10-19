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

#include "Network.hpp"
#include "N2D2.hpp"
#include "StimuliProvider.hpp"
#include "Cell/Cell.hpp"
#include "ROI/RectangularROI.hpp"
#include "Target/TargetROIs.hpp"
#ifdef _OPENMP
#include <omp.h>
#endif

N2D2::Registrar<N2D2::Target>
N2D2::TargetROIs::mRegistrar("TargetROIs", N2D2::TargetROIs::create);

const char* N2D2::TargetROIs::Type = "TargetROIs";

N2D2::TargetROIs::TargetROIs(const std::string& name,
                             const std::shared_ptr<Cell>& cell,
                             const std::shared_ptr<StimuliProvider>& sp,
                             double targetValue,
                             double defaultValue,
                             unsigned int targetTopN,
                             const std::string& labelsMapping,
                             bool createMissingLabels)
    : Target(
          name, cell, sp, targetValue, defaultValue, targetTopN, labelsMapping, createMissingLabels),
      mMinSize(this, "MinSize", 0U),
      mMinOverlap(this, "MinOverlap", 0.5),
      mFilterMinHeight(this, "FilterMinHeight", 0U),
      mFilterMinWidth(this, "FilterMinWidth", 0U),
      mFilterMinAspectRatio(this, "FilterMinAspectRatio", 0.0),
      mFilterMaxAspectRatio(this, "FilterMaxAspectRatio", 0.0),
      mMergeMaxHDist(this, "MergeMaxHDist", 1U),
      mMergeMaxVDist(this, "MergeMaxVDist", 1U),
      mGenerateLabelsROIs(this, "GenerateLabelsROIs", true),
      mScoreTopN(this, "ScoreTopN", 1U)
{
    // ctor
}

unsigned int N2D2::TargetROIs::getNbTargets() const
{
    return (mROIsLabelTarget) ? mROIsLabelTarget->getNbTargets()
                              : Target::getNbTargets();
}

void N2D2::TargetROIs::setROIsLabelTarget(const std::shared_ptr<Target>& target)
{
    mROIsLabelTarget = target;

    // Create target_rois_label.dat for target_rois_viewer.py
    const std::string fileName = mName + "/target_rois_label.dat";
    std::ofstream roisLabelData(fileName);

    if (!roisLabelData.good())
        throw std::runtime_error("Could not save data file: " + fileName);

    roisLabelData << mROIsLabelTarget->getName();
    roisLabelData.close();
}

void N2D2::TargetROIs::logConfusionMatrix(const std::string& fileName,
                                          Database::StimuliSet set) const
{
    (*mScoreSet.find(set)).second.confusionMatrix.log(
        mName + "/ConfusionMatrix_" + fileName + ".dat", getTargetLabelsName());
}

void N2D2::TargetROIs::clearConfusionMatrix(Database::StimuliSet set)
{
    mScoreSet[set].confusionMatrix.clear();
}

void N2D2::TargetROIs::processEstimatedLabels(Database::StimuliSet set,
                                              Float_T* values)
{
    const bool validDatabase
        = (mStimuliProvider->getDatabase().getNbStimuli() > 0);

    const unsigned int nbTargets = getNbTargets();
    ConfusionMatrix<unsigned long long int>& confusionMatrix
        = mScoreSet[set].confusionMatrix;

    if (confusionMatrix.empty())
        confusionMatrix.resize(nbTargets, nbTargets, 0);

    const Tensor<int>& labels = mStimuliProvider->getLabelsData();
    const double xRatio = labels.dimX() / (double)mCell->getOutputsWidth();
    const double yRatio = labels.dimY() / (double)mCell->getOutputsHeight();

    int dev = 0;
#ifdef CUDA
    CHECK_CUDA_STATUS(cudaGetDevice(&dev));
#endif

    const Tensor<int>& targets = mTargetData[dev].targets;
    const TensorLabels_T& estimatedLabels = mTargetData[dev].estimatedLabels;

    estimatedLabels.synchronizeDBasedToH();

    mDetectedBB.assign(targets.dimB(), std::vector<DetectedBB>());

#pragma omp parallel for if (targets.dimB() > 4)
    for (int batchPos = 0; batchPos < (int)targets.dimB(); ++batchPos) {
#ifdef CUDA
        CHECK_CUDA_STATUS(cudaSetDevice(dev));
#endif

        const int id = mStimuliProvider->getBatch()[batchPos];

        if (id < 0) {
            // Invalid stimulus in batch (can occur for the last batch of the
            // set)
            continue;
        }

        std::vector<DetectedBB> detectedBB;

        // Extract estimated BB
        const Tensor<int> estLabels = estimatedLabels[batchPos][0];

        ComputerVision::LSL_Box lsl(mMinSize);
        lsl.process(Matrix<int>(estLabels.dimY(),
                                estLabels.dimX(),
                                estLabels.begin(),
                                estLabels.end()));

        std::vector<ComputerVision::ROI::Roi_T> estimatedROIs = lsl.getRoi();

        if (mFilterMinHeight > 0 || mFilterMinWidth > 0
            || mFilterMinAspectRatio > 0.0 || mFilterMaxAspectRatio > 0.0) {
            ComputerVision::ROI::filterSize(estimatedROIs,
                                            mFilterMinHeight,
                                            mFilterMinWidth,
                                            mFilterMinAspectRatio,
                                            mFilterMaxAspectRatio);
        }

        ComputerVision::ROI::filterSeparability(
            estimatedROIs, mMergeMaxHDist, mMergeMaxVDist);

        for (std::vector<ComputerVision::ROI::Roi_T>::const_iterator it
             = estimatedROIs.begin(),
             itEnd = estimatedROIs.end();
             it != itEnd;
             ++it)
        {
            const int bbLabel = (*it).cls;
            DetectedBB dbb(std::make_shared<RectangularROI<int> >(
                               bbLabel,
                               // RectangularROI<>() bottom right is exclusive,
                               // but LSL_Box b.r. is inclusive
                               cv::Point(Utils::round(xRatio * (*it).j0),
                                         Utils::round(yRatio * (*it).i0)),
                               cv::Point(Utils::round(xRatio * ((*it).j1 + 1)),
                                         Utils::round(yRatio * ((*it).i1 + 1)))),
                           0.0,
                           std::shared_ptr<ROI>(),
                           0.0,
                           false);

            if (mScoreTopN > 1) {
                TensorLabelsValue_T bbLabels = (mROIsLabelTarget)
                    ? mROIsLabelTarget->getEstimatedLabels(dbb.bb, batchPos, values)
                    : getEstimatedLabels(dbb.bb, batchPos, values);

                std::vector<int> labelIdx(bbLabels.size());
                std::iota(labelIdx.begin(), labelIdx.end(), 0);

                // sort indexes based on comparing values in
                // bbLabels. Sort in descending order.
                std::partial_sort(labelIdx.begin(),
                    labelIdx.begin() + mScoreTopN,
                    labelIdx.end(),
                    [&bbLabels](int i1, int i2)
                        {return bbLabels(i1) > bbLabels(i2);});

                dbb.bb->setLabel(labelIdx[0]);
                dbb.score = bbLabels(labelIdx[0]);

                for (unsigned int i = 1; i < mScoreTopN; ++i) {
                    dbb.scoreTopN.push_back(
                        std::make_pair(labelIdx[i], bbLabels(labelIdx[i])));
                }
            }
            else {
                int label;
                Float_T score;

                std::tie(label, score) = (mROIsLabelTarget)
                    ? mROIsLabelTarget->getEstimatedLabel(dbb.bb, batchPos, values)
                    : getEstimatedLabel(dbb.bb, batchPos, values);

                dbb.bb->setLabel(label);
                dbb.score = score;
            }

            detectedBB.push_back(dbb);
        }

        // Sort BB by highest score
        std::sort(detectedBB.begin(), detectedBB.end(), scoreCompare);

        if (validDatabase) {
            const Tensor<int> target = targets[batchPos];

            // Extract ground true ROIs
            std::vector<std::shared_ptr<ROI> > labelROIs
                = mStimuliProvider->getLabelsROIs(batchPos);

            if (labelROIs.empty() && target.size() == 1) {
                // The whole image has a single label
                if (target(0) >= 0) {
                    // Confusion computation
                    for (std::vector<DetectedBB>::const_iterator itBB
                        = detectedBB.begin(),
                        itBBEnd = detectedBB.end();
                        itBB != itBBEnd;
                        ++itBB) {
                        const int bbLabel = (*itBB).bb->getLabel();
    #pragma omp atomic
                        confusionMatrix(target(0), bbLabel) += 1ULL;
                    }
                }
            }
            else {
                if (mGenerateLabelsROIs
                    && labelROIs.empty() && target.size() > 1)
                {
                    labelROIs = generateLabelsROIs(labels[batchPos][0]);
                }

                // ROI and BB association
                for (std::vector<DetectedBB>::iterator
                    itBB = detectedBB.begin(), itBBEnd = detectedBB.end();
                    itBB != itBBEnd; ++itBB)
                {
                    for (std::vector<std::shared_ptr<ROI> >::const_iterator
                        itLabel = labelROIs.begin(),
                        itLabelEnd = labelROIs.end(); itLabel != itLabelEnd;
                        ++itLabel)
                    {
                        const cv::Rect bbRect = (*itBB).bb->getBoundingRect();
                        cv::Rect labelRect = (*itLabel)->getBoundingRect();

                        // Crop labelRect to the slice for correct overlap area
                        // calculation
                        if (labelRect.tl().x < 0) {
                            labelRect.width+= labelRect.tl().x;
                            labelRect.x = 0;
                        }
                        if (labelRect.tl().y < 0) {
                            labelRect.height+= labelRect.tl().y;
                            labelRect.y = 0;
                        }
                        if (labelRect.br().x > (int)labels.dimX())
                            labelRect.width = labels.dimX() - labelRect.x;
                        if (labelRect.br().y > (int)labels.dimY())
                            labelRect.height = labels.dimY() - labelRect.y;

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
                            const int unionArea = labelRect.area()
                                + bbRect.area() - interArea;
                            const double overlapFraction = interArea
                                / (double)unionArea;

                            if (overlapFraction > mMinOverlap) {
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
                    itBB = detectedBB.begin(), itBBEnd = detectedBB.end();
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

                            const int targetLabel = getLabelTarget((*itBB).roi
                                                            ->getLabel());

                            if (targetLabel >= 0) {
#pragma omp atomic
                                confusionMatrix(targetLabel, bbLabel) += 1ULL;
                            }
                        }
                    } else {
                        // False positive
#pragma omp atomic
                        confusionMatrix(0, bbLabel) += 1ULL;
                    }
                }

                // False negative (miss) for remaining unmatched label ROIs
                for (std::vector<std::shared_ptr<ROI> >::const_iterator itLabel
                    = labelROIs.begin(), itLabelEnd = labelROIs.end();
                    itLabel != itLabelEnd; ++itLabel)
                {
                    const int targetLabel
                        = getLabelTarget((*itLabel)->getLabel());

                    if (targetLabel >= 0) {
#pragma omp atomic
                        confusionMatrix(targetLabel, 0) += 1ULL;
                    }
                }
            }
        }

        mDetectedBB[batchPos].swap(detectedBB);
    }
}
void N2D2::TargetROIs::process(Database::StimuliSet set)
{
    Target::process(set);
    processEstimatedLabels(set);
}

cv::Mat N2D2::TargetROIs::drawEstimatedLabels(unsigned int batchPos) const
{
    const std::vector<DetectedBB>& detectedBB = mDetectedBB[batchPos];
    const std::vector<std::string>& labelsName = getTargetLabelsName();

    // Input image
    cv::Mat img = (cv::Mat)mStimuliProvider->getDataChannel(0, batchPos);
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
    std::vector<std::shared_ptr<ROI> > labelROIs
        = mStimuliProvider->getLabelsROIs(batchPos);

    int dev = 0;
#ifdef CUDA
    CHECK_CUDA_STATUS(cudaGetDevice(&dev));
#endif

    const Tensor<int>& targets = mTargetData[dev].targets;

    if (mGenerateLabelsROIs
        && labelROIs.empty() && targets[batchPos].size() > 1)
    {
        const Tensor<int>& labels = mStimuliProvider->getLabelsData();
        labelROIs = generateLabelsROIs(labels[batchPos][0]);
    }

    for (std::vector<std::shared_ptr<ROI> >::const_iterator itLabel
         = labelROIs.begin(),
         itLabelEnd = labelROIs.end();
         itLabel != itLabelEnd;
         ++itLabel) {
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
         it != itEnd;
         ++it)
    {
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

cv::Mat N2D2::TargetROIs::getBBData(const DetectedBB& bb,
                                    unsigned int batchPos) const
{
    // Input image
    cv::Mat img = (cv::Mat)mStimuliProvider->getDataChannel(0, batchPos);
    cv::Mat img8U;
    img.convertTo(img8U, CV_8U, 255.0);

    return bb.bb->extract(img8U);
}

void N2D2::TargetROIs::logDetectedBB(const std::string& fileName,
                                     unsigned int batchPos) const
{
    const bool validDatabase
        = (mStimuliProvider->getDatabase().getNbStimuli() > 0);
    const std::vector<DetectedBB>& detectedBB = mDetectedBB[batchPos];

    if (!detectedBB.empty()) {
        const std::vector<std::string>& labelsName = getTargetLabelsName();
        const int id = mStimuliProvider->getBatch()[batchPos];

        const std::string logFileName = Utils::fileBaseName(fileName)
            + ".log";
        std::ofstream roisData(logFileName.c_str());

        if (!roisData.good())
            throw std::runtime_error("Could not save data file: "
                                        + logFileName);

        std::ostringstream imgFile;

        if (validDatabase)
            imgFile << mStimuliProvider->getDatabase().getStimulusName(id);
        else
            imgFile << std::setw(10) << std::setfill('0') << id;

        for (std::vector<DetectedBB>::const_iterator it
                = detectedBB.begin(),
                itEnd = detectedBB.end();
                it != itEnd;
                ++it) {
            const cv::Rect rect = (*it).bb->getBoundingRect();

            roisData << Utils::quoted(imgFile.str())
                << " " << rect.x << " " << rect.y
                << " " << rect.width << " " << rect.height
                << " " << labelsName[(*it).bb->getLabel()]
                << " " << (*it).score;

            if ((*it).roi)
                roisData << " 1";
            else
                roisData << " 0";

            roisData << "\n";
        }
    }
}

void N2D2::TargetROIs::logEstimatedLabels(const std::string& dirName) const
{
    const bool validDatabase
        = (mStimuliProvider->getDatabase().getNbStimuli() > 0);

    Target::logEstimatedLabels(dirName);

    const std::string dirPath = mName + "/" + dirName;
    Utils::createDirectories(dirPath);

    // Remove symlink created in Target::logEstimatedLabels()
    int ret = remove((dirPath + ".py").c_str());
    if (ret < 0) {
    } // avoid ignoring return value warning

#if !defined(WIN32) && !defined(__CYGWIN__) && !defined(_WIN32)
    ret = symlink(N2D2_PATH("tools/target_rois_viewer.py"),
                  (dirPath + ".py").c_str());
    if (ret < 0) {
    } // avoid ignoring return value warning
#endif

    int dev = 0;
#ifdef CUDA
    CHECK_CUDA_STATUS(cudaGetDevice(&dev));
#endif

    const Tensor<int>& targets = mTargetData[dev].targets;

#pragma omp parallel for if (targets.dimB() > 4)
    for (int batchPos = 0; batchPos < (int)targets.dimB(); ++batchPos) {
#ifdef CUDA
        CHECK_CUDA_STATUS(cudaSetDevice(dev));
#endif

        const int id = mStimuliProvider->getBatch()[batchPos];

        if (id < 0) {
            // Invalid stimulus in batch (can occur for the last batch of the
            // set)
            continue;
        }

        std::ostringstream imgFile;
        std::string fileName;

        if (validDatabase) {
            imgFile << mStimuliProvider->getDatabase().getStimulusName(id);

            const std::string baseName = Utils::baseName(imgFile.str());
            const std::string fileBaseName = Utils::fileBaseName(baseName);
            std::string fileExtension = Utils::fileExtension(baseName);

            if (!((std::string)mImageLogFormat).empty()) {
                // Keep "[x,y]" after file extension, appended by
                // getStimulusName() in case of slicing
                fileExtension.replace(0, fileExtension.find_first_of('['),
                                    mImageLogFormat);
            }

            fileName = dirPath + "/" + fileBaseName + "." + fileExtension;
        }
        else {
            imgFile << std::setw(10) << std::setfill('0') << id;

            const std::string fileExtension
                = (!((std::string)mImageLogFormat).empty())
                    ? (std::string)mImageLogFormat
                    : std::string("jpg");

            fileName = dirPath + "/" + imgFile.str() + "." + fileExtension;
        }

        // Draw image
        if (!cv::imwrite(fileName, drawEstimatedLabels(batchPos))) {
#pragma omp critical(TargetROIs__logEstimatedLabels)
            throw std::runtime_error("Unable to write image: " + fileName);
        }

        // Log ROIs
        logDetectedBB(fileName, batchPos);
    }

    // Merge all ROIs logs
#ifdef WIN32
    const std::string cmd = "type " + dirPath + "/*.log > " + dirPath + ".log";
#else
    const std::string cmd = "cat " + dirPath + "/*.log > " + dirPath + ".log";
#endif
    Utils::exec(cmd);
}

void N2D2::TargetROIs::logEstimatedLabelsJSON(const std::string& dirName,
                                              const std::string& fileName,
                                              unsigned int xOffset,
                                              unsigned int yOffset,
                                              bool append) const
{
    const std::string dirPath = mName + "/" + dirName;
    Utils::createDirectories(dirPath);

    const bool validDatabase
        = (mStimuliProvider->getDatabase().getNbStimuli() > 0);

    const time_t now = std::time(0);
    tm* localNow = std::localtime(&now);
    std::string time = std::asctime(localNow);
    time.pop_back(); // remove \n introduced by std::asctime()

    const std::vector<std::string>& labelsName = getTargetLabelsName();

    int dev = 0;
#ifdef CUDA
    CHECK_CUDA_STATUS(cudaGetDevice(&dev));
#endif

    const Tensor<int>& targets = mTargetData[dev].targets;

#ifdef _OPENMP
    omp_lock_t appendLock;
    omp_init_lock(&appendLock);
#endif

#pragma omp parallel for if (targets.dimB() > 4) schedule(dynamic)
    for (int batchPos = 0; batchPos < (int)targets.dimB(); ++batchPos) {
#ifdef CUDA
        CHECK_CUDA_STATUS(cudaSetDevice(dev));
#endif

        const int id = mStimuliProvider->getBatch()[batchPos];

        if (id < 0) {
            // Invalid stimulus in batch (can occur for the last batch of the
            // set)
            continue;
        }

        std::string jsonName(fileName);

        if (jsonName.empty()) {
            std::ostringstream imgFile;

            if (validDatabase) {
                imgFile << mStimuliProvider->getDatabase()
                                        .getStimulusName(id, false);

                const std::string baseName = Utils::baseName(imgFile.str());
                const std::string fileBaseName = Utils::fileBaseName(baseName);
                std::string fileExtension = Utils::fileExtension(baseName);
/*
                if (!((std::string)mImageLogFormat).empty()) {
                    // Keep "[x,y]" after file extension, appended by
                    // getStimulusName() in case of slicing
                    fileExtension.replace(0, fileExtension.find_first_of('['),
                                        mImageLogFormat);
                }
*/
                jsonName = dirPath + "/" + fileBaseName + "." + fileExtension;
            }
            else {
                imgFile << std::setw(10) << std::setfill('0') << id;

                const std::string fileExtension
                    = (!((std::string)mImageLogFormat).empty())
                        ? (std::string)mImageLogFormat
                        : std::string("jpg");

                jsonName = dirPath + "/" + imgFile.str() + "." + fileExtension;
            }
        }

        jsonName += ".box.json";

        std::ostringstream jsonDataBuffer;

        const std::vector<DetectedBB>& detectedBB = mDetectedBB[batchPos];

        if (detectedBB.empty())
            continue;

        for (std::vector<DetectedBB>::const_iterator it
                = detectedBB.begin(),
                itEnd = detectedBB.end();
                it != itEnd;
                ++it)
        {
            const cv::Rect rect = (*it).bb->getBoundingRect();

            if (it != detectedBB.begin())
                jsonDataBuffer << ",";

            unsigned int xSliceOffset = 0;
            unsigned int ySliceOffset = 0;

            if (validDatabase) {
                const ROI* slice
                    = mStimuliProvider->getDatabase().getStimulusSlice(id);

                if (slice != NULL) {
                    const cv::Rect bbRect = slice->getBoundingRect();
                    xSliceOffset = bbRect.x;
                    ySliceOffset = bbRect.y;

                    // If there is multiple slices, append MUST be true
                    // because 2nd argument of getStimulusName() is true
                    append = true;
                }
            }

            jsonDataBuffer << "{\"class_id\": " << (*it).bb->getLabel() << ","
                "\"class_name\": \"" << labelsName[(*it).bb->getLabel()] << "\","
                "\"info\": [\"BOX_" << (*it).score << "\","
                    "false,"
                    "{\"CreationDate\": \"" << time << "\","
                        "\"Source\": \"N2D2\","
                        "\"ClassName\": \"" << labelsName[(*it).bb->getLabel()]
                            << "\","
                        "\"Confidence\": \"" << (*it).score << "\"";

            for (std::vector<std::pair<int, double> >::const_iterator itScore
                    = (*it).scoreTopN.begin(),
                    itScoreBegin = (*it).scoreTopN.begin(),
                    itScoreEnd = (*it).scoreTopN.end();
                    itScore != itScoreEnd;
                    ++itScore)
            {
                jsonDataBuffer << ",\"ClassName-" << (itScore - itScoreBegin + 1)
                    << "\": \"" << labelsName[(*itScore).first] << "\""
                    << ",\"Confidence-" << (itScore - itScoreBegin + 1)
                    << "\": \"" << (*itScore).second << "\"";
            }

            jsonDataBuffer << "}"
                "],"
                "\"type\": \"polygon\","
                "\"points\": ["
                    "[" << xSliceOffset + xOffset + rect.x << ", "
                        << ySliceOffset + yOffset + rect.y << "],"
                    "[" << xSliceOffset + xOffset + rect.x + rect.width << ", "
                        << ySliceOffset + yOffset + rect.y << "],"
                    "[" << xSliceOffset + xOffset + rect.x + rect.width << ", "
                        << ySliceOffset + yOffset + rect.y + rect.height << "],"
                    "[" << xSliceOffset + xOffset + rect.x << ", "
                        << ySliceOffset + yOffset + rect.y + rect.height << "]]}";
        }

        jsonDataBuffer << "]}";

#ifdef _OPENMP
        if (append && omp_in_parallel())
            omp_set_lock(&appendLock);
#endif
        std::fstream jsonData;
        bool newFile = true;

        if (append) {
            newFile = false;
            jsonData.open(jsonName.c_str(),
                          std::ofstream::in | std::ofstream::out);
        }
        else
            jsonData.open(jsonName.c_str(), std::ofstream::out);

        if (append && !jsonData.good()) {
            newFile = true;
            jsonData.open(jsonName.c_str(),
                          std::ofstream::in | std::ofstream::out | std::ofstream::app);
        }

        //std::ofstream jsonData(jsonName.c_str(),
        //    (append) ? std::fstream::app
        //             : std::fstream::out);

        if (!jsonData.good()) {
#pragma omp critical(TargetROIs__logEstimatedLabelsJSON)
            throw std::runtime_error("Could not create JSON file: " + jsonName);
        }

        if (newFile)
            jsonData << "{\"annotations\": [" << jsonDataBuffer.str();
        else {
            jsonData.seekp(-2, jsonData.end); // Go before "]}"
            jsonData.write(",", sizeof(char));
            jsonData.write(jsonDataBuffer.str().c_str(),
                           sizeof(char) * jsonDataBuffer.str().size());
        }

#ifdef _OPENMP
        if (append && omp_in_parallel())
            omp_unset_lock(&appendLock);
#endif
    }

#ifdef _OPENMP
    omp_destroy_lock(&appendLock);
#endif
}

void N2D2::TargetROIs::log(const std::string& fileName,
                           Database::StimuliSet set)
{
    logConfusionMatrix(fileName, set);
}

void N2D2::TargetROIs::clear(Database::StimuliSet set)
{
    Target::clear(set);
    clearConfusionMatrix(set);
}

std::vector<std::shared_ptr<N2D2::ROI> > N2D2::TargetROIs::generateLabelsROIs(
    const Tensor<int>& labels) const
{
    // Compute label ROIs from pixel-wise annotations.
    // In this case, it makes sense to apply the same filtering
    // to the obtained label ROIs, than the estimated ROIs.
    std::vector<std::shared_ptr<ROI> > labelROIs;

    ComputerVision::LSL_Box lsl(mMinSize);
    lsl.process(Matrix<int>(labels.dimY(),
                            labels.dimX(),
                            labels.begin(),
                            labels.end()));

    std::vector<ComputerVision::ROI::Roi_T> estimatedROIs
        = lsl.getRoi();

    if (mFilterMinHeight > 0 || mFilterMinWidth > 0
        || mFilterMinAspectRatio > 0.0 || mFilterMaxAspectRatio > 0.0) {
        ComputerVision::ROI::filterSize(estimatedROIs,
                                        mFilterMinHeight,
                                        mFilterMinWidth,
                                        mFilterMinAspectRatio,
                                        mFilterMaxAspectRatio);
    }

    ComputerVision::ROI::filterSeparability(
        estimatedROIs, mMergeMaxHDist, mMergeMaxVDist);

    for (std::vector<ComputerVision::ROI::Roi_T>::iterator
        it = estimatedROIs.begin(), itEnd = estimatedROIs.end();
        it != itEnd; ++it)
    {
        std::map<int, unsigned int> labelsFreq;

        for (size_t y = (*it).i0; y <= (*it).i1; ++y) {
            for (size_t x = (*it).j0; x <= (*it).j1; ++x)
                ++labelsFreq[labels(x, y)];
        }
    
        const std::map<int, unsigned int>::const_iterator itFreq
            = std::max_element(labelsFreq.begin(), labelsFreq.end(),
                            Utils::PairSecondPred<int, unsigned int>());
        
        int label = (*itFreq).first;
    
        if (mWeakTarget >= -1 && getLabelTarget(label) == mWeakTarget) {
            labelsFreq.erase(label);

            const std::map<int, unsigned int>::const_iterator itFreq
                = std::max_element(labelsFreq.begin(), labelsFreq.end(),
                                Utils::PairSecondPred<int, unsigned int>());
            
            if (itFreq != labelsFreq.end())
                label = (*itFreq).first;
        }

        if (label >= 0) {
            labelROIs.push_back(std::make_shared<RectangularROI<int> >(
                label,
                // RectangularROI<>() bottom right is exclusive,
                // but LSL_Box b.r. is inclusive
                cv::Point((*it).j0, (*it).i0),
                cv::Point((*it).j1 + 1, (*it).i1 + 1)));
        }
    }

    return labelROIs;
}