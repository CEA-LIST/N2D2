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

#include "Target/TargetROIs.hpp"

N2D2::Registrar<N2D2::Target>
N2D2::TargetROIs::mRegistrar("TargetROIs", N2D2::TargetROIs::create);

const char* N2D2::TargetROIs::Type = "TargetROIs";

N2D2::TargetROIs::TargetROIs(const std::string& name,
                             const std::shared_ptr<Cell>& cell,
                             const std::shared_ptr<StimuliProvider>& sp,
                             double targetValue,
                             double defaultValue,
                             unsigned int targetTopN,
                             const std::string& labelsMapping)
    : Target(
          name, cell, sp, targetValue, defaultValue, targetTopN, labelsMapping),
      mMinSize(this, "MinSize", 0U),
      mMinOverlap(this, "MinOverlap", 0.5),
      mFilterMinHeight(this, "FilterMinHeight", 0U),
      mFilterMinWidth(this, "FilterMinWidth", 0U),
      mFilterMinAspectRatio(this, "FilterMinAspectRatio", 0.0),
      mFilterMaxAspectRatio(this, "FilterMaxAspectRatio", 0.0),
      mMergeMaxHDist(this, "MergeMaxHDist", 1U),
      mMergeMaxVDist(this, "MergeMaxVDist", 1U)
{
    // ctor
}

unsigned int N2D2::TargetROIs::getNbTargets() const
{
    return (mROIsLabelTarget) ? mROIsLabelTarget->getNbTargets()
                              : Target::getNbTargets();
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

void N2D2::TargetROIs::process(Database::StimuliSet set)
{
    Target::process(set);

    mDetectedBB.assign(mTargets.dimB(), std::vector<DetectedBB>());

    const unsigned int nbTargets = getNbTargets();
    ConfusionMatrix<unsigned long long int>& confusionMatrix
        = mScoreSet[set].confusionMatrix;

    if (confusionMatrix.empty())
        confusionMatrix.resize(nbTargets, nbTargets, 0);

    const Tensor4d<int>& labels = mStimuliProvider->getLabelsData();
    const double xRatio = (labels.dimX() - 1)
                          / (double)(mCell->getOutputsWidth() - 1);
    const double yRatio = (labels.dimY() - 1)
                          / (double)(mCell->getOutputsHeight() - 1);

#pragma omp parallel for if (mTargets.dimB() > 4)
    for (int batchPos = 0; batchPos < (int)mTargets.dimB(); ++batchPos) {
        const int id = mStimuliProvider->getBatch()[batchPos];

        if (id < 0) {
            // Invalid stimulus in batch (can occur for the last batch of the
            // set)
            continue;
        }

        std::vector<DetectedBB> detectedBB;

        // Extract estimated BB
        const Tensor3d<int> target = mTargets[batchPos];
        const Tensor2d<int> estimatedLabels = mEstimatedLabels[batchPos][0];
        const Tensor2d<Float_T> estimatedLabelsValue
            = mEstimatedLabelsValue[batchPos][0];

        ComputerVision::LSL_Box lsl(mMinSize);
        lsl.process(Matrix<int>(estimatedLabels.dimY(),
                                estimatedLabels.dimX(),
                                estimatedLabels.begin(),
                                estimatedLabels.end()));

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
             ++it) {
            const int bbLabel = (*it).cls;

            DetectedBB dbb(std::make_shared<RectangularROI<int> >(
                               bbLabel,
                               cv::Point(Utils::round(xRatio * (*it).j0),
                                         Utils::round(yRatio * (*it).i0)),
                               cv::Point(Utils::round(xRatio * (*it).j1),
                                         Utils::round(yRatio * (*it).i1))),
                           0.0,
                           std::shared_ptr<ROI>(),
                           0.0,
                           false);

            if (mROIsLabelTarget) {
                int label;
                Float_T score;
                std::tie(label, score)
                    = mROIsLabelTarget->getEstimatedLabel(dbb.bb, batchPos);

                dbb.bb->setLabel(label);
                dbb.score = score;
            } else {
                // Compute BB score
                int nbPixels = 0;

                for (unsigned int x = (*it).j0; x < (*it).j1; ++x) {
                    for (unsigned int y = (*it).i0; y < (*it).i1; ++y) {
                        if (estimatedLabels(x, y) == bbLabel) {
                            dbb.score += estimatedLabelsValue(x, y);
                            ++nbPixels;
                        }
                    }
                }

                if (nbPixels > 0)
                    dbb.score /= nbPixels;
            }

            detectedBB.push_back(dbb);
        }

        // Sort BB by highest score
        std::sort(detectedBB.begin(), detectedBB.end(), scoreCompare);

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
            // ROI and BB association
            for (std::vector<DetectedBB>::iterator itBB = detectedBB.begin(),
                                                   itBBEnd = detectedBB.end();
                 itBB != itBBEnd;
                 ++itBB) {
                for (std::vector<std::shared_ptr<ROI> >::const_iterator itLabel
                     = labelROIs.begin(),
                     itLabelEnd = labelROIs.end();
                     itLabel != itLabelEnd;
                     ++itLabel) {
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
                        const int unionArea = labelRect.area() + bbRect.area()
                                              - interArea;
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
                 = labelROIs.begin(),
                 itLabelEnd = labelROIs.end();
                 itLabel != itLabelEnd;
                 ++itLabel) {
                const int targetLabel = getLabelTarget((*itLabel)->getLabel());

                if (targetLabel >= 0) {
#pragma omp atomic
                    confusionMatrix(targetLabel, 0) += 1ULL;
                }
            }
        }

        mDetectedBB[batchPos].swap(detectedBB);
    }
}

cv::Mat N2D2::TargetROIs::drawEstimatedLabels(unsigned int batchPos) const
{
    const std::vector<DetectedBB>& detectedBB = mDetectedBB[batchPos];
    const std::vector<std::string> labelsName = getTargetLabelsName();

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
    cv::cvtColor(img8U, imgBB, CV_GRAY2BGR);

    // Draw targets ROIs
    const std::vector<std::shared_ptr<ROI> >& labelROIs
        = mStimuliProvider->getLabelsROIs(batchPos);

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
                        CV_AA);
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
                    CV_AA);
    }

    return imgBB;
}

cv::Mat N2D2::TargetROIs::getBBData(const DetectedBB& bb,
                                    unsigned int batchPos) const
{
    // Input image
    cv::Mat img = (cv::Mat)mStimuliProvider->getData(0, batchPos);
    cv::Mat img8U;
    img.convertTo(img8U, CV_8U, 255.0);

    return bb.bb->extract(img8U);
}

void N2D2::TargetROIs::logEstimatedLabels(const std::string& dirName) const
{
    Target::logEstimatedLabels(dirName);

    const std::string dirPath = mName + "/" + dirName;
    Utils::createDirectories(dirPath);

    if (mROIsLabelTarget) {
        const std::string fileName = mName + "/target_rois_label.dat";
        std::ofstream roisLabelData(fileName);

        if (!roisLabelData.good())
            throw std::runtime_error("Could not save data file: " + fileName);

        roisLabelData << mROIsLabelTarget->getName();
        roisLabelData.close();
    }

    // Remove symlink created in Target::logEstimatedLabels()
    int ret = remove((dirPath + ".py").c_str());
    if (ret < 0) {
    } // avoid ignoring return value warning

#ifndef WIN32
    ret = symlink(N2D2_PATH("tools/target_rois_viewer.py"),
                  (dirPath + ".py").c_str());
    if (ret < 0) {
    } // avoid ignoring return value warning
#endif

    const std::vector<std::string> labelsName = getTargetLabelsName();

#pragma omp parallel for if (mTargets.dimB() > 4)
    for (int batchPos = 0; batchPos < (int)mTargets.dimB(); ++batchPos) {
        const int id = mStimuliProvider->getBatch()[batchPos];

        if (id < 0) {
            // Invalid stimulus in batch (can occur for the last batch of the
            // set)
            continue;
        }

        const std::string imgFile
            = mStimuliProvider->getDatabase().getStimulusName(id);
        const std::string baseName = Utils::baseName(imgFile);
        const std::string fileName = dirPath + "/" + baseName;

        // Draw image
        if (!cv::imwrite(fileName, drawEstimatedLabels(batchPos)))
            throw std::runtime_error("Unable to write image: " + fileName);

        // Log ROIs
        const std::vector<DetectedBB>& detectedBB = mDetectedBB[batchPos];

        if (!detectedBB.empty()) {
            const std::string logFileName = Utils::fileBaseName(fileName)
                                            + ".log";
            std::ofstream roisData(logFileName.c_str());

            if (!roisData.good())
                throw std::runtime_error("Could not save data file: "
                                         + logFileName);

            for (std::vector<DetectedBB>::const_iterator it
                 = detectedBB.begin(),
                 itEnd = detectedBB.end();
                 it != itEnd;
                 ++it) {
                const cv::Rect rect = (*it).bb->getBoundingRect();

                roisData << baseName << " " << rect.x << " " << rect.y << " "
                         << rect.width << " " << rect.height << " "
                         << labelsName[(*it).bb->getLabel()] << " "
                         << (*it).score;

                if ((*it).roi)
                    roisData << " 1";
                else
                    roisData << " 0";

                roisData << "\n";
            }
        }
    }

    // Merge all ROIs logs
#ifdef WIN32
    const std::string cmd = "type " + dirPath + "/*.log > " + dirPath + ".log";
#else
    const std::string cmd = "cat " + dirPath + "/*.log > " + dirPath + ".log";
#endif
    ret = system(cmd.c_str());
    if (ret < 0) {
    } // avoid ignoring return value warning
}

void N2D2::TargetROIs::log(const std::string& fileName,
                           Database::StimuliSet set)
{
    logConfusionMatrix(fileName, set);
}

void N2D2::TargetROIs::clear(Database::StimuliSet set)
{
    clearConfusionMatrix(set);
}
