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

#include "Target/TargetRP.hpp"

N2D2::Registrar<N2D2::Target>
N2D2::TargetRP::mRegistrar("TargetRP", N2D2::TargetRP::create);

const char* N2D2::TargetRP::Type = "TargetRP";
std::map<std::string, std::map<N2D2::TargetRP::TargetType, N2D2::TargetRP*> >
N2D2::TargetRP::mTargetRP;

N2D2::TargetRP::TargetRP(const std::string& name,
                         const std::shared_ptr<Cell>& cell,
                         const std::shared_ptr<StimuliProvider>& sp,
                         double targetValue,
                         double defaultValue,
                         unsigned int targetTopN,
                         const std::string& labelsMapping)
    : Target(
          name, cell, sp, targetValue, defaultValue, targetTopN, labelsMapping),
      mMinOverlap(this, "MinOverlap", 0.5),
      mLossLambda(this, "LossLambda", 1.0)
{
    // ctor
}

void N2D2::TargetRP::logConfusionMatrix(const std::string& fileName,
                                          Database::StimuliSet set) const
{
    if (mTargetType == Cls) {
        (*mScoreSet.find(set)).second.confusionMatrix.log(
            mName + "/ConfusionMatrix_" + fileName + ".dat",
            getTargetLabelsName());
    }
}

void N2D2::TargetRP::clearConfusionMatrix(Database::StimuliSet set)
{
    if (mTargetType == Cls)
        mScoreSet[set].confusionMatrix.clear();
}

void N2D2::TargetRP::initialize(TargetType targetType,
                                const std::shared_ptr<RPCell>& RPCell,
                                const std::shared_ptr<AnchorCell>& anchorCell)
{
    std::shared_ptr<Cell_Frame_Top> targetCell = std::dynamic_pointer_cast
        <Cell_Frame_Top>(mCell);
    const Tensor4d<Float_T>& values = targetCell->getOutputs();

    if (targetType == BBox && values.dimZ() != 4) {
        throw std::runtime_error("TargetRP::initialize(): cell must have 4"
                                 " output channels for BBox TargetRP " + mName);
    }

    mTargetType = targetType;
    mRPCell = RPCell;
    mAnchorCell = anchorCell;
    mTargets.resize(mCell->getOutputsWidth(),
                    mCell->getOutputsHeight(),
                    (mTargetType == BBox) ? 4 : 1,
                    values.dimB());

    if (mTargetType == Cls) {
        mEstimatedLabels.resize(mCell->getOutputsWidth(),
                                mCell->getOutputsHeight(),
                                mTargetTopN,
                                values.dimB());
        mEstimatedLabelsValue.resize(mCell->getOutputsWidth(),
                                     mCell->getOutputsHeight(),
                                     mTargetTopN,
                                     values.dimB());
    }

    const std::string targetRP = anchorCell->getName()
        + "+" + RPCell->getName();
    mTargetRP[targetRP][targetType] = this;
}

void N2D2::TargetRP::process(Database::StimuliSet set)
{
    if (!mRPCell) {
        throw std::runtime_error("TargetRP::process(): uninitialized TargetRP "
                                 + mName);
    }

    if (mTargetType == Cls)
        processCls(set);
    else if (mTargetType == BBox)
        processBBox(set);
}

void N2D2::TargetRP::processCls(Database::StimuliSet set)
{
    std::shared_ptr<Cell_Frame_Top> targetCell = std::dynamic_pointer_cast
        <Cell_Frame_Top>(mCell);
    const Tensor4d<Float_T>& values = targetCell->getOutputs();
    const std::vector<Tensor4d<int>::Index> anchors = mRPCell->getAnchors();

    ConfusionMatrix<unsigned long long int>& confusionMatrix
        = mScoreSet[set].confusionMatrix;

    if (confusionMatrix.empty())
        confusionMatrix.resize(values.size() / values.dimB(),
                               values.size() / values.dimB(),
                               0);

    assert(anchors.size() == mTargets.dimB());

    const int defaultLabel = getLabelTarget(mStimuliProvider->getDatabase()
                                                .getDefaultLabelID());

#pragma omp parallel for if (mTargets.dimB() > 16)
    for (int batchPos = 0; batchPos < (int)mTargets.dimB(); ++batchPos) {
        const Tensor3d<Float_T> value = values[batchPos];
        Tensor3d<int> estimatedLabels = mEstimatedLabels[batchPos];
        Tensor3d<Float_T> estimatedLabelsValue
            = mEstimatedLabelsValue[batchPos];

        if (value.size() > 1) {
            std::vector
                <std::pair<Float_T, size_t> > sortedLabelsValues;
            sortedLabelsValues.reserve(value.size());

            for (unsigned int index = 0; index < value.size();
                 ++index)
                sortedLabelsValues.push_back(
                    std::make_pair(value(index), index));

            std::partial_sort(
                sortedLabelsValues.begin(),
                sortedLabelsValues.begin() + mTargetTopN,
                sortedLabelsValues.end(),
                std::greater<std::pair<Float_T, size_t> >());

            for (unsigned int i = 0; i < mTargetTopN; ++i) {
                estimatedLabels(i) = sortedLabelsValues[i].second;
                estimatedLabelsValue(i)
                    = sortedLabelsValues[i].first;
            }
        }
        else {
            estimatedLabels(0) = (value(0) > mBinaryThreshold);
            estimatedLabelsValue(0) = value(0);
        }

        const double IoU = mAnchorCell->getAnchorIoU(anchors[batchPos]);

        if (IoU >= mMinOverlap) {
            const std::shared_ptr<ROI> ROI
                = mAnchorCell->getAnchorROI(anchors[batchPos]);
            const int targetLabel = getLabelTarget(ROI->getLabel());

            mTargets(0, batchPos) = targetLabel;
            confusionMatrix(targetLabel, estimatedLabels(0)) += 1ULL;
        }
        else {
            mTargets(0, batchPos) = defaultLabel;
            confusionMatrix(defaultLabel, estimatedLabels(0)) += 1ULL;
        }
    }

    if (set == Database::Learn) {
        targetCell->setOutputTarget(
            mTargets, mTargetValue, mDefaultValue);
    }
}

void N2D2::TargetRP::processBBox(Database::StimuliSet set)
{
    std::shared_ptr<Cell_Frame_Top> targetCell = std::dynamic_pointer_cast
        <Cell_Frame_Top>(mCell);
    const std::vector<Tensor4d<int>::Index> anchors = mRPCell->getAnchors();
    const Tensor4d<Float_T>& values = targetCell->getOutputs();

    Tensor4d<Float_T> smoothTargets(mCell->getOutputsWidth(),
                                    mCell->getOutputsHeight(),
                                    mTargets.dimZ(),
                                    mTargets.dimB(),
                                    0.0);

#pragma omp parallel for if (mTargets.dimB() > 16)
    for (int batchPos = 0; batchPos < (int)mTargets.dimB(); ++batchPos) {
        const AnchorCell_Frame_Kernels::BBox_T& gt
            = mAnchorCell->getAnchorGT(anchors[batchPos]);
/*
        mTargets(0, batchPos) = gt.x;
        mTargets(1, batchPos) = gt.y;
        mTargets(2, batchPos) = gt.w;
        mTargets(3, batchPos) = gt.h;
*/
        const double IoU = mAnchorCell->getAnchorIoU(anchors[batchPos]);

        if (IoU >= mMinOverlap && set == Database::Learn) {
            const AnchorCell_Frame_Kernels::BBox_T& bb
                = mAnchorCell->getAnchorBBox(anchors[batchPos]);

            // Parameterized Ground Truth coordinates
            const Float_T txgt = (gt.x - bb.x) / bb.w;
            const Float_T tygt = (gt.y - bb.y) / bb.h;
            const Float_T twgt = std::log(gt.w / bb.w);
            const Float_T thgt = std::log(gt.h / bb.h);

            smoothTargets(0, batchPos) = mLossLambda
                * smoothL1(txgt, values(0, batchPos));
            smoothTargets(1, batchPos) = mLossLambda
                * smoothL1(tygt, values(1, batchPos));
            smoothTargets(2, batchPos) = mLossLambda
                * smoothL1(twgt, values(2, batchPos));
            smoothTargets(3, batchPos) = mLossLambda
                * smoothL1(thgt, values(3, batchPos));
        }
    }

    if (set == Database::Learn)
        targetCell->setOutputErrors(smoothTargets);
}

cv::Mat N2D2::TargetRP::drawEstimatedLabels(unsigned int batchPos) const
{
    if (mTargetType != Cls)
        return cv::Mat();

    const std::vector<std::string> labelsName = getTargetLabelsName();
    const int defaultLabel = getLabelTarget(mStimuliProvider->getDatabase()
                                                .getDefaultLabelID());

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
                        CV_AA);
        }
    }

    const std::string targetRP = mAnchorCell->getName()
        + "+" + mRPCell->getName();

    if (mTargetRP[targetRP].find(BBox) == mTargetRP[targetRP].end())
        throw std::runtime_error("Missing TargetRP of type: " + BBox);

    // Draw detected BB
    std::shared_ptr<Cell_Frame_Top> targetCell = std::dynamic_pointer_cast
        <Cell_Frame_Top>(mCell);
    std::shared_ptr<Cell_Frame_Top> targetBBoxCell = std::dynamic_pointer_cast
        <Cell_Frame_Top>(mTargetRP[targetRP][BBox]->getCell());

    const Tensor4d<Float_T>& valuesBBox = targetBBoxCell->getOutputs();

    const std::vector<Tensor4d<int>::Index> anchors = mRPCell->getAnchors();

    for (unsigned int n = 0, nbProposals = mRPCell->getNbProposals();
        n < nbProposals; ++n)
    {
        const unsigned int outputBatchPos = n + batchPos * nbProposals;
        const int cls = mEstimatedLabels(0, outputBatchPos);

        const AnchorCell_Frame_Kernels::BBox_T& bb
            = mAnchorCell->getAnchorBBox(anchors[outputBatchPos]);

        // Parameterized coordinates
        const Float_T tx = valuesBBox(0, outputBatchPos);
        const Float_T ty = valuesBBox(1, outputBatchPos);
        const Float_T tw = valuesBBox(2, outputBatchPos);
        const Float_T th = valuesBBox(3, outputBatchPos);

        // Predicted box coordinates
        const Float_T x = tx * bb.w + bb.x;
        const Float_T y = ty * bb.h + bb.y;
        const Float_T w = bb.w * std::exp(tw);
        const Float_T h = bb.h * std::exp(th);

        const double IoU = mAnchorCell->getAnchorIoU(anchors[outputBatchPos]);

        cv::Scalar color = cv::Scalar(0, 0, 255);   // red = miss

        if (IoU >= mMinOverlap) {
            // Found a matching ROI
            const std::shared_ptr<ROI> ROI
                = mAnchorCell->getAnchorROI(anchors[outputBatchPos]);

            const bool match = (getLabelTarget(ROI->getLabel()) == cls);

            if (match)
                color = cv::Scalar(0, 255, 0);  // green
            else if (cls != defaultLabel)
                color = cv::Scalar(0, 255, 255);  // yellow = wrong label
        }
        else if (cls == defaultLabel)
            color = cv::Scalar(0, 0, 0);  // black = background

        RectangularROI<int> bbox(cls, cv::Point(x, y), w, h);
        bbox.draw(imgBB, color);

        // Draw legend
        if (cls != defaultLabel) {
            const cv::Rect rect = bbox.getBoundingRect();
            std::stringstream legend;
            legend << bbox.getLabel() << " "
                   << labelsName[bbox.getLabel()];

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
    }

    return imgBB;
}

void N2D2::TargetRP::logEstimatedLabels(const std::string& dirName) const
{
    if (mTargetType != Cls)
        return;

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
        const std::string fileName = dirPath + "/" + baseName;

        // Draw image
        if (!cv::imwrite(fileName, drawEstimatedLabels(batchPos)))
            throw std::runtime_error("Unable to write image: " + fileName);
    }
}

void N2D2::TargetRP::log(const std::string& fileName,
                           Database::StimuliSet set)
{
    logConfusionMatrix(fileName, set);
}

void N2D2::TargetRP::clear(Database::StimuliSet set)
{
    clearConfusionMatrix(set);
}

N2D2::TargetRP::~TargetRP()
{
    // dtor
    if (mRPCell) {
        const std::string targetRP = mAnchorCell->getName()
            + "+" + mRPCell->getName();
        mTargetRP[targetRP].erase(mTargetType);
    }
}
