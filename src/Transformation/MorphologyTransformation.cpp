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

#include "Transformation/MorphologyTransformation.hpp"

const char* N2D2::MorphologyTransformation::Type = "Morphology";

N2D2::MorphologyTransformation::MorphologyTransformation(Operation operation,
                                                         unsigned int size,
                                                         bool applyToLabels)
    : mOperation(operation),
      mSize(size),
      mApplyToLabels(applyToLabels),
      mLabelsIgnoreDiff(this, "LabelsIgnoreDiff", false),
      mShape(this, "Shape", Rectangular),
      mNbIterations(this, "NbIterations", 1U),
      mLabel(this, "Label", std::vector<int>())
{
    // ctor
}

N2D2::MorphologyTransformation::MorphologyTransformation(
    const MorphologyTransformation& trans)
    : mOperation(trans.mOperation),
      mSize(trans.mSize),
      mApplyToLabels(trans.mApplyToLabels),
      mLabelsIgnoreDiff(this, "LabelsIgnoreDiff", trans.mLabelsIgnoreDiff),
      mShape(this, "Shape", trans.mShape),
      mNbIterations(this, "NbIterations", trans.mNbIterations),
      mLabel(this, "Label", trans.mLabel)
{
    // copy-ctor
}

void
N2D2::MorphologyTransformation::apply(cv::Mat& frame,
                                      cv::Mat& labels,
                                      std::vector
                                      <std::shared_ptr<ROI> >& /*labelsROI*/,
                                      int /*id*/)
{
    if (!mKernel.data) {
        const int shape = (mShape == Rectangular)
                              ? cv::MORPH_RECT
                              : (mShape == Elliptic) ? cv::MORPH_ELLIPSE
                                                     : cv::MORPH_CROSS;

        mKernel = cv::getStructuringElement(shape, cv::Size(mSize, mSize));
    }

    if (mApplyToLabels) {
        cv::Mat newLabels;

        if (mLabel->empty()) {
            // CV_32S not supported by cv::morphologyEx()
            // CV_64F not supported by OpenCV 2.0.0
            cv::Mat mat;
            labels.convertTo(mat, CV_32F);
            applyMorphology(mat);
            mat.convertTo(newLabels, CV_32S);
        }
        else if (mLabel->size() == 1) {
            // Simple case with only one label
            cv::Mat mask = (labels == (*mLabel->begin()));
            applyMorphology(mask);

            cv::Mat newLabels = cv::Mat(cv::Size(labels.cols, labels.rows),
                                        CV_32S, cv::Scalar((*mLabel->begin())));
            labels.copyTo(newLabels, (mask == 0));
        }
        else {
            // General case with multiple labels (slower)
            // Create a mask with all the labels we want to apply a morpho on
            cv::Mat mask = (labels == (*mLabel->begin()));

            for (std::vector<int>::const_iterator it = mLabel->begin() + 1,
                 itEnd = mLabel->end(); it != itEnd; ++it)
            {
                mask = mask | (labels == (*it));
            }

            // morphoLabels contains only the masked labels and the background
            // is -2
            cv::Mat morphoLabels = cv::Mat(cv::Size(labels.cols, labels.rows),
                                        CV_32S, cv::Scalar(-2));
            labels.copyTo(morphoLabels, mask);

            // Apply the morpho on morphoLabels
            // CV_32S not supported by cv::morphologyEx()
            // CV_64F not supported by OpenCV 2.0.0
            cv::Mat morphoMat;
            morphoLabels.convertTo(morphoMat, CV_32F);
            applyMorphology(morphoMat);
            morphoMat.convertTo(morphoLabels, CV_32S);

            // Retrieves only the morphed labels into labels and keep the
            newLabels = labels.clone();
            morphoLabels.copyTo(newLabels, (morphoLabels > -2));
        }

        if (mLabelsIgnoreDiff)
            labels.setTo(-1, newLabels != labels);
        else
            labels = newLabels;
    } else
        applyMorphology(frame);
}

void N2D2::MorphologyTransformation::applyMorphology(cv::Mat& mat) const
{
    if (mOperation == Erode)
        cv::erode(mat, mat, mKernel, cv::Point(-1, -1), mNbIterations);
    else if (mOperation == Dilate)
        cv::dilate(mat, mat, mKernel, cv::Point(-1, -1), mNbIterations);
    else {
        const int operation = (mOperation == Opening)
                                  ? cv::MORPH_OPEN
                                  : (mOperation == Closing)
                                        ? cv::MORPH_CLOSE
                                        : (mOperation == Gradient)
                                              ? cv::MORPH_GRADIENT
                                              : (mOperation == TopHat)
                                                    ? cv::MORPH_TOPHAT
                                                    : cv::MORPH_BLACKHAT;

        cv::morphologyEx(
            mat, mat, operation, mKernel, cv::Point(-1, -1), mNbIterations);
    }
}

N2D2::MorphologyTransformation::~MorphologyTransformation() {
    
}
