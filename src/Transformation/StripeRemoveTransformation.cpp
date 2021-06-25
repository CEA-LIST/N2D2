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

#include "Transformation/StripeRemoveTransformation.hpp"

const char* N2D2::StripeRemoveTransformation::Type = "StripeRemove";

N2D2::StripeRemoveTransformation::StripeRemoveTransformation(int axis,
                                                   unsigned int offset,
                                                   unsigned int length)
    : mAxis(axis),
      mOffset(offset),
      mLength(length),
      mRandomOffset(this, "RandomOffset", false),
      mNbIterations(this, "NbIterations", 1U),
      mStepOffset(this, "StepOffset", offset)
{
    // ctor
}

N2D2::StripeRemoveTransformation::StripeRemoveTransformation(
    const StripeRemoveTransformation& trans)
    : mAxis(trans.mAxis),
      mOffset(trans.mOffset),
      mLength(trans.mLength),
      mRandomOffset(this, "RandomOffset", trans.mRandomOffset),
      mNbIterations(this, "NbIterations", trans.mNbIterations),
      mStepOffset(this, "StepOffset", trans.mStepOffset)
{
    // copy-ctor
}

void N2D2::StripeRemoveTransformation::apply(cv::Mat& frame,
                                        cv::Mat& labels,
                                        std::vector
                                        <std::shared_ptr<ROI> >& /*labelsROI*/,
                                        int /*id*/)
{
    unsigned int offset = mOffset;

    for (unsigned int i = 0; i < mNbIterations; ++i) {
        const unsigned int iterOffset
            = (mRandomOffset) ? ((mAxis == 0)
                        ? Random::randUniform(0, frame.cols - mLength)
                        : Random::randUniform(0, frame.rows - mLength))
                    : offset;

        stripeRemove(frame, iterOffset);

        if (labels.rows > 1 || labels.cols > 1) {
            stripeRemove(labels, iterOffset);
        }

        offset += mStepOffset;
    }
}

std::pair<unsigned int, unsigned int> 
N2D2::StripeRemoveTransformation::getOutputsSize(unsigned int width,
                                                 unsigned int height) const
{
    unsigned int newWidth = width;
    unsigned int newHeight = height;

    if (mAxis == 0) {
        // Col
        newWidth -= mLength * mNbIterations;
    }
    else if (mAxis == 1) {
        // Row
        newHeight -= mLength * mNbIterations;
    }

    return std::make_pair(newWidth, newHeight);
}

void
N2D2::StripeRemoveTransformation::stripeRemove(cv::Mat& mat,
                                               unsigned int offset) const
{
    cv::Mat before, after;

    if (mAxis == 0) {
        // Col
        if (offset > 0) {
            mat(cv::Range(0, mat.rows),
                cv::Range(0, offset)).copyTo(before);
        }

        if (offset + mLength < (unsigned int) mat.cols) {
            mat(cv::Range(0, mat.rows),
                cv::Range(offset + mLength, mat.cols)).copyTo(after);
        }
    }
    else if (mAxis == 1) {
        // Row
        if (offset > 0) {
            mat(cv::Range(0, offset),
                cv::Range(0, mat.cols)).copyTo(before);
        }

        if (offset + mLength < (unsigned int) mat.cols) {
            mat(cv::Range(offset + mLength, mat.rows),
                cv::Range(0, mat.cols)).copyTo(after);
        }
    }
    else {
        throw std::runtime_error("StripeRemoveTransformation::apply(): axis "
                                 "not within range");
    }

    if (!before.empty() && !after.empty()) {
        if (mAxis == 0)
            cv::hconcat(before, after, mat);
        else
            cv::vconcat(before, after, mat);
    }
    else if (!before.empty())
        mat = before;
    else
        mat = after;
}
