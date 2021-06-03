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

#include "Transformation/RangeAffineTransformation.hpp"

const char* N2D2::RangeAffineTransformation::Type = "RangeAffine";

namespace N2D2 {
class ParallelTruncate : public cv::ParallelLoopBody {
public:
    ParallelTruncate(cv::Mat& frame_)
        : frame(frame_) {}
    virtual void operator ()(const cv::Range& range) const override {
        for (int r = range.start; r < range.end; r++) {
            int i = r / frame.cols;
            int j = r % frame.cols;
            Float_T& value = frame.ptr<Float_T>(i)[j];
            value = std::trunc(value);
        }
    }
    ParallelTruncate& operator=(const ParallelTruncate&) {
        return *this;
    };
private:
    cv::Mat& frame;
};
}

N2D2::RangeAffineTransformation::RangeAffineTransformation(
    Operator firstOperator,
    const std::vector<double>& firstValue,
    Operator secondOperator,
    const std::vector<double>& secondValue)
    : mFirstOperator(firstOperator),
      mFirstValue(firstValue),
      mSecondOperator(secondOperator),
      mSecondValue(secondValue),
      mTruncate(this, "Truncate", false)
{
    // ctor
}

N2D2::RangeAffineTransformation::RangeAffineTransformation(
    Operator firstOperator,
    double firstValue,
    Operator secondOperator,
    double secondValue)
    : mFirstOperator(firstOperator),
      mFirstValue(std::vector<double>(1, firstValue)),
      mSecondOperator(secondOperator),
      mSecondValue(std::vector<double>(1, secondValue)),
      mTruncate(this, "Truncate", false)
{
    // ctor
}

N2D2::RangeAffineTransformation::RangeAffineTransformation(
    const RangeAffineTransformation& trans)
    : mFirstOperator(trans.mFirstOperator),
      mFirstValue(trans.mFirstValue),
      mSecondOperator(trans.mSecondOperator),
      mSecondValue(trans.mSecondValue),
      mTruncate(this, "Truncate", trans.mTruncate)
{
    // copy-ctor
}

void
N2D2::RangeAffineTransformation::apply(cv::Mat& frame,
                                       cv::Mat& /*labels*/,
                                       std::vector
                                       <std::shared_ptr<ROI> >& /*labelsROI*/,
                                       int /*id*/)
{
    cv::Mat frameF;
    frame.convertTo(frameF, opencv_data_type<Float_T>::value);

    const unsigned int nbChannels = frameF.channels();

    if (mFirstValue.size() > 1 && mFirstValue.size() != nbChannels) {
        throw std::runtime_error("RangeAffineTransformation::apply(): the "
                                 "number of values for the first operator must "
                                 "be 1 or match the number of image channels.");
    }

    if (mSecondValue.size() > 1 && mSecondValue.size() != nbChannels) {
        throw std::runtime_error("RangeAffineTransformation::apply(): the "
                                 "number of values for the second operator "
                                 "must be 1 or match the number of image "
                                 "channels.");
    }

    std::vector<cv::Mat> channels;
    cv::split(frameF, channels);

    for (unsigned int ch = 0; ch < nbChannels; ++ch) {
        const double firstValue = (mFirstValue.size() > 1)
            ? mFirstValue[ch] : mFirstValue[0];

        applyOperator(channels[ch], mFirstOperator, firstValue);

        if (!mSecondValue.empty()) {
            const double secondValue = (mSecondValue.size() > 1)
                ? mSecondValue[ch] : mSecondValue[0];

            applyOperator(channels[ch], mSecondOperator, secondValue);
        }

        if (mTruncate) {
            ParallelTruncate parallelTruncate(channels[ch]);
            cv::parallel_for_(
                cv::Range(0, channels[ch].rows * channels[ch].cols),
                parallelTruncate);
        }
    }

    cv::merge(channels, frame);
}

void N2D2::RangeAffineTransformation::applyOperator(
    cv::Mat& mat,
    const Operator& op,
    double value) const
{
    switch (op) {
    case Plus:
        mat += value;
        break;
    case Minus:
        mat -= value;
        break;
    case Multiplies:
        mat *= value;
        break;
    case Divides:
        mat /= value;
        break;
    }
}
