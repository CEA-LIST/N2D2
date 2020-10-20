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

#ifndef N2D2_SLICEEXTRACTIONTRANSFORMATION_H
#define N2D2_SLICEEXTRACTIONTRANSFORMATION_H

#include "Transformation/Transformation.hpp"

namespace N2D2 {
class SliceExtractionTransformation : public Transformation {
public:
    using Transformation::apply;

    enum BorderType {
        ConstantBorder = cv::BORDER_CONSTANT,
        ReplicateBorder = cv::BORDER_REPLICATE,
        ReflectBorder = cv::BORDER_REFLECT,
        WrapBorder = cv::BORDER_WRAP,
        MinusOneReflectBorder = cv::BORDER_REFLECT_101,
        MeanBorder
    };

    static const char* Type;

    SliceExtractionTransformation(unsigned int width,
                                  unsigned int height,
                                  unsigned int offsetX = 0,
                                  unsigned int offsetY = 0);
    SliceExtractionTransformation(const SliceExtractionTransformation& trans);
    const char* getType() const
    {
        return Type;
    };
    void apply(cv::Mat& frame,
               cv::Mat& labels,
               std::vector<std::shared_ptr<ROI> >& labelsROI,
               int id = -1);
    void reverse(cv::Mat& frame,
                 cv::Mat& labels,
                 std::vector<std::shared_ptr<ROI> >& labelsROI,
                 int /*id*/ = -1);
    std::shared_ptr<SliceExtractionTransformation> clone() const
    {
        return std::shared_ptr<SliceExtractionTransformation>(doClone());
    }
    std::pair<unsigned int, unsigned int>
    getOutputsSize(unsigned int /*width*/, unsigned int /*height*/) const
    {
        return std::make_pair(mWidth, mHeight);
    };
    int getOutputsDepth(int depth) const
    {
        return depth;
    };
    virtual ~SliceExtractionTransformation();

    static cv::Rect extract(unsigned int x,
                            unsigned int y,
                            unsigned int width,
                            unsigned int height,
                            double rotation,
                            int borderType,
                            const cv::Scalar& bgColor,
                            cv::Mat& frame,
                            cv::Mat& labels,
                            std::vector<std::shared_ptr<ROI> >& labelsROI,
                            int /*id*/ = -1);

private:
    virtual SliceExtractionTransformation* doClone() const
    {
        return new SliceExtractionTransformation(*this);
    }

    const unsigned int mWidth;
    const unsigned int mHeight;

    Parameter<unsigned int> mOffsetX;
    Parameter<unsigned int> mOffsetY;
    Parameter<bool> mRandomOffsetX;
    Parameter<bool> mRandomOffsetY;
    /// If true, enable random rotations on slices
    Parameter<bool> mRandomRotation;
    /// Range of the random rotation (in deg, counterclockwise),
    /// default is [0.0 360.0] (any rotation)
    Parameter<std::vector<double> > mRandomRotationRange;
    /// If true, enable random scaling on slices
    Parameter<bool> mRandomScaling;
    /// Range of the random scaling, default is [0.8 1.2] (20% variation)
    Parameter<std::vector<double> > mRandomScalingRange;
    /// Allow padding (if false and padding should occur, triggers an exception)
    Parameter<bool> mAllowPadding;
    Parameter<BorderType> mBorderType;
    Parameter<std::vector<double> > mBorderValue;
};
}

namespace {
template <>
const char* const EnumStrings
    <N2D2::SliceExtractionTransformation::BorderType>::data[]
    = {"ConstantBorder", "ReplicateBorder", "ReflectBorder", "WrapBorder",
        "MinusOneReflectBorder", "MeanBorder"};
}

#endif // N2D2_SLICEEXTRACTIONTRANSFORMATION_H
