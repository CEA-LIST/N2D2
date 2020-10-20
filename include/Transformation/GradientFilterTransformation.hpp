/*
    (C) Copyright 2017 CEA LIST. All Rights Reserved.
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

#ifndef N2D2_GRADIENTFILTERTRANSFORMATION_H
#define N2D2_GRADIENTFILTERTRANSFORMATION_H

#include "Transformation/Transformation.hpp"

namespace N2D2 {
class GradientFilterTransformation : public Transformation {
public:
    using Transformation::apply;

    enum GradientFilter {
        Sobel,
        Scharr,
        Laplacian
    };

    static const char* Type;

    GradientFilterTransformation(double scale = 1.0,
                                 double delta = 0.0,
                                 bool applyToLabels = false);
    GradientFilterTransformation(const GradientFilterTransformation& trans);
    const char* getType() const
    {
        return Type;
    };
    void apply(cv::Mat& frame,
               cv::Mat& /*labels*/,
               std::vector<std::shared_ptr<ROI> >& /*labelsROI*/,
               int /*id*/ = -1);
    std::shared_ptr<GradientFilterTransformation> clone() const
    {
        return std::shared_ptr<GradientFilterTransformation>(doClone());
    }
    std::pair<unsigned int, unsigned int>
    getOutputsSize(unsigned int width, unsigned int height) const
    {
        return std::make_pair(width, height);
    };
    int getOutputsDepth(int depth) const
    {
        if (mApplyToLabels)
            return depth;
        else
            return CV_32F;
    };
    virtual ~GradientFilterTransformation();

private:
    virtual GradientFilterTransformation* doClone() const
    {
        return new GradientFilterTransformation(*this);
    }

    const double mScale;
    const double mDelta;
    const bool mApplyToLabels;

    Parameter<GradientFilter> mGradientFilter;
    Parameter<int> mKernelSize;
    Parameter<bool> mInvThreshold;
    Parameter<double> mThreshold;
    Parameter<std::vector<int> > mLabel;
    Parameter<double> mGradientScale;
};
}

namespace {
template <>
const char* const EnumStrings<N2D2::GradientFilterTransformation
    ::GradientFilter>::data[] = {"Sobel", "Scharr", "Laplacian"};
}

#endif // N2D2_GRADIENTFILTERTRANSFORMATION_H
