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

#ifndef N2D2_MORPHOLOGYTRANSFORMATION_H
#define N2D2_MORPHOLOGYTRANSFORMATION_H

#include "Transformation/Transformation.hpp"

namespace N2D2 {
class MorphologyTransformation : public Transformation {
public:
    using Transformation::apply;

    enum Operation {
        Erode,
        Dilate,
        Opening,
        Closing,
        Gradient,
        TopHat,
        BlackHat
    };
    enum Shape {
        Rectangular,
        Elliptic,
        Cross
    };

    static const char* Type;

    MorphologyTransformation(Operation operation,
                             unsigned int size,
                             bool applyToLabels = false);
    MorphologyTransformation(const MorphologyTransformation& trans);
    const char* getType() const
    {
        return Type;
    };
    void apply(cv::Mat& frame,
               cv::Mat& labels,
               std::vector<std::shared_ptr<ROI> >& /*labelsROI*/,
               int /*id*/ = -1);
    std::shared_ptr<MorphologyTransformation> clone() const
    {
        return std::shared_ptr<MorphologyTransformation>(doClone());
    }
    std::pair<unsigned int, unsigned int>
    getOutputsSize(unsigned int width, unsigned int height) const
    {
        return std::make_pair(width, height);
    };
    int getOutputsDepth(int depth) const
    {
        return depth;
    };
    virtual ~MorphologyTransformation();

private:
    virtual MorphologyTransformation* doClone() const
    {
        return new MorphologyTransformation(*this);
    }
    void applyMorphology(cv::Mat& mat) const;

    const Operation mOperation;
    const unsigned int mSize;
    const bool mApplyToLabels;

    Parameter<bool> mLabelsIgnoreDiff;
    Parameter<Shape> mShape;
    Parameter<unsigned int> mNbIterations;
    Parameter<std::vector<int> > mLabel;

    cv::Mat mKernel;
};
}

namespace {
template <>
const char* const EnumStrings<N2D2::MorphologyTransformation::Operation>::data[]
    = {"Erode",    "Dilate", "Opening", "Closing",
       "Gradient", "TopHat", "BlackHat"};
}

namespace {
template <>
const char* const EnumStrings<N2D2::MorphologyTransformation::Shape>::data[]
    = {"Rectangular", "Elliptic", "Cross"};
}

#endif // N2D2_MORPHOLOGYTRANSFORMATION_H
