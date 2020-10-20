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

#ifndef N2D2_MORPHOLOGICALRECONSTRUCTIONTRANSFORMATION_H
#define N2D2_MORPHOLOGICALRECONSTRUCTIONTRANSFORMATION_H

#include "Transformation/Transformation.hpp"

namespace N2D2 {
class MorphologicalReconstructionTransformation : public Transformation {
public:
    using Transformation::apply;

    enum Operation {
        ReconstructionByErosion,
        ReconstructionByDilation,
        OpeningByReconstruction,
        ClosingByReconstruction
    };
    enum Shape {
        Rectangular,
        Elliptic,
        Cross
    };

    static const char* Type;

    MorphologicalReconstructionTransformation(Operation operation,
                                              unsigned int size,
                                              bool applyToLabels = false);
    MorphologicalReconstructionTransformation(
        const MorphologicalReconstructionTransformation& trans);
    const char* getType() const
    {
        return Type;
    };
    void apply(cv::Mat& frame,
               cv::Mat& labels,
               std::vector<std::shared_ptr<ROI> >& /*labelsROI*/,
               int /*id*/ = -1);
    std::shared_ptr<MorphologicalReconstructionTransformation> clone() const
    {
        return std::shared_ptr
            <MorphologicalReconstructionTransformation>(doClone());
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
    virtual ~MorphologicalReconstructionTransformation();

private:
    enum ReconstructionType {
        ByErosion,
        ByDilation
    };

    virtual MorphologicalReconstructionTransformation* doClone() const
    {
        return new MorphologicalReconstructionTransformation(*this);
    }
    void applyReconstruction(cv::Mat& mat) const;
    cv::Mat geodesicErosion(const cv::Mat& image,
                            const cv::Mat& kernel,
                            const cv::Mat& mask) const;
    cv::Mat geodesicDilation(const cv::Mat& image,
                             const cv::Mat& kernel,
                             const cv::Mat& mask) const;
    cv::Mat reconstruction(const cv::Mat& image,
                           const cv::Mat& kernel,
                           const cv::Mat& mask,
                           ReconstructionType type) const;

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
const char* const EnumStrings
    <N2D2::MorphologicalReconstructionTransformation::Operation>::data[]
    = {"ReconstructionByErosion", "ReconstructionByDilation",
       "OpeningByReconstruction", "ClosingByReconstruction"};
}

namespace {
template <>
const char* const EnumStrings
    <N2D2::MorphologicalReconstructionTransformation::Shape>::data[]
    = {"Rectangular", "Elliptic", "Cross"};
}

#endif // N2D2_MORPHOLOGICALRECONSTRUCTIONTRANSFORMATION_H
