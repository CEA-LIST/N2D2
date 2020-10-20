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

#ifndef N2D2_LABELSLICEEXTRACTIONTRANSFORMATION_H
#define N2D2_LABELSLICEEXTRACTIONTRANSFORMATION_H

#include "Transformation/Transformation.hpp"
#include "utils/Parameterizable.hpp"

namespace N2D2 {
class LabelSliceExtractionTransformation : public Transformation {
public:
    using Transformation::apply;

    static const char* Type;

    LabelSliceExtractionTransformation(unsigned int width,
                                       unsigned int height,
                                       int label = -1);
    LabelSliceExtractionTransformation(
        const LabelSliceExtractionTransformation& trans);
    const char* getType() const
    {
        return Type;
    };
    void apply(cv::Mat& frame,
               cv::Mat& labels,
               std::vector<std::shared_ptr<ROI> >& labelsROI,
               int /*id*/ = -1);
    std::shared_ptr<LabelSliceExtractionTransformation> clone() const
    {
        return std::shared_ptr<LabelSliceExtractionTransformation>(doClone());
    }
    int getLastLabel() const
    {
        return mLastLabel;
    };
    cv::Rect getLastSlice() const
    {
        return mLastSlice;
    };
    std::pair<unsigned int, unsigned int>
    getOutputsSize(unsigned int /*width*/, unsigned int /*height*/) const
    {
        return std::make_pair(mWidth, mHeight);
    };
    int getOutputsDepth(int depth) const
    {
        return depth;
    };
    virtual ~LabelSliceExtractionTransformation();

private:
    struct Pos_T {
        Pos_T(int x_ = 0, int y_ = 0) : x(x_), y(y_)
        {
        }
        int x;
        int y;
    };

    virtual LabelSliceExtractionTransformation* doClone() const
    {
        return new LabelSliceExtractionTransformation(*this);
    }
    std::vector<int> unique(const cv::Mat& mat) const;
    bool loadLabelRandomPos(const std::string& fileName, Pos_T& pos);
    void loadLabelPosCache(const std::string& fileName,
                           std::vector<Pos_T>& labelPos);
    void saveLabelPosCache(const std::string& fileName,
                           const std::vector<Pos_T>& labelPos);

    const unsigned int mWidth;
    const unsigned int mHeight;
    const int mLabel;
    int mLastLabel;
    cv::Rect mLastSlice;
    cv::Mat mElementDilate;
    cv::Mat mElementErode;
    std::map<int, std::vector<int> > mUniqueLabels;

    Parameter<int> mSlicesMargin;
    Parameter<bool> mKeepComposite;
    /// If true, enable random rotations on slices
    Parameter<bool> mRandomRotation;
    /// Range of the random rotation (in deg, counterclockwise),
    /// default is [0.0 360.0] (any rotation)
    Parameter<std::vector<double> > mRandomRotationRange;
};
}

#endif // N2D2_LABELSLICEEXTRACTIONTRANSFORMATION_H
