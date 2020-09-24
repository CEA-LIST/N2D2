/*
    (C) Copyright 2016 CEA LIST. All Rights Reserved.
    Contributor(s): Olivier BICHLER (olivier.bichler@cea.fr)
                    Benjamin BERTELONE (benjamin.bertelone@cea.fr)
                    Alexandre CARBON (alexandre.carbon@cea.fr)

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

#ifndef N2D2_LABELEXTRACTIONTRANSFORMATION_H
#define N2D2_LABELEXTRACTIONTRANSFORMATION_H

#include "Transformation/Transformation.hpp"

namespace N2D2 {
class LabelExtractionTransformation : public Transformation {
public:
    using Transformation::apply;

    static const char* Type;

    LabelExtractionTransformation(const std::string& widths,
                                  const std::string& heights,
                                  int label = -1,
                                  std::string distributions = "Auto");
    const char* getType() const
    {
        return Type;
    };

    void apply(cv::Mat& frame,
               cv::Mat& labels,
               std::vector<std::shared_ptr<ROI> >& labelsROI,
               int id = -1);

    std::shared_ptr<LabelExtractionTransformation> clone() const
    {
        return std::shared_ptr<LabelExtractionTransformation>(doClone());
    }

    int getLastLabel() const
    {
        return mLastLabel;
    };
    cv::Rect getLastSlice() const
    {
        return mLastSlice;
    };

    int getOutputsDepth(int depth) const
    {
        return depth;
    };
    virtual ~LabelExtractionTransformation();

private:
    struct Pos_T {
        Pos_T(int x_ = 0, int y_ = 0) : x(x_), y(y_)
        {
        }
        int x;
        int y;
    };

    virtual LabelExtractionTransformation* doClone() const
    {
        return new LabelExtractionTransformation(*this);
    }
    std::vector<int> unique(const cv::Mat& mat) const;

    bool loadLabelRandomPos(const std::string& fileName, Pos_T& pos);

    void loadLabelPosCache(const std::string& fileName,
                           std::vector<Pos_T>& labelPos);
    void saveLabelPosCache(const std::string& fileName,
                           const std::vector<Pos_T>& labelPos);

    int getRandomLabel(std::vector<int>& labels);

    void smartErode(cv::Mat& labels,
                    std::vector<std::shared_ptr<ROI> >& labelsROI,
                    int width,
                    int height,
                    int id);

    std::vector<int> mWidths;
    std::vector<int> mHeights;
    const int mLabel;

    std::map<int, std::map<unsigned int, std::vector<int> > > mUniqueLabels;
    std::map<int, int> mLabelDistribution;

    bool mAutoWantedLabelDistribution;
    std::map<int, float> mWantedLabelDistribution;

    int mLastLabel;
    cv::Rect mLastSlice;

    double mDuration[1000];

    // int mLastLabel;

    // cv::Mat mElement;
    // std::map<int, std::vector<int> > mUniqueLabels;

    // Parameter<int> mSlicesMargin;
};
}

#endif // N2D2_LABELEXTRACTIONTRANSFORMATION_H
