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

#ifndef N2D2_FLIPTRANSFORMATION_H
#define N2D2_FLIPTRANSFORMATION_H

#include "Transformation.hpp"

namespace N2D2 {
class FlipTransformation : public Transformation {
public:
    using Transformation::apply;

    static const char* Type;

    FlipTransformation(bool horizontalFlip = false, bool verticalFlip = false);
    FlipTransformation(const FlipTransformation& trans);
    const char* getType() const
    {
        return Type;
    };
    void apply(cv::Mat& frame,
               cv::Mat& labels,
               std::vector<std::shared_ptr<ROI> >& labelsROI,
               int /*id*/ = -1);
    void reverse(cv::Mat& frame,
                 cv::Mat& labels,
                 std::vector<std::shared_ptr<ROI> >& labelsROI,
                 int /*id*/ = -1);
    std::shared_ptr<FlipTransformation> clone() const
    {
        return std::shared_ptr<FlipTransformation>(doClone());
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
    virtual ~FlipTransformation() {};
    bool getHorizontalFlip(){
        return mHorizontalFlip;
    };
    bool getVerticalFlip(){
        return mVerticalFlip;
    };
    bool getRandomHorizontalFlip(){
        return mRandomHorizontalFlip;
    };
    bool getRandomVerticalFlip(){
        return mRandomVerticalFlip;
    };

private:
    virtual FlipTransformation* doClone() const
    {
        return new FlipTransformation(*this);
    }
    void flip(cv::Mat& mat, int flipCode) const;

    const bool mHorizontalFlip;
    const bool mVerticalFlip;

    /// Enable random image horizontal flip
    Parameter<bool> mRandomHorizontalFlip;
    /// Enable random image vertical flip
    Parameter<bool> mRandomVerticalFlip;
};
}

#endif // N2D2_FLIPTRANSFORMATION_H
