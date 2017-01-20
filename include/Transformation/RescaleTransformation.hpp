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

#ifndef N2D2_RESCALETRANSFORMATION_H
#define N2D2_RESCALETRANSFORMATION_H

#include "Transformation.hpp"

namespace N2D2 {
class RescaleTransformation : public Transformation {
public:
    using Transformation::apply;

    RescaleTransformation(unsigned int width, unsigned int height);
    void apply(cv::Mat& frame,
               cv::Mat& labels,
               std::vector<std::shared_ptr<ROI> >& labelsROI,
               int /*id*/ = -1);
    void reverse(cv::Mat& frame,
                 cv::Mat& labels,
                 std::vector<std::shared_ptr<ROI> >& labelsROI,
                 int /*id*/ = -1);
    std::shared_ptr<RescaleTransformation> clone() const
    {
        return std::shared_ptr<RescaleTransformation>(doClone());
    }
    std::pair<unsigned int, unsigned int>
    getOutputsSize(unsigned int /*width*/, unsigned int /*height*/) const
    {
        return (!mKeepAspectRatio) ? std::make_pair(mWidth, mHeight)
                                   : std::make_pair(0U, 0U);
    };
    virtual ~RescaleTransformation() {};

private:
    virtual RescaleTransformation* doClone() const
    {
        return new RescaleTransformation(*this);
    }
    void resize(cv::Mat& mat,
                int interpolation,
                std::vector<std::shared_ptr<ROI> >& labelsROI) const;

    const unsigned int mWidth;
    const unsigned int mHeight;

    /// If true, keep the aspect ratio of the images and use padding
    Parameter<bool> mKeepAspectRatio;
    /// If true, resize along the longest dimension when keep the aspect ratio
    /// is true
    Parameter<bool> mResizeToFit;
};
}

#endif // N2D2_RESCALETRANSFORMATION_H
