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

#ifndef N2D2_RANDOMRESIZECROPTRANSFORMATION_H
#define N2D2_RANDOMRESIZECROPTRANSFORMATION_H

#include "Transformation/Transformation.hpp"

namespace N2D2 {
class RandomResizeCropTransformation : public Transformation {
public:
    using Transformation::apply;

    struct BBox_T {
        int x;
        int y;
        int w;
        int h;

        BBox_T() {}
        BBox_T(int x_, int y_, int w_, int h_):
            x(x_), y(y_), w(w_), h(h_) {}
    };

    static const char* Type;

    RandomResizeCropTransformation(unsigned int width,
                                  unsigned int height);
    RandomResizeCropTransformation(const RandomResizeCropTransformation& trans);
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
    std::shared_ptr<RandomResizeCropTransformation> clone() const
    {
        return std::shared_ptr<RandomResizeCropTransformation>(doClone());
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
    virtual ~RandomResizeCropTransformation();

    static cv::Rect extract(unsigned int x,
                            unsigned int y,
                            unsigned int width,
                            unsigned int height,
                            cv::Mat& frame,
                            cv::Mat& labels,
                            std::vector
                            <std::shared_ptr<ROI> >& labelsROI,
                            int id = 1);
    unsigned int getWidth(){
        return mWidth;
    };
    unsigned int getHeight(){
        return mHeight;
    };
    float getScaleMin(){
        return mScaleMin;
    };
    float getScaleMax(){
        return mScaleMax;
    };
    float getRatioMin(){
        return mRatioMin;
    };
    float getRatioMax(){
        return mRatioMax;
    };

private:
    virtual RandomResizeCropTransformation* doClone() const
    {
        return new RandomResizeCropTransformation(*this);
    }

    const unsigned int mWidth;
    const unsigned int mHeight;
    Parameter<float> mScaleMin;
    Parameter<float> mScaleMax;
    Parameter<float> mRatioMin;
    Parameter<float> mRatioMax;
};
}

#endif // N2D2_RANDOMRESIZECROPTRANSFORMATION_H
