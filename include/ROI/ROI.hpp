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

#ifndef N2D2_ROI_H
#define N2D2_ROI_H

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "containers/Tensor2d.hpp"
#include "utils/Utils.hpp"

namespace N2D2 {
class ROI {
public:
    ROI(int label) : mLabel(label) {};
    int getLabel() const
    {
        return mLabel;
    };
    void setLabel(int label)
    {
        mLabel = label;
    };
    /// Return the bounding rectangle enclosing the ROI shape. In the recturned
    /// cv::Rect, the top and left boundaries of the
    /// rectangle are inclusive, while the right and bottom boundaries are not.
    virtual cv::Rect getBoundingRect() const = 0;
    inline virtual cv::Mat extract(const cv::Mat& stimulus) const;
    virtual cv::Mat draw(cv::Mat& stimulus,
                         const cv::Scalar& color = cv::Scalar(0, 0, 255),
                         int thickness = 1) const = 0;
    virtual void append(cv::Mat& labels,
                        unsigned int outsideMargin = 0,
                        int outsideLabel = 0) const = 0;
    inline void append(Tensor2d<int>& labels,
                       unsigned int outsideMargin = 0,
                       int outsideLabel = 0) const;
    virtual void rescale(double xRatio, double yRatio) = 0;
    virtual void
    padCrop(int offsetX, int offsetY, unsigned int width, unsigned int height)
        = 0;
    virtual void
    flip(unsigned int width, unsigned int height, bool hFlip, bool vFlip) = 0;
    std::shared_ptr<ROI> clone() const
    {
        return std::shared_ptr<ROI>(doClone());
    }
    ROI* clonePtr() const
    {
        return doClone();
    }
    virtual ~ROI() {};

protected:
    int mLabel;

private:
    virtual ROI* doClone() const = 0;
};
}

cv::Mat N2D2::ROI::extract(const cv::Mat& stimulus) const
{
    cv::Rect rect = getBoundingRect();

    if (rect.x < 0) {
        std::cout << Utils::cwarning << "ROI::extract(): BB.x0 (" << rect.x
                  << ") < 0" << Utils::cdef << std::endl;
        rect.width+= rect.x;
        rect.x = 0;
    } else if (rect.x > stimulus.cols) {
        std::cout << Utils::cwarning << "ROI::extract(): BB.x0 (" << rect.x
                  << ") > " << stimulus.cols << Utils::cdef << std::endl;
        rect.x = stimulus.cols;
    }

    if (rect.y < 0) {
        std::cout << Utils::cwarning << "ROI::extract(): BB.y0 (" << rect.y
                  << ") < 0" << Utils::cdef << std::endl;
        rect.height+= rect.y;
        rect.y = 0;
    } else if (rect.y > stimulus.rows) {
        std::cout << Utils::cwarning << "ROI::extract(): BB.y0 (" << rect.y
                  << ") > " << stimulus.rows << Utils::cdef << std::endl;
        rect.y = stimulus.rows;
    }

    if (rect.x + rect.width > stimulus.cols) {
        std::cout << Utils::cwarning << "ROI::extract(): BB.x1 (" << rect.x
                  << " + " << rect.width << ") > " << stimulus.cols
                  << Utils::cdef << std::endl;
        rect.width = stimulus.cols - rect.x;
    }

    if (rect.y + rect.height > stimulus.rows) {
        std::cout << Utils::cwarning << "ROI::extract(): BB.y1 (" << rect.y
                  << " + " << rect.height << ") > " << stimulus.rows
                  << Utils::cdef << std::endl;
        rect.height = stimulus.rows - rect.y;
    }

    return stimulus(rect);
}

void N2D2::ROI::append(Tensor2d<int>& labels,
                       unsigned int outsideMargin,
                       int outsideLabel) const
{
    cv::Mat mat = (cv::Mat)labels;
    append(mat, outsideMargin, outsideLabel);
    labels = Tensor2d<int>(mat);
}

#endif // N2D2_ROI_H
