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

#ifndef N2D2_CIRCULARROI_H
#define N2D2_CIRCULARROI_H

#include "Geometric/Circle.hpp"
#include "ROI/ROI.hpp"

namespace N2D2 {
template <class T> class CircularROI : public ROI, public Geometric::Circle<T> {
public:
    using ROI::append;

    CircularROI(int label,
                const typename Geometric::Circle<T>::Point_T& center,
                double radius)
        : ROI(label), Geometric::Circle<T>(center, radius) {};
    inline cv::Rect getBoundingRect() const;
    inline cv::Mat draw(cv::Mat& stimulus,
                        const cv::Scalar& color = cv::Scalar(0, 0, 255),
                        int thickness = 1) const;
    inline void append(cv::Mat& labels,
                       unsigned int outsideMargin = 0,
                       int outsideLabel = 0) const;
    inline void rescale(double xRatio, double yRatio);
    inline void
    padCrop(int offsetX, int offsetY, unsigned int width, unsigned int height);
    inline void
    flip(unsigned int width, unsigned int height, bool hFlip, bool vFlip);
    std::shared_ptr<CircularROI<T> > clone() const
    {
        return std::shared_ptr<CircularROI<T> >(doClone());
    }
    virtual ~CircularROI() {};

    using Geometric::Circle<T>::center;
    using Geometric::Circle<T>::radius;

private:
    virtual CircularROI<T>* doClone() const
    {
        return new CircularROI<T>(*this);
    }
};
}

template <class T> cv::Rect N2D2::CircularROI<T>::getBoundingRect() const
{
    const cv::Point tl(center.x - radius, center.y - radius);
    const cv::Point br(center.x + radius, center.y + radius);
    return cv::Rect(tl, br);
}

template <class T>
cv::Mat N2D2::CircularROI
    <T>::draw(cv::Mat& stimulus, const cv::Scalar& color, int thickness) const
{
    cv::Mat res(stimulus);
    cv::circle(res, center, radius, color, thickness);
    return res;
}

template <class T>
void N2D2::CircularROI<T>::append(cv::Mat& labels,
                                  unsigned int outsideMargin,
                                  int outsideLabel) const
{
    if (outsideMargin > 0) {
        cv::Mat labelsOverlap = labels.clone();
        cv::circle(
            labelsOverlap, center, radius + outsideMargin, cv::Scalar(-1), -1);
        labelsOverlap.copyTo(labels, (labels == outsideLabel));
    }

    cv::circle(labels, center, radius, cv::Scalar(mLabel), -1);
}

template <class T>
void N2D2::CircularROI<T>::rescale(double xRatio, double yRatio)
{
    center.x *= xRatio;
    center.y *= yRatio;

    if (xRatio != yRatio)
        throw std::runtime_error("CircularROI::rescale() with different x and "
                                 "y ratio is not supported (use EllipticROI "
                                 "instead)");

    radius *= xRatio;
}

template <class T>
void N2D2::CircularROI<T>::padCrop(int offsetX,
                                   int offsetY,
                                   unsigned int /*width*/,
                                   unsigned int /*height*/)
{
    center.x -= offsetX;
    center.y -= offsetY;
}

template <class T>
void N2D2::CircularROI
    <T>::flip(unsigned int width, unsigned int height, bool hFlip, bool vFlip)
{
    if (hFlip)
        center.x = width - center.x;

    if (vFlip)
        center.y = height - center.y;
}

#endif // N2D2_CIRCULARROI_H
