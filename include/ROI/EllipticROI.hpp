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

#ifndef N2D2_ELLIPTICROI_H
#define N2D2_ELLIPTICROI_H

#include "Geometric/Ellipse.hpp"
#include "ROI/ROI.hpp"
#include "utils/Utils.hpp"

namespace N2D2 {
template <class T>
class EllipticROI : public ROI, public Geometric::Ellipse<T> {
public:
    using ROI::append;

    EllipticROI(int label,
                const typename Geometric::Ellipse<T>::Point_T& center,
                double majorRadius,
                double minorRadius,
                double angle)
        : ROI(label),
          Geometric::Ellipse<T>(center, majorRadius, minorRadius, angle) {};
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
    std::shared_ptr<EllipticROI<T> > clone() const
    {
        return std::shared_ptr<EllipticROI<T> >(doClone());
    }
    virtual ~EllipticROI() {};

    using Geometric::Ellipse<T>::center;
    using Geometric::Ellipse<T>::majorRadius;
    using Geometric::Ellipse<T>::minorRadius;
    using Geometric::Ellipse<T>::angle;

private:
    virtual EllipticROI<T>* doClone() const
    {
        return new EllipticROI<T>(*this);
    }
};
}

template <class T> cv::Rect N2D2::EllipticROI<T>::getBoundingRect() const
{
    // Parametrized equations for an ellipse rotated at an arbitrary angle:
    // x = center.x + majorRadius*cos(t)*cos(angle) -
    // minorRadius*sin(t)*sin(angle)
    // y = center.y + minorRadius*sin(t)*cos(angle) +
    // majorRadius*cos(t)*sin(angle)

    // Differentiate and solve for gradient = 0:
    // 0 = dx/dt = -majorRadius*sin(t)*cos(angle) -
    // minorRadius*cos(t)*sin(angle)
    // => tan(t) = -minorRadius*tan(angle)/majorRadius
    // 0 = dy/dt = minorRadius*cos(t)*cos(angle) - majorRadius*sin(t)*sin(angle)
    // => tan(t) = minorRadius*cotan(angle)/majorRadius = minorRadius*tan(PI/2 -
    // angle)/majorRadius

    const double tx1 = std::atan(-minorRadius * std::tan(angle) / majorRadius);
    const double tx2 = tx1 + M_PI;

    const double x1 = center.x + majorRadius * std::cos(tx1) * std::cos(angle)
                      - minorRadius * std::sin(tx1) * std::sin(angle);
    const double x2 = center.x + majorRadius * std::cos(tx2) * std::cos(angle)
                      - minorRadius * std::sin(tx2) * std::sin(angle);

    const double ty1
        = std::atan(minorRadius * std::tan(M_PI / 2.0 - angle) / majorRadius);
    const double ty2 = ty1 + M_PI;

    const double y1 = center.y + minorRadius * std::sin(ty1) * std::cos(angle)
                      + majorRadius * std::cos(ty1) * std::sin(angle);
    const double y2 = center.y + minorRadius * std::sin(ty2) * std::cos(angle)
                      + majorRadius * std::cos(ty2) * std::sin(angle);

    const cv::Point tl(std::min(x1, x2), std::min(y1, y2));
    const cv::Point br(std::max(x1, x2), std::max(y1, y2));
    return cv::Rect(tl, br);
}

template <class T>
cv::Mat N2D2::EllipticROI
    <T>::draw(cv::Mat& stimulus, const cv::Scalar& color, int thickness) const
{
    cv::Mat res(stimulus);
    cv::ellipse(res,
                center,
                cv::Size(majorRadius, minorRadius),
                Utils::radToDeg(angle),
                0.0,
                360.0,
                color,
                thickness);
    return res;
}

template <class T>
void N2D2::EllipticROI<T>::append(cv::Mat& labels,
                                  unsigned int outsideMargin,
                                  int outsideLabel) const
{
    if (outsideMargin > 0) {
        cv::Mat labelsOverlap = labels.clone();
        cv::ellipse(
            labelsOverlap,
            center,
            cv::Size(majorRadius + outsideMargin, minorRadius + outsideMargin),
            Utils::radToDeg(angle),
            0.0,
            360.0,
            cv::Scalar(-1),
            -1);
        labelsOverlap.copyTo(labels, (labels == outsideLabel));
    }

    cv::ellipse(labels,
                center,
                cv::Size(majorRadius, minorRadius),
                Utils::radToDeg(angle),
                0.0,
                360.0,
                cv::Scalar(mLabel),
                -1);
}

template <class T>
void N2D2::EllipticROI<T>::rescale(double xRatio, double yRatio)
{
    center.x *= xRatio;
    center.y *= yRatio;

    // x0=xc+M*cos(a) & y0=yc+M*sin(a)
    // x0*xr=M'*cos(a') & y0*yr=M'*sin(a')
    // tan(a')=(y0*yr)/(x0*xr)
    // a'=atan((y0*yr)/(x0*xr))
    // (x0*xr)^2+(y0*yr)^2=(M')^2*(cos(a')^2+sin(a')^2)
    // M'=sqrt((x0*xr)^2+(y0*yr)^2)

    // x1=-m*sin(a) &  y1=m*cos(a)
    // x1*xr=-m'*sin(a') & y1*yr=m'*cos(a')
    // (x1*xr)^2+(y1*yr)^2=(m')^2*(cos(a')^2+sin(a')^2)
    // m'=sqrt((x1*xr)^2+(y1*yr)^2)

    majorRadius *= std::sqrt(std::pow(std::cos(angle) * xRatio, 2)
                             + std::pow(std::sin(angle) * yRatio, 2));
    minorRadius *= std::sqrt(std::pow(std::sin(angle) * xRatio, 2)
                             + std::pow(std::cos(angle) * yRatio, 2));
    angle = std::atan((yRatio / xRatio) * std::tan(angle));
}

template <class T>
void N2D2::EllipticROI<T>::padCrop(int offsetX,
                                   int offsetY,
                                   unsigned int /*width*/,
                                   unsigned int /*height*/)
{
    center.x -= offsetX;
    center.y -= offsetY;
}

template <class T>
void N2D2::EllipticROI
    <T>::flip(unsigned int width, unsigned int height, bool hFlip, bool vFlip)
{
    if (hFlip) {
        center.x = width - center.x;
        angle = -angle;
    }

    if (vFlip) {
        center.y = height - center.y;
        angle = M_PI - angle;
    }
}

#endif // N2D2_ELLIPTICROI_H
