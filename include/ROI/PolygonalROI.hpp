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

#ifndef N2D2_POLYGONALROI_H
#define N2D2_POLYGONALROI_H

#include "Geometric/Polygon.hpp"
#include "ROI/ROI.hpp"

namespace N2D2 {
template <class T>
class PolygonalROI : public ROI, public virtual Geometric::Polygon<T> {
public:
    using ROI::append;

    PolygonalROI(int label,
                 const std::vector
                 <typename Geometric::Polygon<T>::Point_T>& points = std::vector
                 <typename Geometric::Polygon<T>::Point_T>())
        : Geometric::Polygon<T>(points), ROI(label) {};
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
    std::shared_ptr<PolygonalROI<T> > clone() const
    {
        return std::shared_ptr<PolygonalROI<T> >(doClone());
    }
    virtual ~PolygonalROI() {};

    using Geometric::Polygon<T>::points;

private:
    virtual PolygonalROI<T>* doClone() const
    {
        return new PolygonalROI<T>(*this);
    }
};
}

template <class T> cv::Rect N2D2::PolygonalROI<T>::getBoundingRect() const
{
    cv::Rect rect = cv::boundingRect(points);
    // If the polygonal is a rectangle, the returned Rect BR point should not be
    // inclusive, but the cv::boundingRect() includes it.
    // That's why the width and height of the returned Rect must be reduced by
    // 1.
    rect.width -= 1;
    rect.height -= 1;
    return rect;
}

template <class T>
cv::Mat N2D2::PolygonalROI
    <T>::draw(cv::Mat& stimulus, const cv::Scalar& color, int thickness) const
{
    const typename Geometric::Polygon<T>::Point_T* pts[1] = {&points[0]};
    int npts = (int)points.size();

    cv::Mat res(stimulus);
    cv::polylines(res, pts, &npts, 1, true, color, thickness);
    return res;
}

template <class T>
void N2D2::PolygonalROI<T>::append(cv::Mat& labels,
                                   unsigned int outsideMargin,
                                   int outsideLabel) const
{
    // Draw on a sub-image, because drawing functions overflows for coordinates
    // > 65535
    const cv::Rect bb = cv::boundingRect(points);
    const int x0 = std::max(0, bb.x - 2 * (int)outsideMargin);
    const int y0 = std::max(0, bb.y - 2 * (int)outsideMargin);
    cv::Rect workArea(
        x0,
        y0,
        std::min(labels.cols - x0, bb.width + 4 * (int)outsideMargin),
        std::min(labels.rows - y0, bb.height + 4 * (int)outsideMargin));

    std::vector<typename Geometric::Polygon<T>::Point_T> newPoints;

    for (unsigned int i = 0; i < points.size(); ++i)
        newPoints.push_back(typename Geometric::Polygon<T>::Point_T(
            points[i].x - workArea.x, points[i].y - workArea.y));

    const typename Geometric::Polygon<T>::Point_T* newPts[1] = {&newPoints[0]};
    int newNpts = (int)newPoints.size();

    cv::Mat labelsWorkArea = labels(workArea);

    if (outsideMargin > 0) {
        cv::Mat labelsOverlap = labels(workArea).clone();
        cv::polylines(labelsOverlap,
                      newPts,
                      &newNpts,
                      1,
                      true,
                      cv::Scalar(-1),
                      2 * outsideMargin);
        labelsOverlap.copyTo(labelsWorkArea, (labelsWorkArea == outsideLabel));
    }

    cv::fillPoly(labelsWorkArea, newPts, &newNpts, 1, cv::Scalar(mLabel));

    /*
        // Original code:
        const typename Geometric::Polygon<T>::Point_T* pts[1] = { &points[0] };
        int npts = (int) points.size();

        if (outsideMargin > 0) {
            cv::Mat labelsOverlap = labels.clone();
            cv::polylines(labelsOverlap, pts, &npts, 1, true, cv::Scalar(-1),
       2*outsideMargin);
            labelsOverlap.copyTo(labels, (labels == outsideLabel));
        }

        cv::fillPoly(labels, pts, &npts, 1, cv::Scalar(mLabel));
    */
}

template <class T>
void N2D2::PolygonalROI<T>::rescale(double xRatio, double yRatio)
{
    for (unsigned int i = 0; i < points.size(); ++i) {
        points[i].x *= xRatio;
        points[i].y *= yRatio;
    }
}

template <class T>
void N2D2::PolygonalROI<T>::padCrop(int offsetX,
                                    int offsetY,
                                    unsigned int /*width*/,
                                    unsigned int /*height*/)
{
    for (unsigned int i = 0; i < points.size(); ++i) {
        points[i].x -= offsetX;
        points[i].y -= offsetY;
    }
}

template <class T>
void N2D2::PolygonalROI
    <T>::flip(unsigned int width, unsigned int height, bool hFlip, bool vFlip)
{
    for (unsigned int i = 0; i < points.size(); ++i) {
        if (hFlip)
            points[i].x = width - points[i].x;

        if (vFlip)
            points[i].y = height - points[i].y;
    }
}

#endif // N2D2_POLYGONALROI_H
