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

#ifndef N2D2_BITMAPROI_H
#define N2D2_BITMAPROI_H

#include "ROI/ROI.hpp"

namespace N2D2 {
template <class T> class BitmapROI : public ROI {
public:
    using ROI::append;

    BitmapROI(int label,
              cv::Point origin,
              int scale,
              const cv::Mat& data)
        : ROI(label),
          mOrigin(origin),
          mScale(scale),
          mData(data) {};
    inline cv::Rect getBoundingRect() const;
    inline cv::Mat draw(cv::Mat& stimulus,
                        const cv::Scalar& color = cv::Scalar(0, 0, 255),
                        int thickness = 1) const;
    inline void append(cv::Mat& labels,
                       unsigned int outsideMargin = 0,
                       int outsideLabel = 0) const;
    inline void rescale(double xRatio, double yRatio);
    inline void rotate(int centerX, int centerY, double angle);
    inline void
    padCrop(int offsetX, int offsetY, unsigned int width, unsigned int height);
    inline void
    flip(unsigned int width, unsigned int height, bool hFlip, bool vFlip);
    std::shared_ptr<BitmapROI<T> > clone() const
    {
        return std::shared_ptr<BitmapROI<T> >(doClone());
    }
    virtual ~BitmapROI() {};

private:
    cv::Point mOrigin;
    int mScale;
    cv::Mat mData;

    virtual BitmapROI<T>* doClone() const
    {
        return new BitmapROI<T>(*this);
    }
};
}

template <class T> cv::Rect N2D2::BitmapROI<T>::getBoundingRect() const
{
    const cv::Point tl(mOrigin);
    const cv::Point br(mOrigin.x + mData.cols * std::abs(mScale),
                       mOrigin.y + mData.rows * std::abs(mScale));
    return cv::Rect(tl, br);
}

template <class T>
cv::Mat N2D2::BitmapROI
    <T>::draw(cv::Mat& stimulus, const cv::Scalar& color, int /*thickness*/) const
{
    if (mData.empty()
        || mOrigin.x >= stimulus.cols || mOrigin.y >= stimulus.rows)
    {
        // mOrigin may be outside stimulus after rescale()
        return cv::Mat(stimulus);
    }

    // Rescale mData
    cv::Mat dataToScale;
    cv::resize(mData,
        dataToScale,
        cv::Size(mData.cols * std::abs(mScale), mData.rows * std::abs(mScale)),
        0.0, 0.0, cv::INTER_NEAREST);

    // Crop dataToScale if necessary (it can be the case after rescale())
    dataToScale = dataToScale(cv::Rect(0, 0,
        std::min(stimulus.cols - mOrigin.x, dataToScale.cols),
        std::min(stimulus.rows - mOrigin.y, dataToScale.rows)));

    // Insert rescaled data patch
    const cv::Rect patch = cv::Rect(mOrigin.x, mOrigin.y,
                                    dataToScale.cols, dataToScale.rows);

    cv::Mat res(stimulus);
    res(patch).setTo(color, dataToScale);
    return res;
}

template <class T>
void N2D2::BitmapROI<T>::append(cv::Mat& labels,
                                  unsigned int outsideMargin,
                                  int outsideLabel) const
{
    if (mData.empty()
        || mOrigin.x >= labels.cols || mOrigin.y >= labels.rows)
    {
        // mOrigin may be outside stimulus after rescale()
        return;
    }

    // Rescale mData
    cv::Mat dataToScale;
    cv::resize(mData,
        dataToScale,
        cv::Size(mData.cols * std::abs(mScale), mData.rows * std::abs(mScale)),
        0.0, 0.0, cv::INTER_NEAREST);

    // Crop dataToScale if necessary (it can be the case after rescale())
    dataToScale = dataToScale(cv::Rect(0, 0,
        std::min(labels.cols - std::max(mOrigin.x, 0), dataToScale.cols),
        std::min(labels.rows - std::max(mOrigin.y, 0), dataToScale.rows)));

    // Insert rescaled data patch
    const cv::Rect patch = cv::Rect(
        std::max(mOrigin.x, 0),
        std::max(mOrigin.y, 0),
        dataToScale.cols,
        dataToScale.rows);

    if (outsideMargin > 0) {
        cv::Mat bitmap = cv::Mat::zeros(labels.size(), dataToScale.type());
        bitmap(patch).setTo(cv::Scalar(1), dataToScale);

        const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, 
            cv::Size(outsideMargin, outsideMargin));
        cv::dilate(bitmap, bitmap, kernel);

        cv::Mat mask;
        cv::bitwise_and(labels == outsideLabel, bitmap, mask);
        labels.setTo(cv::Scalar(-1), mask);
    }

    labels(patch).setTo(cv::Scalar(mLabel), dataToScale);
}

template <class T>
void N2D2::BitmapROI<T>::rescale(double xRatio, double yRatio)
{
    mOrigin.x *= xRatio;
    mOrigin.y *= yRatio;

    cv::Mat resizedData;
    cv::resize(mData,
        resizedData,
        cv::Size(mData.cols * xRatio, mData.rows * yRatio),
        0.0, 0.0, cv::INTER_NEAREST);
    mData = resizedData;
}

template <class T>
void N2D2::BitmapROI<T>::rotate(int centerX, int centerY, double angle)
{
    const int scale = std::abs(mScale);

    // get the center of the bitmap
    const double cx = mOrigin.x + mData.cols * scale / 2.0;
    const double cy = mOrigin.y + mData.rows * scale / 2.0;

    // compute new, rotated center
    const double ox = centerX + (cx - centerX) * std::cos(angle)
                              - (cy - centerY) * std::sin(angle);
    const double oy = centerY + (cx - centerX) * std::sin(angle)
                              + (cy - centerY) * std::cos(angle);

    // get the rotation matrix, relative to the center of bitmap
    const cv::Point center = cv::Point(mData.cols / 2.0, mData.rows / 2.0);
    const cv::Mat rotMat
        = cv::getRotationMatrix2D(center, -Utils::radToDeg(angle), 1.0);

    // perform the affine transformation
    cv::Mat rotated;
    cv::warpAffine(mData, rotated, rotMat, mData.size(),
                    cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar(0));
    mData = rotated;

    // compute new origin
    mOrigin.x = Utils::round(ox - mData.cols * scale / 2.0);
    mOrigin.y = Utils::round(oy - mData.rows * scale / 2.0);
}

template <class T>
void N2D2::BitmapROI<T>::padCrop(int offsetX,
                                 int offsetY,
                                 unsigned int width,
                                 unsigned int height)
{
    mOrigin.x -= offsetX;
    mOrigin.y -= offsetY;

    // Cropping values for mData
    int top = 0;
    int bottom = 0;
    int left = 0;
    int right = 0;

    const int scale = std::abs(mScale);

    if (mOrigin.x < 0) {
        left = (int)std::ceil((-mOrigin.x) / (double)scale);
        mOrigin.x += left * scale;
    }

    if (mOrigin.y < 0) {
        top = (int)std::ceil((-mOrigin.y) / (double)scale);
        mOrigin.y += top * scale;
    }

    const int dw = width - (mOrigin.x + (mData.cols - left) * scale);
    const int dh = height - (mOrigin.y + (mData.rows - top) * scale);

    if (dw < 0)
        right = (int)std::ceil((-dw) / (double)scale);

    if (dh < 0)
        bottom = (int)std::ceil((-dh) / (double)scale);

    // Cropping
    if (left + right >= mData.cols || top + bottom >= mData.rows) {
        // out of area
        mData = cv::Mat::zeros(cv::Size(0, 0), mData.type());
    }
    else if (left > 0 || top > 0 || right > 0 || bottom > 0) {
        const cv::Rect crop(
            cv::Point(left, top),
            cv::Point(mData.cols - right, mData.rows - bottom));
        mData = mData(crop);
    }
}

template <class T>
void N2D2::BitmapROI
    <T>::flip(unsigned int width, unsigned int height, bool hFlip, bool vFlip)
{
    if (hFlip) {
        mOrigin.x = width - mOrigin.x;
        mOrigin.x -= mData.cols * std::abs(mScale);
    }

    if (vFlip) {
        mOrigin.y = height - mOrigin.y;
        mOrigin.y -= mData.rows * std::abs(mScale);
    }

    const int flipCode = (hFlip && vFlip) ? -1
                       : (hFlip) ? 1
                       : (vFlip) ? 0
                       : 2;

    if (flipCode != 2) {
        cv::Mat matFlip;
        cv::flip(mData, matFlip, flipCode);
        mData = matFlip;
    }
}

#endif // N2D2_BITMAPROI_H
