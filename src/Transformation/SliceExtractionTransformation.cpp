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

#include "Transformation/SliceExtractionTransformation.hpp"

const char* N2D2::SliceExtractionTransformation::Type = "SliceExtraction";

N2D2::SliceExtractionTransformation::SliceExtractionTransformation(
    unsigned int width,
    unsigned int height,
    unsigned int offsetX,
    unsigned int offsetY)
    : mWidth(width),
      mHeight(height),
      mOffsetX(this, "OffsetX", offsetX),
      mOffsetY(this, "OffsetY", offsetY),
      mRandomOffsetX(this, "RandomOffsetX", false),
      mRandomOffsetY(this, "RandomOffsetY", false),
      mRandomRotation(this, "RandomRotation", false),
      mRandomRotationRange(this, "RandomRotationRange",
                           std::vector<double>({0.0, 360.0})),
      mRandomScaling(this, "RandomScaling", false),
      mRandomScalingRange(this, "RandomScalingRange",
                          std::vector<double>({0.8, 1.2})),
      mAllowPadding(this, "AllowPadding", false),
      mBorderType(this, "BorderType", MinusOneReflectBorder),
      mBorderValue(this, "BorderValue", std::vector<double>())
{
    // ctor
}

N2D2::SliceExtractionTransformation::SliceExtractionTransformation(
    const SliceExtractionTransformation& trans)
    : mWidth(trans.mWidth),
      mHeight(trans.mHeight),
      mOffsetX(this, "OffsetX", trans.mOffsetX),
      mOffsetY(this, "OffsetY", trans.mOffsetY),
      mRandomOffsetX(this, "RandomOffsetX", trans.mRandomOffsetX),
      mRandomOffsetY(this, "RandomOffsetY", trans.mRandomOffsetY),
      mRandomRotation(this, "RandomRotation", trans.mRandomRotation),
      mRandomRotationRange(this, "RandomRotationRange",
                           trans.mRandomRotationRange),
      mRandomScaling(this, "RandomScaling", trans.mRandomScaling),
      mRandomScalingRange(this, "RandomScalingRange",
                           trans.mRandomScalingRange),
      mAllowPadding(this, "AllowPadding", trans.mAllowPadding),
      mBorderType(this, "BorderType", trans.mBorderType),
      mBorderValue(this, "BorderValue", trans.mBorderValue)
{
    // copy-ctor
}

void
N2D2::SliceExtractionTransformation::apply(cv::Mat& frame,
                                           cv::Mat& labels,
                                           std::vector
                                           <std::shared_ptr<ROI> >& labelsROI,
                                           int id)
{
    if (mRandomScaling && mRandomScalingRange->size() != 2) {
        throw std::runtime_error("SliceExtractionTransformation::apply(): "
                                 "RandomScalingRange must have two value "
                                 "(\"min max\")");
    }

    const double scaling = (mRandomScaling)
        ? Random::randUniform(*(mRandomScalingRange->begin()),
                              *(mRandomScalingRange->begin() + 1))
        : 1.0;

    const unsigned int targetWidth
        = (mWidth > 0) ? mWidth : frame.cols / scaling;
    const unsigned int targetHeight
        = (mHeight > 0) ? mHeight : frame.rows / scaling;
    const unsigned int width = Utils::round(targetWidth * scaling);
    const unsigned int height = Utils::round(targetHeight * scaling);

    const unsigned int frameOffsetX
        = (mRandomOffsetX) ? ((frame.cols > (int)width)
                                  ? Random::randUniform(0, frame.cols - width)
                                  : 0)
                           : mOffsetX;
    const unsigned int frameOffsetY
        = (mRandomOffsetY) ? ((frame.rows > (int)height)
                                  ? Random::randUniform(0, frame.rows - height)
                                  : 0)
                           : mOffsetY;

    const int padWidth = (int)frameOffsetX + width - frame.cols;
    const int padHeight = (int)frameOffsetY + height - frame.rows;

    if ((padWidth > 0 || padHeight > 0) && !mAllowPadding) {
        std::ostringstream msgStr;
        msgStr << "SliceExtractionTransformation::apply(): cannot extract a"
            " slice with an image size (" << frame.cols << "x" << frame.rows
            << ") smaller than the slice size (" << width << "x" << height
            << "), when padding is not allowed";

        throw std::runtime_error(msgStr.str());
    }

    if (mRandomRotation && mRandomRotationRange->size() != 2) {
        throw std::runtime_error("SliceExtractionTransformation::apply(): "
                                 "RandomRotationRange must have two value "
                                 "(\"min max\")");
    }

    const double rotation = (mRandomRotation)
        ? Random::randUniform(*(mRandomRotationRange->begin()),
                              *(mRandomRotationRange->begin() + 1))
        : 0.0;

    const int borderType = (mBorderType == MeanBorder)
                                ? cv::BORDER_CONSTANT
                                : (int)mBorderType;

    std::vector<double> bgColorValue = mBorderValue;
    bgColorValue.resize(4, 0.0);
    const cv::Scalar bgColor = (mBorderType == MeanBorder)
        ? cv::mean(frame)
        : cv::Scalar(bgColorValue[0], bgColorValue[1],
                    bgColorValue[2], bgColorValue[3]);

    extract(frameOffsetX,
            frameOffsetY,
            width,
            height,
            rotation,
            borderType,
            bgColor,
            frame,
            labels,
            labelsROI,
            id);

    if (scaling != 1.0) {
        double xRatio = targetWidth / (double)frame.cols;
        double yRatio = targetHeight / (double)frame.rows;

        cv::Mat frameResized;
        cv::resize(frame, frameResized, cv::Size(targetWidth, targetHeight), 0, 0,
                   cv::INTER_LINEAR);
        frame = frameResized;

        if (labels.rows > 1 || labels.cols > 1) {
            cv::Mat labelsResized;
            cv::resize(labels, labelsResized, cv::Size(targetWidth, targetHeight), 0, 0,
                       cv::INTER_NEAREST);
            labels = labelsResized;
        }

        std::for_each(
            labelsROI.begin(),
            labelsROI.end(),
            std::bind(&ROI::rescale, std::placeholders::_1, xRatio, yRatio));
    }
}

void
N2D2::SliceExtractionTransformation::reverse(cv::Mat& frame,
                                             cv::Mat& labels,
                                             std::vector
                                             <std::shared_ptr<ROI> >& labelsROI,
                                             int /*id*/)
{
    if (mRandomOffsetX || mRandomOffsetY || mRandomRotation || mRandomScaling)
        throw std::runtime_error("SliceExtractionTransformation::reverse(): "
                                 "cannot reverse random transformation.");

    if (!labels.empty()) {
        const unsigned int targetWidth = (mWidth > 0) ? mWidth : frame.cols;
        const unsigned int targetHeight = (mHeight > 0) ? mHeight : frame.rows;
        const int bottom = frame.cols - targetWidth - mOffsetX;
        const int right = frame.rows - targetHeight - mOffsetY;

        cv::Mat labelsBorder;
        cv::copyMakeBorder(labels,
                           labelsBorder,
                           mOffsetY,
                           std::max(0, bottom),
                           mOffsetX,
                           std::max(0, right),
                           cv::BORDER_CONSTANT,
                           cv::Scalar::all(-1));
        labels = labelsBorder;

        const cv::Rect crop(cv::Point(0, 0),
                            cv::Point(labels.cols + std::min(0, right),
                                      labels.rows + std::min(0, bottom)));
        labels = labels(crop);
    }

    padCropLabelsROI(labelsROI,
                     -(int)mOffsetX,
                     -(int)mOffsetY,
                     frame.cols,
                     frame.rows);
}

cv::Rect
N2D2::SliceExtractionTransformation::extract(unsigned int x,
                                             unsigned int y,
                                             unsigned int width,
                                             unsigned int height,
                                             double rotation,
                                             int borderType,
                                             const cv::Scalar& bgColor,
                                             cv::Mat& frame,
                                             cv::Mat& labels,
                                             std::vector
                                             <std::shared_ptr<ROI> >& labelsROI,
                                             int /*id*/)
{
    if (rotation != 0.0) {
        const cv::RotatedRect rotatedRect(cv::Point(x + width / 2.0,
                                                  y + height / 2.0),
                                          cv::Size(width, height),
                                          -rotation);
        cv::Rect rect = rotatedRect.boundingRect();

        // Debug: plot center of rotation
        //cv::circle(frame, cv::Point(x + width / 2.0,
        //                            y + height / 2.0), 2,
        //            cv::Scalar(255, 0, 0));

        const unsigned int padH = std::max((int)width - rect.width, 0);
        const unsigned int padV = std::max((int)height - rect.height, 0);

        const unsigned int padLeft = (rect.x < 0) ? -rect.x : 0;
        const unsigned int padTop = (rect.y < 0) ? -rect.y : 0;
        const unsigned int padRight = (rect.x + rect.width > frame.cols)
                                        ? rect.x + rect.width - frame.cols : 0;
        const unsigned int padBottom = (rect.y + rect.height > frame.rows)
                                        ? rect.y + rect.height - frame.rows : 0;

        if (padLeft > 0) {
            rect.x += padLeft;
            rect.width -= padLeft;
        }

        if (padTop > 0) {
            rect.y += padTop;
            rect.height -= padTop;
        }

        if (padRight > 0)
            rect.width -= padRight;

        if (padBottom > 0)
            rect.height -= padBottom;

        frame = frame(rect);

        cv::Mat frameBorder;
        cv::copyMakeBorder(frame,
                           frameBorder,
                           padTop + padV / 2,
                           padBottom + padV - padV / 2,
                           padLeft + padH / 2,
                           padRight + padH - padH / 2,
                           borderType,
                           bgColor);
        frame = frameBorder;

        // Debug: plot rectangle to extract
        //cv::Point2f vertices[4];
        //rotatedRect.center.x = frame.cols / 2.0;
        //rotatedRect.center.y = frame.rows / 2.0;
        //rotatedRect.points(vertices);

        //for (int i = 0; i < 4; i++) {
        //    cv::line(frame, vertices[i], vertices[(i+1)%4],
        //             cv::Scalar(255, 0, 0));
        //}

        // get the rotation matrix
        const cv::Point center = cv::Point(frame.cols / 2.0, frame.rows / 2.0);
        const cv::Mat rotMat = cv::getRotationMatrix2D(center, -rotation, 1.0);

        // perform the affine transformation
        cv::Mat rotated;
        cv::warpAffine(frame, rotated, rotMat, frame.size(),
                       cv::INTER_LINEAR, borderType, bgColor);

        // crop the resulting image
        const cv::Rect subRect((rotated.cols - (int)width) / 2.0,
                               (rotated.rows - (int)height) / 2.0,
                               width, height);
        frame = rotated(subRect);

        if (labels.rows > 1 || labels.cols > 1) {
            labels = labels(rect);

            cv::Mat labelsBorder;
            cv::copyMakeBorder(labels,
                               labelsBorder,
                               padTop + padV / 2,
                               padBottom + padV - padV / 2,
                               padLeft + padH / 2,
                               padRight + padH - padH / 2,
                               cv::BORDER_CONSTANT,
                               cv::Scalar::all(-1));
            labels = labelsBorder;

            cv::Mat labelsRotated;
            cv::warpAffine(labels, labelsRotated, rotMat, labels.size(),
                           cv::INTER_NEAREST, cv::BORDER_CONSTANT,
                           cv::Scalar::all(-1));

            labels = labelsRotated(subRect);
        }

        // pad crop to bounding rect
        padCropLabelsROI(labelsROI, rect.x, rect.y, rect.width, rect.height);

        // rotate
        // angle is in rad and counterclockwise, contrary to OpenCV functions
        std::for_each(
            labelsROI.begin(),
            labelsROI.end(),
            std::bind(&ROI::rotate, std::placeholders::_1,
                      rect.width / 2, rect.height / 2,
                      Utils::degToRad(rotation)));

        // pad crop to patch rect
        padCropLabelsROI(labelsROI, (rect.width - (int)width) / 2.0,
                                    (rect.height - (int)height) / 2.0,
                         width, height);

        return rect;
    }
    else {
        cv::Rect rect(x, y, width, height);

        const int padWidth = (int)x + rect.width - frame.cols;
        const int padHeight = (int)y + rect.height - frame.rows;

        if (padWidth > 0)
            rect.width = frame.cols - (int)x;
        if (padHeight > 0)
            rect.height = frame.rows - (int)y;

        frame = frame(rect);

        if (padWidth > 0 || padHeight > 0) {
            cv::Mat frameBorder;
            cv::copyMakeBorder(frame,
                               frameBorder,
                               0,
                               std::max(0, padHeight),
                               0,
                               std::max(0, padWidth),
                               borderType,
                               bgColor);
            frame = frameBorder;
        }

        if (labels.rows > 1 || labels.cols > 1) {
            labels = labels(rect);

            if (padWidth > 0 || padHeight > 0) {
                cv::Mat labelsBorder;
                cv::copyMakeBorder(labels,
                                   labelsBorder,
                                   0,
                                   std::max(0, padHeight),
                                   0,
                                   std::max(0, padWidth),
                                   cv::BORDER_CONSTANT,
                                   cv::Scalar::all(-1));
                labels = labelsBorder;
            }
        }

        padCropLabelsROI(labelsROI, x, y, width, height);

        return rect;
    }
}

N2D2::SliceExtractionTransformation::~SliceExtractionTransformation() {
    
}
