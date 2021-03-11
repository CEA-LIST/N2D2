/*
    (C) Copyright 2021 CEA LIST. All Rights Reserved.
    Contributor(s): David BRIAND (david.briand@cea.fr)

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

#include "Transformation/RandomResizeCropTransformation.hpp"

const char* N2D2::RandomResizeCropTransformation::Type = "RandomResizeCrop";

N2D2::RandomResizeCropTransformation::RandomResizeCropTransformation(
    unsigned int width,
    unsigned int height,
    unsigned int offsetX,
    unsigned int offsetY)
    : mWidth(width),
      mHeight(height),
      mOffsetX(this, "OffsetX", offsetX),
      mOffsetY(this, "OffsetY", offsetY),
      mScaleMin(this, "ScaleMin", 1.0),
      mScaleMax(this, "ScaleMax", 1.0),
      mRatioMin(this, "RatioMin", 1.0),
      mRatioMax(this, "RatioMax", 1.0)
{
    // ctor
}

N2D2::RandomResizeCropTransformation::RandomResizeCropTransformation(
    const RandomResizeCropTransformation& trans)
    : mWidth(trans.mWidth),
      mHeight(trans.mHeight),
      mOffsetX(this, "OffsetX", trans.mOffsetX),
      mOffsetY(this, "OffsetY", trans.mOffsetY),
      mScaleMin(this, "ScaleMin", trans.mScaleMin),
      mScaleMax(this, "ScaleMax", trans.mScaleMax),
      mRatioMin(this, "RatioMin", trans.mRatioMin),
      mRatioMax(this, "RatioMax", trans.mRatioMax)
{
    // copy-ctor
}

void
N2D2::RandomResizeCropTransformation::apply(cv::Mat& frame,
                                           cv::Mat& labels,
                                           std::vector
                                           <std::shared_ptr<ROI> >& labelsROI,
                                           int id)
{
    if(mScaleMin > 1.0 || mScaleMin > 1.0) {
        throw std::runtime_error("RandomResizeCropTransformation::apply(): "
                                 "RandomScalingRange must be inferior to 1.0");
    }

    const int frameWidth = (int) frame.cols;
    const int frameHeight = (int) frame.rows;
    const double areaFrame = (double) frameWidth * frameHeight;
    BBox_T imageCrop = BBox_T(0,0,0,0);

    for (std::size_t i = 0; i < 10; ++i) {
        const double targetArea = areaFrame*Random::randUniform(mScaleMin, mScaleMax);
        const double aspect_ratio = std::exp(Random::randUniform(std::log(mRatioMin), std::log(mRatioMax)));
        const int width = int(std::round(std::sqrt(targetArea*aspect_ratio)));
        const int height = int(std::round(std::sqrt(targetArea/aspect_ratio)));
        if((width <= frameWidth) && (0 < width)
            && (height <= frameHeight) && (0 < height)) {
            imageCrop.x = Random::randUniform(0, frameWidth - width );
            imageCrop.y = Random::randUniform(0, frameHeight - height );
            imageCrop.w = width;
            imageCrop.h = height;
            break;

        }
    }

    if(imageCrop.w == 0 || imageCrop.h == 0) {
        const double inRatio = (double) frameWidth / frameHeight;
        const double ratioMin = std::min(mRatioMin, mRatioMax);
        const double ratioMax = std::max(mRatioMin, mRatioMax);

        if(inRatio < ratioMin) {
            imageCrop.w = frameWidth;
            imageCrop.h = int(std::round(frameWidth / ratioMin)) ;
        }
        else if (inRatio > ratioMax) {
            imageCrop.h = frameHeight;
            imageCrop.w = int(std::round(frameHeight * ratioMax)) ;
        }
        else {
            imageCrop.h = frameHeight;
            imageCrop.w = frameWidth;
        }
    }
    //std::cout << "RandomResizeCrop: {" << imageCrop.x 
    //    << ", " << imageCrop.y
    //    << ", " << imageCrop.w 
    //    << ", " << imageCrop.h << "}" 
    //    <<"(" << frameWidth << "," << frameHeight << ")"
    //    << std::endl;

    extract(imageCrop.x,
            imageCrop.y,
            imageCrop.w ,
            imageCrop.h ,
            frame,
            labels,
            labelsROI,
            id);

    cv::Mat frameResized;

    cv::resize(frame, frameResized, cv::Size(mWidth, mHeight), 0, 0, cv::INTER_LINEAR);

    frame = frameResized;
}

void
N2D2::RandomResizeCropTransformation::reverse(cv::Mat& frame,
                                             cv::Mat& labels,
                                             std::vector
                                             <std::shared_ptr<ROI> >& labelsROI,
                                             int /*id*/)
{
    /*
    if (mRandomOffsetX || mRandomOffsetY || mRandomRotation || mRandomScaling)
        throw std::runtime_error("RandomResizeCropTransformation::reverse(): "
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
    */
}

cv::Rect
N2D2::RandomResizeCropTransformation::extract(unsigned int x,
                                             unsigned int y,
                                             unsigned int width,
                                             unsigned int height,
                                             cv::Mat& frame,
                                             cv::Mat& labels,
                                             std::vector
                                             <std::shared_ptr<ROI> >& labelsROI,
                                             int /*id*/)
{
    cv::Point pTopLeft;
    cv::Point pBottomRight;
    pTopLeft.x = (int)x;
    pTopLeft.y = (int)y;
    pBottomRight.x = (int)(x + width) ;
    pBottomRight.y = (int)(y + height) ;
    cv::Rect rect(pTopLeft, pBottomRight);


    //cv::Rect rect(x, y, width, height);
    frame = frame(rect);
    //padCropLabelsROI(labelsROI, x, y, width, height);
    return rect;
}

N2D2::RandomResizeCropTransformation::~RandomResizeCropTransformation() {
    
}
