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

#include "Transformation/TrimTransformation.hpp"

N2D2::TrimTransformation::TrimTransformation(unsigned int nbLevels,
                                             const cv::Mat& kernel)
    : mNbLevels(nbLevels), mKernel(kernel), mMethod(this, "Method", Discretize)
{
    // ctor
}

void N2D2::TrimTransformation::apply(cv::Mat& frame,
                                     cv::Mat& labels,
                                     std::vector
                                     <std::shared_ptr<ROI> >& labelsROI,
                                     int /*id*/)
{
    cv::Mat frameRed;

    if (frame.channels() > 1)
        cv::cvtColor(frame, frameRed, CV_BGR2GRAY);
    else
        frameRed = frame.clone();

    if (mNbLevels > 0) {
        if (mMethod == Discretize)
            Utils::colorDiscretize(frameRed, mNbLevels);
        else
            Utils::colorReduce(frameRed, mNbLevels);
    }
    /*
        cv::namedWindow("Reduced color frame", CV_WINDOW_NORMAL);
        cv::imshow("Reduced color frame", frameRed);
        cv::waitKey(0);
    */
    // Find the most occuring border value
    std::map<unsigned char, unsigned int> borderValFreq;

    for (unsigned int i = 0; i < 255; ++i)
        borderValFreq.insert(std::make_pair(i, 0));

    for (int i = 0; i < frameRed.rows; ++i) {
        ++borderValFreq[frameRed.at<unsigned char>(i, 0)];
        ++borderValFreq[frameRed.at<unsigned char>(i, frameRed.cols - 1)];
    }

    for (int j = 1; j < frameRed.cols - 1; ++j) {
        ++borderValFreq[frameRed.at<unsigned char>(0, j)];
        ++borderValFreq[frameRed.at<unsigned char>(frameRed.rows - 1, j)];
    }

    const std::pair<unsigned char, unsigned int> borderVal =
        *std::max_element(borderValFreq.begin(),
                          borderValFreq.end(),
                          Utils::PairSecondPred<unsigned char, unsigned int>());

    if (!mKernel.empty()) {
        if (borderVal.first > 127)
            cv::dilate(frameRed, frameRed, mKernel);
        else
            cv::erode(frameRed, frameRed, mKernel);
    }

    cv::Rect frameBorder(0, 0, frameRed.cols, frameRed.rows);

    bool trimLeft = true;

    while (trimLeft) {
        for (int y = 0; y < frameRed.rows; ++y) {
            if (frameRed.at<unsigned char>(y, frameBorder.x)
                != borderVal.first) {
                trimLeft = false;
                break;
            }
        }

        if (trimLeft)
            ++frameBorder.x;
    }

    bool trimRight = true;

    while (trimRight) {
        for (int y = 0; y < frameRed.rows; ++y) {
            if (frameRed.at<unsigned char>(y, frameBorder.width - 1)
                != borderVal.first) {
                trimRight = false;
                break;
            }
        }

        if (trimRight)
            --frameBorder.width;
    }

    frameBorder.width -= frameBorder.x;

    bool trimTop = true;

    while (trimTop) {
        for (int x = 0; x < frameRed.cols; ++x) {
            if (frameRed.at<unsigned char>(frameBorder.y, x)
                != borderVal.first) {
                trimTop = false;
                break;
            }
        }

        if (trimTop)
            ++frameBorder.y;
    }

    bool trimBottom = true;

    while (trimBottom) {
        for (int x = 0; x < frameRed.cols; ++x) {
            if (frameRed.at<unsigned char>(frameBorder.height - 1, x)
                != borderVal.first) {
                trimBottom = false;
                break;
            }
        }

        if (trimBottom)
            --frameBorder.height;
    }

    frameBorder.height -= frameBorder.y;

    frame = frame(frameBorder);

    if (labels.rows > 1 || labels.cols > 1)
        labels = labels(frameBorder);

    padCropLabelsROI(labelsROI,
                     frameBorder.x,
                     frameBorder.y,
                     frameBorder.width,
                     frameBorder.height);
}
