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

#include "Transformation/AffineTransformation.hpp"

N2D2::AffineTransformation::AffineTransformation(Operator firstOperator,
                                                 cv::Mat firstValue,
                                                 Operator secondOperator,
                                                 cv::Mat secondValue)
    : mFirstOperator(firstOperator),
      mFirstValue(firstValue),
      mSecondOperator(secondOperator),
      mSecondValue(secondValue)
{
    // ctor
}

N2D2::AffineTransformation::AffineTransformation(Operator firstOperator,
                                                 const std::string& firstValue,
                                                 Operator secondOperator,
                                                 const std::string& secondValue)
    : mFirstOperator(firstOperator), mSecondOperator(secondOperator)
{
    // ctor
    if (Utils::fileExtension(firstValue) == "bin")
        BinaryCvMat::read(firstValue, mFirstValue);
    else {
        mFirstValue = cv::imread(firstValue, CV_LOAD_IMAGE_UNCHANGED);

        if (!mFirstValue.data)
            throw std::runtime_error(
                "AffineTransformation: unable to read image: " + firstValue);
    }

    if (!secondValue.empty()) {
        if (Utils::fileExtension(secondValue) == "bin")
            BinaryCvMat::read(secondValue, mSecondValue);
        else {
            mSecondValue = cv::imread(secondValue, CV_LOAD_IMAGE_UNCHANGED);

            if (!mSecondValue.data)
                throw std::runtime_error(
                    "AffineTransformation: unable to read image: "
                    + secondValue);
        }
    }
}

void N2D2::AffineTransformation::apply(cv::Mat& frame,
                                       cv::Mat& /*labels*/,
                                       std::vector
                                       <std::shared_ptr<ROI> >& /*labelsROI*/,
                                       int /*id*/)
{
    applyOperator(frame, mFirstOperator, mFirstValue);

    if (mSecondValue.data)
        applyOperator(frame, mSecondOperator, mSecondValue);
}

void N2D2::AffineTransformation::applyOperator(cv::Mat& frame,
                                               const Operator& op,
                                               const cv::Mat& valueFrame) const
{
    cv::Mat mat = frame;
    cv::Mat valueMat = valueFrame;

    if (mat.cols != valueMat.cols || mat.rows != valueMat.rows) {
        std::stringstream msg;
        msg << "AffineTransformation: value matrix for operator " << op
            << " has size " << valueMat.cols << "x" << valueMat.rows
            << ", whereas data size is " << mat.cols << "x" << mat.rows;

        throw std::runtime_error(msg.str());
    }

    if (frame.type() != valueFrame.type()) {
        if (frame.depth() == CV_32F || frame.depth() == CV_64F)
            valueFrame.convertTo(valueMat, frame.type());
        else if (valueFrame.depth() == CV_32F || valueFrame.depth() == CV_64F)
            frame.convertTo(mat, valueFrame.type());
    }

    switch (op) {
    case Plus:
        mat += valueMat;
        break;
    case Minus:
        mat -= valueMat;
        break;
    case Multiplies:
        mat = mat.mul(valueMat);
        break;
    case Divides:
        cv::divide(mat, valueMat, mat);
        break;
    }

    frame = mat;
}
