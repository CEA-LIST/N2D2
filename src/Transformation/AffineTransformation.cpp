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
      mSecondValue(secondValue),
      mDivByZeroWarn(0)
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
#if CV_MAJOR_VERSION >= 3
        mFirstValue = cv::imread(firstValue, cv::IMREAD_UNCHANGED);
#else
        mFirstValue = cv::imread(firstValue, CV_LOAD_IMAGE_UNCHANGED);
#endif

        if (!mFirstValue.data)
            throw std::runtime_error(
                "AffineTransformation: unable to read image: " + firstValue);
    }

    if (!secondValue.empty()) {
        if (Utils::fileExtension(secondValue) == "bin")
            BinaryCvMat::read(secondValue, mSecondValue);
        else {
#if CV_MAJOR_VERSION >= 3
            mSecondValue = cv::imread(secondValue, cv::IMREAD_UNCHANGED);
#else
            mSecondValue = cv::imread(secondValue, CV_LOAD_IMAGE_UNCHANGED);
#endif

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
        const int nonZero = cv::countNonZero(valueMat);
        assert(nonZero <= (int)valueMat.total());

        if (nonZero < (int)valueMat.total()) {
            // Divide by 0 will occur...
            const int warnLimit = 5;

            if (mDivByZeroWarn < warnLimit) {
                std::cout << Utils::cwarning << "Warning:"
                    " AffineTransformation: divide by 0 will occur (found "
                    << (valueMat.total() - nonZero) << " 0s in the denominator)."
                    " Values divided by 0 will be set to 0.0."
                    << Utils::cdef << std::endl;

                ++mDivByZeroWarn;

                if (mDivByZeroWarn == warnLimit) {
                    std::cout << Utils::cwarning
                        << "Future divide by 0 warning will be ignored!"
                        << Utils::cdef << std::endl;
                }
            }

#if !defined(WIN32) && !defined(__APPLE__) && !defined(__CYGWIN__) && !defined(_WIN32)
            const int excepts = fegetexcept();
            fedisableexcept(FE_INVALID | FE_DIVBYZERO);
#endif

            // Divide by 0 = 0 in OpenCV 3.x, but is IEEE conformant in 4.x
            cv::divide(mat, valueMat, mat);

            // Converts NaN's to 0.0 (0/0 => FE_INVALID)
            cv::patchNaNs(mat, 0.0);

            // Converts inf values to 0.0 (0/x => FE_DIVBYZERO)
            if (mat.depth() == CV_32F) {
                mat.setTo(0.0f, mat == std::numeric_limits<float>::infinity());
                mat.setTo(0.0f, mat == -std::numeric_limits<float>::infinity());
            }
            else if (mat.depth() == CV_64F) {
                mat.setTo(0.0, mat == std::numeric_limits<double>::infinity());
                mat.setTo(0.0, mat == -std::numeric_limits<double>::infinity());
            }

#if !defined(WIN32) && !defined(__APPLE__) && !defined(__CYGWIN__) && !defined(_WIN32)
            feenableexcept(excepts);
#endif
        }
        else {
            cv::divide(mat, valueMat, mat);
        }

        break;
    }

    frame = mat;
}
