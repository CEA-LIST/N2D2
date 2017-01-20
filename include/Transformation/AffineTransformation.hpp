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

#ifndef N2D2_AFFINETRANSFORMATION_H
#define N2D2_AFFINETRANSFORMATION_H

#include "Transformation.hpp"
#include "utils/BinaryCvMat.hpp"
#include "utils/Utils.hpp"

namespace N2D2 {
class AffineTransformation : public Transformation {
public:
    using Transformation::apply;

    enum Operator {
        Plus,
        Minus,
        Multiplies,
        Divides
    };

    AffineTransformation(Operator firstOperator,
                         cv::Mat firstValue,
                         Operator secondOperator = Plus,
                         cv::Mat secondValue = cv::Mat());
    AffineTransformation(Operator firstOperator,
                         const std::string& firstValue,
                         Operator secondOperator = Plus,
                         const std::string& secondValue = "");
    void apply(cv::Mat& frame,
               cv::Mat& /*labels*/,
               std::vector<std::shared_ptr<ROI> >& /*labelsROI*/,
               int /*id*/ = -1);
    std::shared_ptr<AffineTransformation> clone() const
    {
        return std::shared_ptr<AffineTransformation>(doClone());
    }
    std::pair<unsigned int, unsigned int>
    getOutputsSize(unsigned int width, unsigned int height) const
    {
        return std::make_pair(width, height);
    };
    virtual ~AffineTransformation() {};

private:
    virtual AffineTransformation* doClone() const
    {
        return new AffineTransformation(*this);
    }
    void applyOperator(cv::Mat& frame,
                       const Operator& op,
                       const cv::Mat& frameValue) const;

    const Operator mFirstOperator;
    cv::Mat mFirstValue;
    const Operator mSecondOperator;
    cv::Mat mSecondValue;
};
}

namespace {
template <>
const char* const EnumStrings<N2D2::AffineTransformation::Operator>::data[]
    = {"Plus", "Minus", "Multiplies", "Divides"};
}

#endif // N2D2_AFFINETRANSFORMATION_H
