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

#ifndef N2D2_RANGEAFFINETRANSFORMATION_H
#define N2D2_RANGEAFFINETRANSFORMATION_H

#include "Transformation.hpp"

namespace N2D2 {
class RangeAffineTransformation : public Transformation {
public:
    using Transformation::apply;

    enum Operator {
        Plus,
        Minus,
        Multiplies,
        Divides
    };

    RangeAffineTransformation(Operator firstOperator,
                              const std::vector<double>& firstValue,
                              Operator secondOperator = Plus,
                              const std::vector<double>& secondValue
                                = std::vector<double>());
    RangeAffineTransformation(Operator firstOperator,
                              double firstValue,
                              Operator secondOperator = Plus,
                              double secondValue = 0.0);
    void apply(cv::Mat& frame,
               cv::Mat& /*labels*/,
               std::vector<std::shared_ptr<ROI> >& /*labelsROI*/,
               int /*id*/ = -1);
    std::shared_ptr<RangeAffineTransformation> clone() const
    {
        return std::shared_ptr<RangeAffineTransformation>(doClone());
    }
    std::pair<unsigned int, unsigned int>
    getOutputsSize(unsigned int width, unsigned int height) const
    {
        return std::make_pair(width, height);
    };
    virtual ~RangeAffineTransformation() {};

private:
    virtual RangeAffineTransformation* doClone() const
    {
        return new RangeAffineTransformation(*this);
    }
    void applyOperator(cv::Mat& mat,
                       const Operator& op,
                       double value) const;

    const Operator mFirstOperator;
    const std::vector<double> mFirstValue;
    const Operator mSecondOperator;
    const std::vector<double> mSecondValue;
};
}

namespace {
template <>
const char* const EnumStrings<N2D2::RangeAffineTransformation::Operator>::data[]
    = {"Plus", "Minus", "Multiplies", "Divides"};
}

#endif // N2D2_RANGEAFFINETRANSFORMATION_H
