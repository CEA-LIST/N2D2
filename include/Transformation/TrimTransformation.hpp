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

#ifndef N2D2_TRIMTRANSFORMATION_H
#define N2D2_TRIMTRANSFORMATION_H

#include "Transformation.hpp"
#include "utils/Utils.hpp"

namespace N2D2 {
class TrimTransformation : public Transformation {
public:
    using Transformation::apply;

    enum Method {
        Discretize,
        Reduce
    };

    static const char* Type;

    TrimTransformation(unsigned int nbLevels,
                       const cv::Mat& kernel
                       = cv::getStructuringElement(cv::MORPH_RECT,
                                                   cv::Size(3, 3)));
    TrimTransformation(const TrimTransformation& trans);
    const char* getType() const
    {
        return Type;
    };
    void apply(cv::Mat& frame,
               cv::Mat& labels,
               std::vector<std::shared_ptr<ROI> >& labelsROI,
               int /*id*/ = -1);
    std::shared_ptr<TrimTransformation> clone() const
    {
        return std::shared_ptr<TrimTransformation>(doClone());
    }
    int getOutputsDepth(int depth) const
    {
        return depth;
    };
    virtual ~TrimTransformation() {};

private:
    virtual TrimTransformation* doClone() const
    {
        return new TrimTransformation(*this);
    }

    const unsigned int mNbLevels;
    const cv::Mat mKernel;

    Parameter<Method> mMethod;
};
}

namespace {
template <>
const char* const EnumStrings<N2D2::TrimTransformation::Method>::data[]
    = {"Discretize", "Reduce"};
}

#endif // N2D2_TRIMTRANSFORMATION_H
