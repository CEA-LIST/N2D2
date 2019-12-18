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

#ifndef N2D2_DCTTRANSFORMATION_H
#define N2D2_DCTTRANSFORMATION_H

#include "Transformation.hpp"
#include "utils/DSP.hpp"

namespace N2D2 {
class DCTTransformation : public Transformation {
public:
    using Transformation::apply;

    static const char* Type;

    DCTTransformation(bool twoDimensional = true);
    const char* getType() const
    {
        return Type;
    };
    void apply(cv::Mat& frame,
               cv::Mat& /*labels*/,
               std::vector<std::shared_ptr<ROI> >& /*labelsROI*/,
               int /*id*/ = -1);
    std::shared_ptr<DCTTransformation> clone() const
    {
        return std::shared_ptr<DCTTransformation>(doClone());
    }
    int getOutputsDepth(int depth) const
    {
        return depth;
    };
    virtual ~DCTTransformation() {};

private:
    virtual DCTTransformation* doClone() const
    {
        return new DCTTransformation(*this);
    }
    size_t getOptimalDCTSize(size_t N) const;

    const bool mTwoDimensional;
};
}

#endif // N2D2_DCTTRANSFORMATION_H
