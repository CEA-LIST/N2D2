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

#ifndef N2D2_EQUALIZETRANSFORMATION_H
#define N2D2_EQUALIZETRANSFORMATION_H

#include "Transformation/Transformation.hpp"

namespace N2D2 {
class EqualizeTransformation : public Transformation {
public:
    using Transformation::apply;

    enum Method {
        Standard,
        CLAHE
    };

    static const char* Type;

    EqualizeTransformation();
    EqualizeTransformation(const EqualizeTransformation& trans);
    const char* getType() const
    {
        return Type;
    };
    void apply(cv::Mat& frame,
               cv::Mat& /*labels*/,
               std::vector<std::shared_ptr<ROI> >& /*labelsROI*/,
               int /*id*/ = -1);
    std::shared_ptr<EqualizeTransformation> clone() const
    {
        return std::shared_ptr<EqualizeTransformation>(doClone());
    }
    std::pair<unsigned int, unsigned int>
    getOutputsSize(unsigned int width, unsigned int height) const
    {
        return std::make_pair(width, height);
    };
    int getOutputsDepth(int depth) const
    {
        return depth;
    };
    virtual ~EqualizeTransformation();

private:
    virtual EqualizeTransformation* doClone() const
    {
        return new EqualizeTransformation(*this);
    }

    Parameter<Method> mMethod;
    /// Threshold for contrast limiting.
    Parameter<double> mCLAHE_ClipLimit;
    /// Size of grid for histogram equalization. Input image will be divided
    /// into equally sized rectangular tiles. This
    /// parameter defines the number of tiles in row and column.
    Parameter<unsigned int> mCLAHE_GridSize;
};
}

namespace {
template <>
const char* const EnumStrings<N2D2::EqualizeTransformation::Method>::data[]
    = {"Standard", "CLAHE"};
}

#endif // N2D2_EQUALIZETRANSFORMATION_H
