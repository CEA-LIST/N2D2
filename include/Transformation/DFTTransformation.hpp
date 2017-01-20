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

#ifndef N2D2_DFTTRANSFORMATION_H
#define N2D2_DFTTRANSFORMATION_H

#include "Transformation.hpp"
#include "utils/DSP.hpp"
#include "utils/Utils.hpp"

namespace N2D2 {
class DFTTransformation : public Transformation {
public:
    using Transformation::apply;

    DFTTransformation(bool twoDimensional = true);
    void apply(cv::Mat& frame,
               cv::Mat& /*labels*/,
               std::vector<std::shared_ptr<ROI> >& /*labelsROI*/,
               int /*id*/ = -1);
    std::shared_ptr<DFTTransformation> clone() const
    {
        return std::shared_ptr<DFTTransformation>(doClone());
    }
    virtual ~DFTTransformation() {};

private:
    virtual DFTTransformation* doClone() const
    {
        return new DFTTransformation(*this);
    }

    const bool mTwoDimensional;
};
}

#endif // N2D2_DFTTRANSFORMATION_H
