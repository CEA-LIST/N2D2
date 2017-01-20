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

#ifndef N2D2_APODIZATIONTRANSFORMATION_H
#define N2D2_APODIZATIONTRANSFORMATION_H

#include "Transformation.hpp"
#include "utils/Utils.hpp"
#include "utils/WindowFunction.hpp"

namespace N2D2 {
class ApodizationTransformation : public Transformation {
public:
    using Transformation::apply;

    ApodizationTransformation(const WindowFunction<double>& func,
                              unsigned int size);
    void apply(cv::Mat& frame,
               cv::Mat& /*labels*/,
               std::vector<std::shared_ptr<ROI> >& /*labelsROI*/,
               int /*id*/ = -1);
    std::shared_ptr<ApodizationTransformation> clone() const
    {
        return std::shared_ptr<ApodizationTransformation>(doClone());
    }
    std::pair<unsigned int, unsigned int>
    getOutputsSize(unsigned int width, unsigned int height) const
    {
        return std::make_pair(width, height);
    };
    virtual ~ApodizationTransformation() {};

private:
    virtual ApodizationTransformation* doClone() const
    {
        return new ApodizationTransformation(*this);
    }
    template <class T> void applyApodization(cv::Mat& mat) const;

    const std::vector<double> mWindow;
};
}

template <class T>
void N2D2::ApodizationTransformation::applyApodization(cv::Mat& mat) const
{
    for (int i = 0; i < mat.rows; ++i) {
        T* rowPtr = mat.ptr<T>(i);

        for (int j = 0; j < mat.cols; ++j)
            rowPtr[j] *= mWindow[j];
    }
}

#endif // N2D2_APODIZATIONTRANSFORMATION_H
