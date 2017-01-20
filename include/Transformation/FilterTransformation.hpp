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

/**
 * @file      FilterTransformation.h
 * @author    Olivier BICHLER (olivier.bichler@cea.fr)
 * @brief     Define filter object and common-specific ones.
 *
 * @details   These classes build filter from Kernel and apply it as a
 *convolution.
*/

#ifndef N2D2_FILTERTRANSFORMATION_H
#define N2D2_FILTERTRANSFORMATION_H

#include "Transformation.hpp"
#include "utils/Kernel.hpp"

namespace N2D2 {
/**
 * @class   FilterTransformation
 * @brief   This class allow to convolve an OpenCV matrix with a previously
 *built Kernel.
 *
*/
class FilterTransformation : public Transformation {
public:
    using Transformation::apply;

    FilterTransformation(const Kernel<double>& kernel,
                         double orientation = 0.0);
    void apply(cv::Mat& frame,
               cv::Mat& /*labels*/,
               std::vector<std::shared_ptr<ROI> >& /*labelsROI*/,
               int /*id*/ = -1);
    std::shared_ptr<FilterTransformation> clone() const
    {
        return std::shared_ptr<FilterTransformation>(doClone());
    }
    std::pair<unsigned int, unsigned int>
    getOutputsSize(unsigned int width, unsigned int height) const
    {
        return std::make_pair(width, height);
    };

    const Kernel<double>& getKernel() const
    {
        return mKernel;
    }

    /**
     * Get the orientation of the filter
     *
     * @return the orientation of the filter
    */
    double getOrientation() const
    {
        return mOrientation;
    }
    virtual ~FilterTransformation() {};

    friend FilterTransformation operator-(const FilterTransformation& filter);

private:
    virtual FilterTransformation* doClone() const
    {
        return new FilterTransformation(*this);
    }

    const Kernel<double> mKernel;
    const double mOrientation;
};

// DEPRECATED: legacy special pre-defined filters
extern const FilterTransformation FilterTransformationLaplacian;
extern const FilterTransformation FilterTransformationAerPositive;
extern const FilterTransformation FilterTransformationAerNegative;
}

#endif // N2D2_FILTERTRANSFORMATION_H
