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

#ifndef N2D2_RANDOMAFFINETRANSFORMATION_H
#define N2D2_RANDOMAFFINETRANSFORMATION_H

#include "Transformation/Transformation.hpp"

namespace N2D2 {
class RandomAffineTransformation : public Transformation {
public:
    using Transformation::apply;

    static const char* Type;

    RandomAffineTransformation(double gainVar, double biasVar = 0.0);
    RandomAffineTransformation(
        const std::vector<std::pair<double, double> >& gainRange,
        const std::vector<std::pair<double, double> >& biasRange
            = std::vector<std::pair<double, double> >(),
        const std::vector<std::pair<double, double> >& gammaRange
            = std::vector<std::pair<double, double> >(),
        const std::vector<double>& gainVarProb = std::vector<double>(),
        const std::vector<double>& biasVarProb = std::vector<double>(),
        const std::vector<double>& gammaVarProb = std::vector<double>());
    RandomAffineTransformation(const RandomAffineTransformation& trans);
    const char* getType() const
    {
        return Type;
    };
    void apply(cv::Mat& frame,
               cv::Mat& /*labels*/,
               std::vector<std::shared_ptr<ROI> >& /*labelsROI*/,
               int /*id*/ = -1);
    std::shared_ptr<RandomAffineTransformation> clone() const
    {
        return std::shared_ptr<RandomAffineTransformation>(doClone());
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
    virtual ~RandomAffineTransformation();

private:
    virtual RandomAffineTransformation* doClone() const
    {
        return new RandomAffineTransformation(*this);
    }
    template <class T>
    void applyRandomAffine(cv::Mat& mat, double gain, double bias, double gamma)
        const;

    std::vector<std::pair<double, double> > mGainRange;
    std::vector<std::pair<double, double> > mBiasRange;
    std::vector<std::pair<double, double> > mGammaRange;
    std::vector<double> mGainVarProb;
    std::vector<double> mBiasVarProb;
    std::vector<double> mGammaVarProb;

    Parameter<bool> mDisjointGamma;
    Parameter<std::vector<bool> > mChannelsMask;
};
}

template <class T>
void N2D2::RandomAffineTransformation::applyRandomAffine(cv::Mat& mat,
                                                         double gain,
                                                         double bias,
                                                         double gamma) const
{
    const double range = (std::numeric_limits<T>::is_integer)
                           ? std::numeric_limits<T>::max()
                           : 1.0;

    for (int i = 0; i < mat.rows; ++i) {
        T* rowPtr = mat.ptr<T>(i);

        for (int j = 0; j < mat.cols; ++j) {
            if (gamma != 1.0) {
                rowPtr[j] = cv::saturate_cast<T>(gain
                    * (std::pow(rowPtr[j] / range, gamma) * range)
                                                 + bias * range);
            }
            else {
                rowPtr[j] = cv::saturate_cast<T>(gain * rowPtr[j]
                                                 + bias * range);
            }
        }
    }
}

#endif // N2D2_RANDOMAFFINETRANSFORMATION_H
