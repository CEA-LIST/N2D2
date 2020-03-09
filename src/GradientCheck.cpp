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

#include <iomanip>
#include <set>

#include "GradientCheck.hpp"
#include "containers/Tensor.hpp"
#include "third_party/half.hpp"
#include "utils/Random.hpp"

template <class T>
N2D2::GradientCheck<T>::GradientCheck(double epsilon, double maxError)
    : mEpsilon(epsilon), mMaxError(maxError)
{
    // ctor
}

template <class T>
void N2D2::GradientCheck<T>::initialize(Interface<>& inputs,
                                        Tensor<T>& outputs,
                                        Tensor<T>& diffInputs,
                                        PropagateType propagate,
                                        BackPropagateType backPropagate,
                                        bool avoidDiscontinuity)
{
    mOutputs = &outputs;
    mDiffInputs = &diffInputs;
    mPropagate = propagate;

    // Initialize input values
    for (Interface<>::iterator itTensor = inputs.begin(),
        itTensorEnd = inputs.end(); itTensor != itTensorEnd; ++itTensor)
    {
        Tensor<T> input = tensor_cast_nocopy<T>(*(*itTensor));

        if ((*itTensor)->isValid()) {
            std::cout << "GradientCheck::initialize(): do not initialize valid"
                " input #" << (itTensor - inputs.begin()) << std::endl;
            continue;
        }

        if (avoidDiscontinuity) {
            // This special case is for MAX pooling.
            // Each value must be at least one mEpsilon appart, to avoid
            // changing the MAX during numerical gradient computation.
            std::set<T> values;

            if ((*itTensor)->size() > (unsigned int)(1.0 / mEpsilon)) {
                throw std::runtime_error("GradientCheck::initialize():"
                    " avoidDiscontinuity not possible");
            }

            for (unsigned int index = 0; index < (*itTensor)->size(); ++index) {
                T value;

                do {
                    value = ((int)Random::randUniform(-1.0 / mEpsilon,
                                                      1.0 / mEpsilon))
                                                            * mEpsilon;
                }
                while (values.find(value) != values.end());

                input(index) = value;
                values.insert(value);
            }
        }
        else {
            for (unsigned int index = 0; index < (*itTensor)->size(); ++index)
                input(index) = Random::randUniform(-1.0, 1.0);
        }

        *(*itTensor) = input;
        (*itTensor)->synchronizeHToD();
    }

    propagate(false);
    outputs.synchronizeDToH();

#pragma omp parallel for if (outputs.size() > 32)
    for (int index = 0; index < (int)outputs.size(); ++index)
        diffInputs(index) = 1.0 - outputs(index);

    diffInputs.setValid();
    diffInputs.synchronizeHToD();
    backPropagate();
}

template <class T>
template <class U>
void N2D2::GradientCheck<T>::check(const std::string& tensorName,
                                   Tensor<U>& tensor,
                                   Tensor<U>& diffTensor)
{
    double cumulativeError = 0.0;
    unsigned int nbGradients = 0;

    tensor.synchronizeDToH();
    diffTensor.synchronizeDToH();

    for (unsigned int b = 0; b < tensor.dimB(); ++b) {
        for (unsigned int z = 0; z < tensor.dimZ(); ++z) {
            for (unsigned int y = 0; y < tensor.dimY(); ++y) {
                for (unsigned int x = 0; x < tensor.dimX(); ++x) {
                    const U value = tensor(x, y, z, b);

                    // Compute approx. gradient
                    tensor(x, y, z, b) = value + mEpsilon / 2.0;
                    tensor.synchronizeHToD(x, y, z, b, 1);
                    mPropagate(false);

                    double approxGradient = cost();

                    tensor(x, y, z, b) = value - mEpsilon / 2.0;
                    tensor.synchronizeHToD(x, y, z, b, 1);
                    mPropagate(false);

                    approxGradient = (approxGradient - cost()) / mEpsilon;

                    // Computed gradient
                    tensor(x, y, z, b) = value;
                    tensor.synchronizeHToD(x, y, z, b, 1);

                    const U gradient = -diffTensor(x, y, z, b);
                    const double error = std::fabs(gradient - approxGradient);

                    cumulativeError += error;
                    ++nbGradients;

                    if (error >= mMaxError || std::isnan(error)) {
                        std::cout << "Gradient check error for \"" << tensorName
                                  << "\""
                                  << " @ (" << x << ", " << y << ", " << z
                                  << ", " << b << ")\n"
                                  << std::setprecision(std::numeric_limits
                                                       <double>::digits10 + 1)
                                  << "  Computed = " << gradient
                                  << "\n"
                                     "  Approximated = " << approxGradient
                                  << "\n"
                                     "  Error = " << error
                                  << " > max. error = " << mMaxError
                                  << std::endl;

                        throw std::runtime_error("Gradient check failed!");
                    }
                }
            }
        }
    }

    std::cout << "Gradient check for \"" << tensorName
              << "\" PASSED! (avg error = "
              << (cumulativeError / ((nbGradients == 0) ? 1.0 : nbGradients))
              << ")" << std::endl;
}

template <class T>
void N2D2::GradientCheck<T>::check(const std::string& tensorName,
                                   BaseTensor& tensor,
                                   BaseTensor& diffTensor)
{
    assert(tensor.getType() == diffTensor.getType());

    if (tensor.getType() == &typeid(float)) {
        check(tensorName,
              dynamic_cast<Tensor<float>&>(tensor),
              dynamic_cast<Tensor<float>&>(diffTensor));
    }
    else if (tensor.getType() == &typeid(half_float::half)) {
        check(tensorName,
              dynamic_cast<Tensor<half_float::half>&>(tensor),
              dynamic_cast<Tensor<half_float::half>&>(diffTensor));
    }
    else if (tensor.getType() == &typeid(double)) {
        check(tensorName,
              dynamic_cast<Tensor<double>&>(tensor),
              dynamic_cast<Tensor<double>&>(diffTensor));
    }
    else
        throw std::runtime_error("GradientCheck::check(): type not supported");
}

template <class T>
N2D2::GradientCheck<T>::~GradientCheck()
{
    mDiffInputs->fill(T(0.0));
}

template <class T>
double N2D2::GradientCheck<T>::cost() const
{
    mOutputs->synchronizeDToH();
    const int outputSize = mOutputs->size();

    double cost = 0.0;

#pragma omp parallel for reduction(+ : cost)
    for (int index = 0; index < outputSize; ++index) {
        const double error = 1.0 - (*mOutputs)(index);
        cost += error * error;
    }

    return (1.0 / 2.0) * cost;
}

namespace N2D2 {
    template class GradientCheck<half_float::half>;
    template class GradientCheck<float>;
    template class GradientCheck<double>;
}
