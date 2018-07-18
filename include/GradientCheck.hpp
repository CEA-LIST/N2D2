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

#ifndef N2D2_GRADIENT_CHECK_H
#define N2D2_GRADIENT_CHECK_H

#include "Environment.hpp" // Defines Float_T
#include "containers/Tensor.hpp"
#include "controler/Interface.hpp"

namespace N2D2 {
template <class T>
class GradientCheck {
public:
    typedef std::function<void(bool)> PropagateType;
    typedef std::function<void(void)> BackPropagateType;

    GradientCheck(double epsilon = 1.0e-4, double maxError = 1.0e-6);
    void initialize(Interface<>& inputs,
                    Tensor<T>& outputs,
                    Tensor<T>& diffInputs,
                    PropagateType propagate,
                    BackPropagateType backPropagate,
                    bool avoidDiscontinuity = false);
    template <class U>
    void check(const std::string& tensorName,
               Tensor<U>& inputs,
               Tensor<U>& diffOutputs);
    void check(const std::string& tensorName,
               BaseTensor& inputs,
               BaseTensor& diffOutputs);
    virtual ~GradientCheck();

private:
    double cost() const;

    double mEpsilon;
    double mMaxError;

    Tensor<T>* mOutputs;
    Tensor<T>* mDiffInputs;
    PropagateType mPropagate;
};
}

#endif // N2D2_GRADIENT_CHECK_H
