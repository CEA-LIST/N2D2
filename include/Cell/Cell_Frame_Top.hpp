/*
    (C) Copyright 2015 CEA LIST. All Rights Reserved.
    Contributor(s): Olivier BICHLER (olivier.bichler@cea.fr)
                    Victor GACOIN

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

#ifndef N2D2_CELL_FRAME_TOP_H
#define N2D2_CELL_FRAME_TOP_H

#include <string>
#include <vector>

#include "Activation/Activation.hpp"

namespace N2D2 {

class BaseTensor;
template<typename T>
class Tensor;

class Cell_Frame_Top {
public:
    enum Signals {
        In = 1,
        Out = 2,
        InOut = 3   // = In | Out
    };

    Cell_Frame_Top(const std::shared_ptr<Activation>& activation
                   = std::shared_ptr<Activation>())
        : mActivation(activation)
    {
    }
    virtual void save(const std::string& dirName) const {
        if (mActivation)
            mActivation->save(dirName + "/Activation");
    }
    virtual void load(const std::string& dirName) {
        if (mActivation)
            mActivation->load(dirName + "/Activation");
    }
    virtual void addInput(BaseTensor& inputs,
                          BaseTensor& diffOutputs) = 0;
    virtual void propagate(bool inference = false) = 0;
    virtual void backPropagate() = 0;
    virtual void update() = 0;
    virtual void checkGradient(double /*epsilon*/, double /*maxError*/) = 0;
    virtual void discretizeSignals(unsigned int /*nbLevels*/,
                                   const Signals& /*signals*/ = In) = 0;
    virtual void setOutputTarget(const Tensor<int>& targets,
                                 double targetVal = 1.0,
                                 double defaultVal = 0.0) = 0;
    virtual void setOutputTargets(const Tensor<int>& targets,
                                  double targetVal = 1.0,
                                  double defaultVal = 0.0) = 0;
    virtual void setOutputTargets(const BaseTensor& targets) = 0;
    virtual void setOutputErrors(const BaseTensor& errors) = 0;
    virtual BaseTensor& getOutputs() = 0;
    virtual const BaseTensor& getOutputs() const = 0;
    virtual BaseTensor& getDiffInputs() = 0;
    virtual const BaseTensor& getDiffInputs() const = 0;
    virtual unsigned int getMaxOutput(unsigned int batchPos = 0) const = 0;
    const std::shared_ptr<Activation>& getActivation() const
    {
        return mActivation;
    };
    void setActivation(const std::shared_ptr<Activation>& activation)
    {
        mActivation = activation;
    };
    virtual bool isCuda() const = 0;
    virtual ~Cell_Frame_Top() {};

protected:
    std::shared_ptr<Activation> mActivation;
};
}

#endif // N2D2_CELL_FRAME_TOP_H
