/*
    (C) Copyright 2015 CEA LIST. All Rights Reserved.
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

#ifndef N2D2_CELL_FRAME_H
#define N2D2_CELL_FRAME_H

#ifdef _OPENMP
#include <omp.h>
#endif

#include "Cell.hpp"
#include "Cell_Frame_Top.hpp"
#include "controler/Interface.hpp"

namespace N2D2 {

class DeepNet;

template <class T>
class Cell_Frame : public virtual Cell, public Cell_Frame_Top {
public:
    /**
     * Abstract frame-based cell constructor
     *
     * @param name          Name of the cell
     * @param type          Type of the cell
     * @param nbOutputs     Number of outputs maps of the cell (if 1D = number
     *of outputs)
    */
    Cell_Frame(const DeepNet& deepNet, const std::string& name,
               unsigned int nbOutputs,
               const std::shared_ptr<Activation>& activation
               = std::shared_ptr<Activation>());
    virtual void save(const std::string& fileName) const;
    virtual void load(const std::string& fileName);

    /**
     * Manage inputs, in particular the transmission input(i) = output(i-1)
     */
    virtual void addInput(StimuliProvider& sp,
                          unsigned int channel,
                          unsigned int x0,
                          unsigned int y0,
                          unsigned int width,
                          unsigned int height,
                          const Tensor<bool>& mapping = Tensor<bool>());
    virtual void addInput(StimuliProvider& sp,
                          unsigned int x0 = 0,
                          unsigned int y0 = 0,
                          unsigned int width = 0,
                          unsigned int height = 0,
                          const Tensor<bool>& mapping = Tensor<bool>());
    virtual void addInput(Cell* cell,
                          const Tensor<bool>& mapping = Tensor<bool>());
    virtual void addInput(Cell* cell,
                          unsigned int x0,
                          unsigned int y0,
                          unsigned int width = 0,
                          unsigned int height = 0);
    virtual void addInput(BaseTensor& inputs,
                          BaseTensor& diffOutputs);
    
    virtual void clearInputs();

    virtual void replaceInput(BaseTensor& oldInputs,
                              BaseTensor& newInputs,
                              BaseTensor& newDiffOutputs);
    virtual void propagate(bool inference = false);
    virtual void backPropagate();
    virtual void update();
    virtual void setOutputTarget(const Tensor<int>& targets);
    virtual double applyLoss(double targetVal,
                             double defaultVal);
    virtual void setOutputTargets(const BaseTensor& targets);
    virtual double applyLoss();
    virtual double applyLossDistribWeighted(unsigned int quantSteps,
                                            double rangeMin,
                                            double rangeMax);
    virtual double applyLossThroughKernel(const BaseTensor& kernel,
        std::function<double()> lossFunc);
    virtual void setOutputErrors(const BaseTensor& errors);
    virtual BaseTensor& getInputs(unsigned int index = 0) {
        return mInputs[index];
    }
    virtual const BaseTensor& getInputs(unsigned int index = 0) const {
        return mInputs[index];
    }
    virtual BaseTensor& getOutputs()
    {
        return mOutputs;
    }
    virtual const BaseTensor& getOutputs() const
    {
        return mOutputs;
    }
    virtual BaseTensor& getDiffInputs()
    {
        return mDiffInputs;
    }
    virtual const BaseTensor& getDiffInputs() const
    {
        return mDiffInputs;
    }
    void setDiffInputsValid()
    {
        mDiffInputs.setValid();        
    }    
    void setDiffInputs(Tensor<float>& diffInputs)
    {
        mDiffInputs = diffInputs;
    }

    virtual BaseTensor& getDiffOutputs(unsigned int index = 0) {
        return mDiffOutputs[index];
    }
    virtual const BaseTensor& getDiffOutputs(unsigned int index = 0) const {
        return mDiffOutputs[index];
    }
    virtual unsigned int getMaxOutput(unsigned int batchPos = 0) const;
    void exportActivationParameters(const std::string& dirName) const;
    void importActivationParameters(const std::string& dirName, bool ignoreNotExists);
    bool isCuda() const
    {
        return false;
    }
    virtual ~Cell_Frame() {};

protected:
    // Internal
    // Forward
    Interface<> mInputs;
    Tensor<T> mOutputs;

    // Backward
    Tensor<T> mDiffInputs;
    Interface<> mDiffOutputs;
};
}

#endif // N2D2_CELL_FRAME_H
