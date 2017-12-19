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

#include "Activation/LogisticActivation_Frame.hpp"
#include "Activation/RectifierActivation_Frame.hpp"
#include "Activation/SaturationActivation_Frame.hpp"
#include "Activation/SoftplusActivation_Frame.hpp"
#include "Activation/TanhActivation_Frame.hpp"
#include "Cell.hpp"
#include "Cell_Frame_Top.hpp"
#include "Environment.hpp" // Defines Float_T
#include "GradientCheck.hpp"
#include "controler/Interface.hpp"

namespace N2D2 {
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
    Cell_Frame(const std::string& name,
               unsigned int nbOutputs,
               const std::shared_ptr<Activation<Float_T> >& activation
               = std::shared_ptr<Activation<Float_T> >());
    virtual unsigned int getNbChannels() const
    {
        return mNbChannels;
    };
    virtual bool isConnection(unsigned int channel, unsigned int output) const
    {
        return mMaps(output, channel);
    };

    /**
     * Manage inputs, in particular the transmission input(i) = output(i-1)
     */
    virtual void addInput(StimuliProvider& sp,
                          unsigned int channel,
                          unsigned int x0,
                          unsigned int y0,
                          unsigned int width,
                          unsigned int height,
                          const std::vector<bool>& mapping = std::vector
                          <bool>());
    virtual void addInput(StimuliProvider& sp,
                          unsigned int x0 = 0,
                          unsigned int y0 = 0,
                          unsigned int width = 0,
                          unsigned int height = 0,
                          const Matrix<bool>& mapping = Matrix<bool>());
    virtual void addInput(Cell* cell,
                          const Matrix<bool>& mapping = Matrix<bool>());
    virtual void addInput(Cell* cell,
                          unsigned int x0,
                          unsigned int y0,
                          unsigned int width = 0,
                          unsigned int height = 0);
    virtual void addInput(Tensor4d<Float_T>& inputs,
                          Tensor4d<Float_T>& diffOutputs);
    virtual void propagate(bool inference = false);
    virtual void backPropagate();
    virtual void setOutputTarget(const Tensor4d<int>& targets,
                                 double targetVal = 1.0,
                                 double defaultVal = 0.0);
    virtual void setOutputTargets(const Tensor4d<int>& targets,
                                  double targetVal = 1.0,
                                  double defaultVal = 0.0);
    virtual void setOutputTargets(const Tensor4d<Float_T>& targets);
    virtual void setOutputErrors(const Tensor4d<Float_T>& errors);
    virtual Tensor4d<Float_T>& getOutputs()
    {
        return mOutputs;
    }
    virtual const Tensor4d<Float_T>& getOutputs() const
    {
        return mOutputs;
    }
    virtual Tensor4d<Float_T>& getDiffInputs()
    {
        return mDiffInputs;
    }
    virtual const Tensor4d<Float_T>& getDiffInputs() const
    {
        return mDiffInputs;
    }
    virtual unsigned int getMaxOutput(unsigned int batchPos = 0) const;
    void discretizeSignals(unsigned int nbLevels, const Signals& signals = In);
    virtual ~Cell_Frame() {};

protected:
    // Internal
    // Forward
    Interface<Float_T> mInputs;
    Tensor4d<Float_T> mOutputs;

    // Backward
    Tensor4d<Float_T> mDiffInputs;
    Interface<Float_T> mDiffOutputs;
};
}

#endif // N2D2_CELL_FRAME_H
