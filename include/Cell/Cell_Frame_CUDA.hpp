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

#ifndef N2D2_CELL_FRAME_CUDA_H
#define N2D2_CELL_FRAME_CUDA_H

#include "Activation/LogisticActivation_Frame_CUDA.hpp"
#include "Activation/RectifierActivation_Frame_CUDA.hpp"
#include "Activation/SaturationActivation_Frame_CUDA.hpp"
#include "Activation/SoftplusActivation_Frame_CUDA.hpp"
#include "Activation/TanhActivation_Frame_CUDA.hpp"
#include "Cell.hpp"
#include "Cell_Frame_Top.hpp"
#include "CudaContext.hpp"
#include "CudaUtils.hpp"
#include "Environment.hpp" // Defines Float_T
#include "GradientCheck.hpp"
#include "containers/CudaTensor4d.hpp"
#include "controler/CudaInterface.hpp"

namespace N2D2 {
class Cell_Frame_CUDA : public virtual Cell, public Cell_Frame_Top {
public:
    /**
     * Abstract frame-based cell constructor
     *
     * @param name          Name of the cell
     * @param type          Type of the cell
     * @param nbOutputs     Number of outputs maps of the cell (if 1D = number
     *of outputs)
    */
    Cell_Frame_CUDA(const std::string& name,
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
    virtual CudaTensor4d<Float_T>& getOutputs();
    virtual const CudaTensor4d<Float_T>& getOutputs() const;
    virtual CudaTensor4d<Float_T>& getDiffInputs();
    virtual const CudaTensor4d<Float_T>& getDiffInputs() const;
    virtual unsigned int getMaxOutput(unsigned int batchPos = 0) const;
    void discretizeSignals(unsigned int nbLevels, const Signals& signals = In);
    virtual ~Cell_Frame_CUDA();

protected:
    // Internal
    /*
        Structures shared by all kind of layer :

           *mInputs -------->|    |--------> mOutputs
                             |    |
                             |    |
       mDiffOutputs <--------|    |<-------- *mDiffInputs
    */

    CudaInterface<Float_T> mInputs;
    CudaTensor4d<Float_T> mOutputs;

    CudaTensor4d<Float_T> mDiffInputs;
    CudaInterface<Float_T> mDiffOutputs;

#if CUDNN_VERSION >= 5000
    cudnnActivationDescriptor_t mActivationDesc;
#else
    cudnnActivationMode_t mActivationDesc;
#endif
};
}

#endif // N2D2_CELL_FRAME_CUDA_H
