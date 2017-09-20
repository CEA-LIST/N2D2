/*
    (C) Copyright 2013 CEA LIST. All Rights Reserved.
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

#ifndef N2D2_DECONVCELL_FRAME_H
#define N2D2_DECONVCELL_FRAME_H

#include "Cell_Frame.hpp"
#include "ConvCell_Frame_Kernels.hpp"
#include "DeconvCell.hpp"
#include "Solver/SGDSolver_Frame.hpp"

namespace N2D2 {
class DeconvCell_Frame : public virtual DeconvCell, public Cell_Frame {
public:
    DeconvCell_Frame(const std::string& name,
                     unsigned int kernelWidth,
                     unsigned int kernelHeight,
                     unsigned int nbOutputs,
                     unsigned int strideX = 1,
                     unsigned int strideY = 1,
                     int paddingX = 0,
                     int paddingY = 0,
                     const std::shared_ptr<Activation<Float_T> >& activation
                     = std::make_shared<TanhActivation_Frame<Float_T> >());
    static std::shared_ptr<DeconvCell>
    create(Network& /*net*/,
           const std::string& name,
           unsigned int kernelWidth,
           unsigned int kernelHeight,
           unsigned int nbOutputs,
           unsigned int strideX = 1,
           unsigned int strideY = 1,
           int paddingX = 0,
           int paddingY = 0,
           const std::shared_ptr<Activation<Float_T> >& activation
           = std::make_shared<TanhActivation_Frame<Float_T> >())
    {
        return std::make_shared<DeconvCell_Frame>(name,
                                                  kernelWidth,
                                                  kernelHeight,
                                                  nbOutputs,
                                                  strideX,
                                                  strideY,
                                                  paddingX,
                                                  paddingY,
                                                  activation);
    }

    virtual void initialize();
    virtual void propagate(bool inference = false);
    virtual void backPropagate();
    virtual void update();
    inline Float_T getWeight(unsigned int output,
                             unsigned int channel,
                             unsigned int sx,
                             unsigned int sy) const
    {
        return mSharedSynapses(sx, sy, output, channel);
    };
    inline Float_T getBias(unsigned int output) const
    {
        return (*mBias)(output);
    };
    inline Interface<Float_T>* getWeights()
    {
        return &mSharedSynapses;
    };
    void setWeights(unsigned int k,
                    Interface<Float_T>* weights,
                    unsigned int offset);
    inline std::shared_ptr<Tensor4d<Float_T> > getBiases()
    {
        return mBias;
    };
    inline void setBiases(const std::shared_ptr<Tensor4d<Float_T> >& biases)
    {
        mBias = biases;
    }
    void checkGradient(double epsilon = 1.0e-4, double maxError = 1.0e-6);
    void saveFreeParameters(const std::string& fileName) const;
    void loadFreeParameters(const std::string& fileName,
                            bool ignoreNotExists = false);
    void exportSolverParameters(const std::string& fileName) const;
    virtual ~DeconvCell_Frame();

protected:
    inline void setWeight(unsigned int output,
                          unsigned int channel,
                          unsigned int sx,
                          unsigned int sy,
                          Float_T value)
    {
        mSharedSynapses(sx, sy, output, channel) = value;
    }
    inline void setBias(unsigned int output, Float_T value)
    {
        (*mBias)(output) = value;
    };

    // Internal
    std::vector<std::shared_ptr<Solver<Float_T> > > mWeightsSolvers;
    Interface<Float_T> mSharedSynapses;
    std::map<unsigned int,
        std::pair<Interface<Float_T>*, unsigned int> > mExtSharedSynapses;
    std::shared_ptr<Tensor4d<Float_T> > mBias;
    Interface<Float_T> mDiffSharedSynapses;
    Tensor4d<Float_T> mDiffBias;
    ConvCell_Frame_Kernels::Descriptor mConvDesc;

private:
    static Registrar<DeconvCell> mRegistrar;
};
}

#endif // N2D2_DECONVCELL_FRAME_H
