/*
    (C) Copyright 2014 CEA LIST. All Rights Reserved.
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

#ifndef N2D2_FCCELL_FRAME_H
#define N2D2_FCCELL_FRAME_H

#include "Cell_Frame.hpp"
#include "FcCell.hpp"
#include "Solver/SGDSolver_Frame.hpp"

namespace N2D2 {
class FcCell_Frame : public virtual FcCell, public Cell_Frame {
public:
    FcCell_Frame(const std::string& name,
                 unsigned int nbOutputs,
                 const std::shared_ptr<Activation<Float_T> >& activation
                 = std::make_shared<TanhActivation_Frame<Float_T> >());
    static std::shared_ptr<FcCell> create(Network& /*net*/,
                                          const std::string& name,
                                          unsigned int nbOutputs,
                                          const std::shared_ptr
                                          <Activation<Float_T> >& activation
                                          = std::make_shared
                                          <TanhActivation_Frame<Float_T> >())
    {
        return std::make_shared<FcCell_Frame>(name, nbOutputs, activation);
    }

    virtual void initialize();
    virtual void propagate(bool inference = false);
    virtual void backPropagate();
    virtual void update();
    inline Float_T getWeight(unsigned int output, unsigned int channel) const
    {
        return mSynapses(0, 0, channel, output);
    };
    inline Float_T getBias(unsigned int output) const
    {
        return mBias(output);
    };
    void checkGradient(double epsilon = 1.0e-4, double maxError = 1.0e-6);
    void saveFreeParameters(const std::string& fileName) const;
    void loadFreeParameters(const std::string& fileName,
                            bool ignoreNotExists = false);
    virtual ~FcCell_Frame();

protected:
    inline void
    setWeight(unsigned int output, unsigned int channel, Float_T value)
    {
        mSynapses(0, 0, channel, output) = value;
    };
    inline void setBias(unsigned int output, Float_T value)
    {
        mBias(output) = value;
    };

    Parameter<double> mDropConnect;

    // Internal
    std::vector<std::shared_ptr<Solver<Float_T> > > mWeightsSolvers;
    Interface<Float_T> mSynapses;
    Tensor4d<Float_T> mBias;
    Interface<Float_T> mDiffSynapses;
    Tensor4d<Float_T> mDiffBias;

    Interface<bool> mDropConnectMask;
    bool mLockRandom;

private:
    static Registrar<FcCell> mRegistrar;
};
}

#endif // N2D2_FCCELL_FRAME_H
