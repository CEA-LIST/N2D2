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

#ifndef N2D2_FCCELL_FRAME_CUDA_H
#define N2D2_FCCELL_FRAME_CUDA_H

#include "Cell_Frame_CUDA.hpp"
#include "CudaContext.hpp"
#include "CudaUtils.hpp"
#include "FcCell.hpp"
#include "Network.hpp"
#include "Solver/SGDSolver_Frame_CUDA.hpp"
#include "containers/CudaTensor4d.hpp"

namespace N2D2 {
class FcCell_Frame_CUDA : public virtual FcCell, public Cell_Frame_CUDA {
public:
    FcCell_Frame_CUDA(const std::string& name,
                      unsigned int nbOutputs,
                      const std::shared_ptr<Activation<Float_T> >& activation
                      = std::make_shared
                      <TanhActivation_Frame_CUDA<Float_T> >());
    static std::shared_ptr<FcCell>
    create(Network& /*net*/,
           const std::string& name,
           unsigned int nbOutputs,
           const std::shared_ptr<Activation<Float_T> >& activation
           = std::make_shared<TanhActivation_Frame_CUDA<Float_T> >())
    {
        return std::make_shared<FcCell_Frame_CUDA>(name, nbOutputs, activation);
    }

    virtual void initialize();
    virtual void propagate(bool inference = false);
    virtual void backPropagate();
    virtual void update();
    inline Float_T getWeight(unsigned int output, unsigned int channel) const;
    inline Float_T getBias(unsigned int output) const;
    void checkGradient(double epsilon = 1.0e-4, double maxError = 1.0e-6);
    void logFreeParameters(const std::string& fileName,
                           unsigned int output) const;
    void logFreeParameters(const std::string& dirName) const;
    void saveFreeParameters(const std::string& fileName) const;
    void loadFreeParameters(const std::string& fileName,
                            bool ignoreNotExists = false);
    void exportFreeParameters(const std::string& fileName) const;
    void importFreeParameters(const std::string& fileName,
                              bool ignoreNotExists = false);
    void logFreeParametersDistrib(const std::string& fileName) const;
    void exportSolverParameters(const std::string& fileName) const;
    void discretizeFreeParameters(unsigned int nbLevels);
    std::pair<Float_T, Float_T> getFreeParametersRange() const;
    void processFreeParameters(const std::function
                               <double(const double&)>& func);
    virtual ~FcCell_Frame_CUDA();

protected:
    inline void
    setWeight(unsigned int output, unsigned int channel, Float_T value);
    inline void setBias(unsigned int output, Float_T value);

    // Internal
    std::vector<std::shared_ptr<Solver<Float_T> > > mWeightsSolvers;
    CudaInterface<Float_T> mSynapses;
    CudaTensor4d<Float_T> mBias;
    CudaInterface<Float_T> mDiffSynapses;
    CudaTensor4d<Float_T> mDiffBias;

    Float_T* mOnesVector; // Bias inputs
    mutable bool mSynchronized;

private:
    static Registrar<FcCell> mRegistrar;
};
}

void N2D2::FcCell_Frame_CUDA::setWeight(unsigned int output,
                                        unsigned int channel,
                                        Float_T value)
{
    mSynapses(0, 0, channel, output) = value;

    if (!mSynchronized)
        mSynapses.synchronizeHToD(0, 0, channel, output, 1);
}

N2D2::Float_T N2D2::FcCell_Frame_CUDA::getWeight(unsigned int output,
                                                 unsigned int channel) const
{
    if (!mSynchronized)
        mSynapses.synchronizeDToH(0, 0, channel, output, 1);

    return mSynapses(0, 0, channel, output);
}

void N2D2::FcCell_Frame_CUDA::setBias(unsigned int output, Float_T value)
{
    mBias(output) = value;

    if (!mSynchronized)
        mBias.synchronizeHToD(output, 1);
}

N2D2::Float_T N2D2::FcCell_Frame_CUDA::getBias(unsigned int output) const
{
    if (!mSynchronized)
        mBias.synchronizeDToH(output, 1);

    return mBias(output);
}

#endif // N2D2_FCCELL_FRAME_CUDA_H
