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

#ifndef N2D2_BATCHNORMCELL_FRAME_H
#define N2D2_BATCHNORMCELL_FRAME_H

#include <tuple>
#include <unordered_map>
#include <vector>

#include "BatchNormCell.hpp"
#include "Cell_Frame.hpp"
#include "Solver/SGDSolver_Frame.hpp"

#ifdef WIN32
// For static library
#ifdef CUDA
#pragma comment(                                                               \
    linker,                                                                    \
    "/include:?mRegistrar@BatchNormCell_Frame_CUDA@N2D2@@0U?$Registrar@VBatchNormCell@N2D2@@@2@A")
#endif
#endif

namespace N2D2 {
class BatchNormCell_Frame : public virtual BatchNormCell, public Cell_Frame {
public:
    BatchNormCell_Frame(const std::string& name,
                        unsigned int nbOutputs,
                        const std::shared_ptr<Activation<Float_T> >& activation
                        = std::make_shared<TanhActivation_Frame<Float_T> >());
    static std::shared_ptr<BatchNormCell>
    create(const std::string& name,
           unsigned int nbOutputs,
           const std::shared_ptr<Activation<Float_T> >& activation
           = std::make_shared<TanhActivation_Frame<Float_T> >())
    {
        return std::make_shared
            <BatchNormCell_Frame>(name, nbOutputs, activation);
    }

    virtual void initialize();
    virtual void propagate(bool inference = false);
    virtual void backPropagate();
    virtual void update();
    inline Float_T
    getScale(unsigned int channel, unsigned int sx, unsigned int sy) const
    {
        return mScale(sx, sy, channel, 0);
    }
    inline Float_T
    getBias(unsigned int channel, unsigned int sx, unsigned int sy) const
    {
        return mBias(sx, sy, channel, 0);
    }
    inline Float_T
    getMean(unsigned int channel, unsigned int sx, unsigned int sy) const
    {
        return mMean(sx, sy, channel, 0);
    }
    inline Float_T
    getVariance(unsigned int channel, unsigned int sx, unsigned int sy) const
    {
        return mVariance(sx, sy, channel, 0);
    }
    void checkGradient(double epsilon = 1.0e-4, double maxError = 1.0e-6);
    void saveFreeParameters(const std::string& fileName) const;
    void loadFreeParameters(const std::string& fileName,
                            bool ignoreNotExists = false);
    void discretizeFreeParameters(unsigned int /*nbLevels*/) {}; // no free
    // parameter to
    // discretize
    virtual ~BatchNormCell_Frame();

protected:
    inline void setScale(unsigned int channel,
                         unsigned int sx,
                         unsigned int sy,
                         Float_T value)
    {
        mScale(sx, sy, channel, 0) = value;
    }
    inline void setBias(unsigned int channel,
                        unsigned int sx,
                        unsigned int sy,
                        Float_T value)
    {
        mBias(sx, sy, channel, 0) = value;
    }
    inline void setMean(unsigned int channel,
                        unsigned int sx,
                        unsigned int sy,
                        Float_T value)
    {
        mMean(sx, sy, channel, 0) = value;
    }
    inline void setVariance(unsigned int channel,
                            unsigned int sx,
                            unsigned int sy,
                            Float_T value)
    {
        mVariance(sx, sy, channel, 0) = value;
    }

    unsigned int mNbPropagate;
    Tensor4d<Float_T> mScale;
    Tensor4d<Float_T> mBias;
    Tensor4d<Float_T> mMean;
    Tensor4d<Float_T> mVariance;
    Tensor4d<Float_T> mDiffScale;
    Tensor4d<Float_T> mDiffBias;
    Tensor4d<Float_T> mDiffSavedMean;
    Tensor4d<Float_T> mDiffSavedVariance;
    Tensor4d<Float_T> mSavedMean;
    Tensor4d<Float_T> mSavedVariance;

private:
    static Registrar<BatchNormCell> mRegistrar;
};
}

#endif // N2D2_BATCHNORMCELL_FRAME_H
