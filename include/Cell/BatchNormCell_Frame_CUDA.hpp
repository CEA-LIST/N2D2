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

#ifndef N2D2_BATCHNORMCELL_FRAME_CUDA_H
#define N2D2_BATCHNORMCELL_FRAME_CUDA_H

#include <tuple>
#include <unordered_map>
#include <vector>

#include "BatchNormCell.hpp"
#include "Cell_Frame_CUDA.hpp"
#include "Solver/SGDSolver_Frame_CUDA.hpp"

#include "CudaContext.hpp"
#include "CudaUtils.hpp"
#include "containers/CudaTensor4d.hpp"

namespace N2D2 {
class BatchNormCell_Frame_CUDA : public virtual BatchNormCell,
                                 public Cell_Frame_CUDA {
public:
    BatchNormCell_Frame_CUDA(const std::string& name,
                             unsigned int nbOutputs,
                             const std::shared_ptr
                             <Activation<Float_T> >& activation
                             = std::make_shared
                             <TanhActivation_Frame_CUDA<Float_T> >());
    static std::shared_ptr<BatchNormCell>
    create(const std::string& name,
           unsigned int nbOutputs,
           const std::shared_ptr<Activation<Float_T> >& activation
           = std::make_shared<TanhActivation_Frame_CUDA<Float_T> >())
    {
        return std::make_shared
            <BatchNormCell_Frame_CUDA>(name, nbOutputs, activation);
    }

    virtual void initialize();
    virtual void propagate(bool inference = false);
    virtual void backPropagate();
    virtual void update();
    inline Float_T
    getScale(unsigned int channel, unsigned int sx, unsigned int sy) const;
    inline Float_T
    getBias(unsigned int channel, unsigned int sx, unsigned int sy) const;
    inline Float_T
    getMean(unsigned int channel, unsigned int sx, unsigned int sy) const;
    inline Float_T
    getVariance(unsigned int channel, unsigned int sx, unsigned int sy) const;
    inline std::shared_ptr<Tensor4d<Float_T> > getScales()
    {
        return mScale;
    };
    void setScales(const std::shared_ptr<Tensor4d<Float_T> >& scales);
    inline std::shared_ptr<Tensor4d<Float_T> > getBiases()
    {
        return mBias;
    };
    void setBiases(const std::shared_ptr<Tensor4d<Float_T> >& biases);
    inline std::shared_ptr<Tensor4d<Float_T> > getMeans()
    {
        return mMean;
    };
    void setMeans(const std::shared_ptr<Tensor4d<Float_T> >& means);
    inline std::shared_ptr<Tensor4d<Float_T> > getVariances()
    {
        return mVariance;
    };
    void setVariances(const std::shared_ptr<Tensor4d<Float_T> >& variances);
    void checkGradient(double epsilon = 1.0e-4, double maxError = 1.0e-6);
    void saveFreeParameters(const std::string& fileName) const;
    void loadFreeParameters(const std::string& fileName,
                            bool ignoreNotExists = false);
    void exportFreeParameters(const std::string& fileName) const;
    void importFreeParameters(const std::string& fileName,
                              bool ignoreNotExists = false);
    virtual ~BatchNormCell_Frame_CUDA();

protected:
    inline void setScale(unsigned int channel,
                         unsigned int sx,
                         unsigned int sy,
                         Float_T value);
    inline void setBias(unsigned int channel,
                        unsigned int sx,
                        unsigned int sy,
                        Float_T value);
    inline void setMean(unsigned int channel,
                        unsigned int sx,
                        unsigned int sy,
                        Float_T value);
    inline void setVariance(unsigned int channel,
                            unsigned int sx,
                            unsigned int sy,
                            Float_T value);

    cudnnBatchNormMode_t mMode;
    unsigned int mNbPropagate;
    std::shared_ptr<CudaTensor4d<Float_T> > mScale;
    std::shared_ptr<CudaTensor4d<Float_T> > mBias;
    CudaTensor4d<Float_T> mDiffScale;
    CudaTensor4d<Float_T> mDiffBias;
    std::shared_ptr<CudaTensor4d<Float_T> > mMean;
    std::shared_ptr<CudaTensor4d<Float_T> > mVariance;
    CudaTensor4d<Float_T> mSavedMean;
    CudaTensor4d<Float_T> mSavedVariance;
    mutable bool mSynchronized;

private:
    static Registrar<BatchNormCell> mRegistrar;
};
}

void N2D2::BatchNormCell_Frame_CUDA::setScale(unsigned int channel,
                                              unsigned int sx,
                                              unsigned int sy,
                                              Float_T value)
{
    (*mScale)(sx, sy, channel, 0) = value;

    if (!mSynchronized)
        mScale->synchronizeHToD(sx, sy, channel, 0, 1);
}

N2D2::Float_T N2D2::BatchNormCell_Frame_CUDA::getScale(unsigned int channel,
                                                       unsigned int sx,
                                                       unsigned int sy) const
{
    if (!mSynchronized)
        mScale->synchronizeDToH(sx, sy, channel, 0, 1);

    return (*mScale)(sx, sy, channel, 0);
}

void N2D2::BatchNormCell_Frame_CUDA::setBias(unsigned int channel,
                                             unsigned int sx,
                                             unsigned int sy,
                                             Float_T value)
{
    (*mBias)(sx, sy, channel, 0) = value;

    if (!mSynchronized)
        mBias->synchronizeHToD(sx, sy, channel, 0, 1);
}

N2D2::Float_T N2D2::BatchNormCell_Frame_CUDA::getBias(unsigned int channel,
                                                      unsigned int sx,
                                                      unsigned int sy) const
{
    if (!mSynchronized)
        mBias->synchronizeDToH(sx, sy, channel, 0, 1);

    return (*mBias)(sx, sy, channel, 0);
}

void N2D2::BatchNormCell_Frame_CUDA::setMean(unsigned int channel,
                                             unsigned int sx,
                                             unsigned int sy,
                                             Float_T value)
{
    (*mMean)(sx, sy, channel, 0) = value;

    if (!mSynchronized)
        mMean->synchronizeHToD(sx, sy, channel, 0, 1);
}

N2D2::Float_T N2D2::BatchNormCell_Frame_CUDA::getMean(unsigned int channel,
                                                      unsigned int sx,
                                                      unsigned int sy) const
{
    if (!mSynchronized)
        mMean->synchronizeDToH(sx, sy, channel, 0, 1);

    return (*mMean)(sx, sy, channel, 0);
}

void N2D2::BatchNormCell_Frame_CUDA::setVariance(unsigned int channel,
                                                 unsigned int sx,
                                                 unsigned int sy,
                                                 Float_T value)
{
    (*mVariance)(sx, sy, channel, 0) = value;

    if (!mSynchronized)
        mVariance->synchronizeHToD(sx, sy, channel, 0, 1);
}

N2D2::Float_T N2D2::BatchNormCell_Frame_CUDA::getVariance(unsigned int channel,
                                                          unsigned int sx,
                                                          unsigned int sy) const
{
    if (!mSynchronized)
        mVariance->synchronizeDToH(sx, sy, channel, 0, 1);

    return (*mVariance)(sx, sy, channel, 0);
}

#endif // N2D2_BATCHNORMCELL_FRAME_CUDA_H
