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

#ifndef N2D2_DECONVCELL_FRAME_CUDA_H
#define N2D2_DECONVCELL_FRAME_CUDA_H

#include "Cell_Frame_CUDA.hpp"
#include "CudaContext.hpp"
#include "CudaUtils.hpp"
#include "DeconvCell.hpp"
#include "Solver/SGDSolver_Frame_CUDA.hpp"
#include "containers/CudaTensor4d.hpp"

namespace N2D2 {
class DeconvCell_Frame_CUDA : public virtual DeconvCell,
                              public Cell_Frame_CUDA {
public:
    DeconvCell_Frame_CUDA(const std::string& name,
                          unsigned int kernelWidth,
                          unsigned int kernelHeight,
                          unsigned int nbOutputs,
                          unsigned int strideX = 1,
                          unsigned int strideY = 1,
                          int paddingX = 0,
                          int paddingY = 0,
                          const std::shared_ptr
                          <Activation<Float_T> >& activation = std::make_shared
                          <TanhActivation_Frame_CUDA<Float_T> >());
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
           = std::make_shared<TanhActivation_Frame_CUDA<Float_T> >())
    {
        return std::make_shared<DeconvCell_Frame_CUDA>(name,
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
                             unsigned int sy) const;
    inline Float_T getBias(unsigned int output) const;
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
    void setBiases(const std::shared_ptr<Tensor4d<Float_T> >& biases);
    void checkGradient(double /*epsilon*/ = 1.0e-4,
                       double /*maxError*/ = 1.0e-6);
    void logFreeParameters(const std::string& fileName,
                           unsigned int output,
                           unsigned int channel) const;
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
    virtual ~DeconvCell_Frame_CUDA();

protected:
    inline void setWeight(unsigned int output,
                          unsigned int channel,
                          unsigned int sx,
                          unsigned int sy,
                          Float_T value);
    inline void setBias(unsigned int output, Float_T value);

    // Internal
    std::vector<std::shared_ptr<Solver<Float_T> > > mWeightsSolvers;
    CudaInterface<Float_T> mSharedSynapses;
    std::map<unsigned int,
        std::pair<CudaInterface<Float_T>*, unsigned int> > mExtSharedSynapses;
    std::shared_ptr<CudaTensor4d<Float_T> > mBias;
    CudaInterface<Float_T> mDiffSharedSynapses;
    CudaTensor4d<Float_T> mDiffBias;

    size_t mWorkspaceSize;
    void* mWorkspace;

    std::vector<cudnnFilterDescriptor_t> mFilterDesc;
    std::vector<cudnnConvolutionFwdAlgo_t> mFwdAlgo;
#if CUDNN_VERSION >= 5000
    std::vector<cudnnConvolutionBwdFilterAlgo_t> mBwdFilterAlgo;
    std::vector<cudnnConvolutionBwdDataAlgo_t> mBwdDataAlgo;
#endif
    cudnnConvolutionDescriptor_t mConvDesc;
    mutable bool mSynchronized;

private:
    static Registrar<DeconvCell> mRegistrar;
};
}

void N2D2::DeconvCell_Frame_CUDA::setWeight(unsigned int output,
                                            unsigned int channel,
                                            unsigned int sx,
                                            unsigned int sy,
                                            Float_T value)
{
    mSharedSynapses(sx, sy, output, channel) = value;

    if (!mSynchronized)
        mSharedSynapses.synchronizeHToD(sx, sy, output, channel, 1);
}

N2D2::Float_T N2D2::DeconvCell_Frame_CUDA::getWeight(unsigned int output,
                                                     unsigned int channel,
                                                     unsigned int sx,
                                                     unsigned int sy) const
{
    if (!mSynchronized)
        mSharedSynapses.synchronizeDToH(sx, sy, output, channel, 1);

    return mSharedSynapses(sx, sy, output, channel);
}

void N2D2::DeconvCell_Frame_CUDA::setBias(unsigned int output, Float_T value)
{
    (*mBias)(output) = value;

    if (!mSynchronized)
        mBias->synchronizeHToD(output, 1);
}

N2D2::Float_T N2D2::DeconvCell_Frame_CUDA::getBias(unsigned int output) const
{
    if (!mSynchronized)
        mBias->synchronizeDToH(output, 1);

    return (*mBias)(output);
}

#endif // N2D2_DECONVCELL_FRAME_CUDA_H
