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

#ifdef CUDA
#include <cudnn.h>
#if CUDNN_VERSION >= 4000

#include "Cell/BatchNormCell_Frame_CUDA.hpp"

N2D2::Registrar<N2D2::BatchNormCell> N2D2::BatchNormCell_Frame_CUDA::mRegistrar(
    "Frame_CUDA", N2D2::BatchNormCell_Frame_CUDA::create);

N2D2::BatchNormCell_Frame_CUDA::BatchNormCell_Frame_CUDA(
    const std::string& name,
    unsigned int nbOutputs,
    const std::shared_ptr<Activation<Float_T> >& activation)
    : Cell(name, nbOutputs),
      BatchNormCell(name, nbOutputs),
      Cell_Frame_CUDA(name, nbOutputs, activation),
      mScale(std::make_shared<CudaTensor4d<Float_T> >()),
      mBias(std::make_shared<CudaTensor4d<Float_T> >()),
      mMean(std::make_shared<CudaTensor4d<Float_T> >()),
      mVariance(std::make_shared<CudaTensor4d<Float_T> >())
{
    // ctor
    mScaleSolver = std::make_shared<SGDSolver_Frame_CUDA<Float_T> >();
    mBiasSolver = std::make_shared<SGDSolver_Frame_CUDA<Float_T> >();
}

void N2D2::BatchNormCell_Frame_CUDA::initialize()
{
    if (mInputs.size() > 1)
        throw std::domain_error("BatchNormCell_Frame_CUDA::initialize(): "
                                "inputs concatenation is not supported.");

    mMode = CUDNN_BATCHNORM_SPATIAL;
    mNbPropagate = 0;

    if (mEpsilon == 0.0)
        mEpsilon = CUDNN_BN_MIN_EPSILON;

    cudnnTensorDescriptor_t derivedBnDesc;
    CHECK_CUDNN_STATUS(cudnnCreateTensorDescriptor(&derivedBnDesc));
    CHECK_CUDNN_STATUS(cudnnDeriveBNTensorDescriptor(
        derivedBnDesc, mInputs[0].getCudnnTensorDesc(), mMode));

    cudnnDataType_t dataType;
    int n, c, h, w;
    int nStride, cStride, hStride, wStride;

    CHECK_CUDNN_STATUS(cudnnGetTensor4dDescriptor(derivedBnDesc,
                                                  &dataType,
                                                  &n,
                                                  &c,
                                                  &h,
                                                  &w,
                                                  &nStride,
                                                  &cStride,
                                                  &hStride,
                                                  &wStride));

    CHECK_CUDNN_STATUS(cudnnDestroyTensorDescriptor(derivedBnDesc));

    if (mScale->empty())
        mScale->resize(w, h, c, n, 1.0);
    else {
        if ((int)mScale->dimX() != w || (int)mScale->dimY() != h
            || (int)mScale->dimZ() != c || (int)mScale->dimB() != n)
        {
            throw std::runtime_error("BatchNormCell_Frame_CUDA::initialize():"
                " in cell " + mName + ", wrong size for shared scale");
        }
    }

    if (mBias->empty())
        mBias->resize(w, h, c, n, 0.0);
    else {
        if ((int)mBias->dimX() != w || (int)mBias->dimY() != h
            || (int)mBias->dimZ() != c || (int)mBias->dimB() != n)
        {
            throw std::runtime_error("BatchNormCell_Frame_CUDA::initialize():"
                " in cell " + mName + ", wrong size for shared bias");
        }
    }

    if (mMean->empty())
        mMean->resize(w, h, c, n, 0.0);
    else {
        if ((int)mMean->dimX() != w || (int)mMean->dimY() != h
            || (int)mMean->dimZ() != c || (int)mMean->dimB() != n)
        {
            throw std::runtime_error("BatchNormCell_Frame_CUDA::initialize():"
                " in cell " + mName + ", wrong size for shared mean");
        }
    }

    if (mVariance->empty())
        mVariance->resize(w, h, c, n, 0.0);
    else {
        if ((int)mVariance->dimX() != w || (int)mVariance->dimY() != h
            || (int)mVariance->dimZ() != c || (int)mVariance->dimB() != n)
        {
            throw std::runtime_error("BatchNormCell_Frame_CUDA::initialize():"
                " in cell " + mName + ", wrong size for shared variance");
        }
    }

    mSavedMean.resize(w, h, c, n);
    mSavedVariance.resize(w, h, c, n);

    mDiffScale.resize(w, h, c, n);
    mDiffBias.resize(w, h, c, n);
}

void N2D2::BatchNormCell_Frame_CUDA::propagate(bool inference)
{
    mInputs.synchronizeHBasedToD();

    const float alpha = 1.0f;
    const float beta = 0.0f;

    if (inference) {
        CHECK_CUDNN_STATUS(cudnnBatchNormalizationForwardInference(
            CudaContext::cudnnHandle(),
            mMode,
            &alpha,
            &beta,
            mInputs[0].getCudnnTensorDesc(),
            mInputs[0].getDevicePtr(),
            mOutputs.getCudnnTensorDesc(),
            mOutputs.getDevicePtr(),
            mScale->getCudnnTensorDesc(),
            mScale->getDevicePtr(),
            mBias->getDevicePtr(),
            mMean->getDevicePtr(),
            mVariance->getDevicePtr(),
            mEpsilon));
    } else {
        // Cumulative Moving Average (CMA)
        const double expAverageFactor = 1.0 / (1.0 + mNbPropagate);

        CHECK_CUDNN_STATUS(cudnnBatchNormalizationForwardTraining(
            CudaContext::cudnnHandle(),
            mMode,
            &alpha,
            &beta,
            mInputs[0].getCudnnTensorDesc(),
            mInputs[0].getDevicePtr(),
            mOutputs.getCudnnTensorDesc(),
            mOutputs.getDevicePtr(),
            mScale->getCudnnTensorDesc(),
            mScale->getDevicePtr(),
            mBias->getDevicePtr(),
            expAverageFactor,
            mMean->getDevicePtr(),
            mVariance->getDevicePtr(),
            mEpsilon,
            mSavedMean.getDevicePtr(),
            mSavedVariance.getDevicePtr()));
    }

    if (!inference)
        ++mNbPropagate;

    Cell_Frame_CUDA::propagate();
    mDiffInputs.clearValid();
}

void N2D2::BatchNormCell_Frame_CUDA::backPropagate()
{
    Cell_Frame_CUDA::backPropagate();

    const float alpha = 1.0f;
    const float alphaData = 1.0f;
    assert(mScaleSolver->isNewIteration() == mBiasSolver->isNewIteration());
    const float beta = (mScaleSolver->isNewIteration()) ? 0.0f : 1.0f;
    const float betaData = (mDiffOutputs[0].isValid()) ? 1.0f : 0.0f;

    CHECK_CUDNN_STATUS(
        cudnnBatchNormalizationBackward(CudaContext::cudnnHandle(),
                                        mMode,
                                        &alphaData,
                                        &betaData,
                                        &alpha,
                                        &beta,
                                        mInputs[0].getCudnnTensorDesc(),
                                        mInputs[0].getDevicePtr(),
                                        mOutputs.getCudnnTensorDesc(),
                                        mDiffInputs.getDevicePtr(),
                                        mDiffOutputs[0].getCudnnTensorDesc(),
                                        mDiffOutputs[0].getDevicePtr(),
                                        mScale->getCudnnTensorDesc(),
                                        mScale->getDevicePtr(),
                                        mDiffScale.getDevicePtr(),
                                        mDiffBias.getDevicePtr(),
                                        mEpsilon,
                                        mSavedMean.getDevicePtr(),
                                        mSavedVariance.getDevicePtr()));

    mDiffOutputs[0].setValid();
    mDiffOutputs.synchronizeDToHBased();
}

void N2D2::BatchNormCell_Frame_CUDA::update()
{
    mScaleSolver->update(&(*mScale), &mDiffScale, mInputs.dimB());
    mBiasSolver->update(&(*mBias), &mDiffBias, mInputs.dimB());
}

void N2D2::BatchNormCell_Frame_CUDA::setScales(
    const std::shared_ptr<Tensor4d<Float_T> >& scales)
{
    std::shared_ptr<CudaTensor4d<Float_T> > cudaScales
        = std::dynamic_pointer_cast<CudaTensor4d<Float_T> >(scales);

    if (!cudaScales) {
        throw std::runtime_error("BatchNormCell_Frame_CUDA::setBiases(): scales"
                                 " must be a CudaTensor4d");
    }

    mScale = cudaScales;
}

void N2D2::BatchNormCell_Frame_CUDA::setBiases(
    const std::shared_ptr<Tensor4d<Float_T> >& biases)
{
    std::shared_ptr<CudaTensor4d<Float_T> > cudaBiases
        = std::dynamic_pointer_cast<CudaTensor4d<Float_T> >(biases);

    if (!cudaBiases) {
        throw std::runtime_error("BatchNormCell_Frame_CUDA::setBiases(): biases"
                                 " must be a CudaTensor4d");
    }

    mBias = cudaBiases;
}

void N2D2::BatchNormCell_Frame_CUDA::setMeans(
    const std::shared_ptr<Tensor4d<Float_T> >& means)
{
    std::shared_ptr<CudaTensor4d<Float_T> > cudaMeans
        = std::dynamic_pointer_cast<CudaTensor4d<Float_T> >(means);

    if (!cudaMeans) {
        throw std::runtime_error("BatchNormCell_Frame_CUDA::setBiases(): means"
                                 " must be a CudaTensor4d");
    }

    mMean = cudaMeans;
}

void N2D2::BatchNormCell_Frame_CUDA::setVariances(
    const std::shared_ptr<Tensor4d<Float_T> >& variances)
{
    std::shared_ptr<CudaTensor4d<Float_T> > cudaVariances
        = std::dynamic_pointer_cast<CudaTensor4d<Float_T> >(variances);

    if (!cudaVariances) {
        throw std::runtime_error("BatchNormCell_Frame_CUDA::setBiases():"
                                 " variances must be a CudaTensor4d");
    }

    mVariance = cudaVariances;
}

void N2D2::BatchNormCell_Frame_CUDA::checkGradient(double epsilon,
                                                   double maxError)
{
    GradientCheck gc(epsilon, maxError);
    gc.initialize(mInputs,
                  mOutputs,
                  mDiffInputs,
                  std::bind(&BatchNormCell_Frame_CUDA::propagate, this, false),
                  std::bind(&BatchNormCell_Frame_CUDA::backPropagate, this));
    gc.check(mName + "_mDiffScale", (*mScale), mDiffScale);
    gc.check(mName + "_mDiffBias", (*mBias), mDiffBias);

    if (!mDiffOutputs.empty()) {
        for (unsigned int in = 0; in < mInputs.size(); ++in) {
            std::stringstream name;
            name << mName + "_mDiffOutputs[" << in << "]";

            gc.check(name.str(), mInputs[in], mDiffOutputs[in]);
        }
    } else {
        std::cout << Utils::cwarning << "Empty diff. outputs for cell " << mName
                  << ", could not check the gradient!" << Utils::cdef
                  << std::endl;
    }
}

void N2D2::BatchNormCell_Frame_CUDA::saveFreeParameters(const std::string
                                                        & fileName) const
{
    std::ofstream syn(fileName.c_str(), std::fstream::binary);

    if (!syn.good())
        throw std::runtime_error("Could not create parameter file (.SYN): "
                                 + fileName);

    mScale->synchronizeDToH();

    for (std::vector<Float_T>::const_iterator it = mScale->begin();
         it != mScale->end();
         ++it)
        syn.write(reinterpret_cast<const char*>(&(*it)), sizeof(*it));

    mBias->synchronizeDToH();

    for (std::vector<Float_T>::const_iterator it = mBias->begin();
         it != mBias->end();
         ++it)
        syn.write(reinterpret_cast<const char*>(&(*it)), sizeof(*it));

    mMean->synchronizeDToH();

    for (std::vector<Float_T>::const_iterator it = mMean->begin();
         it != mMean->end();
         ++it)
        syn.write(reinterpret_cast<const char*>(&(*it)), sizeof(*it));

    mVariance->synchronizeDToH();

    for (std::vector<Float_T>::const_iterator it = mVariance->begin();
         it != mVariance->end();
         ++it)
        syn.write(reinterpret_cast<const char*>(&(*it)), sizeof(*it));

    if (!syn.good())
        throw std::runtime_error("Error writing parameter file: " + fileName);
}

void N2D2::BatchNormCell_Frame_CUDA::loadFreeParameters(const std::string
                                                        & fileName,
                                                        bool ignoreNotExists)
{
    std::ifstream syn(fileName.c_str(), std::fstream::binary);

    if (!syn.good()) {
        if (ignoreNotExists) {
            std::cout << Utils::cnotice
                      << "Notice: Could not open parameter file (.SYN): "
                      << fileName << Utils::cdef << std::endl;
            return;
        } else
            throw std::runtime_error("Could not open parameter file (.SYN): "
                                     + fileName);
    }

    for (std::vector<Float_T>::iterator it = mScale->begin(); it != mScale->end();
         ++it)
        syn.read(reinterpret_cast<char*>(&(*it)), sizeof(*it));

    mScale->synchronizeHToD();

    for (std::vector<Float_T>::iterator it = mBias->begin(); it != mBias->end();
         ++it)
        syn.read(reinterpret_cast<char*>(&(*it)), sizeof(*it));

    mBias->synchronizeHToD();

    for (std::vector<Float_T>::iterator it = mMean->begin(); it != mMean->end();
         ++it)
        syn.read(reinterpret_cast<char*>(&(*it)), sizeof(*it));

    mMean->synchronizeHToD();

    for (std::vector<Float_T>::iterator it = mVariance->begin();
         it != mVariance->end();
         ++it)
        syn.read(reinterpret_cast<char*>(&(*it)), sizeof(*it));

    mVariance->synchronizeHToD();

    if (syn.eof())
        throw std::runtime_error(
            "End-of-file reached prematurely in parameter file (.SYN): "
            + fileName);
    else if (!syn.good())
        throw std::runtime_error("Error while reading parameter file (.SYN): "
                                 + fileName);
    else if (syn.get() != std::fstream::traits_type::eof())
        throw std::runtime_error(
            "Synaptic file (.SYN) size larger than expected: " + fileName);
}

void N2D2::BatchNormCell_Frame_CUDA::exportFreeParameters(const std::string
                                                          & fileName) const
{
    mScale->synchronizeDToH();
    mBias->synchronizeDToH();
    mMean->synchronizeDToH();
    mVariance->synchronizeDToH();

    mSynchronized = true;
    BatchNormCell::exportFreeParameters(fileName);
    mSynchronized = false;
}

void N2D2::BatchNormCell_Frame_CUDA::importFreeParameters(const std::string
                                                          & fileName,
                                                          bool ignoreNotExists)
{
    mSynchronized = true;
    BatchNormCell::importFreeParameters(fileName, ignoreNotExists);
    mSynchronized = false;

    mScale->synchronizeHToD();
    mBias->synchronizeHToD();
    mMean->synchronizeHToD();
    mVariance->synchronizeHToD();
}

N2D2::BatchNormCell_Frame_CUDA::~BatchNormCell_Frame_CUDA()
{
}

#endif
#endif
