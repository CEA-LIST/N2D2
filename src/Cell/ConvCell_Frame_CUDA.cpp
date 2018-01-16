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

#ifdef CUDA

#include "Cell/ConvCell_Frame_CUDA.hpp"

N2D2::Registrar<N2D2::ConvCell>
N2D2::ConvCell_Frame_CUDA::mRegistrar("Frame_CUDA",
                                      N2D2::ConvCell_Frame_CUDA::create);

N2D2::ConvCell_Frame_CUDA::ConvCell_Frame_CUDA(
    const std::string& name,
    unsigned int kernelWidth,
    unsigned int kernelHeight,
    unsigned int nbOutputs,
    unsigned int subSampleX,
    unsigned int subSampleY,
    unsigned int strideX,
    unsigned int strideY,
    int paddingX,
    int paddingY,
    const std::shared_ptr<Activation<Float_T> >& activation)
    : Cell(name, nbOutputs),
      ConvCell(name,
               kernelWidth,
               kernelHeight,
               nbOutputs,
               subSampleX,
               subSampleY,
               strideX,
               strideY,
               paddingX,
               paddingY),
      Cell_Frame_CUDA(name, nbOutputs, activation),
      // IMPORTANT: Do not change the value of the parameters here! Use
      // setParameter() or loadParameters().
      mBias(std::make_shared<CudaTensor4d<Float_T> >()),
      mDiffBias(1, 1, mNbOutputs, 1),
      mWorkspaceSize(0),
      mWorkspace(NULL),
      mSynchronized(false)
{
    mWeightsSolver = std::make_shared<SGDSolver_Frame_CUDA<Float_T> >();
    mBiasSolver = std::make_shared<SGDSolver_Frame_CUDA<Float_T> >();

    CHECK_CUDNN_STATUS(cudnnCreateConvolutionDescriptor(&mConvDesc));
}

void N2D2::ConvCell_Frame_CUDA::initialize()
{
    if (!mNoBias) {
        if (mBias->empty()) {
            mBias->resize(1, 1, mNbOutputs, 1);
            mBiasFiller->apply((*mBias));
            mBias->synchronizeHToD();
        }
        else {
            if (mBias->dimX() != 1 || mBias->dimY() != 1
                || mBias->dimZ() != mNbOutputs || mBias->dimB() != 1)
            {
                throw std::runtime_error("ConvCell_Frame_CUDA::initialize():"
                    " in cell " + mName + ", wrong size for shared bias");
            }
        }
    }

#if CUDNN_VERSION >= 6000
    CHECK_CUDNN_STATUS(
        cudnnSetConvolution2dDescriptor(mConvDesc,
                                        mPaddingY,
                                        mPaddingX,
                                        mStrideY,
                                        mStrideX,
                                        1,
                                        1,
                                        CUDNN_CROSS_CORRELATION,
                                        CudaContext::data_type));
#else
    CHECK_CUDNN_STATUS(
        cudnnSetConvolution2dDescriptor(mConvDesc,
                                        mPaddingY,
                                        mPaddingX,
                                        mStrideY,
                                        mStrideX,
                                        1,
                                        1,
                                        CUDNN_CROSS_CORRELATION));
#endif

    size_t workspaceSize = 0;

    for (unsigned int k = 0, size = mInputs.size(); k < size; ++k) {
        if (mInputs[k].size() == 0)
            throw std::runtime_error("Zero-sized input for ConvCell " + mName);

        mWeightsSolvers.push_back(mWeightsSolver->clone());

        std::map<unsigned int,
            std::pair<CudaInterface<Float_T>*, unsigned int> >::const_iterator
                it = mExtSharedSynapses.find(k);

        if (it != mExtSharedSynapses.end()) {
            CudaTensor4d<Float_T>* extWeights
                = &(*((*it).second.first))[(*it).second.second];

            if (extWeights->dimX() != mKernelWidth
                || extWeights->dimY() != mKernelHeight
                || extWeights->dimZ() != mInputs[k].dimZ()
                || extWeights->dimB() != mNbOutputs)
            {
                std::stringstream errorStr;
                errorStr << "ConvCell_Frame_CUDA::initialize(): in cell "
                    << mName << ", mismatch between external weights dim. ("
                    << extWeights->dimX() << "x"
                    << extWeights->dimY() << "x"
                    << extWeights->dimZ() << "x"
                    << extWeights->dimB() << ") and expected dim. ("
                    << mKernelWidth << "x" << mKernelHeight << "x"
                    << mInputs[k].dimZ() << "x" << mNbOutputs << ")";

                throw std::runtime_error(errorStr.str());
            }

            mSharedSynapses.push_back(extWeights);
        }
        else {
            mSharedSynapses.push_back(new CudaTensor4d<Float_T>(
                mKernelWidth, mKernelHeight, mInputs[k].dimZ(), mNbOutputs));
            mWeightsFiller->apply(mSharedSynapses.back());

            if (!isFullMap()) {
                // Set the non-connected kernels coefficients to 0
                for (unsigned int output = 0; output < mNbOutputs; ++output) {
                    for (unsigned int channel = 0; channel < mInputs[k].dimZ();
                         ++channel) {
                        if (!isConnection(channel, output))
                            mSharedSynapses.back()[output][channel] = Tensor2d
                                <Float_T>(mKernelWidth, mKernelHeight, 0.0);
                    }
                }
            }

            mSharedSynapses.back().synchronizeHToD();
        }

        mDiffSharedSynapses.push_back(new CudaTensor4d<Float_T>(
            mKernelWidth, mKernelHeight, mInputs[k].dimZ(), mNbOutputs));

        mFilterDesc.push_back(cudnnFilterDescriptor_t());

        CHECK_CUDNN_STATUS(cudnnCreateFilterDescriptor(&mFilterDesc.back()));
#if CUDNN_VERSION >= 5000
        CHECK_CUDNN_STATUS(cudnnSetFilter4dDescriptor(mFilterDesc.back(),
                                                      CudaContext::data_type,
                                                      CUDNN_TENSOR_NCHW,
                                                      mNbOutputs,
                                                      mInputs[k].dimZ(),
                                                      mKernelHeight,
                                                      mKernelWidth));
#else
        CHECK_CUDNN_STATUS(cudnnSetFilter4dDescriptor(mFilterDesc.back(),
                                                      CudaContext::data_type,
                                                      mNbOutputs,
                                                      mInputs[k].dimZ(),
                                                      mKernelHeight,
                                                      mKernelWidth));
#endif

        mFwdAlgo.push_back(cudnnConvolutionFwdAlgo_t());

        CHECK_CUDNN_STATUS(cudnnGetConvolutionForwardAlgorithm(
            CudaContext::cudnnHandle(),
            mInputs[k].getCudnnTensorDesc(),
            mFilterDesc.back(),
            mConvDesc,
            mOutputs.getCudnnTensorDesc(),
            CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
            0,
            &mFwdAlgo.back()));

#if CUDNN_VERSION >= 5000
        mBwdFilterAlgo.push_back(cudnnConvolutionBwdFilterAlgo_t());

        CHECK_CUDNN_STATUS(cudnnGetConvolutionBackwardFilterAlgorithm(
            CudaContext::cudnnHandle(),
            mInputs[k].getCudnnTensorDesc(),
            mOutputs.getCudnnTensorDesc(),
            mConvDesc,
            mFilterDesc.back(),
            CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST,
            0,
            &mBwdFilterAlgo.back()));

        mBwdDataAlgo.push_back(cudnnConvolutionBwdDataAlgo_t());

        CHECK_CUDNN_STATUS(cudnnGetConvolutionBackwardDataAlgorithm(
            CudaContext::cudnnHandle(),
            mFilterDesc.back(),
            mOutputs.getCudnnTensorDesc(),
            mConvDesc,
            mInputs[k].getCudnnTensorDesc(),
            CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST,
            0,
            &mBwdDataAlgo.back()));
#endif

        CHECK_CUDNN_STATUS(cudnnGetConvolutionForwardWorkspaceSize(
            CudaContext::cudnnHandle(),
            mInputs[k].getCudnnTensorDesc(),
            mFilterDesc.back(),
            mConvDesc,
            mOutputs.getCudnnTensorDesc(),
            mFwdAlgo.back(),
            &workspaceSize));

        if (workspaceSize > mWorkspaceSize)
            mWorkspaceSize = workspaceSize;

#if CUDNN_VERSION >= 5000
        CHECK_CUDNN_STATUS(cudnnGetConvolutionBackwardFilterWorkspaceSize(
            CudaContext::cudnnHandle(),
            // same arguments as cudnnGetConvolutionBackwardFilterAlgorithm()
            // -->
            mInputs[k].getCudnnTensorDesc(),
            mOutputs.getCudnnTensorDesc(),
            mConvDesc,
            mFilterDesc.back(),
            // <--
            mBwdFilterAlgo.back(),
            &workspaceSize));

        if (workspaceSize > mWorkspaceSize)
            mWorkspaceSize = workspaceSize;

        CHECK_CUDNN_STATUS(cudnnGetConvolutionBackwardDataWorkspaceSize(
            CudaContext::cudnnHandle(),
            // same arguments as cudnnGetConvolutionBackwardDataAlgorithm() -->
            mFilterDesc.back(),
            mOutputs.getCudnnTensorDesc(),
            mConvDesc,
            mInputs[k].getCudnnTensorDesc(),
            // <--
            mBwdDataAlgo.back(),
            &workspaceSize));
#endif

        if (workspaceSize > mWorkspaceSize)
            mWorkspaceSize = workspaceSize;
    }

    if (mWorkspaceSize > 0)
        CHECK_CUDA_STATUS(cudaMalloc(&mWorkspace, mWorkspaceSize));
}

void N2D2::ConvCell_Frame_CUDA::propagate(bool /*inference*/)
{
    mInputs.synchronizeHBasedToD();

    /**
     * 1.0
     * Corps de la procédure de convolution via CuDNN
     * Pour plus de détails, cf. doc : cuDNN Library
     */
    const float alpha = 1.0f;
    float beta = 0.0f;

    for (unsigned int k = 0, size = mInputs.size(); k < size; ++k) {
        if (k > 0)
            beta = 1.0f;

        CHECK_CUDNN_STATUS(
            cudnnConvolutionForward(CudaContext::cudnnHandle(),
                                    &alpha,
                                    mInputs[k].getCudnnTensorDesc(),
                                    mInputs[k].getDevicePtr(),
                                    mFilterDesc[k],
                                    mSharedSynapses[k].getDevicePtr(),
                                    mConvDesc,
                                    mFwdAlgo[k],
                                    mWorkspace,
                                    mWorkspaceSize,
                                    &beta,
                                    mOutputs.getCudnnTensorDesc(),
                                    mOutputs.getDevicePtr()));
    }

    if (!mNoBias) {
/**
 * 2.0
 * Ajoute le biais au tenseur de destination.
 */
#if CUDNN_VERSION >= 5000
        CHECK_CUDNN_STATUS(cudnnAddTensor(CudaContext::cudnnHandle(),
                                          &alpha,
                                          mBias->getCudnnTensorDesc(),
                                          mBias->getDevicePtr(),
                                          &alpha,
                                          mOutputs.getCudnnTensorDesc(),
                                          mOutputs.getDevicePtr()));
#else
        CHECK_CUDNN_STATUS(cudnnAddTensor(CudaContext::cudnnHandle(),
                                          CUDNN_ADD_SAME_C,
                                          &alpha,
                                          mBias->getCudnnTensorDesc(),
                                          mBias->getDevicePtr(),
                                          &alpha,
                                          mOutputs.getCudnnTensorDesc(),
                                          mOutputs.getDevicePtr()));
#endif
    }

    Cell_Frame_CUDA::propagate();
    mDiffInputs.clearValid();
}

void N2D2::ConvCell_Frame_CUDA::backPropagate()
{
    Cell_Frame_CUDA::backPropagate();

    const float alpha = 1.0f;
    const Float_T alphaMask = 0.0f;

    for (unsigned int k = 0, size = mInputs.size(); k < size; ++k) {
        const float beta = (mWeightsSolvers[k]->isNewIteration()) ? 0.0f : 1.0f;

#if CUDNN_VERSION >= 5000
        CHECK_CUDNN_STATUS(cudnnConvolutionBackwardFilter(
            CudaContext::cudnnHandle(),
            &alpha,
            mInputs[k].getCudnnTensorDesc(),
            mInputs[k].getDevicePtr(),
            mDiffInputs.getCudnnTensorDesc(),
            mDiffInputs.getDevicePtr(),
            mConvDesc,
            mBwdFilterAlgo[k],
            mWorkspace,
            mWorkspaceSize,
            &beta,
            mFilterDesc[k],
            mDiffSharedSynapses[k].getDevicePtr()));
#else
        CHECK_CUDNN_STATUS(cudnnConvolutionBackwardFilter(
            CudaContext::cudnnHandle(),
            &alpha,
            mInputs[k].getCudnnTensorDesc(),
            mInputs[k].getDevicePtr(),
            mDiffInputs.getCudnnTensorDesc(),
            mDiffInputs.getDevicePtr(),
            mConvDesc,
            &beta,
            mFilterDesc[k],
            mDiffSharedSynapses[k].getDevicePtr()));
#endif

        if (!isFullMap()) {
            // Set the non-connected kernels diff to 0
            unsigned int offset = 0;

            for (unsigned int output = 0; output < mNbOutputs; ++output) {
                for (unsigned int channel = 0; channel < mInputs[k].dimZ();
                     ++channel) {
                    if (!isConnection(channel, output)) {
                        CHECK_CUBLAS_STATUS(cublasSscal(
                            CudaContext::cublasHandle(),
                            mKernelWidth * mKernelHeight, // size of data
                            &alphaMask,
                            mDiffSharedSynapses[k].getDevicePtr() + offset,
                            1));
                    }

                    offset += mKernelWidth * mKernelHeight;
                }
            }
        }
    }

    if (!mNoBias) {
        const float beta = (mBiasSolver->isNewIteration()) ? 0.0f : 1.0f;

        CHECK_CUDNN_STATUS(
            cudnnConvolutionBackwardBias(CudaContext::cudnnHandle(),
                                         &alpha,
                                         mDiffInputs.getCudnnTensorDesc(),
                                         mDiffInputs.getDevicePtr(),
                                         &beta,
                                         mDiffBias.getCudnnTensorDesc(),
                                         mDiffBias.getDevicePtr()));
    }

    /** Si il ne s'agit pas de la première couche */
    if (!mDiffOutputs.empty() && mBackPropagate) {
        for (unsigned int k = 0, size = mInputs.size(); k < size; ++k) {
            const float beta = (mDiffOutputs[k].isValid()) ? 1.0f : 0.0f;

#if CUDNN_VERSION >= 5000
            CHECK_CUDNN_STATUS(cudnnConvolutionBackwardData(
                CudaContext::cudnnHandle(),
                &alpha,
                mFilterDesc[k],
                mSharedSynapses[k].getDevicePtr(),
                mDiffInputs.getCudnnTensorDesc(),
                mDiffInputs.getDevicePtr(),
                mConvDesc,
                mBwdDataAlgo[k],
                mWorkspace,
                mWorkspaceSize,
                &beta,
                mDiffOutputs[k].getCudnnTensorDesc(),
                mDiffOutputs[k].getDevicePtr()));
#else
            CHECK_CUDNN_STATUS(cudnnConvolutionBackwardData(
                CudaContext::cudnnHandle(),
                &alpha,
                mFilterDesc[k],
                mSharedSynapses[k].getDevicePtr(),
                mDiffInputs.getCudnnTensorDesc(),
                mDiffInputs.getDevicePtr(),
                mConvDesc,
                &beta,
                mDiffOutputs[k].getCudnnTensorDesc(),
                mDiffOutputs[k].getDevicePtr()));
#endif
            mDiffOutputs[k].setValid();
        }

        mDiffOutputs.synchronizeDToHBased();
    }
}

void N2D2::ConvCell_Frame_CUDA::update()
{

    for (unsigned int k = 0, size = mSharedSynapses.size(); k < size; ++k)
        mWeightsSolvers[k]->update(
            &mSharedSynapses[k], &mDiffSharedSynapses[k], mInputs.dimB());

    if (!mNoBias)
        mBiasSolver->update(&(*mBias), &mDiffBias, mInputs.dimB());
}

void N2D2::ConvCell_Frame_CUDA::setWeights(unsigned int k,
                                           Interface<Float_T>* weights,
                                           unsigned int offset)
{
    CudaInterface<Float_T>* cudaWeights
        = dynamic_cast<CudaInterface<Float_T>*>(weights);

    if (cudaWeights == NULL) {
        throw std::runtime_error("ConvCell_Frame_CUDA::setWeights(): weights"
                                 " must be a CudaInterface");
    }

    mExtSharedSynapses[k] = std::make_pair(cudaWeights, offset);
}

void N2D2::ConvCell_Frame_CUDA::setBiases(
    const std::shared_ptr<Tensor4d<Float_T> >& biases)
{
    std::shared_ptr<CudaTensor4d<Float_T> > cudaBiases
        = std::dynamic_pointer_cast<CudaTensor4d<Float_T> >(biases);

    if (!cudaBiases) {
        throw std::runtime_error("ConvCell_Frame_CUDA::setBiases(): biases"
                                 " must be a CudaTensor4d");
    }

    mBias = cudaBiases;
}

void N2D2::ConvCell_Frame_CUDA::checkGradient(double epsilon, double maxError)
{
    GradientCheck gc(epsilon, maxError);
    gc.initialize(mInputs,
                  mOutputs,
                  mDiffInputs,
                  std::bind(&ConvCell_Frame_CUDA::propagate, this, false),
                  std::bind(&ConvCell_Frame_CUDA::backPropagate, this));

    for (unsigned int k = 0, size = mSharedSynapses.size(); k < size; ++k) {
        std::stringstream name;
        name << mName + "_mDiffSharedSynapses[" << k << "]";

        gc.check(name.str(), mSharedSynapses[k], mDiffSharedSynapses[k]);
    }

    if (!mNoBias)
        gc.check(mName + "_mDiffBias", (*mBias), mDiffBias);

    if (!mDiffOutputs.empty()) {
        for (unsigned int k = 0; k < mInputs.size(); ++k) {
            std::stringstream name;
            name << mName + "_mDiffOutputs[" << k << "]";

            gc.check(name.str(), mInputs[k], mDiffOutputs[k]);
        }
    } else {
        std::cout << Utils::cwarning << "Empty diff. outputs for cell " << mName
                  << ", could not check the gradient!" << Utils::cdef
                  << std::endl;
    }
}

void N2D2::ConvCell_Frame_CUDA::logFreeParameters(const std::string& fileName,
                                                  unsigned int output,
                                                  unsigned int channel) const
{
    for (unsigned int i = 0; i < mInputs.size(); ++i)
        mSharedSynapses[i].synchronizeDToH();

    mBias->synchronizeDToH();

    mSynchronized = true;
    ConvCell::logFreeParameters(fileName, output, channel);
    mSynchronized = false;
}

void N2D2::ConvCell_Frame_CUDA::logFreeParameters(const std::string& fileName,
                                                  unsigned int output) const
{
    for (unsigned int i = 0; i < mInputs.size(); ++i)
        mSharedSynapses[i].synchronizeDToH();

    mBias->synchronizeDToH();

    mSynchronized = true;
    ConvCell::logFreeParameters(fileName, output);
    mSynchronized = false;
}

void N2D2::ConvCell_Frame_CUDA::logFreeParameters(const std::string
                                                  & dirName) const
{
    for (unsigned int i = 0; i < mInputs.size(); ++i)
        mSharedSynapses[i].synchronizeDToH();

    mBias->synchronizeDToH();

    mSynchronized = true;
    ConvCell::logFreeParameters(dirName);
    mSynchronized = false;
}

void N2D2::ConvCell_Frame_CUDA::saveFreeParameters(const std::string
                                                   & fileName) const
{
    std::ofstream syn(fileName.c_str(), std::fstream::binary);

    if (!syn.good())
        throw std::runtime_error("Could not create synaptic file (.SYN): "
                                 + fileName);

    mSharedSynapses.synchronizeDToH();

    for (unsigned int k = 0; k < mSharedSynapses.size(); ++k) {
        for (std::vector<Float_T>::const_iterator it
             = mSharedSynapses[k].begin();
             it != mSharedSynapses[k].end();
             ++it)
            syn.write(reinterpret_cast<const char*>(&(*it)), sizeof(*it));
    }

    if (!mNoBias) {
        mBias->synchronizeDToH();

        for (std::vector<Float_T>::const_iterator it = mBias->begin();
             it != mBias->end();
             ++it)
            syn.write(reinterpret_cast<const char*>(&(*it)), sizeof(*it));
    }

    if (!syn.good())
        throw std::runtime_error("Error writing synaptic file: " + fileName);
}

void N2D2::ConvCell_Frame_CUDA::loadFreeParameters(const std::string& fileName,
                                                   bool ignoreNotExists)
{
    std::ifstream syn(fileName.c_str(), std::fstream::binary);

    if (!syn.good()) {
        if (ignoreNotExists) {
            std::cout << Utils::cnotice
                      << "Notice: Could not open synaptic file (.SYN): "
                      << fileName << Utils::cdef << std::endl;
            return;
        } else
            throw std::runtime_error("Could not open synaptic file (.SYN): "
                                     + fileName);
    }

    for (unsigned int k = 0; k < mSharedSynapses.size(); ++k) {
        for (std::vector<Float_T>::iterator it = mSharedSynapses[k].begin();
             it != mSharedSynapses[k].end();
             ++it)
            syn.read(reinterpret_cast<char*>(&(*it)), sizeof(*it));
    }

    mSharedSynapses.synchronizeHToD();

    if (!mNoBias) {
        for (std::vector<Float_T>::iterator it = mBias->begin();
             it != mBias->end();
             ++it)
            syn.read(reinterpret_cast<char*>(&(*it)), sizeof(*it));
    }

    mBias->synchronizeHToD();

    if (syn.eof())
        throw std::runtime_error(
            "End-of-file reached prematurely in synaptic file (.SYN): "
            + fileName);
    else if (!syn.good())
        throw std::runtime_error("Error while reading synaptic file (.SYN): "
                                 + fileName);
    else if (syn.get() != std::fstream::traits_type::eof())
        throw std::runtime_error(
            "Synaptic file (.SYN) size larger than expected: " + fileName);
}

void N2D2::ConvCell_Frame_CUDA::exportFreeParameters(const std::string
                                                     & fileName) const
{
    for (unsigned int i = 0; i < mInputs.size(); ++i)
        mSharedSynapses[i].synchronizeDToH();

    mBias->synchronizeDToH();

    mSynchronized = true;
    ConvCell::exportFreeParameters(fileName);
    mSynchronized = false;
}

void N2D2::ConvCell_Frame_CUDA::exportSolverParameters(const std::string
                                                       & fileName) const
{
    for (unsigned int i = 0; i < mSharedSynapses.size(); ++i)
        mWeightsSolvers[i]->exportFreeParameters(fileName);
}

void N2D2::ConvCell_Frame_CUDA::importFreeParameters(const std::string
                                                     & fileName,
                                                     bool ignoreNotExists)
{
    mSynchronized = true;
    ConvCell::importFreeParameters(fileName, ignoreNotExists);
    mSynchronized = false;

    for (unsigned int i = 0; i < mInputs.size(); ++i)
        mSharedSynapses[i].synchronizeHToD();

    mBias->synchronizeHToD();
}

void N2D2::ConvCell_Frame_CUDA::logFreeParametersDistrib(const std::string
                                                         & fileName) const
{
    for (unsigned int i = 0; i < mInputs.size(); ++i)
        mSharedSynapses[i].synchronizeDToH();
    mBias->synchronizeDToH();

    mSynchronized = true;
    ConvCell::logFreeParametersDistrib(fileName);
    mSynchronized = false;
}

void N2D2::ConvCell_Frame_CUDA::discretizeFreeParameters(unsigned int nbLevels)
{
    for (unsigned int i = 0; i < mInputs.size(); ++i)
        mSharedSynapses[i].synchronizeDToH();

    mBias->synchronizeDToH();

    mSynchronized = true;
    ConvCell::discretizeFreeParameters(nbLevels);
    mSynchronized = false;

    for (unsigned int i = 0; i < mInputs.size(); ++i)
        mSharedSynapses[i].synchronizeHToD();

    mBias->synchronizeHToD();
}

std::pair<N2D2::Float_T, N2D2::Float_T>
N2D2::ConvCell_Frame_CUDA::getFreeParametersRange() const
{
    for (unsigned int i = 0; i < mInputs.size(); ++i)
        mSharedSynapses[i].synchronizeDToH();

    mBias->synchronizeDToH();

    mSynchronized = true;
    const std::pair<Float_T, Float_T> range
        = ConvCell::getFreeParametersRange();
    mSynchronized = false;

    for (unsigned int i = 0; i < mInputs.size(); ++i)
        mSharedSynapses[i].synchronizeHToD();

    mBias->synchronizeHToD();

    return range;
}

void N2D2::ConvCell_Frame_CUDA::processFreeParameters(const std::function
                                                <double(const double&)>& func)
{
    for (unsigned int i = 0; i < mInputs.size(); ++i)
        mSharedSynapses[i].synchronizeDToH();

    mBias->synchronizeDToH();

    mSynchronized = true;
    ConvCell::processFreeParameters(func);
    mSynchronized = false;

    for (unsigned int i = 0; i < mInputs.size(); ++i)
        mSharedSynapses[i].synchronizeHToD();

    mBias->synchronizeHToD();
}

N2D2::ConvCell_Frame_CUDA::~ConvCell_Frame_CUDA()
{

    for (unsigned int k = 0, size = mSharedSynapses.size(); k < size; ++k) {
        if (mExtSharedSynapses.find(k) == mExtSharedSynapses.end())
            delete &mSharedSynapses[k];
    }

    for (unsigned int k = 0, size = mFilterDesc.size(); k < size; ++k)
        CHECK_CUDNN_STATUS(cudnnDestroyFilterDescriptor(mFilterDesc[k]));

    if (mWorkspaceSize > 0)
        CHECK_CUDA_STATUS(cudaFree(mWorkspace));

    CHECK_CUDNN_STATUS(cudnnDestroyConvolutionDescriptor(mConvDesc));
}

#endif
