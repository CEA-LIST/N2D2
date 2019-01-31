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

#include "Filler/Filler.hpp"
#include "Filler/NormalFiller.hpp"
#include "GradientCheck.hpp"
#include "Cell/DeconvCell_Frame_CUDA.hpp"
#include "third_party/half.hpp"

template <>
N2D2::Registrar<N2D2::DeconvCell>
N2D2::DeconvCell_Frame_CUDA<half_float::half>::mRegistrar("Frame_CUDA",
        N2D2::DeconvCell_Frame_CUDA<half_float::half>::create,
        N2D2::Registrar<N2D2::DeconvCell>::Type<half_float::half>());

template <>
N2D2::Registrar<N2D2::DeconvCell>
N2D2::DeconvCell_Frame_CUDA<float>::mRegistrar("Frame_CUDA",
        N2D2::DeconvCell_Frame_CUDA<float>::create,
        N2D2::Registrar<N2D2::DeconvCell>::Type<float>());

template <>
N2D2::Registrar<N2D2::DeconvCell>
N2D2::DeconvCell_Frame_CUDA<double>::mRegistrar("Frame_CUDA",
        N2D2::DeconvCell_Frame_CUDA<double>::create,
        N2D2::Registrar<N2D2::DeconvCell>::Type<double>());

template <class T>
N2D2::DeconvCell_Frame_CUDA<T>::DeconvCell_Frame_CUDA(
    const std::string& name,
    const std::vector<unsigned int>& kernelDims,
    unsigned int nbOutputs,
    const std::vector<unsigned int>& strideDims,
    const std::vector<int>& paddingDims,
    const std::vector<unsigned int>& dilationDims,
    const std::shared_ptr<Activation>& activation)
    : Cell(name, nbOutputs),
      DeconvCell(name,
                 kernelDims,
                 nbOutputs,
                 strideDims,
                 paddingDims,
                 dilationDims),
      Cell_Frame_CUDA<T>(name, nbOutputs, activation),
      // IMPORTANT: Do not change the value of the parameters here! Use
      // setParameter() or loadParameters().
      mBias(std::make_shared<CudaTensor<T> >()),
      mDiffBias({1, 1, getNbOutputs(), 1}),
      mWorkspaceSize(0),
      mWorkspace(NULL),
      mSynchronized(false)
{
    if (strideDims.size() != kernelDims.size()) {
        throw std::domain_error("DeconvCell_Frame_CUDA: the number of dimensions"
                                " of stride must match the number of"
                                " dimensions of the kernel.");
    }

    if (paddingDims.size() != kernelDims.size()) {
        throw std::domain_error("DeconvCell_Frame_CUDA: the number of dimensions"
                                " of padding must match the number of"
                                " dimensions of the kernel.");
    }

    if (dilationDims.size() != kernelDims.size()) {
        throw std::domain_error("DeconvCell_Frame_CUDA: the number of dimensions"
                                " of dilation must match the number of"
                                " dimensions of the kernel.");
    }

    mWeightsFiller = std::make_shared<NormalFiller<T> >(0.0, 0.05);
    mBiasFiller = std::make_shared<NormalFiller<T> >(0.0, 0.05);
    mWeightsSolver = std::make_shared<SGDSolver_Frame_CUDA<T> >();
    mBiasSolver = std::make_shared<SGDSolver_Frame_CUDA<T> >();

    CHECK_CUDNN_STATUS(cudnnCreateConvolutionDescriptor(&mConvDesc));
}

template <class T>
void N2D2::DeconvCell_Frame_CUDA<T>::initialize()
{
    if (!mNoBias) {
        if (mBias->empty()) {
            mBias->resize({1, 1, getNbOutputs(), 1});
            mBiasFiller->apply((*mBias));
            mBias->synchronizeHToD();
        }
        else {
            if (mBias->dimX() != 1 || mBias->dimY() != 1
                || mBias->dimZ() != getNbOutputs() || mBias->dimB() != 1)
            {
                throw std::runtime_error("DeconvCell_Frame_CUDA<T>::initialize():"
                    " in cell " + mName + ", wrong size for shared bias");
            }
        }
    }

    const std::vector<int> strides(mStrideDims.rbegin(), mStrideDims.rend());
    const std::vector<int> paddings(mPaddingDims.rbegin(), mPaddingDims.rend());
    const std::vector<int> upscales(mDilationDims.rbegin(),
                                    mDilationDims.rend());

    CHECK_CUDNN_STATUS(
        cudnnSetConvolutionNdDescriptor(mConvDesc,
                                        mKernelDims.size(),
                                        &paddings[0],
                                        &strides[0],
                                        &upscales[0],
                                        CUDNN_CROSS_CORRELATION,
                                        CudaContext::data_type<T>::value));

    size_t workspaceSize = 0;
    unsigned int nbChannels = 0;

    for (unsigned int k = 0, size = mInputs.size(); k < size; ++k) {
        if (mInputs[k].size() == 0) {
            throw std::runtime_error("Zero-sized input for DeconvCell "
                                     + mName);
        }

        mNbGroups.push_back(getNbGroups(mMapping.rows(nbChannels,
                                                   mInputs[k].dimZ())));

        mWeightsSolvers.push_back(mWeightsSolver->clone());

        typename std::map<unsigned int,
            std::pair<CudaInterface<T>*, unsigned int> >::iterator
                it = mExtSharedSynapses.find(k);

        std::vector<size_t> kernelDims(mKernelDims.begin(), mKernelDims.end());

#if CUDNN_VERSION >= 7000
        if (mNbGroups[k] > 1)
            kernelDims.push_back(getNbOutputs() / mNbGroups[k]);
        else
            kernelDims.push_back(getNbOutputs());
#endif

        kernelDims.push_back(mInputs[k].dimZ());

        if (it != mExtSharedSynapses.end()) {
            CudaTensor<T>* extWeights
                = &((*(*it).second.first)[(*it).second.second]);

            if (!std::equal(kernelDims.begin(), kernelDims.end(),
                            extWeights->dims().begin()))
            {
                std::stringstream errorStr;
                errorStr << "DeconvCell_Frame_CUDA<T>::initialize(): in cell "
                    << mName << ", mismatch between external weights dim. ("
                    << extWeights->dims() << ") and expected dim. ("
                    << kernelDims << ")";

                throw std::runtime_error(errorStr.str());
            }

            mSharedSynapses.push_back(extWeights);
        }
        else {
            // Weight filler expect dimZ as input and dimB as output
            std::vector<size_t> fillerKernelDims(kernelDims);
            std::swap(fillerKernelDims.back(),
                      fillerKernelDims[kernelDims.size() - 2]);

            CudaTensor<T>* sharedSynapses = new CudaTensor<T>(fillerKernelDims);
            mWeightsFiller->apply(*sharedSynapses);
            // Inverse dimZ and dimB for Deconv
            sharedSynapses->reshape(kernelDims);

            mSharedSynapses.push_back(sharedSynapses, 0);

#if CUDNN_VERSION >= 7000
            if (mNbGroups[k] > 1)
                cudnnSetConvolutionGroupCount(mConvDesc, mNbGroups[k]);
            else
#endif
            if (mNbGroups[k] == 0) {
                // Set the non-connected kernels coefficients to 0
                for (unsigned int output = 0; output < getNbOutputs(); ++output) {
                    for (unsigned int channel = 0; channel < mInputs[k].dimZ();
                         ++channel) {
                        if (!isConnection(nbChannels + channel, output))
                            mSharedSynapses.back()[channel][output] = Tensor
                                <T>(mKernelDims, T(0.0));
                    }
                }
            }

            mSharedSynapses.back().synchronizeHToD();
        }

        mDiffSharedSynapses.push_back(new CudaTensor<T>(kernelDims), 0);

        mFilterDesc.push_back(cudnnFilterDescriptor_t());

        const std::vector<int> cudaKernelDims(kernelDims.rbegin(),
                                              kernelDims.rend());

        CHECK_CUDNN_STATUS(cudnnCreateFilterDescriptor(&mFilterDesc.back()));
#if CUDNN_VERSION >= 5000
        CHECK_CUDNN_STATUS(cudnnSetFilterNdDescriptor(mFilterDesc.back(),
                                                      CudaContext::data_type<T>::value,
                                                      CUDNN_TENSOR_NCHW,
                                                      cudaKernelDims.size(),
                                                      &cudaKernelDims[0]));
#else
        CHECK_CUDNN_STATUS(cudnnSetFilterNdDescriptor(mFilterDesc.back(),
                                                      CudaContext::data_type<T>::value,
                                                      cudaKernelDims.size(),
                                                      &cudaKernelDims[0]));
#endif

        mFwdAlgo.push_back(cudnnConvolutionFwdAlgo_t());

        // Need to cast mInputs[k] so that getCudnnTensorDesc() returns the
        // right data type. No need to actually copy any data.
        std::shared_ptr<CudaDeviceTensor<T> > input
            = cuda_device_tensor_cast_nocopy<T>(mInputs[k]);

        CHECK_CUDNN_STATUS(cudnnGetConvolutionForwardAlgorithm(
            CudaContext::cudnnHandle(),
            mOutputs.getCudnnTensorDesc(),
            mFilterDesc.back(),
            mConvDesc,
            input->getCudnnTensorDesc(),
            CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
            0,
            &mFwdAlgo.back()));

#if CUDNN_VERSION >= 5000
        mBwdFilterAlgo.push_back(cudnnConvolutionBwdFilterAlgo_t());

        CHECK_CUDNN_STATUS(cudnnGetConvolutionBackwardFilterAlgorithm(
            CudaContext::cudnnHandle(),
            mOutputs.getCudnnTensorDesc(),
            input->getCudnnTensorDesc(),
            mConvDesc,
            mFilterDesc.back(),
            CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST,
            0,
            &mBwdFilterAlgo.back()));

        mBwdDataAlgo.push_back(cudnnConvolutionBwdDataAlgo_t());

        CHECK_CUDNN_STATUS(cudnnGetConvolutionBackwardDataAlgorithm(
            CudaContext::cudnnHandle(),
            mFilterDesc.back(),
            input->getCudnnTensorDesc(),
            mConvDesc,
            mOutputs.getCudnnTensorDesc(),
            CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST,
            0,
            &mBwdDataAlgo.back()));
#endif

        CHECK_CUDNN_STATUS(cudnnGetConvolutionForwardWorkspaceSize(
            CudaContext::cudnnHandle(),
            mOutputs.getCudnnTensorDesc(),
            mFilterDesc.back(),
            mConvDesc,
            input->getCudnnTensorDesc(),
            mFwdAlgo.back(),
            &workspaceSize));

        if (workspaceSize > mWorkspaceSize)
            mWorkspaceSize = workspaceSize;

#if CUDNN_VERSION >= 5000
        CHECK_CUDNN_STATUS(cudnnGetConvolutionBackwardFilterWorkspaceSize(
            CudaContext::cudnnHandle(),
            // same arguments as cudnnGetConvolutionBackwardFilterAlgorithm()
            // -->
            mOutputs.getCudnnTensorDesc(),
            input->getCudnnTensorDesc(),
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
            input->getCudnnTensorDesc(),
            mConvDesc,
            mOutputs.getCudnnTensorDesc(),
            // <--
            mBwdDataAlgo.back(),
            &workspaceSize));
#endif

        if (workspaceSize > mWorkspaceSize)
            mWorkspaceSize = workspaceSize;

        nbChannels += mInputs[k].dimZ();
    }

    if (mWorkspaceSize > 0)
        CHECK_CUDA_STATUS(cudaMalloc(&mWorkspace, mWorkspaceSize));
}

template <class T>
void N2D2::DeconvCell_Frame_CUDA<T>::propagate(bool inference)
{
    mInputs.synchronizeHBasedToD();

    /**
     * 1.0
     * Corps de la procédure de convolution via CuDNN
     * Pour plus de détails, cf. doc : cuDNN Library
     */
    const typename Cuda::cudnn_scaling_type<T>::type alpha = 1.0f;
    typename Cuda::cudnn_scaling_type<T>::type beta = 0.0f;

    for (unsigned int k = 0, size = mInputs.size(); k < size; ++k) {
        if (k > 0)
            beta = 1.0f;

        std::shared_ptr<CudaDeviceTensor<T> > input
            = cuda_device_tensor_cast<T>(mInputs[k]);

#if CUDNN_VERSION >= 5000
        CHECK_CUDNN_STATUS(
            cudnnConvolutionBackwardData(CudaContext::cudnnHandle(),
                                         &alpha,
                                         mFilterDesc[k],
                                         mSharedSynapses[k].getDevicePtr(),
                                         input->getCudnnTensorDesc(),
                                         input->getDevicePtr(),
                                         mConvDesc,
                                         mBwdDataAlgo[k],
                                         mWorkspace,
                                         mWorkspaceSize,
                                         &beta,
                                         mOutputs.getCudnnTensorDesc(),
                                         mOutputs.getDevicePtr()));
#else
        CHECK_CUDNN_STATUS(
            cudnnConvolutionBackwardData(CudaContext::cudnnHandle(),
                                         &alpha,
                                         mFilterDesc[k],
                                         mSharedSynapses[k].getDevicePtr(),
                                         input->getCudnnTensorDesc(),
                                         input->getDevicePtr(),
                                         mConvDesc,
                                         &beta,
                                         mOutputs.getCudnnTensorDesc(),
                                         mOutputs.getDevicePtr()));
#endif
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

    Cell_Frame_CUDA<T>::propagate(inference);
    mDiffInputs.clearValid();
}

template <class T>
void N2D2::DeconvCell_Frame_CUDA<T>::backPropagate()
{
    Cell_Frame_CUDA<T>::backPropagate();

    const typename Cuda::cudnn_scaling_type<T>::type alpha = 1.0f;

    const unsigned int kernelSize = (!mKernelDims.empty())
        ? std::accumulate(mKernelDims.begin(), mKernelDims.end(),
                          1U, std::multiplies<unsigned int>())
        : 0U;

    unsigned int nbChannels = 0;

    for (unsigned int k = 0, size = mInputs.size(); k < size; ++k) {
        const typename Cuda::cudnn_scaling_type<T>::type beta
            = (mWeightsSolvers[k]->isNewIteration()) ? 0.0f : 1.0f;

        std::shared_ptr<CudaDeviceTensor<T> > input
            = cuda_device_tensor_cast_nocopy<T>(mInputs[k]);

#if CUDNN_VERSION >= 5000
        CHECK_CUDNN_STATUS(cudnnConvolutionBackwardFilter(
            CudaContext::cudnnHandle(),
            &alpha,
            mDiffInputs.getCudnnTensorDesc(),
            mDiffInputs.getDevicePtr(),
            input->getCudnnTensorDesc(),
            input->getDevicePtr(),
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
            mDiffInputs.getCudnnTensorDesc(),
            mDiffInputs.getDevicePtr(),
            input->getCudnnTensorDesc(),
            input->getDevicePtr(),
            mConvDesc,
            &beta,
            mFilterDesc[k],
            mDiffSharedSynapses[k].getDevicePtr()));
#endif

#if CUDNN_VERSION >= 7000
        if (mNbGroups[k] > 1) {
            // Nothing to do!
        }
        else
#endif
        if (mNbGroups[k] == 0) {
            // Set the non-connected kernels diff to 0
            unsigned int offset = 0;

            for (unsigned int channel = 0; channel < mInputs[k].dimZ();
                ++channel)
            {
                for (unsigned int output = 0; output < getNbOutputs(); ++output)
                {
                    if (!isConnection(nbChannels + channel, output)) {
                        thrust_fill<T>(mDiffSharedSynapses[k].getDevicePtr()
                                            + offset,
                                       kernelSize,
                                       T(0.0));
                    }

                    offset += kernelSize;
                }
            }
        }

        nbChannels += mInputs[k].dimZ();
    }

    if (!mNoBias) {
        const typename Cuda::cudnn_scaling_type<T>::type beta
            = (mBiasSolver->isNewIteration()) ? 0.0f : 1.0f;

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
            const typename Cuda::cudnn_scaling_type<T>::type beta
                = (mDiffOutputs[k].isValid()) ? 1.0f : 0.0f;

            std::shared_ptr<CudaDeviceTensor<T> > diffOutput
                = (mDiffOutputs[k].isValid())
                    ? cuda_device_tensor_cast<T>(mDiffOutputs[k])
                    : cuda_device_tensor_cast_nocopy<T>(mDiffOutputs[k]);

            CHECK_CUDNN_STATUS(
                cudnnConvolutionForward(CudaContext::cudnnHandle(),
                                        &alpha,
                                        mDiffInputs.getCudnnTensorDesc(),
                                        mDiffInputs.getDevicePtr(),
                                        mFilterDesc[k],
                                        mSharedSynapses[k].getDevicePtr(),
                                        mConvDesc,
                                        mFwdAlgo[k],
                                        mWorkspace,
                                        mWorkspaceSize,
                                        &beta,
                                        diffOutput->getCudnnTensorDesc(),
                                        diffOutput->getDevicePtr()));

            mDiffOutputs[k].deviceTensor() = *diffOutput;
            mDiffOutputs[k].setValid();
        }

        mDiffOutputs.synchronizeDToHBased();
    }
}

template <class T>
void N2D2::DeconvCell_Frame_CUDA<T>::update()
{
    for (unsigned int k = 0, size = mSharedSynapses.size(); k < size; ++k)
        mWeightsSolvers[k]->update(
            mSharedSynapses[k], mDiffSharedSynapses[k], mInputs.dimB());

    if (!mNoBias)
        mBiasSolver->update(*mBias, mDiffBias, mInputs.dimB());
}

template <class T>
void N2D2::DeconvCell_Frame_CUDA<T>::setWeights(unsigned int k,
                                                BaseInterface* weights,
                                                unsigned int offset)
{
    CudaInterface<T>* cudaWeightsInterface
        = dynamic_cast<CudaInterface<T>*>(weights);

    if (!cudaWeightsInterface) {
        throw std::runtime_error("DeconvCell_Frame_CUDA<T>::setWeights(): "
                                 "incompatible types.");
    }

    mExtSharedSynapses[k] = std::make_pair(cudaWeightsInterface, offset);
}

template <class T>
void N2D2::DeconvCell_Frame_CUDA<T>::setBiases(
    const std::shared_ptr<BaseTensor>& biases)
{
    std::shared_ptr<CudaTensor<T> > cudaBiases
        = std::dynamic_pointer_cast<CudaTensor<T> >(biases);

    if (!cudaBiases) {
        throw std::runtime_error("DeconvCell_Frame_CUDA<T>::setBiases(): biases"
                                 " must be a CudaTensor");
    }

    mBias = cudaBiases;
}

template <class T>
void N2D2::DeconvCell_Frame_CUDA<T>::checkGradient(double epsilon, double maxError)
{
    GradientCheck<T> gc(epsilon, maxError);
    gc.initialize(mInputs,
                  mOutputs,
                  mDiffInputs,
                  std::bind(&DeconvCell_Frame_CUDA<T>::propagate, this, false),
                  std::bind(&DeconvCell_Frame_CUDA<T>::backPropagate, this));

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

template <class T>
void N2D2::DeconvCell_Frame_CUDA<T>::logFreeParameters(const std::string& fileName,
                                                    unsigned int output,
                                                    unsigned int channel) const
{
    mSharedSynapses.synchronizeDToH();
    mBias->synchronizeDToH();

    mSynchronized = true;
    DeconvCell::logFreeParameters(fileName, output, channel);
    mSynchronized = false;
}

template <class T>
void N2D2::DeconvCell_Frame_CUDA<T>::logFreeParameters(const std::string& fileName,
                                                    unsigned int output) const
{
    mSharedSynapses.synchronizeDToH();
    mBias->synchronizeDToH();

    mSynchronized = true;
    DeconvCell::logFreeParameters(fileName, output);
    mSynchronized = false;
}

template <class T>
void N2D2::DeconvCell_Frame_CUDA<T>::logFreeParameters(const std::string
                                                    & dirName) const
{
    mSharedSynapses.synchronizeDToH();
    mBias->synchronizeDToH();

    mSynchronized = true;
    DeconvCell::logFreeParameters(dirName);
    mSynchronized = false;
}

template <class T>
void N2D2::DeconvCell_Frame_CUDA<T>::saveFreeParameters(const std::string
                                                     & fileName) const
{
    std::ofstream syn(fileName.c_str(), std::fstream::binary);

    if (!syn.good())
        throw std::runtime_error("Could not create synaptic file (.SYN): "
                                 + fileName);

    mSharedSynapses.synchronizeDToH();

    for (unsigned int k = 0; k < mSharedSynapses.size(); ++k)
        mSharedSynapses[k].save(syn);

    if (!mNoBias) {
        mBias->synchronizeDToH();
        mBias->save(syn);
    }

    if (!syn.good())
        throw std::runtime_error("Error writing synaptic file: " + fileName);
}

template <class T>
void N2D2::DeconvCell_Frame_CUDA<T>::loadFreeParameters(const std::string
                                                     & fileName,
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

    for (unsigned int k = 0; k < mSharedSynapses.size(); ++k)
        mSharedSynapses[k].load(syn);

    mSharedSynapses.synchronizeHToD();

    if (!mNoBias) {
        mBias->load(syn);
        mBias->synchronizeHToD();
    }

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

template <class T>
void N2D2::DeconvCell_Frame_CUDA<T>::exportFreeParameters(const std::string
                                                       & fileName) const
{
    mSharedSynapses.synchronizeDToH();
    mBias->synchronizeDToH();

    mSynchronized = true;
    DeconvCell::exportFreeParameters(fileName);
    mSynchronized = false;
}

template <class T>
void N2D2::DeconvCell_Frame_CUDA<T>::importFreeParameters(const std::string
                                                       & fileName,
                                                       bool ignoreNotExists)
{
    mSynchronized = true;
    DeconvCell::importFreeParameters(fileName, ignoreNotExists);
    mSynchronized = false;

    mSharedSynapses.synchronizeHToD();
    mBias->synchronizeHToD();
}

template <class T>
void N2D2::DeconvCell_Frame_CUDA<T>::logFreeParametersDistrib(const std::string
                                                           & fileName) const
{
    mSharedSynapses.synchronizeDToH();
    mBias->synchronizeDToH();

    mSynchronized = true;
    DeconvCell::logFreeParametersDistrib(fileName);
    mSynchronized = false;
}

template <class T>
N2D2::DeconvCell_Frame_CUDA<T>::~DeconvCell_Frame_CUDA()
{
    for (unsigned int k = 0, size = mFilterDesc.size(); k < size; ++k)
        cudnnDestroyFilterDescriptor(mFilterDesc[k]);

    if (mWorkspaceSize > 0)
        cudaFree(mWorkspace);

    cudnnDestroyConvolutionDescriptor(mConvDesc);
}

namespace N2D2 {
    template class DeconvCell_Frame_CUDA<half_float::half>;
    template class DeconvCell_Frame_CUDA<float>;
    template class DeconvCell_Frame_CUDA<double>;
}

#endif
