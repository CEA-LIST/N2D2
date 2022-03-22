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

#ifdef CUDA

#include "Cell/Cell_Frame_CUDA.hpp"
#include "DeepNet.hpp"
#include "third_party/half.hpp"
#include "Cell/ConvCell_Frame_Kernels.hpp"
#include "Adversarial.hpp"

template <class T>
N2D2::Cell_Frame_CUDA<T>::Cell_Frame_CUDA(const DeepNet& deepNet, const std::string& name,
                                       unsigned int nbOutputs,
                                       const std::shared_ptr
                                       <Activation>& activation)
    : Cell(deepNet, name, nbOutputs),
      Cell_Frame_Top(activation),
      mKeepInSync(true)
{
    // ctor
}

template <class T>
void N2D2::Cell_Frame_CUDA<T>::save(const std::string& dirName) const
{
    Cell::save(dirName);
    Cell_Frame_Top::save(dirName);
}

template <class T>
void N2D2::Cell_Frame_CUDA<T>::load(const std::string& dirName)
{
    Cell::load(dirName);
    Cell_Frame_Top::load(dirName);
}

template <class T>
void N2D2::Cell_Frame_CUDA<T>::applyMapping(Cell* cell,const Tensor<bool>&  mapping){
    // Define input-output connections
    const unsigned int cellNbOutputs = cell->getNbOutputs();

    if (!mapping.empty() && mapping.dimY() != cellNbOutputs)
        throw std::runtime_error("Cell_Frame::addInput(): number of mapping "
                                 "rows must be equal to the number of input "
                                 "channels");

    mMapping.append((!mapping.empty())
        ? mapping
        : Tensor<bool>({getNbOutputs(), cellNbOutputs}, true));
}
template <class T>
void N2D2::Cell_Frame_CUDA<T>::addInput(StimuliProvider& /*sp*/,
                                     unsigned int /*channel*/,
                                     unsigned int /*x0*/,
                                     unsigned int /*y0*/,
                                     unsigned int /*width*/,
                                     unsigned int /*height*/,
                                     const Tensor<bool>& /*mapping*/)
{
    throw std::runtime_error("Cell_Frame_CUDA<T>::addInput(): adding a single "
                             "environment channel as input is not supported");
}

template <class T>
void N2D2::Cell_Frame_CUDA<T>::addInput(StimuliProvider& sp,
                                     unsigned int x0,
                                     unsigned int y0,
                                     unsigned int width,
                                     unsigned int height,
                                     const Tensor<bool>& mapping)
{
    if (width == 0)
        width = sp.getSizeX() - x0;
    if (height == 0)
        height = sp.getSizeY() - y0;

    if (x0 > 0 || y0 > 0 || width < sp.getSizeX() || height < sp.getSizeY())
        throw std::runtime_error("Cell_Frame_CUDA<T>::addInput(): adding a "
                                 "cropped environment channel map as input is "
                                 "not supported");

    // Define input-output sizes
    setInputsDims(sp.getSize());
    mInputs.push_back(&sp.getDataInput());

    // For some adversarial attacks, it is required to backpropagate
    // the gradiants to the inputs
    if (sp.getAdversarialAttack()->getAttackName() != Adversarial::Attack_T::None) {
        std::vector<size_t> inputsDims(mInputsDims);
        inputsDims.push_back(sp.getBatchSize());
        mDiffOutputs.push_back(new CudaTensor<T>(inputsDims), 0);
    }
    else
        mDiffOutputs.push_back(new CudaTensor<T>(), 0);

    setOutputsDims();

    if (mOutputs.empty()) {
        std::vector<size_t> outputsDims(mOutputsDims);
        outputsDims.push_back(sp.getBatchSize());

        mOutputs.resize(outputsDims);
        mDiffInputs.resize(outputsDims);
    }

    // Define input-output connections
    if (!mapping.empty() && mapping.dimY() != sp.getNbChannels()) {
        throw std::runtime_error("Cell_Frame_CUDA<T>::addInput(): number of "
                                 "mapping rows must be equal to the number of "
                                 "input"
                                 " channels");
    }

    mMapping.append((!mapping.empty())
        ? mapping
        : Tensor<bool>({getNbOutputs(), sp.getNbChannels()}, true));
}

template <class T>
void N2D2::Cell_Frame_CUDA<T>::addInput(Cell* cell, const Tensor<bool>& mapping)
{
    // Define input-output sizes
    setInputsDims(cell->getOutputsDims());

    Cell_Frame_Top* cellFrame = dynamic_cast<Cell_Frame_Top*>(cell);

    if (cellFrame != NULL) {
        mInputs.push_back(&cellFrame->getOutputs());
        mDiffOutputs.push_back(&cellFrame->getDiffInputs());
    }
    else {
        throw std::runtime_error(
            "Cell_Frame::addInput(): cannot mix Spike and Frame models");
    }

    setOutputsDims();

    if (mOutputs.empty()) {
        std::vector<size_t> outputsDims(mOutputsDims);
        outputsDims.push_back(mInputs.dimB());

        mOutputs.resize(outputsDims);
        mDiffInputs.resize(outputsDims);
    }

    applyMapping(cell, mapping);
}

template <class T>
void N2D2::Cell_Frame_CUDA<T>::addInput(Cell* cell,
                                     unsigned int x0,
                                     unsigned int y0,
                                     unsigned int width,
                                     unsigned int height)
{
    if (width == 0)
        width = cell->getOutputsWidth() - x0;
    if (height == 0)
        height = cell->getOutputsHeight() - y0;

    if (x0 > 0 || y0 > 0 || width < cell->getOutputsWidth()
        || height < cell->getOutputsHeight())
        throw std::runtime_error("Cell_Frame_CUDA<T>::addInput(): adding a "
                                 "cropped output map as input is not "
                                 "supported");

    Cell_Frame_CUDA<T>::addInput(cell);
}

template <class T>
void N2D2::Cell_Frame_CUDA<T>::addInput(BaseTensor& inputs,
                                     BaseTensor& diffOutputs)
{
    // Define input-output sizes
    std::vector<size_t> inputsDims = inputs.dims();
    inputsDims.pop_back();      // Remove batch

    setInputsDims(inputsDims);
    mInputs.push_back(&inputs);
    mDiffOutputs.push_back(&diffOutputs);

    setOutputsDims();

    if (mOutputs.empty()) {
        std::vector<size_t> outputsDims(mOutputsDims);
        outputsDims.push_back(mInputs.dimB());

        mOutputs.resize(outputsDims);
        mDiffInputs.resize(outputsDims);
    }

    mMapping.resize({getNbOutputs(), getNbChannels()}, true);
}

template <class T>
void N2D2::Cell_Frame_CUDA<T>::clearInputs() {
    mInputs.clear();
    mDiffOutputs.clear();

    mInputsDims.clear();
    mMapping.clear();
}

template <class T>
void N2D2::Cell_Frame_CUDA<T>::replaceInput(BaseTensor& oldInputs,
                                            BaseTensor& newInputs,
                                            BaseTensor& newDiffOutputs)
{
    assert(!newInputs.dims().empty());

    // Define input-output sizes
    std::vector<size_t> inputsDims = newInputs.dims();
    inputsDims.pop_back();      // Remove batch

    if (mInputs.size() > 1) {
        std::vector<size_t> oldInputsDims = oldInputs.dims();
        oldInputsDims.pop_back();      // Remove batch

        mInputsDims.back() -= oldInputsDims.back();
        setInputsDims(inputsDims);
    }
    else
        mInputsDims = inputsDims;

    bool foundOldInputs = false;
    for (unsigned int i = 0; i < mInputs.size(); ++i) {
        if (&mInputs[i] == &oldInputs) {
            foundOldInputs = true;
            mInputs.replace(i, &newInputs);

            if (!newDiffOutputs.empty()) {
                assert(i < mDiffOutputs.size());
                mDiffOutputs.replace(i, &newDiffOutputs);
            }
        }
    }

    if(!foundOldInputs) {
        throw std::runtime_error("Cell_Frame_CUDA::replaceInput(): can't"
            " replace input, the input has not been found.");
    }

    std::vector<size_t> oldOutputsDims(mOutputsDims);
    setOutputsDims();

    if (mInputs.size() > 1 && mOutputsDims != oldOutputsDims) {
        throw std::runtime_error("Cell_Frame_CUDA::replaceInput(): can't"
            " replace input, the output dimension has changed and doesn't"
            " match the other inputs!");
    }
}



// BEGIN code used exlusively in python API

/**
 * This is run every time the input dimensions have changed to relink the cell input/diffOutput tensors.
**/
template <class T>
void N2D2::Cell_Frame_CUDA<T>::clearInputTensors() {
    mInputs.clear();
    mDiffOutputs.clear();

    mInputsDims.clear();
}

/**
 * This is run every time the input dimensions have changed to reinitialized the cell output/diffInput tensors.
**/
template <class T>
void N2D2::Cell_Frame_CUDA<T>::clearOutputTensors() {
    mOutputs.clear();
    mDiffInputs.clear();

    mOutputsDims.clear();
}

/**
 * Initialized like addInputs but without initializing mapping. This is run every time the input dimensions have changed
**/
template <class T>
void N2D2::Cell_Frame_CUDA<T>::initializeDataDependent()
{
    if (mInputs.size() == 0){
         throw std::runtime_error(
            "Cell_Frame<T>::initializeDataDependent(): cell has no inputs");
    }

    setOutputsDims();

    /**
     * At the moment data tensor reinitialisation is not supported, but technically it would be possible. **/
    if (mOutputs.empty() && mDiffInputs.empty()) {
        std::vector<size_t> outputsDims(mOutputsDims);
        outputsDims.push_back(mInputs.dimB());

        mOutputs.resize(outputsDims);
        mDiffInputs.resize(outputsDims);
    }
    else {
        throw std::runtime_error(
            "Cell_Frame_CUDA<T>::initializeDataDependent(): data tensors are already initialized");
    }
}

/**
 * Link an input that has to be of the same size as the current input dimensions of the cell.
 * If the current input dimensions are empty, the input dimensions are initialized to 
 * correspond to the cell output dimensions.
**/
template <class T>
void N2D2::Cell_Frame_CUDA<T>::linkInput(Cell* cell)
{
    /*if (cell->getNbOutputs() != getNbChannels()){
        throw std::runtime_error("Cell has different number of channels than input");
    }*/

    // Define input-output sizes
    setInputsDims(cell->getOutputsDims());

    Cell_Frame_Top* cellFrame = dynamic_cast<Cell_Frame_Top*>(cell);

    if (cellFrame != NULL) {
        mInputs.push_back(&cellFrame->getOutputs());
        mDiffOutputs.push_back(&cellFrame->getDiffInputs());
    }
    else {
        throw std::runtime_error(
            "Cell_Frame<T>::linkInput(): cannot mix Spike and Frame models");
    }
}

/**
 * Link an input that has to be of the same size as the current input dimensions of the cell.
 * If the current input dimensions are empty, the input dimensions are initialized to 
 * correspond to the cell output dimensions.
**/
template <class T>
void N2D2::Cell_Frame_CUDA<T>::linkInput(StimuliProvider& sp,  
                                unsigned int x0,
                                unsigned int y0,
                                unsigned int width,
                                unsigned int height)
{
    if (width == 0)
        width = sp.getSizeX() - x0;
    if (height == 0)
        height = sp.getSizeY() - y0;

    if (x0 > 0 || y0 > 0 || width < sp.getSizeX() || height < sp.getSizeY())
        throw std::runtime_error("Cell_Frame<T>::linkInput(): adding a cropped "
                                 "environment channel map as input is not "
                                 "supported");
    
    /*if (sp.getNbChannels() != getNbChannels()){
        throw std::runtime_error("Cell has different number of channels than input");
    }*/

     // Define input-output sizes
    setInputsDims(sp.getSize());
    mInputs.push_back(&sp.getData());
    mDiffOutputs.push_back(new CudaTensor<T>(), 0);
}
// END code used exlusively in python API


template <class T>
void N2D2::Cell_Frame_CUDA<T>::exportActivationParameters(const std::string& dirName) const
{    
    if (mActivation)
        mActivation->exportParameters(dirName, mName);
}
template <class T>
void N2D2::Cell_Frame_CUDA<T>::importActivationParameters(const std::string& dirName, bool ignoreNotExists)
{    
    if (mActivation)
        mActivation->importParameters(dirName, mName, ignoreNotExists);
}

template <class T>
void N2D2::Cell_Frame_CUDA<T>::propagate(bool inference)
{
    if (mActivation)
        mActivation->propagate(*this, mOutputs, inference);
}

template <class T>
void N2D2::Cell_Frame_CUDA<T>::backPropagate()
{
    if (mActivation)
        mActivation->backPropagate(*this, mOutputs, mDiffInputs);
}

template <class T>
void N2D2::Cell_Frame_CUDA<T>::update()
{
    if (mActivation)
        mActivation->update(mInputs.dimB());
}

template <class T>
void N2D2::Cell_Frame_CUDA<T>::setOutputTarget(const Tensor<int>& targets)
{
    if (targets.dimB() != mOutputs.dimB())
        throw std::domain_error("Cell_Frame_CUDA<T>::setOutputTarget(): target "
                                "and output batch sizes don't match.");

    if (targets.dimX() != mOutputsDims[0] || targets.dimY() != mOutputsDims[1])
    {
        std::ostringstream errorStr;
        errorStr << "Cell_Frame_CUDA<T>::setOutputTarget(): wrong target "
            "size. Expected " << mOutputsDims << ", got "
            << targets.dims() << std::endl;

        throw std::domain_error(errorStr.str());
    }

    if (targets.size() / targets.dimB() == 1) {
        // Single target value per stimulus (ex: images classification)
        const unsigned int outputSize = mOutputs.size() / mOutputs.dimB();

        Tensor<T> diffInputs(mDiffInputs.dims());

        for (unsigned int batchPos = 0; batchPos < mOutputs.dimB(); ++batchPos)
        {
            if (targets(0, batchPos) >= 0) {
                if ((outputSize > 1 && targets(0, batchPos) >= (int)outputSize)
                    || (outputSize == 1 && (targets(0, batchPos) < 0
                                            || targets(0, batchPos) > 1)))
                {
                    throw std::domain_error("Cell_Frame_CUDA<T>"
                        "::setOutputTarget(): target not within output range.");
                }
            }

            for (unsigned int index = 0; index < outputSize; ++index) {
                if (targets(0, batchPos) >= 0) {
                    diffInputs(index, batchPos)
                        = ((outputSize > 1
                            && (int)index == targets(0, batchPos))
                        || (outputSize == 1 && targets(0, batchPos) == 1))
                            ? 1.0
                            : -1.0;
                }
                else
                    diffInputs(index, batchPos) = 0.0;
            }
        }

        mDiffInputs.synchronizeToD(diffInputs);
    }
    else {
        // 2D target matrix per stimulus (ex: images segmentation)
        if (mTargets.empty() || mNbTargetOutputs.empty()) {
#pragma omp critical(Cell_Frame_CUDA__setOutputTarget)
            if (mTargets.empty() || mNbTargetOutputs.empty()) {
                mTargets.resize(targets.dims());
                mNbTargetOutputs.resize(
                    {(getNbOutputs() > 1) ? getNbOutputs() : 2, mOutputs.dimB()});
            }
        }

        mTargets.synchronizeToD(targets);
        mNbTargetOutputs.deviceTensor().fill(0U);

        cudaPopulateNbTargetOutputs(CudaContext::getDeviceProp(),
                                    mTargets.getDevicePtr(),
                                    mNbTargetOutputs.getDevicePtr(),
                                    getNbOutputs(),
                                    mTargets.dimY(),
                                    mTargets.dimX(),
                                    mTargets.dimB());

        cudaSetOutputTargets<T>(CudaContext::getDeviceProp(),
                                    mTargets.getDevicePtr(),
                                    mNbTargetOutputs.getDevicePtr(),
                                    mDiffInputs.getDevicePtr(),
                                    mDiffInputs.dimZ(), // = getNbOutputs()
                                    mDiffInputs.dimY(),
                                    mDiffInputs.dimX(),
                                    mDiffInputs.dimB());
    }
}

template <class T>
double N2D2::Cell_Frame_CUDA<T>::applyLoss(double targetVal,
                                           double defaultVal)
{
    mLossMem.resize(mOutputs.dims());
  
    const double loss = cudaApplyLoss<T>(CudaContext::getDeviceProp(),
                                 mLossMem.getDevicePtr(),
                                 mOutputs.getDevicePtr(),
                                 mDiffInputs.getDevicePtr(),
                                 mDiffInputs.dimZ(), // = getNbOutputs()
                                 mDiffInputs.dimY(),
                                 mDiffInputs.dimX(),
                                 mDiffInputs.dimB(),
                                 T(targetVal),
                                 T(defaultVal));

    mDiffInputs.setValid();
  
    return (loss / mOutputs.dimB());
}

template <class T>
void N2D2::Cell_Frame_CUDA<T>::setOutputTargets(const BaseTensor& baseTargets)
{
    if (baseTargets.dimB() != mOutputs.dimB())
        throw std::domain_error("Cell_Frame_CUDA<T>::setOutputTargets(): target "
                                "and output batch sizes don't match.");

    if (baseTargets.dimX() != mOutputsDims[0]
        || baseTargets.dimY() != mOutputsDims[1]
        || baseTargets.dimZ() != getNbOutputs())
    {
        std::ostringstream errorStr;
        errorStr << "Cell_Frame_CUDA<T>::setOutputTargets(): wrong target "
            "matrix size. Expected " << mOutputsDims << ", got "
            << baseTargets.dims() << std::endl;

        throw std::domain_error(errorStr.str());
    }

    const Tensor<T>& targets = tensor_cast<T>(baseTargets);
    mDiffInputs.synchronizeToD(targets);
}

template <class T>
double N2D2::Cell_Frame_CUDA<T>::applyLoss()
{
    Tensor<T> outputs(mOutputs.dims());
    Tensor<T> diffInputs(mDiffInputs.dims());

    mOutputs.synchronizeToH(outputs);
    mDiffInputs.synchronizeToH(diffInputs);

    const double loss = Cell_Frame_Top::applyLoss<T>(diffInputs, outputs);

    mDiffInputs.setValid();
    mDiffInputs.synchronizeToD(diffInputs);
    return loss;
}

template <class T>
double N2D2::Cell_Frame_CUDA<T>::applyLossDistribWeighted(
    unsigned int quantSteps,
    double rangeMin,
    double rangeMax)
{
    Tensor<T> outputs(mOutputs.dims());
    Tensor<T> diffInputs(mDiffInputs.dims());

    mOutputs.synchronizeToH(outputs);
    mDiffInputs.synchronizeToH(diffInputs);

    const double loss = Cell_Frame_Top::applyLossDistribWeighted<T>(
        diffInputs, outputs,
        quantSteps, rangeMin, rangeMax);

    mDiffInputs.setValid();
    mDiffInputs.synchronizeToD(diffInputs);
    return loss;
}

template <class T>
double N2D2::Cell_Frame_CUDA<T>::applyLossThroughKernel(
    const BaseTensor& baseKernel,
    std::function<double()> lossFunc)
{
    const Tensor<T>& kernel = tensor_cast<T>(baseKernel);
    CudaTensor<T> cudaKernel(kernel.dims());
    cudaKernel.synchronizeToD(kernel);

    const int paddings[2] = {((int)kernel.dimY() - 1) / 2,
                           ((int)kernel.dimX() - 1) / 2};
    const int strides[2] = {1, 1};
    const int upscales[2] = {1, 1};

    cudnnConvolutionDescriptor_t convDesc;
    CHECK_CUDNN_STATUS(cudnnCreateConvolutionDescriptor(&convDesc));
    CHECK_CUDNN_STATUS(
        cudnnSetConvolutionNdDescriptor(convDesc,
                                        2,
                                        &paddings[0],
                                        &strides[0],
                                        &upscales[0],
                                        CUDNN_CROSS_CORRELATION,
                                        CudaContext::data_type<T>::value));

    const std::vector<int> kernels(cudaKernel.dims().rbegin(),
                                   cudaKernel.dims().rend());

    cudnnFilterDescriptor_t filterDesc;
    CHECK_CUDNN_STATUS(cudnnCreateFilterDescriptor(&filterDesc));
#if CUDNN_VERSION >= 5000
    CHECK_CUDNN_STATUS(cudnnSetFilterNdDescriptor(filterDesc,
                                                    CudaContext::data_type<T>::value,
                                                    CUDNN_TENSOR_NCHW,
                                                    4,
                                                    &kernels[0]));
#else
    CHECK_CUDNN_STATUS(cudnnSetFilterNdDescriptor(filterDesc,
                                                    CudaContext::data_type<T>::value,
                                                    4,
                                                    &kernels[0]));
#endif
    std::vector<cudnnConvolutionFwdAlgo_t> fwdAlgo_cudnn;

#if CUDNN_VERSION >= 7000
        int maxAlgoIterations = 0;
        cudnnGetConvolutionForwardAlgorithmMaxCount(CudaContext::cudnnHandle(),
                                                    &maxAlgoIterations);
        if (maxAlgoIterations == 0)
            throw std::runtime_error("No available CUDNN ConvolutionForwardAlgorithm for cell  " + mName);

        int returnAlgoCounts = 0;

        std::vector<cudnnConvolutionFwdAlgoPerf_t> returnFwdAlgo(maxAlgoIterations);
/**************************************************************************************************************
https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnFindConvolutionForwardAlgorithm
This function attempts all cuDNN algorithms (including CUDNN_TENSOR_OP_MATH and CUDNN_DEFAULT_MATH
versions of algorithms where CUDNN_TENSOR_OP_MATH may be available) for cudnnConvolutionForward(),
using memory allocated via cudaMalloc(), and outputs performance metrics to a user-allocated array
of cudnnConvolutionFwdAlgoPerf_t. These metrics are written in sorted fashion where the first element
has the lowest compute time. The total number of resulting algorithms can be queried through
the API cudnnGetConvolutionForwardMaxCount().
***************************************************************************************************************/

        CHECK_CUDNN_STATUS(cudnnFindConvolutionForwardAlgorithm(
                            CudaContext::cudnnHandle(),
                            mOutputs.getCudnnTensorDesc(),
                            filterDesc,
                            convDesc,
                            mOutputs.getCudnnTensorDesc(),
                            maxAlgoIterations,
                            &returnAlgoCounts,
                            &returnFwdAlgo[0]));
        // std::cout << "Layer " << mName << "(" << k  << ")"
        //     << " cuDNN forward algorithm heuristic results: " << std::endl;

        for(unsigned int fwdAlgo = 0; fwdAlgo < (unsigned int) maxAlgoIterations; ++fwdAlgo)
        {


            std::string algoName
                                = (returnFwdAlgo[fwdAlgo].algo
                                        == CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM)
                                    ? "CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM"
                                : (returnFwdAlgo[fwdAlgo].algo
                                        == CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM)
                                    ? "CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM"
                                : (returnFwdAlgo[fwdAlgo].algo
                                        == CUDNN_CONVOLUTION_FWD_ALGO_GEMM)
                                    ? "CUDNN_CONVOLUTION_FWD_ALGO_GEMM"
                                : (returnFwdAlgo[fwdAlgo].algo
                                        == CUDNN_CONVOLUTION_FWD_ALGO_DIRECT)
                                    ? "CUDNN_CONVOLUTION_FWD_ALGO_DIRECT"
                                : (returnFwdAlgo[fwdAlgo].algo
                                        == CUDNN_CONVOLUTION_FWD_ALGO_FFT)
                                    ? "CUDNN_CONVOLUTION_FWD_ALGO_FFT"
                                : (returnFwdAlgo[fwdAlgo].algo
                                        == CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING)
                                    ? "CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING"
                                : (returnFwdAlgo[fwdAlgo].algo
                                        == CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD)
                                    ? "CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD"
                                : (returnFwdAlgo[fwdAlgo].algo
                                        == CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED)
                                    ? "CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED"
                                : (returnFwdAlgo[fwdAlgo].algo
                                        == CUDNN_CONVOLUTION_FWD_ALGO_COUNT)
                                    ? "CUDNN_CONVOLUTION_FWD_ALGO_COUNT"
                                : "Undetermined Algorithm";


            // std::cout << "----> Forward convolution algorithm: " << algoName
            //     << " [" << returnFwdAlgo[fwdAlgo].time << " ms][" << returnFwdAlgo[fwdAlgo].memory / 1.0e6 << " MB]"
            //     << std::endl;
        }
        fwdAlgo_cudnn.push_back(returnFwdAlgo[0].algo);
#else

    fwdAlgo_cudnn.push_back(cudnnConvolutionFwdAlgo_t());


        CHECK_CUDNN_STATUS(cudnnGetConvolutionForwardAlgorithm(
            CudaContext::cudnnHandle(),
            mOutputs.getCudnnTensorDesc(),
            filterDesc,
            convDesc,
            mOutputs.getCudnnTensorDesc(),
            CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
            0,
            &fwdAlgo_cudnn.back()));
#endif

    size_t workspaceSize = 0;
    void* workspace;

    CHECK_CUDNN_STATUS(cudnnGetConvolutionForwardWorkspaceSize(
        CudaContext::cudnnHandle(),
        mOutputs.getCudnnTensorDesc(),
        filterDesc,
        convDesc,
        mOutputs.getCudnnTensorDesc(),
        fwdAlgo_cudnn.back(),
        &workspaceSize));

#if CUDNN_VERSION >= 5000
    cudnnConvolutionBwdDataAlgo_t bwdAlgo = cudnnConvolutionBwdDataAlgo_t();
    size_t bwdWorkspaceSize = 0;

    CHECK_CUDNN_STATUS(cudnnGetConvolutionBackwardDataWorkspaceSize(
        CudaContext::cudnnHandle(),
        // same arguments as cudnnGetConvolutionBackwardDataAlgorithm() -->
        filterDesc,
        mOutputs.getCudnnTensorDesc(),
        convDesc,
        mOutputs.getCudnnTensorDesc(),
        // <--
        bwdAlgo,
        &bwdWorkspaceSize));

    if (bwdWorkspaceSize > workspaceSize)
        workspaceSize = bwdWorkspaceSize;
#endif

    if (workspaceSize > 0)
        CHECK_CUDA_STATUS(cudaMalloc(&workspace, workspaceSize));

    const T alpha = T(1.0);
    const T beta = T(0.0);

    CudaTensor<T> outputs(mOutputs.dims());
    CudaTensor<T> diffInputs(mDiffInputs.dims());

    mOutputs.swap(outputs);
    mDiffInputs.swap(diffInputs);

    CHECK_CUDNN_STATUS(
        cudnnConvolutionForward(CudaContext::cudnnHandle(),
                                &alpha,
                                outputs.getCudnnTensorDesc(),
                                outputs.getDevicePtr(),
                                filterDesc,
                                cudaKernel.getDevicePtr(),
                                convDesc,
                                fwdAlgo_cudnn.back(),
                                workspace,
                                workspaceSize,
                                &beta,
                                mOutputs.getCudnnTensorDesc(),
                                mOutputs.getDevicePtr()));

    CHECK_CUDNN_STATUS(
        cudnnConvolutionForward(CudaContext::cudnnHandle(),
                                &alpha,
                                diffInputs.getCudnnTensorDesc(),
                                diffInputs.getDevicePtr(),
                                filterDesc,
                                cudaKernel.getDevicePtr(),
                                convDesc,
                                fwdAlgo_cudnn.back(),
                                workspace,
                                workspaceSize,
                                &beta,
                                mDiffInputs.getCudnnTensorDesc(),
                                mDiffInputs.getDevicePtr()));

    // Workaround for multi-GPU: to be refactored/optimized
    double loss;

#pragma omp critical
{
    mOutputs.synchronizeDToH();
    mDiffInputs.synchronizeDToH();

    loss = lossFunc();

    mDiffInputs.synchronizeHToD();
}

#if CUDNN_VERSION >= 5000
    CHECK_CUDNN_STATUS(cudnnConvolutionBackwardData(
        CudaContext::cudnnHandle(),
        &alpha,
        filterDesc,
        cudaKernel.getDevicePtr(),
        mDiffInputs.getCudnnTensorDesc(),
        mDiffInputs.getDevicePtr(),
        convDesc,
        bwdAlgo,
        workspace,
        workspaceSize,
        &beta,
        diffInputs.getCudnnTensorDesc(),
        diffInputs.getDevicePtr()));
#else
    CHECK_CUDNN_STATUS(cudnnConvolutionBackwardData(
        CudaContext::cudnnHandle(),
        &alpha,
        filterDesc,
        cudaKernel.getDevicePtr(),
        mDiffInputs.getCudnnTensorDesc(),
        mDiffInputs.getDevicePtr(),
        convDesc,
        &beta,
        diffInputs.getCudnnTensorDesc(),
        diffInputs.getDevicePtr()));
#endif

    diffInputs.swap(mDiffInputs);
    outputs.swap(mOutputs);

    mDiffInputs.setValid();

    // Clean-up CuDNN
    cudnnDestroyFilterDescriptor(filterDesc);
    cudnnDestroyConvolutionDescriptor(convDesc);

    if (workspaceSize > 0)
        cudaFree(workspace);

    return loss;
}

template <class T>
void N2D2::Cell_Frame_CUDA<T>::setOutputErrors(const BaseTensor& baseErrors)
{
    if (baseErrors.dimB() != mOutputs.dimB())
        throw std::domain_error("Cell_Frame_CUDA<T>::setOutputErrors(): target "
                                "and output batch sizes don't match.");

    if (baseErrors.dimX() != mOutputsDims[0]
        || baseErrors.dimY() != mOutputsDims[1]
        || baseErrors.dimZ() != getNbOutputs())
    {
        std::ostringstream errorStr;
        errorStr << "Cell_Frame_CUDA<T>::setOutputErrors(): wrong target "
            "matrix size. Expected " << mOutputsDims << ", got "
            << baseErrors.dims() << std::endl;

        throw std::domain_error(errorStr.str());
    }

    const Tensor<T>& errors = tensor_cast<T>(baseErrors);

    for (unsigned int index = 0; index < mOutputs.size(); ++index)
        mDiffInputs(index) = errors(index);

    mDiffInputs.setValid();
    mDiffInputs.synchronizeHToD();
}

template <class T>
N2D2::BaseTensor& N2D2::Cell_Frame_CUDA<T>::getInputs(unsigned int index)
{
    return mInputs[index];
}

template <class T>
const N2D2::BaseTensor&
N2D2::Cell_Frame_CUDA<T>::getInputs(unsigned int index) const
{
    return mInputs[index];
}

template <class T>
N2D2::BaseTensor& N2D2::Cell_Frame_CUDA<T>::getOutputs()
{
    return mOutputs;
}

template <class T>
const N2D2::BaseTensor& N2D2::Cell_Frame_CUDA<T>::getOutputs() const
{
    return mOutputs;
}

template <class T>
void N2D2::Cell_Frame_CUDA<T>::setDiffInputs(N2D2::BaseTensor& diffInputs)
{
    mDiffInputs = diffInputs;
}

template <class T>
void N2D2::Cell_Frame_CUDA<T>::setDiffInputsValid()
{
    mDiffInputs.setValid();        
}



template <class T>
N2D2::BaseTensor& N2D2::Cell_Frame_CUDA<T>::getDiffInputs()
{
    return mDiffInputs;
}

template <class T>
const N2D2::BaseTensor&
N2D2::Cell_Frame_CUDA<T>::getDiffInputs() const
{
    return mDiffInputs;
}

template <class T>
N2D2::BaseTensor& N2D2::Cell_Frame_CUDA<T>::getDiffOutputs(unsigned int index)
{
    return mDiffOutputs[index];
}

template <class T>
const N2D2::BaseTensor&
N2D2::Cell_Frame_CUDA<T>::getDiffOutputs(unsigned int index) const
{
    return mDiffOutputs[index];
}

template <class T>
unsigned int N2D2::Cell_Frame_CUDA<T>::getMaxOutput(unsigned int batchPos) const
{
    mOutputs.synchronizeDToH();
    const Tensor<T> output = mOutputs[batchPos];
    return std::distance(output.begin(),
                         std::max_element(output.begin(), output.end()));
}

template <class T>
void N2D2::Cell_Frame_CUDA<T>::keepInSync(bool keepInSync_) const
{
    mKeepInSync = keepInSync_;
}

template <class T>
N2D2::Cell_Frame_CUDA<T>::~Cell_Frame_CUDA()
{
    // dtor
}

template <class T>
std::string N2D2::Cell_Frame_CUDA<T>::getPyDataType()
{
    if (typeid(T) == typeid(double)) {
        return std::string("double");
    }
    else if (typeid(T) == typeid(float)) {
        return std::string("float");
    }
    else if (typeid(T) == typeid(half_float::half)) {
        return std::string("half_float");
    }
    else {
        // This case should not happen
        // If it happens, the type string returned by this function
        // might be weird
        return std::string(typeid(T).name());
    }
}

template <class T>
std::string N2D2::Cell_Frame_CUDA<T>::getPyModel()
{
    return std::string(N2D2::Cell_Frame_Top::FRAME_CUDA_TYPE);
}

namespace N2D2 {
    template class Cell_Frame_CUDA<half_float::half>;
    template class Cell_Frame_CUDA<float>;
    template class Cell_Frame_CUDA<double>;
}

#endif
