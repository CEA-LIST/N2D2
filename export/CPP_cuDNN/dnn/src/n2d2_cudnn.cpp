/*
    (C) Copyright 2015 CEA LIST. All Rights Reserved.
    Contributor(s): Olivier BICHLER (olivier.bichler@cea.fr)
                    David BRIAND (david.briand@cea.fr)

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

#include "n2d2_cudnn.hpp"
#include <iostream>

#if CUDNN_VERSION >= 5000
cudnnNanPropagation_t NanPolicy = CUDNN_PROPAGATE_NAN;
#endif


oclHandleStruct oclHandles;

void set_profiling()
{
    oclHandles.isActivated = true;
}

void report_per_layer_profiling(unsigned int nbIter)
{
    double totalProcessTime = 0.0;
    for (std::vector<oclProfiling>::iterator it
         = oclHandles.profiling.begin(),
         itEnd = oclHandles.profiling.end();
         it != itEnd;
         ++it) {
        totalProcessTime += (*it).processTime / (nbIter);
    }

    for (std::vector<oclProfiling>::iterator it
         = oclHandles.profiling.begin(),
         itEnd = oclHandles.profiling.end();
         it != itEnd;
         ++it) {
        const double processTimeUs = (*it).processTime / (nbIter);
        const double workLoad = (processTimeUs / totalProcessTime) * 100.0;
        std::string barrelLoad(((unsigned int)workLoad + 1) * 2, '*');
        std::cout << "(" << std::setfill('0') << std::setw(2)
                  << (unsigned int)workLoad << "%)  " << barrelLoad
                  << "    " << (*it).name << ": " << processTimeUs << " us"
                  << std::endl;
     }
}

/**** Convolution Layer ****/
void setConvolution(unsigned int batchSize,
                    std::vector<int> channelsPerInputLayer,
                    std::vector<int> channelsHeightPerInputLayer,
                    std::vector<int> channelsWidthPerInputLayer,
                    unsigned int paddingY,
                    unsigned int paddingX,
                    unsigned int strideY,
                    unsigned int strideX,
                    unsigned int subSampleY,
                    unsigned int subSampleX,
                    const DATA_T* weights_flatten,
                    std::vector<DATA_T*>& weights_cudnn,
                    const DATA_T *bias_flatten,
                    DATA_T *& bias_cudnn,
                    cudnnHandle_t& context_handle,
                    cudnnTensorFormat_t context_tensorFormat,
                    cudnnDataType_t context_dataType,
                    std::vector<cudnnTensorDescriptor_t>& inputsTensor,
                    cudnnTensorDescriptor_t& outputsTensor,
                    ActivationFunction_T func,
                    std::vector<cudnnConvolutionFwdAlgo_t>& algo,
                    size_t& workSpaceSize,
                    void** workSpace,
                    unsigned int nbOutputs,
                    unsigned int outputHeight,
                    unsigned int outputWidth,
                    unsigned int kernelHeight,
                    unsigned int kernelWidth,
                    cudnnTensorDescriptor_t &biasDesc,
                    std::vector<cudnnFilterDescriptor_t>& filterDesc,
                    cudnnConvolutionDescriptor_t& convDesc)
{
    int n = batchSize;
    int k = nbOutputs;
    std::vector<int> c = channelsPerInputLayer;

    int hK = kernelHeight;
    int wK = kernelWidth;

    std::vector<int> hCh = channelsHeightPerInputLayer;
    std::vector<int> wCh = channelsWidthPerInputLayer;

    if(oclHandles.isActivated)
        oclHandles.profiling.push_back(oclProfiling({"Convolution", 0.0}));

    cudnnCreateConvolutionDescriptor(&convDesc);


#if CUDNN_VERSION >= 6000
    CHECK_CUDNN_STATUS(
        cudnnSetConvolution2dDescriptor(convDesc,
                                        paddingY,
                                        paddingX,
                                        strideY,
                                        strideX,
                                        subSampleY,
                                        subSampleX,
                                        CUDNN_CROSS_CORRELATION,
                                        context_dataType) );
#else
    CHECK_CUDNN_STATUS(
        cudnnSetConvolution2dDescriptor(convDesc,
                                        paddingY,
                                        paddingX,
                                        strideY,
                                        strideX,
                                        subSampleY,
                                        subSampleX,
                                        CUDNN_CROSS_CORRELATION) );
#endif

    cudnnCreateTensorDescriptor(&biasDesc);
    cudnnCreateTensorDescriptor(&outputsTensor);

    CHECK_CUDNN_STATUS(
        cudnnSetTensor4dDescriptor(biasDesc,
                                   context_tensorFormat,
                                   context_dataType,
                                   1,
                                   k,
                                   1,
                                   1));

    CHECK_CUDNN_STATUS(
        cudnnSetTensor4dDescriptor(outputsTensor,
                                   context_tensorFormat,
                                   context_dataType,
                                   n,
                                   k,
                                   outputHeight,
                                   outputWidth));

    for (unsigned int i = 0; i < channelsPerInputLayer.size(); ++ i) {

        filterDesc.push_back(cudnnFilterDescriptor_t());
        cudnnCreateFilterDescriptor(&filterDesc.back());

        inputsTensor.push_back(cudnnTensorDescriptor_t());
        cudnnCreateTensorDescriptor(&inputsTensor.back());

#if CUDNN_VERSION >= 5000
        CHECK_CUDNN_STATUS(
            cudnnSetFilter4dDescriptor(filterDesc.back(),
                                       context_dataType,
                                       context_tensorFormat,
                                       k,
                                       c[i],
                                       hK,
                                       wK));
#else
        CHECK_CUDNN_STATUS(
            cudnnSetFilter4dDescriptor(filterDesc.back(),
                                       context_dataType,
                                       k,
                                       c[i],
                                       hK,
                                       wK));
#endif

        CHECK_CUDNN_STATUS(
            cudnnSetTensor4dDescriptor(inputsTensor.back(),
                                       context_tensorFormat,
                                       context_dataType,
                                       n,
                                       c[i],
                                       hCh[i],
                                       wCh[i]));

        CHECK_CUDNN_STATUS(
            cudnnGetConvolution2dForwardOutputDim(convDesc,
                                                  inputsTensor.back(),
                                                  filterDesc.back(),
                                                  &n,
                                                  &c[i],
                                                  &hCh[i],
                                                  &wCh[i]));

        algo.push_back(cudnnConvolutionFwdAlgo_t());

        CHECK_CUDNN_STATUS(
            cudnnGetConvolutionForwardAlgorithm(
                context_handle,
                inputsTensor.back(),
                filterDesc.back(),
                convDesc,
                outputsTensor,
                CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                0,
                &algo.back()));

        size_t algoSize = 0;

        CHECK_CUDNN_STATUS(
            cudnnGetConvolutionForwardWorkspaceSize(context_handle,
                                                    inputsTensor.back(),
                                                    filterDesc.back(),
                                                    convDesc,
                                                    outputsTensor,
                                                    algo.back(),
                                                    &algoSize));

        if (algoSize > workSpaceSize)
            workSpaceSize = algoSize;

        const unsigned int nbParams
            = channelsPerInputLayer[i]*nbOutputs*kernelHeight*kernelWidth;

        weights_cudnn.push_back(new DATA_T());

        CHECK_CUDA_STATUS(
            cudaMalloc(&weights_cudnn.back(), nbParams*sizeof(DATA_T)));

    }

    unsigned int memOffset = 0;
    unsigned int cudaOffset = 0;

    for (unsigned int out = 0; out < nbOutputs; ++out) {
        for (unsigned int i = 0 ; i < channelsPerInputLayer.size(); ++i) {
            unsigned int blockSize = channelsPerInputLayer[i]
                *kernelHeight*kernelWidth;

            cudaOffset = blockSize*out;

            CHECK_CUDA_STATUS(
                cudaMemcpy(weights_cudnn[i] + cudaOffset,
                           weights_flatten + memOffset,
                           blockSize*sizeof(DATA_T),
                           cudaMemcpyHostToDevice));

            memOffset += blockSize;
        }
    }
    CHECK_CUDA_STATUS(
        cudaMalloc(&bias_cudnn, nbOutputs*sizeof(DATA_T)));

    CHECK_CUDA_STATUS(
        cudaMemcpy(bias_cudnn,
                   bias_flatten,
                   nbOutputs*sizeof(DATA_T),
                   cudaMemcpyHostToDevice));

    if(workSpaceSize > 0)
        CHECK_CUDA_STATUS(
            cudaMalloc(workSpace, workSpaceSize));

}

void convcell(cudnnHandle_t& context_handle,
              ActivationFunction_T func,
              std::vector<cudnnConvolutionFwdAlgo_t> algo,
              void* workSpace, size_t sizeInBytes,
              std::vector<cudnnTensorDescriptor_t> inputsTensor,
              std::vector<DATA_T*> inputs_data,
              int noBias,
              cudnnTensorDescriptor_t outputsTensor,
              DATA_T** outputs_data,
              cudnnTensorDescriptor_t biasDesc,
              DATA_T* bias_data,
              std::vector<cudnnFilterDescriptor_t> filterDesc,
              cudnnConvolutionDescriptor_t convDesc,
              std::vector<DATA_T*> weights_data)
{
    double elapsed = 0.0;
    std::chrono::high_resolution_clock::time_point start;

    if(oclHandles.isActivated)
        start = std::chrono::high_resolution_clock::now();

    DATA_T ONE_T = DATA_T(1);   // Alpha must be set to 1 for all steps
    DATA_T BETA_T = DATA_T(0);
    DATA_T ZERO_T = DATA_T(0);   // Beta must be set to 0 for POOLING FORWARD

    cudnnActivationMode_t cudnnActivation;

    ((func == Tanh || func == TanhLeCun) ? cudnnActivation
        = CUDNN_ACTIVATION_TANH :
    ((func == Rectifier) ? cudnnActivation
        = CUDNN_ACTIVATION_RELU :
    ((func == FastSigmoid) ? cudnnActivation
        = CUDNN_ACTIVATION_SIGMOID :
    cudnnActivation= CUDNN_ACTIVATION_RELU)));

#if CUDNN_VERSION >= 5000
    cudnnActivationDescriptor_t activationDesc;

    CHECK_CUDNN_STATUS(
        cudnnCreateActivationDescriptor(&activationDesc));

    CHECK_CUDNN_STATUS(
        cudnnSetActivationDescriptor(activationDesc,
                                     cudnnActivation,
                                     NanPolicy,
                                     0.0));
#else
    cudnnActivationMode_t activationDesc = cudnnActivation;
#endif

    for (unsigned int i = 0; i < inputsTensor.size(); ++ i) {
        if (i > 0)
            BETA_T = DATA_T(1);

        CHECK_CUDNN_STATUS(
            cudnnConvolutionForward(context_handle,
                                    &ONE_T,
                                    inputsTensor[i],
                                    inputs_data[i],
                                    filterDesc[i],
                                    weights_data[i],
                                    convDesc,
                                    algo[i],
                                    workSpace,
                                    sizeInBytes,
                                    &BETA_T,
                                    outputsTensor,
                                    *outputs_data));

    }
    if (!noBias) {
#if CUDNN_VERSION >= 5000
        CHECK_CUDNN_STATUS(
            cudnnAddTensor(context_handle,
                           &ONE_T,
                           biasDesc,
                           bias_data,
                           &ONE_T,
                           outputsTensor,
                           *outputs_data));
#else
        CHECK_CUDNN_STATUS(
            cudnnAddTensor(context_handle,
                           CUDNN_ADD_SAME_C,
                           &ONE_T,
                           biasDesc,
                           bias_data,
                           &ONE_T,
                           outputsTensor,
                           outputs_data));
#endif
    }

    if(func != Linear) {
        CHECK_CUDNN_STATUS(
            cudnnActivationForward(context_handle,
                                   activationDesc,
                                   &ONE_T,
                                   outputsTensor,
                                   *outputs_data,
                                   &ZERO_T,
                                   outputsTensor,
                                   *outputs_data));
    }

    if(oclHandles.isActivated) {
        CHECK_CUDA_STATUS( cudaDeviceSynchronize() );
        elapsed= 1.0e6*std::chrono::duration_cast<std::chrono::duration<double> >
            (std::chrono::high_resolution_clock::now() - start).count();
        oclHandles.events.push_back(elapsed);
    }

}
/**** BatchNorm Layer ****/
void setBatchnorm(unsigned int batchSize,
                  unsigned int nbChannels,
                  unsigned int channelsHeight,
                  unsigned int channelsWidth,
                  cudnnTensorDescriptor_t scaleDesc,
                  cudnnTensorFormat_t context_tensorFormat,
                  cudnnDataType_t context_dataType,
                  cudnnTensorDescriptor_t inputsTensor,
                  cudnnTensorDescriptor_t outputsTensor)
{

    if(oclHandles.isActivated)
        oclHandles.profiling.push_back(oclProfiling({"Batchnorm", 0.0}));

#if CUDNN_VERSION >= 4000

    cudnnBatchNormMode_t mMode = CUDNN_BATCHNORM_SPATIAL;

    CHECK_CUDNN_STATUS(cudnnSetTensor4dDescriptor(inputsTensor,
                                                  context_tensorFormat,
                                                  context_dataType,
                                                  batchSize,
                                                  nbChannels,
                                                  channelsHeight,
                                                  channelsWidth));

    CHECK_CUDNN_STATUS(cudnnSetTensor4dDescriptor(outputsTensor,
                                                  context_tensorFormat,
                                                  context_dataType,
                                                  batchSize,
                                                  nbChannels,
                                                  channelsHeight,
                                                  channelsWidth));

    cudnnTensorDescriptor_t derivedBnDesc;
    CHECK_CUDNN_STATUS(cudnnCreateTensorDescriptor(&derivedBnDesc));
    CHECK_CUDNN_STATUS(
        cudnnDeriveBNTensorDescriptor(derivedBnDesc, inputsTensor, mMode));

    cudnnDataType_t dataType;

    int n;
    int c;
    int h;
    int w;
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

    CHECK_CUDNN_STATUS(cudnnSetTensor4dDescriptor(
        scaleDesc, context_tensorFormat, context_dataType, n, c, h, w));
/*
    CHECK_CUDA_STATUS(cudaMalloc ( &scale, w*h*n*c*sizeof(DATA_T) ));
    CHECK_CUDA_STATUS(cudaMalloc ( &bias, w*h*n*c*sizeof(DATA_T) ));
    CHECK_CUDA_STATUS(cudaMalloc ( &mean, w*h*n*c*sizeof(DATA_T) ));
    CHECK_CUDA_STATUS(cudaMalloc ( &variance, w*h*n*c*sizeof(DATA_T) ));
*/
#endif
}
void batchnormcell(cudnnHandle_t& context_handle,
                   unsigned int batchSize,
                   unsigned int nbChannels,
                   unsigned int channelsHeight,
                   unsigned int channelsWidth,
                   cudnnTensorDescriptor_t inputsTensor,
                   std::vector<DATA_T*> inputs_data,
                   DATA_T* scale,
                   cudnnTensorDescriptor_t scaleDesc,
                   DATA_T* bias,
                   DATA_T* mean,
                   DATA_T* variance,
                   DATA_T epsilon,
                   cudnnTensorDescriptor_t outputsTensor,
                   DATA_T** outputs_data,
                   ActivationFunction_T func)
{
    double elapsed = 0.0;
    std::chrono::high_resolution_clock::time_point start;

    if(oclHandles.isActivated)
        start = std::chrono::high_resolution_clock::now();
#if CUDNN_VERSION >= 4000

    cudnnActivationMode_t cudnnActivation;

    ((func == Tanh || func == TanhLeCun) ? cudnnActivation
        = CUDNN_ACTIVATION_TANH
    : ((func == Rectifier) ? cudnnActivation
        = CUDNN_ACTIVATION_RELU
    : ((func == FastSigmoid) ? cudnnActivation
        = CUDNN_ACTIVATION_SIGMOID
    : cudnnActivation = CUDNN_ACTIVATION_RELU)));

#if CUDNN_VERSION >= 5000
    cudnnActivationDescriptor_t activationDesc;
    CHECK_CUDNN_STATUS(cudnnCreateActivationDescriptor(&activationDesc));

    CHECK_CUDNN_STATUS(cudnnSetActivationDescriptor(
        activationDesc, cudnnActivation, NanPolicy, 0.0));
#else
    cudnnActivationMode_t activationDesc = cudnnActivation;
#endif

    cudnnBatchNormMode_t mMode = CUDNN_BATCHNORM_SPATIAL;
    DATA_T ONE_T = DATA_T(1); // Alpha must be set to 1 for all steps
    DATA_T ZERO_T = DATA_T(0); // Beta must be set to 0 for POOLING FORWARD

    CHECK_CUDNN_STATUS(
        cudnnBatchNormalizationForwardInference(context_handle,
                                                mMode,
                                                &ONE_T,
                                                &ZERO_T,
                                                inputsTensor,
                                                inputs_data[0],
                                                outputsTensor,
                                                *outputs_data,
                                                scaleDesc,
                                                scale,
                                                bias,
                                                mean,
                                                variance,
                                                epsilon));

#else
    cudaSBNPropagate(inputs_data[0],
                     bias,
                     variance,
                     mean,
                     scale,
                     epsilon,
                     &outputs_data,
                     nbChannels,
                     channelsHeight,
                     channelsWidth,
                     batchSize);

#endif

    if (func != Linear) {

        CHECK_CUDNN_STATUS(
            cudnnActivationForward(context_handle,
                                   activationDesc,
                                   &ONE_T,
                                   outputsTensor,
                                   *outputs_data,
                                   &ZERO_T,
                                   outputsTensor,
                                   *outputs_data)); // Activation
    }

    if(oclHandles.isActivated){
        CHECK_CUDA_STATUS(cudaDeviceSynchronize());
        elapsed = 1.0e6
                  * std::chrono::duration_cast<std::chrono::duration<double> >(
                        std::chrono::high_resolution_clock::now() - start).count();
        oclHandles.events.push_back(elapsed);
    }
}

/**** Pooling Layer ****/
void setPooling(unsigned int batchSize,
                std::vector<int> channelsPerInputLayer,
                std::vector<int> channelsHeightPerInputLayer,
                std::vector<int> channelsWidthPerInputLayer,
                unsigned int paddingY,
                unsigned int paddingX,
                unsigned int strideY,
                unsigned int strideX,
                unsigned int outputHeight,
                unsigned int outputWidth,
                cudnnTensorFormat_t context_tensorFormat,
                cudnnDataType_t context_dataType,
                std::vector<cudnnTensorDescriptor_t>& inputsTensor,
                std::vector<cudnnTensorDescriptor_t>& outputsTensor,
                Pooling_T func,
                cudnnPoolingMode_t& cudnnPooling,
                unsigned int nbOutputs,
                unsigned int poolHeight,
                unsigned int poolWidth,
                cudnnPoolingDescriptor_t& mapping)
{
    int n = batchSize;
    std::vector<int> c = channelsPerInputLayer;
    std::vector<int> h = channelsHeightPerInputLayer;
    std::vector<int> w = channelsWidthPerInputLayer;

    if(oclHandles.isActivated)
        oclHandles.profiling.push_back(oclProfiling({ "Pooling", 0.0}));

    ((func == Average) ? cudnnPooling
        = CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING :
    ((func == Max) ? cudnnPooling
        = CUDNN_POOLING_MAX :
    cudnnPooling = CUDNN_POOLING_MAX));

    cudnnCreatePoolingDescriptor(&mapping);

#if CUDNN_VERSION >= 5000
    CHECK_CUDNN_STATUS(
        cudnnSetPooling2dDescriptor(mapping,
                                    cudnnPooling,
                                    NanPolicy,
                                    poolHeight,
                                    poolWidth,
                                    paddingY,
                                    paddingX,
                                    strideY,
                                    strideX));
#else
    CHECK_CUDNN_STATUS(
        cudnnSetPooling2dDescriptor(mapping,
                                    cudnnPooling,
                                    poolHeight,
                                    poolWidth,
                                    paddingY,
                                    paddingX,
                                    strideX,  // BUG in cuDNN v3 (order of the last 2 arguments was inverted), resolved with cuDNN v5
                                    strideY) );
#endif
    for (unsigned int k = 0; k < channelsPerInputLayer.size(); ++k) {

        inputsTensor.push_back(cudnnTensorDescriptor_t());
        cudnnCreateTensorDescriptor(&inputsTensor.back());
        outputsTensor.push_back(cudnnTensorDescriptor_t());
        cudnnCreateTensorDescriptor(&outputsTensor.back());

        CHECK_CUDNN_STATUS(
            cudnnSetTensor4dDescriptor(inputsTensor.back(),
                                       context_tensorFormat,
                                       context_dataType,
                                       n,
                                       c[k],
                                       h[k],
                                       w[k]));

        CHECK_CUDNN_STATUS(cudnnSetTensor4dDescriptorEx(
            outputsTensor.back(),
            context_dataType,
            n,
            c[k],
            outputHeight,
            outputWidth,
            outputWidth * outputHeight * nbOutputs,
            outputWidth * outputHeight,
            outputWidth,
            1));
    }

}
void poolcell(cudnnHandle_t& context_handle,
              std::vector<cudnnTensorDescriptor_t> inputsTensor,
              std::vector<DATA_T*> inputs_data,
              std::vector<cudnnTensorDescriptor_t> outputsTensor,
              DATA_T** outputs_data,
              cudnnPoolingDescriptor_t mapping)
{
    double elapsed = 0.0;
    std::chrono::high_resolution_clock::time_point start;

    if(oclHandles.isActivated)
        start = std::chrono::high_resolution_clock::now();

    DATA_T ONE_T = DATA_T(1);   // Alpha must be set to 1 for all steps
    DATA_T ZERO_T = DATA_T(0);   // Beta must be set to 0 for POOLING FORWARD

    int inputDimBatch = 0;
    int inputDimChannel = 0;

    int outputDimY = 0;
    int outputDimX = 0;
    int outputStrideBatch = 0;
    int outputStrideFeaturesMap = 0;
    int outputStrideHeight = 0;
    int outputStrideWidth = 0;

    unsigned int offset = 0;
    cudnnDataType_t tensorDataType;

    for (unsigned int k = 0; k < inputsTensor.size(); ++k) {

        CHECK_CUDNN_STATUS(
            cudnnGetTensor4dDescriptor(outputsTensor[k],
                                       &tensorDataType,
                                       &inputDimBatch,
                                       &inputDimChannel,
                                       &outputDimY,
                                       &outputDimX,
                                       &outputStrideBatch,
                                       &outputStrideFeaturesMap,
                                       &outputStrideHeight,
                                       &outputStrideWidth ));

        CHECK_CUDNN_STATUS(
            cudnnPoolingForward(context_handle,
                                mapping,
                                &ONE_T,
                                inputsTensor[k],
                                inputs_data[k],
                                &ZERO_T,
                                outputsTensor[k],
                                *outputs_data + offset) );

        offset += inputDimChannel * outputDimY * outputDimX ;

    }

    if(oclHandles.isActivated) {
        CHECK_CUDA_STATUS(cudaDeviceSynchronize());
        elapsed = 1.0e6
                  * std::chrono::duration_cast<std::chrono::duration<double> >(
                        std::chrono::high_resolution_clock::now() - start).count();
        oclHandles.events.push_back(elapsed);
    }
}
/**** FractionnalMaxPooling Layer ****/
void setFmp()
{
    if(oclHandles.isActivated)
        oclHandles.profiling.push_back(oclProfiling({"FMP", 0.0}));
}

void fmpcell(cudnnHandle_t& context_handle,
             unsigned int batchSize,
             unsigned int nbChannels,
             unsigned int channelsHeight,
             unsigned int channelsWidth,
             unsigned int* gridx,
             unsigned int* gridy,
             const bool overlapping,
             std::vector<DATA_T*> inputs_data,
             unsigned int nbOutputs_,
             unsigned int outputsHeight,
             unsigned int outputsWidth,
             unsigned int nbOutputs,
             unsigned int outputOffset,
             DATA_T* outputs_data)
{
    double elapsed = 0.0;
    std::chrono::high_resolution_clock::time_point start;

    if(oclHandles.isActivated)
        start = std::chrono::high_resolution_clock::now();

    unsigned int* gridXRand
        = new unsigned int[outputsWidth]; // use array new.  Note that length
    // does not need to be constant!
    unsigned int* gridYRand
        = new unsigned int[outputsHeight]; // use array new.  Note that length
    // does not need to be constant!
    fmpcell_propagate_generateRegions(&gridXRand[0],
                                      channelsWidth,
                                      outputsWidth);

    fmpcell_propagate_generateRegions(&gridYRand[0],
                                      channelsHeight,
                                      outputsHeight);
    CHECK_CUDA_STATUS(
        cudaMemcpy(gridx,
                   gridXRand,
                   outputsWidth * sizeof(unsigned int),
                   cudaMemcpyHostToDevice));

    CHECK_CUDA_STATUS(
        cudaMemcpy(gridy,
                   gridYRand,
                   outputsHeight * sizeof(unsigned int),
                   cudaMemcpyHostToDevice));

    cudaSFMPPropagate(inputs_data[0],
                      gridx,
                      gridy,
                      outputs_data,
                      nbChannels,
                      channelsHeight,
                      channelsWidth,
                      nbOutputs,
                      outputsHeight,
                      outputsWidth,
                      batchSize,
                      overlapping);
    if(oclHandles.isActivated){
        CHECK_CUDA_STATUS(cudaDeviceSynchronize());
        elapsed = 1.0e6
                  * std::chrono::duration_cast<std::chrono::duration<double> >(
                        std::chrono::high_resolution_clock::now() - start).count();
        oclHandles.events.push_back(elapsed);
    }
}

void fmpcell_propagate_generateRegions(unsigned int* grid,
                                       unsigned int sizeIn,
                                       unsigned int sizeOut)
{
    // Compute the true scaling ratio
    // This is important to obtain the correct range
    const double scalingRatio = sizeIn / (double)sizeOut;
    const double u = Random::randUniform(0.0, 1.0, Random::OpenInterval);
    for (unsigned int i = 0; i < sizeOut; ++i)
        grid[i] = (unsigned int)std::ceil(scalingRatio * (i + u));
}

/**** FullyConnected Layer ****/
void setFc(unsigned int batchSize,
           std::vector<int> channelsPerInputLayer,
           std::vector<int> channelsHeightPerInputLayer,
           std::vector<int> channelsWidthPerInputLayer,
           std::vector<cudnnTensorDescriptor_t>& inputsTensor,
           const DATA_T* weights_flatten,
           std::vector<DATA_T*>& weights_cudnn,
           const DATA_T *bias_flatten,
           DATA_T *& bias_cudnn,
           cudnnTensorFormat_t context_tensorFormat,
           cudnnDataType_t context_dataType,
           cudnnTensorDescriptor_t& outputsTensor,
           ActivationFunction_T func,
           unsigned int nbOutputs)
{
    int n = batchSize;
    int k = nbOutputs;
    int hOut = 1;
    int wOut = 1;
    std::vector<int> c = channelsPerInputLayer;
    std::vector<int> h = channelsHeightPerInputLayer;
    std::vector<int> w = channelsWidthPerInputLayer;

    if(oclHandles.isActivated)
        oclHandles.profiling.push_back(oclProfiling({"FullyConnected", 0.0}));

    for (unsigned int k = 0; k < channelsPerInputLayer.size(); ++k) {
        const unsigned int nbParams = c[k]*h[k]*w[k]*nbOutputs;

        inputsTensor.push_back(cudnnTensorDescriptor_t());
        cudnnCreateTensorDescriptor(&inputsTensor.back());

        CHECK_CUDNN_STATUS(
            cudnnSetTensor4dDescriptor(inputsTensor.back(),
                                       context_tensorFormat,
                                       context_dataType,
                                       n,
                                       c[k],
                                       h[k],
                                       w[k]));

        weights_cudnn.push_back(new DATA_T());
        CHECK_CUDA_STATUS(
            cudaMalloc(&weights_cudnn.back(), nbParams*sizeof(DATA_T)));
    }

    unsigned int memOffset = 0;
    unsigned int cudaOffset = 0;
    for (unsigned int out = 0; out < nbOutputs; ++out) {
        for (unsigned int k = 0 ; k < channelsPerInputLayer.size(); ++k) {
            unsigned int blockSize = c[k]*h[k]*w[k];

            cudaOffset = blockSize*out;

            CHECK_CUDA_STATUS( cudaMemcpy(weights_cudnn[k] + cudaOffset,
                    weights_flatten + memOffset,
                    blockSize*sizeof(DATA_T),
                    cudaMemcpyHostToDevice) );

            memOffset += blockSize;
        }
    }

    CHECK_CUDA_STATUS(
        cudaMalloc(&bias_cudnn, nbOutputs*sizeof(DATA_T)));

    CHECK_CUDA_STATUS(
        cudaMemcpy(bias_cudnn,
                   bias_flatten,
                   nbOutputs*sizeof(DATA_T),
                   cudaMemcpyHostToDevice));

    cudnnCreateTensorDescriptor(&outputsTensor);

    CHECK_CUDNN_STATUS(
        cudnnSetTensor4dDescriptor(outputsTensor,
                                   context_tensorFormat,
                                   context_dataType,
                                   n,
                                   k,
                                   hOut,
                                   wOut));
}

void fullyConnected(unsigned int nbChannels,
                    cudnnHandle_t& context_handle,
                    cublasHandle_t& context_cublasHandle,
                    ActivationFunction_T func,
                    std::vector<cudnnTensorDescriptor_t> inputsTensor,
                    std::vector<DATA_T*> inputs_data,
                    cudnnTensorDescriptor_t outputsTensor,
                    unsigned int nbOutputs,
                    int noBias,
                    DATA_T** outputs_data,
                    DATA_T *bias_data,
                    DATA_T *ones_vec_data,
                    std::vector<DATA_T*> weights_data)
{
    double elapsed = 0.0;
    std::chrono::high_resolution_clock::time_point start;

    if(oclHandles.isActivated)
        start = std::chrono::high_resolution_clock::now();

    /********CublasSgemm function used for the Fully Connected Layers**********/
    /**  This function performs the matrix-matrix multiplication

    C = α*op( A )*op*( B ) + β*C

    where α and β are scalars, and A , B and C are matrices stored in
    column-major format with dimensions op ( A ) m × k , op ( B ) k × n and C m
    × n , respectively.
    Also, for matrix A:
            op ( A ) = A if   transa == CUBLAS_OP_N
                 = AT if  transa == CUBLAS_OP_T
                 = AH if  transa == CUBLAS_OP_C

        More informations: http://docs.nvidia.com/cuda/cublas/
    ***************************************************************************/
    DATA_T ONE_T = DATA_T(1);    //ONE_T must be set to 1
    DATA_T ZERO_T = DATA_T(0);    //ZERO_T must be set to 0
    DATA_T BETA_T = DATA_T(0);

    cudnnActivationMode_t cudnnActivation;

    ((func == Tanh || func == TanhLeCun) ? cudnnActivation
        = CUDNN_ACTIVATION_TANH :
    ((func == Rectifier) ? cudnnActivation
        = CUDNN_ACTIVATION_RELU :
    ((func == FastSigmoid) ? cudnnActivation
        = CUDNN_ACTIVATION_SIGMOID :
    cudnnActivation= CUDNN_ACTIVATION_RELU)));

#if CUDNN_VERSION >= 5000
    cudnnActivationDescriptor_t activationDesc;
    CHECK_CUDNN_STATUS(cudnnCreateActivationDescriptor(&activationDesc));

    CHECK_CUDNN_STATUS(
        cudnnSetActivationDescriptor(activationDesc,
                                     cudnnActivation,
                                     NanPolicy,
                                     0.0));
#else
    cudnnActivationMode_t activationDesc = cudnnActivation;
#endif

    cublasOperation_t transA = CUBLAS_OP_T;
    cublasOperation_t transB = CUBLAS_OP_N;
    int m = nbOutputs;

    int batch;
    int ch ;
    int chH ;
    int chW ;
    int nStride ;
    int cStride ;
    int hStride ;
    int wStride ;
    cudnnDataType_t tensorDataType;

    for (unsigned int i = 0; i < inputs_data.size(); ++i) {
        if(i > 0)
            BETA_T = DATA_T(1);

        CHECK_CUDNN_STATUS(
            cudnnGetTensor4dDescriptor(inputsTensor[i],
                                       &tensorDataType,
                                       &batch,
                                       &ch,
                                       &chH,
                                       &chW,
                                       &nStride,
                                       &cStride,
                                       &hStride,
                                       &wStride ));

        const unsigned int inputSize = ch * chH * chW;

        cublasSgemm(context_cublasHandle,
                    transA,
                    transB,
                    m,
                    batch,
                    inputSize,
                    &ONE_T,
                    weights_data[i],
                    inputSize,
                    inputs_data[i],
                    inputSize,
                    &BETA_T,
                    *outputs_data,
                    m);
    }
    if(!noBias){

        CHECK_CUBLAS_STATUS(
            cublasSgemm(context_cublasHandle,
                        CUBLAS_OP_N,
                        CUBLAS_OP_N,
                        m,
                        batch,
                        1,
                        &ONE_T,
                        bias_data,
                        m,
                        ones_vec_data,
                        1,
                        &ONE_T,
                        *outputs_data,
                        m) );
    }

    if(func != Linear) {

        CHECK_CUDNN_STATUS(
            cudnnActivationForward(context_handle,
                                   activationDesc,
                                   &ONE_T,
                                   outputsTensor,
                                   *outputs_data,
                                   &ZERO_T,
                                   outputsTensor,
                                   *outputs_data) );
    }

    if(oclHandles.isActivated) {
        CHECK_CUDA_STATUS(cudaDeviceSynchronize());
        elapsed = 1.0e6
                  * std::chrono::duration_cast<std::chrono::duration<double> >(
                        std::chrono::high_resolution_clock::now() - start).count();
        oclHandles.events.push_back(elapsed);
    }
}

/**** SoftMax Layer ****/

void setSoftmax(unsigned int batchSize,
                unsigned int nbChannels,
                unsigned int channelsHeight,
                unsigned int channelsWidth,
                cudnnTensorFormat_t context_tensorFormat,
                cudnnDataType_t context_dataType,
                cudnnTensorDescriptor_t inputsTensor,
                cudnnTensorDescriptor_t outputsTensor)
{
    int n = batchSize;
    int c = nbChannels;
    int h = channelsHeight;
    int w = channelsWidth;

    if(oclHandles.isActivated)
        oclHandles.profiling.push_back(oclProfiling({"SoftMax", 0.0}));

    CHECK_CUDNN_STATUS(cudnnSetTensor4dDescriptor(
        inputsTensor, context_tensorFormat, context_dataType, n, c, h, w));

    CHECK_CUDNN_STATUS(cudnnSetTensor4dDescriptor(
        outputsTensor, context_tensorFormat, context_dataType, n, c, h, w));
}

void softmax(cudnnHandle_t& context_handle,
             cudnnTensorDescriptor_t inputsTensor,
             std::vector<DATA_T*> inputs_data,
             cudnnTensorDescriptor_t outputsTensor,
             DATA_T** outputs_data)
{
    double elapsed = 0.0;
    std::chrono::high_resolution_clock::time_point start;

    if(oclHandles.isActivated)
        start = std::chrono::high_resolution_clock::now();

    DATA_T alpha = DATA_T(1);
    DATA_T beta = DATA_T(0);

    CHECK_CUDNN_STATUS(cudnnSoftmaxForward(context_handle,
                                           CUDNN_SOFTMAX_ACCURATE,
                                           CUDNN_SOFTMAX_MODE_CHANNEL,
                                           &alpha,
                                           inputsTensor,
                                           inputs_data[0],
                                           &beta,
                                           outputsTensor,
                                           *outputs_data));

    if(oclHandles.isActivated) {
        CHECK_CUDA_STATUS(cudaDeviceSynchronize());
        elapsed = 1.0e6
                  * std::chrono::duration_cast<std::chrono::duration<double> >(
                        std::chrono::high_resolution_clock::now() - start).count();
        oclHandles.events.push_back(elapsed);
    }
}
/****Targets Layers ****/
void output_generation(unsigned int batchSize,
                       unsigned int nbOutputs,
                       DATA_T* dataIn,
                       uint32_t* outputEstimated)
{
    DATA_T* outputsData(NULL);

    if (outputsData == NULL) {
        outputsData = new DATA_T[batchSize * nbOutputs];

        if (!outputsData)
            throw std::runtime_error(
                "output_generation(): could not allocate memory");
    }

    CHECK_CUDA_STATUS(cudaMemcpy(outputsData,
                                 dataIn,
                                 batchSize * nbOutputs * sizeof(DATA_T),
                                 cudaMemcpyDeviceToHost));

    if(oclHandles.isActivated) {
        for (std::vector<double>::iterator it = oclHandles.events.begin(),
                                           itBegin = oclHandles.events.begin(),
                                           itEnd = oclHandles.events.end();
             it != itEnd;
             ++it) {
            oclHandles.profiling[it - itBegin].processTime += (*it);
        }

        oclHandles.events.clear();
    }

    for (unsigned int i = 0; i < batchSize; i++) {

        DATA_T maxVal = outputsData[i * nbOutputs];
        unsigned int outputMax = 0;

        for (unsigned int output = 1 + nbOutputs * i;
             output < nbOutputs * (i + 1);
             ++output) {

            if (outputsData[output] > maxVal) {
                maxVal = outputsData[output];
                outputMax = output - i * nbOutputs;
            }
        }
        outputEstimated[i] = outputMax;
    }
    delete[] outputsData;
}
void set_output(unsigned int nbTarget)
{
    if(oclHandles.isActivated)
      for(unsigned int i = 0; i < nbTarget; ++i)
          oclHandles.profiling.push_back(oclProfiling({"memcpy_DevToHost", 0.0}));
}

void spatial_output_generation(unsigned int batchSize,
                               unsigned int nbOutputs,
                               unsigned int outputsHeight,
                               unsigned int outputsWidth,
                               DATA_T* dataIn,
                               uint32_t* outputEstimated)
{
    double elapsed = 0.0;
    std::chrono::high_resolution_clock::time_point start;

    if(oclHandles.isActivated)
        start = std::chrono::high_resolution_clock::now();

    const unsigned int size = nbOutputs * outputsHeight * outputsWidth;
    DATA_T* outputsData(NULL);
    if (outputsData == NULL) {
        outputsData = new DATA_T[batchSize * size];

        if (!outputsData)
            throw std::runtime_error(
                "spatial_output_generation(): could not allocate memory");
    }

    CHECK_CUDA_STATUS(cudaMemcpy(outputsData,
                                 dataIn,
                                 batchSize * size * sizeof(DATA_T),
                                 cudaMemcpyDeviceToHost));

    for (unsigned int i = 0; i < batchSize; i++) {
        for (unsigned int oy = 0; oy < outputsHeight; ++oy) {
            for (unsigned int ox = 0; ox < outputsWidth; ++ox) {
                const unsigned int inputsIdx
                    = ox + oy * outputsWidth
                      + i * (outputsHeight * outputsWidth * nbOutputs);
                DATA_T maxVal = outputsData[inputsIdx];
                unsigned int outputMax = 0;
                if(nbOutputs > 1)
                {
                    for (unsigned int output = 1; output < nbOutputs; ++output) {
                        const unsigned int outputsIdx
                            = ox + (oy + output * outputsHeight) * outputsWidth
                              + i * (outputsHeight * outputsWidth * nbOutputs);
                        if (outputsData[outputsIdx] > maxVal) {
                            outputMax = output;
                            maxVal = outputsData[outputsIdx];
                        }
                    }
                }
                else
                {
                    if(maxVal > 0.0)
                      outputMax = 1;
                }
                outputEstimated[ox + oy * outputsWidth
                                + i * (outputsHeight * outputsWidth)]
                    = outputMax;
            }
        }
    }

    if(oclHandles.isActivated) {
        CHECK_CUDA_STATUS(cudaDeviceSynchronize());
        elapsed = 1.0e6
                  * std::chrono::duration_cast<std::chrono::duration<double> >(
                        std::chrono::high_resolution_clock::now() - start).count();
        oclHandles.events.push_back(elapsed);
    }

    if(oclHandles.isActivated) {
        for (std::vector<double>::iterator it = oclHandles.events.begin(),
                                           itBegin = oclHandles.events.begin(),
                                           itEnd = oclHandles.events.end();
             it != itEnd;
             ++it) {
            oclHandles.profiling[it - itBegin].processTime += (*it);
        }

        oclHandles.events.clear();
    }

    delete[] outputsData;
}

/**** Debug Function ****/
void dumpMem(int size, DATA_T* data, std::string fileName)
{

    std::ofstream file;
    file.open(fileName.c_str());

    DATA_T* eagleEyes(NULL);
    eagleEyes = new DATA_T[size];

    CHECK_CUDA_STATUS(cudaMemcpy(
        eagleEyes, data, size * sizeof(DATA_T), cudaMemcpyDeviceToHost));

    for (int i = 0; i < size; i++)
#if NB_BITS < 0
        file << "data[" << i << "]= " << eagleEyes[i] << "\n";
#else
        file << "data[" << i << "]= " << (int)eagleEyes[i] << "\n";
#endif
    std::cout << "dump mem in file " << fileName.c_str() << "done"
              << "\n";
    file.close();
    delete[] eagleEyes;
}
