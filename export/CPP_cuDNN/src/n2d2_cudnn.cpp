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

void getFilesList(const std::string dir, std::vector<std::string>& files)
{
    struct dirent* pFile;
    DIR* pDir = opendir(dir.c_str());
    if (pDir == NULL)
        throw std::runtime_error(
            "Couldn't open the directory for input patterns: " + dir);

    while ((pFile = readdir(pDir)) != NULL) {
        if (pFile->d_name[0] != '.')
            files.push_back(std::string(dir + "/" + pFile->d_name));
    }
    closedir(pDir);
    std::sort(files.begin(), files.end());
}

void envRead(const std::string& fileName,
             unsigned int size,
             unsigned int channelsHeight,
             unsigned int channelsWidth,
             DATA_T* data,
             unsigned int outputsSize,
             int32_t* outputTargets)
{
    std::ifstream stimuli(fileName.c_str(), std::fstream::binary);

    if (!stimuli.good())
        throw std::runtime_error("Could not open file: " + fileName);

    char header[2];
    stimuli.read(reinterpret_cast<char*>(&header[0]), sizeof(header));

    if (header[0] != 'P' || header[1] != '5')
        throw std::runtime_error("Unknown PGM file format for file: "
                                 + fileName);

    int pixelWidth;
    int pixelHeight;
    int maxValue;

    if (!(stimuli >> pixelWidth) || !(stimuli >> pixelHeight)
        || !(stimuli >> maxValue))
        throw std::runtime_error("Error reading PGM image file: " + fileName);

    stimuli.get();

    if (pixelWidth != (int)channelsWidth || pixelHeight != (int)channelsHeight)
        throw std::runtime_error(
            "PGM image size does not match array size for file: " + fileName);

#if NB_BITS > 0 && NB_BITS != 8 && NB_BITS != 16 && NB_BITS != 32 && NB_BITS   \
                                                                     != 64
#if NB_BITS > 0 && NB_BITS < 8
    char inputsFixed[size];
#elif NB_BITS > 8 && NB_BITS < 16
    short inputsFixed[size];
#elif NB_BITS > 16 && NB_BITS < 32
    int inputsFixed[size];
#elif NB_BITS > 32 && NB_BITS < 64
    long long int inputsFixed[size];
#endif
    stimuli.read(reinterpret_cast<char*>(&inputsFixed[0]),
                 size * sizeof(inputsFixed[0]));

    for (unsigned int i = 0; i < size; ++i)
        data[i] = (DATA_T)inputsFixed[i];
#else
    stimuli.read(reinterpret_cast<char*>(&data[0]), size * sizeof(data[0]));
#endif
    stimuli.read(reinterpret_cast<char*>(&outputTargets[0]),
                 outputsSize * sizeof(outputTargets[0]));

    if (stimuli.eof())
        throw std::runtime_error(
            "End-of-file reached prematurely in data file: " + fileName);
    else if (!stimuli.good())
        throw std::runtime_error("Error while reading data file: " + fileName);
    else if (stimuli.get() != std::fstream::traits_type::eof())
        throw std::runtime_error("Data file size larger than expected: "
                                 + fileName);
}

/************************************CONVOLUTION*************************************************/
/************************************************************************************************/
void setConvolution(unsigned int batchSize,
                    unsigned int nbChannels,
                    unsigned int channelsHeight,
                    unsigned int channelsWidth,
                    unsigned int paddingY,
                    unsigned int paddingX,
                    unsigned int strideY,
                    unsigned int strideX,
                    unsigned int subSampleY,
                    unsigned int subSampleX,
                    cudnnHandle_t& context_handle,
                    cudnnTensorFormat_t context_tensorFormat,
                    cudnnDataType_t context_dataType,
                    cudnnTensorDescriptor_t inputsTensor,
                    cudnnTensorDescriptor_t outputsTensor,
                    ActivationFunction_T func,
                    cudnnConvolutionFwdAlgo_t algo,
                    size_t sizeInBytes,
                    void* workSpace,
                    unsigned int nbOutputs,
                    unsigned int outputOffset,
                    unsigned int kernelHeight,
                    unsigned int kernelWidth,
                    cudnnTensorDescriptor_t biasDesc,
                    cudnnFilterDescriptor_t filterDesc,
                    cudnnConvolutionDescriptor_t convDesc)
{
    int n = batchSize;
    int k = nbOutputs;
    int c = nbChannels;

    int hK = kernelHeight;
    int wK = kernelWidth;

    int hCh = channelsHeight;
    int wCh = channelsWidth;

#ifdef PROFILING
    oclHandles.profiling.push_back(oclProfiling({"Convolution", 0.0}));
#endif

    CHECK_CUDNN_STATUS(cudnnSetTensor4dDescriptor(
        biasDesc, context_tensorFormat, context_dataType, 1, k, 1, 1));

#if CUDNN_VERSION >= 5000
    CHECK_CUDNN_STATUS(cudnnSetFilter4dDescriptor(
        filterDesc, context_dataType, context_tensorFormat, k, c, hK, wK));
#else
    CHECK_CUDNN_STATUS(
        cudnnSetFilter4dDescriptor(filterDesc, context_dataType, k, c, hK, wK));
#endif

    CHECK_CUDNN_STATUS(cudnnSetConvolution2dDescriptor(convDesc,
                                                       paddingY,
                                                       paddingX,
                                                       strideY,
                                                       strideX,
                                                       subSampleY,
                                                       subSampleX,
                                                       CUDNN_CONVOLUTION));

    CHECK_CUDNN_STATUS(cudnnSetTensor4dDescriptor(
        inputsTensor, context_tensorFormat, context_dataType, n, c, hCh, wCh));

    CHECK_CUDNN_STATUS(cudnnGetConvolution2dForwardOutputDim(
        convDesc, inputsTensor, filterDesc, &n, &c, &hCh, &wCh));

    CHECK_CUDNN_STATUS(cudnnSetTensor4dDescriptor(
        outputsTensor, context_tensorFormat, context_dataType, n, c, hCh, wCh));

    CHECK_CUDNN_STATUS(cudnnGetConvolutionForwardAlgorithm(
        context_handle,
        inputsTensor,
        filterDesc,
        convDesc,
        outputsTensor,
        CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
        0,
        &algo));

    CHECK_CUDNN_STATUS(cudnnGetConvolutionForwardWorkspaceSize(context_handle,
                                                               inputsTensor,
                                                               filterDesc,
                                                               convDesc,
                                                               outputsTensor,
                                                               algo,
                                                               &sizeInBytes));
}

void convcell(cudnnHandle_t& context_handle,
              ActivationFunction_T func,
              cudnnConvolutionFwdAlgo_t algo,
              void* workSpace,
              size_t sizeInBytes,
              cudnnTensorDescriptor_t inputsTensor,
              DATA_T* inputs_data,
              unsigned int outputOffset,
              int noBias,
              cudnnTensorDescriptor_t outputsTensor,
              DATA_T** outputs_data,
              cudnnTensorDescriptor_t biasDesc,
              DATA_T* bias_data,
              cudnnFilterDescriptor_t filterDesc,
              cudnnConvolutionDescriptor_t convDesc,
              DATA_T* weights_data)
{
#ifdef PROFILING
    double elapsed = 0.0;
    const std::chrono::high_resolution_clock::time_point start
        = std::chrono::high_resolution_clock::now();
#endif

    DATA_T ONE_T = DATA_T(1); // Alpha must be set to 1 for all steps
    DATA_T ZERO_T = DATA_T(0); // Beta must be set to 0 for CONVOLUTION FORWARD

    cudnnActivationMode_t cudnnActivation;

    ((func == Tanh || func == TanhLeCun)
     ? cudnnActivation = CUDNN_ACTIVATION_TANH
     : ((func == Rectifier)
        ? cudnnActivation = CUDNN_ACTIVATION_RELU
        : ((func == FastSigmoid) ? cudnnActivation = CUDNN_ACTIVATION_SIGMOID
                                 : cudnnActivation = CUDNN_ACTIVATION_RELU)));

#if CUDNN_VERSION >= 5000
    cudnnActivationDescriptor_t activationDesc;
    CHECK_CUDNN_STATUS(cudnnCreateActivationDescriptor(&activationDesc));

    CHECK_CUDNN_STATUS(cudnnSetActivationDescriptor(
        activationDesc, cudnnActivation, NanPolicy, 0.0));
#else
    cudnnActivationMode_t activationDesc = cudnnActivation;
#endif

    /********CONVOLUTION FORWARD***********/
    CHECK_CUDNN_STATUS(cudnnConvolutionForward(context_handle,
                                               &ONE_T,
                                               inputsTensor,
                                               inputs_data,
                                               filterDesc,
                                               weights_data,
                                               convDesc,
                                               algo,
                                               workSpace,
                                               sizeInBytes,
                                               &ZERO_T,
                                               outputsTensor,
                                               *outputs_data + outputOffset));

    if (!noBias) {
/********CONVOLUTION ADD BIAS***********/
#if CUDNN_VERSION >= 5000
        CHECK_CUDNN_STATUS(
            cudnnAddTensor(context_handle,
                           &ONE_T,
                           biasDesc,
                           bias_data,
                           &ONE_T,
                           outputsTensor,
                           *outputs_data + outputOffset)); // Bias
#else
        CHECK_CUDNN_STATUS(cudnnAddTensor(context_handle,
                                          CUDNN_ADD_SAME_C,
                                          &ONE_T,
                                          biasDesc,
                                          bias_data,
                                          &ONE_T,
                                          outputsTensor,
                                          *outputs_data + outputOffset));
#endif
    }

    if (func != Linear) {
        /********CONVOLUTION ACTIVATION***********/
        CHECK_CUDNN_STATUS(
            cudnnActivationForward(context_handle,
                                   activationDesc,
                                   &ONE_T,
                                   outputsTensor,
                                   *outputs_data + outputOffset,
                                   &ZERO_T,
                                   outputsTensor,
                                   *outputs_data + outputOffset)); // Activation
    }

    if (sizeInBytes != 0)
        CHECK_CUDA_STATUS(cudaFree(workSpace));

#ifdef PROFILING
    CHECK_CUDA_STATUS(cudaDeviceSynchronize());
    elapsed = 1.0e6
              * std::chrono::duration_cast<std::chrono::duration<double> >(
                    std::chrono::high_resolution_clock::now() - start).count();
    oclHandles.events.push_back(elapsed);
#endif
}
/************************************BATCHNORM*****************************************************/
/************************************************************************************************/
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

#ifdef PROFILING
    oclHandles.profiling.push_back(oclProfiling({"Batchnorm", 0.0}));
#endif
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
                   DATA_T* inputs_data,
                   DATA_T* scale,
                   cudnnTensorDescriptor_t scaleDesc,
                   DATA_T* bias,
                   DATA_T* mean,
                   DATA_T* variance,
                   DATA_T epsilon,
                   unsigned int outputOffset,
                   cudnnTensorDescriptor_t outputsTensor,
                   DATA_T** outputs_data,
                   ActivationFunction_T func)
{
#ifdef PROFILING
    double elapsed = 0.0;
    const std::chrono::high_resolution_clock::time_point start
        = std::chrono::high_resolution_clock::now();
#endif
#if CUDNN_VERSION >= 4000

    cudnnActivationMode_t cudnnActivation;

    ((func == Tanh || func == TanhLeCun)
     ? cudnnActivation = CUDNN_ACTIVATION_TANH
     : ((func == Rectifier)
        ? cudnnActivation = CUDNN_ACTIVATION_RELU
        : ((func == FastSigmoid) ? cudnnActivation = CUDNN_ACTIVATION_SIGMOID
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
    /********BATCHNORM FORWARD***********/
    CHECK_CUDNN_STATUS(cudnnBatchNormalizationForwardInference(context_handle,
                                                               mMode,
                                                               &ONE_T,
                                                               &ZERO_T,
                                                               inputsTensor,
                                                               inputs_data,
                                                               outputsTensor,
                                                               *outputs_data,
                                                               scaleDesc,
                                                               scale,
                                                               bias,
                                                               mean,
                                                               variance,
                                                               epsilon));

#else
    cudaSBNPropagate(inputs_data,
                     bias,
                     variance,
                     mean,
                     scale,
                     epsilon,
                     &outputs_data + outputOffset,
                     nbChannels,
                     channelsHeight,
                     channelsWidth,
                     batchSize);

#endif

    if (func != Linear) {
        /********CONVOLUTION ACTIVATION***********/
        CHECK_CUDNN_STATUS(
            cudnnActivationForward(context_handle,
                                   activationDesc,
                                   &ONE_T,
                                   outputsTensor,
                                   *outputs_data + outputOffset,
                                   &ZERO_T,
                                   outputsTensor,
                                   *outputs_data + outputOffset)); // Activation
    }

#ifdef PROFILING
    CHECK_CUDA_STATUS(cudaDeviceSynchronize());
    elapsed = 1.0e6
              * std::chrono::duration_cast<std::chrono::duration<double> >(
                    std::chrono::high_resolution_clock::now() - start).count();
    oclHandles.events.push_back(elapsed);
#endif
}

/************************************POOLING*****************************************************/
/************************************************************************************************/
void setPooling(unsigned int batchSize,
                unsigned int nbChannels,
                unsigned int channelsHeight,
                unsigned int channelsWidth,
                unsigned int strideY,
                unsigned int strideX,
                unsigned int outputHeight,
                unsigned int outputWidth,
                cudnnTensorFormat_t context_tensorFormat,
                cudnnDataType_t context_dataType,
                cudnnTensorDescriptor_t inputsTensor,
                cudnnTensorDescriptor_t outputsTensor,
                Pooling_T func,
                cudnnPoolingMode_t& cudnnPooling,
                unsigned int nbOutputs,
                unsigned int outputOffset,
                unsigned int poolHeight,
                unsigned int poolWidth,
                cudnnPoolingDescriptor_t mapping)
{
    int n = batchSize;
    int c = nbChannels;
    int h = channelsHeight;
    int w = channelsWidth;

#ifdef PROFILING
    oclHandles.profiling.push_back(oclProfiling({"Pooling", 0.0}));
#endif

    ((func == Average)
     ? cudnnPooling = CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING
     : ((func == Max) ? cudnnPooling = CUDNN_POOLING_MAX : cudnnPooling
        = CUDNN_POOLING_MAX));

#if CUDNN_VERSION >= 5000
    CHECK_CUDNN_STATUS(cudnnSetPooling2dDescriptor(mapping,
                                                   cudnnPooling,
                                                   NanPolicy,
                                                   poolHeight,
                                                   poolWidth,
                                                   0,
                                                   0,
                                                   strideY,
                                                   strideX));
#else
    CHECK_CUDNN_STATUS(cudnnSetPooling2dDescriptor(
        mapping,
        cudnnPooling,
        poolHeight,
        poolWidth,
        0,
        0,
        strideX, // BUG in cuDNN v3 (order of the last 2 arguments was
        // inverted), resolved with cuDNN v5
        strideY));
#endif

    CHECK_CUDNN_STATUS(cudnnSetTensor4dDescriptor(
        inputsTensor, context_tensorFormat, context_dataType, n, c, h, w));

    CHECK_CUDNN_STATUS(cudnnSetTensor4dDescriptor(outputsTensor,
                                                  context_tensorFormat,
                                                  context_dataType,
                                                  n,
                                                  nbOutputs,
                                                  outputHeight,
                                                  outputWidth));
}
void poolcell(cudnnHandle_t& context_handle,
              cudnnTensorDescriptor_t inputsTensor,
              DATA_T* inputs_data,
              unsigned int outputOffset,
              cudnnTensorDescriptor_t outputsTensor,
              DATA_T** outputs_data,
              cudnnPoolingDescriptor_t mapping)
{
#ifdef PROFILING
    double elapsed = 0.0;
    const std::chrono::high_resolution_clock::time_point start
        = std::chrono::high_resolution_clock::now();
#endif

    DATA_T ONE_T = DATA_T(1); // Alpha must be set to 1 for all steps
    DATA_T ZERO_T = DATA_T(0); // Beta must be set to 0 for POOLING FORWARD
    /********POOLING FORWARD***********/
    CHECK_CUDNN_STATUS(cudnnPoolingForward(context_handle,
                                           mapping,
                                           &ONE_T,
                                           inputsTensor,
                                           inputs_data,
                                           &ZERO_T,
                                           outputsTensor,
                                           *outputs_data + outputOffset));

#ifdef PROFILING
    CHECK_CUDA_STATUS(cudaDeviceSynchronize());
    elapsed = 1.0e6
              * std::chrono::duration_cast<std::chrono::duration<double> >(
                    std::chrono::high_resolution_clock::now() - start).count();
    oclHandles.events.push_back(elapsed);
#endif
}
/************************************FRACTIONNAL MAX
 * POOLING*************************************/
/************************************************************************************************/
void setFmp()
{
#ifdef PROFILING
    oclHandles.profiling.push_back(oclProfiling({"FMP", 0.0}));
#endif
}

void fmpcell(cudnnHandle_t& context_handle,
             unsigned int batchSize,
             unsigned int nbChannels,
             unsigned int channelsHeight,
             unsigned int channelsWidth,
             unsigned int* gridx,
             unsigned int* gridy,
             const bool overlapping,
             const DATA_T* inputs_data,
             unsigned int nbOutputs_,
             unsigned int outputsHeight,
             unsigned int outputsWidth,
             unsigned int nbOutputs,
             unsigned int outputOffset,
             DATA_T* outputs_data)
{
#ifdef PROFILING
    double elapsed = 0.0;
    const std::chrono::high_resolution_clock::time_point start
        = std::chrono::high_resolution_clock::now();
#endif
    unsigned int* gridXRand
        = new unsigned int[outputsWidth]; // use array new.  Note that length
    // does not need to be constant!
    unsigned int* gridYRand
        = new unsigned int[outputsHeight]; // use array new.  Note that length
    // does not need to be constant!
    fmpcell_propagate_generateRegions(
        &gridXRand[0], channelsWidth, outputsWidth);
    fmpcell_propagate_generateRegions(
        &gridYRand[0], channelsHeight, outputsHeight);
    CHECK_CUDA_STATUS(cudaMemcpy(gridx,
                                 gridXRand,
                                 outputsWidth * sizeof(unsigned int),
                                 cudaMemcpyHostToDevice));
    CHECK_CUDA_STATUS(cudaMemcpy(gridy,
                                 gridYRand,
                                 outputsHeight * sizeof(unsigned int),
                                 cudaMemcpyHostToDevice));

    cudaSFMPPropagate(inputs_data,
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
#ifdef PROFILING
    CHECK_CUDA_STATUS(cudaDeviceSynchronize());
    elapsed = 1.0e6
              * std::chrono::duration_cast<std::chrono::duration<double> >(
                    std::chrono::high_resolution_clock::now() - start).count();
    oclHandles.events.push_back(elapsed);
#endif
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

/************************************FULLY
 * CONNECTED*********************************************/
/************************************************************************************************/
void setFc(unsigned int batchSize,
           unsigned int nbChannels,
           unsigned int channelsHeight,
           unsigned int channelsWidth,
           cudnnHandle_t& context_handle,
           cudnnTensorFormat_t context_tensorFormat,
           cudnnDataType_t context_dataType,
           cudnnTensorDescriptor_t inputsTensor,
           cudnnTensorDescriptor_t outputsTensor,
           ActivationFunction_T func,
           unsigned int nbOutputs,
           cudnnTensorDescriptor_t biasDesc)
{
    int n = batchSize;
    int c = nbChannels;
    int hCh = channelsHeight;
    int wCh = channelsWidth;
    int k = nbOutputs;
    int hOut = 1;
    int wOut = 1;

#ifdef PROFILING
    oclHandles.profiling.push_back(oclProfiling({"FullyConnected", 0.0}));
#endif

    CHECK_CUDNN_STATUS(cudnnSetTensor4dDescriptor(
        biasDesc, context_tensorFormat, context_dataType, 1, k, 1, 1));

    CHECK_CUDNN_STATUS(cudnnSetTensor4dDescriptor(
        inputsTensor, context_tensorFormat, context_dataType, n, c, hCh, wCh));

    CHECK_CUDNN_STATUS(cudnnSetTensor4dDescriptor(outputsTensor,
                                                  context_tensorFormat,
                                                  context_dataType,
                                                  n,
                                                  k,
                                                  hOut,
                                                  wOut));
}

void fullyConnected(unsigned int batchSize,
                    unsigned int nbChannels,
                    cudnnHandle_t& context_handle,
                    cublasHandle_t& context_cublasHandle,
                    ActivationFunction_T func,
                    cudnnTensorDescriptor_t inputsTensor,
                    DATA_T* inputs_data,
                    cudnnTensorDescriptor_t outputsTensor,
                    unsigned int nbOutputs,
                    unsigned int outputOffset,
                    int noBias,
                    DATA_T** outputs_data,
                    cudnnTensorDescriptor_t biasDesc,
                    DATA_T* bias_data,
                    DATA_T* ones_vec_data,
                    DATA_T* weights_data)
{
#ifdef PROFILING
    double elapsed = 0.0;
    const std::chrono::high_resolution_clock::time_point start
        = std::chrono::high_resolution_clock::now();
#endif

    /************************CublasSgemm function used for the Fully Connected
     * Layers************************************/
    /********************************************************************************************************************
      This function performs the matrix-matrix multiplication

    C = α*op( A )*op*( B ) + β*C

    where α and β are scalars, and A , B and C are matrices stored in
    column-major format with dimensions op ( A ) m × k , op ( B ) k × n and C m
    × n , respectively.
    Also, for matrix A:
            op ( A ) = A if   transa == CUBLAS_OP_N
                 = AT if  transa == CUBLAS_OP_T
                 = AH if  transa == CUBLAS_OP_C



        More informations: http://docs.nvidia.com/cuda/cublas/
    ***************************************************************************************************************************/
    DATA_T ONE_T
        = DATA_T(1); // Alpha must be set to 1 for all fully connected steps
    DATA_T ZERO_T
        = DATA_T(0); // Beta must be set to 0 for cublasSgemv processing

    cudnnActivationMode_t cudnnActivation;

    ((func == Tanh || func == TanhLeCun)
     ? cudnnActivation = CUDNN_ACTIVATION_TANH
     : ((func == Rectifier)
        ? cudnnActivation = CUDNN_ACTIVATION_RELU
        : ((func == FastSigmoid) ? cudnnActivation = CUDNN_ACTIVATION_SIGMOID
                                 : cudnnActivation = CUDNN_ACTIVATION_RELU)));

#if CUDNN_VERSION >= 5000
    cudnnActivationDescriptor_t activationDesc;
    CHECK_CUDNN_STATUS(cudnnCreateActivationDescriptor(&activationDesc));

    CHECK_CUDNN_STATUS(cudnnSetActivationDescriptor(
        activationDesc, cudnnActivation, NanPolicy, 0.0));
#else
    cudnnActivationMode_t activationDesc = cudnnActivation;
#endif

    cublasOperation_t transA = CUBLAS_OP_T; // Operation op(A)
    cublasOperation_t transB = CUBLAS_OP_N; // Operation op(B)
    int m = nbOutputs; // Number of rows of matrix op(A) and C
    int n = batchSize; // Number of columns of matrix op(B) and C
    int k = nbChannels; // Number of columns of matrix op(A) and rows of matrix
    // op(B)

    int ldA = k; // Leading dimension of two-dimensional array used to store
    // matrix A.
    int ldB = k; // Leading dimension of two-dimensional array used to store
    // matrix B.
    int ldC = m; // Leading dimension of two-dimensional array used to store
    // matrix C.

    cublasSgemm(context_cublasHandle,
                transA,
                transB,
                m,
                n,
                k,
                &ONE_T,
                weights_data,
                ldA,
                inputs_data,
                ldB,
                &ZERO_T,
                *outputs_data + outputOffset,
                ldC);

    /******************************************************************************************************************************/
    if (!noBias) {
        /*************************ADD BIAS to FC
         * Layers*********************************/
        CHECK_CUBLAS_STATUS(cublasSgemm(context_cublasHandle,
                                        CUBLAS_OP_N,
                                        CUBLAS_OP_N,
                                        m,
                                        n,
                                        1,
                                        &ONE_T,
                                        bias_data,
                                        m,
                                        ones_vec_data,
                                        1,
                                        &ONE_T,
                                        *outputs_data + outputOffset,
                                        m));
    }

    if (func != Linear) {
        /*************************Activation
         * Function*********************************/
        CHECK_CUDNN_STATUS(
            cudnnActivationForward(context_handle,
                                   activationDesc,
                                   &ONE_T,
                                   outputsTensor,
                                   *outputs_data + outputOffset,
                                   &ZERO_T,
                                   outputsTensor,
                                   *outputs_data + outputOffset)); // Activation
    }

#ifdef PROFILING
    CHECK_CUDA_STATUS(cudaDeviceSynchronize());
    elapsed = 1.0e6
              * std::chrono::duration_cast<std::chrono::duration<double> >(
                    std::chrono::high_resolution_clock::now() - start).count();
    oclHandles.events.push_back(elapsed);
#endif
}

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

#ifdef PROFILING
    oclHandles.profiling.push_back(oclProfiling({"SoftMax", 0.0}));
#endif

    CHECK_CUDNN_STATUS(cudnnSetTensor4dDescriptor(
        inputsTensor, context_tensorFormat, context_dataType, n, c, h, w));

    CHECK_CUDNN_STATUS(cudnnSetTensor4dDescriptor(
        outputsTensor, context_tensorFormat, context_dataType, n, c, h, w));
}

/************************SoftMax
 * Layer*****************************************************************/
void softmax(cudnnHandle_t& context_handle,
             cudnnTensorDescriptor_t inputsTensor,
             DATA_T* inputs_data,
             cudnnTensorDescriptor_t outputsTensor,
             DATA_T** outputs_data)
{
#ifdef PROFILING
    double elapsed = 0.0;
    const std::chrono::high_resolution_clock::time_point start
        = std::chrono::high_resolution_clock::now();
#endif

    DATA_T alpha = DATA_T(1);
    DATA_T beta = DATA_T(0);

    CHECK_CUDNN_STATUS(cudnnSoftmaxForward(context_handle,
                                           CUDNN_SOFTMAX_ACCURATE,
                                           CUDNN_SOFTMAX_MODE_CHANNEL,
                                           &alpha,
                                           inputsTensor,
                                           inputs_data,
                                           &beta,
                                           outputsTensor,
                                           *outputs_data));

#ifdef PROFILING
    CHECK_CUDA_STATUS(cudaDeviceSynchronize());
    elapsed = 1.0e6
              * std::chrono::duration_cast<std::chrono::duration<double> >(
                    std::chrono::high_resolution_clock::now() - start).count();
    oclHandles.events.push_back(elapsed);
#endif
}

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

#ifdef PROFILING
    for (std::vector<double>::iterator it = oclHandles.events.begin(),
                                       itBegin = oclHandles.events.begin(),
                                       itEnd = oclHandles.events.end();
         it != itEnd;
         ++it) {
        oclHandles.profiling[it - itBegin].processTime += (*it);
    }

    oclHandles.events.clear();
#endif

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

void spatial_output_generation(unsigned int batchSize,
                               unsigned int nbOutputs,
                               unsigned int outputsHeight,
                               unsigned int outputsWidth,
                               DATA_T* dataIn,
                               uint32_t* outputEstimated)
{
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
#ifdef PROFILING
    for (std::vector<double>::iterator it = oclHandles.events.begin(),
                                       itBegin = oclHandles.events.begin(),
                                       itEnd = oclHandles.events.end();
         it != itEnd;
         ++it) {
        oclHandles.profiling[it - itBegin].processTime += (*it);
    }

    oclHandles.events.clear();
#endif
    for (unsigned int i = 0; i < batchSize; i++) {
        for (unsigned int oy = 0; oy < outputsHeight; ++oy) {
            for (unsigned int ox = 0; ox < outputsWidth; ++ox) {
                const unsigned int inputsIdx
                    = ox + oy * outputsWidth
                      + i * (outputsHeight * outputsWidth * nbOutputs);
                DATA_T maxVal = outputsData[inputsIdx];
                unsigned int outputMax = 0;
                for (unsigned int output = 1; output < nbOutputs; ++output) {
                    const unsigned int outputsIdx
                        = ox + (oy + output * outputsHeight) * outputsWidth
                          + i * (outputsHeight * outputsWidth * nbOutputs);
                    if (outputsData[outputsIdx] > maxVal) {
                        outputMax = output;
                        maxVal = outputsData[outputsIdx];
                    }
                }
                outputEstimated[ox + oy * outputsWidth
                                + i * (outputsHeight * outputsWidth)]
                    = outputMax;
            }
        }
    }

    delete[] outputsData;
}

void confusion_print(unsigned int nbOutputs, unsigned int* confusion)
{
    std::cout << "\nConfusion matrix:\n";
    std::cout << std::string(9 + 10 * nbOutputs, '-') << "\n";
    std::cout << "| T \\ E |";

    for (unsigned int estimated = 0; estimated < nbOutputs; ++estimated)
        std::cout << " " << std::setfill(' ') << std::setw(7) << estimated
                  << " |";

    std::cout << "\n" << std::string(9 + 10 * nbOutputs, '-') << "\n";

    unsigned int total = 0;
    unsigned int totalCorrect = 0;

    for (unsigned int target = 0; target < nbOutputs; ++target) {
        unsigned int targetCount = 0;

        for (unsigned int estimated = 0; estimated < nbOutputs; ++estimated)
            targetCount += confusion[estimated + target * nbOutputs];

        total += targetCount;
        totalCorrect += confusion[target + target * nbOutputs];

        std::cout << "| " << std::setfill(' ') << std::setw(5) << target
                  << " |";

        for (unsigned int estimated = 0; estimated < nbOutputs; ++estimated)
            std::cout << " " << std::setfill(' ') << std::setw(7)
                      << confusion[estimated + target * nbOutputs] << " |";

        std::cout << "\n";
        std::cout << "|       |";

        for (unsigned int estimated = 0; estimated < nbOutputs; ++estimated) {
            std::cout << " " << ESC_BG_LIGHT_YELLOW << std::setfill(' ')
                      << std::setw(6) << std::fixed << std::setprecision(2)
                      << 100.0
                         * ((targetCount > 0)
                                ? (confusion[estimated + target * nbOutputs]
                                   / (double)targetCount)
                                : 0.0) << "%" << ESC_ALL_OFF << " |";
        }
        std::cout << "\n";
    }

    std::cout << std::string(9 + 10 * nbOutputs, '-') << "\n"
              << "T: Target    E: Estimated" << std::endl;
}

void dumpMem(int size, DATA_T* data, std::string fileName)
{

    std::ofstream file;
    file.open(fileName.c_str());

    DATA_T* watch_eagle(NULL);
    watch_eagle = new DATA_T[size];

    CHECK_CUDA_STATUS(cudaMemcpy(
        watch_eagle, data, size * sizeof(DATA_T), cudaMemcpyDeviceToHost));

    for (int i = 0; i < size; i++)
#if NB_BITS < 0
        file << "data[" << i << "]= " << watch_eagle[i] << "\n";
#else
        file << "data[" << i << "]= " << (int)watch_eagle[i] << "\n";
#endif
    std::cout << "dump mem in file " << fileName.c_str() << "done"
              << "\n";
    file.close();
    delete[] watch_eagle;
}
