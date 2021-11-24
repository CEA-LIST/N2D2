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
#include "PluginLayers.hpp"
#include "NetworkTensorRT.hpp"

#ifndef ONNX
#include "../dnn/include/env.hpp"
#endif

PluginFactory mPluginFactory;

N2D2::Network::Network()
{
    //ctor
}

void N2D2::Network::setProfiling()
{
    mContext->setProfiler(&gProfiler);
}

void N2D2::Network::setTensorRTPrecision() {
    if(mNbBits == -32) {
        mDataType = nvinfer1::DataType::kFLOAT;
        std::cout << "TensorRT DataType is now set to : kFLOAT" << std::endl;
    }
    else if(mNbBits == -16) {
        mDataType = nvinfer1::DataType::kHALF;
        std::cout << "TensorRT DataType is now set to : kHALF" << std::endl;
    }
    else if(mNbBits == 8) {
        mDataType = nvinfer1::DataType::kFLOAT;
        std::cout << "INT8 Mode is set ON, TensorRT DataType is now set to : kFLOAT" << std::endl;
    }
    else {
        throw std::runtime_error(
                    "TensorRT doesn't support this Data Precision ");
    }

}

#ifndef ONNX
void N2D2::Network::setInternalDimensions() {
    std::cout << "INPUTS/OUTPUTS Dimensions set as follow :" << std::endl;

    setInputDims(ENV_SIZE_X, ENV_SIZE_Y, ENV_NB_OUTPUTS);

    std::cout << "Inputs Dimensions { " << getInputDimX() << ", "
                << getInputDimY() << ", " << getInputDimZ() << "}" 
                << std::endl;

    setOutputNbTargets(NETWORK_TARGETS);
    std::cout << "Outputs Dimensions with NbTargets: " << NETWORK_TARGETS << std::endl;
    for(unsigned int target = 0; target < getOutputNbTargets(); ++target)
    {
        setOutputTarget(NB_TARGET[target], 
                        NB_OUTPUTS[target], 
                        OUTPUTS_HEIGHT[target], 
                        OUTPUTS_WIDTH[target], 
                        target);

        std::cout << "=====> Target " << target << " with output targets "
            << getOutputTarget(target) << " have dimensions " 
            << "{" << getOutputDimX(target) << ", " << getOutputDimY(target) << ", "
            << getOutputDimZ(target) << "}" << std::endl;
    }
}
#endif

void N2D2::Network::initialize() {
    std::cout << "==== INITIALIZE ==== " << std::endl;
    cudaSetDevice(mDeviceID);
#ifndef ONNX
    setInternalDimensions();
#endif
    setIOMemory();
    std::cout << "====> Set device I/O Memory Workspace" << std::endl;
    if(mInputEngine.empty()){
        setTensorRTPrecision();
        std::cout << "====> Set TensorRT Precision" << std::endl;
        mNetBuilder = nvinfer1::createInferBuilder(gLogger);

#ifndef ONNX
    #if (NV_TENSORRT_MAJOR + NV_TENSORRT_MINOR) > 7
        //To Be Improve : Network Definition FLAGS
        //nvinfer1::NetworkDefinitionCreationFlags creationFlag;
        //creationFlag = 1 << int(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_PRECISION);
        //creationFlag |= 1 << int(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        mNetBuilderConfig = mNetBuilder->createBuilderConfig();
        mNetDef.push_back(mNetBuilder->createNetworkV2(0));
    #else
        mNetDef.push_back(mNetBuilder->createNetwork());
    #endif
#else
    #if NV_TENSORRT_MAJOR > 5
        nvinfer1::NetworkDefinitionCreationFlags creationFlag;
        creationFlag = 1 << int(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_PRECISION);
        creationFlag |= 1 << int(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        mNetDef.push_back(mNetBuilder->createNetworkV2(creationFlag));
    #else
        mNetDef.push_back(mNetBuilder->createNetwork());
    #endif
#endif

#if (NV_TENSORRT_MAJOR + NV_TENSORRT_MINOR) > 7
        if(mDataType == nvinfer1::DataType::kHALF) {
            mNetBuilderConfig->setFlag(nvinfer1::BuilderFlag::kFP16);
        }
#else 
    #if NV_TENSORRT_MAJOR > 4
        if(mDataType == nvinfer1::DataType::kHALF) {
            mNetBuilder->setFp16Mode(true);
        }
    #endif
#endif


#ifdef ONNX
        nvonnxparser::IParser* parser = 
            nvonnxparser::createParser(*mNetDef.back(), 
                                            gLogger);
        parser->parseFromFile(mONNXmodel.c_str(), 1);
        std::cout << "Set output the layer number : " << mNetDef.back()->getNbLayers() - 1 << std::endl;

        auto last_layer = mNetDef.back()->getLayer(mNetDef.back()->getNbLayers() - 1);
        std::cout << "Last layer name: " << last_layer->getName() << std::endl;
        //auto last_tensor = last_layer->getOutput(0);
        //mNetDef.back()->markOutput(*last_tensor);

        std::cout << "====> ONNX Model Network Description"
                << " Load from file : " << mONNXmodel << std::endl;
#else
        networkDefinition();

        std::cout << "====> Network Description  load from "
            << " internal N2D2 definition" << std::endl;
#endif
        std::cout << "====> Set TensorRT Network Definition done" << std::endl;
    }
    createContext();
    std::cout << "====> Set TensorRT context done" << std::endl;
}

void N2D2::Network::setIOMemory() {
    //Add +1 for Input buffer
    mInOutBuffer.resize(1U + mTargetsDimensions.size());
    size_t InputBufferSize 
        = mInputDimensions.d[0]*mInputDimensions.d[1]*mInputDimensions.d[2]*mMaxBatchSize*sizeof(float);
    CHECK_CUDA_STATUS( cudaMalloc(&mInOutBuffer[0], InputBufferSize) );

    for(size_t i = 0; i < mTargetsDimensions.size(); ++i) {
        size_t buffSize = mMaxBatchSize*mTargetsDimensions[i].d[0]*mTargetsDimensions[i].d[1]*mTargetsDimensions[i].d[2] * sizeof(float);
        CHECK_CUDA_STATUS( cudaMalloc(&mInOutBuffer[1 + i], buffSize));
    }

    //optional, usefull for acceleration of semantic map
    CHECK_CUDA_STATUS( cudaMalloc(&mWorkspaceGPU,
                                  mMaxBatchSize*mInputDimensions.d[1]*mInputDimensions.d[2]*3*sizeof(unsigned char)));
    CHECK_CUDA_STATUS(cudaStreamCreate(&mDataStream));
}


void N2D2::Network::createContext()
{
    mCudaEngine = nullptr;

    std::cout << "Start createContext" << std::endl;
    if(mInputEngine.empty())
    {
#if (NV_TENSORRT_MAJOR + NV_TENSORRT_MINOR) > 7

        mNetBuilder->setMaxBatchSize(mMaxBatchSize);
        mNetBuilderConfig->setMinTimingIterations(mIterBuild);
        mNetBuilderConfig->setMaxWorkspaceSize(128<<20);
        if(mDataType == nvinfer1::DataType::kHALF) {
            mNetBuilderConfig->setFlag(nvinfer1::BuilderFlag::kFP16);
        }
        if(mNbBits == 8)
        {
            std::string calibDir = mCalibrationFolder.empty() ?  
                                    "./batches_calib/"
                                    : mCalibrationFolder;
            std::vector<std::string> filesCalib;
            struct dirent* pFile;
            DIR* pDir = opendir(calibDir.c_str());
            if (pDir == NULL) {
                //throw std::runtime_error(
                //    "Couldn't open the directory for input patterns: " + calibDir);
                std::cout << "No directory for batches calibration" << std::endl;
            }
            else {
              while ((pFile = readdir(pDir)) != NULL) {
                  if (pFile->d_name[0] != '.')
                      filesCalib.push_back(std::string(calibDir + pFile->d_name));
              }
              closedir(pDir);
            }
            unsigned int nbCalibFiles = filesCalib.size();
            if(nbCalibFiles == 0)
                //throw std::runtime_error("Cannot find calibration files in dir " + calibDir);
                std::cout << "Cannot find calibration files in dir " << calibDir << std::endl;

            std::cout << "Using Entropy Calibrator" << std::endl;
            BatchStream calibrationStream(  1, //batchsize
                                            getInputDimZ(), 
                                            getInputDimY(), 
                                            getInputDimX(), 
                                            nbCalibFiles, 
                                            calibDir + "batch_calibration");

            mCalibrator.reset(new Int8EntropyCalibrator(calibrationStream, 0, mCalibrationCacheName));
            mNetBuilderConfig->setFlag(nvinfer1::BuilderFlag::kINT8);
            mNetBuilderConfig->setInt8Calibrator(mCalibrator.get());
#else

        mNetBuilder->setMaxBatchSize(mMaxBatchSize);
        mNetBuilder->setMinFindIterations(mIterBuild);
        mNetBuilder->setMaxWorkspaceSize(128<<20);
#if NV_TENSORRT_MAJOR < 5
        if(mDataType == nvinfer1::DataType::kHALF)
            mNetBuilder->setHalf2Mode(true);
#else
        if(mDataType == nvinfer1::DataType::kHALF)
            mNetBuilder->setFp16Mode(true);
#endif
        if(mNbBits == 8)
        {
#if NV_TENSORRT_MAJOR > 2

            std::string calibDir = mCalibrationFolder.empty() ?  
                                    "./batches_calib/"
                                    : mCalibrationFolder;
            std::vector<std::string> filesCalib;
            struct dirent* pFile;
            DIR* pDir = opendir(calibDir.c_str());
            if (pDir == NULL) {
                //throw std::runtime_error(
                //    "Couldn't open the directory for input patterns: " + calibDir);
                std::cout << "No directory for batches calibration" << std::endl;
            }
            else {
              while ((pFile = readdir(pDir)) != NULL) {
                  if (pFile->d_name[0] != '.')
                      filesCalib.push_back(std::string(calibDir + pFile->d_name));
              }
              closedir(pDir);
            }
            unsigned int nbCalibFiles = filesCalib.size();
            if(nbCalibFiles == 0)
                //throw std::runtime_error("Cannot find calibration files in dir " + calibDir);
                std::cout << "Cannot find calibration files in dir " << calibDir << std::endl;

            std::cout << "Using Entropy Calibrator" << std::endl;
            BatchStream calibrationStream(  1, //batchsize
                                            getInputDimZ(), 
                                            getInputDimY(), 
                                            getInputDimX(), 
                                            nbCalibFiles, 
                                            calibDir + "batch_calibration");

            mCalibrator.reset(new Int8EntropyCalibrator(calibrationStream, 0, mCalibrationCacheName));
            mNetBuilder->setInt8Mode(true);
            mNetBuilder->setInt8Calibrator(mCalibrator.get());
#endif
#endif

        }
#if (NV_TENSORRT_MAJOR + NV_TENSORRT_MINOR) > 7
        mCudaEngine = mNetBuilder->buildEngineWithConfig(*mNetDef.back(), *mNetBuilderConfig);
        mNetBuilderConfig->destroy();
#else
        mCudaEngine = mNetBuilder->buildCudaEngine(*mNetDef.back());
#endif
        std::cout << "buildCudaEngine done" << std::endl;
        mNetBuilder->destroy();

    }
    else
    {
        std::ifstream cudaEngineStream(mInputEngine);

        if(!cudaEngineStream.good())
            throw std::runtime_error("Could not open cuda engine file: " + mInputEngine);

        nvinfer1::IRuntime* infer = nvinfer1::createInferRuntime(gLogger);
#if NV_TENSORRT_MAJOR > 1
        // support for stringstream deserialization was deprecated in TensorRT v2
        // instead, read the stringstream into a memory buffer and pass that to TRT.
        cudaEngineStream.seekg(0, std::ios::end);
        const int modelSize = cudaEngineStream.tellg();
        cudaEngineStream.seekg(0, std::ios::beg);
        void* modelMem = malloc(modelSize);
        if( !modelMem )
            throw std::runtime_error("Could not allocate enough memory for load cuda engine file " + mInputEngine);

        cudaEngineStream.read((char*)modelMem, modelSize);
        mCudaEngine = infer->deserializeCudaEngine(modelMem, 
                                                  modelSize, 
                                                  &mPluginFactory);
        free(modelMem);
#else
       // TensorRT v1 can deserialize directly from stringstream
       mCudaEngine = infer->deserializeCudaEngine(cudaEngineStream);
#endif
    }
        std::cout << "mCudaEngine->serialize" << std::endl;

    nvinfer1::IHostMemory *gieModelStream = mCudaEngine->serialize();
        std::cout << "serialize done" << std::endl;

    if(!mOutputEngine.empty())
    {
        std::cout << "Save cuda engine file at " << mOutputEngine << std::endl;
        std::ofstream engineSerializedFile; //Create output file stream
        engineSerializedFile.open(mOutputEngine, std::ios::out | std::ios::binary); // Open a new file

        if (engineSerializedFile.is_open() && engineSerializedFile.good() && !engineSerializedFile.fail()) {
           //Save the serialized engine data into the file
           engineSerializedFile.write(reinterpret_cast<const char *>(gieModelStream->data()), gieModelStream->size());
           engineSerializedFile.close();
        }
        else
            throw std::runtime_error("Could not save cuda engine file at  " + mOutputEngine);

    }

    mCudaEngine->destroy();
#if NV_TENSORRT_MAJOR > 2
    // Once the engine is built. Its safe to destroy the mCalibrator.
    mCalibrator.reset();
#endif

    mPluginFactory.destroyPlugin();
    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);
    nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(gieModelStream->data(),
                                                                   gieModelStream->size(),
                                                                   &mPluginFactory);
#if NV_TENSORRT_MAJOR > 4
    if(runtime->getNbDLACores() > 1)
        runtime->setDLACore(runtime->getNbDLACores() - 1) ;

    std::cout << "Available DLA Cores / Used DLA Cores: " << runtime->getNbDLACores() << " / " << runtime->getDLACore() << std::endl;
#endif



    if (gieModelStream)
        gieModelStream->destroy();

    mContext = engine->createExecutionContext();

    mPluginFactory.destroyPlugin();
}

/*
void N2D2::Network::output(uint32_t* out_data, unsigned int target) {

   spatial_output_generation(   mMaxBatchSize,
                                NB_OUTPUTS[target],
                                OUTPUTS_HEIGHT[target],
                                OUTPUTS_WIDTH[target],
                                mInOutBuffer[target + 1],
                                out_data,
                                mDataStream);
}
*/

void N2D2::Network::estimated(uint32_t* out_data, unsigned int target, bool useGPU, float threshold) {

   spatial_output_generation(mMaxBatchSize,
                            mTargetsDimensions[target].d[1],
                            mTargetsDimensions[target].d[2],
                            mTargetsDimensions[target].d[3],
                            mInOutBuffer[target + 1],
                            out_data,
                            mDataStream,
                            threshold,
                            useGPU);
}

void N2D2::Network::log_output(float* out_data, unsigned int target) {

   get_output(  mTargetsDimensions[target].d[1],
                mTargetsDimensions[target].d[2],
                mTargetsDimensions[target].d[3],
                mInOutBuffer[target + 1],
                out_data);
}

void N2D2::Network::addOverlay(unsigned char* overlay_data, unsigned int target, float alpha) {

   add_weighted(mTargetsDimensions[target].d[1],
                mTargetsDimensions[target].d[2],
                mTargetsDimensions[target].d[3],
                reinterpret_cast<float *>(mInOutBuffer[target + 1]),
                mInputDimensions.d[0],
                mInputDimensions.d[1],
                mInputDimensions.d[2],
                reinterpret_cast<float *>(mInOutBuffer[0]),
                overlay_data,
                alpha);
}


void* N2D2::Network::getDevicePtr(unsigned int target) {

    return mInOutBuffer[target + 1];
}



void N2D2::Network::add_target(std::vector<nvinfer1::ITensor *> outputs_tensor,
                unsigned int targetIdx)
{
    for(unsigned int i = 0; i < outputs_tensor.size(); ++i)
    {
        std::string target_name = "Target_" + std::to_string(targetIdx)
                                    + "_" + std::to_string(i);
        outputs_tensor[i]->setType(nvinfer1::DataType::kFLOAT);
        outputs_tensor[i]->setName(target_name.c_str());

        mNetDef.back()->markOutput(*outputs_tensor[i]);
    }
}

std::vector<nvinfer1::ITensor *>
      N2D2::Network::add_activation(std::string layerName,
                        nvinfer1::ActivationType activation,
                        double alpha,
                        double beta,
                        std::vector<nvinfer1::ITensor *> inputs_tensor)
{
        std::vector<nvinfer1::ITensor *> output_tensor;
        std::cout << "Add activation layer: " << layerName << std::endl;
        for(unsigned int i = 0; i < inputs_tensor.size(); ++i)
        {
            std::string outName = layerName + "_Activation_" + std::to_string(i);
            nvinfer1::ActivationType finalActivation = activation;


            //Special cases for Clip Relu (Relu6) and Leaky Relu:
            if(activation == nvinfer1::ActivationType::kRELU){
//Need a special case for TensorRT 5.0.X while clipped Relu and Leaky Relu are only supported since TensorRT 5.1.0 : 
// ==> https://docs.nvidia.com/deeplearning/tensorrt/release-notes/tensorrt-5.html#rel_5-0-RC
#if ((NV_TENSORRT_MAJOR + NV_TENSORRT_MINOR) > 5) 
                //ReluClipped:               
                if(beta != 0.0 && alpha == 0.0) {
                   finalActivation = nvinfer1::ActivationType::kCLIP;
                 }
                if(alpha != 0.0 && beta == 0.0) {
                    finalActivation = nvinfer1::ActivationType::kLEAKY_RELU;
                }
#endif
            }
            auto layer = mNetDef.back()->addActivation(*inputs_tensor[i],
                                            finalActivation);

            layer->setAlpha(alpha);
            layer->setBeta(beta);
            layer->setName(outName.c_str());
#if NV_TENSORRT_MAJOR > 4
            if(mUseDLA)
            {
                bool devSuccess = false;
                if((mDataType == nvinfer1::DataType::kHALF 
                        || mDataType == nvinfer1::DataType::kINT8))
                {
    #if (NV_TENSORRT_MAJOR + NV_TENSORRT_MINOR) > 7
                    if(mNetBuilderConfig->canRunOnDLA(layer)) { 
                        layer->setPrecision(mDataType);
                        std::cout << "Layer: " << layer->getName() 
                                    << " will run on DLA (batch size max: " 
                                    <<  mNetBuilder->getMaxDLABatchSize()
                                    << std::endl;
                        mNetBuilderConfig->setDeviceType(layer, nvinfer1::DeviceType::kDLA);
                        devSuccess = true;
                    }
    #else
                    if(mNetBuilder->canRunOnDLA(layer)) { 
                        layer->setPrecision(mDataType);
                        std::cout << "Layer: " << layer->getName() 
                                    << " will run on DLA (batch size max: " 
                                    <<  mNetBuilder->getMaxDLABatchSize()
                                    << std::endl;
                        mNetBuilder->setDeviceType(layer, nvinfer1::DeviceType::kDLA);
                        devSuccess = true;
                    }
    #endif
                }
                if(!devSuccess)
                    throw std::runtime_error("Cannot use DLA for layer " + outName);
            }
#endif

           output_tensor.push_back(layer->getOutput(0));
           output_tensor.back()->setType(mDataType);
           output_tensor.back()->setName(outName.c_str());
           nvinfer1::Dims tensor_in_dims = inputs_tensor[i]->getDimensions();
           std::cout << "               " << inputs_tensor[i]->getName()
                     << "---> " << output_tensor.back()->getName() << std::endl;

           std::cout << "               ";
           std::cout << "{";
           for(unsigned int d = 0; d < tensor_in_dims.nbDims; ++d)
                std::cout << tensor_in_dims.d[d] << " ";
           std::cout << "} ----> ";

           nvinfer1::Dims tensor_dims = output_tensor.back()->getDimensions();
           std::cout << "{";
           for(unsigned int d = 0; d < tensor_dims.nbDims; ++d)
                std::cout << tensor_dims.d[d] << " ";
           std::cout << "}" << std::endl;        }
        return output_tensor;

}
std::vector<nvinfer1::ITensor *>
        N2D2::Network::add_activation_cell(std::string layerName,
                                            LayerActivation activation,
                                            std::vector<nvinfer1::ITensor *> inputs_tensor)
{
        if(activation.status)
            return(add_activation(layerName,
                                  activation.type,
                                  activation.alpha,
                                  activation.beta,
                                  inputs_tensor));
        else
            return inputs_tensor;

}

std::vector<nvinfer1::ITensor *>
        N2D2::Network::add_convolution(std::string layerName,
                        LayerActivation activation,
                        unsigned int nbOutputs,
                        unsigned int strideX,
                        unsigned int strideY,
                        unsigned int paddingX,
                        unsigned int paddingY,
                        unsigned int kernelW,
                        unsigned int kernelH,
                        std::vector<nvinfer1::ITensor *> inputs_tensor,
                        //const float* weights,
                        std::string wFile,
                        unsigned int weights_size,
                        //const float* bias,
                        std::string bFile,
                        unsigned int bias_size)
{

        std::vector<nvinfer1::ITensor *> output_tensor;
        std::cout << "Add convolution layer: " << layerName << std::endl;

        std::ifstream weights(wFile.c_str());
        if (!weights.good())
                throw std::runtime_error("Could not open synaptic file: "
                                        + wFile);

        float* weight_wdata;
        __half* weight_hdata;

        if(mDataType != nvinfer1::DataType::kHALF)
            weight_wdata = new float[weights_size];
        else
            weight_hdata = new __half[weights_size];

        float w;

        for (unsigned int i = 0; i < weights_size; ++i) {
            if (!(weights >> w))
                throw std::runtime_error( "Error while reading synaptic file: " + wFile);

            if(mDataType != nvinfer1::DataType::kHALF)
                weight_wdata[i] = w;
            else
                weight_hdata[i] = fp16::__float2half(w);
        }
        weights.close();

        std::ifstream bias(bFile.c_str());
        if (!bias.good())
                throw std::runtime_error("Could not open synaptic file: "
                                        + bFile);

        float* bias_wdata;
        __half* bias_hdata;

        if(mDataType != nvinfer1::DataType::kHALF)
            bias_wdata = new float[bias_size];
        else
            bias_hdata = new __half[bias_size];

        float b;

        for (unsigned int i = 0; i < bias_size; ++i) {
            if (!(bias >> b))
                throw std::runtime_error( "Error while reading synaptic file: " + bFile);
            if(mDataType != nvinfer1::DataType::kHALF)
                bias_wdata[i] = b;
            else
                bias_hdata[i] = fp16::__float2half(b);
        }
        bias.close();

        //nvinfer1::Weights weights_trt  = {mDataType, weights, weights_size};
        nvinfer1::Weights weights_trt;

        if(mDataType != nvinfer1::DataType::kHALF)
            weights_trt = {mDataType, weight_wdata, weights_size};
        else
            weights_trt = {mDataType, weight_hdata, weights_size};

        //nvinfer1::Weights bias_trt  = {mDataType, bias, bias_size};
        nvinfer1::Weights bias_trt;
        if(mDataType != nvinfer1::DataType::kHALF)
            bias_trt = {mDataType, bias_wdata, bias_size};
        else
            bias_trt = {mDataType, bias_hdata, bias_size};

        //delete[] bias_data;
        //delete[] weight_data;

        trt_DimsHW kernelDims = {(int) kernelH, (int)kernelW};
        trt_DimsHW strideDims = {(int)strideY, (int)strideX};
        trt_DimsHW paddingDims = {(int)paddingY, (int)paddingX};

        for(unsigned int i = 0; i < inputs_tensor.size(); ++i)
        {
            std::string outName = layerName + "_" + std::to_string(i);
#if NV_TENSORRT_MAJOR < 6
            auto layer = mNetDef.back()->addConvolution(*inputs_tensor[i],
                                             nbOutputs,
                                             kernelDims,
                                             weights_trt,
                                             bias_trt);
            layer->setStride(strideDims);
            layer->setPadding(paddingDims);
#else
            auto layer = mNetDef.back()->addConvolutionNd(*inputs_tensor[i],
                                                            nbOutputs,
                                                            kernelDims,
                                                            weights_trt,
                                                            bias_trt);
            layer->setStrideNd(strideDims);
            layer->setPaddingNd(paddingDims);
#endif
            layer->setName(outName.c_str());
#if NV_TENSORRT_MAJOR > 4
            if(mUseDLA)
            {
                bool devSuccess = false;
                if((mDataType == nvinfer1::DataType::kHALF 
                        || mDataType == nvinfer1::DataType::kINT8))
                {
    #if (NV_TENSORRT_MAJOR + NV_TENSORRT_MINOR) > 7
                    if(mNetBuilderConfig->canRunOnDLA(layer)) { 
                        layer->setPrecision(mDataType);
                        std::cout << "Layer: " << layer->getName() 
                                    << " will run on DLA (batch size max: " 
                                    <<  mNetBuilder->getMaxDLABatchSize()
                                    << std::endl;
                        mNetBuilderConfig->setDeviceType(layer, nvinfer1::DeviceType::kDLA);
                        devSuccess = true;
                    }
    #else
                    if(mNetBuilder->canRunOnDLA(layer)) { 
                        layer->setPrecision(mDataType);
                        std::cout << "Layer: " << layer->getName() 
                                    << " will run on DLA (batch size max: " 
                                    <<  mNetBuilder->getMaxDLABatchSize()
                                    << std::endl;
                        mNetBuilder->setDeviceType(layer, nvinfer1::DeviceType::kDLA);
                        devSuccess = true;
                    }
    #endif
                }
                if(!devSuccess)
                    throw std::runtime_error("Cannot use DLA for layer " + outName);
            }
#endif

            nvinfer1::Dims tensor_in_dims = inputs_tensor[i]->getDimensions();

            output_tensor.push_back(layer->getOutput(0));
            output_tensor.back()->setType(mDataType);
            output_tensor.back()->setName(outName.c_str());


            std::cout << "               " << inputs_tensor[i]->getName()
                     << "---> " << output_tensor.back()->getName() << std::endl;

            std::cout << "               ";
            std::cout << "{";
            for(unsigned int d = 0; d < tensor_in_dims.nbDims; ++d)
                std::cout << tensor_in_dims.d[d] << " ";
            std::cout << "} ----> ";

            nvinfer1::Dims tensor_dims = output_tensor.back()->getDimensions();
            std::cout << "{";
            for(unsigned int d = 0; d < tensor_dims.nbDims; ++d)
                std::cout << tensor_dims.d[d] << " ";
            std::cout << "}" << std::endl;

        }

        if(activation.status)
            return(add_activation(layerName,
                                  activation.type,
                                  activation.alpha,
                                  activation.beta,
                                  output_tensor));
        else
            return output_tensor;

}

std::vector<nvinfer1::ITensor *>
      N2D2::Network::add_deconvolution(std::string layerName,
                        LayerActivation activation,
                        unsigned int nbOutputs,
                        unsigned int strideX,
                        unsigned int strideY,
                        unsigned int paddingX,
                        unsigned int paddingY,
                        unsigned int kernelW,
                        unsigned int kernelH,
                        std::vector<nvinfer1::ITensor *> inputs_tensor,
                        //const float* weights,
                        std::string wFile,
                        unsigned int weights_size,
                        //const float* bias,
                        std::string bFile,
                        unsigned int bias_size)
{

        std::vector<nvinfer1::ITensor *> output_tensor;
        std::cout << "Add deconvolution layer: " << layerName << std::endl;

        std::ifstream weights(wFile.c_str());
        if (!weights.good())
                throw std::runtime_error("Could not open synaptic file: "
                                        + wFile);

        float* weight_wdata;
        __half* weight_hdata;

        if(mDataType != nvinfer1::DataType::kHALF)
            weight_wdata = new float[weights_size];
        else
            weight_hdata = new __half[weights_size];

        float w;

        for (unsigned int i = 0; i < weights_size; ++i) {
            if (!(weights >> w))
                throw std::runtime_error( "Error while reading synaptic file: " + wFile);

            if(mDataType != nvinfer1::DataType::kHALF)
                weight_wdata[i] = w;
            else
                weight_hdata[i] = fp16::__float2half(w);
        }
        weights.close();

        std::ifstream bias(bFile.c_str());
        if (!bias.good())
                throw std::runtime_error("Could not open synaptic file: "
                                        + bFile);

        float* bias_wdata;
        __half* bias_hdata;

        if(mDataType != nvinfer1::DataType::kHALF)
            bias_wdata = new float[bias_size];
        else
            bias_hdata = new __half[bias_size];

        float b;

        for (unsigned int i = 0; i < bias_size; ++i) {
            if (!(bias >> b))
                throw std::runtime_error( "Error while reading synaptic file: " + bFile);
            if(mDataType != nvinfer1::DataType::kHALF)
                bias_wdata[i] = b;
            else
                bias_hdata[i] = fp16::__float2half(b);
        }

        bias.close();

        //nvinfer1::Weights weights_trt  = {mDataType, weights, weights_size};
        nvinfer1::Weights weights_trt;

        if(mDataType != nvinfer1::DataType::kHALF)
            weights_trt = {mDataType, weight_wdata, weights_size};
        else
            weights_trt = {mDataType, weight_hdata, weights_size};

        //nvinfer1::Weights bias_trt  = {mDataType, bias, bias_size};
        nvinfer1::Weights bias_trt;
        if(mDataType != nvinfer1::DataType::kHALF)
            bias_trt = {mDataType, bias_wdata, bias_size};
        else
            bias_trt = {mDataType, bias_hdata, bias_size};
        //delete[] bias_data;
        //delete[] weight_data;

        trt_DimsHW kernelDims = {(int) kernelH, (int)kernelW};
        trt_DimsHW strideDims = {(int)strideY, (int)strideX};
        trt_DimsHW paddingDims = {(int)paddingY, (int)paddingX};

        for(unsigned int i = 0; i < inputs_tensor.size(); ++i)
        {
            std::string outName = layerName + "_" + std::to_string(i);
#if NV_TENSORRT_MAJOR < 6
            auto layer = mNetDef.back()->addDeconvolution(*inputs_tensor[i],
                                             nbOutputs,
                                             kernelDims,
                                             weights_trt,
                                             bias_trt);
            layer->setStride(strideDims);
            layer->setPadding(paddingDims);
#else
            auto layer = mNetDef.back()->addDeconvolutionNd(*inputs_tensor[i],
                                             nbOutputs,
                                             kernelDims,
                                             weights_trt,
                                             bias_trt);
            layer->setStrideNd(strideDims);
            layer->setPaddingNd(paddingDims);
#endif
            layer->setName(outName.c_str());
#if NV_TENSORRT_MAJOR > 4
            if(mUseDLA)
            {
                bool devSuccess = false;
                if((mDataType == nvinfer1::DataType::kHALF 
                        || mDataType == nvinfer1::DataType::kINT8))
                {
    #if (NV_TENSORRT_MAJOR + NV_TENSORRT_MINOR) > 7
                    if(mNetBuilderConfig->canRunOnDLA(layer)) { 
                        layer->setPrecision(mDataType);
                        std::cout << "Layer: " << layer->getName() 
                                    << " will run on DLA (batch size max: " 
                                    <<  mNetBuilder->getMaxDLABatchSize()
                                    << std::endl;
                        mNetBuilderConfig->setDeviceType(layer, nvinfer1::DeviceType::kDLA);
                        devSuccess = true;
                    }
    #else
                    if(mNetBuilder->canRunOnDLA(layer)) { 
                        layer->setPrecision(mDataType);
                        std::cout << "Layer: " << layer->getName() 
                                    << " will run on DLA (batch size max: " 
                                    <<  mNetBuilder->getMaxDLABatchSize()
                                    << std::endl;
                        mNetBuilder->setDeviceType(layer, nvinfer1::DeviceType::kDLA);
                        devSuccess = true;
                    }
    #endif
                }
                if(!devSuccess)
                    throw std::runtime_error("Cannot use DLA for layer " + outName);
            }
#endif

            nvinfer1::Dims tensor_in_dims = inputs_tensor[i]->getDimensions();

            output_tensor.push_back(layer->getOutput(0));
            output_tensor.back()->setType(mDataType);
            output_tensor.back()->setName(outName.c_str());


            std::cout << "               " << inputs_tensor[i]->getName()
                     << "---> " << output_tensor.back()->getName() << std::endl;

            std::cout << "               ";
            std::cout << "{";
            for(unsigned int d = 0; d < tensor_in_dims.nbDims; ++d)
                std::cout << tensor_in_dims.d[d] << " ";
            std::cout << "} ----> ";

            nvinfer1::Dims tensor_dims = output_tensor.back()->getDimensions();
            std::cout << "{";
            for(unsigned int d = 0; d < tensor_dims.nbDims; ++d)
                std::cout << tensor_dims.d[d] << " ";
            std::cout << "}" << std::endl;

        }

        if(activation.status)
            return(add_activation(layerName,
                                  activation.type,
                                  activation.alpha,
                                  activation.beta,
                                  output_tensor));
        else
            return output_tensor;

}


std::vector<nvinfer1::ITensor *>
          N2D2::Network::add_padding(std::string layerName,
                        unsigned int nbOutputs,
                        std::vector<nvinfer1::ITensor *> inputs_tensor,
                        const int pad_top,
                        const int pad_bottom,
                        const int pad_left,
                        const int pad_right)
{
    std::vector<nvinfer1::ITensor *> output_tensor;
    std::cout << "Add paddinglayer: " << layerName << std::endl;
    trt_DimsHW prePad = {pad_top, pad_left};
    trt_DimsHW postPad = {pad_bottom, pad_right};

    for(unsigned int i = 0; i < inputs_tensor.size(); ++i)
    {
        std::string outName = layerName + "_" + std::to_string(i);
#if NV_TENSORRT_MAJOR < 6
        auto layer = mNetDef.back()->addPadding(*inputs_tensor[i],
                                       prePad,
                                       postPad);
#else 
        auto layer = mNetDef.back()->addPaddingNd(*inputs_tensor[i],
                                       prePad,
                                       postPad);
#endif

        layer->setName(outName.c_str());
        output_tensor.push_back(layer->getOutput(0));
        output_tensor.back()->setType(mDataType);
        output_tensor.back()->setName(outName.c_str());

        nvinfer1::Dims tensor_in_dims = inputs_tensor[i]->getDimensions();
        std::cout << "               " << inputs_tensor[i]->getName()
                    << "---> " << output_tensor.back()->getName() << std::endl;

        std::cout << "               ";
        std::cout << "{";
        for(unsigned int d = 0; d < tensor_in_dims.nbDims; ++d)
                std::cout << tensor_in_dims.d[d] << " ";
        std::cout << "} ----> ";

        nvinfer1::Dims tensor_dims = output_tensor.back()->getDimensions();
        std::cout << "{";
        for(unsigned int d = 0; d < tensor_dims.nbDims; ++d)
                std::cout << tensor_dims.d[d] << " ";
        std::cout << "}" << std::endl;
    }

    return output_tensor;
}



std::vector<nvinfer1::ITensor *>
          N2D2::Network::add_lrn(std::string layerName,
                        unsigned int nbOutputs,
                        std::vector<nvinfer1::ITensor *> inputs_tensor,
                        const int windows,
                        const float alpha,
                        const float beta,
                        const float k)
{
    std::vector<nvinfer1::ITensor *> output_tensor;
    std::cout << "Add LRNlayer: " << layerName << std::endl;

    for(unsigned int i = 0; i < inputs_tensor.size(); ++i)
    {
        std::string outName = layerName + "_" + std::to_string(i);
        auto layer = mNetDef.back()->addLRN(*inputs_tensor[i],
                                 windows,
                                 alpha,
                                 beta,
                                 k);

        layer->setName(outName.c_str());

#if NV_TENSORRT_MAJOR > 4
            if(mUseDLA)
            {
                bool devSuccess = false;
                if((mDataType == nvinfer1::DataType::kHALF 
                        || mDataType == nvinfer1::DataType::kINT8))
                {
    #if (NV_TENSORRT_MAJOR + NV_TENSORRT_MINOR) > 7
                    if(mNetBuilderConfig->canRunOnDLA(layer)) { 
                        layer->setPrecision(mDataType);
                        std::cout << "Layer: " << layer->getName() 
                                    << " will run on DLA (batch size max: " 
                                    <<  mNetBuilder->getMaxDLABatchSize()
                                    << std::endl;
                        mNetBuilderConfig->setDeviceType(layer, nvinfer1::DeviceType::kDLA);
                        devSuccess = true;
                    }
    #else
                    if(mNetBuilder->canRunOnDLA(layer)) { 
                        layer->setPrecision(mDataType);
                        std::cout << "Layer: " << layer->getName() 
                                    << " will run on DLA (batch size max: " 
                                    <<  mNetBuilder->getMaxDLABatchSize()
                                    << std::endl;
                        mNetBuilder->setDeviceType(layer, nvinfer1::DeviceType::kDLA);
                        devSuccess = true;
                    }
    #endif
                }
                if(!devSuccess)
                    throw std::runtime_error("Cannot use DLA for layer " + outName);
            }
#endif

        output_tensor.push_back(layer->getOutput(0));
        output_tensor.back()->setName(outName.c_str());

        nvinfer1::Dims tensor_in_dims = inputs_tensor[i]->getDimensions();
        std::cout << "               " << inputs_tensor[i]->getName()
                    << "---> " << output_tensor.back()->getName() << std::endl;

        std::cout << "               ";
        std::cout << "{";
        for(unsigned int d = 0; d < tensor_in_dims.nbDims; ++d)
                std::cout << tensor_in_dims.d[d] << " ";
        std::cout << "} ----> ";

        nvinfer1::Dims tensor_dims = output_tensor.back()->getDimensions();
        std::cout << "{";
        for(unsigned int d = 0; d < tensor_dims.nbDims; ++d)
                std::cout << tensor_dims.d[d] << " ";
        std::cout << "}" << std::endl;
    }

    return output_tensor;
}


std::vector<nvinfer1::ITensor *>
      N2D2::Network::add_fc(std::string layerName,
                LayerActivation activation,
                unsigned int nbOutputs,
                std::vector<nvinfer1::ITensor *> inputs_tensor,
                std::string wFile,
                unsigned int weights_size,
                std::string bFile)
{
        std::vector<nvinfer1::ITensor *> output_tensor;
        std::cout << "Add fully connected layer: " << layerName << std::endl;
        std::ifstream weights(wFile.c_str());
        if (!weights.good())
                throw std::runtime_error("Could not open synaptic file: "
                                        + wFile);

        float* weight_wdata;
        __half* weight_hdata;

        if(mDataType != nvinfer1::DataType::kHALF)
            weight_wdata = new float[weights_size];
        else
            weight_hdata = new __half[weights_size];

        float w;

        for (unsigned int i = 0; i < weights_size; ++i) {
            if (!(weights >> w))
                throw std::runtime_error( "Error while reading synaptic file: " + wFile);

            if(mDataType != nvinfer1::DataType::kHALF)
                weight_wdata[i] = w;
            else
                weight_hdata[i] = fp16::__float2half(w);
        }
        weights.close();

        std::ifstream bias(bFile.c_str());
        if (!bias.good())
                throw std::runtime_error("Could not open synaptic file: "
                                        + bFile);

        float* bias_wdata;
        __half* bias_hdata;

        if(mDataType != nvinfer1::DataType::kHALF)
            bias_wdata = new float[nbOutputs];
        else
            bias_hdata = new __half[nbOutputs];

        float b;

        for (unsigned int i = 0; i < nbOutputs; ++i) {
            if (!(bias >> b))
                throw std::runtime_error( "Error while reading synaptic file: " + bFile);
            if(mDataType != nvinfer1::DataType::kHALF)
                bias_wdata[i] = b;
            else
                bias_hdata[i] = fp16::__float2half(b);
        }
        bias.close();

        //nvinfer1::Weights weights_trt  = {mDataType, weights, weights_size};
        nvinfer1::Weights weights_trt;

        if(mDataType != nvinfer1::DataType::kHALF)
            weights_trt = {mDataType, weight_wdata, weights_size};
        else
            weights_trt = {mDataType, weight_hdata, weights_size};

        //nvinfer1::Weights bias_trt  = {mDataType, bias, bias_size};
        nvinfer1::Weights bias_trt;
        if(mDataType != nvinfer1::DataType::kHALF)
            bias_trt = {mDataType, bias_wdata, nbOutputs};
        else
            bias_trt = {mDataType, bias_hdata, nbOutputs};

        for(unsigned int i = 0; i < inputs_tensor.size(); ++i)
        {
            std::string outName = layerName + "_" + std::to_string(i);
            auto layer = mNetDef.back()->addFullyConnected(*inputs_tensor[i],
                                             nbOutputs,
                                             weights_trt,
                                             bias_trt);

           layer->setName(outName.c_str());
           output_tensor.push_back(layer->getOutput(0));
           output_tensor.back()->setType(mDataType);
           output_tensor.back()->setName(outName.c_str());

           nvinfer1::Dims tensor_in_dims = inputs_tensor[i]->getDimensions();
           std::cout << "               " << inputs_tensor[i]->getName()
                     << "---> " << output_tensor.back()->getName() << std::endl;

           std::cout << "               ";
           std::cout << "{";
           for(unsigned int d = 0; d < tensor_in_dims.nbDims; ++d)
                std::cout << tensor_in_dims.d[d] << " ";
           std::cout << "} ----> ";

           nvinfer1::Dims tensor_dims = output_tensor.back()->getDimensions();
           std::cout << "{";
           for(unsigned int d = 0; d < tensor_dims.nbDims; ++d)
                std::cout << tensor_dims.d[d] << " ";
           std::cout << "}" << std::endl;
        }

        if(activation.status)
            return(add_activation(layerName,
                                  activation.type,
                                  activation.alpha,
                                  activation.beta,
                                  output_tensor));
        else
            return output_tensor;
}


std::vector<nvinfer1::ITensor *>
      N2D2::Network::add_concat(std::string layerName,
                   unsigned int nbInputs,
                    std::vector<std::vector<nvinfer1::ITensor *> *> inputs_tensor)
{
        std::vector<nvinfer1::ITensor *> output_tensor;

        for(unsigned int i = 0; i < inputs_tensor[0]->size(); ++i)
        {
            std::vector<nvinfer1::ITensor *> concat_tensor;

            for(unsigned int k = 0; k < inputs_tensor.size(); ++k)
            {
              nvinfer1::ITensor * input_tensor = (inputs_tensor[k])->data()[i];
              concat_tensor.push_back(input_tensor);
            }

            std::string outName = layerName + "_" + std::to_string(i);

            auto layer = mNetDef.back()->addConcatenation(&concat_tensor[0],
                                             nbInputs);
           layer->setName(outName.c_str());

#if NV_TENSORRT_MAJOR > 4
            if(mUseDLA)
            {
                bool devSuccess = false;
                if((mDataType == nvinfer1::DataType::kHALF 
                        || mDataType == nvinfer1::DataType::kINT8))
                {
    #if (NV_TENSORRT_MAJOR + NV_TENSORRT_MINOR) > 7
                    if(mNetBuilderConfig->canRunOnDLA(layer)) { 
                        layer->setPrecision(mDataType);
                        std::cout << "Layer: " << layer->getName() 
                                    << " will run on DLA (batch size max: " 
                                    <<  mNetBuilder->getMaxDLABatchSize()
                                    << std::endl;
                        mNetBuilderConfig->setDeviceType(layer, nvinfer1::DeviceType::kDLA);
                        devSuccess = true;
                    }
    #else
                    if(mNetBuilder->canRunOnDLA(layer)) { 
                        layer->setPrecision(mDataType);
                        std::cout << "Layer: " << layer->getName() 
                                    << " will run on DLA (batch size max: " 
                                    <<  mNetBuilder->getMaxDLABatchSize()
                                    << std::endl;
                        mNetBuilder->setDeviceType(layer, nvinfer1::DeviceType::kDLA);
                        devSuccess = true;
                    }
    #endif
                }
                if(!devSuccess)
                    throw std::runtime_error("Cannot use DLA for layer " + outName);
            }
#endif
           output_tensor.push_back(layer->getOutput(0));
           output_tensor.back()->setName(outName.c_str());
           output_tensor.back()->setType(mDataType);

           std::cout << "               " << output_tensor.back()->getName() << std::endl;

           nvinfer1::Dims tensor_dims = output_tensor.back()->getDimensions();
           std::cout << "{";
           for(unsigned int d = 0; d < tensor_dims.nbDims; ++d)
                std::cout << tensor_dims.d[d] << " ";
           std::cout << "}" << std::endl;
        }
        return output_tensor;


}

std::vector<nvinfer1::ITensor *>
      N2D2::Network::add_elementwise(std::string layerName,
                        LayerActivation activation,
                        unsigned int nbOutputs,
                        unsigned int outputHeight,
                        unsigned int outputWidth,
                        std::vector<std::vector<nvinfer1::ITensor *> > inputs_tensor,
                        nvinfer1::ElementWiseOperation op,
                        CoeffMode_T coeffMode,
                        float* scales,
                        float* shift,
                        float* power)
{
        std::vector<nvinfer1::ITensor *> output_tensor;
        std::vector<std::vector<nvinfer1::ITensor *>> scale_tensor;

        std::cout << "Add elementwize layer: " << layerName << std::endl;
        nvinfer1::ScaleMode modeScale = (coeffMode == PerChannel) ? 
                                            nvinfer1::ScaleMode::kCHANNEL 
                                            : nvinfer1::ScaleMode::kUNIFORM;
        const std::size_t coeffSize =  coeffMode == PerChannel ? 
                                        nbOutputs : inputs_tensor.size();
        /**
            This layer applies a per-elements tensor computation to its inputA and B:
                output = (input* scale + shift)^ power
        **/
        __half* scale_half;
        __half* shift_half;
        __half* power_half;

        if(mDataType == nvinfer1::DataType::kHALF)
        {
            scale_half = new __half[coeffSize];
            shift_half = new __half[coeffSize];
            power_half = new __half[coeffSize];

            for(unsigned int c = 0; c < coeffSize;  ++c)
            {
                scale_half[c] = fp16::__float2half(scales[c]);
                shift_half[c] = fp16::__float2half(shift[c]);
                power_half[c] = fp16::__float2half(power[c]);
            }
        }


        for(unsigned int input = 0; input < inputs_tensor.size(); ++input)
        {
            nvinfer1::Weights scale_trt;
            nvinfer1::Weights shift_trt;
            nvinfer1::Weights power_trt;
            const std::size_t coeffIdx =  coeffMode == PerChannel ? 
                                            0 : input;
            const std::size_t coeffLength =  coeffMode == PerChannel ? 
                                            nbOutputs : 1;

            if(mDataType != nvinfer1::DataType::kHALF)
            {
                scale_trt  = {mDataType, scales + coeffIdx, coeffLength};
                shift_trt  = {mDataType, shift + coeffIdx, coeffLength};
                power_trt  = {mDataType, power + coeffIdx, coeffLength};
            }
            else
            {
                scale_trt  = {mDataType, scale_half + coeffIdx, coeffLength};
                shift_trt  = {mDataType, shift_half + coeffIdx, coeffLength};
                power_trt  = {mDataType, power_half + coeffIdx, coeffLength};
            }


            std::vector<nvinfer1::ITensor *> scaleVecTensor;

            for(unsigned int vecIn = 0; vecIn < inputs_tensor[input].size();
                    ++vecIn)
            {

                if(scales[input] != 1.0 || shift[input] != 0.0)
                {
                    std::string outName = layerName + "_scale_"
											+ std::to_string(vecIn)
											+ std::to_string(input) ;

                    auto layer = mNetDef.back()->addScale(*inputs_tensor[input][vecIn],
                                                modeScale,
                                                shift_trt,
                                                scale_trt,
                                                power_trt);

                    layer->setName(outName.c_str());
#if NV_TENSORRT_MAJOR > 4
            if(mUseDLA)
            {
                bool devSuccess = false;
                if((mDataType == nvinfer1::DataType::kHALF 
                        || mDataType == nvinfer1::DataType::kINT8))
                {
    #if (NV_TENSORRT_MAJOR + NV_TENSORRT_MINOR) > 7
                    if(mNetBuilderConfig->canRunOnDLA(layer)) { 
                        layer->setPrecision(mDataType);
                        std::cout << "Layer: " << layer->getName() 
                                    << " will run on DLA (batch size max: " 
                                    <<  mNetBuilder->getMaxDLABatchSize()
                                    << std::endl;
                        mNetBuilderConfig->setDeviceType(layer, nvinfer1::DeviceType::kDLA);
                        devSuccess = true;
                    }
    #else
                    if(mNetBuilder->canRunOnDLA(layer)) { 
                        layer->setPrecision(mDataType);
                        std::cout << "Layer: " << layer->getName() 
                                    << " will run on DLA (batch size max: " 
                                    <<  mNetBuilder->getMaxDLABatchSize()
                                    << std::endl;
                        mNetBuilder->setDeviceType(layer, nvinfer1::DeviceType::kDLA);
                        devSuccess = true;
                    }
    #endif
                }
                if(!devSuccess)
                    throw std::runtime_error("Cannot use DLA for layer " + outName);
            }
#endif
                    scaleVecTensor.push_back(layer->getOutput(0));
                    scaleVecTensor.back()->setType(mDataType);

				    nvinfer1::Dims tensor_in_dims = inputs_tensor[input][vecIn]->getDimensions();
				    std::cout << "               " << inputs_tensor[input][vecIn]->getName()
				            << "---> " << scaleVecTensor.back()->getName() << std::endl;

				    std::cout << "               ";
				    std::cout << "{";
				    for(unsigned int d = 0; d < tensor_in_dims.nbDims; ++d)
				        std::cout << tensor_in_dims.d[d] << " ";
				    std::cout << "} ----> ";

				    nvinfer1::Dims tensor_dims = scaleVecTensor.back()->getDimensions();
				    std::cout << "{";
				    for(unsigned int d = 0; d < tensor_dims.nbDims; ++d)
				        std::cout << tensor_dims.d[d] << " ";
				    std::cout << "}" << std::endl;

                }
                else
                {
                    scaleVecTensor.push_back(inputs_tensor[input][vecIn]);
                    scaleVecTensor.back()->setType(mDataType);
                }
            }

            scale_tensor.push_back(scaleVecTensor);
        }

        for(unsigned int i = 0; i < scale_tensor[0].size(); ++i)
        {

            if(scale_tensor.size() == 2)
            {
                std::string outName = layerName + "_" + std::to_string(i);

                auto layer = mNetDef.back()->addElementWise(*scale_tensor[0][i],
                                                    *scale_tensor[1][i],
                                                    op);
                layer->setName(outName.c_str());

#if NV_TENSORRT_MAJOR > 4
            if(mUseDLA)
            {
                bool devSuccess = false;
                if((mDataType == nvinfer1::DataType::kHALF 
                        || mDataType == nvinfer1::DataType::kINT8))
                {
    #if (NV_TENSORRT_MAJOR + NV_TENSORRT_MINOR) > 7
                    if(mNetBuilderConfig->canRunOnDLA(layer)) { 
                        layer->setPrecision(mDataType);
                        std::cout << "Layer: " << layer->getName() 
                                    << " will run on DLA (batch size max: " 
                                    <<  mNetBuilder->getMaxDLABatchSize()
                                    << std::endl;
                        mNetBuilderConfig->setDeviceType(layer, nvinfer1::DeviceType::kDLA);
                        devSuccess = true;
                    }
    #else
                    if(mNetBuilder->canRunOnDLA(layer)) { 
                        layer->setPrecision(mDataType);
                        std::cout << "Layer: " << layer->getName() 
                                    << " will run on DLA (batch size max: " 
                                    <<  mNetBuilder->getMaxDLABatchSize()
                                    << std::endl;
                        mNetBuilder->setDeviceType(layer, nvinfer1::DeviceType::kDLA);
                        devSuccess = true;
                    }
    #endif
                }
                if(!devSuccess)
                    throw std::runtime_error("Cannot use DLA for layer " + outName);
            }
#endif

                output_tensor.push_back(layer->getOutput(0));
                output_tensor.back()->setName(outName.c_str());
                output_tensor.back()->setType(mDataType);
		        nvinfer1::Dims tensor_in_dims = scale_tensor[0][i]->getDimensions();
		        std::cout << "               {" << scale_tensor[0][i]->getName() << ", " << scale_tensor[1][i]->getName()
		                << "}---> " << output_tensor.back()->getName() << std::endl;

		        std::cout << "               ";
		        std::cout << "{";
		        for(unsigned int d = 0; d < tensor_in_dims.nbDims; ++d)
		            std::cout << tensor_in_dims.d[d] << " ";
		        std::cout << "} ----> ";

		        nvinfer1::Dims tensor_dims = output_tensor.back()->getDimensions();
		        std::cout << "{";
		        for(unsigned int d = 0; d < tensor_dims.nbDims; ++d)
		            std::cout << tensor_dims.d[d] << " ";
		        std::cout << "}" << std::endl;
            }
            else if(scale_tensor.size() == 1)
            {
                std::string outName = layerName + "_scale_" + std::to_string(i);
                output_tensor.push_back(scale_tensor[0][i]);
                output_tensor.back()->setName(outName.c_str());
                output_tensor.back()->setType(mDataType);
		        nvinfer1::Dims tensor_in_dims = scale_tensor[0][i]->getDimensions();
		        std::cout << "               " << scale_tensor[0][i]->getName()
		                << "---> " << output_tensor.back()->getName() << std::endl;

		        std::cout << "               ";
		        std::cout << "{";
		        for(unsigned int d = 0; d < tensor_in_dims.nbDims; ++d)
		            std::cout << tensor_in_dims.d[d] << " ";
		        std::cout << "} ----> ";

		        nvinfer1::Dims tensor_dims = output_tensor.back()->getDimensions();
		        std::cout << "{";
		        for(unsigned int d = 0; d < tensor_dims.nbDims; ++d)
		            std::cout << tensor_dims.d[d] << " ";
		        std::cout << "}" << std::endl;
            }
            else
                throw std::runtime_error( "ElementWise Layer: could not have layer size different than 1 or 2");


        }

        if(activation.status)
            return(add_activation(layerName,
                                  activation.type,
                                  activation.alpha,
                                  activation.beta,
                                  output_tensor));
        else
            return output_tensor;
}



std::vector<nvinfer1::ITensor *>
      N2D2::Network::add_scale(std::string layerName,
                        LayerActivation activation,
                        unsigned int nbOutputs,
                        unsigned int outputHeight,
                        unsigned int outputWidth,
                        std::vector<nvinfer1::ITensor *> inputs_tensor,
                        float* scales,
                        float* shift,
                        float* power)
{

        std::vector<nvinfer1::ITensor *> output_tensor;
        std::cout << "Add scale layer: " << layerName << std::endl;

        nvinfer1::Weights scale_trt  = {mDataType, scales, 1};
        nvinfer1::Weights shift_trt  = {mDataType, shift, 1};
        nvinfer1::Weights power_trt  = {mDataType, power, 1};
        nvinfer1::ScaleMode modeScale = nvinfer1::ScaleMode::kUNIFORM;
        /**
            This layer applies a per-tensor computation to its input:
                output = (input* scale + shift)^ power
        **/
        for(unsigned int i = 0; i < inputs_tensor.size(); ++i)
        {
            std::string outName = layerName + "_" + std::to_string(i);

            auto layer = mNetDef.back()->addScale(*inputs_tensor[i],
                                        modeScale,
                                        shift_trt,
                                        scale_trt,
					                    power_trt);
#if NV_TENSORRT_MAJOR > 4
            if(mUseDLA)
            {
                bool devSuccess = false;
                if((mDataType == nvinfer1::DataType::kHALF 
                        || mDataType == nvinfer1::DataType::kINT8))
                {
    #if (NV_TENSORRT_MAJOR + NV_TENSORRT_MINOR) > 7
                    if(mNetBuilderConfig->canRunOnDLA(layer)) { 
                        layer->setPrecision(mDataType);
                        std::cout << "Layer: " << layer->getName() 
                                    << " will run on DLA (batch size max: " 
                                    <<  mNetBuilder->getMaxDLABatchSize()
                                    << std::endl;
                        mNetBuilderConfig->setDeviceType(layer, nvinfer1::DeviceType::kDLA);
                        devSuccess = true;
                    }
    #else
                    if(mNetBuilder->canRunOnDLA(layer)) { 
                        layer->setPrecision(mDataType);
                        std::cout << "Layer: " << layer->getName() 
                                    << " will run on DLA (batch size max: " 
                                    <<  mNetBuilder->getMaxDLABatchSize()
                                    << std::endl;
                        mNetBuilder->setDeviceType(layer, nvinfer1::DeviceType::kDLA);
                        devSuccess = true;
                    }
    #endif
                }
                if(!devSuccess)
                    throw std::runtime_error("Cannot use DLA for layer " + outName);
            }
#endif
            layer->setName(outName.c_str());
            output_tensor.push_back(layer->getOutput(0));
            output_tensor.back()->setName(outName.c_str());

            nvinfer1::Dims tensor_in_dims = inputs_tensor[i]->getDimensions();
            std::cout << "               " << inputs_tensor[i]->getName()
                     << "---> " << output_tensor.back()->getName() << std::endl;

            std::cout << "               ";
            std::cout << "{";
            for(unsigned int d = 0; d < tensor_in_dims.nbDims; ++d)
                std::cout << tensor_in_dims.d[d] << " ";
            std::cout << "} ----> ";

            nvinfer1::Dims tensor_dims = output_tensor.back()->getDimensions();
            std::cout << "{";
            for(unsigned int d = 0; d < tensor_dims.nbDims; ++d)
                std::cout << tensor_dims.d[d] << " ";
            std::cout << "}" << std::endl;
        }

        if(activation.status)
            return(add_activation(layerName,
                                  activation.type,
                                  activation.alpha,
                                  activation.beta,
                                  output_tensor));
        else
            return output_tensor;

}


std::vector<nvinfer1::ITensor *>
      N2D2::Network::add_batchnorm(std::string layerName,
                        LayerActivation activation,
                        unsigned int nbOutputs,
                        unsigned int outputHeight,
                        unsigned int outputWidth,
                        std::vector<nvinfer1::ITensor *> inputs_tensor,
                        float* scales,
                        float* biases,
                        float* means,
                        float* variances,
                        float epsilon)
{

        std::vector<nvinfer1::ITensor *> output_tensor;
        std::cout << "Add batchnorm layer: " << layerName << std::endl;

        nvinfer1::Weights scale_trt;
        nvinfer1::Weights shift_trt;
        nvinfer1::Weights power_trt;

        float* scale_wdata;
        float* shift_wdata;
        float* power_wdata;
        __half* scale_hdata;
        __half* shift_hdata;
        __half* power_hdata;


        if(mDataType != nvinfer1::DataType::kHALF)
        {
            scale_wdata = new float[nbOutputs];
            shift_wdata = new float[nbOutputs];
            power_wdata = new float[nbOutputs];
        }
        else
        {
            scale_hdata = new __half[nbOutputs];
            shift_hdata = new __half[nbOutputs];
            power_hdata = new __half[nbOutputs];
        }

        for(unsigned int  out = 0; out < nbOutputs; ++out)
        {
            if(mDataType != nvinfer1::DataType::kHALF)
            {
                scale_wdata[out] = scales[out]/sqrt(variances[out] + epsilon);
                shift_wdata[out] = ((-means[out]*scales[out])/sqrt(variances[out] + epsilon)) + biases[out];
                power_wdata[out] = (float) 1;
            }
            else
            {
                scale_hdata[out] = fp16::__float2half( scales[out]/sqrt(variances[out] + epsilon) );
                shift_hdata[out] = fp16::__float2half( ((-means[out]*scales[out])/sqrt(variances[out] + epsilon)) + biases[out] );
                power_hdata[out] = fp16::__float2half(1);
            }
        }
        if(mDataType != nvinfer1::DataType::kHALF)
        {

            scale_trt  = {mDataType, scale_wdata, nbOutputs};
            shift_trt  = {mDataType, shift_wdata, nbOutputs};
            power_trt  = {mDataType, power_wdata, nbOutputs};
        }
        else
        {
            scale_trt  = {mDataType, scale_hdata, nbOutputs};
            shift_trt  = {mDataType, shift_hdata, nbOutputs};
            power_trt  = {mDataType, power_hdata, nbOutputs};
        }
        nvinfer1::ScaleMode modeScale = nvinfer1::ScaleMode::kCHANNEL;

        /**
            This layer applies a per-element computation to its input:
                output = (input* scale + shift)^ power
        **/
        for(unsigned int i = 0; i < inputs_tensor.size(); ++i)
        {
            std::string outName = layerName + "_" + std::to_string(i);

            auto layer = mNetDef.back()->addScale(*inputs_tensor[i],
                                        modeScale,
                                        shift_trt,
                                        scale_trt,
					                    power_trt);
#if NV_TENSORRT_MAJOR > 4
            if(mUseDLA)
            {
                bool devSuccess = false;
                if((mDataType == nvinfer1::DataType::kHALF 
                        || mDataType == nvinfer1::DataType::kINT8))
                {
    #if (NV_TENSORRT_MAJOR + NV_TENSORRT_MINOR) > 7
                    if(mNetBuilderConfig->canRunOnDLA(layer)) { 
                        layer->setPrecision(mDataType);
                        std::cout << "Layer: " << layer->getName() 
                                    << " will run on DLA (batch size max: " 
                                    <<  mNetBuilder->getMaxDLABatchSize()
                                    << std::endl;
                        mNetBuilderConfig->setDeviceType(layer, nvinfer1::DeviceType::kDLA);
                        devSuccess = true;
                    }
    #else
                    if(mNetBuilder->canRunOnDLA(layer)) { 
                        layer->setPrecision(mDataType);
                        std::cout << "Layer: " << layer->getName() 
                                    << " will run on DLA (batch size max: " 
                                    <<  mNetBuilder->getMaxDLABatchSize()
                                    << std::endl;
                        mNetBuilder->setDeviceType(layer, nvinfer1::DeviceType::kDLA);
                        devSuccess = true;
                    }
    #endif
                }
                if(!devSuccess)
                    throw std::runtime_error("Cannot use DLA for layer " + outName);
            }
#endif

            layer->setName(outName.c_str());
            output_tensor.push_back(layer->getOutput(0));
            output_tensor.back()->setName(outName.c_str());
            output_tensor.back()->setType(mDataType);

            nvinfer1::Dims tensor_in_dims = inputs_tensor[i]->getDimensions();
            std::cout << "               " << inputs_tensor[i]->getName()
                     << "---> " << output_tensor.back()->getName() << std::endl;

            std::cout << "               ";
            std::cout << "{";
            for(unsigned int d = 0; d < tensor_in_dims.nbDims; ++d)
                std::cout << tensor_in_dims.d[d] << " ";
            std::cout << "} ----> ";

            nvinfer1::Dims tensor_dims = output_tensor.back()->getDimensions();
            std::cout << "{";
            for(unsigned int d = 0; d < tensor_dims.nbDims; ++d)
                std::cout << tensor_dims.d[d] << " ";
            std::cout << "}" << std::endl;
        }
        /*
        for(unsigned int i = 0; i < inputs_tensor.size(); ++i)
        {
            std::string outName = layerName + "_" + std::to_string(i);
            nvinfer1::IPlugin* pluginBn = mPluginFactory.createPlugin(outName.c_str(),
                                                                mMaxBatchSize,
                                                                nbOutputs,
                                                                outputHeight,
                                                                outputWidth,
                                                                scales,
                                                                biases,
                                                                means,
                                                                variances,
                                                                epsilon);

            auto layer = mNetDef->addPlugin(&inputs_tensor[i],
                                        1,
                                        *pluginBn);

           layer->setName(outName.c_str());

           output_tensor.push_back(layer->getOutput(0));
           output_tensor.back()->setName(outName.c_str());
           nvinfer1::Dims tensor_in_dims = inputs_tensor[i]->getDimensions();
           std::cout << "               " << inputs_tensor[i]->getName()
                     << "---> " << output_tensor.back()->getName() << std::endl;

           std::cout << "               ";
           std::cout << "{";
           for(unsigned int d = 0; d < tensor_in_dims.nbDims; ++d)
                std::cout << tensor_in_dims.d[d] << " ";
           std::cout << "} ----> ";

           nvinfer1::Dims tensor_dims = output_tensor.back()->getDimensions();
           std::cout << "{";
           for(unsigned int d = 0; d < tensor_dims.nbDims; ++d)
                std::cout << tensor_dims.d[d] << " ";
           std::cout << "}" << std::endl;
        }
*/
        if(activation.status)
            return(add_activation(layerName,
                                  activation.type,
                                  activation.alpha,
                                  activation.beta,
                                  output_tensor));
        else
            return output_tensor;

}

std::vector<nvinfer1::ITensor *>
      N2D2::Network::add_pooling(std::string layerName,
                        LayerActivation activation,
                        unsigned int poolH,
                        unsigned int poolW,
                        unsigned int strideX,
                        unsigned int strideY,
                        unsigned int paddingX,
                        unsigned int paddingY,
                        std::vector<nvinfer1::ITensor *> inputs_tensor,
                        nvinfer1::PoolingType poolType)
{
        std::vector<nvinfer1::ITensor *> output_tensor;
        std::cout << "Add pooling layer: " << layerName << std::endl;
        trt_DimsHW poolDims = {(int)poolH, (int)poolW};
        trt_DimsHW strideDims = {(int)strideY, (int)strideX};
        trt_DimsHW paddingDims = {(int)paddingY, (int)paddingX};

        for(unsigned int i = 0; i < inputs_tensor.size(); ++i)
        {
            std::string outName = layerName + "_" + std::to_string(i);
#if NV_TENSORRT_MAJOR < 6
            auto layer = mNetDef.back()->addPooling(*inputs_tensor[i],
                                             poolType,
                                             poolDims);
           layer->setStride(strideDims);
           layer->setPadding(paddingDims);
#else 
            auto layer = mNetDef.back()->addPoolingNd(*inputs_tensor[i],
                                             poolType,
                                             poolDims);
           layer->setStrideNd(strideDims);
           layer->setPaddingNd(paddingDims);

#endif
           layer->setName(outName.c_str());
#if NV_TENSORRT_MAJOR > 4
            if(mUseDLA)
            {
                bool devSuccess = false;
                if((mDataType == nvinfer1::DataType::kHALF 
                        || mDataType == nvinfer1::DataType::kINT8))
                {
    #if (NV_TENSORRT_MAJOR + NV_TENSORRT_MINOR) > 7
                    if(mNetBuilderConfig->canRunOnDLA(layer)) { 
                        layer->setPrecision(mDataType);
                        std::cout << "Layer: " << layer->getName() 
                                    << " will run on DLA (batch size max: " 
                                    <<  mNetBuilder->getMaxDLABatchSize()
                                    << std::endl;
                        mNetBuilderConfig->setDeviceType(layer, nvinfer1::DeviceType::kDLA);
                        devSuccess = true;
                    }
    #else
                    if(mNetBuilder->canRunOnDLA(layer)) { 
                        layer->setPrecision(mDataType);
                        std::cout << "Layer: " << layer->getName() 
                                    << " will run on DLA (batch size max: " 
                                    <<  mNetBuilder->getMaxDLABatchSize()
                                    << std::endl;
                        mNetBuilder->setDeviceType(layer, nvinfer1::DeviceType::kDLA);
                        devSuccess = true;
                    }
    #endif
                }
                if(!devSuccess)
                    throw std::runtime_error("Cannot use DLA for layer " + outName);
            }
#endif
           output_tensor.push_back(layer->getOutput(0));
           output_tensor.back()->setName(outName.c_str());
           output_tensor.back()->setType(mDataType);

           nvinfer1::Dims tensor_in_dims = inputs_tensor[i]->getDimensions();
           std::cout << "               " << inputs_tensor[i]->getName()
                     << "---> " << output_tensor.back()->getName() << std::endl;

           std::cout << "               ";
           std::cout << "{";
           for(unsigned int d = 0; d < tensor_in_dims.nbDims; ++d)
                std::cout << tensor_in_dims.d[d] << " ";
           std::cout << "} ----> ";

           nvinfer1::Dims tensor_dims = output_tensor.back()->getDimensions();
           std::cout << "{";
           for(unsigned int d = 0; d < tensor_dims.nbDims; ++d)
                std::cout << tensor_dims.d[d] << " ";
           std::cout << "}" << std::endl;
        }
        if(activation.status)
            return(add_activation(layerName,
                                  activation.type,
                                  activation.alpha,
                                  activation.beta,
                                  output_tensor));
        else
            return output_tensor;

}
std::vector<nvinfer1::ITensor *>
      N2D2::Network::add_HWC2CHW(std::string layerName,
                    std::vector<nvinfer1::ITensor *> inputs_tensor)
{
    std::vector<nvinfer1::ITensor *> output_tensor;
    std::cout << "Add Shufflelayer: " << layerName << std::endl;
    for(unsigned int i = 0; i < inputs_tensor.size(); ++i)
    {

        nvinfer1::Dims tensor_dims = inputs_tensor[0]->getDimensions();

        //const unsigned int residualBatch = tensor_dims.nbDims > 3 ?
        //                                    tensor_dims.d[0]
        //                                    : 1;

        std::string outName = layerName + "_" + std::to_string(i);
        auto layer = mNetDef.back()->addShuffle(*inputs_tensor[i]);
        assert(layer != nullptr);
        if(tensor_dims.nbDims == 3)
        {
            
#if NV_TENSORRT_MAJOR < 4
            nvinfer1::DimsCHW reshape_dims;
            reshape_dims.d[0] = tensor_dims.d[2];
            reshape_dims.d[1] = tensor_dims.d[0];
            reshape_dims.d[2] = tensor_dims.d[1];
            layer->setReshapeDimensions(reshape_dims);
#endif
            layer->setFirstTranspose(nvinfer1::Permutation{2,0,1});
        }
		//layer->setFirstTranspose(nvinfer1::Permutation{0,1,2});
		//layer->setSecondTranspose(nvinfer1::Permutation{0,1,2});

        layer->setName(outName.c_str());
        output_tensor.push_back(layer->getOutput(0));
        output_tensor.back()->setName(outName.c_str());

        nvinfer1::Dims tensor_in_dims = inputs_tensor[i]->getDimensions();
        std::cout << "               " << inputs_tensor[i]->getName()
                    << "---> " << output_tensor.back()->getName() << std::endl;

        std::cout << "               ";
        std::cout << "{";
        for(unsigned int d = 0; d < tensor_in_dims.nbDims; ++d)
                std::cout << tensor_in_dims.d[d] << " ";
        std::cout << "} ----> ";

        nvinfer1::Dims tensor_output_dims = output_tensor.back()->getDimensions();
        std::cout << "{";
        for(unsigned int d = 0; d < tensor_output_dims.nbDims; ++d)
                std::cout << tensor_output_dims.d[d] << " ";
        std::cout << "}" << std::endl;
    }

    return output_tensor;
}

std::vector<nvinfer1::ITensor *>
      N2D2::Network::add_reshape(std::string layerName,
                    unsigned int nbDims,
                    const int shape[],
                    std::vector<nvinfer1::ITensor *> inputs_tensor)
{
    std::vector<nvinfer1::ITensor *> output_tensor;
    std::cout << "Add reshape layer: " << layerName << std::endl;
    for(unsigned int i = 0; i < inputs_tensor.size(); ++i)
    {

        nvinfer1::Dims tensor_dims = inputs_tensor[0]->getDimensions();

        std::string outName = layerName + "_" + std::to_string(i);
        auto layer = mNetDef.back()->addShuffle(*inputs_tensor[i]);
        assert(layer != nullptr);

        nvinfer1::Dims reshape_dims;
        reshape_dims.nbDims = nbDims;

        for (int dim = 0; dim < nbDims; ++dim) {
            if (shape[dim] != 0)
                reshape_dims.d[dim] = shape[dim];
            else
                reshape_dims.d[dim] = tensor_dims.d[dim];
        }

        layer->setReshapeDimensions(reshape_dims);

        layer->setName(outName.c_str());
        output_tensor.push_back(layer->getOutput(0));
        output_tensor.back()->setName(outName.c_str());

        nvinfer1::Dims tensor_in_dims = inputs_tensor[i]->getDimensions();
        std::cout << "               " << inputs_tensor[i]->getName()
                    << "---> " << output_tensor.back()->getName() << std::endl;

        std::cout << "               ";
        std::cout << "{";
        for(unsigned int d = 0; d < tensor_in_dims.nbDims; ++d)
                std::cout << tensor_in_dims.d[d] << " ";
        std::cout << "} ----> ";

        nvinfer1::Dims tensor_output_dims = output_tensor.back()->getDimensions();
        std::cout << "{";
        for(unsigned int d = 0; d < tensor_output_dims.nbDims; ++d)
                std::cout << tensor_output_dims.d[d] << " ";
        std::cout << "}" << std::endl;
    }

    return output_tensor;
}

std::vector<nvinfer1::ITensor *>
      N2D2::Network::add_transpose(std::string layerName,
                    unsigned int nbDims,
                    const int perm[],
                    std::vector<nvinfer1::ITensor *> inputs_tensor)
{
    std::vector<nvinfer1::ITensor *> output_tensor;
    std::cout << "Add transpose layer: " << layerName << std::endl;
    for(unsigned int i = 0; i < inputs_tensor.size(); ++i)
    {

        nvinfer1::Dims tensor_dims = inputs_tensor[0]->getDimensions();

        std::string outName = layerName + "_" + std::to_string(i);
        auto layer = mNetDef.back()->addShuffle(*inputs_tensor[i]);
        assert(layer != nullptr);

        nvinfer1::Dims reshape_dims;
        nvinfer1::Permutation perm_dims;

        reshape_dims.nbDims = nbDims;

        for (int dim = 0; dim < nbDims; ++dim) {
            reshape_dims.d[dim] = tensor_dims.d[perm[dim]];
            perm_dims.order[dim] = perm[dim];
        }

#if NV_TENSORRT_MAJOR < 4
        layer->setReshapeDimensions(reshape_dims);
#endif
        layer->setFirstTranspose(perm_dims);

        layer->setName(outName.c_str());
        output_tensor.push_back(layer->getOutput(0));
        output_tensor.back()->setName(outName.c_str());

        nvinfer1::Dims tensor_in_dims = inputs_tensor[i]->getDimensions();
        std::cout << "               " << inputs_tensor[i]->getName()
                    << "---> " << output_tensor.back()->getName() << std::endl;

        std::cout << "               ";
        std::cout << "{";
        for(unsigned int d = 0; d < tensor_in_dims.nbDims; ++d)
                std::cout << tensor_in_dims.d[d] << " ";
        std::cout << "} ----> ";

        nvinfer1::Dims tensor_output_dims = output_tensor.back()->getDimensions();
        std::cout << "{";
        for(unsigned int d = 0; d < tensor_output_dims.nbDims; ++d)
                std::cout << tensor_output_dims.d[d] << " ";
        std::cout << "}" << std::endl;
    }

    return output_tensor;
}

std::vector<nvinfer1::ITensor *>
          N2D2::Network::add_group_reshape(std::string layerName,
                        unsigned int groupSize,
                        bool restoreShape,
                        std::vector<nvinfer1::ITensor *> inputs_tensor)
{
    std::vector<nvinfer1::ITensor *> output_tensor;
    std::cout << "Add group reshape layer: " << layerName << std::endl;


    for(unsigned int i = 0; i < inputs_tensor.size(); ++i)
    {

        nvinfer1::Dims tensor_dims = inputs_tensor[0]->getDimensions();

        //const unsigned int residualBatch = tensor_dims.nbDims > 3 ?
        //                                    tensor_dims.d[0]
        //                                    : 1;

        std::string outName = layerName + "_" + std::to_string(i);
        auto layer = mNetDef.back()->addShuffle(*inputs_tensor[i]);
        assert(layer != nullptr);
        if(tensor_dims.nbDims == 3)
        {
            trt_Dims3 reshape_dims;
            const unsigned int nbOutputs = tensor_dims.d[0];
            const unsigned int dimY = tensor_dims.d[1];
            const unsigned int dimX = tensor_dims.d[2];
            std::cout << "groupSize: " << groupSize
                        << " nbOutputs: " << nbOutputs
                        << std::endl;

            if(!(groupSize % nbOutputs))
                throw std::runtime_error(
                    "add_group_reshape(): groupsize must be divisible by nbOutputs");

            reshape_dims.d[0] = groupSize;
            reshape_dims.d[1] = dimY * (nbOutputs / groupSize);
            reshape_dims.d[2] = dimX;

            layer->setReshapeDimensions(reshape_dims);

        }
        else if(tensor_dims.nbDims == 4)
        {
            trt_Dims4 reshape_dims;
            const unsigned int batch = tensor_dims.d[0];
            const unsigned int nbOutputs = tensor_dims.d[1];
            const unsigned int dimY = tensor_dims.d[2];
            const unsigned int dimX = tensor_dims.d[3];
            std::cout << "groupSize: " << groupSize
                        << " nbOutputs: " << nbOutputs
                        << std::endl;

            if(!(groupSize % nbOutputs))
                throw std::runtime_error(
                    "add_group_reshape(): groupsize must be divisible by nbOutputs");
            if( (dimY > 1 || dimX > 1) && !restoreShape)
                throw std::runtime_error(
                    "add_group_reshape(): can only be applied on 1 dimension tensor");

            reshape_dims.d[0] = batch;
            reshape_dims.d[1] = !restoreShape ? groupSize : dimY * nbOutputs;
            reshape_dims.d[2] = !restoreShape ? dimY * (nbOutputs / groupSize) : 1;
            reshape_dims.d[3] = dimX;

            layer->setReshapeDimensions(reshape_dims);

        }
		//layer->setFirstTranspose(nvinfer1::Permutation{0,1,2});
		//layer->setSecondTranspose(nvinfer1::Permutation{0,1,2});

        layer->setName(outName.c_str());
        output_tensor.push_back(layer->getOutput(0));
        output_tensor.back()->setName(outName.c_str());

        nvinfer1::Dims tensor_in_dims = inputs_tensor[i]->getDimensions();
        std::cout << "               " << inputs_tensor[i]->getName()
                    << "---> " << output_tensor.back()->getName() << std::endl;

        std::cout << "               ";
        std::cout << "{";
        for(unsigned int d = 0; d < tensor_in_dims.nbDims; ++d)
                std::cout << tensor_in_dims.d[d] << " ";
        std::cout << "} ----> ";

        nvinfer1::Dims tensor_output_dims = output_tensor.back()->getDimensions();
        std::cout << "{";
        for(unsigned int d = 0; d < tensor_output_dims.nbDims; ++d)
                std::cout << tensor_output_dims.d[d] << " ";
        std::cout << "}" << std::endl;
    }

    return output_tensor;
}


std::vector<nvinfer1::ITensor *>
      N2D2::Network::add_softmax(std::string layerName,
                        std::vector<nvinfer1::ITensor *> inputs_tensor)
{
    std::vector<nvinfer1::ITensor *> output_tensor;
    std::cout << "Add softmax layer: " << layerName << std::endl;

    for(unsigned int i = 0; i < inputs_tensor.size(); ++i)
    {
        std::string outName = layerName + "_" + std::to_string(i);
        auto layer = mNetDef.back()->addSoftMax(*inputs_tensor[i]);

        layer->setName(outName.c_str());

        output_tensor.push_back(layer->getOutput(0));
        output_tensor.back()->setName(outName.c_str());
        output_tensor.back()->setType(nvinfer1::DataType::kFLOAT);

        nvinfer1::Dims tensor_in_dims = inputs_tensor[i]->getDimensions();
        std::cout << "               " << inputs_tensor[i]->getName()
                    << "---> " << output_tensor.back()->getName() << std::endl;

        std::cout << "               ";
        std::cout << "{";
        for(unsigned int d = 0; d < tensor_in_dims.nbDims; ++d)
            std::cout << tensor_in_dims.d[d] << " ";
        std::cout << "} ----> ";

        nvinfer1::Dims tensor_dims = output_tensor.back()->getDimensions();
        std::cout << "{";
        for(unsigned int d = 0; d < tensor_dims.nbDims; ++d)
            std::cout << tensor_dims.d[d] << " ";
        std::cout << "}" << std::endl;
    }

    return output_tensor;
}

std::vector<nvinfer1::ITensor *>
      N2D2::Network::add_anchors(std::string layerName,
                        unsigned int nbOutputs,
                        unsigned int outputHeight,
                        unsigned int outputWidth,
                        std::vector<nvinfer1::ITensor *> inputs_tensor,
                        unsigned int stimuliHeight,
                        unsigned int stimuliWidth,
                        unsigned int featureMapWidth,
                        unsigned int featureMapHeight,
                        unsigned int scoreCls,
                        bool isCoordinatesAnchors,
                        bool isFlip,
                        unsigned int nbAnchors,
                        const float* anchor)
{

        std::vector<nvinfer1::ITensor *> output_tensor;
        std::cout << "Add anchors layer: " << layerName << std::endl;

        for(unsigned int i = 0; i < inputs_tensor.size(); ++i)
        {
            std::string outName = layerName + "_" + std::to_string(i);
            nvinfer1::IPlugin* pluginAnc = mPluginFactory.createPlugin(outName.c_str(),
                                                                mMaxBatchSize,
                                                                nbOutputs,
                                                                outputHeight,
                                                                outputWidth,
                                                                stimuliHeight,
                                                                stimuliWidth,
                                                                featureMapWidth,
                                                                featureMapHeight,
                                                                scoreCls,
                                                                isCoordinatesAnchors,
                                                                isFlip,
                                                                nbAnchors,
                                                                anchor);

            auto layer = mNetDef.back()->addPlugin(&inputs_tensor[i],
                                        1,
                                        *pluginAnc);

           layer->setName(outName.c_str());

           output_tensor.push_back(layer->getOutput(0));
           output_tensor.back()->setName(outName.c_str());
           nvinfer1::Dims tensor_in_dims = inputs_tensor[i]->getDimensions();
           std::cout << "               " << inputs_tensor[i]->getName()
                     << "---> " << output_tensor.back()->getName() << std::endl;

           std::cout << "               ";
           std::cout << "{";
           for(unsigned int d = 0; d < tensor_in_dims.nbDims; ++d)
                std::cout << tensor_in_dims.d[d] << " ";
           std::cout << "} ----> ";

           nvinfer1::Dims tensor_dims = output_tensor.back()->getDimensions();
           std::cout << "{";
           for(unsigned int d = 0; d < tensor_dims.nbDims; ++d)
                std::cout << tensor_dims.d[d] << " ";
           std::cout << "}" << std::endl;
        }
        return output_tensor;

}


std::vector<nvinfer1::ITensor *>
      N2D2::Network::add_resize(std::string layerName,
                    unsigned int nbOutputs,
                    unsigned int outputHeight,
                    unsigned int outputWidth,
                    std::vector<nvinfer1::ITensor *> inputs_tensor,
                    unsigned int featureHeight,
                    unsigned int featureWidth,
                    Pooling_T resizeType,
                    bool alignCorner)
{
    std::vector<nvinfer1::ITensor *> output_tensor;
    std::cout << "Add resize layer: " << layerName << std::endl;

    for(unsigned int i = 0; i < inputs_tensor.size(); ++i)
    {
        std::string outName = layerName + "_" + std::to_string(i);
#if NV_TENSORRT_MAJOR < 6
        nvinfer1::IPlugin* pluginResize = mPluginFactory.createPlugin(outName.c_str(),
                                                                mMaxBatchSize,
                                                                nbOutputs,
                                                                outputHeight,
                                                                outputWidth,
                                                                featureHeight,
                                                                featureWidth,
                                                                resizeType,
                                                                alignCorner);

        auto layer = mNetDef.back()->addPlugin(&inputs_tensor[i],
                                    1,
                                    *pluginResize);
        layer->setName(outName.c_str());
        output_tensor.push_back(layer->getOutput(0));
        output_tensor.back()->setType(mDataType);
        output_tensor.back()->setName(outName.c_str());

        nvinfer1::Dims tensor_in_dims = inputs_tensor[i]->getDimensions();
        std::cout << "               " << inputs_tensor[i]->getName()
                    << "---> " << output_tensor.back()->getName() << std::endl;

        std::cout << "               ";
        std::cout << "{";
        for(unsigned int d = 0; d < tensor_in_dims.nbDims; ++d)
                std::cout << tensor_in_dims.d[d] << " ";
        std::cout << "} ----> ";

        nvinfer1::Dims tensor_dims = output_tensor.back()->getDimensions();
        std::cout << "{";
        for(unsigned int d = 0; d < tensor_dims.nbDims; ++d)
                std::cout << tensor_dims.d[d] << " ";
        std::cout << "}" << std::endl;

#else
        const nvinfer1::Dims3 outputsDims = {(int)nbOutputs, (int)outputHeight, (int)outputWidth};
        auto layer = mNetDef.back()->addResize(*inputs_tensor[i]);
        layer->setOutputDimensions(outputsDims);
        layer->setAlignCorners(alignCorner);
        const nvinfer1::ResizeMode mode = (resizeType == BilinearTF) 
                                    ? nvinfer1::ResizeMode::kLINEAR
                                    : nvinfer1::ResizeMode::kNEAREST;
        layer->setResizeMode(mode);
        layer->setName(outName.c_str());
        output_tensor.push_back(layer->getOutput(0));
        output_tensor.back()->setType(mDataType);
        output_tensor.back()->setName(outName.c_str());

        nvinfer1::Dims tensor_in_dims = inputs_tensor[i]->getDimensions();
        std::cout << "               " << inputs_tensor[i]->getName()
                    << "---> " << output_tensor.back()->getName() << std::endl;

        std::cout << "               ";
        std::cout << "{";
        for(unsigned int d = 0; d < tensor_in_dims.nbDims; ++d)
                std::cout << tensor_in_dims.d[d] << " ";
        std::cout << "} ----> ";

        nvinfer1::Dims tensor_dims = output_tensor.back()->getDimensions();
        std::cout << "{";
        for(unsigned int d = 0; d < tensor_dims.nbDims; ++d)
                std::cout << tensor_dims.d[d] << " ";
        std::cout << "}" << std::endl;
#endif
    }

    return output_tensor;

}

std::vector<nvinfer1::ITensor *>
      N2D2::Network::add_objectdetect(std::string layerName,
                            unsigned int nbOutputs,
                            unsigned int outputHeight,
                            unsigned int outputWidth,
                            unsigned int channelHeight,
                            unsigned int channelWidth,
                            unsigned int stimuliWidth,
                            unsigned int stimuliHeight,
                            unsigned int featureMapWidth,
                            unsigned int featureMapHeight,
                            unsigned int nbProposals,
                            unsigned int nbCls,
                            unsigned int nbAnchors,
                            //std::vector<nvinfer1::ITensor *> inputs_tensor,
                            std::vector<std::vector<nvinfer1::ITensor *> *> inputs_tensor,
                            bool isCoordinatesAnchors,
                            bool isPixelFormatXY,
                            double nmsIoU,
                            const float* scoreThreshold,
                            unsigned int maxParts,
                            unsigned int maxTemplates,
                            const unsigned int* numPartsPerClass,
                            const unsigned int* numTemplatesPerClass,
                            const float* anchor)
{
    std::vector<nvinfer1::ITensor *> output_tensor;
    std::cout << "Add object detect layer: " << layerName << std::endl;
    bool useInternalThresholds = false;
    bool useInternalNMS = false;
    if(mDetectorThresholds != NULL)
        useInternalThresholds = true;
    if(mDetectorNMS >= 0.0)
        useInternalNMS = true;

    for(unsigned int i = 0; i < inputs_tensor[0]->size(); ++i)
    {
            std::vector<nvinfer1::ITensor *> concat_tensor;

            for(unsigned int k = 0; k < inputs_tensor.size(); ++k)
            {
              nvinfer1::ITensor * input_tensor = (inputs_tensor[k])->data()[i];
              concat_tensor.push_back(input_tensor);
            }

            std::string outName = layerName + "_" + std::to_string(i);
            nvinfer1::IPlugin* pluginObjDet = mPluginFactory.createPlugin(outName.c_str(),
                                                                mMaxBatchSize,
                                                                nbOutputs,
                                                                outputHeight,
                                                                outputWidth,
                                                                channelHeight,
                                                                channelWidth,
                                                                stimuliWidth,
                                                                stimuliHeight,
                                                                featureMapWidth,
                                                                featureMapHeight,
                                                                nbProposals,
                                                                nbCls,
                                                                nbAnchors,
                                                                isCoordinatesAnchors,
                                                                isPixelFormatXY,
                                                                useInternalNMS ? mDetectorNMS : nmsIoU,
                                                                useInternalThresholds ? mDetectorThresholds : scoreThreshold,
                                                                maxParts,
                                                                maxTemplates,
                                                                numPartsPerClass,
                                                                numTemplatesPerClass,
                                                                anchor);

        auto layer = mNetDef.back()->addPlugin(&concat_tensor[0],
                                    inputs_tensor.size(),
                                    *pluginObjDet);
        layer->setName(outName.c_str());
        output_tensor.push_back(layer->getOutput(0));
        nvinfer1::Dims tensor_dims = output_tensor.back()->getDimensions();
        std::cout << "{";

        for(unsigned int d = 0; d < tensor_dims.nbDims; ++d)
            std::cout << tensor_dims.d[d] << " ";

        std::cout << "}" << std::endl;

    /*
        layer->setName(outName.c_str());
        output_tensor.push_back(layer->getOutput(0));
        output_tensor.back()->setType(mDataType);
        output_tensor.back()->setName(outName.c_str());

        nvinfer1::Dims tensor_in_dims = inputs_tensor[i]->getDimensions();
        std::cout << "               " << inputs_tensor[i]->getName()
                    << "---> " << output_tensor.back()->getName() << std::endl;

        std::cout << "               ";
        std::cout << "{";
        for(unsigned int d = 0; d < tensor_in_dims.nbDims; ++d)
                std::cout << tensor_in_dims.d[d] << " ";
        std::cout << "} ----> ";

        nvinfer1::Dims tensor_dims = output_tensor.back()->getDimensions();
        std::cout << "{";
        for(unsigned int d = 0; d < tensor_dims.nbDims; ++d)
                std::cout << tensor_dims.d[d] << " ";
        std::cout << "}" << std::endl;
        */
    }
    return output_tensor;
}

std::vector<nvinfer1::ITensor *>
      N2D2::Network::add_proposals( std::string layerName,
                        unsigned int nbOutputs,
                        unsigned int outputHeight,
                        unsigned int outputWidth,
                        unsigned int nbProposals,
                        unsigned int nbCls,
                        unsigned int stimuliHeight,
                        unsigned int stimuliWidth,
                        std::vector<std::vector<nvinfer1::ITensor *> *> inputs_tensor,
                        double nmsIoU,
                        unsigned int scoreIndex,
                        double scoreThreshold,
                        unsigned int maxParts,
                        unsigned int maxTemplates,
                        const unsigned int* numPartsPerClass,
                        const unsigned int* numTemplatesPerClass,
                        bool applyNMS,
                        bool keepMax,
                        const float* means,
                        const float* std)
{

    std::vector<nvinfer1::ITensor *> output_tensor;
    std::cout << "Add Proposals layer: " << layerName << std::endl;

    const double normX = 1.0 / (stimuliWidth - 1) ;
    const double normY = 1.0 / (stimuliHeight - 1) ;

    for(unsigned int i = 0; i < inputs_tensor[0]->size()/*inputs_tensor[0].size()*/; ++i)
    {
        std::vector<nvinfer1::ITensor *> concat_tensor;

        for(unsigned int k = 0; k < inputs_tensor.size(); ++k)
        {
            nvinfer1::ITensor * input_tensor = (inputs_tensor[k])->data()[i];
            concat_tensor.push_back(input_tensor);
        }

        std::string outName = layerName + "_" + std::to_string(i);
        nvinfer1::IPlugin* pluginProposals = mPluginFactory.createPlugin(outName.c_str(),
                                                            mMaxBatchSize,
                                                            nbOutputs,
                                                            outputHeight,
                                                            outputWidth,
                                                            nbProposals,
                                                            nbCls,
                                                            nmsIoU,
                                                            scoreIndex,
                                                            scoreThreshold,
                                                            maxParts,
                                                            maxTemplates,
                                                            numPartsPerClass,
                                                            numTemplatesPerClass,
                                                            means,
                                                            std,
                                                            applyNMS,
                                                            keepMax,
                                                            normX,
                                                            normY);


        auto layer = mNetDef.back()->addPlugin(&concat_tensor[0],
                                    inputs_tensor.size(),
                                    *pluginProposals);

        layer->setName(outName.c_str());
        output_tensor.push_back(layer->getOutput(0));
        nvinfer1::Dims tensor_dims = output_tensor.back()->getDimensions();
        std::cout << "{";

        for(unsigned int d = 0; d < tensor_dims.nbDims; ++d)
            std::cout << tensor_dims.d[d] << " ";

        std::cout << "}" << std::endl;

/*
        for(unsigned int k = 0; k < nbProposals; ++k)
        {
            std::string proposalName = outName + "_" + std::to_string(k);
            layer->setName(outName.c_str());
            output_tensor.push_back(layer->getOutput(k));
            output_tensor.back()->setName(proposalName.c_str());
            std::cout << "               " << output_tensor.back()->getName() << std::endl;

            nvinfer1::Dims tensor_dims = output_tensor.back()->getDimensions();
            std::cout << "{";
            for(unsigned int d = 0; d < tensor_dims.nbDims; ++d)
            std::cout << tensor_dims.d[d] << " ";
            std::cout << "}" << std::endl;
        }
*/
    }

    return output_tensor;

}

std::vector<nvinfer1::ITensor *>
      N2D2::Network::add_regionproposal(std::string layerName,
                            unsigned int nbOutputs,
                            unsigned int outputHeight,
                            unsigned int outputWidth,
                            std::vector<nvinfer1::ITensor *> inputs_tensor,
                            unsigned int nbAnchors,
                            unsigned int channelHeight,
                            unsigned int channelWidth,
                            unsigned int nbProposals,
                            unsigned int preNMsTopN,
                            double nmsIoU,
                            double minHeight,
                            double minWidth,
                            unsigned int scoreIndex,
                            unsigned int iouIndex)
{

        std::vector<nvinfer1::ITensor *> output_tensor;
        std::cout << "Add RP layer: " << layerName << std::endl;

        for(unsigned int i = 0; i < inputs_tensor.size(); ++i)
        {
            std::string outName = layerName + "_" + std::to_string(i);
            nvinfer1::IPlugin* pluginRP = mPluginFactory.createPlugin(outName.c_str(),
                                                                mMaxBatchSize,
                                                                nbOutputs,
                                                                outputHeight,
                                                                outputWidth,
                                                                nbAnchors,
                                                                channelHeight,
                                                                channelWidth,
                                                                nbProposals,
                                                                preNMsTopN,
                                                                nmsIoU,
                                                                minHeight,
                                                                minWidth,
                                                                scoreIndex,
                                                                iouIndex);


            auto layer = mNetDef.back()->addPlugin(&inputs_tensor[i],
                                        1,
                                        *pluginRP);

           layer->setName(outName.c_str());

           output_tensor.push_back(layer->getOutput(0));
           output_tensor.back()->setName(outName.c_str());
           nvinfer1::Dims tensor_in_dims = inputs_tensor[i]->getDimensions();
           std::cout << "               " << inputs_tensor[i]->getName()
                     << "---> " << output_tensor.back()->getName() << std::endl;

           std::cout << "               ";
           std::cout << "{";
           for(unsigned int d = 0; d < tensor_in_dims.nbDims; ++d)
                std::cout << tensor_in_dims.d[d] << " ";
           std::cout << "} ----> ";

           nvinfer1::Dims tensor_dims = output_tensor.back()->getDimensions();
           std::cout << "{";
           for(unsigned int d = 0; d < tensor_dims.nbDims; ++d)
                std::cout << tensor_dims.d[d] << " ";
           std::cout << "}" << std::endl;
        }

        return output_tensor;
}


std::vector<nvinfer1::ITensor *>
      N2D2::Network::add_ROIpooling(std::string layerName,
                            unsigned int nbOutputs,
                            unsigned int outputHeight,
                            unsigned int outputWidth,
                            /*std::vector<nvinfer1::ITensor *> const* inputs_tensor,*/
                            std::vector<std::vector<nvinfer1::ITensor *> *> inputs_tensor,
                            unsigned int stimuliHeight,
                            unsigned int stimuliWidth,
                            unsigned int nbFeature,
                            unsigned int* featureChannels,
                            unsigned int* featureHeight,
                            unsigned int* featureWidth,
                            Pooling_T poolType,
                            unsigned int nbProposals,
                            bool ignorePadding,
                            bool isFlip)
{

        std::vector<nvinfer1::ITensor *> output_tensor;
        std::cout << "Add ROIPooling layer: " << layerName << std::endl;
/*
        std::vector<std::vector<nvinfer1::ITensor *> *> tmp_tensor;

        std::vector<nvinfer1::ITensor *> tmp_j;
        for(unsigned int j = 0; j < inputs_tensor[0]->size(); ++j){
          tmp_j.clear();
          for(unsigned int k = 0; k < inputs_tensor.size(); ++k)
          {
            nvinfer1::ITensor * tmp = (inputs_tensor[k])->data()[j];
            tmp_j.push_back(tmp);
          }
          tmp_tensor.push_back(&tmp_j);
        }
*/
        nvinfer1::ITensor * input_proposal = (inputs_tensor[0])->data()[0];
        nvinfer1::Dims proposal_dims = input_proposal->getDimensions();
        unsigned int nbP = proposal_dims.d[0];

        for(unsigned int i = 0; i < inputs_tensor[0]->size()/*inputs_tensor[0].size()*/; ++i)
        {
            std::vector<nvinfer1::ITensor *> concat_tensor;

            for(unsigned int k = 0; k < inputs_tensor.size(); ++k)
            {
              nvinfer1::ITensor * input_tensor = (inputs_tensor[k])->data()[i];
              concat_tensor.push_back(input_tensor);
            }

            std::string outName = layerName + "_" + std::to_string(i);
            nvinfer1::IPlugin* pluginROIpool =
                                mPluginFactory.createPlugin(outName.c_str(),
                                                        mMaxBatchSize,
                                                        nbOutputs,
                                                        outputHeight,
                                                        outputWidth,
                                                        stimuliHeight,
                                                        stimuliWidth,
                                                        nbFeature,
                                                        featureChannels,
                                                        featureHeight,
                                                        featureWidth,
                                                        poolType,
                                                        /*nbProposals,*/
                                                        nbP,
                                                        ignorePadding,
                                                        isFlip);


            auto layer = mNetDef.back()->addPlugin(&concat_tensor[0],
                                        nbFeature + 1,
                                        *pluginROIpool);

            layer->setName(outName.c_str());
            output_tensor.push_back(layer->getOutput(0));
            nvinfer1::Dims tensor_dims = output_tensor.back()->getDimensions();
            std::cout << "{";

            for(unsigned int d = 0; d < tensor_dims.nbDims; ++d)
                std::cout << tensor_dims.d[d] << " ";

            std::cout << "}" << std::endl;

/*
            for(unsigned int k = 0; k < nbProposals; ++k)
            {
                std::string proposalName = outName + "_" + std::to_string(k);
                layer->setName(outName.c_str());
                output_tensor.push_back(layer->getOutput(k));
                output_tensor.back()->setName(proposalName.c_str());
                std::cout << "               " << output_tensor.back()->getName() << std::endl;

                nvinfer1::Dims tensor_dims = output_tensor.back()->getDimensions();
                std::cout << "{";
                for(unsigned int d = 0; d < tensor_dims.nbDims; ++d)
                std::cout << tensor_dims.d[d] << " ";
                std::cout << "}" << std::endl;
            }
*/
        }

        return output_tensor;
}

void N2D2::Network::add_weighted(unsigned int nbOutputs,
                                    unsigned int outputsHeight,
                                    unsigned int outputsWidth,
                                    float* estimated_labels,
                                    unsigned int nbChannels,
                                    unsigned int image_height,
                                    unsigned int image_width,
                                    float* input_image,
                                    unsigned char* overlay_data,
                                    float alpha)
{
/*
    unsigned char* gpuWorkspace;
    const size_t outputWorkspace = image_height * image_width * nbChannels * mMaxBatchSize * sizeof(unsigned char);
    CHECK_CUDA_STATUS(cudaMalloc((void **)&gpuWorkspace, outputWorkspace));
*/
    const unsigned int groupSize = std::min(32, (int)(image_height * image_width));
    const unsigned int blockSize = std::ceil((int)image_height * image_width / groupSize);

    const dim3 threadsPerBlocks = {groupSize, 1, 1};
    const dim3 blocksPerGrid = {blockSize, 1, mMaxBatchSize};

    //Use INTERNEAREST resize factor if output image and input image dont have the same size
    const float multy = ((float) outputsHeight)/((float) image_height);
    const float multx = ((float) outputsWidth)/((float) image_width);

    cuda_add_weighted( mMaxBatchSize,
                       nbOutputs,
                       outputsHeight,
                       outputsWidth,
                       estimated_labels,
                       nbChannels,
                       image_height,
                       image_width,
                       input_image,
                       mWorkspaceGPU,
                       alpha,
                       threadsPerBlocks,
                       blocksPerGrid,
                       mDataStream);

    CHECK_CUDA_STATUS(cudaMemcpy(overlay_data,
                                 mWorkspaceGPU,
                                 nbChannels*image_height*image_width*mMaxBatchSize,
                                 cudaMemcpyDeviceToHost));

}


/****Targets Layers ****/
void N2D2::Network::output_generation(unsigned int mMaxBatchSize,
                       unsigned int nbOutputs,
                       void* dataIn,
                       uint32_t* outputEstimated,
                       cudaStream_t stream)
{
    float* outputsData(NULL);

    if (outputsData == NULL) {
        outputsData = new float[mMaxBatchSize * nbOutputs];

        if (!outputsData)
            throw std::runtime_error(
                "output_generation(): could not allocate memory");
    }

    CHECK_CUDA_STATUS(cudaMemcpy(outputsData,
                                 dataIn,
                                 mMaxBatchSize * nbOutputs * sizeof(float),
                                 cudaMemcpyDeviceToHost));

    CHECK_CUDA_STATUS(cudaStreamSynchronize(stream));

    for (unsigned int i = 0; i < mMaxBatchSize; i++) {

        float maxVal = outputsData[i * nbOutputs];
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

void N2D2::Network::spatial_output_generation(unsigned int mMaxBatchSize,
                               unsigned int nbOutputs,
                               unsigned int outputsHeight,
                               unsigned int outputsWidth,
                               void* dataIn,
                               uint32_t* outputEstimated,
                               cudaStream_t stream,
                               float threshold,
                               bool useGPU)
{


    if(useGPU)
    {
        uint32_t *gpuEstimated;
        const size_t outputWorkspace = outputsHeight * outputsWidth * mMaxBatchSize * sizeof(uint32_t);
        CHECK_CUDA_STATUS(cudaMalloc((void **)&gpuEstimated, outputWorkspace));

        const unsigned int groupSize = std::min(32, (int)(outputsHeight * outputsWidth));
        const unsigned int blockSize = std::ceil((int)outputsHeight * outputsWidth / groupSize);

        const dim3 threadsPerBlocks = {groupSize, 1, 1};
        const dim3 blocksPerGrid = {blockSize, 1, mMaxBatchSize};

        cuda_spatial_outputs(nbOutputs,
                             outputsHeight,
                             outputsWidth,
                             mMaxBatchSize,
                             threshold,
                             reinterpret_cast<float *>(dataIn),
                             gpuEstimated,
                             threadsPerBlocks,
                             blocksPerGrid,
                             stream);

        CHECK_CUDA_STATUS(cudaMemcpy(outputEstimated,
                                     gpuEstimated,
                                     outputWorkspace,
                                     cudaMemcpyDeviceToHost));

        CHECK_CUDA_STATUS(cudaFree(gpuEstimated));
    }
    else
    {
        const unsigned int size = nbOutputs * outputsHeight * outputsWidth;
        float* outputsData(NULL);
        if (outputsData == NULL) {
            outputsData = new float[mMaxBatchSize * size];

            if (!outputsData)
                throw std::runtime_error(
                    "spatial_output_generation(): could not allocate memory");
        }
        CHECK_CUDA_STATUS(cudaMemcpy(outputsData,
                                    dataIn,
                                    mMaxBatchSize * nbOutputs * outputsWidth * outputsHeight * sizeof(float),
                                    cudaMemcpyDeviceToHost));

        for (unsigned int i = 0; i < mMaxBatchSize; i++) {
            for (unsigned int oy = 0; oy < outputsHeight; ++oy) {
                for (unsigned int ox = 0; ox < outputsWidth; ++ox) {
                    const unsigned int inputsIdx
                        = ox + oy * outputsWidth
                        + i * (outputsHeight * outputsWidth * nbOutputs);
                    float maxVal = outputsData[inputsIdx];
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

        delete[] outputsData;
    }
}

void N2D2::Network::get_output( unsigned int nbOutputs,
                                unsigned int outputsHeight,
                                unsigned int outputsWidth,
                                void* dataIn,
                                float* outputEstimated)
{
    const unsigned int size = nbOutputs * outputsHeight * outputsWidth;

    if (outputEstimated == NULL) {
        std::cout << "outputEstimated is NULL" << std::endl;
        outputEstimated = new float[mMaxBatchSize * size];

        if (!outputEstimated)
            throw std::runtime_error(
                "get_output(): could not allocate memory");
    }

    CHECK_CUDA_STATUS(cudaMemcpy(outputEstimated,
                                 dataIn,
                                 mMaxBatchSize * nbOutputs * outputsWidth * outputsHeight * sizeof(float),
                                 cudaMemcpyDeviceToHost));
    
}

void N2D2::Network::reportProfiling(unsigned int nbIter)
{
    double totalProcessTime = 0.0;

    for (size_t i = 0; i < gProfiler.mProfile.size(); i++)
        totalProcessTime += gProfiler.mProfile[i].second / (nbIter);

    for (size_t i = 0; i < gProfiler.mProfile.size(); i++)
    {
        const double processTimeMs = gProfiler.mProfile[i].second / (nbIter);
        const double workLoad = (processTimeMs / totalProcessTime) * 100.0;
        std::string barrelLoad(((unsigned int)workLoad + 1) * 2, '*');
        std::cout << std::setprecision(10)
                  << "(" << std::setfill('0') << std::setw(2)
                  << (unsigned int)workLoad << "%)  " << barrelLoad
                  << "    " << gProfiler.mProfile[i].first << ": "
                  << processTimeMs << " ms"
                  << std::endl;
    }
    std::cout << "Average profiled tensorRT process time per stimulus = "
              << totalProcessTime <<  " ms" << std::endl;

}
/**** Debug Function ****/
void N2D2::Network::dumpMem(int size, float* data, std::string fileName)
{

    std::ofstream file;
    file.open(fileName.c_str());

    float* eagleEyes(NULL);
    eagleEyes = new float[size];

    CHECK_CUDA_STATUS(cudaMemcpy(
        eagleEyes, data, size * sizeof(float), cudaMemcpyDeviceToHost));

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

#ifdef WRAPPER_PYTHON

template<class T>
struct VecToList
{
    static PyObject* convert(const std::vector<T>& vec)
    {
        p::list* l = new p::list();
        for(size_t i = 0; i < vec.size(); i++) {
            l->append(vec[i]);
        }

        return l->ptr();
    }
};

void N2D2::Network::asyncExePy(np::ndarray const & in_data, unsigned int batchSize)
{
    asyncExe<float>(reinterpret_cast<float*>(in_data.get_data()), batchSize);
}

void N2D2::Network::syncExePy(np::ndarray const & in_data, unsigned int batchSize)
{
    syncExe<float>(reinterpret_cast<float*>(in_data.get_data()), batchSize);
}

void N2D2::Network::cpyOutputPy(np::ndarray const & output, unsigned int target)
{
    log_output(reinterpret_cast<float*>(output.get_data()), target);
}

void N2D2::Network::estimatedPy(np::ndarray const & in_data, unsigned int target, bool useGPU, float threshold)
{
    estimated(reinterpret_cast<unsigned int*>(in_data.get_data()), target, useGPU, threshold);
}

void N2D2::Network::addOverlayPy(np::ndarray const & overlay_data, unsigned int target, float threshold)
{
    addOverlay(reinterpret_cast<unsigned char*>(overlay_data.get_data()), target, threshold);
}

using namespace boost::python;

BOOST_PYTHON_MODULE(N2D2)
{

    np::initialize();

    class_<N2D2::Network>("N2D2_Network")
        .def(init<>())
        .def("initialize", &N2D2::Network::initialize)

        .def("setPrecision", &N2D2::Network::setPrecision)
        .def("reportProfiling", &N2D2::Network::reportProfiling)
        .def("setProfiling", &N2D2::Network::setProfiling)
        .def("setMaxBatchSize", &N2D2::Network::setMaxBatchSize)
        .def("setDeviceID", &N2D2::Network::setDeviceID)
        .def("setIterBuild", &N2D2::Network::setIterBuild)
        .def("setInputEngine", &N2D2::Network::setInputEngine)
        .def("setOutputEngine", &N2D2::Network::setOutputEngine)
        .def("setCalibCache", &N2D2::Network::setCalibCache)
        .def("setCalibFolder", &N2D2::Network::setCalibFolder)
        .def("setParamPath", &N2D2::Network::setParamPath)
        .def("setDetectorNMS", &N2D2::Network::setDetectorNMS)
#ifdef ONNX
        .def("setONNXModel", &N2D2::Network::setONNXModel)
#endif
        .def("setInputDims", &N2D2::Network::setInputDims)
        .def("setOutputNbTargets", &N2D2::Network::setOutputNbTargets)
        .def("setOutputTarget", &N2D2::Network::setOutputTarget)
           
        .def("estimated", &N2D2::Network::estimatedPy)

        .def("getOutputNbTargets", &N2D2::Network::getOutputNbTargets)
        .def("getOutputTarget", &N2D2::Network::getOutputTarget)
        .def("getOutputDimZ", &N2D2::Network::getOutputDimZ)
        .def("getOutputDimY", &N2D2::Network::getOutputDimY)
        .def("getOutputDimX", &N2D2::Network::getOutputDimX)
        .def("getInputDimZ", &N2D2::Network::getInputDimZ)
        .def("getInputDimY", &N2D2::Network::getInputDimY)
        .def("getInputDimX", &N2D2::Network::getInputDimX)

        .def("cpyOutput", &N2D2::Network::cpyOutputPy)
        .def("asyncExe", &N2D2::Network::asyncExePy)
        .def("syncExe", &N2D2::Network::syncExePy)
        .def("addOverlay", &N2D2::Network::addOverlayPy)
        .def("setDetectorThresholds", &N2D2::Network::setDetectorThresholdsPy)
       
//        .def("syncExeGPU", &N2D2::Network::syncExeGPUPy)
    ;
    p::to_python_converter<std::vector<unsigned int, std::allocator<unsigned int> >, VecToList<unsigned int> >();

}


#endif
