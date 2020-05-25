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

#include "n2d2_tensorRT.hpp"
#include "fp16.h"
#include <iostream>

tsrRTHandleStruct tsrRTHandles;

void set_profiling()
{
  tsrRTHandles.context->setProfiler(&gProfiler);
}

void createContext(unsigned int batchSize,
                   unsigned int iterBuild,
                   PluginFactory& factory,
                   std::string inputEngine,
                   std::string outputEngine,
                   bool useINT8)
{
    nvinfer1::ICudaEngine* cudaEngine = nullptr;

#if NV_TENSORRT_MAJOR > 2
    std::shared_ptr<nvinfer1::IInt8Calibrator> calibrator;
#endif

    std::cout << "Start createContext" << std::endl;
    if(inputEngine.empty())
    {

        tsrRTHandles.netBuilder->setMaxBatchSize(batchSize);
        tsrRTHandles.netBuilder->setMinFindIterations(iterBuild);
        tsrRTHandles.netBuilder->setMaxWorkspaceSize(128<<20);
#if NV_TENSORRT_MAJOR < 5
        if(tsrRTHandles.dT == nvinfer1::DataType::kHALF)
            tsrRTHandles.netBuilder->setHalf2Mode(true);
#else
        if(tsrRTHandles.dT == nvinfer1::DataType::kHALF)
            tsrRTHandles.netBuilder->setFp16Mode(true);
#endif
        if(useINT8)
        {
#if NV_TENSORRT_MAJOR > 2

            std::string calibDir = "./batches_calib/";
            std::vector<std::string> filesCalib;

            struct dirent* pFile;
            DIR* pDir = opendir(calibDir.c_str());
            if (pDir == NULL)
                throw std::runtime_error(
                    "Couldn't open the directory for input patterns: " + calibDir);

            while ((pFile = readdir(pDir)) != NULL) {
                if (pFile->d_name[0] != '.')
                    filesCalib.push_back(std::string(calibDir + pFile->d_name));
            }
            closedir(pDir);
            unsigned int nbCalibFiles = filesCalib.size();
            if(nbCalibFiles == 0)
                throw std::runtime_error("Cannot find calibration files in dir " + calibDir);

            std::cout << "Using Entropy Calibrator" << std::endl;
            BatchStream calibrationStream(1, nbCalibFiles, calibDir + "batch_calibration");
            calibrator.reset(new Int8EntropyCalibrator(calibrationStream, 0));
            tsrRTHandles.netBuilder->setInt8Mode(true);
            tsrRTHandles.netBuilder->setInt8Calibrator(calibrator.get());
#endif
        }

        cudaEngine = tsrRTHandles.netBuilder->buildCudaEngine(*tsrRTHandles.netDef.back());
        std::cout << "buildCudaEngine done" << std::endl;
        tsrRTHandles.netBuilder->destroy();
        std::cout << "netBuilder->destroy" << std::endl;

    }
    else
    {
        std::ifstream cudaEngineStream(inputEngine);

        if(!cudaEngineStream.good())
            throw std::runtime_error("Could not open cuda engine file: " + inputEngine);

        nvinfer1::IRuntime* infer = nvinfer1::createInferRuntime(gLogger);
#if NV_TENSORRT_MAJOR > 1
        // support for stringstream deserialization was deprecated in TensorRT v2
        // instead, read the stringstream into a memory buffer and pass that to TRT.
        cudaEngineStream.seekg(0, std::ios::end);
        const int modelSize = cudaEngineStream.tellg();
        cudaEngineStream.seekg(0, std::ios::beg);
        void* modelMem = malloc(modelSize);
        if( !modelMem )
            throw std::runtime_error("Could not allocate enough memory for load cuda engine file " + inputEngine);

        cudaEngineStream.read((char*)modelMem, modelSize);
        cudaEngine = infer->deserializeCudaEngine(modelMem, 
                                                  modelSize, 
                                                  &factory);
        free(modelMem);
#else
       // TensorRT v1 can deserialize directly from stringstream
       cudaEngine = infer->deserializeCudaEngine(cudaEngineStream);
#endif
    }
        std::cout << "cudaEngine->serialize" << std::endl;

    nvinfer1::IHostMemory *gieModelStream = cudaEngine->serialize();
        std::cout << "serialize done" << std::endl;

    if(!outputEngine.empty())
    {
        std::cout << "Save cuda engine file at " << outputEngine << std::endl;
        std::ofstream engineSerializedFile; //Create output file stream
        engineSerializedFile.open(outputEngine, std::ios::out | std::ios::binary); // Open a new file

        if (engineSerializedFile.is_open() && engineSerializedFile.good() && !engineSerializedFile.fail()) {
           //Save the serialized engine data into the file
           engineSerializedFile.write(reinterpret_cast<const char *>(gieModelStream->data()), gieModelStream->size());
           engineSerializedFile.close();
        }
        else
            throw std::runtime_error("Could not save cuda engine file at  " + outputEngine);

    }

    cudaEngine->destroy();
#if NV_TENSORRT_MAJOR > 2
    // Once the engine is built. Its safe to destroy the calibrator.
    calibrator.reset();
#endif

    factory.destroyPlugin();
    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);
    nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(gieModelStream->data(),
                                                                   gieModelStream->size(),
                                                                   &factory);
#if NV_TENSORRT_MAJOR > 4
    if(runtime->getNbDLACores() > 1)
        runtime->setDLACore(runtime->getNbDLACores() - 1) ;

    std::cout << "Available DLA Cores / Used DLA Cores: " << runtime->getNbDLACores() << " / " << runtime->getDLACore() << std::endl;
#endif



    if (gieModelStream)
        gieModelStream->destroy();

    tsrRTHandles.context = engine->createExecutionContext();

    factory.destroyPlugin();

}

void add_target(nvinfer1::INetworkDefinition* net,
                std::vector<nvinfer1::ITensor *> outputs_tensor,
                unsigned int targetIdx)
{
    for(unsigned int i = 0; i < outputs_tensor.size(); ++i)
    {
        std::string target_name = "Target_" + std::to_string(targetIdx)
                                    + "_" + std::to_string(i);
        outputs_tensor[i]->setType(nvinfer1::DataType::kFLOAT);
        outputs_tensor[i]->setName(target_name.c_str());

        net->markOutput(*outputs_tensor[i]);
    }

}

std::vector<nvinfer1::ITensor *>
        add_activation(nvinfer1::INetworkDefinition* net,
                        nvinfer1::IBuilder* netBuilder,
                        bool useDLA,
                        nvinfer1::DataType dT,
                        std::string layerName,
                        nvinfer1::ActivationType activation,
                        std::vector<nvinfer1::ITensor *> inputs_tensor)
{
        std::vector<nvinfer1::ITensor *> output_tensor;
        std::cout << "Add activation layer: " << layerName << std::endl;
        for(unsigned int i = 0; i < inputs_tensor.size(); ++i)
        {
            std::string outName = layerName + "_Activation_" + std::to_string(i);
            auto layer = net->addActivation(*inputs_tensor[i],
                                            activation);

           layer->setName(outName.c_str());

#if NV_TENSORRT_MAJOR > 4
            if(useDLA)
            {
                if(tsrRTHandles.netBuilder->canRunOnDLA(layer) 
                        && (dT == nvinfer1::DataType::kHALF || dT == nvinfer1::DataType::kINT8))
                {
                    layer->setPrecision(dT);
                    std::cout << "Layer: " << layer->getName() 
                                << " will run on DLA (batch size max: " 
                                <<  netBuilder->getMaxDLABatchSize()
                                << std::endl;
                    netBuilder->setDeviceType(layer, nvinfer1::DeviceType::kDLA);
                }
                else
                    throw std::runtime_error("Cannot use DLA for layer " + outName);
            }
#endif

           output_tensor.push_back(layer->getOutput(0));
           output_tensor.back()->setType(dT);
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
        add_convolution(nvinfer1::INetworkDefinition* net,
                        nvinfer1::IBuilder* netBuilder,
                        bool useDLA,
                        nvinfer1::DataType dT,
                        std::string layerName,
                        LayerActivation activation,
                        unsigned int nbOutputs,
                        unsigned int strideX,
                        unsigned int strideY,
                        unsigned int paddingX,
                        unsigned int paddingY,
                        unsigned int kernelW,
                        unsigned int kernelH,
                        std::vector<nvinfer1::ITensor *> inputs_tensor,
                        //const WDATA_T* weights,
                        std::string wFile,
                        unsigned int weights_size,
                        //const WDATA_T* bias,
                        std::string bFile,
                        unsigned int bias_size)
{

        std::vector<nvinfer1::ITensor *> output_tensor;
        std::cout << "Add convolution layer: " << layerName << std::endl;

        std::ifstream weights(wFile.c_str());
        if (!weights.good())
                throw std::runtime_error("Could not open synaptic file: "
                                        + wFile);

        WDATA_T* weight_wdata;
        __half* weight_hdata;

        if(dT != nvinfer1::DataType::kHALF)
            weight_wdata = new WDATA_T[weights_size];
        else
            weight_hdata = new __half[weights_size];

        WDATA_T w;

        for (unsigned int i = 0; i < weights_size; ++i) {
            if (!(weights >> w))
                throw std::runtime_error( "Error while reading synaptic file: " + wFile);

            if(dT != nvinfer1::DataType::kHALF)
                weight_wdata[i] = w;
            else
                weight_hdata[i] = fp16::__float2half(w);
        }
        weights.close();

        std::ifstream bias(bFile.c_str());
        if (!bias.good())
                throw std::runtime_error("Could not open synaptic file: "
                                        + bFile);

        WDATA_T* bias_wdata;
        __half* bias_hdata;

        if(dT != nvinfer1::DataType::kHALF)
            bias_wdata = new WDATA_T[bias_size];
        else
            bias_hdata = new __half[bias_size];

        WDATA_T b;

        for (unsigned int i = 0; i < bias_size; ++i) {
            if (!(bias >> b))
                throw std::runtime_error( "Error while reading synaptic file: " + bFile);
            if(dT != nvinfer1::DataType::kHALF)
                bias_wdata[i] = b;
            else
                bias_hdata[i] = fp16::__float2half(b);
        }
        bias.close();

        //nvinfer1::Weights weights_trt  = {dT, weights, weights_size};
        nvinfer1::Weights weights_trt;

        if(dT != nvinfer1::DataType::kHALF)
            weights_trt = {dT, weight_wdata, weights_size};
        else
            weights_trt = {dT, weight_hdata, weights_size};

        //nvinfer1::Weights bias_trt  = {dT, bias, bias_size};
        nvinfer1::Weights bias_trt;
        if(dT != nvinfer1::DataType::kHALF)
            bias_trt = {dT, bias_wdata, bias_size};
        else
            bias_trt = {dT, bias_hdata, bias_size};

        //delete[] bias_data;
        //delete[] weight_data;

        nvinfer1::DimsHW kernelDims = {(int) kernelH, (int)kernelW};
        nvinfer1::DimsHW strideDims = {(int)strideY, (int)strideX};
        nvinfer1::DimsHW paddingDims = {(int)paddingY, (int)paddingX};

        for(unsigned int i = 0; i < inputs_tensor.size(); ++i)
        {
            std::string outName = layerName + "_" + std::to_string(i);
            auto layer = net->addConvolution(*inputs_tensor[i],
                                             nbOutputs,
                                             kernelDims,
                                             weights_trt,
                                             bias_trt);
            layer->setStride(strideDims);
            layer->setPadding(paddingDims);
            layer->setName(outName.c_str());
#if NV_TENSORRT_MAJOR > 4
            if(useDLA)
            {
                if(tsrRTHandles.netBuilder->canRunOnDLA(layer) 
                        && (dT == nvinfer1::DataType::kHALF || dT == nvinfer1::DataType::kINT8))
                {
                    layer->setPrecision(dT);
                    std::cout << "Layer: " << layer->getName() 
                                << " will run on DLA (batch size max: " 
                                <<  netBuilder->getMaxDLABatchSize()
                                << std::endl;
                    netBuilder->setDeviceType(layer, nvinfer1::DeviceType::kDLA);
                }
                else
                    throw std::runtime_error("Cannot use DLA for layer " + outName);
            }
#endif

            nvinfer1::Dims tensor_in_dims = inputs_tensor[i]->getDimensions();

            output_tensor.push_back(layer->getOutput(0));
            output_tensor.back()->setType(dT);
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
            return(add_activation(net,
                                  netBuilder,
                                  useDLA,
                                  dT,
                                  layerName,
                                  activation.type,
                                  output_tensor));
        else
            return output_tensor;

}

std::vector<nvinfer1::ITensor *>
        add_deconvolution(nvinfer1::INetworkDefinition* net,
                        nvinfer1::IBuilder* netBuilder,
                        bool useDLA,
                        nvinfer1::DataType dT,
                        std::string layerName,
                        LayerActivation activation,
                        unsigned int nbOutputs,
                        unsigned int strideX,
                        unsigned int strideY,
                        unsigned int paddingX,
                        unsigned int paddingY,
                        unsigned int kernelW,
                        unsigned int kernelH,
                        std::vector<nvinfer1::ITensor *> inputs_tensor,
                        //const WDATA_T* weights,
                        std::string wFile,
                        unsigned int weights_size,
                        //const WDATA_T* bias,
                        std::string bFile,
                        unsigned int bias_size)
{

        std::vector<nvinfer1::ITensor *> output_tensor;
        std::cout << "Add deconvolution layer: " << layerName << std::endl;

        std::ifstream weights(wFile.c_str());
        if (!weights.good())
                throw std::runtime_error("Could not open synaptic file: "
                                        + wFile);

        WDATA_T* weight_wdata;
        __half* weight_hdata;

        if(dT != nvinfer1::DataType::kHALF)
            weight_wdata = new WDATA_T[weights_size];
        else
            weight_hdata = new __half[weights_size];

        WDATA_T w;

        for (unsigned int i = 0; i < weights_size; ++i) {
            if (!(weights >> w))
                throw std::runtime_error( "Error while reading synaptic file: " + wFile);

            if(dT != nvinfer1::DataType::kHALF)
                weight_wdata[i] = w;
            else
                weight_hdata[i] = fp16::__float2half(w);
        }
        weights.close();

        std::ifstream bias(bFile.c_str());
        if (!bias.good())
                throw std::runtime_error("Could not open synaptic file: "
                                        + bFile);

        WDATA_T* bias_wdata;
        __half* bias_hdata;

        if(dT != nvinfer1::DataType::kHALF)
            bias_wdata = new WDATA_T[bias_size];
        else
            bias_hdata = new __half[bias_size];

        WDATA_T b;

        for (unsigned int i = 0; i < bias_size; ++i) {
            if (!(bias >> b))
                throw std::runtime_error( "Error while reading synaptic file: " + bFile);
            if(dT != nvinfer1::DataType::kHALF)
                bias_wdata[i] = b;
            else
                bias_hdata[i] = fp16::__float2half(b);
        }

        bias.close();

        //nvinfer1::Weights weights_trt  = {dT, weights, weights_size};
        nvinfer1::Weights weights_trt;

        if(dT != nvinfer1::DataType::kHALF)
            weights_trt = {dT, weight_wdata, weights_size};
        else
            weights_trt = {dT, weight_hdata, weights_size};

        //nvinfer1::Weights bias_trt  = {dT, bias, bias_size};
        nvinfer1::Weights bias_trt;
        if(dT != nvinfer1::DataType::kHALF)
            bias_trt = {dT, bias_wdata, bias_size};
        else
            bias_trt = {dT, bias_hdata, bias_size};
        //delete[] bias_data;
        //delete[] weight_data;

        nvinfer1::DimsHW kernelDims = {(int) kernelH, (int)kernelW};
        nvinfer1::DimsHW strideDims = {(int)strideY, (int)strideX};
        nvinfer1::DimsHW paddingDims = {(int)paddingY, (int)paddingX};

        for(unsigned int i = 0; i < inputs_tensor.size(); ++i)
        {
            std::string outName = layerName + "_" + std::to_string(i);
            auto layer = net->addDeconvolution(*inputs_tensor[i],
                                             nbOutputs,
                                             kernelDims,
                                             weights_trt,
                                             bias_trt);
            layer->setStride(strideDims);
            layer->setPadding(paddingDims);
            layer->setName(outName.c_str());
#if NV_TENSORRT_MAJOR > 4
            if(useDLA)
            {
                if(tsrRTHandles.netBuilder->canRunOnDLA(layer) 
                        && (dT == nvinfer1::DataType::kHALF || dT == nvinfer1::DataType::kINT8))
                {
                    layer->setPrecision(dT);
                    std::cout << "Layer: " << layer->getName() 
                                << " will run on DLA (batch size max: " 
                                <<  netBuilder->getMaxDLABatchSize()
                                << std::endl;
                    netBuilder->setDeviceType(layer, nvinfer1::DeviceType::kDLA);
                }
                else
                    throw std::runtime_error("Cannot use DLA for layer " + outName);
            }
#endif

            nvinfer1::Dims tensor_in_dims = inputs_tensor[i]->getDimensions();

            output_tensor.push_back(layer->getOutput(0));
            output_tensor.back()->setType(dT);
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
            return(add_activation(net,
                                  netBuilder,
                                  useDLA,
                                  dT,
                                  layerName,
                                  activation.type,
                                  output_tensor));
        else
            return output_tensor;

}


std::vector<nvinfer1::ITensor *>
            add_padding(nvinfer1::INetworkDefinition* net,
                        nvinfer1::IBuilder* netBuilder,
                        nvinfer1::DataType dT,
                        std::string layerName,
                        unsigned int nbOutputs,
                        std::vector<nvinfer1::ITensor *> inputs_tensor,
                        const int pad_top,
                        const int pad_bottom,
                        const int pad_left,
                        const int pad_right)
{
    std::vector<nvinfer1::ITensor *> output_tensor;
    std::cout << "Add paddinglayer: " << layerName << std::endl;
    nvinfer1::DimsHW prePad = {pad_top, pad_left};
    nvinfer1::DimsHW postPad = {pad_bottom, pad_right};

    for(unsigned int i = 0; i < inputs_tensor.size(); ++i)
    {
        std::string outName = layerName + "_" + std::to_string(i);
        auto layer = net->addPadding(*inputs_tensor[i],
                                       prePad,
                                       postPad);
        layer->setName(outName.c_str());
        output_tensor.push_back(layer->getOutput(0));
        output_tensor.back()->setType(dT);
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
            add_lrn(nvinfer1::INetworkDefinition* net,
                        nvinfer1::IBuilder* netBuilder,
                        bool useDLA,
                        nvinfer1::DataType dT,
                        std::string layerName,
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
        auto layer = net->addLRN(*inputs_tensor[i],
                                 windows,
                                 alpha,
                                 beta,
                                 k);

        layer->setName(outName.c_str());

#if NV_TENSORRT_MAJOR > 4
        if(useDLA)
        {
            if(tsrRTHandles.netBuilder->canRunOnDLA(layer) 
                    && (dT == nvinfer1::DataType::kHALF || dT == nvinfer1::DataType::kINT8))
            {
                layer->setPrecision(dT);
                std::cout << "Layer: " << layer->getName() 
                            << " will run on DLA (batch size max: " 
                            <<  netBuilder->getMaxDLABatchSize()
                            << std::endl;
                netBuilder->setDeviceType(layer, nvinfer1::DeviceType::kDLA);
            }
            else
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
        add_fc(nvinfer1::INetworkDefinition* net,
               nvinfer1::IBuilder* netBuilder,
                nvinfer1::DataType dT,
                std::string layerName,
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

        WDATA_T* weight_wdata;
        __half* weight_hdata;

        if(dT != nvinfer1::DataType::kHALF)
            weight_wdata = new WDATA_T[weights_size];
        else
            weight_hdata = new __half[weights_size];

        WDATA_T w;

        for (unsigned int i = 0; i < weights_size; ++i) {
            if (!(weights >> w))
                throw std::runtime_error( "Error while reading synaptic file: " + wFile);

            if(dT != nvinfer1::DataType::kHALF)
                weight_wdata[i] = w;
            else
                weight_hdata[i] = fp16::__float2half(w);
        }
        weights.close();

        std::ifstream bias(bFile.c_str());
        if (!bias.good())
                throw std::runtime_error("Could not open synaptic file: "
                                        + bFile);

        WDATA_T* bias_wdata;
        __half* bias_hdata;

        if(dT != nvinfer1::DataType::kHALF)
            bias_wdata = new WDATA_T[nbOutputs];
        else
            bias_hdata = new __half[nbOutputs];

        WDATA_T b;

        for (unsigned int i = 0; i < nbOutputs; ++i) {
            if (!(bias >> b))
                throw std::runtime_error( "Error while reading synaptic file: " + bFile);
            if(dT != nvinfer1::DataType::kHALF)
                bias_wdata[i] = b;
            else
                bias_hdata[i] = fp16::__float2half(b);
        }
        bias.close();

        //nvinfer1::Weights weights_trt  = {dT, weights, weights_size};
        nvinfer1::Weights weights_trt;

        if(dT != nvinfer1::DataType::kHALF)
            weights_trt = {dT, weight_wdata, weights_size};
        else
            weights_trt = {dT, weight_hdata, weights_size};

        //nvinfer1::Weights bias_trt  = {dT, bias, bias_size};
        nvinfer1::Weights bias_trt;
        if(dT != nvinfer1::DataType::kHALF)
            bias_trt = {dT, bias_wdata, nbOutputs};
        else
            bias_trt = {dT, bias_hdata, nbOutputs};

        for(unsigned int i = 0; i < inputs_tensor.size(); ++i)
        {
            std::string outName = layerName + "_" + std::to_string(i);
            auto layer = net->addFullyConnected(*inputs_tensor[i],
                                             nbOutputs,
                                             weights_trt,
                                             bias_trt);

           layer->setName(outName.c_str());
           output_tensor.push_back(layer->getOutput(0));
           output_tensor.back()->setType(dT);
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
            return(add_activation(net,
                                  netBuilder,
                                  false,
                                  dT,
                                  layerName,
                                  activation.type,
                                  output_tensor));
        else
            return output_tensor;
}


std::vector<nvinfer1::ITensor *>
        add_concat(nvinfer1::INetworkDefinition* net,
                    nvinfer1::IBuilder* netBuilder,
                    bool useDLA,
                   nvinfer1::DataType dT,
                   std::string layerName,
                   unsigned int nbInputs,
                   /*std::vector<nvinfer1::ITensor *> const* inputs_tensor*/
                    std::vector<std::vector<nvinfer1::ITensor *> *> inputs_tensor)
{
        std::vector<nvinfer1::ITensor *> output_tensor;

/*

        for(unsigned int i = 0; i < inputs_tensor[0].size(); ++i)
        {
            std::string outName = layerName + "_" + std::to_string(i);
            auto layer = net->addConcatenation(&inputs_tensor[i][0],
                                             nbInputs);
           layer->setName(outName.c_str());

           output_tensor.push_back(layer->getOutput(0));
           output_tensor.back()->setName(outName.c_str());
        }
        return output_tensor;
*/
/*
        std::vector<nvinfer1::ITensor *> output_tensor;
        std::cout << "Add concat layer: " << layerName << std::endl;
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
        for(unsigned int i = 0; i < inputs_tensor[0]->size(); ++i)
        {
            std::vector<nvinfer1::ITensor *> concat_tensor;

            for(unsigned int k = 0; k < inputs_tensor.size(); ++k)
            {
              nvinfer1::ITensor * input_tensor = (inputs_tensor[k])->data()[i];
              concat_tensor.push_back(input_tensor);
            }

            std::string outName = layerName + "_" + std::to_string(i);
            //auto layer = net->addConcatenation((tmp_tensor[i])->data(),
            //                                 nbInputs);
            auto layer = net->addConcatenation(&concat_tensor[0],
                                             nbInputs);
           layer->setName(outName.c_str());

#if NV_TENSORRT_MAJOR > 4
            if(useDLA)
            {
                if(tsrRTHandles.netBuilder->canRunOnDLA(layer) 
                        && (dT == nvinfer1::DataType::kHALF || dT == nvinfer1::DataType::kINT8))
                {
                    layer->setPrecision(dT);
                    std::cout << "Layer: " << layer->getName() 
                                << " will run on DLA (batch size max: " 
                                <<  netBuilder->getMaxDLABatchSize()
                                << std::endl;
                    netBuilder->setDeviceType(layer, nvinfer1::DeviceType::kDLA);
                }
                else
                    throw std::runtime_error("Cannot use DLA for layer " + outName);
            }
#endif
           output_tensor.push_back(layer->getOutput(0));
           output_tensor.back()->setName(outName.c_str());
           output_tensor.back()->setType(dT);

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
        add_elementwise(nvinfer1::INetworkDefinition* net,
                        nvinfer1::IBuilder* netBuilder,
                        bool useDLA,
                        nvinfer1::DataType dT,
                        std::string layerName,
                        LayerActivation activation,
                        unsigned int batchSize,
                        unsigned int nbOutputs,
                        unsigned int outputHeight,
                        unsigned int outputWidth,
                        std::vector<std::vector<nvinfer1::ITensor *> > inputs_tensor,
                        nvinfer1::ElementWiseOperation op,
                        WDATA_T* scales,
                        WDATA_T* shift,
                        WDATA_T* power)
{
        std::vector<nvinfer1::ITensor *> output_tensor;
        std::vector<std::vector<nvinfer1::ITensor *>> scale_tensor;

        std::cout << "Add elementwize layer: " << layerName << std::endl;
        nvinfer1::ScaleMode modeScale = nvinfer1::ScaleMode::kUNIFORM;

        /**
            This layer applies a per-elements tensor computation to its inputA and B:
                output = (input* scale + shift)^ power
        **/
        __half* scale_half;
        __half* shift_half;
        __half* power_half;

        if(dT == nvinfer1::DataType::kHALF)
        {
            scale_half = new __half[inputs_tensor.size()];
            shift_half = new __half[inputs_tensor.size()];
            power_half = new __half[inputs_tensor.size()];

            for(unsigned int input = 0;
                    input < inputs_tensor.size();
                    ++input)
            {
                scale_half[input] = fp16::__float2half(scales[input]);
                shift_half[input] = fp16::__float2half(shift[input]);
                power_half[input] = fp16::__float2half(power[input]);
            }
        }


        for(unsigned int input = 0; input < inputs_tensor.size(); ++input)
        {
            nvinfer1::Weights scale_trt;
            nvinfer1::Weights shift_trt;
            nvinfer1::Weights power_trt;

            if(dT != nvinfer1::DataType::kHALF)
            {
                scale_trt  = {dT, scales + input, 1};
                shift_trt  = {dT, shift + input, 1};
                power_trt  = {dT, power + input, 1};
            }
            else
            {
                scale_trt  = {dT, scale_half + input, 1};
                shift_trt  = {dT, shift_half + input, 1};
                power_trt  = {dT, power_half + input, 1};
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

                    auto layer = net->addScale(*inputs_tensor[input][vecIn],
                                                modeScale,
                                                shift_trt,
                                                scale_trt,
                                                power_trt);

                    layer->setName(outName.c_str());
#if NV_TENSORRT_MAJOR > 4
                    if(useDLA)
                    {
                        if(tsrRTHandles.netBuilder->canRunOnDLA(layer) 
                                && (dT == nvinfer1::DataType::kHALF || dT == nvinfer1::DataType::kINT8))
                        {
                            layer->setPrecision(dT);
                            std::cout << "Layer: " << layer->getName() 
                                        << " will run on DLA (batch size max: " 
                                        <<  netBuilder->getMaxDLABatchSize()
                                        << std::endl;
                            netBuilder->setDeviceType(layer, nvinfer1::DeviceType::kDLA);
                        }
                        else
                            throw std::runtime_error("Cannot use DLA for layer " + outName);
                    }
#endif
                    scaleVecTensor.push_back(layer->getOutput(0));
                    scaleVecTensor.back()->setType(dT);

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
                    scaleVecTensor.back()->setType(dT);
                }
            }

            scale_tensor.push_back(scaleVecTensor);
        }

        for(unsigned int i = 0; i < scale_tensor[0].size(); ++i)
        {

            if(scale_tensor.size() == 2)
            {
                std::string outName = layerName + "_" + std::to_string(i);

                auto layer = net->addElementWise(*scale_tensor[0][i],
                                                    *scale_tensor[1][i],
                                                    op);
                layer->setName(outName.c_str());

#if NV_TENSORRT_MAJOR > 4
                if(useDLA)
                {
                    if(tsrRTHandles.netBuilder->canRunOnDLA(layer) 
                            && (dT == nvinfer1::DataType::kHALF || dT == nvinfer1::DataType::kINT8))
                    {
                        layer->setPrecision(dT);
                        std::cout << "Layer: " << layer->getName() 
                                    << " will run on DLA (batch size max: " 
                                    <<  netBuilder->getMaxDLABatchSize()
                                    << std::endl;
                        netBuilder->setDeviceType(layer, nvinfer1::DeviceType::kDLA);
                    }
                    else
                        throw std::runtime_error("Cannot use DLA for layer " + outName);
                }
#endif

                output_tensor.push_back(layer->getOutput(0));
                output_tensor.back()->setName(outName.c_str());
                output_tensor.back()->setType(dT);
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
                output_tensor.back()->setType(dT);
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
            return(add_activation(net,
                                  netBuilder,
                                  useDLA,
                                  dT,
                                  layerName,
                                  activation.type,
                                  output_tensor));
        else
            return output_tensor;
}



std::vector<nvinfer1::ITensor *>
        add_scale(nvinfer1::INetworkDefinition* net,
                        nvinfer1::IBuilder* netBuilder,
                        bool useDLA,
                        nvinfer1::DataType dT,
                        std::string layerName,
                        LayerActivation activation,
                        unsigned int batchSize,
                        unsigned int nbOutputs,
                        unsigned int outputHeight,
                        unsigned int outputWidth,
                        std::vector<nvinfer1::ITensor *> inputs_tensor,
                        DATA_T* scales,
                        DATA_T* shift,
                        DATA_T* power)
{

        std::vector<nvinfer1::ITensor *> output_tensor;
        std::cout << "Add scale layer: " << layerName << std::endl;

        nvinfer1::Weights scale_trt  = {dT, scales, 1};
        nvinfer1::Weights shift_trt  = {dT, shift, 1};
        nvinfer1::Weights power_trt  = {dT, power, 1};
        nvinfer1::ScaleMode modeScale = nvinfer1::ScaleMode::kUNIFORM;
        /**
            This layer applies a per-tensor computation to its input:
                output = (input* scale + shift)^ power
        **/
        for(unsigned int i = 0; i < inputs_tensor.size(); ++i)
        {
            std::string outName = layerName + "_" + std::to_string(i);

            auto layer = net->addScale(*inputs_tensor[i],
                                        modeScale,
                                        shift_trt,
                                        scale_trt,
					                    power_trt);
#if NV_TENSORRT_MAJOR > 4
            if(useDLA)
            {
                if(tsrRTHandles.netBuilder->canRunOnDLA(layer) 
                        && (dT == nvinfer1::DataType::kHALF || dT == nvinfer1::DataType::kINT8))
                {
                    layer->setPrecision(dT);
                    std::cout << "Layer: " << layer->getName() 
                                << " will run on DLA (batch size max: " 
                                <<  netBuilder->getMaxDLABatchSize()
                                << std::endl;
                    netBuilder->setDeviceType(layer, nvinfer1::DeviceType::kDLA);
                }
                else
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
            return(add_activation(net,
                                  netBuilder,
                                  useDLA,
                                  dT,
                                  layerName,
                                  activation.type,
                                  output_tensor));
        else
            return output_tensor;

}


std::vector<nvinfer1::ITensor *>
        add_batchnorm(nvinfer1::INetworkDefinition* net,
                        nvinfer1::IBuilder* netBuilder,
                        bool useDLA,
                        PluginFactory& factory,
                        nvinfer1::DataType dT,
                        std::string layerName,
                        LayerActivation activation,
                        unsigned int batchSize,
                        unsigned int nbOutputs,
                        unsigned int outputHeight,
                        unsigned int outputWidth,
                        std::vector<nvinfer1::ITensor *> inputs_tensor,
                        WDATA_T* scales,
                        WDATA_T* biases,
                        WDATA_T* means,
                        WDATA_T* variances,
                        WDATA_T epsilon)
{

        std::vector<nvinfer1::ITensor *> output_tensor;
        std::cout << "Add batchnorm layer: " << layerName << std::endl;

        nvinfer1::Weights scale_trt;
        nvinfer1::Weights shift_trt;
        nvinfer1::Weights power_trt;

        WDATA_T* scale_wdata;
        WDATA_T* shift_wdata;
        WDATA_T* power_wdata;
        __half* scale_hdata;
        __half* shift_hdata;
        __half* power_hdata;


        if(dT != nvinfer1::DataType::kHALF)
        {
            scale_wdata = new WDATA_T[nbOutputs];
            shift_wdata = new WDATA_T[nbOutputs];
            power_wdata = new WDATA_T[nbOutputs];
        }
        else
        {
            scale_hdata = new __half[nbOutputs];
            shift_hdata = new __half[nbOutputs];
            power_hdata = new __half[nbOutputs];
        }

        for(unsigned int  out = 0; out < nbOutputs; ++out)
        {
            if(dT != nvinfer1::DataType::kHALF)
            {
                scale_wdata[out] = scales[out]/sqrt(variances[out] + epsilon);
                shift_wdata[out] = ((-means[out]*scales[out])/sqrt(variances[out] + epsilon)) + biases[out];
                power_wdata[out] = (WDATA_T) 1;
            }
            else
            {
                scale_hdata[out] = fp16::__float2half( scales[out]/sqrt(variances[out] + epsilon) );
                shift_hdata[out] = fp16::__float2half( ((-means[out]*scales[out])/sqrt(variances[out] + epsilon)) + biases[out] );
                power_hdata[out] = fp16::__float2half(1);
            }
        }
        if(dT != nvinfer1::DataType::kHALF)
        {

            scale_trt  = {dT, scale_wdata, nbOutputs};
            shift_trt  = {dT, shift_wdata, nbOutputs};
            power_trt  = {dT, power_wdata, nbOutputs};
        }
        else
        {
            scale_trt  = {dT, scale_hdata, nbOutputs};
            shift_trt  = {dT, shift_hdata, nbOutputs};
            power_trt  = {dT, power_hdata, nbOutputs};
        }
        nvinfer1::ScaleMode modeScale = nvinfer1::ScaleMode::kCHANNEL;

        /**
            This layer applies a per-element computation to its input:
                output = (input* scale + shift)^ power
        **/
        for(unsigned int i = 0; i < inputs_tensor.size(); ++i)
        {
            std::string outName = layerName + "_" + std::to_string(i);

            auto layer = net->addScale(*inputs_tensor[i],
                                        modeScale,
                                        shift_trt,
                                        scale_trt,
					                    power_trt);
#if NV_TENSORRT_MAJOR > 4
            if(useDLA)
            {
                if(tsrRTHandles.netBuilder->canRunOnDLA(layer) 
                        && (dT == nvinfer1::DataType::kHALF || dT == nvinfer1::DataType::kINT8))
                {
                    layer->setPrecision(dT);
                    std::cout << "Layer: " << layer->getName() 
                                << " will run on DLA (batch size max: " 
                                <<  netBuilder->getMaxDLABatchSize()
                                << std::endl;
                    netBuilder->setDeviceType(layer, nvinfer1::DeviceType::kDLA);
                }
                else
                    throw std::runtime_error("Cannot use DLA for layer " + outName);
            }
#endif

            layer->setName(outName.c_str());
            output_tensor.push_back(layer->getOutput(0));
            output_tensor.back()->setName(outName.c_str());
            output_tensor.back()->setType(dT);

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
            nvinfer1::IPlugin* pluginBn = factory.createPlugin(outName.c_str(),
                                                                batchSize,
                                                                nbOutputs,
                                                                outputHeight,
                                                                outputWidth,
                                                                scales,
                                                                biases,
                                                                means,
                                                                variances,
                                                                epsilon);

            auto layer = net->addPlugin(&inputs_tensor[i],
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
            return(add_activation(net,
                                  netBuilder,
                                  useDLA,
                                  dT,
                                  layerName,
                                  activation.type,
                                  output_tensor));
        else
            return output_tensor;

}

std::vector<nvinfer1::ITensor *>
        add_pooling(nvinfer1::INetworkDefinition* net,
                        nvinfer1::IBuilder* netBuilder,
                        bool useDLA,
                        nvinfer1::DataType dT,
                        std::string layerName,
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
        nvinfer1::DimsHW poolDims = {(int)poolH, (int)poolW};
        nvinfer1::DimsHW strideDims = {(int)strideY, (int)strideX};
        nvinfer1::DimsHW paddingDims = {(int)paddingY, (int)paddingX};

        for(unsigned int i = 0; i < inputs_tensor.size(); ++i)
        {
            std::string outName = layerName + "_" + std::to_string(i);
            auto layer = net->addPooling(*inputs_tensor[i],
                                             poolType,
                                             poolDims);
           layer->setStride(strideDims);
           layer->setPadding(paddingDims);
           layer->setName(outName.c_str());
#if NV_TENSORRT_MAJOR > 4
            if(useDLA)
            {
                if(tsrRTHandles.netBuilder->canRunOnDLA(layer) 
                        && (dT == nvinfer1::DataType::kHALF || dT == nvinfer1::DataType::kINT8))
                {
                    layer->setPrecision(dT);
                    std::cout << "Layer: " << layer->getName() 
                                << " will run on DLA (batch size max: " 
                                <<  netBuilder->getMaxDLABatchSize()
                                << std::endl;
                    netBuilder->setDeviceType(layer, nvinfer1::DeviceType::kDLA);
                }
                else
                    throw std::runtime_error("Cannot use DLA for layer " + outName);
            }
#endif
           output_tensor.push_back(layer->getOutput(0));
           output_tensor.back()->setName(outName.c_str());
           output_tensor.back()->setType(dT);

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
            return(add_activation(net,
                                  netBuilder,
                                  useDLA,
                                  dT,
                                  layerName,
                                  activation.type,
                                  output_tensor));
        else
            return output_tensor;

}
std::vector<nvinfer1::ITensor *>
        add_HWC2CHW(nvinfer1::INetworkDefinition* net,
                    nvinfer1::IBuilder* netBuilder,
                    nvinfer1::DataType dT,
                    std::string layerName,
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
        auto layer = net->addShuffle(*inputs_tensor[i]);
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
            add_reshape(nvinfer1::INetworkDefinition* net,
                        nvinfer1::IBuilder* netBuilder,
                        nvinfer1::DataType dT,
                        std::string layerName,
                        unsigned int groupSize,
                        bool restoreShape,
                        std::vector<nvinfer1::ITensor *> inputs_tensor)
{
    std::vector<nvinfer1::ITensor *> output_tensor;
    std::cout << "Add Reshapelayer: " << layerName << std::endl;


    for(unsigned int i = 0; i < inputs_tensor.size(); ++i)
    {

        nvinfer1::Dims tensor_dims = inputs_tensor[0]->getDimensions();

        //const unsigned int residualBatch = tensor_dims.nbDims > 3 ?
        //                                    tensor_dims.d[0]
        //                                    : 1;

        std::string outName = layerName + "_" + std::to_string(i);
        auto layer = net->addShuffle(*inputs_tensor[i]);
        assert(layer != nullptr);
        if(tensor_dims.nbDims == 3)
        {
            nvinfer1::DimsCHW reshape_dims;
            const unsigned int nbOutputs = tensor_dims.d[0];
            const unsigned int dimY = tensor_dims.d[1];
            const unsigned int dimX = tensor_dims.d[2];
            std::cout << "groupSize: " << groupSize
                        << " nbOutputs: " << nbOutputs
                        << std::endl;

            if(!(groupSize % nbOutputs))
                throw std::runtime_error(
                    "add_reshape(): groupsize must be divisible by nbOutputs");

            reshape_dims.d[0] = groupSize;
            reshape_dims.d[1] = dimY * (nbOutputs / groupSize);
            reshape_dims.d[2] = dimX;

            layer->setReshapeDimensions(reshape_dims);

        }
        else if(tensor_dims.nbDims == 4)
        {
            nvinfer1::DimsNCHW reshape_dims;
            const unsigned int batch = tensor_dims.d[0];
            const unsigned int nbOutputs = tensor_dims.d[1];
            const unsigned int dimY = tensor_dims.d[2];
            const unsigned int dimX = tensor_dims.d[3];
            std::cout << "groupSize: " << groupSize
                        << " nbOutputs: " << nbOutputs
                        << std::endl;

            if(!(groupSize % nbOutputs))
                throw std::runtime_error(
                    "add_reshape(): groupsize must be divisible by nbOutputs");
            if( (dimY > 1 || dimX > 1) && !restoreShape)
                throw std::runtime_error(
                    "add_reshape(): can only be applied on 1 dimension tensor");

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
        add_softmax(nvinfer1::INetworkDefinition* net,
                        nvinfer1::IBuilder* netBuilder,
                        nvinfer1::DataType dT,
                        std::string layerName,
                        std::vector<nvinfer1::ITensor *> inputs_tensor)
{
    std::vector<nvinfer1::ITensor *> output_tensor;
    std::cout << "Add softmax layer: " << layerName << std::endl;

    for(unsigned int i = 0; i < inputs_tensor.size(); ++i)
    {
        std::string outName = layerName + "_" + std::to_string(i);
        auto layer = net->addSoftMax(*inputs_tensor[i]);

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
        add_anchors(nvinfer1::INetworkDefinition* net,
                        PluginFactory& factory,
                        nvinfer1::DataType dT,
                        std::string layerName,
                        unsigned int batchSize,
                        unsigned int nbOutputs,
                        unsigned int outputHeight,
                        unsigned int outputWidth,
                        std::vector<nvinfer1::ITensor *> inputs_tensor,
                        unsigned int stimuliHeight,
                        unsigned int stimuliWidth,
                        unsigned int featureMapWidth,
                        unsigned int featureMapHeight,
                        unsigned int scoreCls,
                        bool isFlip,
                        unsigned int nbAnchors,
                        const WDATA_T* anchor)
{

        std::vector<nvinfer1::ITensor *> output_tensor;
        std::cout << "Add anchors layer: " << layerName << std::endl;

        for(unsigned int i = 0; i < inputs_tensor.size(); ++i)
        {
            std::string outName = layerName + "_" + std::to_string(i);
            nvinfer1::IPlugin* pluginAnc = factory.createPlugin(outName.c_str(),
                                                                batchSize,
                                                                nbOutputs,
                                                                outputHeight,
                                                                outputWidth,
                                                                stimuliHeight,
                                                                stimuliWidth,
                                                                featureMapWidth,
                                                                featureMapHeight,
                                                                scoreCls,
                                                                isFlip,
                                                                nbAnchors,
                                                                anchor);

            auto layer = net->addPlugin(&inputs_tensor[i],
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
        add_resize(nvinfer1::INetworkDefinition* net,
                    PluginFactory& factory,
                    nvinfer1::DataType dT,
                    std::string layerName,
                    unsigned int batchSize,
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
        nvinfer1::IPlugin* pluginResize = factory.createPlugin(outName.c_str(),
                                                                batchSize,
                                                                nbOutputs,
                                                                outputHeight,
                                                                outputWidth,
                                                                featureHeight,
                                                                featureWidth,
                                                                resizeType,
                                                                alignCorner);

        auto layer = net->addPlugin(&inputs_tensor[i],
                                    1,
                                    *pluginResize);
        layer->setName(outName.c_str());
        output_tensor.push_back(layer->getOutput(0));
        output_tensor.back()->setType(dT);
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
        auto layer = net->addResize(*inputs_tensor[i]);
        layer->setOutputDimensions(outputsDims);
        layer->setAlignCorners(alignCorner);
        const nvinfer1::ResizeMode mode = (resizeType == BilinearTF) 
                                    ? nvinfer1::ResizeMode::kLINEAR
                                    : nvinfer1::ResizeMode::kNEAREST;
        layer->setResizeMode(mode);
        layer->setName(outName.c_str());
        output_tensor.push_back(layer->getOutput(0));
        output_tensor.back()->setType(dT);
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
        add_objectdetect(nvinfer1::INetworkDefinition* net,
                            PluginFactory& factory,
                            nvinfer1::DataType dT,
                            std::string layerName,
                            unsigned int batchSize,
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
                            double nmsIoU,
                            const float* scoreThreshold,
                            unsigned int maxParts,
                            unsigned int maxTemplates,
                            const unsigned int* numPartsPerClass,
                            const unsigned int* numTemplatesPerClass,
                            const WDATA_T* anchor)
{
    std::vector<nvinfer1::ITensor *> output_tensor;
    std::cout << "Add object detect layer: " << layerName << std::endl;

    for(unsigned int i = 0; i < inputs_tensor[0]->size(); ++i)
    {
            std::vector<nvinfer1::ITensor *> concat_tensor;

            for(unsigned int k = 0; k < inputs_tensor.size(); ++k)
            {
              nvinfer1::ITensor * input_tensor = (inputs_tensor[k])->data()[i];
              concat_tensor.push_back(input_tensor);
            }

            std::string outName = layerName + "_" + std::to_string(i);
            nvinfer1::IPlugin* pluginObjDet = factory.createPlugin(outName.c_str(),
                                                                batchSize,
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
                                                                nmsIoU,
                                                                scoreThreshold,
                                                                maxParts,
                                                                maxTemplates,
                                                                numPartsPerClass,
                                                                numTemplatesPerClass,
                                                                anchor);

        auto layer = net->addPlugin(&concat_tensor[0],
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
        output_tensor.back()->setType(dT);
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
        add_proposals(  nvinfer1::INetworkDefinition* net,
                        PluginFactory& factory,
                        nvinfer1::DataType dT,
                        std::string layerName,
                        unsigned int batchSize,
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
                        const WDATA_T* means,
                        const WDATA_T* std)
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
        nvinfer1::IPlugin* pluginProposals = factory.createPlugin(outName.c_str(),
                                                            batchSize,
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


        auto layer = net->addPlugin(&concat_tensor[0],
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
        add_regionproposal(nvinfer1::INetworkDefinition* net,
                            PluginFactory& factory,
                            nvinfer1::DataType dT,
                            std::string layerName,
                            unsigned int batchSize,
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
            nvinfer1::IPlugin* pluginRP = factory.createPlugin(outName.c_str(),
                                                                batchSize,
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


            auto layer = net->addPlugin(&inputs_tensor[i],
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
        add_ROIpooling(nvinfer1::INetworkDefinition* net,
                            PluginFactory& factory,
                            nvinfer1::DataType dT,
                            std::string layerName,
                            unsigned int batchSize,
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
                                factory.createPlugin(outName.c_str(),
                                                        batchSize,
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


            auto layer = net->addPlugin(&concat_tensor[0],
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

void add_weighted(  unsigned int batchSize,
                    unsigned int nbOutputs,
                    unsigned int outputsHeight,
                    unsigned int outputsWidth,
                    DATA_T* estimated_labels,
                    unsigned int nbChannels,
                    unsigned int image_height,
                    unsigned int image_width,
                    DATA_T* input_image,
                    unsigned char* overlay_data,
                    unsigned char* workspace,
                    float alpha,
                    cudaStream_t stream)
{
/*
    unsigned char* gpuWorkspace;
    const size_t outputWorkspace = image_height * image_width * nbChannels * batchSize * sizeof(unsigned char);
    CHECK_CUDA_STATUS(cudaMalloc((void **)&gpuWorkspace, outputWorkspace));
*/
    const unsigned int groupSize = std::min(32, (int)(image_height * image_width));
    const unsigned int blockSize = std::ceil((int)image_height * image_width / groupSize);

    const dim3 threadsPerBlocks = {groupSize, 1, 1};
    const dim3 blocksPerGrid = {blockSize, 1, batchSize};

    //Use INTERNEAREST resize factor if output image and input image dont have the same size
    const float multy = ((float) outputsHeight)/((float) image_height);
    const float multx = ((float) outputsWidth)/((float) image_width);

    cuda_add_weighted( batchSize,
                       nbOutputs,
                       outputsHeight,
                       outputsWidth,
                       estimated_labels,
                       nbChannels,
                       image_height,
                       image_width,
                       input_image,
                       workspace,
                       alpha,
                       threadsPerBlocks,
                       blocksPerGrid,
                       stream);

    CHECK_CUDA_STATUS(cudaMemcpy(overlay_data,
                                 workspace,
                                 nbChannels*image_height*image_width*batchSize,
                                 cudaMemcpyDeviceToHost));

}


/****Targets Layers ****/
void output_generation(unsigned int batchSize,
                       unsigned int nbOutputs,
                       void* dataIn,
                       uint32_t* outputEstimated,
                       cudaStream_t stream)
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

    CHECK_CUDA_STATUS(cudaStreamSynchronize(stream));

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
                               void* dataIn,
                               uint32_t* outputEstimated,
                               cudaStream_t stream,
                               float threshold,
                               bool useGPU)
{


    if(useGPU)
    {
        uint32_t *gpuEstimated;
        const size_t outputWorkspace = outputsHeight * outputsWidth * batchSize * sizeof(uint32_t);
        CHECK_CUDA_STATUS(cudaMalloc((void **)&gpuEstimated, outputWorkspace));

        const unsigned int groupSize = std::min(32, (int)(outputsHeight * outputsWidth));
        const unsigned int blockSize = std::ceil((int)outputsHeight * outputsWidth / groupSize);

        const dim3 threadsPerBlocks = {groupSize, 1, 1};
        const dim3 blocksPerGrid = {blockSize, 1, batchSize};

        cuda_spatial_outputs(nbOutputs,
                             outputsHeight,
                             outputsWidth,
                             batchSize,
                             threshold,
                             reinterpret_cast<DATA_T *>(dataIn),
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
        DATA_T* outputsData(NULL);
        if (outputsData == NULL) {
            outputsData = new DATA_T[batchSize * size];

            if (!outputsData)
                throw std::runtime_error(
                    "spatial_output_generation(): could not allocate memory");
        }
        CHECK_CUDA_STATUS(cudaMemcpy(outputsData,
                                    dataIn,
                                    batchSize * nbOutputs * outputsWidth * outputsHeight * sizeof(DATA_T),
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

        delete[] outputsData;
    }
}

void get_output(unsigned int batchSize,
                unsigned int nbOutputs,
                unsigned int outputsHeight,
                unsigned int outputsWidth,
                void* dataIn,
                DATA_T* outputEstimated)
{
    const unsigned int size = nbOutputs * outputsHeight * outputsWidth;

    if (outputEstimated == NULL) {
        outputEstimated = new DATA_T[batchSize * size];

        if (!outputEstimated)
            throw std::runtime_error(
                "get_output(): could not allocate memory");
    }

    CHECK_CUDA_STATUS(cudaMemcpy(outputEstimated,
                                 dataIn,
                                 batchSize * nbOutputs * outputsWidth * outputsHeight * sizeof(DATA_T),
                                 cudaMemcpyDeviceToHost));

}

void report_per_layer_profiling(unsigned int nbIter)
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