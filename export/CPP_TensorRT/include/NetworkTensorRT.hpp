/*
    (C) Copyright 2020 CEA LIST. All Rights Reserved.
    Contributor(s): David BRIAND (david.briand@cea.fr)

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

#ifndef NETWORKTENSORRT_HPP
#define NETWORKTENSORRT_HPP

#include "dnn_utils.hpp"
#include "kernels_cpu.hpp"
#include "kernels_gpu.hpp"
#include "BatchStream.hpp"
#include "fp16.h"

#include "../dnn/include/env.hpp"

#if NV_TENSORRT_MAJOR > 2
#include "IInt8EntropyCalibrator.hpp"
#endif

#ifdef WRAPPER_PYTHON

#ifndef BOOST_NATIVE
#include <boost/python/numpy.hpp>
namespace np = boost::python::numpy;
#else
#include <boost/numpy.hpp>
namespace np = boost::numpy;
#endif

#include <boost/python.hpp>
#include <boost/scoped_array.hpp>
namespace p = boost::python;

#endif


///TENSOR_RT Specific class override
static struct Profiler : public nvinfer1::IProfiler
{
    typedef std::pair<std::string, float> Record;
    std::vector<Record> mProfile;

    virtual void reportLayerTime(const char* layerName, float ms)
    {
        auto record = std::find_if(mProfile.begin(), mProfile.end(), [&](const Record& r){ return r.first == layerName; });
        if (record == mProfile.end())
            mProfile.push_back(std::make_pair(layerName, ms));
        else
            record->second += ms;
    }

} gProfiler;

static class Logger : public nvinfer1::ILogger
{
    void log(Severity severity, const char* msg) override
    {
        std::cout << msg << std::endl;
    }
} gLogger;



namespace N2D2 {

class Network {
public:
    Network();

    void initialize();
    void networkDefinition();

    template<typename Input_T>
    void asyncExe(Input_T* in_data, unsigned int batchSize);
#ifdef WRAPPER_PYTHON
    void asyncExePy(np::ndarray const & in_data, unsigned int batchSize);
#endif

    template<typename Input_T>
    void syncExe(Input_T* in_data, unsigned int batchSize);
#ifdef WRAPPER_PYTHON
    void syncExePy(np::ndarray const & in_data, unsigned int batchSize);
#endif

    template<typename Input_T>
    void syncExeGPU(Input_T** externalInOut, unsigned int batchSize);

    void log_output(float* out_data, unsigned int target);
#ifdef WRAPPER_PYTHON
    void cpyOutputPy(np::ndarray const & in_data, unsigned int target);
#endif

    void* getDevicePtr(unsigned int target);

    void estimated(uint32_t* out_data, unsigned int target, bool useGPU = false, float threshold = 0.0f);
#ifdef WRAPPER_PYTHON
    void estimatedPy(np::ndarray const & in_data, unsigned int target, bool useGPU, float threshold);
#endif

    void addOverlay(unsigned char* overlay_data, unsigned int target, float alpha);
#ifdef WRAPPER_PYTHON
    void addOverlayPy(np::ndarray const & in_data, unsigned int target, float alpha);
#endif


    //void output(uint32_t* out_data, unsigned int target);

    //void getOutput(uint32_t* out_data, unsigned int batchSize, unsigned int target);

    //void get2DOutput(   uint32_t* out_data, 
    //                    unsigned int batchSize, 
    //                    unsigned int target, 
    //                    float threshold);


    void output_generation(unsigned int batchSize,
                        unsigned int nbOutputs,
                        void* dataIn,
                        uint32_t* outputEstimated,
                        cudaStream_t stream);

    void spatial_output_generation(unsigned int batchSize,
                                unsigned int nbOutputs,
                                unsigned int outputsHeight,
                                unsigned int outputsWidth,
                                void* dataIn,
                                uint32_t* outputEstimated,
                                cudaStream_t stream,
                                float threshold=0.0,
                                bool useGPU=false);

    void get_output(unsigned int nbOutputs,
                    unsigned int outputsHeight,
                    unsigned int outputsWidth,
                    void* dataIn,
                    float* outputEstimated);

    void report_per_layer_profiling(unsigned int nbIter);

    void add_weighted(  unsigned int nbOutputs,
                        unsigned int outputsHeight,
                        unsigned int outputsWidth,
                        float* estimated_labels,
                        unsigned int nbChannels,
                        unsigned int image_height,
                        unsigned int image_width,
                        float* input_image,
                        unsigned char* overlay_data,
                        float alpha,
                        cudaStream_t stream);


    void dumpMem(int size, float* data, std::string fileName);

    void setProfiling();
    void reportProfiling(unsigned int nbIter);
    unsigned int getOutputNbTargets(){ return NETWORK_TARGETS; };
    unsigned int getOutputTarget(unsigned int target){ return NB_TARGET[target]; };
    unsigned int getOutputDimZ(unsigned int target){ return NB_OUTPUTS[target]; };
    unsigned int getOutputDimY(unsigned int target){ return OUTPUTS_HEIGHT[target]; };
    unsigned int getOutputDimX(unsigned int target){ return OUTPUTS_WIDTH[target]; };
    unsigned int getInputDimZ(){ return ENV_NB_OUTPUTS; };
    unsigned int getInputDimY(){ return ENV_SIZE_Y; };
    unsigned int getInputDimX(){ return ENV_SIZE_X; };

    void setPrecision(int nbBits) { mNbBits = nbBits ; };

    void setMaxBatchSize(std::size_t batchSize) {
        mMaxBatchSize = batchSize;
    };
    void setDeviceID(std::size_t devID) {
        mDeviceID = devID;
    };
    void setIterBuild(std::size_t iterBuild) {
        mIterBuild = iterBuild;
    };
    void setInputEngine(std::string path) {
        mInputEngine = path;
    };
    void setOutputEngine(std::string path) {
        mOutputEngine = path;
    };
    void setCalibCache(std::string path) {
        mCalibrationCacheName = path;
    };
    void setCalibFolder(std::string path) {
        mCalibrationFolder = path;
    };
    void setParamPath(std::string path) {
        mParametersPath = path;
    };

    std::size_t mMaxBatchSize = 1;
    std::size_t mDeviceID = 0;
    std::size_t mIterBuild = 1;
    int mNbBits = -32;

    std::string mInputEngine = "";
    std::string mOutputEngine = "";
    std::string mCalibrationCacheName = "";
    std::string mCalibrationFolder = "";
    std::string mParametersPath = "dnn/";

    /// Destructor
    ~Network() { /*free_memory();*/ };

    protected :

#if NV_TENSORRT_MAJOR > 2
    std::shared_ptr<nvinfer1::IInt8Calibrator> mCalibrator;
#endif
    cudaStream_t mDataStream;
    std::vector<void* > mInOutBuffer;
    unsigned char* mWorkspaceGPU;
    bool mUseDLA = false;
    nvinfer1::ICudaEngine* mCudaEngine;
    nvinfer1::IExecutionContext* mContext;
    nvinfer1::IBuilder* mNetBuilder;
    std::vector<nvinfer1::INetworkDefinition*> mNetDef;
    nvinfer1::DataType mDataType = nvinfer1::DataType::kFLOAT;

    void createContext();
    void setIOMemory();
    void setTensorRTPrecision();

    void add_target(std::vector<nvinfer1::ITensor *> outputs_tensor,
                    unsigned int targetIdx);

    std::vector<nvinfer1::ITensor *>
            add_activation( std::string layerName,
                            nvinfer1::ActivationType activation,
                            std::vector<nvinfer1::ITensor *> inputs_tensor);

    std::vector<nvinfer1::ITensor *>
            add_convolution(std::string layerName,
                            LayerActivation activation,
                            unsigned int nbOutputs,
                            unsigned int strideX,
                            unsigned int strideY,
                            unsigned int paddingX,
                            unsigned int paddingY,
                            unsigned int kernelW,
                            unsigned int kernelH,
                            std::vector<nvinfer1::ITensor *> inputs_tensor,
                            std::string wFile,
                            unsigned int weights_size,
                            std::string bFile,
                            unsigned int bias_size);

    std::vector<nvinfer1::ITensor *>
            add_deconvolution(std::string layerName,
                            LayerActivation activation,
                            unsigned int nbOutputs,
                            unsigned int strideX,
                            unsigned int strideY,
                            unsigned int paddingX,
                            unsigned int paddingY,
                            unsigned int kernelW,
                            unsigned int kernelH,
                            std::vector<nvinfer1::ITensor *> inputs_tensor,
                            std::string wFile,
                            unsigned int weights_size,
                            std::string bFile,
                            unsigned int bias_size);

    std::vector<nvinfer1::ITensor *>
            add_fc(std::string layerName,
                    LayerActivation activation,
                    unsigned int nbOutputs,
                    std::vector<nvinfer1::ITensor *> inputs_tensor,
                    std::string wFile,
                    unsigned int weights_size,
                    std::string bFile);

    std::vector<nvinfer1::ITensor *>
            add_HWC2CHW(std::string layerName,
                        std::vector<nvinfer1::ITensor *> inputs_tensor);

    std::vector<nvinfer1::ITensor *>
            add_concat(std::string layerName,
                    unsigned int nbInputs,
                    std::vector<std::vector<nvinfer1::ITensor*>*> inputs_tensor);

    std::vector<nvinfer1::ITensor *>
            add_batchnorm(std::string layerName,
                            LayerActivation activation,
                            unsigned int nbOutputs,
                            unsigned int outputHeight,
                            unsigned int outputWidth,
                            std::vector<nvinfer1::ITensor *> inputs_tensor,
                            float* scales,
                            float* biases,
                            float* means,
                            float* variances,
                            float epsilon);

    std::vector<nvinfer1::ITensor *>
            add_scale(std::string layerName,
                            LayerActivation activation,
                            unsigned int nbOutputs,
                            unsigned int outputHeight,
                            unsigned int outputWidth,
                            std::vector<nvinfer1::ITensor *> inputs_tensor,
                            float* scales,
                            float* shift,
                            float* power);

    std::vector<nvinfer1::ITensor *>
            add_elementwise(std::string layerName,
                            LayerActivation activation,
                            unsigned int nbOutputs,
                            unsigned int outputHeight,
                            unsigned int outputWidth,
                            std::vector<std::vector<nvinfer1::ITensor *> > inputs_tensor,
                            nvinfer1::ElementWiseOperation op,
                            float* scales,
                            float* shift,
                            float* power);

    std::vector<nvinfer1::ITensor *>
            add_pooling(std::string layerName,
                            LayerActivation activation,
                            unsigned int poolH,
                            unsigned int poolW,
                            unsigned int strideX,
                            unsigned int strideY,
                            unsigned int paddingX,
                            unsigned int paddingY,
                            std::vector<nvinfer1::ITensor *> inputs_tensor,
                            nvinfer1::PoolingType poolType);

    std::vector<nvinfer1::ITensor *>
            add_padding(std::string layerName,
                        unsigned int nbOutputs,
                        std::vector<nvinfer1::ITensor *> inputs_tensor,
                        const int pad_top,
                        const int pad_bottom,
                        const int pad_left,
                        const int pad_right);

    std::vector<nvinfer1::ITensor *>
                add_lrn(std::string layerName,
                            unsigned int nbOutputs,
                            std::vector<nvinfer1::ITensor *> inputs_tensor,
                            const int windows,
                            const float alpha,
                            const float beta,
                            const float k);

    std::vector<nvinfer1::ITensor *>
                add_reshape(std::string layerName,
                            unsigned int groupSize,
                            bool restoreShape,
                            std::vector<nvinfer1::ITensor *> inputs_tensor);
    std::vector<nvinfer1::ITensor *>
            add_softmax(std::string layerName,
                            std::vector<nvinfer1::ITensor *> inputs_tensor);


    std::vector<nvinfer1::ITensor *>
            add_anchors(std::string layerName,
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
                            const float* anchor);

    std::vector<nvinfer1::ITensor *>
            add_regionproposal(std::string layerName,
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
                                unsigned int iouIndex);

    std::vector<nvinfer1::ITensor *>
            add_objectdetect(std::string layerName,
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
                                const float* anchor);


    std::vector<nvinfer1::ITensor *>
            add_proposals(std::string layerName,
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
                            const float* std);

    std::vector<nvinfer1::ITensor *>
            add_ROIpooling(std::string layerName,
                                unsigned int nbOutputs,
                                unsigned int outputHeight,
                                unsigned int outputWidth,
                                std::vector<std::vector<nvinfer1::ITensor*>*> inputs_tensor,
                                unsigned int stimuliHeight,
                                unsigned int stimuliWidth,
                                unsigned int nbFeature,
                                unsigned int* featureChannels,
                                unsigned int* featureHeight,
                                unsigned int* featureWidth,
                                Pooling_T poolType,
                                unsigned int nbProposals,
                                bool ignorePadding,
                                bool isFlip);

    std::vector<nvinfer1::ITensor *>
            add_resize(std::string layerName,
                        unsigned int nbOutputs,
                        unsigned int outputHeight,
                        unsigned int outputWidth,
                        std::vector<nvinfer1::ITensor *> inputs_tensor,
                        unsigned int featureHeight,
                        unsigned int featureWidth,
                        Pooling_T resizeType,
                        bool alignCorner);
};
}



template<typename Input_T>
void N2D2::Network::asyncExe(Input_T* in_data, unsigned int batchSize) {

   CHECK_CUDA_STATUS(cudaMemcpyAsync(mInOutBuffer[0],
                                    in_data,
                                    batchSize*ENV_BUFFER_SIZE*sizeof(Input_T),
                                    cudaMemcpyHostToDevice,
                                    mDataStream));

   mContext->enqueue(batchSize, reinterpret_cast<void**>(mInOutBuffer.data()), mDataStream, nullptr);
}

template<typename Input_T>
void N2D2::Network::syncExe(Input_T* in_data, unsigned int batchSize) {

   CHECK_CUDA_STATUS(cudaMemcpy(mInOutBuffer[0],
                                in_data,
                                batchSize*ENV_BUFFER_SIZE*sizeof(Input_T),
                                cudaMemcpyHostToDevice));

   mContext->execute(batchSize, reinterpret_cast<void**>(mInOutBuffer.data()));
}

template<typename Input_T>
void N2D2::Network::syncExeGPU(Input_T** externalInOut, unsigned int batchSize) {
   mContext->execute(batchSize, reinterpret_cast<void**>(externalInOut));
}

#endif // NETWORKTENSORRT_HPP
