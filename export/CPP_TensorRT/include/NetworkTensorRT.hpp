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
#if NV_TENSORRT_MAJOR > 7
    noexcept
#endif
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
    void log(Severity severity, const char* msg) 
#if NV_TENSORRT_MAJOR > 7
    noexcept
#endif
    override
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
                        float alpha);


    void dumpMem(int size, float* data, std::string fileName);

    void setProfiling();
    void reportProfiling(unsigned int nbIter);
    void setInputDims(unsigned int dimX, unsigned int dimY, unsigned int dimZ) 
    { mInputDimensions.d[0] = dimZ;
      mInputDimensions.d[1] = dimY;
      mInputDimensions.d[2] = dimX; };

    void setOutputNbTargets(unsigned int nbTargets) { mTargetsDimensions.resize(nbTargets); };

    void setOutputTarget(unsigned int nbTarget, 
                        unsigned int dimZ, 
                        unsigned int dimY, 
                        unsigned int dimX, 
                        unsigned int t) {
        mTargetsDimensions[t].d[0] = nbTarget;
        mTargetsDimensions[t].d[1] = dimZ;
        mTargetsDimensions[t].d[2] = dimY;
        mTargetsDimensions[t].d[3] = dimX;
    };

    unsigned int getOutputNbTargets(){ return mTargetsDimensions.size(); };
    unsigned int getOutputTarget(unsigned int target){ return mTargetsDimensions[target].d[0]; };
    unsigned int getOutputDimZ(unsigned int target){ return mTargetsDimensions[target].d[1]; };
    unsigned int getOutputDimY(unsigned int target){ return mTargetsDimensions[target].d[2]; };
    unsigned int getOutputDimX(unsigned int target){ return mTargetsDimensions[target].d[3]; };
    unsigned int getInputDimZ(){ return mInputDimensions.d[0]; };
    unsigned int getInputDimY(){ return mInputDimensions.d[1]; };
    unsigned int getInputDimX(){ return mInputDimensions.d[2]; };

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
#ifdef ONNX
    void setONNXModel(std::string path) {
        mONNXmodel = path;
    };
#endif
    void setDetectorThresholds(float* thresholds, unsigned int nbClass) {
        if(mDetectorThresholds == NULL) {
            mDetectorThresholds = new float[nbClass];

            if (!mDetectorThresholds)
                throw std::runtime_error(
                    "setDetectorThresholds(): could not allocate memory");
        }
        std::cout << "Set Object detector Thresholds: " << std::endl;
        for(size_t i = 0; i < nbClass; ++i) {
            std::cout << "[" << i << "]=" <<  thresholds[i] << " ";
            mDetectorThresholds[i] = thresholds[i];
        }
        std::cout << "\n" << std::endl;
    };
#ifdef WRAPPER_PYTHON
    void setDetectorThresholdsPy(np::ndarray const & thresholds, unsigned int nbClass)
    {
        setDetectorThresholds(reinterpret_cast<float*>(thresholds.get_data()), nbClass);
    };
#endif
    void setDetectorNMS(double nmsIoU) {
        std::cout << "Set Object detector NMS IoU: " 
                << nmsIoU << std::endl;
        mDetectorNMS = nmsIoU;
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
#ifdef ONNX
    std::string mONNXmodel = "";
#endif
    /// Destructor
    ~Network() { /*free_memory();*/ };
    trt_Dims3 mInputDimensions;
    std::vector<trt_Dims4> mTargetsDimensions;

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
#if (NV_TENSORRT_MAJOR + NV_TENSORRT_MINOR) > 6
//  Builder Config have been introduce since TensorRT 7.1.0 EA :
// https://docs.nvidia.com/deeplearning/tensorrt/release-notes/tensorrt-7.html#rel_7-1-0-EA
    nvinfer1::IBuilderConfig* mNetBuilderConfig;
#endif
    std::vector<nvinfer1::INetworkDefinition*> mNetDef;
    nvinfer1::DataType mDataType = nvinfer1::DataType::kFLOAT;
    float* mDetectorThresholds = NULL;
    double mDetectorNMS = -1.0;

    void createContext();
    void setIOMemory();
    void setTensorRTPrecision();
#ifndef ONNX
    void setInternalDimensions();
#endif

    void add_target(std::vector<nvinfer1::ITensor *> outputs_tensor,
                    unsigned int targetIdx);

    std::vector<nvinfer1::ITensor *>
            add_activation( std::string layerName,
                            nvinfer1::ActivationType activation,
                            double alpha,
                            double beta,
                            std::vector<nvinfer1::ITensor *> inputs_tensor);
    std::vector<nvinfer1::ITensor *>
        add_activation_cell(std::string layerName,
                            LayerActivation activation,
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
                            CoeffMode_T coeffMode,
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
                            unsigned int nbDims,
                            const int shape[],
                            std::vector<nvinfer1::ITensor *> inputs_tensor);

    std::vector<nvinfer1::ITensor *>
                add_transpose(std::string layerName,
                            unsigned int nbDims,
                            const int perm[],
                            std::vector<nvinfer1::ITensor *> inputs_tensor);

    std::vector<nvinfer1::ITensor *>
                add_group_reshape(std::string layerName,
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
                            bool isCoordinatesAnchors,
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
                                bool isCoordinatesAnchors,
                                bool isPixelFormatXY,
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
                                    batchSize  *mInputDimensions.d[0] * mInputDimensions.d[1] * mInputDimensions.d[2] *sizeof(Input_T),
                                    cudaMemcpyHostToDevice,
                                    mDataStream));

   mContext->enqueue(batchSize, reinterpret_cast<void**>(mInOutBuffer.data()), mDataStream, nullptr);
}

template<typename Input_T>
void N2D2::Network::syncExe(Input_T* in_data, unsigned int batchSize) {

   CHECK_CUDA_STATUS(cudaMemcpy(mInOutBuffer[0],
                                in_data,
                                batchSize  *mInputDimensions.d[0] * mInputDimensions.d[1] * mInputDimensions.d[2] *sizeof(Input_T),
                                cudaMemcpyHostToDevice));

   mContext->execute(batchSize, reinterpret_cast<void**>(mInOutBuffer.data()));
}

template<typename Input_T>
void N2D2::Network::syncExeGPU(Input_T** externalInOut, unsigned int batchSize) {
   mContext->execute(batchSize, reinterpret_cast<void**>(externalInOut));
}

#endif // NETWORKTENSORRT_HPP
