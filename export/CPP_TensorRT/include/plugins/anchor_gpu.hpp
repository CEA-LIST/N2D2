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

#ifndef ANCHOR_GPU_HPP
#define ANCHOR_GPU_HPP

#include "dnn_utils.hpp"
#include "kernels_gpu.hpp"


/**Plugin Layer implementation**/
/**Anchoring GPU implementation**/
class AnchorGPUPlugin: public nvinfer1::IPluginV2Ext
{
public:
	AnchorGPUPlugin(unsigned int batchSize,
                    unsigned int nbOutputs,
                    unsigned int outputHeight,
                    unsigned int outputWidth,
                    unsigned int stimuliHeight,
                    unsigned int stimuliWidth,
                    unsigned int featureMapWidth,
                    unsigned int featureMapHeight,
                    unsigned int scoreCls,
                    bool isCoordinatesAnchors,
                    bool isFlip,
                    unsigned int nbAnchors,
                    const float*  anchors)
	{
        std::cout << "AnchorGPUPlugin()" << std::endl;
        mOutputDims.d[0] = batchSize;
        mOutputDims.d[1] = nbOutputs;
        mOutputDims.d[2] = outputHeight;
        mOutputDims.d[3] = outputWidth;

        mStimuliHeight = stimuliHeight;
        mStimuliWidth = stimuliWidth;
        mFeatureMapWidth = featureMapWidth;
        mFeatureMapHeight = featureMapHeight;

        mRatioX = std::ceil(featureMapWidth / (double) outputWidth);
        mRatioY = std::ceil(featureMapHeight / (double) outputHeight);
        mScoreCls = scoreCls;
        mIsCoordinateAnchors = isCoordinatesAnchors;
        mIsFlip = isFlip;
        mNbAnchors = nbAnchors;
        mAnchors_HOST.resize(4 * mNbAnchors);
        for(unsigned int i = 0; i < 4 * mNbAnchors; ++i) {
            mAnchors_HOST[i] = anchors[i];

        }

        float* mAnchorsPrecompute = new float [outputWidth*outputHeight*mNbAnchors*4];

        for(size_t k = 0; k < mNbAnchors; ++k) {
          for(size_t ya = 0; ya < outputHeight; ++ya) {
            for(size_t xa = 0; xa < outputWidth; ++xa) {

                const size_t anchorsIdx = k*4;
                const size_t anchorsPrecomputeIdx = xa*4 + ya*outputWidth*4 + k*outputWidth*outputHeight*4;
                const float xa0 = (mAnchors_HOST[anchorsIdx] + xa * mRatioX) 
                                    / (float)(mFeatureMapWidth - 1.0);
                const float ya0 = (mAnchors_HOST[anchorsIdx + 1] + ya * mRatioY) 
                                    / (float)(mFeatureMapHeight - 1.0);
                const float xa1 = (mAnchors_HOST[anchorsIdx + 2] + xa * mRatioX) 
                                    / (float)(mFeatureMapWidth - 1.0);
                const float ya1 = (mAnchors_HOST[anchorsIdx + 3] + ya * mRatioY) 
                                    / (float)(mFeatureMapHeight - 1.0);

                // Anchors width and height
                const float wa = xa1 - xa0;
                const float ha = ya1 - ya0;
                // Anchor center coordinates (xac, yac)
                const float xac = xa0 + wa * 0.5;
                const float yac = ya0 + ha * 0.5;
                mAnchorsPrecompute[0 + anchorsPrecomputeIdx] = xac;
                mAnchorsPrecompute[1 + anchorsPrecomputeIdx] = yac;
                mAnchorsPrecompute[2 + anchorsPrecomputeIdx] = wa;
                mAnchorsPrecompute[3 + anchorsPrecomputeIdx] = ha;
            }
          }
        }
        checkCudaErrors( cudaMalloc((void**)&mAnchorsGPU,
                         4*mNbAnchors*outputWidth*outputHeight*sizeof(float)) );
        checkCudaErrors( cudaMemcpy(mAnchorsGPU,
                         mAnchorsPrecompute,
                         4*mNbAnchors*outputWidth*outputHeight*sizeof(float),
                         cudaMemcpyHostToDevice) );
        delete [] mAnchorsPrecompute;
        gpuThreadAllocation();
	}

	AnchorGPUPlugin(const void* dataFromRuntime, size_t length)
	{
		const char* d = reinterpret_cast<const char*>(dataFromRuntime), *a = d;
        mOutputDims = read<nvinfer1::Dims>(d);
        mStimuliHeight = (unsigned int) read<int>(d);
        mStimuliWidth = (unsigned int) read<int>(d);
        mFeatureMapWidth = (unsigned int) read<int>(d);
        mFeatureMapHeight = (unsigned int) read<int>(d);

        mScoreCls = (unsigned int) read<int>(d);
        mIsCoordinateAnchors = read<bool>(d);
        mIsFlip = read<bool>(d);
        mNbAnchors = (unsigned int) read<int>(d);
		mAnchorsGPU = deserializeToDevice(d, mNbAnchors*4*mOutputDims.d[2]*mOutputDims.d[3]);
        mAnchors_HOST.resize(4*mNbAnchors);
        for(unsigned int k = 0; k < mNbAnchors * 4; ++k)
            mAnchors_HOST[k] = read<float>(d);
        
        mThreadX = read<int>(d);
        mThreadY = read<int>(d);
        mThreadZ = read<int>(d);
        mBlockX = read<int>(d);
        mBlockY = read<int>(d);
        mBlockZ = read<int>(d);
		assert(d == a + length);
 	}

	~AnchorGPUPlugin()
	{
        checkCudaErrors(cudaFree(mAnchorsGPU));
	}

	virtual int getNbOutputs() const 
#if NV_TENSORRT_MAJOR > 7
    noexcept
#endif
    override
	{
        return 1;
	}

	virtual nvinfer1::Dims getOutputDimensions(int index,
                                               const nvinfer1::Dims* inputDim,
                                               int nbInputDims)
#if NV_TENSORRT_MAJOR > 7
    noexcept
#endif
    override
	{
        unsigned int batchInput = 1;

        if(inputDim[0].nbDims == 4)
            batchInput = inputDim[0].d[0];

        return trt_Dims3(mOutputDims.d[1], mOutputDims.d[2], mOutputDims.d[3]);
	}

	virtual int initialize() 
#if NV_TENSORRT_MAJOR > 7
    noexcept
#endif
    override
	{
		return 0;
	}

	virtual void terminate() 
#if NV_TENSORRT_MAJOR > 7
    noexcept
#endif
    override
	{

	}

	virtual size_t getWorkspaceSize(int maxBatchSize) const
#if NV_TENSORRT_MAJOR > 7
    noexcept
#endif
    override
	{
		return 0;
	}

#if NV_TENSORRT_MAJOR > 7
	virtual int enqueue(int batchSize,
                        const void*const * inputs,
                        void*const* outputs,
                        void* workspace,
                        cudaStream_t stream) noexcept
#else
	virtual int enqueue(int batchSize,
                        const void*const * inputs,
                        void** outputs,
                        void* workspace,
                        cudaStream_t stream)
#endif
    override
	{
        const dim3 threadGrid = {(unsigned int) mThreadX,
                                 (unsigned int) mThreadY,
                                 (unsigned int) mThreadZ};

        const dim3 blockGrid = {(unsigned int) mBlockX,
                                (unsigned int) mBlockY,
                                (unsigned int) mBlockZ};

        cuda_anchor_propagate(mOutputDims.d[0],
                                mOutputDims.d[1],
                                mOutputDims.d[2],
                                mOutputDims.d[3],
                                mStimuliHeight,
                                mStimuliWidth,
                                mScoreCls,
                                mIsCoordinateAnchors,
                                mIsFlip,
                                mNbAnchors,
                                reinterpret_cast<const float *>(mAnchorsGPU),
                                reinterpret_cast<const float *>(inputs[0]),
                                reinterpret_cast<float *>(outputs[0]),
                                threadGrid,
                                blockGrid,
                                stream);

       return 0;
	}

	virtual size_t getSerializationSize() const 
#if NV_TENSORRT_MAJOR > 7
    noexcept
#endif
    override
	{
        size_t inputDimParamSize = sizeof(nvinfer1::Dims); //mOutputsDims
        size_t stimuliParamSize = 6*sizeof(int); //Stimuliheight and StimuliWidth
        size_t anchorsSize = 4*mNbAnchors*mOutputDims.d[2]*mOutputDims.d[3]*sizeof(float) + sizeof(bool); // mNbAnchors and (x0 y0 x1 y1) * mNbAnchors + mScoreCls
        size_t paramSize = sizeof(bool);
        size_t threadSize = 3*2*sizeof(int);
        size_t anchorHostSize = 4*mNbAnchors*sizeof(float);
        size_t SerializationSize = inputDimParamSize + stimuliParamSize
                                + anchorsSize + threadSize + paramSize + anchorHostSize;

        return SerializationSize;
	}

	virtual void serialize(void* buffer) const 
#if NV_TENSORRT_MAJOR > 7
    noexcept
#endif
    override
	{
        char* d = reinterpret_cast<char*>(buffer);
        char* a = d;

        *reinterpret_cast<nvinfer1::Dims*>(d) = mOutputDims;
        d += sizeof(nvinfer1::Dims);

        *reinterpret_cast<int*>(d) = mStimuliHeight;
        d += sizeof(int);
        *reinterpret_cast<int*>(d) = mStimuliWidth;
        d += sizeof(int);
        *reinterpret_cast<int*>(d) = mFeatureMapWidth;
        d += sizeof(int);
        *reinterpret_cast<int*>(d) = mFeatureMapHeight;
        d += sizeof(int);
        *reinterpret_cast<int*>(d) = mScoreCls;
        d += sizeof(int);
        *reinterpret_cast<bool*>(d) = mIsCoordinateAnchors;
        d += sizeof(bool);
        *reinterpret_cast<bool*>(d) = mIsFlip;
        d += sizeof(bool);
        *reinterpret_cast<int*>(d) = mNbAnchors;
        d += sizeof(int);

        d += serializeFromDevice<float>(d, mAnchorsGPU, mNbAnchors*4*mOutputDims.d[2]*mOutputDims.d[3]);
        for(unsigned int k = 0; k < mNbAnchors*4 ; ++k) {
            *reinterpret_cast<float*>(d) = mAnchors_HOST[k];
            d += sizeof(float);
        }


        *reinterpret_cast<int*>(d) = mThreadX;
        d += sizeof(int);
        *reinterpret_cast<int*>(d) = mThreadY;
        d += sizeof(int);
        *reinterpret_cast<int*>(d) = mThreadZ;
        d += sizeof(int);
        *reinterpret_cast<int*>(d) = mBlockX;
        d += sizeof(int);
        *reinterpret_cast<int*>(d) = mBlockY;
        d += sizeof(int);
        *reinterpret_cast<int*>(d) = mBlockZ;
        d += sizeof(int);
        assert(d == a + getSerializationSize());
	}

    bool supportsFormat(nvinfer1::DataType type, nvinfer1::PluginFormat format) const 
#if NV_TENSORRT_MAJOR > 7
    noexcept
#endif
    override
    {
#if NV_TENSORRT_MAJOR > 5
        return (type == nvinfer1::DataType::kFLOAT && format == nvinfer1::PluginFormat::kLINEAR );
#else
        return (type == nvinfer1::DataType::kFLOAT && format == nvinfer1::PluginFormat::kNCHW );
#endif
    }

    const char* getPluginType() const 
#if NV_TENSORRT_MAJOR > 7
    noexcept
#endif
    override
     {
        const char* ANCHOR_GPU_PLUGIN_NAME{"AnchorGPUPlugin"};
        return ANCHOR_GPU_PLUGIN_NAME;
     }
    const char* getPluginVersion() const 
#if NV_TENSORRT_MAJOR > 7
    noexcept
#endif
    override
    {
        const char* ANCHOR_GPU_PLUGIN_VERSION{"1"};
        return ANCHOR_GPU_PLUGIN_VERSION;
     }
    void destroy() 
#if NV_TENSORRT_MAJOR > 7
    noexcept
#endif
    override {    delete this; }

    IPluginV2Ext* clone() const 
#if NV_TENSORRT_MAJOR > 7
    noexcept
#endif
    override
    {

        IPluginV2Ext* plugin = new AnchorGPUPlugin( mOutputDims.d[0],
                                                    mOutputDims.d[1],
                                                    mOutputDims.d[2],
                                                    mOutputDims.d[3],
                                                    mStimuliHeight,
                                                    mStimuliWidth,
                                                    mFeatureMapWidth,
                                                    mFeatureMapHeight,
                                                    mScoreCls,
                                                    mIsCoordinateAnchors,
                                                    mIsFlip,
                                                    mNbAnchors,
                                                    mAnchors_HOST.data());

        plugin->setPluginNamespace("");
        return plugin;
     }


    void setPluginNamespace(const char* pluginNamespace) 
#if NV_TENSORRT_MAJOR > 7
    noexcept
#endif
    override
     {
        std::string mNamespace = pluginNamespace;
     }
    const char* getPluginNamespace() const 
#if NV_TENSORRT_MAJOR > 7
    noexcept
#endif
    override
     { 
        return mNamespace.c_str();
    }

    nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const 
#if NV_TENSORRT_MAJOR > 7
    noexcept
#endif
    override
     {
        return nvinfer1::DataType::kFLOAT;
     }

    bool isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const 
#if NV_TENSORRT_MAJOR > 7
    noexcept
#endif
    override
     {    
        return false;
     }

    bool canBroadcastInputAcrossBatch(int inputIndex) const 
#if NV_TENSORRT_MAJOR > 7
    noexcept
#endif
    override
     {
        return false;
     }

    void attachToContext(
        cudnnContext* cudnnContext, cublasContext* cublasContext, nvinfer1::IGpuAllocator* gpuAllocator) 
#if NV_TENSORRT_MAJOR > 7
    noexcept
#endif
    override{ }

    void configurePlugin(const nvinfer1::Dims* inputDims, int nbInputs, const nvinfer1::Dims* outputDims, int nbOutputs,
        const nvinfer1::DataType* inputTypes, const nvinfer1::DataType* outputTypes, const bool* inputIsBroadcast,
        const bool* outputIsBroadcast, nvinfer1::PluginFormat floatFormat, int maxBatchSize) 
#if NV_TENSORRT_MAJOR > 7
    noexcept
#endif
    override { }

    void detachFromContext() 
#if NV_TENSORRT_MAJOR > 7
    noexcept
#endif
    override
     { }

private:
    template<typename T> void write(char*& buffer, const T& val)
    {
        *reinterpret_cast<T*>(buffer) = val;
        buffer += sizeof(T);
    }

    template<typename T> T read(const char*& buffer)
    {
        T val = *reinterpret_cast<const T*>(buffer);
        buffer += sizeof(T);
        return val;
    }

    float* deserializeToDevice(const char*& hostBuffer, size_t dataSize)
    {
        float* gpuData;
        checkCudaErrors(cudaMalloc(&gpuData, dataSize*sizeof(float)));
        checkCudaErrors(cudaMemcpy(gpuData, hostBuffer, dataSize*sizeof(float), cudaMemcpyHostToDevice));
        hostBuffer += dataSize*sizeof(float);
        return gpuData;
    }

    template<typename T>
    size_t serializeFromDevice(char*& hostBuffer, const T* deviceWeights, const size_t dataSize) const
    {
        checkCudaErrors(cudaMemcpy(hostBuffer, deviceWeights, dataSize*sizeof(T), cudaMemcpyDeviceToHost));
        return dataSize*sizeof(T);
    }
    void gpuThreadAllocation()
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, 0);

        const unsigned int batchSize = mOutputDims.d[0];
        const unsigned int nbOutputs = mOutputDims.d[1];
        const unsigned int outputsHeight = mOutputDims.d[2];
        const unsigned int outputsWidth = mOutputDims.d[3];

        const unsigned int maxSize
            = (unsigned int)deviceProp.maxThreadsPerBlock;
        const unsigned int prefMultiple = (unsigned int)deviceProp.warpSize;

        const unsigned int groupSize = (outputsWidth * outputsHeight < maxSize)
                                           ? outputsWidth * outputsHeight
                                           : maxSize;
        const unsigned int groupWidth
            = std::min(prefMultiple, nextDivisor(groupSize, outputsWidth));

        mBlockX = mNbAnchors;
        mBlockY = 1;
        mBlockZ = batchSize;
        mThreadX = groupWidth;
        mThreadY = groupSize / groupWidth;
        mThreadZ = 1;



        std::cout << "AnchorCell"
                  << ":\n"
                     "    Max. Threads per Blocks = " << maxSize
                  << "\n"
                     "    Preferred Blocks Size multiple = " << prefMultiple
                  << "\n"
                     "    Blocks size = (" << mThreadX << ", "
                  << mThreadY << ", " << mThreadZ
                  << ") = "
                  << std::max<unsigned long>(mThreadX, 1UL)
                     * std::max<unsigned long>(mThreadY, 1UL)
                     * std::max<unsigned long>(mThreadZ, 1UL)
                  << "\n"
                     "    Grid size = (" << mBlockX << ", "
                  << mBlockY << ", " << mBlockZ << ") = "
                  << std::max<unsigned long>(mBlockX, 1UL)
                     * std::max<unsigned long>(mBlockY, 1UL)
                     * std::max<unsigned long>(mBlockZ, 1UL) << "\n"
                  << "    Multi-Processors used = "
                  << (mBlockX)
                     * (std::max<unsigned long>(mBlockY, 1UL))
                     * (std::max<unsigned long>(mBlockZ, 1UL))
                  << std::endl;


    }

    nvinfer1::Dims mOutputDims;
    unsigned int mStimuliHeight;
    unsigned int mStimuliWidth;
    unsigned int mFeatureMapHeight;
    unsigned int mFeatureMapWidth;
    double mRatioX;
    double mRatioY;
    unsigned int mScoreCls;
    bool mIsCoordinateAnchors;
    bool mIsFlip;
    unsigned int mNbAnchors;
    int mThreadX;
    int mThreadY;
    int mThreadZ;
    int mBlockX;
    int mBlockY;
    int mBlockZ;
    float* mAnchorsGPU;
    std::vector<float> mAnchors_HOST;
    //float* mAnchorsPrecompute;
    std::string mNamespace;

};





class AnchorGPUPluginCreator : public nvinfer1::IPluginCreator
{
public:
    AnchorGPUPluginCreator() {

        mPluginAttributes.emplace_back(nvinfer1::PluginField("batchSize", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
        mPluginAttributes.emplace_back(nvinfer1::PluginField("nbOutputs", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
        mPluginAttributes.emplace_back(nvinfer1::PluginField("outputHeight", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
        mPluginAttributes.emplace_back(nvinfer1::PluginField("outputWidth", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
        mPluginAttributes.emplace_back(nvinfer1::PluginField("stimuliHeight", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
        mPluginAttributes.emplace_back(nvinfer1::PluginField("stimuliWidth", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
        mPluginAttributes.emplace_back(nvinfer1::PluginField("featureMapWidth", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
        mPluginAttributes.emplace_back(nvinfer1::PluginField("featureMapHeight", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
        mPluginAttributes.emplace_back(nvinfer1::PluginField("nbAnchors", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
        mPluginAttributes.emplace_back(nvinfer1::PluginField("scoreCls", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
        mPluginAttributes.emplace_back(nvinfer1::PluginField("isCoordinatesAnchors", nullptr, nvinfer1::PluginFieldType::kCHAR, 1));
        mPluginAttributes.emplace_back(nvinfer1::PluginField("isFlip", nullptr, nvinfer1::PluginFieldType::kCHAR, 1));
        mPluginAttributes.emplace_back(nvinfer1::PluginField("anchor", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 4*nbAnchors));

        mFC.nbFields = mPluginAttributes.size();
        mFC.fields = mPluginAttributes.data();
    }

    ~AnchorGPUPluginCreator() override
    {

    }

    const char* getPluginName() const 
#if NV_TENSORRT_MAJOR > 7
    noexcept
#endif
    override {
        const char* ANCHOR_GPU_PLUGIN_NAME{"AnchorGPUPlugin"};
        return ANCHOR_GPU_PLUGIN_NAME;
    }

    const char* getPluginVersion() const 
#if NV_TENSORRT_MAJOR > 7
    noexcept
#endif
    override {
        const char* ANCHOR_GPU_PLUGIN_VERSION{"1"};
        return ANCHOR_GPU_PLUGIN_VERSION;
    }

    const nvinfer1::PluginFieldCollection* getFieldNames() 
#if NV_TENSORRT_MAJOR > 7
    noexcept
#endif
    override {
        return &mFC;
    }

    nvinfer1::IPluginV2Ext* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) 
#if NV_TENSORRT_MAJOR > 7
    noexcept
#endif
    override {
        const nvinfer1::PluginField* fields = fc->fields;
        int nbFields = fc->nbFields;

        for (int i = 0; i < nbFields; ++i)
        {
            const char* attrName = fields[i].name;
            if (!strcmp(attrName, "batchSize"))
            {
                //ASSERT(fields[i].type == nvinfer1::PluginFieldType::kINT32);
                batchSize = *(static_cast<const int*>(fields[i].data));
            }
            if (!strcmp(attrName, "nbOutputs"))
            {
                //ASSERT(fields[i].type == nvinfer1::PluginFieldType::kINT32);
                nbOutputs = *(static_cast<const int*>(fields[i].data));
            }
            if (!strcmp(attrName, "outputHeight"))
            {
                //ASSERT(fields[i].type == nvinfer1::PluginFieldType::kINT32);
                outputHeight = *(static_cast<const int*>(fields[i].data));
            }
            if (!strcmp(attrName, "outputWidth"))
            {
                //ASSERT(fields[i].type == nvinfer1::PluginFieldType::kINT32);
                outputWidth = *(static_cast<const int*>(fields[i].data));
            }
            if (!strcmp(attrName, "stimuliWidth"))
            {
                //ASSERT(fields[i].type == nvinfer1::PluginFieldType::kINT32);
                stimuliWidth = *(static_cast<const int*>(fields[i].data));
            }
            if (!strcmp(attrName, "stimuliHeight"))
            {
                //ASSERT(fields[i].type == nvinfer1::PluginFieldType::kINT32);
                stimuliHeight = *(static_cast<const int*>(fields[i].data));
            }
            if (!strcmp(attrName, "scoreCls"))
            {
                //ASSERT(fields[i].type == nvinfer1::PluginFieldType::kINT32);
                scoreCls = *(static_cast<const int*>(fields[i].data));
            }
            if (!strcmp(attrName, "isFlip"))
            {
                //ASSERT(fields[i].type == nvinfer1::PluginFieldType::kINT32);
                isFlip = *(static_cast<const bool*>(fields[i].data));
            }

            if (!strcmp(attrName, "featureMapWidth"))
            {
                //ASSERT(fields[i].type == nvinfer1::PluginFieldType::kINT32);
                featureMapWidth = *(static_cast<const int*>(fields[i].data));
            }
            if (!strcmp(attrName, "featureMapHeight"))
            {
                //ASSERT(fields[i].type == nvinfer1::PluginFieldType::kINT32);
                featureMapHeight = *(static_cast<const int*>(fields[i].data));
            }
            if (!strcmp(attrName, "nbAnchors"))
            {
                //ASSERT(fields[i].type == nvinfer1::PluginFieldType::kINT32);
                nbAnchors = *(static_cast<const int*>(fields[i].data));
            }
            if (!strcmp(attrName, "isCoordinatesAnchors"))
            {
                //ASSERT(fields[i].type == nvinfer1::PluginFieldType::kCHAR);
                isCoordinatesAnchors = *(static_cast<const bool*>(fields[i].data));
            }
            if (!strcmp(attrName, "anchors"))
            {
                //ASSERT(fields[i].type == nvinfer1::PluginFieldType::kFLOAT32);
                const float* anc = static_cast<const float*>(fields[i].data);
                for (int j = 0; j < nbAnchors*4; ++j)
                {
                    anchors.push_back(*anc);
                    anc++;
                }
            }
        }

        // This object will be deleted when the network is destroyed, which will
        // call RPROIPlugin::terminate()
        AnchorGPUPlugin* plugin = new AnchorGPUPlugin(    batchSize,
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
                                                            anchors.data());

        //plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;


    }

    nvinfer1::IPluginV2Ext* deserializePlugin(const char* name, const void* serialData, size_t serialLength) 
#if NV_TENSORRT_MAJOR > 7
    noexcept
#endif
    override {
        AnchorGPUPlugin* plugin = new AnchorGPUPlugin(serialData, serialLength);
        return plugin;
    }
    void setPluginNamespace(const char* pluginNamespace) 
#if NV_TENSORRT_MAJOR > 7
    noexcept
#endif
    override {
        mNamespace = pluginNamespace;
    }
    const char* getPluginNamespace() const 
#if NV_TENSORRT_MAJOR > 7
    noexcept
#endif
    override {
        return mNamespace.c_str();
    }


private:
    nvinfer1::PluginFieldCollection mFC;

    unsigned int batchSize;
    unsigned int nbOutputs;
    unsigned int outputHeight;
    unsigned int outputWidth;
    unsigned int stimuliHeight;
    unsigned int stimuliWidth;
    unsigned int featureMapWidth;
    unsigned int featureMapHeight;
    unsigned int scoreCls;
    bool isCoordinatesAnchors;
    bool isFlip;
    unsigned int nbAnchors;
    std::vector<float> anchors;

    std::string mNamespace;
    std::vector<nvinfer1::PluginField> mPluginAttributes;
};

REGISTER_TENSORRT_PLUGIN(AnchorGPUPluginCreator);

#endif