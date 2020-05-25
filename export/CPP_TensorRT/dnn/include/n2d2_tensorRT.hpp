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

#ifndef N2D2_TENSORRT_HPP
#define N2D2_TENSORRT_HPP

#include "dnn_utils.hpp"
#include "kernels_cpu.hpp"
#include "kernels_gpu.hpp"
#include "BatchStream.hpp"

#if NV_TENSORRT_MAJOR > 2
#include "IInt8EntropyCalibrator.hpp"
#endif

struct tsrRTHandleStruct {
    nvinfer1::IExecutionContext* context;
    nvinfer1::IBuilder* netBuilder;
    std::vector<nvinfer1::INetworkDefinition*> netDef;
    nvinfer1::DataType dT;
};
extern tsrRTHandleStruct tsrRTHandles;

void set_profiling();

void logNetworkDef(nvinfer1::INetworkDefinition* network_tensorRT);

template <class T> struct n2d2type{};

/**** Targets Layers ****/
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

void get_output(unsigned int batchSize,
                unsigned int nbOutputs,
                unsigned int outputsHeight,
                unsigned int outputsWidth,
                void* dataIn,
                DATA_T* outputEstimated);

void report_per_layer_profiling(unsigned int nbIter);

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
                    cudaStream_t stream);

/**** Confusion Matrix ****/

void dumpMem(int size, DATA_T* data, std::string fileName);



/**Plugin Layer implementation**/
/**Anchoring CPU implementation**/
class AnchorCPUPlugin: public nvinfer1::IPlugin
{
public:
	AnchorCPUPlugin(unsigned int batchSize,
                    unsigned int nbOutputs,
                    unsigned int outputHeight,
                    unsigned int outputWidth,
                    unsigned int stimuliHeight,
                    unsigned int stimuliWidth,
                    unsigned int featureMapWidth,
                    unsigned int featureMapHeight,
                    unsigned int scoreCls,
                    bool isFlip,
                    unsigned int nbAnchors,
                    const WDATA_T*  anchors)
	{
        mOutputDims.d[0] = batchSize;
        mOutputDims.d[1] = nbOutputs;
        mOutputDims.d[2] = outputHeight;
        mOutputDims.d[3] = outputWidth;

        mStimuliHeight = stimuliHeight;
        mStimuliWidth = stimuliWidth;
        mFeatureMapWidth = featureMapWidth;
        mFeatureMapHeight = featureMapHeight;

        mRatioX = std::ceil(stimuliWidth / (double) outputWidth);
        mRatioY = std::ceil(stimuliHeight / (double) outputHeight);
        mScoreCls = scoreCls;
        mIsFlip = isFlip;
        mNbAnchors = nbAnchors;
        mAnchors.resize(mNbAnchors);
        for(unsigned int i = 0; i < mNbAnchors*4; i += 4)
        {
            mAnchors[i/4].x0 = anchors[i + 0];
            mAnchors[i/4].y0 = anchors[i + 1];
            mAnchors[i/4].x1 = anchors[i + 2];
            mAnchors[i/4].y1 = anchors[i + 3];
        }
	}

	AnchorCPUPlugin(const void* dataFromRuntime, size_t length)
	{
		const char* d = reinterpret_cast<const char*>(dataFromRuntime), *a = d;
        mOutputDims.d[0] = read<int>(d);
		mOutputDims.d[1] = read<int>(d);
		mOutputDims.d[2] = read<int>(d);
		mOutputDims.d[3] = read<int>(d);
        mStimuliHeight = (unsigned int) read<int>(d);
        mStimuliWidth = (unsigned int) read<int>(d);
        mFeatureMapWidth = (unsigned int) read<int>(d);
        mFeatureMapHeight = (unsigned int) read<int>(d);
        mRatioX = read<double>(d);
        mRatioY = read<double>(d);
        mScoreCls = (unsigned int) read<int>(d);
        mIsFlip = read<bool>(d);
        mNbAnchors = (unsigned int) read<int>(d);
        mAnchors.resize(mNbAnchors);
        for(unsigned int i = 0; i < mNbAnchors; ++i)
        {
            mAnchors[i].x0 = read<WDATA_T>(d);
            mAnchors[i].y0 = read<WDATA_T>(d);
            mAnchors[i].x1 = read<WDATA_T>(d);
            mAnchors[i].y1 = read<WDATA_T>(d);
        }
		assert(d == a + length);
	}

	~AnchorCPUPlugin()
	{
        mAnchors = std::vector<Anchor>();
	}

	virtual int getNbOutputs() const override
	{
        return 1;
	}

	virtual nvinfer1::Dims getOutputDimensions(int index,
                                               const nvinfer1::Dims* inputDim,
                                               int nbInputDims) override
	{
        return nvinfer1::DimsCHW(mOutputDims.d[1], mOutputDims.d[2], mOutputDims.d[3]);
	}

	virtual void configure(const nvinfer1::Dims* inputDims,
                   int nbInputs,
                   const nvinfer1::Dims* outputDims,
                   int nbOutputs,
                   int maxBatchSize) override
	{

	}

	virtual int initialize() override
	{
		return 0;
	}

	virtual void terminate() override
	{

	}

	virtual size_t getWorkspaceSize(int maxBatchSize) const override
	{
		return 0;
	}

	virtual int enqueue(int batchSize,
                        const void*const * inputs,
                        void** outputs,
                        void* workspace,
                        cudaStream_t stream) override
	{
        DATA_T* inputDataCPU(NULL);
        DATA_T* outputDataCPU(NULL);
        size_t size_cpy = mOutputDims.d[3]*mOutputDims.d[2]*mOutputDims.d[1]*mOutputDims.d[0];

        inputDataCPU = new DATA_T[size_cpy];
        outputDataCPU = new DATA_T[size_cpy];

        if (!inputDataCPU)
            throw std::runtime_error(
                "enqueue(): could not allocate memory");

        if (!outputDataCPU)
            throw std::runtime_error(
                "enqueue(): could not allocate memory");

        CHECK_CUDA_STATUS(cudaMemcpy(inputDataCPU,
                                     reinterpret_cast<const float*>(inputs[0]),
                                     size_cpy*sizeof(DATA_T),
                                     cudaMemcpyDeviceToHost));

        CHECK_CUDA_STATUS(cudaStreamSynchronize(stream));

        anchor_cpu(mOutputDims.d[0],
                    mOutputDims.d[1],
                    mOutputDims.d[2],
                    mOutputDims.d[3],
                    mStimuliHeight,
                    mStimuliWidth,
                    mScoreCls,
                    mIsFlip,
                    mNbAnchors,
                    mRatioX,
                    mRatioY,
                    mAnchors,
                    inputDataCPU,
                    outputDataCPU);

         CHECK_CUDA_STATUS(cudaMemcpy(outputs[0],
             outputDataCPU,
             size_cpy*sizeof(DATA_T),
             cudaMemcpyHostToDevice));
        delete[] inputDataCPU;
        delete[] outputDataCPU;

        return 0;
	}

	virtual size_t getSerializationSize() override
	{
        size_t inputDimParamSize = 4*sizeof(int); //nbOutputs, nbOutputsHeight, nbOutputWidth = 3
        size_t stimuliParamSize = 4*sizeof(int); //Stimuliheight and StimuliWidth
        size_t ratioParamSize = 2*sizeof(double); //RatioX and RatioY
        size_t anchorsSize = sizeof(int)*2 + 4*mNbAnchors*sizeof(WDATA_T) + sizeof(bool); // mNbAnchors and (x0 y0 x1 y1) * mNbAnchors + mScoreCls

        mSerializationSize = inputDimParamSize + stimuliParamSize
                                + ratioParamSize + anchorsSize;

        return mSerializationSize;
	}

	virtual void serialize(void* buffer) override
	{
        char* d = reinterpret_cast<char*>(buffer);
        char* a = d;
        write<int>(d, (int)mOutputDims.d[0]);
        write<int>(d, (int)mOutputDims.d[1]);
        write<int>(d, (int)mOutputDims.d[2]);
        write<int>(d, (int)mOutputDims.d[3]);
        write<int>(d, (int)mStimuliHeight);
        write<int>(d, (int)mStimuliWidth);
        write<int>(d, (int)mFeatureMapWidth);
        write<int>(d, (int)mFeatureMapHeight);
        write<double>(d, mRatioX);
        write<double>(d, mRatioY);
        write<int>(d, (int)mScoreCls);
        write<bool>(d, mIsFlip);
        write<int>(d, (int)mNbAnchors);

        for(unsigned int i = 0; i < mNbAnchors; ++i)
        {
            write<WDATA_T>(d, mAnchors[i].x0);
            write<WDATA_T>(d, mAnchors[i].y0);
            write<WDATA_T>(d, mAnchors[i].x1);
            write<WDATA_T>(d, mAnchors[i].y1);
        }

        assert(d == a + getSerializationSize());
	}

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

    nvinfer1::Dims mOutputDims;
    unsigned int mStimuliHeight;
    unsigned int mStimuliWidth;
    unsigned int mFeatureMapHeight;
    unsigned int mFeatureMapWidth;
    double mRatioX;
    double mRatioY;
    unsigned int mScoreCls;
    bool mIsFlip;
    unsigned int mNbAnchors;
    std::vector<Anchor> mAnchors;
    size_t mSerializationSize;

};
/**Plugin Layer implementation**/
/**Anchoring GPU implementation**/
class AnchorGPUPlugin: public nvinfer1::IPlugin
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
                    bool isFlip,
                    unsigned int nbAnchors,
                    const WDATA_T*  anchors)
	{
        mOutputDims.d[0] = batchSize;
        mOutputDims.d[1] = nbOutputs;
        mOutputDims.d[2] = outputHeight;
        mOutputDims.d[3] = outputWidth;

        mStimuliHeight = stimuliHeight;
        mStimuliWidth = stimuliWidth;
        mFeatureMapWidth = featureMapWidth;
        mFeatureMapHeight = featureMapHeight;

        mRatioX = std::ceil(mFeatureMapWidth / (double) outputWidth);
        mRatioY = std::ceil(featureMapHeight / (double) outputHeight);
        mScoreCls = scoreCls;
        mIsFlip = isFlip;
        mNbAnchors = nbAnchors;

        checkCudaErrors( cudaMalloc((void**)&mAnchorsGPU,
                         4*mNbAnchors*sizeof(DATA_T)) );
        checkCudaErrors( cudaMemcpy(mAnchorsGPU,
                         anchors,
                         4*mNbAnchors*sizeof(DATA_T),
                         cudaMemcpyHostToDevice) );
        gpuThreadAllocation();
	}

	AnchorGPUPlugin(const void* dataFromRuntime, size_t length)
	{
		const char* d = reinterpret_cast<const char*>(dataFromRuntime), *a = d;
        mOutputDims.d[0] = read<int>(d);
		mOutputDims.d[1] = read<int>(d);
		mOutputDims.d[2] = read<int>(d);
		mOutputDims.d[3] = read<int>(d);
        mStimuliHeight = (unsigned int) read<int>(d);
        mStimuliWidth = (unsigned int) read<int>(d);
        mFeatureMapWidth = (unsigned int) read<int>(d);
        mFeatureMapHeight = (unsigned int) read<int>(d);
        mRatioX = read<double>(d);
        mRatioY = read<double>(d);
        mScoreCls = (unsigned int) read<int>(d);
        mIsFlip = read<bool>(d);
        mNbAnchors = (unsigned int) read<int>(d);
		mAnchorsGPU = deserializeToDevice(d, mNbAnchors*4);
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

	virtual int getNbOutputs() const override
	{
        return 1;
	}

	virtual nvinfer1::Dims getOutputDimensions(int index,
                                               const nvinfer1::Dims* inputDim,
                                               int nbInputDims) override
	{
        unsigned int batchInput = 1;

        if(inputDim[0].nbDims == 4)
            batchInput = inputDim[0].d[0];

        return nvinfer1::DimsCHW(mOutputDims.d[1], mOutputDims.d[2], mOutputDims.d[3]);
	}

	virtual void configure(const nvinfer1::Dims* inputDims,
                   int nbInputs,
                   const nvinfer1::Dims* outputDims,
                   int nbOutputs,
                   int maxBatchSize) override
	{

	}

	virtual int initialize() override
	{
		return 0;
	}

	virtual void terminate() override
	{

	}

	virtual size_t getWorkspaceSize(int maxBatchSize) const override
	{
		return 0;
	}

	virtual int enqueue(int batchSize,
                        const void*const * inputs,
                        void** outputs,
                        void* workspace,
                        cudaStream_t stream) override
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
                            mFeatureMapWidth,
                            mFeatureMapHeight,
                            mScoreCls,
                            mIsFlip,
                            mNbAnchors,
                            mRatioX,
                            mRatioY,
                            reinterpret_cast<const DATA_T *>(mAnchorsGPU),
                            reinterpret_cast<const DATA_T *>(inputs[0]),
                            reinterpret_cast<DATA_T *>(outputs[0]),
                            threadGrid,
                            blockGrid,
                            stream);

       return 0;
	}

	virtual size_t getSerializationSize() override
	{
        size_t inputDimParamSize = 4*sizeof(int); //nbOutputs, nbOutputsHeight, nbOutputWidth = 3
        size_t stimuliParamSize = 4*sizeof(int); //Stimuliheight and StimuliWidth
        size_t ratioParamSize = 2*sizeof(double); //RatioX and RatioY
        size_t anchorsSize = sizeof(int)*2 + 4*mNbAnchors*sizeof(DATA_T) + sizeof(bool); // mNbAnchors and (x0 y0 x1 y1) * mNbAnchors + mScoreCls
        size_t threadSize = 3*2*sizeof(int);
        mSerializationSize = inputDimParamSize + stimuliParamSize
                                + ratioParamSize + anchorsSize + threadSize;

        return mSerializationSize;
	}

	virtual void serialize(void* buffer) override
	{
        char* d = reinterpret_cast<char*>(buffer);
        char* a = d;
        write<int>(d, (int)mOutputDims.d[0]);
        write<int>(d, (int)mOutputDims.d[1]);
        write<int>(d, (int)mOutputDims.d[2]);
        write<int>(d, (int)mOutputDims.d[3]);
        write<int>(d, (int)mStimuliHeight);
        write<int>(d, (int)mStimuliWidth);
        write<int>(d, (int)mFeatureMapWidth);
        write<int>(d, (int)mFeatureMapHeight);

        write<double>(d, mRatioX);
        write<double>(d, mRatioY);
        write<int>(d, (int)mScoreCls);
        write<bool>(d, mIsFlip);
        write<int>(d, (int)mNbAnchors);
        serializeFromDevice(d, mAnchorsGPU, mNbAnchors*4);
        write<int>(d, mThreadX);
        write<int>(d, mThreadY);
        write<int>(d, mThreadZ);
        write<int>(d, mBlockX);
        write<int>(d, mBlockY);
        write<int>(d, mBlockZ);
        assert(d == a + getSerializationSize());
	}

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

    DATA_T* deserializeToDevice(const char*& hostBuffer, size_t dataSize)
    {
        DATA_T* gpuData;
        checkCudaErrors(cudaMalloc(&gpuData, dataSize*sizeof(DATA_T)));
        checkCudaErrors(cudaMemcpy(gpuData, hostBuffer, dataSize*sizeof(DATA_T), cudaMemcpyHostToDevice));
        hostBuffer += dataSize*sizeof(DATA_T);
        return gpuData;
    }

    void serializeFromDevice(char*& hostBuffer, DATA_T* deviceWeights, size_t dataSize)
    {
        checkCudaErrors(cudaMemcpy(hostBuffer, deviceWeights, dataSize*sizeof(DATA_T), cudaMemcpyDeviceToHost));
        hostBuffer += dataSize*sizeof(DATA_T);
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
    bool mIsFlip;
    unsigned int mNbAnchors;
    int mThreadX;
    int mThreadY;
    int mThreadZ;
    int mBlockX;
    int mBlockY;
    int mBlockZ;
    DATA_T* mAnchorsGPU;
    size_t mSerializationSize;
};

/**Plugin Layer implementation**/
/**ROI Pooling GPU implementation**/
class ROIPoolingGPUPlugin: public nvinfer1::IPlugin
{
public:
	ROIPoolingGPUPlugin(unsigned int batchSize,
                        unsigned int nbOutputs,
                        unsigned int outputHeight,
                        unsigned int outputWidth,
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

#if NV_TENSORRT_MAJOR < 3
        throw std::runtime_error(
            "ROIPoolingGPUPlugin: only supported for at least TensorRT 3 version");
#endif
        mBatchSize = batchSize;
        mOutputDims.d[0] = nbProposals;
        mOutputDims.d[1] = nbOutputs;
        mOutputDims.d[2] = outputHeight;
        mOutputDims.d[3] = outputWidth;
        mStimuliHeight = stimuliHeight;
        mStimuliWidth = stimuliWidth;
        for(unsigned int i = 0; i < nbFeature; ++i)
            mFeatureDims.push_back( nvinfer1::DimsCHW(  featureChannels[i],
                                                        featureHeight[i],
                                                        featureWidth[i]));

        mNbInputFeature = nbFeature;
        mPoolType = poolType;
        mNbProposals = nbProposals;
        mFlip = isFlip;
        mIgnorePadding= ignorePadding;
        gpuThreadAllocation();
	}

	ROIPoolingGPUPlugin(const void* dataFromRuntime, size_t length)
	{
		const char* d = reinterpret_cast<const char*>(dataFromRuntime), *a = d;
        mOutputDims.d[0] = read<int>(d);
		mOutputDims.d[1] = read<int>(d);
		mOutputDims.d[2] = read<int>(d);
		mOutputDims.d[3] = read<int>(d);
        mNbInputFeature = read<int>(d);
        mFeatureDims.resize(mNbInputFeature);
        for(unsigned int k = 0; k < mNbInputFeature; ++k)
            mFeatureDims[k] = read<nvinfer1::DimsCHW>(d);

        mThreadX.resize(mNbInputFeature);
        mThreadY.resize(mNbInputFeature);
        mThreadZ.resize(mNbInputFeature);
        for(unsigned int k = 0; k < mNbInputFeature; ++k)
        {
            mThreadX[k] = read<int>(d);
            mThreadY[k] = read<int>(d);
            mThreadZ[k] = read<int>(d);
        }
        mBlockX.resize(mNbInputFeature);
        mBlockY.resize(mNbInputFeature);
        mBlockZ.resize(mNbInputFeature);
        for(unsigned int k = 0; k < mNbInputFeature; ++k)
        {
            mBlockX[k] = read<int>(d);
            mBlockY[k] = read<int>(d);
            mBlockZ[k] = read<int>(d);
        }

        mStimuliHeight = read<int>(d);
        mStimuliWidth = read<int>(d);

        mPoolType= read <Pooling_T>(d);
        mNbProposals = (unsigned int) read<int>(d);
        mFlip = read<bool>(d);
        mIgnorePadding = read<bool>(d);
		assert(d == a + length);
	}

	~ROIPoolingGPUPlugin()
	{

	}

	virtual int getNbOutputs() const override
	{
        return (int) 1;
	}

	virtual nvinfer1::Dims getOutputDimensions(int index,
                                               const nvinfer1::Dims* inputDim,
                                               int nbInputDims) override
	{
        return nvinfer1::DimsNCHW(mOutputDims.d[0], mOutputDims.d[1], mOutputDims.d[2], mOutputDims.d[3]);

	}

	virtual void configure(const nvinfer1::Dims* inputDims,
                   int nbInputs,
                   const nvinfer1::Dims* outputDims,
                   int nbOutputs,
                   int maxBatchSize) override
	{

	}

	virtual int initialize() override
	{
		return 0;
	}

	virtual void terminate() override
	{

	}

	virtual size_t getWorkspaceSize(int maxBatchSize) const override
	{
		return 0;
	}

	virtual int enqueue(int batchSize,
                        const void*const * inputs,
                        void** outputs,
                        void* workspace,
                        cudaStream_t stream) override
	{

    unsigned int outputOffset = 0;
    for(unsigned int k = 0; k < mNbInputFeature; ++k)
    {
        DATA_T alpha = 1;
        DATA_T beta = 0;
        dim3 threadGrid = {mThreadX[k], mThreadY[k], mThreadZ[k]};
        dim3 blockGrid = {mBlockX[k], mBlockY[k], mBlockZ[k]};


        if(k>0)
            beta = 1;

        if(mPoolType == Bilinear || mPoolType == BilinearTF)
            cuda_roipooling_bilinear_propagate( alpha,
                                                reinterpret_cast<const DATA_T*>(inputs[0]),
                                                0,
                                                //proposalIdx,
                                                mNbProposals,
                                                mStimuliHeight,
                                                mStimuliWidth,
                                                reinterpret_cast<const DATA_T*>(inputs[k + 1]),
                                                mFeatureDims[k].d[0],
                                                mFeatureDims[k].d[1],
                                                mFeatureDims[k].d[2],
                                                batchSize,
                                                0,
                                                beta,
                                                //reinterpret_cast<DATA_T*>(outputs[proposalIdx]),
                                                reinterpret_cast<DATA_T*>(outputs[0]),
                                                mOutputDims.d[1],
                                                mOutputDims.d[2],
                                                mOutputDims.d[3],
                                                outputOffset,
                                                mPoolType == BilinearTF ? true : false,
                                                mFlip,
                                                mIgnorePadding,
                                                threadGrid,
                                                blockGrid,
                                                stream);
        else
            throw std::runtime_error(
                "ROIPoolingGPUPlugin: ony bilinears method are implemented");

        //outputOffset += mFeatureDims[k].d[0]*mOutputDims.d[2]*mOutputDims.d[3]*(batchSize*mOutputDims.d[0]);
        outputOffset += mFeatureDims[k].d[0];
    }


        return 0;
	}

	virtual size_t getSerializationSize() override
	{
        size_t inputDimParamSize = 4*sizeof(int); //
        size_t ROIParamI = 4*sizeof(int) + sizeof(Pooling_T) + 6*sizeof(int)*mNbInputFeature + 2*sizeof(bool)
                            + mNbInputFeature*sizeof(nvinfer1::DimsCHW); //

        mSerializationSize = inputDimParamSize + ROIParamI;

        return mSerializationSize;
	}

	virtual void serialize(void* buffer) override
	{
        char* d = reinterpret_cast<char*>(buffer);
        char* a = d;
        write<int>(d, (int)mOutputDims.d[0]);
        write<int>(d, (int)mOutputDims.d[1]);
        write<int>(d, (int)mOutputDims.d[2]);
        write<int>(d, (int)mOutputDims.d[3]);
        write<int>(d, (int)mNbInputFeature);

        for(unsigned int k = 0; k < mNbInputFeature; ++k)
            write<nvinfer1::DimsCHW>(d, mFeatureDims[k]);

        for(unsigned int k = 0; k < mNbInputFeature; ++k)
        {
            write<int>(d, mThreadX[k]);
            write<int>(d, mThreadY[k]);
            write<int>(d, mThreadZ[k]);
        }

        for(unsigned int k = 0; k < mNbInputFeature; ++k)
        {
            write<int>(d, mBlockX[k]);
            write<int>(d, mBlockY[k]);
            write<int>(d, mBlockZ[k]);
        }

        write<int>(d, (int)mStimuliHeight);
        write<int>(d, (int)mStimuliWidth);
        write<Pooling_T>(d , mPoolType);
        write<int>(d, (int)mNbProposals);
        write<bool>(d, mFlip);
        write<bool>(d, mIgnorePadding);
        assert(d == a + getSerializationSize());
	}

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
    void gpuThreadAllocation()
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, 0);
        for(unsigned int i = 0; i < mFeatureDims.size(); ++i)
        {
            const unsigned int nbProposals = mOutputDims.d[0];
            const unsigned int nbOutputs = mOutputDims.d[1];
            const unsigned int outputsHeight = mOutputDims.d[2];
            const unsigned int outputsWidth = mOutputDims.d[3];

            const unsigned int nbChannels = mFeatureDims[i].d[0];
            const unsigned int channelsHeight = mFeatureDims[i].d[1];
            const unsigned int channelsWidth = mFeatureDims[i].d[2];

            const unsigned int maxSize
                = (unsigned int)deviceProp.maxThreadsPerBlock;
            const unsigned int prefMultiple = (unsigned int)deviceProp.warpSize;

            const unsigned int groupSize = (outputsWidth * outputsHeight < maxSize)
                                               ? outputsWidth * outputsHeight
                                               : maxSize;
            const unsigned int groupWidth
                = std::min(prefMultiple, nextDivisor(groupSize, outputsWidth));

            mThreadX.push_back(groupWidth);
            mThreadY.push_back(groupSize / groupWidth);
            mThreadZ.push_back(1);

            mBlockX.push_back(nbChannels);
            mBlockY.push_back(1);
            mBlockZ.push_back(mBatchSize*nbProposals);

            std::cout << "ROI Pooling"
                      << ":\n"
                         "    Max. Threads per Blocks = " << maxSize
                      << "\n"
                         "    Preferred Blocks Size multiple = " << prefMultiple
                      << "\n"
                         "    Blocks size = (" << mThreadX.back() << ", "
                      << mThreadY.back() << ", " << mThreadZ.back()
                      << ") = "
                      << std::max<unsigned long>(mThreadX.back(), 1UL)
                         * std::max<unsigned long>(mThreadY.back(), 1UL)
                         * std::max<unsigned long>(mThreadZ.back(), 1UL)
                      << "\n"
                         "    Grid size = (" << mBlockX.back() << ", "
                      << mBlockY.back() << ", " << mBlockZ.back() << ") = "
                      << std::max<unsigned long>(mBlockX.back(), 1UL)
                         * std::max<unsigned long>(mBlockY.back(), 1UL)
                         * std::max<unsigned long>(mBlockZ.back(), 1UL) << "\n"
                      << "    Multi-Processors used = "
                      << (mBlockX.back())
                         * (std::max<unsigned long>(mBlockY.back(), 1UL))
                         * (std::max<unsigned long>(mBlockZ.back(), 1UL))
                      << std::endl;
        }
    }


    nvinfer1::Dims mOutputDims;
    std::vector<nvinfer1::DimsCHW> mFeatureDims;
    std::vector<unsigned int> mThreadX;
    std::vector<unsigned int> mThreadY;
    std::vector<unsigned int> mThreadZ;
    std::vector<unsigned int> mBlockX;
    std::vector<unsigned int> mBlockY;
    std::vector<unsigned int> mBlockZ;
    unsigned int mBatchSize;
    unsigned int mNbInputFeature;
    unsigned int mStimuliHeight;
    unsigned int mStimuliWidth;
    unsigned int mOutputWidth;
    unsigned int mOutputHeight;
    unsigned int mNbOutputs;
    unsigned int mNbProposals;
    Pooling_T mPoolType;
    bool mFlip;
    bool mIgnorePadding;
    size_t mSerializationSize;

};

/**Plugin Layer implementation**/
/**ROI Pooling CPU implementation**/
class ROIPoolingCPUPlugin: public nvinfer1::IPlugin
{
public:
	ROIPoolingCPUPlugin(unsigned int batchSize,
                        unsigned int nbOutputs,
                        unsigned int outputHeight,
                        unsigned int outputWidth,
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
        mOutputDims.d[0] = nbProposals;
        mOutputDims.d[1] = nbOutputs;
        mOutputDims.d[2] = outputHeight;
        mOutputDims.d[3] = outputWidth;
        mStimuliHeight = stimuliHeight;
        mStimuliWidth = stimuliWidth;
        for(unsigned int i = 0; i < nbFeature; ++i)
            mFeatureDims.push_back( nvinfer1::DimsCHW(  featureChannels[i],
                                                        featureHeight[i],
                                                        featureWidth[i]));

        mNbInputFeature = nbFeature;
        mPoolType = poolType;
        mNbProposals = nbProposals;
	}

	ROIPoolingCPUPlugin(const void* dataFromRuntime, size_t length)
	{
		const char* d = reinterpret_cast<const char*>(dataFromRuntime), *a = d;
        mOutputDims.d[0] = read<int>(d);
		mOutputDims.d[1] = read<int>(d);
		mOutputDims.d[2] = read<int>(d);
		mOutputDims.d[3] = read<int>(d);
        mNbInputFeature = read<int>(d);
        mFeatureDims.resize(mNbInputFeature);
        for(unsigned int k = 0; k < mNbInputFeature; ++k)
            mFeatureDims[k] = read<nvinfer1::DimsCHW>(d);

        mStimuliHeight = read<int>(d);
        mStimuliWidth = read<int>(d);

        mPoolType= read <Pooling_T>(d);
        mNbProposals = (unsigned int) read<int>(d);
		assert(d == a + length);
	}

	~ROIPoolingCPUPlugin()
	{

	}

	virtual int getNbOutputs() const override
	{
        return (int) 1;
	}

	virtual nvinfer1::Dims getOutputDimensions(int index,
                                               const nvinfer1::Dims* inputDim,
                                               int nbInputDims) override
	{
        return nvinfer1::DimsNCHW(mOutputDims.d[0], mOutputDims.d[1], mOutputDims.d[2], mOutputDims.d[3]);
	}

	virtual void configure(const nvinfer1::Dims* inputDims,
                   int nbInputs,
                   const nvinfer1::Dims* outputDims,
                   int nbOutputs,
                   int maxBatchSize) override
	{

	}

	virtual int initialize() override
	{
		return 0;
	}

	virtual void terminate() override
	{

	}

	virtual size_t getWorkspaceSize(int maxBatchSize) const override
	{
		return 0;
	}

	virtual int enqueue(int batchSize,
                        const void*const * inputs,
                        void** outputs,
                        void* workspace,
                        cudaStream_t stream) override
	{
        DATA_T* inputDataCPU(NULL);
        DATA_T* outputDataCPU(NULL);
        size_t size_output_cpy = mOutputDims.d[0]*mOutputDims.d[1]
                                    *mOutputDims.d[2]*mOutputDims.d[3];

        size_t size_regionproposal = 4*mNbProposals;
        size_t size_input_cpy = size_regionproposal;
        size_t mem_offset = size_regionproposal*batchSize;

        for(unsigned int k = 0; k < mNbInputFeature; k++)
            size_input_cpy += mFeatureDims[k].d[0]*mFeatureDims[k].d[1]
                                *mFeatureDims[k].d[2];


        inputDataCPU = new DATA_T[size_input_cpy*batchSize];
        outputDataCPU = new DATA_T[size_output_cpy*batchSize];

        if (!inputDataCPU)
            throw std::runtime_error(
                "enqueue(): could not allocate memory");

        if (!outputDataCPU)
            throw std::runtime_error(
                "enqueue(): could not allocate memory");
        CHECK_CUDA_STATUS(cudaMemcpy(inputDataCPU,
                                     reinterpret_cast<const float*>(inputs[0]),
                                     size_regionproposal*sizeof(DATA_T)*batchSize,
                                     cudaMemcpyDeviceToHost));

        for(unsigned int k = 0; k < mNbInputFeature; ++k)
        {
            size_t feature_size = mFeatureDims[k].d[0]*mFeatureDims[k].d[1]
                                        *mFeatureDims[k].d[2];

            CHECK_CUDA_STATUS(cudaMemcpy(inputDataCPU + mem_offset,
                                         reinterpret_cast<const float*>(inputs[k + 1]),
                                         feature_size*batchSize*sizeof(DATA_T),
                                         cudaMemcpyDeviceToHost));

            mem_offset += feature_size*batchSize;
        }

        CHECK_CUDA_STATUS(cudaStreamSynchronize(stream));


        if(mPoolType == Bilinear)
        {
            ROIPooling_bilinear_cpu(mOutputDims.d[0]*batchSize,
                                    mOutputDims.d[1],
                                    mOutputDims.d[2],
                                    mOutputDims.d[3],
                                    mStimuliHeight,
                                    mStimuliWidth,
                                    mFeatureDims,
                                    mNbProposals,
                                    inputDataCPU,
                                    outputDataCPU);
        }

        size_t outputOffset = 0;
/*
        for(unsigned int kN = 0; kN < mOutputDims.d[0]; ++kN)
        {
             CHECK_CUDA_STATUS(cudaMemcpy(outputs[kN],
                 outputDataCPU + outputOffset,
                 mOutputDims.d[1]*mOutputDims.d[2]
                    *mOutputDims.d[3]*batchSize*sizeof(DATA_T),
                 cudaMemcpyHostToDevice));

             outputOffset += mOutputDims.d[1]*mOutputDims.d[2]*mOutputDims.d[3];
        }
*/

         CHECK_CUDA_STATUS(cudaMemcpy(outputs[0], outputDataCPU,
                                      batchSize*mOutputDims.d[0]*mOutputDims.d[1]
                                      *mOutputDims.d[2]*mOutputDims.d[3]*sizeof(DATA_T),
                                       cudaMemcpyHostToDevice));
        delete[] inputDataCPU;
        delete[] outputDataCPU;

        return 0;
	}

	virtual size_t getSerializationSize() override
	{
        size_t inputDimParamSize = 4*sizeof(int); //
        size_t ROIParamI = 4*sizeof(int) + sizeof(Pooling_T)
                            + mNbInputFeature*sizeof(nvinfer1::DimsCHW); //

        mSerializationSize = inputDimParamSize + ROIParamI;

        return mSerializationSize;
	}

	virtual void serialize(void* buffer) override
	{
        char* d = reinterpret_cast<char*>(buffer);
        char* a = d;
        write<int>(d, (int)mOutputDims.d[0]);
        write<int>(d, (int)mOutputDims.d[1]);
        write<int>(d, (int)mOutputDims.d[2]);
        write<int>(d, (int)mOutputDims.d[3]);
        write<int>(d, (int)mNbInputFeature);

        for(unsigned int k = 0; k < mNbInputFeature; ++k)
            write<nvinfer1::DimsCHW>(d, mFeatureDims[k]);

        write<int>(d, (int)mStimuliHeight);
        write<int>(d, (int)mStimuliWidth);
        write<Pooling_T>(d , mPoolType);
        write<int>(d, (int)mNbProposals);
        assert(d == a + getSerializationSize());
	}

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

    nvinfer1::Dims mOutputDims;
    std::vector<nvinfer1::DimsCHW> mFeatureDims;
    unsigned int mNbInputFeature;
    unsigned int mStimuliHeight;
    unsigned int mStimuliWidth;
    unsigned int mFeatureHeight;
    unsigned int mFeatureWidth;
    unsigned int mOutputWidth;
    unsigned int mOutputHeight;
    unsigned int mNbOutputs;
    unsigned int mNbProposals;
    Pooling_T mPoolType;
    size_t mSerializationSize;

};

/******************************************************************************/
/**Plugin Layer implementation**/
/**Proposal GPU implementation**/
class ProposalGPUPlugin: public nvinfer1::IPlugin
{
public:
	ProposalGPUPlugin(unsigned int batchSize,
                      unsigned int nbOutputs,
                      unsigned int outputHeight,
                      unsigned int outputWidth,
                      unsigned int nbProposals,
                      unsigned int nbCls,
                      double nmsIoU,
                      unsigned int scoreIndex,
                      double scoreThreshold,
                      unsigned int maxParts,
                      unsigned int maxTemplates,
                      const unsigned int* numPartsPerClass,
                      const unsigned int* numTemplatesPerClass,
                      const WDATA_T* means,
                      const WDATA_T* std,
                      bool applyNms,
                      bool keepMax,
                      double normX,
                      double normY)
	{
        mOutputDims.d[0] = batchSize;
        mOutputDims.d[1] = nbOutputs;
        mOutputDims.d[2] = outputHeight;
        mOutputDims.d[3] = outputWidth;

        mNbProposals = nbProposals;
        mNbCls = nbCls;
        mNMS_IoU = nmsIoU;
        mScoreIndex = scoreIndex;
        mScoreThreshold = scoreThreshold;
        mApplyNMS = applyNms;
        mKeepMax = keepMax;
        mNormX = normX;
        mNormY = normY;
        mMaxParts = maxParts;
        mMaxTemplates = maxTemplates;

        checkCudaErrors( cudaMalloc((void**)&mPartsPerClass,
                    mNbCls*sizeof(unsigned int)) );
        checkCudaErrors( cudaMalloc((void**)&mTemplatesPerClass,
                    mNbCls*sizeof(unsigned int)) );
        checkCudaErrors( cudaMemcpy(mPartsPerClass,
                         numPartsPerClass,
                         mNbCls*sizeof(unsigned int),
                         cudaMemcpyHostToDevice) );
        checkCudaErrors( cudaMemcpy(mTemplatesPerClass,
                         numTemplatesPerClass,
                         mNbCls*sizeof(unsigned int),
                         cudaMemcpyHostToDevice) );
        if(mMaxParts > 0)
        {
            checkCudaErrors( cudaMalloc((void**)&mPartsPrediction,
                                        2*mMaxParts*mNbCls*mNbProposals*mOutputDims.d[0]*sizeof(float)) );
            checkCudaErrors( cudaMalloc((void**)&mPartsVisibilityPrediction,
                                        mMaxParts*mNbCls*mNbProposals*mOutputDims.d[0]*sizeof(float)) );
        }

        if(mMaxTemplates > 0)
            checkCudaErrors( cudaMalloc((void**)&mTemplatesPrediction,
                                        3*mMaxTemplates*mNbCls*mNbProposals*mOutputDims.d[0]*sizeof(float)) );


            checkCudaErrors( cudaMalloc((void**)&mKeepIndex,
                                        2*mNbProposals*mOutputDims.d[0]*sizeof(int)) );

            checkCudaErrors( cudaMemset(mKeepIndex,
                                        -1,
                                        2*mNbProposals*mOutputDims.d[0]*sizeof(int)) );


        checkCudaErrors( cudaMalloc((void**)&mMeanGPU,
                         4*sizeof(DATA_T)) );

        checkCudaErrors( cudaMalloc((void**)&mStdGPU,
                         4*sizeof(DATA_T)) );

        checkCudaErrors( cudaMemcpy(mMeanGPU,
                         means,
                         4*sizeof(DATA_T),
                         cudaMemcpyHostToDevice) );

        checkCudaErrors( cudaMemcpy(mStdGPU,
                         std,
                         4*sizeof(DATA_T),
                         cudaMemcpyHostToDevice) );

        checkCudaErrors(cudaMalloc((void**)&mNormalizeROIs,
                                     mNbProposals*4*(mNbCls - mScoreIndex)*batchSize*sizeof(DATA_T)) );

        checkCudaErrors(cudaMalloc((void**)&mMaxCls,
                                     mNbProposals*batchSize*sizeof(int)) );
	}

	ProposalGPUPlugin(const void* dataFromRuntime, size_t length)
	{
		const char* d = reinterpret_cast<const char*>(dataFromRuntime), *a = d;
        mOutputDims.d[0] = read<int>(d);
		mOutputDims.d[1] = read<int>(d);
		mOutputDims.d[2] = read<int>(d);
		mOutputDims.d[3] = read<int>(d);
        mNbProposals = (unsigned int) read<int>(d);
        mNbCls = (unsigned int) read<int>(d);
        mNMS_IoU = read<double>(d);
        mScoreIndex = (unsigned int) read<int>(d);
        mScoreThreshold = (double) read<double>(d);
        mApplyNMS = (bool) read<bool>(d);
        mKeepMax = (bool) read<bool>(d);
        mNormX = (double) read<double>(d);
        mNormY = (double) read<double>(d);

        mMaxParts = read<unsigned int>(d);
        mMaxTemplates = read<unsigned int>(d);

        if((mMaxParts > 0))
            mPartsPerClass = deserializeToDevice<unsigned int>(d, mNbCls);

        if((mMaxTemplates > 0))
            mTemplatesPerClass = deserializeToDevice<unsigned int>(d, mNbCls);

        if((mMaxParts > 0))
        {
            mPartsPrediction = deserializeToDevice<float>(d, 2*mMaxParts*mNbCls*mNbProposals*mOutputDims.d[0]);
            mPartsVisibilityPrediction = deserializeToDevice<float>(d, mMaxParts*mNbCls*mNbProposals*mOutputDims.d[0]);
        }
        if((mMaxTemplates > 0))
            mTemplatesPrediction = deserializeToDevice<float>(d, 3*mMaxTemplates*mNbCls*mNbProposals*mOutputDims.d[0]);

        mKeepIndex = deserializeToDevice<int>(d, 2*mNbProposals*mOutputDims.d[0]);

        mMeanGPU = deserializeToDevice<float>(d, 4);
        mStdGPU = deserializeToDevice<float>(d, 4);
        mNormalizeROIs = deserializeToDevice<float>(d, mOutputDims.d[0]*mNbProposals*4*(mNbCls - mScoreIndex));
        mMaxCls = deserializeToDevice<int>(d, mOutputDims.d[0]*mNbProposals);

		assert(d == a + length);
	}

	~ProposalGPUPlugin()
	{

        checkCudaErrors(cudaFree(mPartsPerClass));
        checkCudaErrors(cudaFree(mTemplatesPerClass));
        checkCudaErrors(cudaFree(mPartsPrediction));
        checkCudaErrors(cudaFree(mPartsVisibilityPrediction));
        checkCudaErrors(cudaFree(mTemplatesPrediction));
        checkCudaErrors( cudaFree(mKeepIndex));

        checkCudaErrors(cudaFree(mMeanGPU));
        checkCudaErrors(cudaFree(mStdGPU));
        checkCudaErrors(cudaFree(mNormalizeROIs));
	}

	virtual int getNbOutputs() const override
	{
        return (int) 1;
	}

	virtual nvinfer1::Dims getOutputDimensions(int index,
                                               const nvinfer1::Dims* inputDim,
                                               int nbInputDims) override
	{
        ///Proposals are store through CHW format with C of size NbProposal*4 (4 for the ROI proposals coordinate)
        //return nvinfer1::DimsCHW(mNbProposals*mOutputDims.d[1], mOutputDims.d[2], mOutputDims.d[3]);
        return nvinfer1::DimsNCHW(mNbProposals, mOutputDims.d[1], mOutputDims.d[2], mOutputDims.d[3]);

	}

	virtual void configure(const nvinfer1::Dims* inputDims,
                   int nbInputs,
                   const nvinfer1::Dims* outputDims,
                   int nbOutputs,
                   int maxBatchSize) override
	{

	}

	virtual int initialize() override
	{
		return 0;
	}

	virtual void terminate() override
	{

	}

	virtual size_t getWorkspaceSize(int maxBatchSize) const override
	{
		return 0;
	}

	virtual int enqueue(int batchSize,
                        const void*const * inputs,
                        void** outputs,
                        void* workspace,
                        cudaStream_t stream) override
	{

        const unsigned int nbBlocks = std::ceil( (float)mNbProposals/ (float)32);
        const dim3 threadsPerBlock = {32, 1, 1};
        const dim3 blocksPerGrid = {(unsigned int) nbBlocks, 1, (unsigned int) batchSize};

        cuda_proposal_normalize( mOutputDims.d[0],
                                 mOutputDims.d[1],
                                 mNbProposals,
                                 batchSize,
                                 mScoreIndex,
                                 mNbCls,
                                 mMaxParts,
                                 mMaxTemplates,
                                 mKeepMax,
                                 mMaxParts > 0 ? true : false,
                                 mMaxTemplates > 0 ? true : false,
                                 mNormX,
                                 mNormY,
                                 reinterpret_cast<const DATA_T *>(mMeanGPU),
                                 reinterpret_cast<const DATA_T *>(mStdGPU),
                                 reinterpret_cast<const unsigned int *>(mPartsPerClass),
                                 reinterpret_cast<const unsigned int *>(mTemplatesPerClass),
                                 reinterpret_cast<const DATA_T *>(inputs[0]),
                                 reinterpret_cast<const DATA_T*>(inputs[1]),
                                 reinterpret_cast<const DATA_T*>(inputs[2]),
                                 (mMaxParts > 0 || mMaxTemplates > 0)  ?
                                    reinterpret_cast<const DATA_T*>(inputs[3])
                                    : reinterpret_cast<const DATA_T*>(inputs[2]),
                                 (mMaxParts > 0) ?
                                    reinterpret_cast<const DATA_T*>(inputs[4])
                                    : reinterpret_cast<const DATA_T*>(inputs[2]),
                                 (mMaxParts > 0 && mMaxTemplates > 0) ?
                                    reinterpret_cast<const DATA_T*>(inputs[5])
                                    : reinterpret_cast<const DATA_T*>(inputs[2]),
                                 reinterpret_cast<DATA_T *>(mNormalizeROIs),
                                 reinterpret_cast<int *>(mMaxCls),
                                 reinterpret_cast<DATA_T *>(mPartsPrediction),
                                 reinterpret_cast<DATA_T *>(mPartsVisibilityPrediction),
                                 reinterpret_cast<DATA_T *>(mTemplatesPrediction),
                                 mScoreThreshold,
                                 threadsPerBlock,
                                 blocksPerGrid);

        int* cpuArgMax = new int[mOutputDims.d[0]*mNbProposals];

        CHECK_CUDA_STATUS(  cudaMemcpy(cpuArgMax,
                            reinterpret_cast<int*>(mMaxCls),
                            mOutputDims.d[0]*mNbProposals*sizeof(int),
                            cudaMemcpyDeviceToHost));

        if(mApplyNMS)
        {
            float* cpuROIs = new float[mOutputDims.d[0]*mNbProposals*4*(mNbCls - mScoreIndex)];
            CHECK_CUDA_STATUS(  cudaMemcpy(cpuROIs,
                                reinterpret_cast<float*>(mNormalizeROIs),
                                mOutputDims.d[0]*mNbProposals*4*(mNbCls - mScoreIndex)*sizeof(float),
                                cudaMemcpyDeviceToHost));

            int* cpuKeepIndex = new int[2*mNbProposals*mOutputDims.d[0]];
            CHECK_CUDA_STATUS(  cudaMemcpy(cpuKeepIndex,
                                reinterpret_cast<int*>(mKeepIndex),
                                2*mNbProposals*mOutputDims.d[0]*sizeof(int),
                                cudaMemcpyDeviceToHost));

            // Non-Maximum Suppression (NMS)
            for(unsigned int n = 0; n < mOutputDims.d[0]; ++n)
            {
                for(unsigned int cls = 0; cls < (mNbCls - mScoreIndex) ; ++cls)
                {
                    for (unsigned int i = 0; i < mNbProposals - 1;
                        ++i)
                    {
                        const unsigned int iIdx
                                        = 4*cls
                                            + 4*i*(mNbCls - mScoreIndex)
                                            + n*4*mNbProposals*(mNbCls - mScoreIndex);

                        const float x0 = cpuROIs[0 + iIdx];
                        const float y0 = cpuROIs[1 + iIdx];
                        const float w0 = cpuROIs[2 + iIdx];
                        const float h0 = cpuROIs[3 + iIdx];

                        for (unsigned int j = i + 1; j < mNbProposals; ) {
                            const unsigned int jIdx
                                            = 4*cls
                                                + 4*j*(mNbCls - mScoreIndex)
                                                + n*4*mNbProposals*(mNbCls - mScoreIndex);

                            const float x = cpuROIs[0 + jIdx];
                            const float y = cpuROIs[1 + jIdx];
                            const float w = cpuROIs[2 + jIdx];
                            const float h = cpuROIs[3 + jIdx];

                            const float interLeft = std::max(x0, x);
                            const float interRight = std::min(x0 + w0, x + w);
                            const float interTop = std::max(y0, y);
                            const float interBottom = std::min(y0 + h0, y + h);

                            if (interLeft < interRight && interTop < interBottom) {
                                const float interArea = (interRight - interLeft)
                                                            * (interBottom - interTop);
                                const float unionArea = w0 * h0 + w * h - interArea;
                                const float IoU = interArea / unionArea;

                                if (IoU > mNMS_IoU) {

                                    // Suppress ROI
                                    cpuROIs[0 + jIdx] = 0.0;
                                    cpuROIs[1 + jIdx] = 0.0;
                                    cpuROIs[2 + jIdx] = 0.0;
                                    cpuROIs[3 + jIdx] = 0.0;
                                    continue;
                                }
                            }
                            ++j;
                        }
                    }
                }

                unsigned int out = 0;

                for(unsigned int cls = 0; cls < (mNbCls - mScoreIndex)
                        && out < mNbProposals; ++cls)
                {
                    for (unsigned int i = 0; i < mNbProposals && out < mNbProposals ;
                        ++i)
                    {
                        //Read before erase and write
                        const unsigned int idx
                                        = 4*cls
                                            + 4*i*(mNbCls - mScoreIndex)
                                            + n*4*mNbProposals*(mNbCls - mScoreIndex);

                        const float x = cpuROIs[0 + idx];
                        const float y = cpuROIs[1 + idx];
                        const float w = cpuROIs[2 + idx];
                        const float h = cpuROIs[3 + idx];


                        if(w > 0.0 && h > 0.0)
                        {
                            const unsigned int outIdx = 4*out*(mNbCls - mScoreIndex)
                                                        + n*4*mNbProposals*(mNbCls - mScoreIndex);

                            //Erase before write
                            cpuROIs[0 + idx] = 0.0;
                            cpuROIs[1 + idx] = 0.0;
                            cpuROIs[2 + idx] = 0.0;
                            cpuROIs[3 + idx] = 0.0;

                            //Write result
                            cpuROIs[0 + outIdx] = x;
                            cpuROIs[1 + outIdx] = y;
                            cpuROIs[2 + outIdx] = w;
                            cpuROIs[3 + outIdx] = h;

                            cpuArgMax[out + n*mNbProposals] = (int) (cls + mScoreIndex);
                            cpuKeepIndex[out*2 + 0] = i;
                            cpuKeepIndex[out*2 + 1] = cls + mScoreIndex;

                            ++out;
                        }
                    }
                }
            }

            CHECK_CUDA_STATUS(cudaMemcpy( mNormalizeROIs,
                                          cpuROIs,
                                          mOutputDims.d[0]*mNbProposals*4*(mNbCls - mScoreIndex)*sizeof(float),
                                          cudaMemcpyHostToDevice));

            CHECK_CUDA_STATUS(cudaMemcpy( mKeepIndex,
                                          cpuKeepIndex,
                                          2*mNbProposals*mOutputDims.d[0]*sizeof(int),
                                          cudaMemcpyHostToDevice));

            delete[] cpuROIs;
            delete[] cpuKeepIndex;

        }

        CHECK_CUDA_STATUS(  cudaMemcpy(mMaxCls,
                            cpuArgMax,
                            mOutputDims.d[0]*mNbProposals*sizeof(int),
                            cudaMemcpyHostToDevice));

        delete[] cpuArgMax;

        cuda_proposal_to_output( mNbProposals,
                                 mScoreIndex,
                                 mNbCls,
                                 mOutputDims.d[3]*mOutputDims.d[2]*mOutputDims.d[1],
                                 mMaxParts,
                                 mMaxTemplates,
                                 mMaxParts > 0 ? true : false,
                                 mMaxTemplates > 0 ? true : false,
                                 reinterpret_cast<const unsigned int *>(mPartsPerClass),
                                 reinterpret_cast<const unsigned int *>(mTemplatesPerClass),
                                 reinterpret_cast<const int *>(mMaxCls),
                                 reinterpret_cast<const DATA_T *>(mNormalizeROIs),
                                 reinterpret_cast<const int *>(mKeepIndex),
                                 reinterpret_cast<const DATA_T *>(mPartsPrediction),
                                 reinterpret_cast<const DATA_T *>(mPartsVisibilityPrediction),
                                 reinterpret_cast<const DATA_T *>(mTemplatesPrediction),
                                 reinterpret_cast<DATA_T *>(outputs[0]),
                                 threadsPerBlock,
                                 blocksPerGrid);

        return 0;
	}

	virtual size_t getSerializationSize() override
	{
        size_t inputDimParamSize = 4*sizeof(int); //
        size_t proposalParamI = 3*sizeof(int) + sizeof(bool)*2; //
        size_t proposalParamD = 4*sizeof(double); //

        size_t proposalParamGPU = 4*2*sizeof(float)
                                + 4*(mNbCls - mScoreIndex)*mNbProposals*mOutputDims.d[0]*sizeof(float)
                                + mNbProposals*mOutputDims.d[0]*sizeof(int);
        size_t partSize = (mMaxParts == 0) ? 1*sizeof(unsigned int)
                            : 1*sizeof(unsigned int)
                                +  mNbCls*sizeof(unsigned int)
                                +  3*mMaxParts*mNbCls*mNbProposals*mOutputDims.d[0]*sizeof(float);

        size_t templatesSize = (mMaxTemplates == 0) ? 1*sizeof(unsigned int)
                            :  1*sizeof(unsigned int)
                                +  mNbCls*sizeof(unsigned int)
                                +  3*mMaxTemplates*mNbCls*mNbProposals*mOutputDims.d[0]*sizeof(float);

        size_t keepIndexSize = 2*mNbProposals*mOutputDims.d[0]*sizeof(int);

        mSerializationSize = inputDimParamSize + proposalParamI
                                + proposalParamD
                                + proposalParamGPU
                                + partSize
                                + templatesSize
                                + keepIndexSize;

        return mSerializationSize;
	}

	virtual void serialize(void* buffer) override
	{
        char* d = reinterpret_cast<char*>(buffer);
        char* a = d;
        write<int>(d, (int)mOutputDims.d[0]);
        write<int>(d, (int)mOutputDims.d[1]);
        write<int>(d, (int)mOutputDims.d[2]);
        write<int>(d, (int)mOutputDims.d[3]);
        write<int>(d, (int)mNbProposals);
        write<int>(d, (int)mNbCls);
        write<double>(d, mNMS_IoU);
        write<int>(d, (int)mScoreIndex);
        write<double>(d, (double)mScoreThreshold);
        write<bool>(d, (bool)mApplyNMS);
        write<bool>(d, (bool) mKeepMax);
        write<double>(d, (double)mNormX);
        write<double>(d, (double)mNormY);

        write<unsigned int>(d, mMaxParts);
        write<unsigned int>(d, mMaxTemplates);

        if((mMaxParts > 0))
            serializeFromDevice<unsigned int>(d, mPartsPerClass, mNbCls);

        if((mMaxTemplates > 0))
            serializeFromDevice<unsigned int>(d, mTemplatesPerClass, mNbCls);

        if((mMaxParts > 0))
        {
            serializeFromDevice<float>(d, mPartsPrediction, 2*mMaxParts*mNbCls*mNbProposals*mOutputDims.d[0]);
            serializeFromDevice<float>(d, mPartsVisibilityPrediction, mMaxParts*mNbCls*mNbProposals*mOutputDims.d[0]);
        }
        if((mMaxTemplates > 0))
            serializeFromDevice<float>(d, mTemplatesPrediction, 3*mMaxTemplates*mNbCls*mNbProposals*mOutputDims.d[0]);

        serializeFromDevice<int>(d, mKeepIndex, 2*mNbProposals*mOutputDims.d[0]);

        serializeFromDevice<float>(d, mMeanGPU, 4);
        serializeFromDevice<float>(d, mStdGPU, 4);
        serializeFromDevice<float>(d, mNormalizeROIs, mNbProposals*mOutputDims.d[0]*4*(mNbCls - mScoreIndex));
        serializeFromDevice<int>(d, mMaxCls, mNbProposals*mOutputDims.d[0]);

        assert(d == a + getSerializationSize());
	}

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

    template<typename T>
    T* deserializeToDevice(const char*& hostBuffer, size_t dataSize)
    {
        T* gpuData;
        checkCudaErrors(cudaMalloc(&gpuData, dataSize*sizeof(T)));
        checkCudaErrors(cudaMemcpy(gpuData, hostBuffer, dataSize*sizeof(T), cudaMemcpyHostToDevice));
        hostBuffer += dataSize*sizeof(T);
        return gpuData;
    }

    template<typename T>
    void serializeFromDevice(char*& hostBuffer, T* deviceWeights, size_t dataSize)
    {
        checkCudaErrors(cudaMemcpy(hostBuffer, deviceWeights, dataSize*sizeof(T), cudaMemcpyDeviceToHost));
        hostBuffer += dataSize*sizeof(T);
    }

    nvinfer1::Dims mOutputDims;
    unsigned int mNbProposals;
    unsigned int mNbCls;
    double mNMS_IoU;
    unsigned int mScoreIndex;
    double mScoreThreshold;
    double mNormX;
    double mNormY;
    unsigned int mMaxParts;
    unsigned int mMaxTemplates;
    unsigned int* mPartsPerClass;
    unsigned int* mTemplatesPerClass;
    float* mPartsPrediction;
    float* mPartsVisibilityPrediction;
    float* mTemplatesPrediction;
    int* mKeepIndex;
    float* mMeanGPU;
    float* mStdGPU;
    float* mNormalizeROIs;
    int* mMaxCls;
    bool mApplyNMS;
    bool mKeepMax;
    size_t mSerializationSize;
};


/******************************************************************************/
/**Plugin Layer implementation**/
/**RegionProposal GPU implementation**/
class RegionProposalGPUPlugin: public nvinfer1::IPlugin
{
public:
	RegionProposalGPUPlugin(unsigned int batchSize,
                            unsigned int nbOutputs,
                            unsigned int outputHeight,
                            unsigned int outputWidth,
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
        mOutputDims.d[0] = batchSize;
        mOutputDims.d[1] = nbOutputs;
        mOutputDims.d[2] = outputHeight;
        mOutputDims.d[3] = outputWidth;
        mChannelHeight = channelHeight;
        mChannelWidth = channelWidth;
        mNbAnchors = nbAnchors;
        mNbProposals = nbProposals;
        mPreNMsTopN = preNMsTopN;

        mNMS_IoU = nmsIoU;
        mMinHeight = minHeight;
        mMinWidth = minWidth;
        mScoreIndex = scoreIndex;
        mIoUIndex = iouIndex;

        mInputMaxSize = mNbAnchors*mChannelHeight*channelWidth;
        mSortSize = mInputMaxSize;
        if(mPreNMsTopN > 0 && mPreNMsTopN < mInputMaxSize)
            mSortSize = mPreNMsTopN;

        mNbThreadsNMS = sizeof(unsigned long long) * 8;
        mNbBlocksNMS = DIVUP(mSortSize, mNbThreadsNMS), DIVUP(mSortSize, mNbThreadsNMS);

        checkCudaErrors(cudaMalloc((void**)&mValues,
                            mOutputDims.d[0]*mInputMaxSize*sizeof(DATA_T)) );
        checkCudaErrors(cudaMalloc((void**)&mIndexI,
                            mOutputDims.d[0]*mInputMaxSize*sizeof(DATA_T)) );
        checkCudaErrors(cudaMalloc((void**)&mIndexJ,
                            mOutputDims.d[0]*mInputMaxSize*sizeof(DATA_T)) );
        checkCudaErrors(cudaMalloc((void**)&mIndexK,
                            mOutputDims.d[0]*mInputMaxSize*sizeof(DATA_T)) );
        checkCudaErrors(cudaMalloc((void**)&mIndexB,
                            mOutputDims.d[0]*mInputMaxSize*sizeof(DATA_T)) );
        checkCudaErrors(cudaMalloc((void**)&mMap,
                            mOutputDims.d[0]*mInputMaxSize*sizeof(unsigned int)) );

        checkCudaErrors(cudaMalloc((void**)&mSortedIndexI,
                            mOutputDims.d[0]*mSortSize*sizeof(DATA_T)) );
        checkCudaErrors(cudaMalloc((void**)&mSortedIndexJ,
                            mOutputDims.d[0]*mSortSize*sizeof(DATA_T)) );
        checkCudaErrors(cudaMalloc((void**)&mSortedIndexK,
                            mOutputDims.d[0]*mSortSize*sizeof(DATA_T)) );
        checkCudaErrors(cudaMalloc((void**)&mSortedIndexB,
                            mOutputDims.d[0]*mSortSize*sizeof(DATA_T)) );

        checkCudaErrors(cudaMalloc((void**)&mMask,
                            mOutputDims.d[0]*mSortSize*mNbBlocksNMS*sizeof(unsigned long long)) );
        checkCudaErrors(cudaMalloc((void**)&mSortedIndex,
                            mOutputDims.d[0]*mSortSize*sizeof(int)) );
	}

	RegionProposalGPUPlugin(const void* dataFromRuntime, size_t length)
	{
		const char* d = reinterpret_cast<const char*>(dataFromRuntime), *a = d;
        mOutputDims.d[0] = read<int>(d);
		mOutputDims.d[1] = read<int>(d);
		mOutputDims.d[2] = read<int>(d);
		mOutputDims.d[3] = read<int>(d);
        mChannelHeight = read<int>(d);
        mChannelWidth = read<int>(d);
        mNbProposals = (unsigned int) read<int>(d);
        mNbAnchors = (unsigned int) read<int>(d);
        mPreNMsTopN = (unsigned int) read<int>(d);
        mNMS_IoU = read<double>(d);
        mMinWidth = read<double>(d);
        mMinHeight = read<double>(d);
        mScoreIndex = (unsigned int) read<int>(d);
        mIoUIndex = (unsigned int) read<int>(d);

        mInputMaxSize = (unsigned int) read<int>(d);
        mSortSize = (unsigned int) read<int>(d);
        mNbThreadsNMS = (unsigned int) read<int>(d);
        mNbBlocksNMS = (unsigned int) read<int>(d);

		mValues = deserializeToDevice<float>(d,
                        mInputMaxSize*mOutputDims.d[0]);
        mIndexI = deserializeToDevice<float>(d,
                        mInputMaxSize*mOutputDims.d[0]);
        mIndexJ = deserializeToDevice<float>(d,
                        mInputMaxSize*mOutputDims.d[0]);
        mIndexK = deserializeToDevice<float>(d,
                        mInputMaxSize*mOutputDims.d[0]);
        mIndexB = deserializeToDevice<float>(d,
                        mInputMaxSize*mOutputDims.d[0]);
        mMap = deserializeToDevice<unsigned int>(d,
                        mInputMaxSize*mOutputDims.d[0]);

        mSortedIndexI = deserializeToDevice<float>(d,
                            mSortSize*mOutputDims.d[0]);
        mSortedIndexJ = deserializeToDevice<float>(d,
                            mSortSize*mOutputDims.d[0]);
        mSortedIndexK = deserializeToDevice<float>(d,
                            mSortSize*mOutputDims.d[0]);
        mSortedIndexB = deserializeToDevice<float>(d,
                            mSortSize*mOutputDims.d[0]);

        mMask = deserializeToDevice<unsigned long long>(d,
                            mSortSize*mNbBlocksNMS*mOutputDims.d[0]);
        mSortedIndex = deserializeToDevice<int>(d,
                            mSortSize*mOutputDims.d[0]);

		assert(d == a + length);
	}

	~RegionProposalGPUPlugin()
	{
        checkCudaErrors(cudaFree(mValues));
        checkCudaErrors(cudaFree(mIndexI));
        checkCudaErrors(cudaFree(mIndexJ));
        checkCudaErrors(cudaFree(mIndexK));
        checkCudaErrors(cudaFree(mIndexB));
        checkCudaErrors(cudaFree(mMap));
        checkCudaErrors(cudaFree(mSortedIndexI));
        checkCudaErrors(cudaFree(mSortedIndexJ));
        checkCudaErrors(cudaFree(mSortedIndexK));
        checkCudaErrors(cudaFree(mSortedIndexB));
        checkCudaErrors(cudaFree(mMask));
        checkCudaErrors(cudaFree(mSortedIndex));

	}

	virtual int getNbOutputs() const override
	{
        //return (int)mNbProposals;
        return (int) 1;

	}

	virtual nvinfer1::Dims getOutputDimensions(int index,
                                               const nvinfer1::Dims* inputDim,
                                               int nbInputDims) override
	{
        //return nvinfer1::DimsCHW(mOutputDims.d[1], mOutputDims.d[2], mOutputDims.d[3]);
        ///Proposals are store through CHW format with C of size NbProposal*4 (4 for the ROI proposals coordinate)
        /*unsigned int batchInput = 1;

        if(inputDim[0].nbDims == 4)
            batchInput = inputDim[0].d[0];*/

        //return nvinfer1::DimsNCHW(batchInput, mNbProposals*mOutputDims.d[1], mOutputDims.d[2], mOutputDims.d[3]);
        return nvinfer1::DimsNCHW(mNbProposals, mOutputDims.d[1], mOutputDims.d[2], mOutputDims.d[3]);

	}

	virtual void configure(const nvinfer1::Dims* inputDims,
                   int nbInputs,
                   const nvinfer1::Dims* outputDims,
                   int nbOutputs,
                   int maxBatchSize) override
	{

	}

	virtual int initialize() override
	{
		return 0;
	}

	virtual void terminate() override
	{

	}

	virtual size_t getWorkspaceSize(int maxBatchSize) const override
	{
		return 0;
	}

	virtual int enqueue(int batchSize,
                        const void*const * inputs,
                        void** outputs,
                        void* workspace,
                        cudaStream_t stream) override
	{

        const unsigned int nbBlocks = std::ceil(mInputMaxSize/32.0);

        /**Reorder i,j,k,b index and create the map vector to allow a fast gpu sorting using thrust**/
        cuda_region_proposal_split_indexes( mChannelWidth,
                                            mChannelHeight,
                                            mNbAnchors,
                                            batchSize,
                                            nbBlocks,
                                            reinterpret_cast<const DATA_T *>(inputs[0]),
                                            reinterpret_cast<DATA_T *>(mValues),
                                            reinterpret_cast<DATA_T *>(mIndexI),
                                            reinterpret_cast<DATA_T *>(mIndexJ),
                                            reinterpret_cast<DATA_T *>(mIndexK),
                                            reinterpret_cast<DATA_T *>(mIndexB),
                                            reinterpret_cast<unsigned int *>(mMap),
                                            mMinWidth,
                                            mMinHeight,
                                            mScoreIndex);

        for(unsigned int n = 0; n < batchSize; ++n)
        {
            unsigned int inputOffset = mInputMaxSize*n;
            unsigned int outputOffset = mSortSize*n;

            thrust_sort_keys(   reinterpret_cast<DATA_T *>(mValues),
                                reinterpret_cast<unsigned int *>(mMap),
                                mInputMaxSize,
                                inputOffset);

            thrust_gather(  reinterpret_cast<unsigned int *>(mMap),
                            reinterpret_cast<DATA_T *>(mIndexI),
                            reinterpret_cast<DATA_T *>(mSortedIndexI),
                            mSortSize,
                            inputOffset,
                            outputOffset);

            thrust_gather(  reinterpret_cast<unsigned int *>(mMap),
                            reinterpret_cast<DATA_T *>(mIndexJ),
                            reinterpret_cast<DATA_T *>(mSortedIndexJ),
                            mSortSize,
                            inputOffset,
                            outputOffset);

            thrust_gather(  reinterpret_cast<unsigned int *>(mMap),
                            reinterpret_cast<DATA_T *>(mIndexK),
                            reinterpret_cast<DATA_T *>(mSortedIndexK),
                            mSortSize,
                            inputOffset,
                            outputOffset);

            thrust_gather(  reinterpret_cast<unsigned int *>(mMap),
                            reinterpret_cast<DATA_T *>(mIndexB),
                            reinterpret_cast<DATA_T *>(mSortedIndexB),
                            mSortSize,
                            inputOffset,
                            outputOffset);
        }


        const int col_blocks = DIVUP(mSortSize, mNbThreadsNMS);

        dim3 blocks(mNbBlocksNMS, mNbBlocksNMS);
        dim3 threads(mNbThreadsNMS);

        for(unsigned int n = 0; n <batchSize; ++n)
        {
            //unsigned int inputOffset = n*mInputs[0].dimX()*mInputs[0].dimY()*mInputs[0].dimZ();
            unsigned int indexOffset = n*mSortSize;
            unsigned int outputOffset = n*mInputMaxSize;

            cuda_region_proposal_nms(   mChannelWidth,
                                        mChannelHeight,
                                        mNbAnchors,
                                        1,
                                        reinterpret_cast<const DATA_T *>(inputs[0]),
                                        reinterpret_cast<DATA_T *>(mSortedIndexI),
                                        reinterpret_cast<DATA_T *>(mSortedIndexJ),
                                        reinterpret_cast<DATA_T *>(mSortedIndexK),
                                        reinterpret_cast<DATA_T *>(mSortedIndexB),
                                        indexOffset,
                                        reinterpret_cast<unsigned long long *>(mMask),
                                        outputOffset,
                                        mNMS_IoU,
                                        mSortSize,
                                        threads,
                                        blocks);
        }
        std::vector<std::vector<unsigned long long> > remv(batchSize,
                                                           std::vector<unsigned long long>(col_blocks, 0));

        int* cpuSortIndex = new int[mSortSize * batchSize];
        unsigned long long* cpuMask = new unsigned long long[mSortSize*mNbBlocksNMS*batchSize];

        CHECK_CUDA_STATUS(cudaMemcpy(cpuMask,
                                     reinterpret_cast<unsigned long long*>(mMask),
                                     batchSize*mSortSize*mNbBlocksNMS*sizeof(unsigned long long),
                                     cudaMemcpyDeviceToHost));

        unsigned int batchNumtoKeep = 0;
        for(unsigned int n = 0; n < batchSize; ++n)
        {
            int num_to_keep = 0;
            const unsigned int sortOffset = n*mSortSize;
            const unsigned int maskOffset = n*mInputMaxSize;

            for (int i = 0; i < mSortSize; i++)
            {
                int nblock = i / mNbThreadsNMS;
                int inblock = i % mNbThreadsNMS;

                if (!(remv[n][nblock] & (1ULL << inblock)))
                {
                    cpuSortIndex[num_to_keep + sortOffset] = i;
                    num_to_keep++;

                    unsigned long long *p = &cpuMask[0] + i * col_blocks + maskOffset;

                    for (int j = nblock; j < col_blocks; j++)
                    {
                        remv[n][j] |= p[j];
                    }
                }
            }

            batchNumtoKeep += num_to_keep;
        }

        CHECK_CUDA_STATUS(cudaMemcpy(   mSortedIndex,
                                        cpuSortIndex,
                                        mSortSize*batchSize*sizeof(int),
                                        cudaMemcpyHostToDevice));

        cuda_region_proposal_gathering( mChannelWidth,
                                        mChannelHeight,
                                        mNbAnchors,
                                        batchSize,
                                        reinterpret_cast<const DATA_T *>(inputs[0]),
                                        reinterpret_cast<const DATA_T *>(mSortedIndexI),
                                        reinterpret_cast<const DATA_T *>(mSortedIndexJ),
                                        reinterpret_cast<const DATA_T *>(mSortedIndexK),
                                        reinterpret_cast<const DATA_T *>(mSortedIndexB),
                                        reinterpret_cast<const int *>(mSortedIndex),
                                        reinterpret_cast<DATA_T *>(outputs[0]),
                                        mSortSize,
                                        mNbProposals,
                                        (unsigned int) std::ceil(mNbProposals/(float)32));


        delete[] cpuSortIndex;
        delete[] cpuMask;
        return 0;
	}

	virtual size_t getSerializationSize() override
	{
        size_t inputDimParamSize = 4*sizeof(int); //
        size_t proposalParamI = 7*sizeof(int); //
        size_t proposalParamD = 3*sizeof(double); //RatioX and RatioY

        size_t proposalParamGPU = 4*sizeof(int)
                                    + mOutputDims.d[0]*5*mInputMaxSize*sizeof(float)
                                    + mOutputDims.d[0]*mInputMaxSize*sizeof(unsigned int)
                                    + mOutputDims.d[0]*4*mSortSize*sizeof(float)
                                    + mOutputDims.d[0]*mSortSize*mNbBlocksNMS*sizeof(unsigned long long)
                                    + mOutputDims.d[0]*mSortSize*sizeof(int);

        mSerializationSize = inputDimParamSize + proposalParamI
                                + proposalParamD
                                + proposalParamGPU;

        return mSerializationSize;
	}

	virtual void serialize(void* buffer) override
	{
        char* d = reinterpret_cast<char*>(buffer);
        char* a = d;
        write<int>(d, (int)mOutputDims.d[0]);
        write<int>(d, (int)mOutputDims.d[1]);
        write<int>(d, (int)mOutputDims.d[2]);
        write<int>(d, (int)mOutputDims.d[3]);
        write<int>(d, (int)mChannelHeight);
        write<int>(d, (int)mChannelWidth);
        write<int>(d, (int)mNbProposals);
        write<int>(d, (int)mNbAnchors);
        write<int>(d, (int)mPreNMsTopN);
        write<double>(d, mNMS_IoU);
        write<double>(d, mMinWidth);
        write<double>(d, mMinHeight);
        write<int>(d, (int)mScoreIndex);
        write<int>(d, (int)mIoUIndex);

        write<int>(d, (int)mInputMaxSize);
        write<int>(d, (int)mSortSize);
        write<int>(d, (int)mNbThreadsNMS);
        write<int>(d, (int)mNbBlocksNMS);

        serializeFromDevice<float>(d, mValues, mInputMaxSize*mOutputDims.d[0]);
        serializeFromDevice<float>(d, mIndexI, mInputMaxSize*mOutputDims.d[0]);
        serializeFromDevice<float>(d, mIndexJ, mInputMaxSize*mOutputDims.d[0]);
        serializeFromDevice<float>(d, mIndexK, mInputMaxSize*mOutputDims.d[0]);
        serializeFromDevice<float>(d, mIndexB, mInputMaxSize*mOutputDims.d[0]);
        serializeFromDevice<unsigned int>(d, mMap, mInputMaxSize*mOutputDims.d[0]);

        serializeFromDevice<float>(d, mSortedIndexI, mSortSize*mOutputDims.d[0]);
        serializeFromDevice<float>(d, mSortedIndexJ, mSortSize*mOutputDims.d[0]);
        serializeFromDevice<float>(d, mSortedIndexK, mSortSize*mOutputDims.d[0]);
        serializeFromDevice<float>(d, mSortedIndexB, mSortSize*mOutputDims.d[0]);

        serializeFromDevice<unsigned long long>(d, mMask, mSortSize*mNbBlocksNMS*mOutputDims.d[0]);
        serializeFromDevice<int>(d, mSortedIndex, mSortSize*mOutputDims.d[0]);

        assert(d == a + getSerializationSize());
	}

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

    template<typename T>
    T* deserializeToDevice(const char*& hostBuffer, size_t dataSize)
    {
        T* gpuData;
        checkCudaErrors(cudaMalloc(&gpuData, dataSize*sizeof(T)));
        checkCudaErrors(cudaMemcpy(gpuData, hostBuffer, dataSize*sizeof(T), cudaMemcpyHostToDevice));
        hostBuffer += dataSize*sizeof(T);
        return gpuData;
    }

    template<typename T>
    void serializeFromDevice(char*& hostBuffer, T* deviceWeights, size_t dataSize)
    {
        checkCudaErrors(cudaMemcpy(hostBuffer, deviceWeights, dataSize*sizeof(T), cudaMemcpyDeviceToHost));
        hostBuffer += dataSize*sizeof(T);
    }

    nvinfer1::Dims mOutputDims;
    unsigned int mChannelHeight;
    unsigned int mChannelWidth;
    unsigned int mNbProposals;
    unsigned int mNbAnchors;
    unsigned int mPreNMsTopN;
    unsigned int mScoreIndex;
    unsigned int mIoUIndex;
    double mNMS_IoU;
    double mMinWidth;
    double mMinHeight;
    unsigned int mInputMaxSize;
    unsigned int mSortSize;
    float* mIndexI;
    float* mSortedIndexI;
    float* mIndexJ;
    float* mSortedIndexJ;
    float* mIndexK;
    float* mSortedIndexK;
    float* mIndexB;
    float* mSortedIndexB;
    float* mValues;
    int* mSortedIndex;
    unsigned int* mMap;
    unsigned long long* mMask;
    unsigned int mNbThreadsNMS;
    unsigned int mNbBlocksNMS;
    size_t mSerializationSize;

};


/******************************************************************************/
/**Plugin Layer implementation**/
/**RegionProposal CPU implementation**/
class RegionProposalCPUPlugin: public nvinfer1::IPlugin
{
public:
	RegionProposalCPUPlugin(unsigned int batchSize,
                    unsigned int nbOutputs,
                    unsigned int outputHeight,
                    unsigned int outputWidth,
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
        mOutputDims.d[0] = batchSize;
        mOutputDims.d[1] = nbOutputs;
        mOutputDims.d[2] = outputHeight;
        mOutputDims.d[3] = outputWidth;
        mChannelHeight = channelHeight;
        mChannelWidth = channelWidth;
        mNbAnchors = nbAnchors;
        mNbProposals = nbProposals;
        mPreNMsTopN = preNMsTopN;

        mNMS_IoU = nmsIoU;
        mMinHeight = minHeight;
        mMinWidth = minWidth;
        mScoreIndex = scoreIndex;
        mIoUIndex = iouIndex;
	}

	RegionProposalCPUPlugin(const void* dataFromRuntime, size_t length)
	{
		const char* d = reinterpret_cast<const char*>(dataFromRuntime), *a = d;
        mOutputDims.d[0] = read<int>(d);
		mOutputDims.d[1] = read<int>(d);
		mOutputDims.d[2] = read<int>(d);
		mOutputDims.d[3] = read<int>(d);
        mChannelHeight = read<int>(d);
        mChannelWidth = read<int>(d);
        mNbProposals = (unsigned int) read<int>(d);
        mNbAnchors = (unsigned int) read<int>(d);
        mPreNMsTopN = (unsigned int) read<int>(d);
        mNMS_IoU = read<double>(d);
        mMinWidth = read<double>(d);
        mMinHeight = read<double>(d);
        mScoreIndex = (unsigned int) read<int>(d);
        mIoUIndex = (unsigned int) read<int>(d);
		assert(d == a + length);
	}

	~RegionProposalCPUPlugin()
	{

	}

	virtual int getNbOutputs() const override
	{
        //return (int)mNbProposals;
        return (int) 1;

	}

	virtual nvinfer1::Dims getOutputDimensions(int index,
                                               const nvinfer1::Dims* inputDim,
                                               int nbInputDims) override
	{
        //return nvinfer1::DimsCHW(mOutputDims.d[1], mOutputDims.d[2], mOutputDims.d[3]);
        ///Proposals are store through CHW format with C of size NbProposal*4 (4 for the ROI proposals coordinate)
        unsigned int batchInput = 1;

        if(inputDim[0].nbDims == 4)
            batchInput = inputDim[0].d[0];

        //return nvinfer1::DimsNCHW(batchInput, mNbProposals*mOutputDims.d[1], mOutputDims.d[2], mOutputDims.d[3]);
        return nvinfer1::DimsNCHW(mNbProposals, mOutputDims.d[1], mOutputDims.d[2], mOutputDims.d[3]);

	}

	virtual void configure(const nvinfer1::Dims* inputDims,
                   int nbInputs,
                   const nvinfer1::Dims* outputDims,
                   int nbOutputs,
                   int maxBatchSize) override
	{

	}

	virtual int initialize() override
	{
		return 0;
	}

	virtual void terminate() override
	{

	}

	virtual size_t getWorkspaceSize(int maxBatchSize) const override
	{
		return 0;
	}

	virtual int enqueue(int batchSize,
                        const void*const * inputs,
                        void** outputs,
                        void* workspace,
                        cudaStream_t stream) override
	{
        DATA_T* inputDataCPU(NULL);
        DATA_T* outputDataCPU(NULL);
        size_t size_input_cpy = 6*mNbAnchors
                                 *mChannelHeight*mChannelWidth*batchSize;
        size_t size_output_cpy = mNbProposals*mOutputDims.d[1]
                                    *mOutputDims.d[2]*mOutputDims.d[3];

        inputDataCPU = new DATA_T[size_input_cpy];
        outputDataCPU = new DATA_T[size_output_cpy*batchSize];

        if (!inputDataCPU)
            throw std::runtime_error(
                "enqueue(): could not allocate memory");

        if (!outputDataCPU)
            throw std::runtime_error(
                "enqueue(): could not allocate memory");

        CHECK_CUDA_STATUS(cudaMemcpy(inputDataCPU,
                                     reinterpret_cast<const float*>(inputs[0]),
                                     size_input_cpy*sizeof(DATA_T),
                                     cudaMemcpyDeviceToHost));

        CHECK_CUDA_STATUS(cudaStreamSynchronize(stream));

        region_proposal_cpu(mOutputDims.d[0],
                            mOutputDims.d[1],
                            mOutputDims.d[2],
                            mOutputDims.d[3],
                            mNbAnchors,
                            mChannelHeight,
                            mChannelWidth,
                            mNbProposals,
                            mPreNMsTopN,
                            mNMS_IoU,
                            mMinHeight,
                            mMinWidth,
                            mScoreIndex,
                            mIoUIndex,
                            inputDataCPU,
                            outputDataCPU);

         CHECK_CUDA_STATUS(cudaMemcpy(outputs[0],
             //outputDataCPU + 4*batchSize*i,
             outputDataCPU,
             size_output_cpy*sizeof(DATA_T),
             cudaMemcpyHostToDevice));

        delete[] inputDataCPU;
        delete[] outputDataCPU;
        return 0;
	}

	virtual size_t getSerializationSize() override
	{
        size_t inputDimParamSize = 4*sizeof(int); //
        size_t proposalParamI = 7*sizeof(int); //
        size_t proposalParamD = 3*sizeof(double); //RatioX and RatioY

        mSerializationSize = inputDimParamSize + proposalParamI
                                + proposalParamD;

        return mSerializationSize;
	}

	virtual void serialize(void* buffer) override
	{
        char* d = reinterpret_cast<char*>(buffer);
        char* a = d;
        write<int>(d, (int)mOutputDims.d[0]);
        write<int>(d, (int)mOutputDims.d[1]);
        write<int>(d, (int)mOutputDims.d[2]);
        write<int>(d, (int)mOutputDims.d[3]);
        write<int>(d, (int)mChannelHeight);
        write<int>(d, (int)mChannelWidth);
        write<int>(d, (int)mNbProposals);
        write<int>(d, (int)mNbAnchors);
        write<int>(d, (int)mPreNMsTopN);
        write<double>(d, mNMS_IoU);
        write<double>(d, mMinWidth);
        write<double>(d, mMinHeight);
        write<int>(d, (int)mScoreIndex);
        write<int>(d, (int)mIoUIndex);

        assert(d == a + getSerializationSize());
	}

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

    nvinfer1::Dims mOutputDims;
    unsigned int mChannelHeight;
    unsigned int mChannelWidth;
    unsigned int mNbProposals;
    unsigned int mNbAnchors;
    unsigned int mPreNMsTopN;
    unsigned int mScoreIndex;
    unsigned int mIoUIndex;
    double mNMS_IoU;
    double mMinWidth;
    double mMinHeight;
    size_t mSerializationSize;

};

/******************************************************************************/
/**Plugin Layer implementation**/
/**ObjectDet CPU implementation**/
class ObjDetCPUPlugin: public nvinfer1::IPlugin
{
public:
	ObjDetCPUPlugin(unsigned int batchSize,
                    unsigned int nbOutputs,
                    unsigned int outputHeight,
                    unsigned int outputWidth,
                    unsigned int channelHeight,
                    unsigned int channelWidth,
                    unsigned int nbProposals,
                    unsigned int nbCls,
                    unsigned int nbAnchors,
                    double nmsIoU,
                    const float* scoreThreshold)
	{
        mOutputDims.d[0] = batchSize;
        mOutputDims.d[1] = nbOutputs;
        mOutputDims.d[2] = outputHeight;
        mOutputDims.d[3] = outputWidth;
        mChannelHeight = channelHeight;
        mChannelWidth = channelWidth;
        mNbAnchors = nbAnchors;
        mNbClass= nbCls;
        mNbProposals = nbProposals;
        mNMS_IoU = nmsIoU;
        mScoreThreshold = new float[mNbClass];

        for(unsigned int i = 0; i < mNbClass; ++i)
            mScoreThreshold[i] = scoreThreshold[i];
	}

	ObjDetCPUPlugin(const void* dataFromRuntime, size_t length)
	{
		const char* d = reinterpret_cast<const char*>(dataFromRuntime), *a = d;
        mOutputDims.d[0] = read<int>(d);
		mOutputDims.d[1] = read<int>(d);
		mOutputDims.d[2] = read<int>(d);
		mOutputDims.d[3] = read<int>(d);
        mChannelHeight = (unsigned int) read<int>(d);
        mChannelWidth = (unsigned int) read<int>(d);
        mNbAnchors = (unsigned int) read<int>(d);
        mNbClass = (unsigned int) read<int>(d);
        mNbProposals = (unsigned int) read<int>(d);
        mNMS_IoU = read<double>(d);
        mScoreThreshold = new float[mNbClass];
        for(unsigned int k = 0; k < mNbClass; ++k)
            mScoreThreshold[k] = read<float>(d);

		assert(d == a + length);
	}

	~ObjDetCPUPlugin()
	{

	}

	virtual int getNbOutputs() const override
	{
        //return (int)mNbProposals;
        return (int) 1;

	}

	virtual nvinfer1::Dims getOutputDimensions(int index,
                                               const nvinfer1::Dims* inputDim,
                                               int nbInputDims) override
	{
        //return nvinfer1::DimsCHW(mOutputDims.d[1], mOutputDims.d[2], mOutputDims.d[3]);
        ///Proposals are store through CHW format with C of size NbProposal*4 (4 for the ROI proposals coordinate)
        unsigned int batchInput = 1;

        if(inputDim[0].nbDims == 4)
            batchInput = inputDim[0].d[0];

        //return nvinfer1::DimsNCHW(batchInput, mNbProposals*mOutputDims.d[1], mOutputDims.d[2], mOutputDims.d[3]);
        return nvinfer1::DimsNCHW(  mNbProposals*mNbClass,
                                    mOutputDims.d[1],
                                    mOutputDims.d[2],
                                    mOutputDims.d[3]);

	}

	virtual void configure(const nvinfer1::Dims* inputDims,
                   int nbInputs,
                   const nvinfer1::Dims* outputDims,
                   int nbOutputs,
                   int maxBatchSize) override
	{

	}

	virtual int initialize() override
	{
		return 0;
	}

	virtual void terminate() override
	{

	}

	virtual size_t getWorkspaceSize(int maxBatchSize) const override
	{
		return 0;
	}

	virtual int enqueue(int batchSize,
                        const void*const * inputs,
                        void** outputs,
                        void* workspace,
                        cudaStream_t stream) override
	{
        const unsigned int outputBatchSize = batchSize*mNbClass*mNbAnchors;

        float* inputDataCPU(NULL);
        float* outputDataCPU(NULL);
        size_t size_input_cpy = 6*mNbAnchors*mNbClass
                                 *mChannelHeight*mChannelWidth*batchSize;
        size_t size_output_cpy = mNbProposals*mNbClass*mOutputDims.d[1]
                                    *mOutputDims.d[2]*mOutputDims.d[3]*batchSize;

        inputDataCPU = new float[size_input_cpy];
        outputDataCPU = new float[size_output_cpy];

        if (!inputDataCPU)
            throw std::runtime_error(
                "enqueue(): could not allocate memory");

        if (!outputDataCPU)
            throw std::runtime_error(
                "enqueue(): could not allocate memory");

        CHECK_CUDA_STATUS(cudaMemcpy(inputDataCPU,
                                     reinterpret_cast<const float*>(inputs[0]),
                                     size_input_cpy*sizeof(float),
                                     cudaMemcpyDeviceToHost));

        CHECK_CUDA_STATUS(cudaStreamSynchronize(stream));

        object_det_cpu( outputBatchSize,
                        mOutputDims.d[1],
                        mOutputDims.d[2],
                        mOutputDims.d[3],
                        mChannelHeight,
                        mChannelWidth,
                        mNbAnchors,
                        mNbProposals,
                        mNbClass,
                        mNMS_IoU,
                        mScoreThreshold,
                        inputDataCPU,
                        outputDataCPU);

         CHECK_CUDA_STATUS(cudaMemcpy(outputs[0],
             //outputDataCPU + 4*batchSize*i,
             outputDataCPU,
             size_output_cpy*sizeof(float),
             cudaMemcpyHostToDevice));

        delete[] inputDataCPU;
        delete[] outputDataCPU;

        return 0;
	}

	virtual size_t getSerializationSize() override
	{
        size_t proposalParamI = 9*sizeof(int); //
        size_t proposalParamD = 1*sizeof(double) + mNbClass*sizeof(float); //RatioX and RatioY

        mSerializationSize = proposalParamI + proposalParamD;

        return mSerializationSize;
	}

	virtual void serialize(void* buffer) override
	{

        char* d = reinterpret_cast<char*>(buffer);
        char* a = d;
        write<int>(d, (int)mOutputDims.d[0]);
        write<int>(d, (int)mOutputDims.d[1]);
        write<int>(d, (int)mOutputDims.d[2]);
        write<int>(d, (int)mOutputDims.d[3]);
        write<int>(d, (int)mChannelHeight);
        write<int>(d, (int)mChannelWidth);
        write<int>(d, (int)mNbAnchors);
        write<int>(d, (int)mNbClass);
        write<int>(d, (int)mNbProposals);
        write<double>(d, mNMS_IoU);

        for(unsigned int k = 0; k < mNbClass; ++k)
            write<float>(d, mScoreThreshold[k]);

        assert(d == a + getSerializationSize());
	}

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

    nvinfer1::Dims mOutputDims;
    unsigned int mChannelHeight;
    unsigned int mChannelWidth;
    unsigned int mNbProposals;
    unsigned int mNbAnchors;
    unsigned int mNbClass;
    double mNMS_IoU;
    float* mScoreThreshold;
    size_t mSerializationSize;
};

/**ObjectDet GPU implementation**/
class ObjDetGPUPlugin: public nvinfer1::IPlugin
{
public:
	ObjDetGPUPlugin(unsigned int batchSize,
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
                    double nmsIoU,
                    const float* scoreThreshold,
                    unsigned int maxParts,
                    unsigned int maxTemplates,
                    const unsigned int* numPartsPerClass,
                    const unsigned int* numTemplatesPerClass,
                    const WDATA_T* anchor)
	{
        mOutputDims.d[0] = batchSize;
        mOutputDims.d[1] = nbOutputs;
        mOutputDims.d[2] = outputHeight;
        mOutputDims.d[3] = outputWidth;
        mChannelHeight = channelHeight;
        mChannelWidth = channelWidth;
        mStimuliWidth = stimuliWidth;
        mStimuliHeight = stimuliHeight;
        mFeatureMapWidth = featureMapWidth;
        mFeatureMapHeight = featureMapHeight;

        mNbAnchors = nbAnchors;
        mNbClass= nbCls;
        mNbProposals = nbProposals;
        mNMS_IoU = nmsIoU;

        mMaxParts = maxParts;
        mMaxTemplates = maxTemplates;

        mPartsPerClass = new unsigned int[mNbClass];
        for(unsigned int i = 0; i < mNbClass; ++i)
            mPartsPerClass[i] = numPartsPerClass[i];

        mTemplatesPerClass = new unsigned int[mNbClass];
        for(unsigned int i = 0; i < mNbClass; ++i)
            mTemplatesPerClass[i] = numTemplatesPerClass[i];
        /*
        mAnchors.resize(mNbAnchors);
        for(unsigned int i = 0; i < mNbAnchors*4; i += 4)
        {
            mAnchors[i/4].x0 = anchor[i + 0];
            mAnchors[i/4].y0 = anchor[i + 1];
            mAnchors[i/4].x1 = anchor[i + 2];
            mAnchors[i/4].y1 = anchor[i + 3];
        }
        */
        checkCudaErrors( cudaMalloc((void**)&mAnchors,
                         4 * mNbAnchors * nbCls * sizeof(WDATA_T)) );
        checkCudaErrors( cudaMemcpy(mAnchors,
                         anchor,
                         4 * mNbAnchors * nbCls *sizeof(WDATA_T),
                         cudaMemcpyHostToDevice) );

        /**Initialize pixels map on GPU **/
        checkCudaErrors( cudaMalloc((void**)&mPixelMap,
                         channelHeight * channelWidth
                         * mNbAnchors * mNbClass
                         * batchSize * sizeof(int)) );
        checkCudaErrors( cudaMemset(mPixelMap,
                                    -1,
                                    channelHeight * channelWidth
                                    * mNbAnchors * mNbClass
                                    * batchSize * sizeof(int)) );

        checkCudaErrors( cudaMalloc((void**)&mPixelMapSorted,
                         channelHeight * channelWidth
                         * mNbAnchors * mNbClass
                         * batchSize * sizeof(int)) );
        checkCudaErrors( cudaMemset(mPixelMapSorted,
                                    -1,
                                    channelHeight * channelWidth
                                    * mNbAnchors * mNbClass
                                    * batchSize * sizeof(int)) );

        checkCudaErrors( cudaMalloc((void**)&mScoresIndex,
                         channelHeight * channelWidth
                         * mNbAnchors * mNbClass
                         * batchSize * sizeof(int)) );
        checkCudaErrors( cudaMalloc((void**)&mScores,
                         channelHeight * channelWidth
                         * mNbAnchors * mNbClass
                         * batchSize * sizeof(float)) );
        checkCudaErrors( cudaMalloc((void**)&mScoresFiltered,
                         channelHeight * channelWidth
                         * mNbAnchors * mNbClass
                         * batchSize * sizeof(float)) );


        checkCudaErrors( cudaMalloc((void**)&mMxGPUIndex,
                         channelHeight * channelWidth
                         * mNbAnchors * mNbClass
                         * batchSize * sizeof(float)) );
        checkCudaErrors( cudaMemset(mMxGPUIndex,
                                    -1.0,
                                    channelHeight * channelWidth
                                    * mNbAnchors * mNbClass
                                    * batchSize * sizeof(float)) );
        checkCudaErrors( cudaMalloc((void**)&mMyGPUIndex,
                         channelHeight * channelWidth
                         * mNbAnchors * mNbClass
                         * batchSize * sizeof(float)) );
        checkCudaErrors( cudaMemset(mMyGPUIndex,
                                    -1.0,
                                    channelHeight * channelWidth
                                    * mNbAnchors * mNbClass
                                    * batchSize * sizeof(float)) );
        checkCudaErrors( cudaMalloc((void**)&mMwGPUIndex,
                         channelHeight * channelWidth
                         * mNbAnchors * mNbClass
                         * batchSize * sizeof(float)) );
        checkCudaErrors( cudaMemset(mMwGPUIndex,
                                    -1.0,
                                    channelHeight * channelWidth
                                    * mNbAnchors * mNbClass
                                    * batchSize * sizeof(float)) );
        checkCudaErrors( cudaMalloc((void**)&mMhGPUIndex,
                         channelHeight * channelWidth
                         * mNbAnchors * mNbClass
                         * batchSize * sizeof(float)) );
        checkCudaErrors( cudaMemset(mMhGPUIndex,
                                    -1.0,
                                    channelHeight * channelWidth
                                    * mNbAnchors * mNbClass
                                    * batchSize * sizeof(float)) );

        mMxCPUIndex = new float[channelHeight * channelWidth * mNbAnchors * mNbClass * batchSize];
        mMyCPUIndex = new float[channelHeight * channelWidth * mNbAnchors * mNbClass * batchSize];
        mMwCPUIndex = new float[channelHeight * channelWidth * mNbAnchors * mNbClass * batchSize];
        mMhCPUIndex = new float[channelHeight * channelWidth * mNbAnchors * mNbClass * batchSize];

        /**Initialize mScoreThreshold parameters on GPU **/
        checkCudaErrors( cudaMalloc((void**)&mScoreThreshold,
                         mNbClass*sizeof(float)) );
        checkCudaErrors( cudaMemcpy(mScoreThreshold,
                         scoreThreshold,
                         mNbClass*sizeof(float),
                         cudaMemcpyHostToDevice) );

        checkCudaErrors( cudaMalloc((void**)&mROIsBBOxFinal,
                            mNbProposals*5*batchSize*sizeof(float)) );

        checkCudaErrors( cudaMalloc((void**)&mROIsMapAnchorsFinal,
                            mNbProposals*5*batchSize*sizeof(float)) );

        checkCudaErrors( cudaMalloc((void**)&mROIsIndexFinal,
                            batchSize*sizeof(unsigned int)) );


        gpuThreadAllocation();
	}

	ObjDetGPUPlugin(const void* dataFromRuntime, size_t length)
	{
		const char* d = reinterpret_cast<const char*>(dataFromRuntime), *a = d;
        mOutputDims.d[0] = read<int>(d);
		mOutputDims.d[1] = read<int>(d);
		mOutputDims.d[2] = read<int>(d);
		mOutputDims.d[3] = read<int>(d);
        mChannelHeight = (unsigned int) read<int>(d);
        mChannelWidth = (unsigned int) read<int>(d);
        mStimuliWidth = read<unsigned int>(d);
        mStimuliHeight = read<unsigned int>(d);
        mFeatureMapWidth = read<unsigned int>(d);
        mFeatureMapHeight = read<unsigned int>(d);

        mNbAnchors = (unsigned int) read<int>(d);
        mNbClass = (unsigned int) read<int>(d);
        mNbProposals = (unsigned int) read<int>(d);
        mMaxParts = (unsigned int) read<int>(d);
        mMaxTemplates = (unsigned int) read<int>(d);
        mNMS_IoU = read<double>(d);
        mThreadX = read<int>(d);
        mThreadY = read<int>(d);
        mThreadZ = read<int>(d);
        mBlockX = read<int>(d);
        mBlockY = read<int>(d);
        mBlockZ = read<int>(d);
		mPixelMap = deserializeToDevice<int>(d, mChannelHeight * mChannelWidth * mNbAnchors * mNbClass * mOutputDims.d[0]);
		mPixelMapSorted = deserializeToDevice<int>(d, mChannelHeight * mChannelWidth * mNbAnchors * mNbClass * mOutputDims.d[0]);
        mScoresIndex = deserializeToDevice<int>(d, mChannelHeight * mChannelWidth * mNbAnchors * mNbClass * mOutputDims.d[0]);
        mScores = deserializeToDevice<float>(d, mChannelHeight * mChannelWidth * mNbAnchors * mNbClass * mOutputDims.d[0]);
        mScoresFiltered = deserializeToDevice<float>(d, mChannelHeight * mChannelWidth * mNbAnchors * mNbClass * mOutputDims.d[0]);

        mMxGPUIndex = deserializeToDevice<float>(d, mChannelHeight * mChannelWidth * mNbAnchors * mNbClass * mOutputDims.d[0]);
		mMyGPUIndex = deserializeToDevice<float>(d, mChannelHeight * mChannelWidth * mNbAnchors * mNbClass * mOutputDims.d[0]);
		mMwGPUIndex = deserializeToDevice<float>(d, mChannelHeight * mChannelWidth * mNbAnchors * mNbClass * mOutputDims.d[0]);
		mMhGPUIndex = deserializeToDevice<float>(d, mChannelHeight * mChannelWidth * mNbAnchors * mNbClass * mOutputDims.d[0]);

        mMxCPUIndex = new float[mChannelHeight * mChannelWidth * mNbAnchors * mNbClass * mOutputDims.d[0]];
        for(unsigned int k = 0; k < mChannelHeight * mChannelWidth * mNbAnchors * mNbClass * mOutputDims.d[0]; ++k)
            mMxCPUIndex[k] = read<float>(d);
        mMyCPUIndex = new float[mChannelHeight * mChannelWidth * mNbAnchors * mNbClass * mOutputDims.d[0]];
        for(unsigned int k = 0; k < mChannelHeight * mChannelWidth * mNbAnchors * mNbClass * mOutputDims.d[0]; ++k)
            mMyCPUIndex[k] = read<float>(d);
        mMwCPUIndex = new float[mChannelHeight * mChannelWidth * mNbAnchors * mNbClass * mOutputDims.d[0]];
        for(unsigned int k = 0; k < mChannelHeight * mChannelWidth * mNbAnchors * mNbClass * mOutputDims.d[0]; ++k)
            mMwCPUIndex[k] = read<float>(d);
        mMhCPUIndex = new float[mChannelHeight * mChannelWidth * mNbAnchors * mNbClass * mOutputDims.d[0]];
        for(unsigned int k = 0; k < mChannelHeight * mChannelWidth * mNbAnchors * mNbClass * mOutputDims.d[0]; ++k)
            mMhCPUIndex[k] = read<float>(d);
        
        mScoreThreshold = deserializeToDevice<float>(d, mNbClass);

        mPartsPerClass = new unsigned int[mNbClass];
        for(unsigned int k = 0; k < mNbClass; ++k)
            mPartsPerClass[k] = read<unsigned int>(d);

        mTemplatesPerClass = new unsigned int[mNbClass];
        for(unsigned int k = 0; k < mNbClass; ++k)
            mTemplatesPerClass[k] = read<unsigned int>(d);
        /*mAnchors.resize(mNbAnchors);
        for(unsigned int i = 0; i < mNbAnchors; ++i)
        {
            mAnchors[i].x0 = read<WDATA_T>(d);
            mAnchors[i].y0 = read<WDATA_T>(d);
            mAnchors[i].x1 = read<WDATA_T>(d);
            mAnchors[i].y1 = read<WDATA_T>(d);
        }*/
        mAnchors = deserializeToDevice<WDATA_T>(d, 4 * mNbAnchors * mNbClass);

        mROIsBBOxFinal = deserializeToDevice<float>(d, mNbProposals*5*mOutputDims.d[0]);
        mROIsMapAnchorsFinal = deserializeToDevice<float>(d, mNbProposals*5*mOutputDims.d[0]);
        mROIsIndexFinal = deserializeToDevice<unsigned int>(d, mOutputDims.d[0]);

		assert(d == a + length);

	}

	~ObjDetGPUPlugin()
	{
        //mAnchors = std::vector<Anchor>();
	}

	virtual int getNbOutputs() const override
	{
        //return (int)mNbProposals;
        return (int) 1;

	}

	virtual nvinfer1::Dims getOutputDimensions(int index,
                                               const nvinfer1::Dims* inputDim,
                                               int nbInputDims) override
	{
        //return nvinfer1::DimsCHW(mOutputDims.d[1], mOutputDims.d[2], mOutputDims.d[3]);
        ///Proposals are store through CHW format with C of size NbProposal*4 (4 for the ROI proposals coordinate)
        unsigned int batchInput = 1;

        if(inputDim[0].nbDims == 4)
            batchInput = inputDim[0].d[0];

        //return nvinfer1::DimsNCHW(batchInput, mNbProposals*mOutputDims.d[1], mOutputDims.d[2], mOutputDims.d[3]);
        return nvinfer1::DimsNCHW(  mNbProposals*mNbClass,
                                    mOutputDims.d[1],
                                    mOutputDims.d[2],
                                    mOutputDims.d[3]);

	}

	virtual void configure(const nvinfer1::Dims* inputDims,
                   int nbInputs,
                   const nvinfer1::Dims* outputDims,
                   int nbOutputs,
                   int maxBatchSize) override
	{

	}

	virtual int initialize() override
	{
		return 0;
	}

	virtual void terminate() override
	{

	}

	virtual size_t getWorkspaceSize(int maxBatchSize) const override
	{
		return 0;
	}

	virtual int enqueue(int batchSize,
                        const void*const * inputs,
                        void** outputs,
                        void* workspace,
                        cudaStream_t stream) override
	{
        //float* outputDataCPU(NULL);

        size_t size_output_cpy = mNbProposals*mNbClass*mOutputDims.d[1]
                                    *mOutputDims.d[2]*mOutputDims.d[3]*batchSize;

        size_t size_map = mChannelHeight * mChannelWidth
                                    * mNbAnchors * mNbClass
                                    * batchSize;

        //float outputDataCPU[size_output_cpy] = {0.0};

        //if (!outputDataCPU)
        //    throw std::runtime_error(
        //        "enqueue(): could not allocate memory");

        //DATA_T* input_parts(NULL);
        //DATA_T* input_templates(NULL);

        //input_parts = new DATA_T[mMaxParts*2];
        //input_templates = new DATA_T[mMaxTemplates*3];

        const unsigned int inputBatchOffset = mChannelWidth*mChannelHeight*(mNbAnchors*mNbClass * 6);
        unsigned int nbTotalPart = 0;
        unsigned int nbTotalTemplate = 0;

        /***TO IMPROVE!!!!!**/
        const double xRatio = std::ceil(mFeatureMapWidth / mChannelWidth);
        const double yRatio = std::ceil(mFeatureMapHeight / mChannelHeight);
        const float xOutputRatio = mStimuliWidth / (float) mFeatureMapWidth;
        const float yOutputRatio = mStimuliHeight / (float) mFeatureMapHeight;

        for(unsigned int c = 0; c < mNbClass; ++c) 
        {
            nbTotalPart += mPartsPerClass[c];
            nbTotalTemplate += mTemplatesPerClass[c];
        }

        checkCudaErrors( cudaMemset(mPixelMapSorted,
                                    -1,
                                    mChannelHeight * mChannelWidth
                                    * mNbAnchors * mNbClass
                                    * batchSize * sizeof(int)) );
        checkCudaErrors( cudaMemset(mPixelMap,
                                    -1,
                                    mChannelHeight * mChannelWidth
                                    * mNbAnchors * mNbClass
                                    * batchSize * sizeof(int)) );


        dim3 blocksPerGrid = {  (unsigned int) mBlockX,
                                (unsigned int) mBlockY,
                                (unsigned int) batchSize,
                            };

        dim3 threadsPerBlock = {(unsigned int) mThreadX,
                                (unsigned int) mThreadY,
                                (unsigned int) mThreadZ
                                };

        cudaSReduceIndex(  mChannelWidth*mChannelHeight*mNbAnchors,
                        inputBatchOffset,
                        mChannelWidth*mChannelHeight*mNbAnchors*mNbClass,
                        mChannelWidth,
                        mChannelHeight,
                        mNbAnchors,
                        reinterpret_cast<const float *>(mScoreThreshold),
                        reinterpret_cast<const DATA_T *>(inputs[0]),
                        reinterpret_cast<int *>(mPixelMap),
                        reinterpret_cast<float *>(mScores),
                        blocksPerGrid,
                        threadsPerBlock);

        std::vector<std::vector <unsigned int> > count(batchSize,
                                                    std::vector<unsigned int>(mNbClass));

        for(unsigned int batchPos = 0; batchPos < batchSize; ++batchPos)
        {
            for(unsigned int cls = 0; cls < mNbClass; ++cls)
            {
                const int pixelOffset = cls*mChannelWidth*mChannelHeight*mNbAnchors 
                                            +  mChannelWidth*mChannelHeight*mNbAnchors*mNbClass*batchPos;

                const int nbMapDet = copy_if_int(    reinterpret_cast<int *>(mPixelMap) + pixelOffset,
                                                   reinterpret_cast<int *>(mPixelMapSorted) + pixelOffset,
                                                   mChannelWidth*mChannelHeight*mNbAnchors);

                const int nbScoreDet = copy_if_float( reinterpret_cast<float *>(mScores) + pixelOffset,
                                                    reinterpret_cast<float *>(mScoresFiltered) + pixelOffset,
                                                    mChannelWidth*mChannelHeight*mNbAnchors);

                if (nbScoreDet != nbMapDet)
                    throw std::runtime_error(
                        "Dont find the same number of valid boxes");

                count[batchPos][cls] = nbMapDet;

            }
        }

        std::vector< std::vector< std::vector<BBox_T >>> ROIs(  mNbClass, 
                                                                std::vector< std::vector <BBox_T>>(batchSize));

        std::vector< std::vector< std::vector<BBox_T >>> ROIsAnchors(   mNbClass, 
                                                                        std::vector< std::vector <BBox_T>>(batchSize));

        for(unsigned int cls = 0; cls < mNbClass; ++cls)
        {
            const int offset = cls*mNbAnchors*mChannelWidth*mChannelHeight;

            for(unsigned int batchPos = 0; batchPos < batchSize; ++batchPos)
            {
                const int batchOffset = batchPos*inputBatchOffset;

                unsigned int totalIdxPerClass = 0;

                if(count[batchPos][cls] > 0)
                {
                    const int offsetBase = mNbClass*mNbAnchors*mChannelWidth*mChannelHeight;

                    const int offsetCpy = cls*mNbAnchors*mChannelWidth*mChannelHeight
                                            + batchPos*mNbClass*mNbAnchors*mChannelWidth*mChannelHeight;

                    unsigned int nbElementNMS =  count[batchPos][cls];

                    thrust_sort_keys_int(   reinterpret_cast<float *>(mScoresFiltered) + offsetCpy,
                                            reinterpret_cast<int *>(mPixelMapSorted) + offsetCpy,
                                            nbElementNMS,
                                            0);


                    thrust_gather_int(reinterpret_cast<int*>(mPixelMapSorted) + offset + batchPos*mNbClass*mNbAnchors*mChannelWidth*mChannelHeight,
                                reinterpret_cast<const DATA_T *>(inputs[0]) + offsetBase + offset + batchOffset,
                                reinterpret_cast<DATA_T *>(mMxGPUIndex),
                                nbElementNMS,
                                0,
                                0);
                    thrust_gather_int(reinterpret_cast<int*>(mPixelMapSorted) + offset + batchPos*mNbClass*mNbAnchors*mChannelWidth*mChannelHeight,
                                reinterpret_cast<const DATA_T *>(inputs[0]) + 2*offsetBase + offset + batchOffset,
                                reinterpret_cast<DATA_T *>(mMyGPUIndex),
                                nbElementNMS,
                                0,
                                0);

                    thrust_gather_int(reinterpret_cast<int*>(mPixelMapSorted) + offset + batchPos*mNbClass*mNbAnchors*mChannelWidth*mChannelHeight,
                                reinterpret_cast<const DATA_T *>(inputs[0]) + 3*offsetBase + offset + batchOffset,
                                reinterpret_cast<DATA_T *>(mMwGPUIndex),
                                nbElementNMS,
                                0,
                                0);

                    thrust_gather_int(reinterpret_cast<int*>(mPixelMapSorted) + offset + batchPos*mNbClass*mNbAnchors*mChannelWidth*mChannelHeight,
                                reinterpret_cast<const DATA_T *>(inputs[0]) + 4*offsetBase + offset + batchOffset,
                                reinterpret_cast<DATA_T *>(mMhGPUIndex),
                                nbElementNMS,
                                0,
                                0);

                    int* pixelMap(NULL);
                    float* scoreMap(NULL);

                    pixelMap = new int[nbElementNMS];
                    scoreMap = new float[nbElementNMS];

                    CHECK_CUDA_STATUS(cudaMemcpy(pixelMap,
                            reinterpret_cast<int*>(mPixelMapSorted) + offsetCpy,
                            nbElementNMS*sizeof(int),
                            cudaMemcpyDeviceToHost));

                    CHECK_CUDA_STATUS(cudaMemcpy(scoreMap,
                            reinterpret_cast<float*>(mScoresFiltered) + offsetCpy,
                            nbElementNMS*sizeof(float),
                            cudaMemcpyDeviceToHost));

                    CHECK_CUDA_STATUS(cudaMemcpy(mMxCPUIndex,
                                                reinterpret_cast<DATA_T*>(mMxGPUIndex),
                                                nbElementNMS*sizeof(DATA_T),
                                                cudaMemcpyDeviceToHost));
                    CHECK_CUDA_STATUS(cudaMemcpy(mMyCPUIndex,
                                                reinterpret_cast<DATA_T*>(mMyGPUIndex),
                                                nbElementNMS*sizeof(DATA_T),
                                                cudaMemcpyDeviceToHost));
                    CHECK_CUDA_STATUS(cudaMemcpy(mMwCPUIndex,
                                                reinterpret_cast<DATA_T*>(mMwGPUIndex),
                                                nbElementNMS*sizeof(DATA_T),
                                                cudaMemcpyDeviceToHost));
                    CHECK_CUDA_STATUS(cudaMemcpy(mMhCPUIndex,
                                                reinterpret_cast<DATA_T*>(mMhGPUIndex),
                                                nbElementNMS*sizeof(DATA_T),
                                                cudaMemcpyDeviceToHost));


                    for(unsigned int idx = 0; idx < nbElementNMS; ++idx)
                    {
                        ROIs[cls][batchPos].push_back(BBox_T(  mMxCPUIndex[idx],
                                                                mMyCPUIndex[idx],
                                                                mMwCPUIndex[idx],
                                                                mMhCPUIndex[idx],
                                                                scoreMap[idx]));
                        ROIsAnchors[cls][batchPos].push_back(BBox_T(   pixelMap[idx]%mChannelWidth,
                                                                        (pixelMap[idx]/mChannelWidth)%mChannelHeight,
                                                                        (pixelMap[idx]/(mChannelWidth*mChannelHeight))%mNbAnchors,
                                                                        0.0,
                                                                        0.0));
                    }
                    delete[] pixelMap;
                    delete[] scoreMap;

                    // Non-Maximum Suppression (NMS)
                    /*for (unsigned int i = 0; i < ROIs.size() - 1 && i < 50; ++i)
                    {
                        const float x0 = ROIs[i].x;
                        const float y0 = ROIs[i].y;
                        const float w0 = ROIs[i].w;
                        const float h0 = ROIs[i].h;

                        for (unsigned int j = i + 1; j < ROIs.size(); ) {

                            const float x = ROIs[j].x;
                            const float y = ROIs[j].y;
                            const float w = ROIs[j].w;
                            const float h = ROIs[j].h;

                            const float interLeft = std::max(x0, x);
                            const float interRight = std::min(x0 + w0, x + w);
                            const float interTop = std::max(y0, y);
                            const float interBottom = std::min(y0 + h0, y + h);

                            if (interLeft < interRight && interTop < interBottom) {
                                const float interArea = (interRight - interLeft)
                                                            * (interBottom - interTop);
                                const float unionArea = w0 * h0 + w * h - interArea;
                                const float IoU = interArea / unionArea;

                                if (IoU > mNMS_IoU) {
                                    // Suppress ROI
                                    ROIs.erase(ROIs.begin() + j);
                                    keepIdx[batchPos][cls].erase(keepIdx[batchPos][cls].begin() + j);

                                    continue;
                                }
                            }
                            ++j;
                        }
                    }*/

                    std::vector<BBox_T> final_rois;
                    std::vector<BBox_T> final_anchors;

                    BBox_T next_candidate;
                    BBox_T next_anchors;
                    std::reverse(ROIs[cls][batchPos].begin(),ROIs[cls][batchPos].end());
                    std::reverse(ROIsAnchors[cls][batchPos].begin(),ROIsAnchors[cls][batchPos].end());

                    while (final_rois.size() < mNbProposals && !ROIs[cls][batchPos].empty()) {
                        next_candidate = ROIs[cls][batchPos].back();
                        ROIs[cls][batchPos].pop_back();
                        next_anchors = ROIsAnchors[cls][batchPos].back();
                        ROIsAnchors[cls][batchPos].pop_back();
                        // Overlapping boxes are likely to have similar scores,
                        // therefore we iterate through the previously selected boxes backwards
                        // in order to see if `next_candidate` should be suppressed.
                        bool should_select = true;
                        const float x0 = next_candidate.x;
                        const float y0 = next_candidate.y;
                        const float w0 = next_candidate.w;
                        const float h0 = next_candidate.h;

                        for (int j = static_cast<int>(final_rois.size()) - 1; j >= 0; --j) {

                            const float x = final_rois[j].x;
                            const float y = final_rois[j].y;
                            const float w = final_rois[j].w;
                            const float h = final_rois[j].h;
                            const float interLeft = std::max(x0, x);
                            const float interRight = std::min(x0 + w0, x + w);
                            const float interTop = std::max(y0, y);
                            const float interBottom = std::min(y0 + h0, y + h);

                            if (interLeft < interRight && interTop < interBottom) {
                                const float interArea = (interRight - interLeft)
                                                            * (interBottom - interTop);
                                const float unionArea = w0 * h0 + w * h - interArea;
                                const float IoU = interArea / unionArea;

                                if (IoU > mNMS_IoU) {
                                    should_select = false;
                                    break;

                                }
                            }

                        }

                        if (should_select) {
                            final_rois.push_back(next_candidate);
                            final_anchors.push_back(next_anchors);
                        }
                    }
                    ROIs[cls][batchPos].resize(final_rois.size());
                    ROIsAnchors[cls][batchPos].resize(final_anchors.size());

                    for(unsigned int f = 0; f < final_rois.size(); ++ f )
                    {
                        ROIs[cls][batchPos][f] = final_rois[f];
                        ROIsAnchors[cls][batchPos][f] = final_anchors[f];
                    }
                    /*
                    for(unsigned int i = 0; i < ROIs[cls][batchPos].size() && i < mNbProposals; ++i)
                    {
                        const unsigned int n = i + cls*mNbProposals + batchPos*mNbProposals*mNbClass;
                        outputDataCPU[0 + n*(5 + mMaxParts*2 + mMaxTemplates*3)] = ROIs[cls][batchPos][i].x;
                        outputDataCPU[1 + n*(5 + mMaxParts*2 + mMaxTemplates*3)] = ROIs[cls][batchPos][i].y;
                        outputDataCPU[2 + n*(5 + mMaxParts*2 + mMaxTemplates*3)] = ROIs[cls][batchPos][i].w;
                        outputDataCPU[3 + n*(5 + mMaxParts*2 + mMaxTemplates*3)] = ROIs[cls][batchPos][i].h;
                        outputDataCPU[4 + n*(5 + mMaxParts*2 + mMaxTemplates*3)] = (float) cls;
                        ++totalIdxPerClass;

                        const unsigned int xa = ROIsAnchors[cls][batchPos][i].x;
                        const unsigned int ya = ROIsAnchors[cls][batchPos][i].y;
                        const unsigned int k = ROIsAnchors[cls][batchPos][i].w;

                        
                        if(mMaxParts > 0 && mMaxTemplates > 0)
                        {

                            unsigned int partsIdx = 0;
                            for(unsigned int c = 0; c < cls; ++c) partsIdx += mPartsPerClass[c] * 2 * mNbAnchors;

                            for(unsigned int part = 0; part < mPartsPerClass[cls]; ++part)
                            {
                                
                                const int yIdx = xa 
                                                + ya*mChannelWidth 
                                                + (k*mPartsPerClass[cls]*2 + partsIdx + part*2)*mChannelHeight*mChannelWidth
                                                + batchPos*mChannelHeight*mChannelWidth*mNbAnchors*2*nbTotalPart;
                                const int xIdx = xa 
                                                + ya*mChannelWidth 
                                                + (k*mPartsPerClass[cls]*2 + partsIdx + part*2 + 1)*mChannelHeight*mChannelWidth
                                                + batchPos*mChannelHeight*mChannelWidth*mNbAnchors*2*nbTotalPart;

                                CHECK_CUDA_STATUS(  cudaMemcpy( input_parts + part*2, 
                                                                reinterpret_cast<const DATA_T *>(inputs[2]) + yIdx,
                                                                1*sizeof(DATA_T),
                                                                cudaMemcpyDeviceToHost));

                                CHECK_CUDA_STATUS(  cudaMemcpy( input_parts + part*2 + 1, 
                                                                reinterpret_cast<const DATA_T *>(inputs[2]) + xIdx,
                                                                1*sizeof(DATA_T),
                                                                cudaMemcpyDeviceToHost));
                                const float partY = input_parts[part*2];
                                const float partX = input_parts[part*2 + 1];

                                const Anchor& anchor = mAnchors[k];
                                const int xa0 = (int)(anchor.x0 + xa * xRatio);
                                const int ya0 = (int)(anchor.y0 + ya * yRatio);
                                const int xa1 = (int)(anchor.x1 + xa * xRatio);
                                const int ya1 = (int)(anchor.y1 + ya * yRatio);

                                // Anchors width and height
                                const int wa = xa1 - xa0;
                                const int ha = ya1 - ya0;

                                // Anchor center coordinates (xac, yac)
                                const float xac = xa0 + wa / 2.0;
                                const float yac = ya0 + ha / 2.0;
                                const float predPartY = ((partY) * ha + yac)*yOutputRatio ;
                                const float predPartX = ((partX) * wa + xac)*xOutputRatio ;

                                outputDataCPU[part*2 + 0 + 5 + n*(5 + mMaxParts*2 + mMaxTemplates*3) ] = predPartY;
                                outputDataCPU[part*2 + 1 + 5 + n*(5 + mMaxParts*2 + mMaxTemplates*3) ] = predPartX;
                                if(cls == 0 && batchPos == 0 && k ==7)
                                    std::cout << "CPU: [" 
                                            << part << "][" 
                                            << yIdx << "][" 
                                            << xIdx << "]("
                                            << anchor.x0 << ","
                                            << anchor.y0 
                                            << ", " << anchor.x1 
                                            << ", " << anchor.y1 << ")" << std::endl;


                            }
                            unsigned int templateIdx = 0;
                            for(unsigned int c = 0; c < cls; ++c) templateIdx += mTemplatesPerClass[c] * 3 * mNbAnchors;
                            
                            for(unsigned int temp = 0; temp < mTemplatesPerClass[cls]; ++temp)
                            {

                                const int yIdx = xa 
                                                + ya*mChannelWidth 
                                                + (k*mTemplatesPerClass[cls]*3 + templateIdx + temp*3)*mChannelHeight*mChannelWidth
                                                + batchPos*mChannelHeight*mChannelWidth*mNbAnchors*3*nbTotalTemplate;
                                const int xIdx = xa 
                                                + ya*mChannelWidth 
                                                + (k*mTemplatesPerClass[cls]*3 + templateIdx + temp*3 + 1)*mChannelHeight*mChannelWidth
                                                + batchPos*mChannelHeight*mChannelWidth*mNbAnchors*3*nbTotalTemplate;
                                const int zIdx = xa 
                                                + ya*mChannelWidth 
                                                + (k*mTemplatesPerClass[cls]*3 + templateIdx + temp*3 + 2)*mChannelHeight*mChannelWidth
                                                + batchPos*mChannelHeight*mChannelWidth*mNbAnchors*3*nbTotalTemplate;

                                CHECK_CUDA_STATUS(  cudaMemcpy( input_templates + temp*3, 
                                                                reinterpret_cast<const DATA_T *>(inputs[1]) + yIdx,
                                                                1*sizeof(DATA_T),
                                                                cudaMemcpyDeviceToHost));
                                CHECK_CUDA_STATUS(  cudaMemcpy( input_templates + temp*3 + 1, 
                                                                reinterpret_cast<const DATA_T *>(inputs[1]) + xIdx,
                                                                1*sizeof(DATA_T),
                                                                cudaMemcpyDeviceToHost));
                                CHECK_CUDA_STATUS(  cudaMemcpy( input_templates + temp*3 + 2, 
                                                                reinterpret_cast<const DATA_T *>(inputs[1]) + zIdx,
                                                                1*sizeof(DATA_T),
                                                                cudaMemcpyDeviceToHost));
                                const float templateY = std::exp(input_templates[temp*3]);
                                const float templateX = std::exp(input_templates[temp*3 + 1]);
                                const float templateZ = std::exp(input_templates[temp*3 + 2]);

                                outputDataCPU[temp*3 + 0 + mMaxParts*2 + 5 + n*(5 + mMaxParts*2 + mMaxTemplates*3)] = templateY;
                                outputDataCPU[temp*3 + 1 + mMaxParts*2 + 5 + n*(5 + mMaxParts*2 + mMaxTemplates*3)] = templateX;
                                outputDataCPU[temp*3 + 2 + mMaxParts*2 + 5 + n*(5 + mMaxParts*2 + mMaxTemplates*3)] = templateZ;

                            }
                        }
                    }*/
                }
                /*
                for(unsigned int rest = totalIdxPerClass; rest < mNbProposals; ++rest)
                {
                    const unsigned int n = rest + cls*mNbProposals + batchPos*mNbProposals*mNbClass;
                    outputDataCPU[0 + n*(5 + mMaxParts*2 + mMaxTemplates*3)] = 0.0;
                    outputDataCPU[1 + n*(5 + mMaxParts*2 + mMaxTemplates*3)] = 0.0;
                    outputDataCPU[2 + n*(5 + mMaxParts*2 + mMaxTemplates*3)] = 0.0;
                    outputDataCPU[3 + n*(5 + mMaxParts*2 + mMaxTemplates*3)] = 0.0;
                    outputDataCPU[4 + n*(5 + mMaxParts*2 + mMaxTemplates*3)] = 0.0;
                    
                    if(mMaxParts > 0 && mMaxTemplates > 0)
                    {
                        for(unsigned int part = 0; part < mMaxParts; ++part)
                        {
                            outputDataCPU[part*2 + 0 + 5 + n*(5 + mMaxParts*2 + mMaxTemplates*3) ] = 0.0;
                            outputDataCPU[part*2 + 1 + 5 + n*(5 + mMaxParts*2 + mMaxTemplates*3) ] = 0.0;
                        }
                        for(unsigned int temp = 0; temp < mMaxTemplates; ++temp)
                        {
                            outputDataCPU[0 + mMaxParts*2 + 5 + n*(5 + mMaxParts*2 + mMaxTemplates*3)] = 0.0;
                            outputDataCPU[1 + mMaxParts*2 + 5 + n*(5 + mMaxParts*2 + mMaxTemplates*3)] = 0.0;
                            outputDataCPU[2 + mMaxParts*2 + 5 + n*(5 + mMaxParts*2 + mMaxTemplates*3)] = 0.0;
                        }
                    }                
                }
                */
            }
        }

        /*float* cpu_anchors(NULL);

        cpu_anchors = new float[4*mNbAnchors];
        for(unsigned int a = 0; a < mNbAnchors; ++a)
        {
            const Anchor& anchor = mAnchors[a];

            cpu_anchors[a*4 + 0] = anchor.x0;
            cpu_anchors[a*4 + 1] = anchor.y0;
            cpu_anchors[a*4 + 2] = anchor.x1;
            cpu_anchors[a*4 + 3] = anchor.y1;

        }


        float* gpu_anchors;

        checkCudaErrors( cudaMalloc((void**)&gpu_anchors,
                         4*mNbAnchors*sizeof(DATA_T)) );

        checkCudaErrors( cudaMemcpy(gpu_anchors,
                         cpu_anchors,
                         4*mNbAnchors*sizeof(DATA_T),
                         cudaMemcpyHostToDevice) );
        */

        /*
        float* gpu_rois_bbox;
        float* gpu_rois_anchors;
        unsigned int* gpu_valid_rois;

        checkCudaErrors( cudaMalloc((void**)&gpu_rois_bbox,
                        mNbProposals*5*batchSize*sizeof(float)) );

        checkCudaErrors( cudaMalloc((void**)&gpu_rois_anchors,
                        mNbProposals*5*batchSize*sizeof(float)) );

        checkCudaErrors( cudaMalloc((void**)&gpu_valid_rois,
                        batchSize*sizeof(unsigned int)) );
        */

        unsigned int* valid_rois(NULL);
        valid_rois = new unsigned int[batchSize];

        for(unsigned int cls = 0; cls < mNbClass; ++cls)
        {
            int mThreadX = 32;
            int mThreadY = 1;
            int mThreadZ = 1;

            int mBlockX = std::ceil(mNbProposals/(float) mThreadX);
            int mBlockY = std::max(mPartsPerClass[cls], mTemplatesPerClass[cls]) > 0 ? 
                            std::max(mPartsPerClass[cls], mTemplatesPerClass[cls]) : 1 ;
            int mBlockZ = batchSize;

            dim3 blocks = {  (unsigned int) mBlockX, (unsigned int) mBlockY, (unsigned int) batchSize};

            dim3 threads = {(unsigned int) mThreadX, (unsigned int) mThreadY, (unsigned int) mThreadZ };
            
            for(int i = 0; i < ROIs[cls].size(); ++i)
            {
                valid_rois[i] = ROIs[cls][i].size();
            }
            unsigned int cumulParts = 0;
            unsigned int cumulTemplates = 0;

            for(unsigned int c = 0; c < cls; ++c) 
            {
                cumulParts += mPartsPerClass[c] * 2 * mNbAnchors;
                cumulTemplates += mTemplatesPerClass[c] * 3 * mNbAnchors;
            }

            for(unsigned int b = 0; b < batchSize; ++b)
            {
                const unsigned int offset = b*5*mNbProposals ;

                if(valid_rois[b] > 0)
                {
                    CHECK_CUDA_STATUS(cudaMemcpy(   mROIsBBOxFinal + offset,
                                                    ROIs[cls][b].data(),
                                                    valid_rois[b]*5*sizeof(float),
                                                    cudaMemcpyHostToDevice));
                                                    
                    CHECK_CUDA_STATUS(cudaMemcpy(   mROIsMapAnchorsFinal + offset,
                                                    ROIsAnchors[cls][b].data(),
                                                    valid_rois[b]*5*sizeof(float),
                                                    cudaMemcpyHostToDevice));

                }
            }

            CHECK_CUDA_STATUS(cudaMemcpy(   mROIsIndexFinal,
                                            valid_rois,
                                            batchSize*sizeof(unsigned int),
                                            cudaMemcpyHostToDevice));

            cudaS_SSD_output_gathering( batchSize,
                                        mNbClass,
                                        mNbAnchors,
                                        mChannelWidth,
                                        mChannelHeight,
                                        mNbProposals,
                                        mROIsIndexFinal,
                                        cls,
                                        nbTotalPart,
                                        nbTotalTemplate,
                                        mMaxParts,
                                        mMaxTemplates,
                                        cumulParts,
                                        cumulTemplates,
                                        mPartsPerClass[cls],
                                        mTemplatesPerClass[cls],
                                        xRatio,
                                        yRatio,
                                        xOutputRatio,
                                        yOutputRatio,
                                        mROIsBBOxFinal,
                                        mROIsMapAnchorsFinal,
                                        mAnchors,
                                        reinterpret_cast<const DATA_T *>(inputs[2]),
                                        reinterpret_cast<const DATA_T *>(inputs[1]),
                                        reinterpret_cast<DATA_T*>(outputs[0]),
                                        blocks,
                                        threads);
            

        }

        //checkCudaErrors( cudaFree(gpu_anchors));
        //checkCudaErrors( cudaFree(gpu_rois_bbox));
        //checkCudaErrors( cudaFree(gpu_rois_anchors));
        //checkCudaErrors( cudaFree(gpu_valid_rois));
        //delete[] cpu_anchors;
        delete[] valid_rois;

        /* CHECK_CUDA_STATUS(cudaMemcpy(outputs[0],
         outputDataCPU,
         size_output_cpy*sizeof(float),
        cudaMemcpyHostToDevice));*/

	   // delete[] pixelMapCPU;
        //delete[] outputDataCPU;
        //delete[] input_parts;
        //delete[] input_templates;
        return 0;
    }

	virtual size_t getSerializationSize() override
	{
        size_t proposalParamI = 15*sizeof(int) + (4 + 2 + 2*mNbClass)*sizeof(unsigned int); //
        size_t proposalParamD = (1)*sizeof(double) + mNbClass*sizeof(float); //RatioX and RatioY
        size_t PixelMapSize = mChannelHeight * mChannelWidth * mNbAnchors * mNbClass * mOutputDims.d[0]*sizeof(int);
        size_t M_Index = mChannelHeight * mChannelWidth * mNbAnchors * mNbClass * mOutputDims.d[0]*sizeof(float);
        size_t anchorsSize = 4*mNbAnchors*mNbClass*sizeof(WDATA_T) ; // mNbAnchors and (x0 y0 x1 y1) * mNbAnchors + mScoreCls

        size_t finalIdxSize = (mNbProposals*5*mOutputDims.d[0])*2*sizeof(float) + mOutputDims.d[0]*sizeof(unsigned int);

        mSerializationSize = proposalParamI + proposalParamD + 3*PixelMapSize + 10*M_Index + anchorsSize + finalIdxSize;

        return mSerializationSize;
	}

	virtual void serialize(void* buffer) override
	{
        char* d = reinterpret_cast<char*>(buffer);
        char* a = d;
        write<int>(d, (int)mOutputDims.d[0]);
        write<int>(d, (int)mOutputDims.d[1]);
        write<int>(d, (int)mOutputDims.d[2]);
        write<int>(d, (int)mOutputDims.d[3]);
        write<int>(d, (int)mChannelHeight);
        write<int>(d, (int)mChannelWidth);
        write<unsigned int>(d, mStimuliWidth);
        write<unsigned int>(d, mStimuliHeight);
        write<unsigned int>(d, mFeatureMapWidth);
        write<unsigned int>(d, mFeatureMapHeight);
        write<int>(d, (int)mNbAnchors);
        write<int>(d, (int)mNbClass);
        write<int>(d, (int)mNbProposals);
        write<int>(d, (unsigned int)mMaxParts);
        write<int>(d, (unsigned int)mMaxTemplates);
        write<double>(d, mNMS_IoU);
        write<int>(d, mThreadX);
        write<int>(d, mThreadY);
        write<int>(d, mThreadZ);
        write<int>(d, mBlockX);
        write<int>(d, mBlockY);
        write<int>(d, mBlockZ);
        serializeFromDevice<int>(d, mPixelMap, mChannelHeight * mChannelWidth * mNbAnchors * mNbClass * mOutputDims.d[0]);
        serializeFromDevice<int>(d, mPixelMapSorted, mChannelHeight * mChannelWidth * mNbAnchors * mNbClass * mOutputDims.d[0]);
        serializeFromDevice<int>(d, mScoresIndex, mChannelHeight * mChannelWidth * mNbAnchors * mNbClass * mOutputDims.d[0]);
        serializeFromDevice<float>(d, mScores, mChannelHeight * mChannelWidth * mNbAnchors * mNbClass * mOutputDims.d[0]);
        serializeFromDevice<float>(d, mScoresFiltered, mChannelHeight * mChannelWidth * mNbAnchors * mNbClass * mOutputDims.d[0]);

        serializeFromDevice<float>(d, mMxGPUIndex, mChannelHeight * mChannelWidth * mNbAnchors * mNbClass * mOutputDims.d[0]);
        serializeFromDevice<float>(d, mMyGPUIndex, mChannelHeight * mChannelWidth * mNbAnchors * mNbClass * mOutputDims.d[0]);
        serializeFromDevice<float>(d, mMwGPUIndex, mChannelHeight * mChannelWidth * mNbAnchors * mNbClass * mOutputDims.d[0]);
        serializeFromDevice<float>(d, mMhGPUIndex, mChannelHeight * mChannelWidth * mNbAnchors * mNbClass * mOutputDims.d[0]);
        for(unsigned int k = 0; k < mChannelHeight * mChannelWidth * mNbAnchors * mNbClass * mOutputDims.d[0]; ++k)
            write<float>(d, mMxCPUIndex[k]);
        for(unsigned int k = 0; k < mChannelHeight * mChannelWidth * mNbAnchors * mNbClass * mOutputDims.d[0]; ++k)
            write<float>(d, mMyCPUIndex[k]);
        for(unsigned int k = 0; k < mChannelHeight * mChannelWidth * mNbAnchors * mNbClass * mOutputDims.d[0]; ++k)
            write<float>(d, mMwCPUIndex[k]);
        for(unsigned int k = 0; k < mChannelHeight * mChannelWidth * mNbAnchors * mNbClass * mOutputDims.d[0]; ++k)
            write<float>(d, mMhCPUIndex[k]);
        
        serializeFromDevice<float>(d, mScoreThreshold, mNbClass);

        for(unsigned int k = 0; k < mNbClass; ++k)
            write<unsigned int>(d, mPartsPerClass[k]);

        for(unsigned int k = 0; k < mNbClass; ++k)
            write<unsigned int>(d, mTemplatesPerClass[k]);

        /*for(unsigned int i = 0; i < mNbAnchors; ++i)
        {
            write<WDATA_T>(d, mAnchors[i].x0);
            write<WDATA_T>(d, mAnchors[i].y0);
            write<WDATA_T>(d, mAnchors[i].x1);
            write<WDATA_T>(d, mAnchors[i].y1);
        }*/
        serializeFromDevice<float>(d, mAnchors, 4 * mNbAnchors * mNbClass);

        serializeFromDevice<float>(d, mROIsBBOxFinal, mNbProposals*5*mOutputDims.d[0]);
        serializeFromDevice<float>(d, mROIsMapAnchorsFinal, mNbProposals*5*mOutputDims.d[0]);
        serializeFromDevice<unsigned int>(d, mROIsIndexFinal, mOutputDims.d[0]);

        assert(d == a + getSerializationSize());
	}

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
    template<typename T>
    T* deserializeToDevice(const char*& hostBuffer, size_t dataSize)
    {
        T* gpuData;
        checkCudaErrors(cudaMalloc(&gpuData, dataSize*sizeof(T)));
        checkCudaErrors(cudaMemcpy(gpuData, hostBuffer, dataSize*sizeof(T), cudaMemcpyHostToDevice));
        hostBuffer += dataSize*sizeof(T);
        return gpuData;
    }

    template<typename T>
    void serializeFromDevice(char*& hostBuffer, T* deviceWeights, size_t dataSize)
    {
        checkCudaErrors(cudaMemcpy(hostBuffer, deviceWeights, dataSize*sizeof(T), cudaMemcpyDeviceToHost));
        hostBuffer += dataSize*sizeof(T);
    }
    void gpuThreadAllocation()
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, 0);

        const unsigned int outputMaxSizePerCls = mNbAnchors * mChannelWidth * mChannelHeight;
        const unsigned int nbBlocks = std::ceil(outputMaxSizePerCls/(float) 32.0);


        const unsigned int maxSize
            = (unsigned int)deviceProp.maxThreadsPerBlock;
        const unsigned int prefMultiple = (unsigned int)deviceProp.warpSize;

        mThreadX = 32;
        mThreadY = 1;
        mThreadZ = 1;

        mBlockX = nbBlocks;
        mBlockY = mNbClass;
        mBlockZ = (int) mOutputDims.d[0];
        std::cout << "ObjectDet: "
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
    unsigned int mStimuliWidth;
    unsigned int mStimuliHeight;
    unsigned int mFeatureMapWidth;
    unsigned int mFeatureMapHeight;
    unsigned int mChannelHeight;
    unsigned int mChannelWidth;
    unsigned int mNbProposals;
    unsigned int mNbAnchors;
    unsigned int mNbClass;
    double mNMS_IoU;
    float* mScoreThreshold;
    unsigned int mMaxParts;
    unsigned int mMaxTemplates;
    unsigned int* mPartsPerClass;
    unsigned int* mTemplatesPerClass;
    float* mAnchors;

    float* mROIsBBOxFinal;
    float* mROIsMapAnchorsFinal;
    unsigned int* mROIsIndexFinal;

    size_t mSerializationSize;

    int* mPixelMapSorted;
    int* mPixelMap;

    int* mScoresIndex;
    float* mScores;
    float* mScoresFiltered;

    float* mMxGPUIndex;
    float* mMyGPUIndex;    
    float* mMwGPUIndex;
    float* mMhGPUIndex;

    float* mMxCPUIndex;
    float* mMyCPUIndex;    
    float* mMwCPUIndex;
    float* mMhCPUIndex;

    int mThreadX;
    int mThreadY;
    int mThreadZ;
    int mBlockX;
    int mBlockY;
    int mBlockZ;
};


/**Plugin Layer implementation**/
/**BatchNormalisation CUDNN implementation**/
class BatchNormCUDNNPlugin: public nvinfer1::IPlugin
{
public:
	BatchNormCUDNNPlugin(unsigned int batchSize,
                    unsigned int nbOutputs,
                    unsigned int outputHeight,
                    unsigned int outputWidth,
                    WDATA_T* scales,
                    WDATA_T* biases,
                    WDATA_T* means,
                    WDATA_T* variances,
                    WDATA_T epsilon)
	{

        mOutputDims.d[0] = batchSize;
        mOutputDims.d[1] = nbOutputs;
        mOutputDims.d[2] = outputHeight;
        mOutputDims.d[3] = outputWidth;

        /**Initialize epsilon parameter**/
        mEpsilon = epsilon;

        /**Initialize scale parameters on GPU **/
        checkCudaErrors( cudaMalloc((void**)&mScalesCuda,
                         nbOutputs*sizeof(WDATA_T)) );
        checkCudaErrors( cudaMemcpy(mScalesCuda,
                         scales,
                         nbOutputs*sizeof(WDATA_T),
                         cudaMemcpyHostToDevice) );

        /**Initialize bias parameters on GPU **/
        checkCudaErrors( cudaMalloc((void**)&mBiasesCuda,
                         nbOutputs*sizeof(WDATA_T)) );
        checkCudaErrors( cudaMemcpy(mBiasesCuda,
                         biases,
                         nbOutputs*sizeof(WDATA_T),
                         cudaMemcpyHostToDevice) );

        /**Initialize means parameters on GPU **/
        checkCudaErrors( cudaMalloc((void**)&mMeansCuda,
                         nbOutputs*sizeof(WDATA_T)) );
        checkCudaErrors( cudaMemcpy(mMeansCuda,
                         means,
                         nbOutputs*sizeof(WDATA_T),
                         cudaMemcpyHostToDevice) );

        /**Initialize variance parameters on GPU **/
        checkCudaErrors( cudaMalloc((void**)&mVariancesCuda,
                         nbOutputs*sizeof(WDATA_T)) );
        checkCudaErrors( cudaMemcpy(mVariancesCuda,
                         variances,
                         nbOutputs*sizeof(WDATA_T),
                         cudaMemcpyHostToDevice) );

	}

	BatchNormCUDNNPlugin(const void* dataFromRuntime, size_t length)
	{
		const char* d = reinterpret_cast<const char*>(dataFromRuntime), *a = d;
        mOutputDims.d[0] = read<int>(d);
		mOutputDims.d[1] = read<int>(d);
		mOutputDims.d[2] = read<int>(d);
		mOutputDims.d[3] = read<int>(d);
        mInputDescriptor = read<cudnnTensorDescriptor_t>(d);
        mOutputDescriptor = read<cudnnTensorDescriptor_t>(d);
        mScaleDescriptor = read<cudnnTensorDescriptor_t>(d);
    	mEpsilon = read<WDATA_T>(d);
		mScalesCuda = deserializeToDevice(d, mOutputDims.d[1]);
		mBiasesCuda = deserializeToDevice(d, mOutputDims.d[1]);
		mMeansCuda = deserializeToDevice(d, mOutputDims.d[1]);
		mVariancesCuda = deserializeToDevice(d, mOutputDims.d[1]);
		assert(d == a + length);
	}

	~BatchNormCUDNNPlugin()
	{
        checkCudaErrors(cudaFree(mVariancesCuda));
        checkCudaErrors(cudaFree(mMeansCuda));
        checkCudaErrors(cudaFree(mBiasesCuda));
        checkCudaErrors(cudaFree(mScalesCuda));
        CHECK_CUDNN_STATUS(cudnnDestroyTensorDescriptor(mScaleDescriptor));
        CHECK_CUDNN_STATUS(cudnnDestroyTensorDescriptor(mInputDescriptor));
        CHECK_CUDNN_STATUS(cudnnDestroyTensorDescriptor(mOutputDescriptor));

	}

	virtual int getNbOutputs() const override
	{
        return 1;
	}

	virtual nvinfer1::Dims getOutputDimensions(int index,
                                               const nvinfer1::Dims* inputDim,
                                               int nbInputDims) override
	{
        unsigned int batchInput = 1;

        if(inputDim[0].nbDims == 4)
            batchInput = inputDim[0].d[0];

        return nvinfer1::DimsNCHW(batchInput, mOutputDims.d[1], mOutputDims.d[2], mOutputDims.d[3]);
	}

	virtual void configure(const nvinfer1::Dims* inputDims,
                   int nbInputs,
                   const nvinfer1::Dims* outputDims,
                   int /*nbOutputs*/,
                   int maxBatchSize) override
	{
        unsigned int batchSize = maxBatchSize;
        unsigned int nbOutputs;
	    unsigned int outputHeight;
        unsigned int outputWidth;

        if(inputDims[0].nbDims == 4)
	    {
            batchSize *= inputDims[0].d[0];
	        nbOutputs = inputDims[0].d[1];
	        outputHeight = inputDims[0].d[2];
	        outputWidth = inputDims[0].d[3];
	    }
	    else
	    {
            nbOutputs = inputDims[0].d[0];
            outputHeight = inputDims[0].d[1];
            outputWidth = inputDims[0].d[2];
        }

        mOutputDims.d[0] = batchSize;

        cudnnDataType_t context_dataType = CUDNN_DATA_FLOAT;
        cudnnTensorFormat_t context_tensorFormat = CUDNN_TENSOR_NCHW;
        cudnnBatchNormMode_t mMode = CUDNN_BATCHNORM_SPATIAL;

        cudnnCreateTensorDescriptor(&mInputDescriptor);
        cudnnCreateTensorDescriptor(&mOutputDescriptor);
        cudnnCreateTensorDescriptor(&mScaleDescriptor);

        CHECK_CUDNN_STATUS(cudnnSetTensor4dDescriptor(mInputDescriptor,
                                                      context_tensorFormat,
                                                      context_dataType,
                                                      mOutputDims.d[0],
                                                      nbOutputs,
                                                      outputHeight,
                                                      outputWidth));

        CHECK_CUDNN_STATUS(cudnnSetTensor4dDescriptor(mOutputDescriptor,
                                                      context_tensorFormat,
                                                      context_dataType,
                                                      mOutputDims.d[0],
                                                      nbOutputs,
                                                      outputHeight,
                                                      outputWidth));

        cudnnTensorDescriptor_t derivedBnDesc;
        CHECK_CUDNN_STATUS(cudnnCreateTensorDescriptor(&derivedBnDesc));
        CHECK_CUDNN_STATUS(
            cudnnDeriveBNTensorDescriptor(derivedBnDesc, mInputDescriptor, mMode));

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
                mScaleDescriptor, context_tensorFormat, context_dataType, n, c, h, w));
	}

	virtual int initialize() override
	{
        CHECK_CUDNN_STATUS(cudnnCreate(&mCudnnContext));
		return 0;
	}

	virtual void terminate() override
	{
		CHECK_CUDNN_STATUS(cudnnDestroy(mCudnnContext));
	}

	virtual size_t getWorkspaceSize(int maxBatchSize) const override
	{
		return 0;
	}

	virtual int enqueue(int batchSize,
                        const void*const * inputs,
                        void** outputs,
                        void* workspace,
                        cudaStream_t stream) override
	{
        cudnnSetStream(mCudnnContext, stream);
        cudnnBatchNormMode_t mMode = CUDNN_BATCHNORM_SPATIAL;
        DATA_T ONE_T = DATA_T(1); // Alpha must be set to 1 for all steps
        DATA_T ZERO_T = DATA_T(0); // Beta must be set to 0 for POOLING FORWARD

        CHECK_CUDNN_STATUS(
            cudnnBatchNormalizationForwardInference(mCudnnContext,
                                                    mMode,
                                                    &ONE_T,
                                                    &ZERO_T,
                                                    mInputDescriptor,
                                                    reinterpret_cast<const float*>(inputs[0]),
                                                    mOutputDescriptor,
                                                    //*outputs,
                                                    outputs[0],
                                                    mScaleDescriptor,
                                                    mScalesCuda,
                                                    mBiasesCuda,
                                                    mMeansCuda,
                                                    mVariancesCuda,
                                                    mEpsilon));

        return 0;
	}

	virtual size_t getSerializationSize() override
	{
        size_t inputDimParamSize = 4*sizeof(int); //nbOutputs, nbOutputsHeight, nbOutputWidth = 3
        size_t scaleDescriptorSize = sizeof(cudnnTensorDescriptor_t)*3;
        size_t biasParamSize = sizeof(WDATA_T)*mOutputDims.d[1];
        size_t variancesParamSize = sizeof(WDATA_T)*mOutputDims.d[1];
        size_t meansParamSize = sizeof(WDATA_T)*mOutputDims.d[1];
        size_t scalesParamSize = sizeof(WDATA_T)*mOutputDims.d[1];
        size_t epsilonParamSize = sizeof(WDATA_T);

        mSerializationSize = inputDimParamSize + biasParamSize + scaleDescriptorSize
                                + variancesParamSize + scalesParamSize
                                + meansParamSize + epsilonParamSize;
        return mSerializationSize;
	}

	virtual void serialize(void* buffer) override
	{
        char* d = reinterpret_cast<char*>(buffer);
        char* a = d;
        write<int>(d, (int)mOutputDims.d[0]);
        write<int>(d, (int)mOutputDims.d[1]);
        write<int>(d, (int)mOutputDims.d[2]);
        write<int>(d, (int)mOutputDims.d[3]);
        write<cudnnTensorDescriptor_t>(d, mInputDescriptor);
        write<cudnnTensorDescriptor_t>(d, mOutputDescriptor);
        write<cudnnTensorDescriptor_t>(d, mScaleDescriptor);
        write<WDATA_T>(d, mEpsilon);
        serializeFromDevice(d, mScalesCuda, mOutputDims.d[1]);
        serializeFromDevice(d, mBiasesCuda, mOutputDims.d[1]);
        serializeFromDevice(d, mMeansCuda, mOutputDims.d[1]);
        serializeFromDevice(d, mVariancesCuda, mOutputDims.d[1]);
        assert(d == a + getSerializationSize());
	}

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

    WDATA_T* deserializeToDevice(const char*& hostBuffer, size_t dataSize)
    {
        WDATA_T* gpuData;
        checkCudaErrors(cudaMalloc(&gpuData, dataSize*sizeof(WDATA_T)));
        checkCudaErrors(cudaMemcpy(gpuData, hostBuffer, dataSize*sizeof(WDATA_T), cudaMemcpyHostToDevice));
        hostBuffer += dataSize*sizeof(WDATA_T);
        return gpuData;
    }

    void serializeFromDevice(char*& hostBuffer, WDATA_T* deviceWeights, size_t dataSize)
    {
        checkCudaErrors(cudaMemcpy(hostBuffer, deviceWeights, dataSize*sizeof(WDATA_T), cudaMemcpyDeviceToHost));
        hostBuffer += dataSize*sizeof(WDATA_T);
    }

    nvinfer1::Dims mOutputDims;
    WDATA_T mEpsilon;
    WDATA_T* mScalesCuda;
    WDATA_T* mBiasesCuda;
    WDATA_T* mMeansCuda;
    WDATA_T* mVariancesCuda;
    cudnnTensorDescriptor_t mInputDescriptor;
    cudnnTensorDescriptor_t mOutputDescriptor;
    cudnnTensorDescriptor_t mScaleDescriptor;
	cudnnHandle_t mCudnnContext;
    size_t mSerializationSize;

};




/**Plugin Layer implementation**/
/**BatchNormalisation CUDA implementation**/
class BatchNormPlugin: public nvinfer1::IPlugin
{
public:
	BatchNormPlugin(unsigned int batchSize,
                    unsigned int nbOutputs,
                    unsigned int outputHeight,
                    unsigned int outputWidth,
                    WDATA_T* scales,
                    WDATA_T* biases,
                    WDATA_T* means,
                    WDATA_T* variances,
                    WDATA_T epsilon)
	{
        mOutputDims.d[0] = batchSize;
        mOutputDims.d[1] = nbOutputs;
        mOutputDims.d[2] = outputHeight;
        mOutputDims.d[3] = outputWidth;


        /**Initialize epsilon parameter**/
        mEpsilon = epsilon;

        /**Initialize scale parameters on GPU **/
        checkCudaErrors( cudaMalloc((void**)&mScalesCuda,
                         nbOutputs*sizeof(WDATA_T)) );
        checkCudaErrors( cudaMemcpy(mScalesCuda,
                         scales,
                         nbOutputs*sizeof(WDATA_T),
                         cudaMemcpyHostToDevice) );

        /**Initialize bias parameters on GPU **/
        checkCudaErrors( cudaMalloc((void**)&mBiasesCuda,
                         nbOutputs*sizeof(WDATA_T)) );
        checkCudaErrors( cudaMemcpy(mBiasesCuda,
                         biases,
                         nbOutputs*sizeof(WDATA_T),
                         cudaMemcpyHostToDevice) );

        /**Initialize means parameters on GPU **/
        checkCudaErrors( cudaMalloc((void**)&mMeansCuda,
                         nbOutputs*sizeof(WDATA_T)) );
        checkCudaErrors( cudaMemcpy(mMeansCuda,
                         means,
                         nbOutputs*sizeof(WDATA_T),
                         cudaMemcpyHostToDevice) );

        /**Initialize variance parameters on GPU **/
        checkCudaErrors( cudaMalloc((void**)&mVariancesCuda,
                         nbOutputs*sizeof(WDATA_T)) );
        checkCudaErrors( cudaMemcpy(mVariancesCuda,
                         variances,
                         nbOutputs*sizeof(WDATA_T),
                         cudaMemcpyHostToDevice) );

	}

	BatchNormPlugin(const void* dataFromRuntime, size_t length)
	{
		const char* d = reinterpret_cast<const char*>(dataFromRuntime), *a = d;
        mOutputDims.d[0] = read<int>(d);
		mOutputDims.d[1] = read<int>(d);
		mOutputDims.d[2] = read<int>(d);
		mOutputDims.d[3] = read<int>(d);
    	mEpsilon = read<WDATA_T>(d);
		mScalesCuda = deserializeToDevice(d, mOutputDims.d[1]);
		mBiasesCuda = deserializeToDevice(d, mOutputDims.d[1]);
		mMeansCuda = deserializeToDevice(d, mOutputDims.d[1]);
		mVariancesCuda = deserializeToDevice(d, mOutputDims.d[1]);
		mThreadX = read<int>(d);
		mThreadY = read<int>(d);
		mThreadZ = read<int>(d);
    	mBlockX = read<int>(d);
		mBlockY = read<int>(d);
		mBlockZ = read<int>(d);
		assert(d == a + length);

		mNbThreads.x = mThreadX;
		mNbThreads.y = mThreadY;
		mNbThreads.z = mThreadZ;
		mNbBlocks.x = mBlockX;
		mNbBlocks.y = mBlockY;
		mNbBlocks.z = mBlockZ;
	}

	~BatchNormPlugin()
	{
        checkCudaErrors(cudaFree(mVariancesCuda));
        checkCudaErrors(cudaFree(mMeansCuda));
        checkCudaErrors(cudaFree(mBiasesCuda));
        checkCudaErrors(cudaFree(mScalesCuda));

	}

	virtual int getNbOutputs() const override
	{
        return 1;
	}

	virtual nvinfer1::Dims getOutputDimensions(int index,
                                               const nvinfer1::Dims* inputDim,
                                               int nbInputDims) override
	{
        unsigned int batchInput = 1;

        if(inputDim[0].nbDims == 4)
            batchInput = inputDim[0].d[0];

        return nvinfer1::DimsNCHW(batchInput, mOutputDims.d[1], mOutputDims.d[2], mOutputDims.d[3]);
	}

	virtual void configure(const nvinfer1::Dims* inputDims,
                   int nbInputs,
                   const nvinfer1::Dims* outputDims,
                   int nbOutputs,
                   int maxBatchSize) override
	{
        unsigned int batchSize = maxBatchSize;
        if(inputDims[0].nbDims == 4)
        {
                batchSize *= inputDims[0].d[0];
        }
        else
        {
            nbOutputs = inputDims[0].d[0];
        }
        mOutputDims.d[0] = batchSize;

        /**GPU thread and block allocation***/
        gpuThreadAllocation();
	}

	virtual int initialize() override
	{
		return 0;
	}

	virtual void terminate() override
	{

	}

	virtual size_t getWorkspaceSize(int maxBatchSize) const override
	{
		return 0;
	}

	virtual int enqueue(int batchSize,
                        const void*const * inputs,
                        void** outputs,
                        void* workspace,
                        cudaStream_t stream) override
	{
        cuda_batchnormcell_propagate(mOutputDims.d[1],
                                     mOutputDims.d[2],
                                     mOutputDims.d[3],
                                     reinterpret_cast<const DATA_T*>(inputs[0]),
                                     mOutputDims.d[1],
                                     0, /*outputoffset*/
                                     reinterpret_cast<DATA_T*>(outputs[0]),
                                     reinterpret_cast<const WDATA_T*>(mBiasesCuda),
                                     reinterpret_cast<const WDATA_T*>(mVariancesCuda),
                                     reinterpret_cast<const WDATA_T*>(mMeansCuda),
                                     reinterpret_cast<const WDATA_T*>(mScalesCuda),
                                     mEpsilon,
                                     mNbThreads,
                                     mNbBlocks,
                                     stream);
        return 0;
	}

	virtual size_t getSerializationSize() override
	{
        size_t inputDimParamSize = 4*sizeof(int); //nbOutputs, nbOutputsHeight, nbOutputWidth = 3
        size_t biasParamSize = sizeof(WDATA_T)*mOutputDims.d[1];
        size_t variancesParamSize = sizeof(WDATA_T)*mOutputDims.d[1];
        size_t meansParamSize = sizeof(WDATA_T)*mOutputDims.d[1];
        size_t scalesParamSize = sizeof(WDATA_T)*mOutputDims.d[1];
        size_t epsilonParamSize = sizeof(WDATA_T);
        size_t threadParam = sizeof(int)*3; //threadX, threadY, threadZ = 3
        size_t blockParam = sizeof(int)*3; //blockX, blockY, blockZ = 3

        mSerializationSize = inputDimParamSize + biasParamSize
                                + variancesParamSize + scalesParamSize
                                + meansParamSize + epsilonParamSize
                                + threadParam + blockParam;
        return mSerializationSize;
	}

	virtual void serialize(void* buffer) override
	{
        char* d = reinterpret_cast<char*>(buffer);
        char* a = d;
        write<int>(d, (int)mOutputDims.d[0]);
        write<int>(d, (int)mOutputDims.d[1]);
        write<int>(d, (int)mOutputDims.d[2]);
        write<int>(d, (int)mOutputDims.d[3]);
        write<WDATA_T>(d, mEpsilon);
        serializeFromDevice(d, mScalesCuda, mOutputDims.d[1]);
        serializeFromDevice(d, mBiasesCuda, mOutputDims.d[1]);
        serializeFromDevice(d, mMeansCuda, mOutputDims.d[1]);
        serializeFromDevice(d, mVariancesCuda, mOutputDims.d[1]);
        write<int>(d, mThreadX);
        write<int>(d, mThreadY);
        write<int>(d, mThreadZ);
        write<int>(d, mBlockX);
        write<int>(d, mBlockY);
        write<int>(d, mBlockZ);
        assert(d == a + getSerializationSize());
	}

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

    WDATA_T* deserializeToDevice(const char*& hostBuffer, size_t dataSize)
    {
        WDATA_T* gpuData;
        checkCudaErrors(cudaMalloc(&gpuData, dataSize*sizeof(WDATA_T)));
        checkCudaErrors(cudaMemcpy(gpuData, hostBuffer, dataSize*sizeof(WDATA_T), cudaMemcpyHostToDevice));
        hostBuffer += dataSize*sizeof(WDATA_T);
        return gpuData;
    }

    void serializeFromDevice(char*& hostBuffer, WDATA_T* deviceWeights, size_t dataSize)
    {
        checkCudaErrors(cudaMemcpy(hostBuffer, deviceWeights, dataSize*sizeof(WDATA_T), cudaMemcpyDeviceToHost));
        hostBuffer += dataSize*sizeof(WDATA_T);
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

        mNbThreads = {groupWidth, groupSize / groupWidth, 1};
        mNbBlocks = {nbOutputs, 1, batchSize};
        std::cout << "BatchNormalization"
                  << ":\n"
                     "    Max. Threads per Blocks = " << maxSize
                  << "\n"
                     "    Preferred Blocks Size multiple = " << prefMultiple
                  << "\n"
                     "    Blocks size = (" << mNbThreads.x << ", "
                  << mNbThreads.y << ", " << mNbThreads.z
                  << ") = "
                  << std::max<unsigned long>(mNbThreads.x, 1UL)
                     * std::max<unsigned long>(mNbThreads.y, 1UL)
                     * std::max<unsigned long>(mNbThreads.z, 1UL)
                  << "\n"
                     "    Grid size = (" << mNbBlocks.x << ", "
                  << mNbBlocks.y << ", " << mNbBlocks.z << ") = "
                  << std::max<unsigned long>(mNbBlocks.x, 1UL)
                     * std::max<unsigned long>(mNbBlocks.y, 1UL)
                     * std::max<unsigned long>(mNbBlocks.z, 1UL) << "\n"
                  << "    Multi-Processors used = "
                  << (mNbBlocks.x)
                     * (std::max<unsigned long>(mNbBlocks.y, 1UL))
                     * (std::max<unsigned long>(mNbBlocks.z, 1UL))
                  << std::endl;


    }

    dim3 mNbThreads;
    int mThreadX;
    int mThreadY;
    int mThreadZ;
    dim3 mNbBlocks;
    int mBlockX;
    int mBlockY;
    int mBlockZ;
    nvinfer1::Dims mOutputDims;
    WDATA_T mEpsilon;
    WDATA_T* mScalesCuda;
    WDATA_T* mBiasesCuda;
    WDATA_T* mMeansCuda;
    WDATA_T* mVariancesCuda;
    size_t mSerializationSize;

};

/**Plugin Layer implementation**/
/**Resize GPU implementation**/
class ResizeGPUPlugin: public nvinfer1::IPlugin
{

public:
	ResizeGPUPlugin(unsigned int batchSize,
                        unsigned int nbOutputs,
                        unsigned int outputHeight,
                        unsigned int outputWidth,
                        unsigned int featureHeight,
                        unsigned int featureWidth,
                        Pooling_T resizeType,
                        bool aligCorner)
	{

        mOutputDims.d[0] = batchSize;
        mOutputDims.d[1] = nbOutputs;
        mOutputDims.d[2] = outputHeight;
        mOutputDims.d[3] = outputWidth;
        mInputHeight = featureHeight;
        mInputWidth = featureWidth;
        mResizeType = resizeType;
        mAlignCorner = aligCorner;

        /***Initialization of the scale and strideIndex for the X,Y dimensions**/
        mScaleX = mAlignCorner ? (mInputWidth - 1) / (float) (mOutputDims.d[3] - 1)
                    : (mInputWidth) / (float) (mOutputDims.d[3]);

        mScaleY = mAlignCorner ? (mInputHeight - 1) / (float) (mOutputDims.d[2] - 1)
                    : (mInputHeight) / (float) (mOutputDims.d[2]);

        std::vector<unsigned int> yStrideLowIndex(mOutputDims.d[2] + 1, 0);
        std::vector<unsigned int> yStrideHightIndex(mOutputDims.d[2] + 1, 0);
        std::vector<float> yStrideInterpolation(mOutputDims.d[2] + 1, 0.0);

        std::vector<unsigned int> xStrideLowIndex(mOutputDims.d[3] + 1, 0);
        std::vector<unsigned int> xStrideHightIndex(mOutputDims.d[3] + 1, 0);
        std::vector<float> xStrideInterpolation(mOutputDims.d[3] + 1, 0.0);

        BilinearInterpolation(  mOutputDims.d[2],
                                mInputHeight,
                                mScaleY,
                                yStrideLowIndex,
                                yStrideHightIndex,
                                yStrideInterpolation);

        BilinearInterpolation(  mOutputDims.d[3],
                                mInputWidth,
                                mScaleX,
                                xStrideLowIndex,
                                xStrideHightIndex,
                                xStrideInterpolation);

        /**Initialize interpolation indexes parameters on GPU **/
        checkCudaErrors( cudaMalloc((void**)&mYStrideLowIndex,
                         (mOutputDims.d[2] + 1)*sizeof(unsigned int)) );
        checkCudaErrors( cudaMalloc((void**)&mYStrideHightIndex,
                         (mOutputDims.d[2] + 1)*sizeof(unsigned int)) );
        checkCudaErrors( cudaMalloc((void**)&mYStrideInterpolation,
                         (mOutputDims.d[2] + 1)*sizeof(float)) );
        checkCudaErrors( cudaMalloc((void**)&mXStrideLowIndex,
                         (mOutputDims.d[3] + 1)*sizeof(unsigned int)) );
        checkCudaErrors( cudaMalloc((void**)&mXStrideHightIndex,
                         (mOutputDims.d[3] + 1)*sizeof(unsigned int)) );
        checkCudaErrors( cudaMalloc((void**)&mXStrideInterpolation,
                         (mOutputDims.d[3] + 1)*sizeof(float)) );

        checkCudaErrors( cudaMemcpy(mYStrideLowIndex,
                         yStrideLowIndex.data(),
                         (mOutputDims.d[2] + 1)*sizeof(unsigned int),
                         cudaMemcpyHostToDevice) );
        checkCudaErrors( cudaMemcpy(mYStrideHightIndex,
                         yStrideHightIndex.data(),
                         (mOutputDims.d[2] + 1)*sizeof(unsigned int),
                         cudaMemcpyHostToDevice) );
        checkCudaErrors( cudaMemcpy(mYStrideInterpolation,
                         yStrideInterpolation.data(),
                         (mOutputDims.d[2] + 1)*sizeof(float),
                         cudaMemcpyHostToDevice) );
        checkCudaErrors( cudaMemcpy(mXStrideLowIndex,
                         xStrideLowIndex.data(),
                         (mOutputDims.d[3] + 1)*sizeof(unsigned int),
                         cudaMemcpyHostToDevice) );
        checkCudaErrors( cudaMemcpy(mXStrideHightIndex,
                         xStrideHightIndex.data(),
                         (mOutputDims.d[3] + 1)*sizeof(unsigned int),
                         cudaMemcpyHostToDevice) );
        checkCudaErrors( cudaMemcpy(mXStrideInterpolation,
                         xStrideInterpolation.data(),
                         (mOutputDims.d[3] + 1)*sizeof(float),
                         cudaMemcpyHostToDevice) );
        yStrideLowIndex.erase( yStrideLowIndex.begin(),
                               yStrideLowIndex.end());
        yStrideHightIndex.erase( yStrideHightIndex.begin(),
                                yStrideHightIndex.end());
        xStrideLowIndex.erase( xStrideLowIndex.begin(),
                               xStrideLowIndex.end());
        xStrideHightIndex.erase( xStrideHightIndex.begin(),
                                xStrideHightIndex.end());
        xStrideInterpolation.erase( xStrideInterpolation.begin(),
                                    xStrideInterpolation.end());
        yStrideInterpolation.erase( yStrideInterpolation.begin(),
                                    yStrideInterpolation.end());

        gpuThreadAllocation();
	}

	ResizeGPUPlugin(const void* dataFromRuntime, size_t length)
	{
		const char* d = reinterpret_cast<const char*>(dataFromRuntime), *a = d;
        mOutputDims.d[0] = read<int>(d);
		mOutputDims.d[1] = read<int>(d);
		mOutputDims.d[2] = read<int>(d);
		mOutputDims.d[3] = read<int>(d);
        mInputHeight = read<int>(d);
        mInputWidth = read<int>(d);
        mResizeType= read <Pooling_T>(d);
        mScaleX = read<float>(d);
        mScaleY = read<float>(d);

        mYStrideLowIndex = deserializeToDevice<unsigned int>(d, mOutputDims.d[2] + 1);
        mYStrideHightIndex = deserializeToDevice<unsigned int>(d, mOutputDims.d[2] + 1);
        mYStrideInterpolation = deserializeToDevice<float>(d, mOutputDims.d[2] + 1);

        mXStrideLowIndex = deserializeToDevice<unsigned int>(d, mOutputDims.d[3] + 1);
        mXStrideHightIndex = deserializeToDevice<unsigned int>(d, mOutputDims.d[3] + 1);
        mXStrideInterpolation = deserializeToDevice<float>(d, mOutputDims.d[3] + 1);

		mThreadX = read<int>(d);
		mThreadY = read<int>(d);
		mThreadZ = read<int>(d);
    	mBlockX = read<int>(d);
		mBlockY = read<int>(d);
		mBlockZ = read<int>(d);

		assert(d == a + length);
	}

	~ResizeGPUPlugin()
	{
        checkCudaErrors(cudaFree(mYStrideLowIndex));
        checkCudaErrors(cudaFree(mYStrideHightIndex));
        checkCudaErrors(cudaFree(mYStrideInterpolation));
        checkCudaErrors(cudaFree(mXStrideLowIndex));
        checkCudaErrors(cudaFree(mXStrideHightIndex));
        checkCudaErrors(cudaFree(mXStrideInterpolation));
	}

	virtual int getNbOutputs() const override
	{
        return (int) 1;
	}

	virtual nvinfer1::Dims getOutputDimensions(int index,
                                               const nvinfer1::Dims* inputDim,
                                               int nbInputDims) override
	{
        return nvinfer1::DimsCHW(   mOutputDims.d[1],
                                    mOutputDims.d[2],
                                    mOutputDims.d[3]);

	}

	virtual void configure(const nvinfer1::Dims* inputDims,
                   int nbInputs,
                   const nvinfer1::Dims* outputDims,
                   int nbOutputs,
                   int maxBatchSize) override
	{

	}

	virtual int initialize() override
	{
		return 0;
	}

	virtual void terminate() override
	{

	}

	virtual size_t getWorkspaceSize(int maxBatchSize) const override
	{
		return 0;
	}

	virtual int enqueue(int batchSize,
                        const void*const * inputs,
                        void** outputs,
                        void* workspace,
                        cudaStream_t stream) override
	{
        const dim3 nbBlocks = {mBlockX, mBlockY, mBlockZ};
        const dim3 nbThreads = {mThreadX, mThreadY, mThreadZ};

        if(mResizeType == BilinearTF)
        {
            cuda_resize_bilinearTF_propagate(   mOutputDims.d[3],
                                                mOutputDims.d[2],
                                                mOutputDims.d[1],
                                                mOutputDims.d[0],
                                                mInputWidth,
                                                mInputHeight,
                                                reinterpret_cast<unsigned int*>(mYStrideLowIndex),
                                                reinterpret_cast<unsigned int*>(mYStrideHightIndex),
                                                reinterpret_cast<float*>(mYStrideInterpolation),
                                                reinterpret_cast<unsigned int*>(mXStrideLowIndex),
                                                reinterpret_cast<unsigned int*>(mXStrideHightIndex),
                                                reinterpret_cast<float*>(mXStrideInterpolation),
                                                reinterpret_cast<const DATA_T*>(inputs[0]),
                                                reinterpret_cast<DATA_T*>(outputs[0]),
                                                nbBlocks,
                                                nbThreads,
                                                stream);
        }
        else
            throw std::runtime_error( "ResizeLayer: Only BilinearTF is implemented");


        return 0;
	}

	virtual size_t getSerializationSize() override
	{
        size_t intSize = 12*sizeof(int);
        size_t floatSize = 2*sizeof(float)
                            + (mOutputDims.d[2] + 1)*sizeof(float)
                            + (mOutputDims.d[3] + 1)*sizeof(float);
        size_t uintSize = (mOutputDims.d[2] + 1)*sizeof(unsigned int)*2
                        + (mOutputDims.d[3] + 1)*sizeof(unsigned int)*2;
        size_t finalSize = intSize + floatSize + uintSize + sizeof(Pooling_T);
        mSerializationSize = finalSize;

        return mSerializationSize;
	}

	virtual void serialize(void* buffer) override
	{
        char* d = reinterpret_cast<char*>(buffer);
        char* a = d;
        write<int>(d, (int)mOutputDims.d[0]);
        write<int>(d, (int)mOutputDims.d[1]);
        write<int>(d, (int)mOutputDims.d[2]);
        write<int>(d, (int)mOutputDims.d[3]);
        write<int>(d, (int)mInputHeight);
        write<int>(d, (int)mInputWidth);
        write<Pooling_T>(d, (Pooling_T)mResizeType);
        write<float>(d, (float)mScaleX);
        write<float>(d, (float)mScaleY);

        serializeFromDevice<unsigned int>(d, mYStrideLowIndex, mOutputDims.d[2] + 1);
        serializeFromDevice<unsigned int>(d, mYStrideHightIndex, mOutputDims.d[2] + 1);
        serializeFromDevice<float>(d, mYStrideInterpolation, mOutputDims.d[2] + 1);

        serializeFromDevice<unsigned int>(d, mXStrideLowIndex, mOutputDims.d[3] + 1);
        serializeFromDevice<unsigned int>(d, mXStrideHightIndex, mOutputDims.d[3] + 1);
        serializeFromDevice<float>(d, mXStrideInterpolation, mOutputDims.d[3] + 1);
        write<int>(d, (int)mThreadX);
        write<int>(d, (int)mThreadY);
        write<int>(d, (int)mThreadZ);
        write<int>(d, (int)mBlockX);
        write<int>(d, (int)mBlockY);
        write<int>(d, (int)mBlockZ);
        assert(d == a + getSerializationSize());
	}

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

    template<typename T>
    T* deserializeToDevice(const char*& hostBuffer, size_t dataSize)
    {
        T* gpuData;
        checkCudaErrors(cudaMalloc(&gpuData, dataSize*sizeof(T)));
        checkCudaErrors(cudaMemcpy(gpuData, hostBuffer, dataSize*sizeof(T), cudaMemcpyHostToDevice));
        hostBuffer += dataSize*sizeof(T);
        return gpuData;
    }

    template<typename T>
    void serializeFromDevice(char*& hostBuffer, T* deviceWeights, size_t dataSize)
    {
        checkCudaErrors(cudaMemcpy(hostBuffer, deviceWeights, dataSize*sizeof(T), cudaMemcpyDeviceToHost));
        hostBuffer += dataSize*sizeof(T);
    }

    void gpuThreadAllocation()
    {

        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, 0);

        const unsigned int maxSize = (unsigned int)deviceProp.maxThreadsPerBlock;
        const unsigned int prefMultiple = (unsigned int)deviceProp.warpSize;

        const unsigned int groupSize = (mOutputDims.d[3] * mOutputDims.d[2] < maxSize)
                                        ? mOutputDims.d[3] * mOutputDims.d[2]
                                        : maxSize;
        const unsigned int reqWidth = (unsigned int) ceilf((float) groupSize / (float) mOutputDims.d[3]);

        const unsigned int groupWidth = std::min(prefMultiple, reqWidth);

        dim3 block_size = {(unsigned int)mOutputDims.d[1], 1, (unsigned int)mOutputDims.d[0]};
        dim3 thread_size = {groupWidth, groupSize / groupWidth, 1};

        mThreadX = thread_size.x;
        mThreadY = thread_size.y;
        mThreadZ = thread_size.z;

        mBlockX = block_size.x;
        mBlockY = block_size.y;
        mBlockZ = block_size.z;


        std::cout << "Resize Layer:"
                    << ":\n"
                        "    Max. Threads per Blocks = " << maxSize
                    << "\n"
                        "    Preferred Blocks Size multiple = " << prefMultiple
                    << "\n"
                        "    Blocks size = (" << mBlockX << ", "
                    << mBlockY << ", " << mBlockZ
                    << ") = "
                    << std::max<unsigned long>(mBlockX, 1UL)
                        * std::max<unsigned long>(mBlockY, 1UL)
                        * std::max<unsigned long>(mBlockZ, 1UL)
                    << "\n"
                        "    Grid size = (" << mThreadX << ", "
                    << mThreadY << ", " << mThreadZ << ") = "
                    << std::max<unsigned long>(mThreadX, 1UL)
                        * std::max<unsigned long>(mThreadY, 1UL)
                        * std::max<unsigned long>(mThreadZ, 1UL) << "\n"
                    << "    Multi-Processors used = "
                    << (mBlockX)
                        * (std::max<unsigned long>(mBlockY, 1UL))
                        * (std::max<unsigned long>(mBlockZ, 1UL))
                    << std::endl;

    }
    void BilinearInterpolation( const int out_size,
                                const int in_size,
                                const float scale,
                                std::vector<unsigned int>& LowIndex,
                                std::vector<unsigned int>& HightIndex,
                                std::vector<float>& Interpolation)
    {
        LowIndex[out_size] = 0;
        HightIndex[out_size] = 0;

        for (int i = out_size - 1; i >= 0; --i) {
            const float in = i * scale;
            LowIndex[i] = (unsigned int) in;
            HightIndex[i] = std::min((int) in + 1, in_size - 1);
            Interpolation[i] = in - LowIndex[i];
        }

    }

    nvinfer1::Dims mOutputDims;
    unsigned int mInputWidth;
    unsigned int mInputHeight;

    unsigned int mThreadX;
    unsigned int mThreadY;
    unsigned int mThreadZ;
    unsigned int mBlockX;
    unsigned int mBlockY;
    unsigned int mBlockZ;

    unsigned int* mYStrideLowIndex;
    unsigned int* mYStrideHightIndex;
    float* mYStrideInterpolation;
    float mScaleY;

    unsigned int* mXStrideLowIndex;
    unsigned int* mXStrideHightIndex;
    float* mXStrideInterpolation;
    float mScaleX;

    Pooling_T mResizeType;
    bool mAlignCorner;

    size_t mSerializationSize;
};


struct pluginBatchnorm_CUDNN {
    std::vector<std::unique_ptr<BatchNormCUDNNPlugin>> mPlugin;
    int mPluginCount = 0;

    void add(unsigned int batchSize, unsigned int nbOutputs, unsigned int outputHeight, unsigned int outputWidth,
            WDATA_T* scales, WDATA_T* biases, WDATA_T* means, WDATA_T* variances, WDATA_T epsilon)

    {
        mPlugin.push_back(std::unique_ptr<BatchNormCUDNNPlugin>(new BatchNormCUDNNPlugin(batchSize,
                                                   nbOutputs,
                                                outputHeight,
                                                outputWidth,
                                                scales,
                                                biases,
                                                means,
                                                variances,
                                                epsilon)));
        ++mPluginCount;
    }

    void add(const void* serialData, size_t serialLength)
    {
        mPlugin.push_back(std::unique_ptr<BatchNormCUDNNPlugin>(new BatchNormCUDNNPlugin(serialData,serialLength)));
        ++mPluginCount;
    }

    nvinfer1::IPlugin* get()
    {
        return mPlugin.back().get();
    }
    void destroy()
    {
      for(int i = mPluginCount - 1; i >= 0; --i)
      {
        mPlugin[i].release();
        mPlugin[i] = nullptr;
      }
      mPlugin = std::vector<std::unique_ptr<BatchNormCUDNNPlugin>>();
      mPluginCount = 0;
    }
};

struct pluginBatchnorm_CUDA{
    std::vector<std::unique_ptr<BatchNormPlugin>> mPlugin;
    int mPluginCount = 0;

    void add(unsigned int batchSize, unsigned int nbOutputs, unsigned int outputHeight, unsigned int outputWidth,
            WDATA_T* scales, WDATA_T* biases, WDATA_T* means, WDATA_T* variances, WDATA_T epsilon)

    {
        mPlugin.push_back(std::unique_ptr<BatchNormPlugin>(new BatchNormPlugin(batchSize,
                                                   nbOutputs,
                                                outputHeight,
                                                outputWidth,
                                                scales,
                                                biases,
                                                means,
                                                variances,
                                                epsilon)));
        ++mPluginCount;
    }

    void add(const void* serialData, size_t serialLength)
    {
        mPlugin.push_back(std::unique_ptr<BatchNormPlugin>(new BatchNormPlugin(serialData,serialLength)));
        ++mPluginCount;
    }

    nvinfer1::IPlugin* get()
    {
        return mPlugin.back().get();
    }
    void destroy()
    {
      for(int i = mPluginCount - 1; i >= 0; --i)
      {
        mPlugin[i].release();
        mPlugin[i] = nullptr;
      }
      mPlugin = std::vector<std::unique_ptr<BatchNormPlugin>>();
      mPluginCount = 0;
    }
};

struct pluginAnchor_CPU{
    std::vector<std::unique_ptr<AnchorCPUPlugin>> mPlugin;
    int mPluginCount = 0;

    void add(unsigned int batchSize,
            unsigned int nbOutputs,
            unsigned int outputHeight,
            unsigned int outputWidth,
            unsigned int stimuliHeight,
            unsigned int stimuliWidth,
            unsigned int featureMapWidth,
            unsigned int featureMapHeight,
            unsigned int scoreCls,
            bool isFlip,
            unsigned int nbAnchors,
            const WDATA_T* anchors)

    {
        mPlugin.push_back(std::unique_ptr
                    <AnchorCPUPlugin>(new AnchorCPUPlugin(batchSize,
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
                                                           anchors)));
        ++mPluginCount;
    }

    void add(const void* serialData, size_t serialLength)
    {
        mPlugin.push_back(std::unique_ptr
                <AnchorCPUPlugin>(new AnchorCPUPlugin(serialData,
                                                      serialLength)));
        ++mPluginCount;
    }

    nvinfer1::IPlugin* get()
    {
        return mPlugin.back().get();
    }
    void destroy()
    {
      for(int i = mPluginCount - 1; i >= 0; --i)
      {
        mPlugin[i].release();
        mPlugin[i] = nullptr;
      }
      mPlugin = std::vector<std::unique_ptr<AnchorCPUPlugin>>();
      mPluginCount = 0;
    }
};

struct pluginAnchor_GPU{
    std::vector<std::unique_ptr<AnchorGPUPlugin>> mPlugin;
    int mPluginCount = 0;

    void add(unsigned int batchSize,
            unsigned int nbOutputs,
            unsigned int outputHeight,
            unsigned int outputWidth,
            unsigned int stimuliHeight,
            unsigned int stimuliWidth,
            unsigned int featureMapWidth,
            unsigned int featureMapHeight,
            unsigned int scoreCls,
            bool isFlip,
            unsigned int nbAnchors,
            const WDATA_T* anchors)

    {
        mPlugin.push_back(std::unique_ptr
                    <AnchorGPUPlugin>(new AnchorGPUPlugin(batchSize,
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
                                                           anchors)));
        ++mPluginCount;
    }

    void add(const void* serialData, size_t serialLength)
    {
        mPlugin.push_back(std::unique_ptr
                <AnchorGPUPlugin>(new AnchorGPUPlugin(serialData,
                                                      serialLength)));
        ++mPluginCount;
    }

    nvinfer1::IPlugin* get()
    {
        return mPlugin.back().get();
    }
    void destroy()
    {
      for(int i = mPluginCount - 1; i >= 0; --i)
      {
        mPlugin[i].release();
        mPlugin[i] = nullptr;
      }
      mPlugin = std::vector<std::unique_ptr<AnchorGPUPlugin>>();
      mPluginCount = 0;
    }
};


struct pluginRegionProposal_CPU{
    std::vector<std::unique_ptr<RegionProposalCPUPlugin>> mPlugin;
    int mPluginCount = 0;

    void add(unsigned int batchSize,
            unsigned int nbOutputs,
            unsigned int outputHeight,
            unsigned int outputWidth,
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
        mPlugin.push_back(std::unique_ptr
                    <RegionProposalCPUPlugin>(new RegionProposalCPUPlugin(batchSize,
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
                                                           iouIndex)));
        ++mPluginCount;
    }

    void add(const void* serialData, size_t serialLength)
    {
        mPlugin.push_back(std::unique_ptr
                <RegionProposalCPUPlugin>(new RegionProposalCPUPlugin(serialData,
                                                      serialLength)));
        ++mPluginCount;
    }

    nvinfer1::IPlugin* get()
    {
        return mPlugin.back().get();
    }
    void destroy()
    {
      for(int i = mPluginCount - 1; i >= 0; --i)
      {
        mPlugin[i].release();
        mPlugin[i] = nullptr;
      }
      mPlugin = std::vector<std::unique_ptr<RegionProposalCPUPlugin>>();
      mPluginCount = 0;
    }
};

struct pluginRegionProposal_GPU{
    std::vector<std::unique_ptr<RegionProposalGPUPlugin>> mPlugin;
    int mPluginCount = 0;

    void add(unsigned int batchSize,
            unsigned int nbOutputs,
            unsigned int outputHeight,
            unsigned int outputWidth,
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
        mPlugin.push_back(std::unique_ptr
                    <RegionProposalGPUPlugin>(new RegionProposalGPUPlugin(batchSize,
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
                                                           iouIndex)));
        ++mPluginCount;
    }

    void add(const void* serialData, size_t serialLength)
    {
        mPlugin.push_back(std::unique_ptr
                <RegionProposalGPUPlugin>(new RegionProposalGPUPlugin(serialData,
                                                      serialLength)));
        ++mPluginCount;
    }

    nvinfer1::IPlugin* get()
    {
        return mPlugin.back().get();
    }
    void destroy()
    {
      for(int i = mPluginCount - 1; i >= 0; --i)
      {
        mPlugin[i].release();
        mPlugin[i] = nullptr;
      }
      mPlugin = std::vector<std::unique_ptr<RegionProposalGPUPlugin>>();
      mPluginCount = 0;
    }
};

struct pluginProposal_GPU{
    std::vector<std::unique_ptr<ProposalGPUPlugin>> mPlugin;
    int mPluginCount = 0;

    void add(unsigned int batchSize,
            unsigned int nbOutputs,
            unsigned int outputHeight,
            unsigned int outputWidth,
            unsigned int nbProposals,
            unsigned int nbCls,
            double nmsIoU,
            unsigned int scoreIndex,
            double scoreThreshold,
            unsigned int maxParts,
            unsigned int maxTemplates,
            const unsigned int* numPartsPerClass,
            const unsigned int* numTemplatesPerClass,
            const WDATA_T* means,
            const WDATA_T* std,
            bool applyNMS,
            bool keepMax,
            double normX,
            double normY)

    {
        mPlugin.push_back(std::unique_ptr
                    <ProposalGPUPlugin>(new ProposalGPUPlugin(batchSize,
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
                                                             normY)));
        ++mPluginCount;
    }

    void add(const void* serialData, size_t serialLength)
    {
        mPlugin.push_back(std::unique_ptr
                <ProposalGPUPlugin>(new ProposalGPUPlugin(serialData,
                                                      serialLength)));
        ++mPluginCount;
    }

    nvinfer1::IPlugin* get()
    {
        return mPlugin.back().get();
    }
    void destroy()
    {
      for(int i = mPluginCount - 1; i >= 0; --i)
      {
        mPlugin[i].release();
        mPlugin[i] = nullptr;
      }
      mPlugin = std::vector<std::unique_ptr<ProposalGPUPlugin>>();
      mPluginCount = 0;
    }
};
struct pluginObjDet_GPU{
    std::vector<std::unique_ptr<ObjDetGPUPlugin>> mPlugin;
    int mPluginCount = 0;

    void add(unsigned int batchSize,
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
            double nmsIoU,
            const float* scoreThreshold,
            unsigned int maxParts,
            unsigned int maxTemplates,
            const unsigned int* numPartsPerClass,
            const unsigned int* numTemplatesPerClass,
            const WDATA_T* anchor)

    {
        mPlugin.push_back(std::unique_ptr
                    <ObjDetGPUPlugin>(new ObjDetGPUPlugin(batchSize,
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
                                                             anchor )));
        ++mPluginCount;
    }

    void add(const void* serialData, size_t serialLength)
    {
        mPlugin.push_back(std::unique_ptr
                <ObjDetGPUPlugin>(new ObjDetGPUPlugin(serialData,
                                                      serialLength)));
        ++mPluginCount;
    }

    nvinfer1::IPlugin* get()
    {
        return mPlugin.back().get();
    }
    void destroy()
    {
      for(int i = mPluginCount - 1; i >= 0; --i)
      {
        mPlugin[i].release();
        mPlugin[i] = nullptr;
      }
      mPlugin = std::vector<std::unique_ptr<ObjDetGPUPlugin>>();
      mPluginCount = 0;
    }
};

struct pluginObjDet_CPU{
    std::vector<std::unique_ptr<ObjDetCPUPlugin>> mPlugin;
    int mPluginCount = 0;

    void add(unsigned int batchSize,
            unsigned int nbOutputs,
            unsigned int outputHeight,
            unsigned int outputWidth,
            unsigned int channelHeight,
            unsigned int channelWidth,
            unsigned int nbProposals,
            unsigned int nbCls,
            unsigned int nbAnchors,
            double nmsIoU,
            const float* scoreThreshold)

    {
        mPlugin.push_back(std::unique_ptr
                    <ObjDetCPUPlugin>(new ObjDetCPUPlugin(batchSize,
                                                             nbOutputs,
                                                             outputHeight,
                                                             outputWidth,
                                                             channelHeight,
                                                             channelWidth,
                                                             nbProposals,
                                                             nbCls,
                                                             nbAnchors,
                                                             nmsIoU,
                                                             scoreThreshold )));
        ++mPluginCount;
    }

    void add(const void* serialData, size_t serialLength)
    {
        mPlugin.push_back(std::unique_ptr
                <ObjDetCPUPlugin>(new ObjDetCPUPlugin(serialData,
                                                      serialLength)));
        ++mPluginCount;
    }

    nvinfer1::IPlugin* get()
    {
        return mPlugin.back().get();
    }
    void destroy()
    {
      for(int i = mPluginCount - 1; i >= 0; --i)
      {
        mPlugin[i].release();
        mPlugin[i] = nullptr;
      }
      mPlugin = std::vector<std::unique_ptr<ObjDetCPUPlugin>>();
      mPluginCount = 0;
    }
};

struct pluginROIPooling_CPU{
    std::vector<std::unique_ptr<ROIPoolingCPUPlugin>> mPlugin;
    int mPluginCount = 0;

    void add(unsigned int batchSize,
            unsigned int nbOutputs,
            unsigned int outputHeight,
            unsigned int outputWidth,
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
        mPlugin.push_back(std::unique_ptr
                    <ROIPoolingCPUPlugin>(new ROIPoolingCPUPlugin(batchSize,
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
                                                           nbProposals,
                                                           ignorePadding,
                                                           isFlip)));
        ++mPluginCount;
    }

    void add(const void* serialData, size_t serialLength)
    {
        mPlugin.push_back(std::unique_ptr
                <ROIPoolingCPUPlugin>(new ROIPoolingCPUPlugin(serialData,
                                                      serialLength)));
        ++mPluginCount;
    }

    nvinfer1::IPlugin* get()
    {
        return mPlugin.back().get();
    }
    void destroy()
    {
      for(int i = mPluginCount - 1; i >= 0; --i)
      {
        mPlugin[i].release();
        mPlugin[i] = nullptr;
      }
      mPlugin = std::vector<std::unique_ptr<ROIPoolingCPUPlugin>>();
      mPluginCount = 0;
    }
};

struct pluginROIPooling_GPU{
    std::vector<std::unique_ptr<ROIPoolingGPUPlugin>> mPlugin;
    int mPluginCount = 0;

    void add(unsigned int batchSize,
            unsigned int nbOutputs,
            unsigned int outputHeight,
            unsigned int outputWidth,
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
        mPlugin.push_back(std::unique_ptr
                    <ROIPoolingGPUPlugin>(new ROIPoolingGPUPlugin(batchSize,
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
                                                           nbProposals,
                                                           ignorePadding,
                                                           isFlip)));
        ++mPluginCount;
    }

    void add(const void* serialData, size_t serialLength)
    {
        mPlugin.push_back(std::unique_ptr
                <ROIPoolingGPUPlugin>(new ROIPoolingGPUPlugin(serialData,
                                                      serialLength)));
        ++mPluginCount;
    }

    nvinfer1::IPlugin* get()
    {
        return mPlugin.back().get();
    }
    void destroy()
    {
      for(int i = mPluginCount - 1; i >= 0; --i)
      {
        mPlugin[i].release();
        mPlugin[i] = nullptr;
      }
      mPlugin = std::vector<std::unique_ptr<ROIPoolingGPUPlugin>>();
      mPluginCount = 0;
    }
};

struct pluginResize_GPU{
    std::vector<std::unique_ptr<ResizeGPUPlugin>> mPlugin;
    int mPluginCount = 0;

    void add(unsigned int batchSize,
            unsigned int nbOutputs,
            unsigned int outputHeight,
            unsigned int outputWidth,
            unsigned int featureHeight,
            unsigned int featureWidth,
            Pooling_T resizeType,
            bool aligCorner)

    {
        mPlugin.push_back(std::unique_ptr
                    <ResizeGPUPlugin>(new ResizeGPUPlugin(batchSize,
                                                           nbOutputs,
                                                           outputHeight,
                                                           outputWidth,
                                                           featureHeight,
                                                           featureWidth,
                                                           resizeType,
                                                           aligCorner)));
        ++mPluginCount;
    }

    void add(const void* serialData, size_t serialLength)
    {
        mPlugin.push_back(std::unique_ptr
                <ResizeGPUPlugin>(new ResizeGPUPlugin(serialData,
                                                      serialLength)));
        ++mPluginCount;
    }

    nvinfer1::IPlugin* get()
    {
        return mPlugin.back().get();
    }
    void destroy()
    {
      for(int i = mPluginCount - 1; i >= 0; --i)
      {
        mPlugin[i].release();
        mPlugin[i] = nullptr;
      }
      mPlugin = std::vector<std::unique_ptr<ResizeGPUPlugin>>();
      mPluginCount = 0;
    }
};



class PluginFactory : public nvinfer1::IPluginFactory
{


public:

    nvinfer1::IPlugin* createPlugin(const char* layerName,
                                    unsigned int batchSize,
                                    unsigned int nbOutputs,
                                    unsigned int outputHeight,
                                    unsigned int outputWidth,
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
        if(!strncmp(layerName, "ROIPooling_CPU", 14))
        {
            mROIPoolingCPUPlugin.add(batchSize,
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
                                 nbProposals,
                                 ignorePadding,
                                 isFlip);

            return mROIPoolingCPUPlugin.get();
        }
        else if(!strncmp(layerName, "ROIPooling_GPU", 14))
        {
            mROIPoolingGPUPlugin.add(batchSize,
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
                                 nbProposals,
                                 ignorePadding,
                                 isFlip);

            return mROIPoolingGPUPlugin.get();
        }
        else
            throw std::runtime_error(
                "PluginFactory::createPlugin this kind of ROIPooling layer is not implemented");

    }

    nvinfer1::IPlugin* createPlugin(const char* layerName,
                                    unsigned int batchSize,
                                    unsigned int nbOutputs,
                                    unsigned int outputHeight,
                                    unsigned int outputWidth,
                                    unsigned int featureHeight,
                                    unsigned int featureWidth,
                                    Pooling_T resizeType,
                                    bool alignCorner)
    {

        if(!strncmp(layerName, "Resize_GPU", 10))
        {
            mResizeGPUPlugin.add(batchSize,
                                 nbOutputs,
                                 outputHeight,
                                 outputWidth,
                                 featureHeight,
                                 featureWidth,
                                 resizeType,
                                 alignCorner);

            return mResizeGPUPlugin.get();
        }
        else
            throw std::runtime_error(
                "PluginFactory::createPlugin this kind of resize layer is not implemented");
    }

    nvinfer1::IPlugin* createPlugin(const char* layerName,
                                    unsigned int batchSize,
                                    unsigned int nbOutputs,
                                    unsigned int outputHeight,
                                    unsigned int outputWidth,
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
        if(!strncmp(layerName, "RegionProposal_CPU", 18))
        {
            mRegionProposalCPUPlugin.add(batchSize,
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
            return mRegionProposalCPUPlugin.get();
        }
        else if(!strncmp(layerName, "RegionProposal_GPU", 18))
        {
            mRegionProposalGPUPlugin.add(batchSize,
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
            return mRegionProposalGPUPlugin.get();
        }
        else
            throw std::runtime_error(
                "PluginFactory::createPlugin this kind of RegionProposal layer is not implemented");

    }



    nvinfer1::IPlugin* createPlugin(const char* layerName,
                                    unsigned int batchSize,
                                    unsigned int nbOutputs,
                                    unsigned int outputHeight,
                                    unsigned int outputWidth,
                                    unsigned int nbProposals,
                                    unsigned int mNbCls,
                                    double nmsIoU,
                                    unsigned int scoreIndex,
                                    double scoreThreshold,
                                    unsigned int maxParts,
                                    unsigned int maxTemplates,
                                    const unsigned int* numPartsPerClass,
                                    const unsigned int* numTemplatesPerClass,
                                    const WDATA_T* means,
                                    const WDATA_T* std,
                                    bool applyNMS,
                                    bool keepMax,
                                    double normX,
                                    double normY)

    {
        if(!strncmp(layerName, "Proposals_GPU", 13))
        {
            mProposalGPUPlugin.add(batchSize,
                                 nbOutputs,
                                 outputHeight,
                                 outputWidth,
                                 nbProposals,
                                 mNbCls,
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
            return mProposalGPUPlugin.get();
        }
        else
            throw std::runtime_error(
                "PluginFactory::createPlugin this kind of Proposal layer is not implemented");

    }

    nvinfer1::IPlugin* createPlugin(const char* layerName,
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
                                    double nmsIoU,
                                    const float* scoreThreshold,
                                    unsigned int maxParts,
                                    unsigned int maxTemplates,
                                    const unsigned int* numPartsPerClass,
                                    const unsigned int* numTemplatesPerClass,
                                    const WDATA_T* anchor)

    {
        if(!strncmp(layerName, "ObjectDet_CPU", 13))
        {
            mObjectDetCPUPlugin.add(batchSize,
                                 nbOutputs,
                                 outputHeight,
                                 outputWidth,
                                 channelHeight,
                                 channelWidth,
                                 nbProposals,
                                 nbCls,
                                 nbAnchors,
                                 nmsIoU,
                                 scoreThreshold);

            return mObjectDetCPUPlugin.get();
        }
        else if(!strncmp(layerName, "ObjectDet_GPU", 13))
        {
            mObjectDetGPUPlugin.add(batchSize,
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

            return mObjectDetGPUPlugin.get();
        }

        else
            throw std::runtime_error(
                "PluginFactory::createPlugin this kind of ObjectDetect layer is not implemented");

    }

    nvinfer1::IPlugin* createPlugin(const char* layerName,
                                    unsigned int batchSize,
                                    unsigned int nbOutputs,
                                    unsigned int outputHeight,
                                    unsigned int outputWidth,
                                    unsigned int stimuliHeight,
                                    unsigned int stimuliWidth,
                                    unsigned int featureMapWidth,
                                    unsigned int featureMapHeight,
                                    unsigned int scoreCls,
                                    bool isFlip,
                                    unsigned int nbAnchors,
                                    const WDATA_T* anchors)
    {
        if(!strncmp(layerName, "Anchor_CPU", 10))
        {
            mAnchorCPUPlugin.add(batchSize,
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
                                 anchors);
            return mAnchorCPUPlugin.get();
        }
        else if(!strncmp(layerName, "Anchor_GPU", 10))
        {
            mAnchorGPUPlugin.add(batchSize,
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
                                 anchors);
            return mAnchorGPUPlugin.get();
        }

        else
            throw std::runtime_error(
                "PluginFactory::createPlugin this kind of Anchor layer is not implemented");

    }

    nvinfer1::IPlugin* createPlugin(const char* layerName,
                                    unsigned int batchSize,
                                    unsigned int nbOutputs,
                                    unsigned int outputHeight,
                                    unsigned int outputWidth,
                                    WDATA_T* scales,
                                    WDATA_T* biases,
                                    WDATA_T* means,
                                    WDATA_T* variances,
                                    WDATA_T epsilon)
    {
        if(!strncmp(layerName, "BatchNorm_CUDA", 14))
        {
            mBatchNormCUDAPlugin.add(batchSize,
                                           nbOutputs,
                                        outputHeight,
                                        outputWidth,
                                        scales,
                                        biases,
                                        means,
                                        variances,
                                        epsilon);
            return mBatchNormCUDAPlugin.get();
        }
        else if (!strncmp(layerName, "BatchNorm_CUDNN", 15))
        {
            mBatchNormCUDNNPlugin.add(batchSize,
                                      nbOutputs,
                                      outputHeight,
                                      outputWidth,
                                      scales,
                                      biases,
                                      means,
                                      variances,
                                      epsilon);

            return mBatchNormCUDNNPlugin.get();
        }
        else
            throw std::runtime_error(
                "PluginFactory::createPlugin this kind of Batchnorm layer is not implemented");


    }

	nvinfer1::IPlugin* createPlugin(const char* layerName, const void* serialData, size_t serialLength) override
	{
        if(!strncmp(layerName, "BatchNorm_CUDA", 14))
        {
	    	mBatchNormCUDAPlugin.add(serialData, serialLength);
            return mBatchNormCUDAPlugin.get();
        }
        else if(!strncmp(layerName, "BatchNorm_CUDNN", 15))
        {
            mBatchNormCUDNNPlugin.add(serialData, serialLength);
            return mBatchNormCUDNNPlugin.get();
        }
        else if(!strncmp(layerName, "Anchor_CPU", 10))
        {
	    	mAnchorCPUPlugin.add(serialData, serialLength);
            return mAnchorCPUPlugin.get();
        }
        else if(!strncmp(layerName, "Anchor_GPU", 10))
        {
	    	mAnchorGPUPlugin.add(serialData, serialLength);
            return mAnchorGPUPlugin.get();
        }
        else if(!strncmp(layerName, "RegionProposal_CPU", 18))
        {
	    	mRegionProposalCPUPlugin.add(serialData, serialLength);
            return mRegionProposalCPUPlugin.get();
        }
        else if(!strncmp(layerName, "RegionProposal_GPU", 18))
        {
	    	mRegionProposalGPUPlugin.add(serialData, serialLength);
            return mRegionProposalGPUPlugin.get();
        }
        else if(!strncmp(layerName, "ROIPooling_CPU", 14))
        {
	    	mROIPoolingCPUPlugin.add(serialData, serialLength);
            return mROIPoolingCPUPlugin.get();
        }
        else if(!strncmp(layerName, "ROIPooling_GPU", 14))
        {
	    	mROIPoolingGPUPlugin.add(serialData, serialLength);
            return mROIPoolingGPUPlugin.get();
        }
        else if(!strncmp(layerName, "Proposals_GPU", 13))
        {
	    	mProposalGPUPlugin.add(serialData, serialLength);
            return mProposalGPUPlugin.get();
        }
        else if(!strncmp(layerName, "Resize_GPU", 10))
        {
	    	mResizeGPUPlugin.add(serialData, serialLength);
            return mResizeGPUPlugin.get();
        }
        else if(!strncmp(layerName, "ObjectDet_CPU", 13))
        {
	    	mObjectDetCPUPlugin.add(serialData, serialLength);
            return mObjectDetCPUPlugin.get();
        }
        else if(!strncmp(layerName, "ObjectDet_GPU", 13))
        {
	    	mObjectDetGPUPlugin.add(serialData, serialLength);
            return mObjectDetGPUPlugin.get();
        }
        else
            throw std::runtime_error(
                "PluginFactory::createPlugin(const char*, const void*, size_t): this kind of layer is not implemented");

    }
    void destroyPlugin()
    {
        //BatchNormPlugin models destroy
        mBatchNormCUDAPlugin.destroy();
        mBatchNormCUDNNPlugin.destroy();
        //AnchorPlugin models destroy
        mAnchorCPUPlugin.destroy();
        mAnchorGPUPlugin.destroy();
        //Proposal models destroy
        mProposalGPUPlugin.destroy();
        //Region Proposal models destroy
        mRegionProposalCPUPlugin.destroy();
        mRegionProposalGPUPlugin.destroy();
        //ROI Pooling models destroy
        mROIPoolingCPUPlugin.destroy();
        mROIPoolingGPUPlugin.destroy();

        mResizeGPUPlugin.destroy();

        mObjectDetCPUPlugin.destroy();
        mObjectDetGPUPlugin.destroy();

    }

    pluginAnchor_CPU mAnchorCPUPlugin;
    pluginAnchor_GPU mAnchorGPUPlugin;

    pluginBatchnorm_CUDA mBatchNormCUDAPlugin;
    pluginBatchnorm_CUDNN mBatchNormCUDNNPlugin;
    pluginRegionProposal_CPU mRegionProposalCPUPlugin;
    pluginRegionProposal_GPU mRegionProposalGPUPlugin;

    pluginProposal_GPU mProposalGPUPlugin;

    pluginROIPooling_CPU mROIPoolingCPUPlugin;
    pluginROIPooling_GPU mROIPoolingGPUPlugin;

    pluginResize_GPU mResizeGPUPlugin;

    pluginObjDet_CPU mObjectDetCPUPlugin;
    pluginObjDet_GPU mObjectDetGPUPlugin;

};

void add_target(nvinfer1::INetworkDefinition* net,
                std::vector<nvinfer1::ITensor *> outputs_tensor,
                unsigned int targetIdx);

std::vector<nvinfer1::ITensor *>
        add_activation(nvinfer1::INetworkDefinition* net,
        			    nvinfer1::IBuilder* netBuilder,
                        bool useDLA,
                        nvinfer1::DataType dT,
                        std::string layerName,
                        nvinfer1::ActivationType activation,
                        std::vector<nvinfer1::ITensor *> inputs_tensor);

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
                        unsigned int bias_size);

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
                        unsigned int bias_size);

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
                std::string bFile);

std::vector<nvinfer1::ITensor *>
        add_HWC2CHW(nvinfer1::INetworkDefinition* net,
                    nvinfer1::IBuilder* netBuilder,
                    nvinfer1::DataType dT,
                    std::string layerName,
                    std::vector<nvinfer1::ITensor *> inputs_tensor);

std::vector<nvinfer1::ITensor *>
        add_concat(nvinfer1::INetworkDefinition* net,
                   nvinfer1::IBuilder* netBuilder,
                   bool useDLA,
                   nvinfer1::DataType dT,
                   std::string layerName,
                   unsigned int nbInputs,
                   /*std::vector<nvinfer1::ITensor *> const* inputs_tensor*/
                   std::vector<std::vector<nvinfer1::ITensor*>*> inputs_tensor);

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
                        WDATA_T epsilon);

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
                        DATA_T* power);

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
                        WDATA_T* power);

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
                        nvinfer1::PoolingType poolType);

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
                    const int pad_right);

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
                        const float k);

std::vector<nvinfer1::ITensor *>
            add_reshape(nvinfer1::INetworkDefinition* net,
                        nvinfer1::IBuilder* netBuilder,
                        nvinfer1::DataType dT,
                        std::string layerName,
                        unsigned int groupSize,
                        bool restoreShape,
                        std::vector<nvinfer1::ITensor *> inputs_tensor);
std::vector<nvinfer1::ITensor *>
        add_softmax(nvinfer1::INetworkDefinition* net,
                        nvinfer1::IBuilder* netBuilder,
                        nvinfer1::DataType dT,
                        std::string layerName,
                        std::vector<nvinfer1::ITensor *> inputs_tensor);


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
                        const WDATA_T* anchor);

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
                            unsigned int iouIndex);

std::vector<nvinfer1::ITensor *>
        add_objectdetect(  nvinfer1::INetworkDefinition* net,
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
                            const WDATA_T* anchor);


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
                        const WDATA_T* std);

std::vector<nvinfer1::ITensor *>
        add_ROIpooling(nvinfer1::INetworkDefinition* net,
                            PluginFactory& factory,
                            nvinfer1::DataType dT,
                            std::string layerName,
                            unsigned int batchSize,
                            unsigned int nbOutputs,
                            unsigned int outputHeight,
                            unsigned int outputWidth,
                            /*std::vector<nvinfer1::ITensor *> inputs_tensor,*/
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
                    bool alignCorner);

void createContext(unsigned int batchSize,
                   unsigned int iterBuild,
                   PluginFactory& factory,
                   std::string inputEngine = "",
                   std::string outputEngine = "",
                   bool useINT8 = false);

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

#endif // N2D2_TENSORRT_HPP
