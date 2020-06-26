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

#ifndef ANCHOR_CPU_HPP
#define ANCHOR_CPU_HPP

#include "dnn_utils.hpp"
#include "kernels_cpu.hpp"

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
                    const float*  anchors)
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
            mAnchors[i].x0 = read<float>(d);
            mAnchors[i].y0 = read<float>(d);
            mAnchors[i].x1 = read<float>(d);
            mAnchors[i].y1 = read<float>(d);
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
        float* inputDataCPU(NULL);
        float* outputDataCPU(NULL);
        size_t size_cpy = mOutputDims.d[3]*mOutputDims.d[2]*mOutputDims.d[1]*mOutputDims.d[0];

        inputDataCPU = new float[size_cpy];
        outputDataCPU = new float[size_cpy];

        if (!inputDataCPU)
            throw std::runtime_error(
                "enqueue(): could not allocate memory");

        if (!outputDataCPU)
            throw std::runtime_error(
                "enqueue(): could not allocate memory");

        CHECK_CUDA_STATUS(cudaMemcpy(inputDataCPU,
                                     reinterpret_cast<const float*>(inputs[0]),
                                     size_cpy*sizeof(float),
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
             size_cpy*sizeof(float),
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
        size_t anchorsSize = sizeof(int)*2 + 4*mNbAnchors*sizeof(float) + sizeof(bool); // mNbAnchors and (x0 y0 x1 y1) * mNbAnchors + mScoreCls

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
            write<float>(d, mAnchors[i].x0);
            write<float>(d, mAnchors[i].y0);
            write<float>(d, mAnchors[i].x1);
            write<float>(d, mAnchors[i].y1);
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
            const float* anchors)

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
#endif