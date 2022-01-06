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

#ifndef ROIPOOLING_GPU_HPP
#define ROIPOOLING_GPU_HPP

#include "dnn_utils.hpp"
#include "kernels_gpu.hpp"


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
            mFeatureDims.push_back( trt_Dims3(  featureChannels[i],
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
            mFeatureDims[k] = read<trt_Dims3>(d);

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
        return trt_Dims4(mOutputDims.d[0], mOutputDims.d[1], mOutputDims.d[2], mOutputDims.d[3]);

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
        float alpha = 1;
        float beta = 0;
        dim3 threadGrid = {mThreadX[k], mThreadY[k], mThreadZ[k]};
        dim3 blockGrid = {mBlockX[k], mBlockY[k], mBlockZ[k]};


        if(k>0)
            beta = 1;

        if(mPoolType == Bilinear || mPoolType == BilinearTF)
            cuda_roipooling_bilinear_propagate( alpha,
                                                reinterpret_cast<const float*>(inputs[0]),
                                                0,
                                                //proposalIdx,
                                                mNbProposals,
                                                mStimuliHeight,
                                                mStimuliWidth,
                                                reinterpret_cast<const float*>(inputs[k + 1]),
                                                mFeatureDims[k].d[0],
                                                mFeatureDims[k].d[1],
                                                mFeatureDims[k].d[2],
                                                batchSize,
                                                0,
                                                beta,
                                                //reinterpret_cast<float*>(outputs[proposalIdx]),
                                                reinterpret_cast<float*>(outputs[0]),
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
                            + mNbInputFeature*sizeof(trt_Dims3); //

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
            write<trt_Dims3>(d, mFeatureDims[k]);

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
    std::vector<trt_Dims3> mFeatureDims;
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
#endif