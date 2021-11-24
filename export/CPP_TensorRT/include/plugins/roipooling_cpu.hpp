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

#ifndef ROIPOOLING_CPU_HPP
#define ROIPOOLING_CPU_HPP

#include "dnn_utils.hpp"
#include "kernels_cpu.hpp"
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
            mFeatureDims.push_back( trt_Dims3(  featureChannels[i],
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
            mFeatureDims[k] = read<trt_Dims3>(d);

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
        float* inputDataCPU(NULL);
        float* outputDataCPU(NULL);
        size_t size_output_cpy = mOutputDims.d[0]*mOutputDims.d[1]
                                    *mOutputDims.d[2]*mOutputDims.d[3];

        size_t size_regionproposal = 4*mNbProposals;
        size_t size_input_cpy = size_regionproposal;
        size_t mem_offset = size_regionproposal*batchSize;

        for(unsigned int k = 0; k < mNbInputFeature; k++)
            size_input_cpy += mFeatureDims[k].d[0]*mFeatureDims[k].d[1]
                                *mFeatureDims[k].d[2];


        inputDataCPU = new float[size_input_cpy*batchSize];
        outputDataCPU = new float[size_output_cpy*batchSize];

        if (!inputDataCPU)
            throw std::runtime_error(
                "enqueue(): could not allocate memory");

        if (!outputDataCPU)
            throw std::runtime_error(
                "enqueue(): could not allocate memory");
        CHECK_CUDA_STATUS(cudaMemcpy(inputDataCPU,
                                     reinterpret_cast<const float*>(inputs[0]),
                                     size_regionproposal*sizeof(float)*batchSize,
                                     cudaMemcpyDeviceToHost));

        for(unsigned int k = 0; k < mNbInputFeature; ++k)
        {
            size_t feature_size = mFeatureDims[k].d[0]*mFeatureDims[k].d[1]
                                        *mFeatureDims[k].d[2];

            CHECK_CUDA_STATUS(cudaMemcpy(inputDataCPU + mem_offset,
                                         reinterpret_cast<const float*>(inputs[k + 1]),
                                         feature_size*batchSize*sizeof(float),
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
                    *mOutputDims.d[3]*batchSize*sizeof(float),
                 cudaMemcpyHostToDevice));

             outputOffset += mOutputDims.d[1]*mOutputDims.d[2]*mOutputDims.d[3];
        }
*/

         CHECK_CUDA_STATUS(cudaMemcpy(outputs[0], outputDataCPU,
                                      batchSize*mOutputDims.d[0]*mOutputDims.d[1]
                                      *mOutputDims.d[2]*mOutputDims.d[3]*sizeof(float),
                                       cudaMemcpyHostToDevice));
        delete[] inputDataCPU;
        delete[] outputDataCPU;

        return 0;
	}

	virtual size_t getSerializationSize() override
	{
        size_t inputDimParamSize = 4*sizeof(int); //
        size_t ROIParamI = 4*sizeof(int) + sizeof(Pooling_T)
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
    std::vector<trt_Dims3> mFeatureDims;
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

#endif