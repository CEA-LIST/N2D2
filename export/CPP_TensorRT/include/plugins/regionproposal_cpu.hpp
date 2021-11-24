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

#ifndef REGIONPROPOSAL_CPU_HPP
#define REGIONPROPOSAL_CPU_HPP

#include "dnn_utils.hpp"
#include "kernels_cpu.hpp"

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
        return trt_Dims4(mNbProposals, mOutputDims.d[1], mOutputDims.d[2], mOutputDims.d[3]);

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
        size_t size_input_cpy = 6*mNbAnchors
                                 *mChannelHeight*mChannelWidth*batchSize;
        size_t size_output_cpy = mNbProposals*mOutputDims.d[1]
                                    *mOutputDims.d[2]*mOutputDims.d[3];

        inputDataCPU = new float[size_input_cpy];
        outputDataCPU = new float[size_output_cpy*batchSize];

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
             size_output_cpy*sizeof(float),
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
#endif