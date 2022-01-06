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

#ifndef OBJECTDETECTION_CPU_HPP
#define OBJECTDETECTION_CPU_HPP

#include "dnn_utils.hpp"
#include "kernels_cpu.hpp"

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
        return trt_Dims4(  mNbProposals*mNbClass,
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
#endif