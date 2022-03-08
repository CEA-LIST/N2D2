
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

#ifndef PROPOSAL_GPU_HPP
#define PROPOSAL_GPU_HPP

#include "dnn_utils.hpp"
#include "kernels_gpu.hpp"
#if NV_TENSORRT_MAJOR < 8

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
                      const float* means,
                      const float* std,
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
                         4*sizeof(float)) );

        checkCudaErrors( cudaMalloc((void**)&mStdGPU,
                         4*sizeof(float)) );

        checkCudaErrors( cudaMemcpy(mMeanGPU,
                         means,
                         4*sizeof(float),
                         cudaMemcpyHostToDevice) );

        checkCudaErrors( cudaMemcpy(mStdGPU,
                         std,
                         4*sizeof(float),
                         cudaMemcpyHostToDevice) );

        checkCudaErrors(cudaMalloc((void**)&mNormalizeROIs,
                                     mNbProposals*4*(mNbCls - mScoreIndex)*batchSize*sizeof(float)) );

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
                                 reinterpret_cast<const float *>(mMeanGPU),
                                 reinterpret_cast<const float *>(mStdGPU),
                                 reinterpret_cast<const unsigned int *>(mPartsPerClass),
                                 reinterpret_cast<const unsigned int *>(mTemplatesPerClass),
                                 reinterpret_cast<const float *>(inputs[0]),
                                 reinterpret_cast<const float*>(inputs[1]),
                                 reinterpret_cast<const float*>(inputs[2]),
                                 (mMaxParts > 0 || mMaxTemplates > 0)  ?
                                    reinterpret_cast<const float*>(inputs[3])
                                    : reinterpret_cast<const float*>(inputs[2]),
                                 (mMaxParts > 0) ?
                                    reinterpret_cast<const float*>(inputs[4])
                                    : reinterpret_cast<const float*>(inputs[2]),
                                 (mMaxParts > 0 && mMaxTemplates > 0) ?
                                    reinterpret_cast<const float*>(inputs[5])
                                    : reinterpret_cast<const float*>(inputs[2]),
                                 reinterpret_cast<float *>(mNormalizeROIs),
                                 reinterpret_cast<int *>(mMaxCls),
                                 reinterpret_cast<float *>(mPartsPrediction),
                                 reinterpret_cast<float *>(mPartsVisibilityPrediction),
                                 reinterpret_cast<float *>(mTemplatesPrediction),
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
                                 reinterpret_cast<const float *>(mNormalizeROIs),
                                 reinterpret_cast<const int *>(mKeepIndex),
                                 reinterpret_cast<const float *>(mPartsPrediction),
                                 reinterpret_cast<const float *>(mPartsVisibilityPrediction),
                                 reinterpret_cast<const float *>(mTemplatesPrediction),
                                 reinterpret_cast<float *>(outputs[0]),
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
                            mOutputDims.d[0]*mInputMaxSize*sizeof(float)) );
        checkCudaErrors(cudaMalloc((void**)&mIndexI,
                            mOutputDims.d[0]*mInputMaxSize*sizeof(float)) );
        checkCudaErrors(cudaMalloc((void**)&mIndexJ,
                            mOutputDims.d[0]*mInputMaxSize*sizeof(float)) );
        checkCudaErrors(cudaMalloc((void**)&mIndexK,
                            mOutputDims.d[0]*mInputMaxSize*sizeof(float)) );
        checkCudaErrors(cudaMalloc((void**)&mIndexB,
                            mOutputDims.d[0]*mInputMaxSize*sizeof(float)) );
        checkCudaErrors(cudaMalloc((void**)&mMap,
                            mOutputDims.d[0]*mInputMaxSize*sizeof(unsigned int)) );

        checkCudaErrors(cudaMalloc((void**)&mSortedIndexI,
                            mOutputDims.d[0]*mSortSize*sizeof(float)) );
        checkCudaErrors(cudaMalloc((void**)&mSortedIndexJ,
                            mOutputDims.d[0]*mSortSize*sizeof(float)) );
        checkCudaErrors(cudaMalloc((void**)&mSortedIndexK,
                            mOutputDims.d[0]*mSortSize*sizeof(float)) );
        checkCudaErrors(cudaMalloc((void**)&mSortedIndexB,
                            mOutputDims.d[0]*mSortSize*sizeof(float)) );

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

        const unsigned int nbBlocks = std::ceil(mInputMaxSize/32.0);

        /**Reorder i,j,k,b index and create the map vector to allow a fast gpu sorting using thrust**/
        cuda_region_proposal_split_indexes( mChannelWidth,
                                            mChannelHeight,
                                            mNbAnchors,
                                            batchSize,
                                            nbBlocks,
                                            reinterpret_cast<const float *>(inputs[0]),
                                            reinterpret_cast<float *>(mValues),
                                            reinterpret_cast<float *>(mIndexI),
                                            reinterpret_cast<float *>(mIndexJ),
                                            reinterpret_cast<float *>(mIndexK),
                                            reinterpret_cast<float *>(mIndexB),
                                            reinterpret_cast<unsigned int *>(mMap),
                                            mMinWidth,
                                            mMinHeight,
                                            mScoreIndex);

        for(unsigned int n = 0; n < batchSize; ++n)
        {
            unsigned int inputOffset = mInputMaxSize*n;
            unsigned int outputOffset = mSortSize*n;

            thrust_sort_keys(   reinterpret_cast<float *>(mValues),
                                reinterpret_cast<unsigned int *>(mMap),
                                mInputMaxSize,
                                inputOffset);

            thrust_gather(  reinterpret_cast<unsigned int *>(mMap),
                            reinterpret_cast<float *>(mIndexI),
                            reinterpret_cast<float *>(mSortedIndexI),
                            mSortSize,
                            inputOffset,
                            outputOffset);

            thrust_gather(  reinterpret_cast<unsigned int *>(mMap),
                            reinterpret_cast<float *>(mIndexJ),
                            reinterpret_cast<float *>(mSortedIndexJ),
                            mSortSize,
                            inputOffset,
                            outputOffset);

            thrust_gather(  reinterpret_cast<unsigned int *>(mMap),
                            reinterpret_cast<float *>(mIndexK),
                            reinterpret_cast<float *>(mSortedIndexK),
                            mSortSize,
                            inputOffset,
                            outputOffset);

            thrust_gather(  reinterpret_cast<unsigned int *>(mMap),
                            reinterpret_cast<float *>(mIndexB),
                            reinterpret_cast<float *>(mSortedIndexB),
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
                                        reinterpret_cast<const float *>(inputs[0]),
                                        reinterpret_cast<float *>(mSortedIndexI),
                                        reinterpret_cast<float *>(mSortedIndexJ),
                                        reinterpret_cast<float *>(mSortedIndexK),
                                        reinterpret_cast<float *>(mSortedIndexB),
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
                                        reinterpret_cast<const float *>(inputs[0]),
                                        reinterpret_cast<const float *>(mSortedIndexI),
                                        reinterpret_cast<const float *>(mSortedIndexJ),
                                        reinterpret_cast<const float *>(mSortedIndexK),
                                        reinterpret_cast<const float *>(mSortedIndexB),
                                        reinterpret_cast<const int *>(mSortedIndex),
                                        reinterpret_cast<float *>(outputs[0]),
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
            const float* means,
            const float* std,
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
#endif
#endif
