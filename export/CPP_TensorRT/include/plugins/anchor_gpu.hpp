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
                    bool isCoordinatesAnchors,
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

        mRatioX = std::ceil(mFeatureMapWidth / (double) outputWidth);
        mRatioY = std::ceil(featureMapHeight / (double) outputHeight);
        mScoreCls = scoreCls;
        mIsCoordinateAnchors = isCoordinatesAnchors;
        mIsFlip = isFlip;
        mNbAnchors = nbAnchors;
        mAnchorsPrecompute = new float [outputWidth*outputHeight*mNbAnchors*4];
        for(size_t k = 0; k < mNbAnchors; ++k) {
          for(size_t ya = 0; ya < outputHeight; ++ya) {
            for(size_t xa = 0; xa < outputWidth; ++xa) {

                const size_t anchorsIdx = k*4;
                const size_t anchorsPrecomputeIdx = xa*4 + ya*outputWidth*4 + k*outputWidth*outputHeight*4;
                const float xa0 = (anchors[anchorsIdx] + xa * mRatioX) 
                                    / (float)(mFeatureMapWidth - 1.0);
                const float ya0 = (anchors[anchorsIdx + 1] + ya * mRatioY) 
                                    / (float)(mFeatureMapHeight - 1.0);
                const float xa1 = (anchors[anchorsIdx + 2] + xa * mRatioX) 
                                    / (float)(mFeatureMapWidth - 1.0);
                const float ya1 = (anchors[anchorsIdx + 3] + ya * mRatioY) 
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
        mScoreCls = (unsigned int) read<int>(d);
        mIsCoordinateAnchors = read<bool>(d);
        mIsFlip = read<bool>(d);
        mNbAnchors = (unsigned int) read<int>(d);
		mAnchorsGPU = deserializeToDevice(d, mNbAnchors*4*mOutputDims.d[2]*mOutputDims.d[3]);
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

	virtual size_t getSerializationSize() override
	{
        size_t inputDimParamSize = 4*sizeof(int); //nbOutputs, nbOutputsHeight, nbOutputWidth = 3
        size_t stimuliParamSize = 4*sizeof(int); //Stimuliheight and StimuliWidth
        size_t anchorsSize = 4*mNbAnchors*mOutputDims.d[2]*mOutputDims.d[3]*sizeof(float) + sizeof(bool); // mNbAnchors and (x0 y0 x1 y1) * mNbAnchors + mScoreCls
        size_t paramSize = sizeof(bool);
        size_t threadSize = 3*2*sizeof(int);
        mSerializationSize = inputDimParamSize + stimuliParamSize
                                + anchorsSize + threadSize + paramSize;

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
        write<int>(d, (int)mScoreCls);
        write<bool>(d, mIsCoordinateAnchors);
        write<bool>(d, mIsFlip);
        write<int>(d, (int)mNbAnchors);
        serializeFromDevice(d, mAnchorsGPU, mNbAnchors*4*mOutputDims.d[2]*mOutputDims.d[3]);
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

    float* deserializeToDevice(const char*& hostBuffer, size_t dataSize)
    {
        float* gpuData;
        checkCudaErrors(cudaMalloc(&gpuData, dataSize*sizeof(float)));
        checkCudaErrors(cudaMemcpy(gpuData, hostBuffer, dataSize*sizeof(float), cudaMemcpyHostToDevice));
        hostBuffer += dataSize*sizeof(float);
        return gpuData;
    }

    void serializeFromDevice(char*& hostBuffer, float* deviceWeights, size_t dataSize)
    {
        checkCudaErrors(cudaMemcpy(hostBuffer, deviceWeights, dataSize*sizeof(float), cudaMemcpyDeviceToHost));
        hostBuffer += dataSize*sizeof(float);
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
    float* mAnchorsPrecompute;
    size_t mSerializationSize;
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
            bool isCoordinatesAnchors,
            bool isFlip,
            unsigned int nbAnchors,
            const float* anchors)

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
                                                           isCoordinatesAnchors,
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

#endif