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

#ifndef BATCHNORM_GPU_HPP
#define BATCHNORM_GPU_HPP

#include "dnn_utils.hpp"
#include "kernels_gpu.hpp"
#if NV_TENSORRT_MAJOR < 8

/**Plugin Layer implementation**/
/**BatchNormalisation CUDA implementation**/
class BatchNormPlugin: public nvinfer1::IPlugin
{
public:
	BatchNormPlugin(unsigned int batchSize,
                    unsigned int nbOutputs,
                    unsigned int outputHeight,
                    unsigned int outputWidth,
                    float* scales,
                    float* biases,
                    float* means,
                    float* variances,
                    float epsilon)
	{
        mOutputDims.d[0] = batchSize;
        mOutputDims.d[1] = nbOutputs;
        mOutputDims.d[2] = outputHeight;
        mOutputDims.d[3] = outputWidth;


        /**Initialize epsilon parameter**/
        mEpsilon = epsilon;

        /**Initialize scale parameters on GPU **/
        checkCudaErrors( cudaMalloc((void**)&mScalesCuda,
                         nbOutputs*sizeof(float)) );
        checkCudaErrors( cudaMemcpy(mScalesCuda,
                         scales,
                         nbOutputs*sizeof(float),
                         cudaMemcpyHostToDevice) );

        /**Initialize bias parameters on GPU **/
        checkCudaErrors( cudaMalloc((void**)&mBiasesCuda,
                         nbOutputs*sizeof(float)) );
        checkCudaErrors( cudaMemcpy(mBiasesCuda,
                         biases,
                         nbOutputs*sizeof(float),
                         cudaMemcpyHostToDevice) );

        /**Initialize means parameters on GPU **/
        checkCudaErrors( cudaMalloc((void**)&mMeansCuda,
                         nbOutputs*sizeof(float)) );
        checkCudaErrors( cudaMemcpy(mMeansCuda,
                         means,
                         nbOutputs*sizeof(float),
                         cudaMemcpyHostToDevice) );

        /**Initialize variance parameters on GPU **/
        checkCudaErrors( cudaMalloc((void**)&mVariancesCuda,
                         nbOutputs*sizeof(float)) );
        checkCudaErrors( cudaMemcpy(mVariancesCuda,
                         variances,
                         nbOutputs*sizeof(float),
                         cudaMemcpyHostToDevice) );

	}

	BatchNormPlugin(const void* dataFromRuntime, size_t length)
	{
		const char* d = reinterpret_cast<const char*>(dataFromRuntime), *a = d;
        mOutputDims.d[0] = read<int>(d);
		mOutputDims.d[1] = read<int>(d);
		mOutputDims.d[2] = read<int>(d);
		mOutputDims.d[3] = read<int>(d);
    	mEpsilon = read<float>(d);
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

        return trt_Dims4(batchInput, mOutputDims.d[1], mOutputDims.d[2], mOutputDims.d[3]);
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
                                     reinterpret_cast<const float*>(inputs[0]),
                                     mOutputDims.d[1],
                                     0, /*outputoffset*/
                                     reinterpret_cast<float*>(outputs[0]),
                                     reinterpret_cast<const float*>(mBiasesCuda),
                                     reinterpret_cast<const float*>(mVariancesCuda),
                                     reinterpret_cast<const float*>(mMeansCuda),
                                     reinterpret_cast<const float*>(mScalesCuda),
                                     mEpsilon,
                                     mNbThreads,
                                     mNbBlocks,
                                     stream);
        return 0;
	}

	virtual size_t getSerializationSize() override
	{
        size_t inputDimParamSize = 4*sizeof(int); //nbOutputs, nbOutputsHeight, nbOutputWidth = 3
        size_t biasParamSize = sizeof(float)*mOutputDims.d[1];
        size_t variancesParamSize = sizeof(float)*mOutputDims.d[1];
        size_t meansParamSize = sizeof(float)*mOutputDims.d[1];
        size_t scalesParamSize = sizeof(float)*mOutputDims.d[1];
        size_t epsilonParamSize = sizeof(float);
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
        write<float>(d, mEpsilon);
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
    float mEpsilon;
    float* mScalesCuda;
    float* mBiasesCuda;
    float* mMeansCuda;
    float* mVariancesCuda;
    size_t mSerializationSize;

};

struct pluginBatchnorm_CUDA{
    std::vector<std::unique_ptr<BatchNormPlugin>> mPlugin;
    int mPluginCount = 0;

    void add(unsigned int batchSize, unsigned int nbOutputs, unsigned int outputHeight, unsigned int outputWidth,
            float* scales, float* biases, float* means, float* variances, float epsilon)

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
#endif
#endif