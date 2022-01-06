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

#ifndef RESIZE_GPU_HPP
#define RESIZE_GPU_HPP

#include "dnn_utils.hpp"
#include "kernels_gpu.hpp"

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
        return trt_Dims3(   mOutputDims.d[1],
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
                                                reinterpret_cast<const float*>(inputs[0]),
                                                reinterpret_cast<float*>(outputs[0]),
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
#endif