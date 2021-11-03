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

#ifndef BATCHNORM_CUDNN_HPP
#define BATCHNORM_CUDNN_HPP

#include "dnn_utils.hpp"
#include "kernels_gpu.hpp"

/**Plugin Layer implementation**/
/**BatchNormalisation CUDNN implementation**/
class BatchNormCUDNNPlugin: public nvinfer1::IPlugin
{
public:
	BatchNormCUDNNPlugin(unsigned int batchSize,
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
    	mEpsilon = read<float>(d);
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

        return trt_Dims4(batchInput, mOutputDims.d[1], mOutputDims.d[2], mOutputDims.d[3]);
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
        float ONE_T = float(1); // Alpha must be set to 1 for all steps
        float ZERO_T = float(0); // Beta must be set to 0 for POOLING FORWARD

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
        size_t biasParamSize = sizeof(float)*mOutputDims.d[1];
        size_t variancesParamSize = sizeof(float)*mOutputDims.d[1];
        size_t meansParamSize = sizeof(float)*mOutputDims.d[1];
        size_t scalesParamSize = sizeof(float)*mOutputDims.d[1];
        size_t epsilonParamSize = sizeof(float);

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
        write<float>(d, mEpsilon);
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

    nvinfer1::Dims mOutputDims;
    float mEpsilon;
    float* mScalesCuda;
    float* mBiasesCuda;
    float* mMeansCuda;
    float* mVariancesCuda;
    cudnnTensorDescriptor_t mInputDescriptor;
    cudnnTensorDescriptor_t mOutputDescriptor;
    cudnnTensorDescriptor_t mScaleDescriptor;
	cudnnHandle_t mCudnnContext;
    size_t mSerializationSize;

};

struct pluginBatchnorm_CUDNN {
    std::vector<std::unique_ptr<BatchNormCUDNNPlugin>> mPlugin;
    int mPluginCount = 0;

    void add(unsigned int batchSize, unsigned int nbOutputs, unsigned int outputHeight, unsigned int outputWidth,
            float* scales, float* biases, float* means, float* variances, float epsilon)

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

#endif