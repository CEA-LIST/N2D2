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

#ifndef OBJECTDETECTION_GPU_HPP
#define OBJECTDETECTION_GPU_HPP

#include "dnn_utils.hpp"
#include "kernels_gpu.hpp"
/**ObjectDet GPU implementation**/

class ObjDetGPUPlugin: public nvinfer1::IPluginV2Ext
{
public:
	ObjDetGPUPlugin(unsigned int batchSize,
                    unsigned int nbOutputs,
                    unsigned int outputHeight,
                    unsigned int outputWidth,
                    unsigned int channelHeight,
                    unsigned int channelWidth,
                    unsigned int stimuliWidth,
                    unsigned int stimuliHeight,
                    unsigned int featureMapWidth,
                    unsigned int featureMapHeight,
                    unsigned int nbProposals,
                    unsigned int nbCls,
                    unsigned int nbAnchors,
                    bool isCoordinatesAnchors,
                    bool isPixelFormatXY,
                    double nmsIoU,
                    const float* scoreThreshold,
                    unsigned int maxParts,
                    unsigned int maxTemplates,
                    const unsigned int* numPartsPerClass,
                    const unsigned int* numTemplatesPerClass,
                    const float* anchor)
	{
        mOutputDims.d[0] = batchSize;
        mOutputDims.d[1] = nbOutputs;
        mOutputDims.d[2] = outputHeight;
        mOutputDims.d[3] = outputWidth;
        mChannelHeight = channelHeight;
        mChannelWidth = channelWidth;
        mStimuliWidth = stimuliWidth;
        mStimuliHeight = stimuliHeight;
        mFeatureMapWidth = featureMapWidth;
        mFeatureMapHeight = featureMapHeight;


        mNbAnchors = nbAnchors;
        mNbClass= nbCls;
        mNbProposals = nbProposals;
        mIsCoordinatesAnchors = isCoordinatesAnchors;
        mIsPixelFormatXY = isPixelFormatXY;
        mNMS_IoU = nmsIoU;

        mMaxParts = maxParts;
        mMaxTemplates = maxTemplates;

        //mPartsPerClass = new unsigned int[mNbClass];
        for(unsigned int i = 0; i < mNbClass; ++i) {
            mPartsPerClass_HOST.push_back(numPartsPerClass[i]);
        }

        //mTemplatesPerClass = new unsigned int[mNbClass];
        for(unsigned int i = 0; i < mNbClass; ++i) {
            mTemplatesPerClass_HOST.push_back(numTemplatesPerClass[i]);
        }
        /*
        mAnchors.resize(mNbAnchors);
        for(unsigned int i = 0; i < mNbAnchors*4; i += 4)
        {
            mAnchors[i/4].x0 = anchor[i + 0];
            mAnchors[i/4].y0 = anchor[i + 1];
            mAnchors[i/4].x1 = anchor[i + 2];
            mAnchors[i/4].y1 = anchor[i + 3];
        }
        */
        //mAnchors_HOST.resize(4 * mNbAnchors * nbCls);
        for(unsigned int i = 0; i < 4 * mNbAnchors * nbCls; ++i) {
            mAnchors_HOST.push_back(anchor[i]);
        }

        checkCudaErrors( cudaMalloc((void**)&mAnchors,
                         4 * mNbAnchors * nbCls * sizeof(float)) );
        checkCudaErrors( cudaMemcpy(mAnchors,
                         mAnchors_HOST.data(),
                         4 * mNbAnchors * nbCls *sizeof(float),
                         cudaMemcpyHostToDevice) );
        /**Initialize pixels map on GPU **/
        checkCudaErrors( cudaMalloc((void**)&mPixelMap,
                         channelHeight * channelWidth
                         * mNbAnchors * mNbClass
                         * batchSize * sizeof(int)) );
        checkCudaErrors( cudaMemset(mPixelMap,
                                    -1,
                                    channelHeight * channelWidth
                                    * mNbAnchors * mNbClass
                                    * batchSize * sizeof(int)) );

        checkCudaErrors( cudaMalloc((void**)&mPixelMapSorted,
                         channelHeight * channelWidth
                         * mNbAnchors * mNbClass
                         * batchSize * sizeof(int)) );
        checkCudaErrors( cudaMemset(mPixelMapSorted,
                                    -1,
                                    channelHeight * channelWidth
                                    * mNbAnchors * mNbClass
                                    * batchSize * sizeof(int)) );

        checkCudaErrors( cudaMalloc((void**)&mScoresIndex,
                         channelHeight * channelWidth
                         * mNbAnchors * mNbClass
                         * batchSize * sizeof(int)) );
        checkCudaErrors( cudaMalloc((void**)&mScores,
                         channelHeight * channelWidth
                         * mNbAnchors * mNbClass
                         * batchSize * sizeof(float)) );
        checkCudaErrors( cudaMalloc((void**)&mScoresFiltered,
                         channelHeight * channelWidth
                         * mNbAnchors * mNbClass
                         * batchSize * sizeof(float)) );


        checkCudaErrors( cudaMalloc((void**)&mMxGPUIndex,
                         channelHeight * channelWidth
                         * mNbAnchors * mNbClass
                         * batchSize * sizeof(float)) );
        checkCudaErrors( cudaMemset(mMxGPUIndex,
                                    -1.0,
                                    channelHeight * channelWidth
                                    * mNbAnchors * mNbClass
                                    * batchSize * sizeof(float)) );
        checkCudaErrors( cudaMalloc((void**)&mMyGPUIndex,
                         channelHeight * channelWidth
                         * mNbAnchors * mNbClass
                         * batchSize * sizeof(float)) );
        checkCudaErrors( cudaMemset(mMyGPUIndex,
                                    -1.0,
                                    channelHeight * channelWidth
                                    * mNbAnchors * mNbClass
                                    * batchSize * sizeof(float)) );
        checkCudaErrors( cudaMalloc((void**)&mMwGPUIndex,
                         channelHeight * channelWidth
                         * mNbAnchors * mNbClass
                         * batchSize * sizeof(float)) );
        checkCudaErrors( cudaMemset(mMwGPUIndex,
                                    -1.0,
                                    channelHeight * channelWidth
                                    * mNbAnchors * mNbClass
                                    * batchSize * sizeof(float)) );
        checkCudaErrors( cudaMalloc((void**)&mMhGPUIndex,
                         channelHeight * channelWidth
                         * mNbAnchors * mNbClass
                         * batchSize * sizeof(float)) );
        checkCudaErrors( cudaMemset(mMhGPUIndex,
                                    -1.0,
                                    channelHeight * channelWidth
                                    * mNbAnchors * mNbClass
                                    * batchSize * sizeof(float)) );

        mMxCPUIndex = new float[channelHeight * channelWidth * mNbAnchors * mNbClass * batchSize];
        mMyCPUIndex = new float[channelHeight * channelWidth * mNbAnchors * mNbClass * batchSize];
        mMwCPUIndex = new float[channelHeight * channelWidth * mNbAnchors * mNbClass * batchSize];
        mMhCPUIndex = new float[channelHeight * channelWidth * mNbAnchors * mNbClass * batchSize];

        //mScoreThreshold_HOST.resize(nbCls);
        for(unsigned int i = 0; i < nbCls; ++i) {
            mScoreThreshold_HOST.push_back(scoreThreshold[i]);
        }

        /**Initialize mScoreThreshold parameters on GPU **/
        checkCudaErrors( cudaMalloc((void**)&mScoreThreshold,
                         mNbClass*sizeof(float)) );
        checkCudaErrors( cudaMemcpy(mScoreThreshold,
                         scoreThreshold,
                         mNbClass*sizeof(float),
                         cudaMemcpyHostToDevice) );

        checkCudaErrors( cudaMalloc((void**)&mROIsBBOxFinal,
                            mNbProposals*5*batchSize*sizeof(float)) );

        checkCudaErrors( cudaMalloc((void**)&mROIsMapAnchorsFinal,
                            mNbProposals*5*batchSize*sizeof(float)) );

        checkCudaErrors( cudaMalloc((void**)&mROIsIndexFinal,
                            batchSize*sizeof(unsigned int)) );


        gpuThreadAllocation();
	}

	ObjDetGPUPlugin(const void* dataFromRuntime, size_t length)
	{
		const char* d = reinterpret_cast<const char*>(dataFromRuntime), *a = d;
        mOutputDims = read<nvinfer1::Dims>(d);
        //mOutputDims.d[0] = read<int>(d);
		//mOutputDims.d[1] = read<int>(d);
		//mOutputDims.d[2] = read<int>(d);
		//mOutputDims.d[3] = read<int>(d);
        mChannelHeight = (unsigned int) read<int>(d);
        mChannelWidth = (unsigned int) read<int>(d);
        mStimuliWidth = read<unsigned int>(d);
        mStimuliHeight = read<unsigned int>(d);
        mFeatureMapWidth = read<unsigned int>(d);
        mFeatureMapHeight = read<unsigned int>(d);

        mNbAnchors = (unsigned int) read<int>(d);
        mNbClass = (unsigned int) read<int>(d);
        mNbProposals = (unsigned int) read<int>(d);
        mMaxParts = (unsigned int) read<int>(d);
        mMaxTemplates = (unsigned int) read<int>(d);
        mIsCoordinatesAnchors = read<bool>(d);;
        mIsPixelFormatXY = read<bool>(d);;
        mNMS_IoU = read<double>(d);
        mThreadX = read<int>(d);
        mThreadY = read<int>(d);
        mThreadZ = read<int>(d);
        mBlockX = read<int>(d);
        mBlockY = read<int>(d);
        mBlockZ = read<int>(d);
		mPixelMap = deserializeToDevice<int>(d, mChannelHeight * mChannelWidth * mNbAnchors * mNbClass * mOutputDims.d[0]);
		mPixelMapSorted = deserializeToDevice<int>(d, mChannelHeight * mChannelWidth * mNbAnchors * mNbClass * mOutputDims.d[0]);
        mScoresIndex = deserializeToDevice<int>(d, mChannelHeight * mChannelWidth * mNbAnchors * mNbClass * mOutputDims.d[0]);
        mScores = deserializeToDevice<float>(d, mChannelHeight * mChannelWidth * mNbAnchors * mNbClass * mOutputDims.d[0]);
        mScoresFiltered = deserializeToDevice<float>(d, mChannelHeight * mChannelWidth * mNbAnchors * mNbClass * mOutputDims.d[0]);

        mMxGPUIndex = deserializeToDevice<float>(d, mChannelHeight * mChannelWidth * mNbAnchors * mNbClass * mOutputDims.d[0]);
		mMyGPUIndex = deserializeToDevice<float>(d, mChannelHeight * mChannelWidth * mNbAnchors * mNbClass * mOutputDims.d[0]);
		mMwGPUIndex = deserializeToDevice<float>(d, mChannelHeight * mChannelWidth * mNbAnchors * mNbClass * mOutputDims.d[0]);
		mMhGPUIndex = deserializeToDevice<float>(d, mChannelHeight * mChannelWidth * mNbAnchors * mNbClass * mOutputDims.d[0]);

        mMxCPUIndex = new float[mChannelHeight * mChannelWidth * mNbAnchors * mNbClass * mOutputDims.d[0]];
        for(unsigned int k = 0; k < mChannelHeight * mChannelWidth * mNbAnchors * mNbClass * mOutputDims.d[0]; ++k)
            mMxCPUIndex[k] = read<float>(d);
        mMyCPUIndex = new float[mChannelHeight * mChannelWidth * mNbAnchors * mNbClass * mOutputDims.d[0]];
        for(unsigned int k = 0; k < mChannelHeight * mChannelWidth * mNbAnchors * mNbClass * mOutputDims.d[0]; ++k)
            mMyCPUIndex[k] = read<float>(d);
        mMwCPUIndex = new float[mChannelHeight * mChannelWidth * mNbAnchors * mNbClass * mOutputDims.d[0]];
        for(unsigned int k = 0; k < mChannelHeight * mChannelWidth * mNbAnchors * mNbClass * mOutputDims.d[0]; ++k)
            mMwCPUIndex[k] = read<float>(d);
        mMhCPUIndex = new float[mChannelHeight * mChannelWidth * mNbAnchors * mNbClass * mOutputDims.d[0]];
        for(unsigned int k = 0; k < mChannelHeight * mChannelWidth * mNbAnchors * mNbClass * mOutputDims.d[0]; ++k)
            mMhCPUIndex[k] = read<float>(d);
        
        mScoreThreshold = deserializeToDevice<float>(d, mNbClass);
        mScoreThreshold_HOST.reserve(mNbClass);
        for(unsigned int k = 0; k <  mNbClass; ++k)
            mScoreThreshold_HOST[k] = read<float>(d);

        mPartsPerClass_HOST.reserve(mNbClass);
        for(unsigned int k = 0; k < mNbClass; ++k)
            mPartsPerClass_HOST[k] = read<unsigned int>(d);

        mTemplatesPerClass_HOST.reserve(mNbClass);
        for(unsigned int k = 0; k < mNbClass; ++k)
            mTemplatesPerClass_HOST[k] = read<unsigned int>(d);
        /*mAnchors.resize(mNbAnchors);
        for(unsigned int i = 0; i < mNbAnchors; ++i)
        {
            mAnchors[i].x0 = read<float>(d);
            mAnchors[i].y0 = read<float>(d);
            mAnchors[i].x1 = read<float>(d);
            mAnchors[i].y1 = read<float>(d);
        }*/
        mAnchors = deserializeToDevice<float>(d, 4 * mNbAnchors * mNbClass);
        mAnchors_HOST.reserve(4 * mNbAnchors * mNbClass);
        for(unsigned int k = 0; k <  mNbAnchors * mNbClass * 4; ++k)
            mAnchors_HOST[k] = read<float>(d);

        mROIsBBOxFinal = deserializeToDevice<float>(d, mNbProposals*5*mOutputDims.d[0]);
        mROIsMapAnchorsFinal = deserializeToDevice<float>(d, mNbProposals*5*mOutputDims.d[0]);
        mROIsIndexFinal = deserializeToDevice<unsigned int>(d, mOutputDims.d[0]);
		assert(d == a + length);

	}

	~ObjDetGPUPlugin()
	{
        /**Initialize pixels map on GPU **/

        //mAnchors = std::vector<Anchor>();
	}

	virtual int getNbOutputs() const 
#if NV_TENSORRT_MAJOR > 7
    noexcept
#endif
    override {
        //return (int)mNbProposals;
        return (int) 1;

	}

	virtual nvinfer1::Dims getOutputDimensions(int index,
                                               const nvinfer1::Dims* inputDim,
                                               int nbInputDims) 
#if NV_TENSORRT_MAJOR > 7
    noexcept
#endif
    override {
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

	virtual int initialize() 
#if NV_TENSORRT_MAJOR > 7
    noexcept
#endif
    override{
		return 0;
	}

	virtual void terminate() 
#if NV_TENSORRT_MAJOR > 7
    noexcept
#endif
    override	{
        checkCudaErrors(cudaFree(mMxGPUIndex));
        checkCudaErrors(cudaFree(mMyGPUIndex));
        checkCudaErrors(cudaFree(mMwGPUIndex));
        checkCudaErrors(cudaFree(mMhGPUIndex));

        checkCudaErrors(cudaFree(mScoresFiltered));
        checkCudaErrors(cudaFree(mScores));
        checkCudaErrors(cudaFree(mScoresIndex));

        checkCudaErrors(cudaFree(mPixelMap));
        checkCudaErrors(cudaFree(mPixelMapSorted));

        checkCudaErrors(cudaFree(mAnchors));

        checkCudaErrors(cudaFree(mScoreThreshold));
        checkCudaErrors(cudaFree(mROIsIndexFinal));
        checkCudaErrors(cudaFree(mROIsMapAnchorsFinal));
        checkCudaErrors(cudaFree(mROIsBBOxFinal));

	}

	virtual size_t getWorkspaceSize(int maxBatchSize) const 
#if NV_TENSORRT_MAJOR > 7
    noexcept
#endif
    override	{
		return 0;
	}

#if NV_TENSORRT_MAJOR > 7
	virtual int enqueue(int batchSize,
                        const void*const * inputs,
                        void*const* outputs,
                        void* workspace,
                        cudaStream_t stream) noexcept
#else
	virtual int enqueue(int batchSize,
                        const void*const * inputs,
                        void** outputs,
                        void* workspace,
                        cudaStream_t stream)
#endif
    override	{
        //float* outputDataCPU(NULL);

        size_t size_output_cpy = mNbProposals*mNbClass*mOutputDims.d[1]
                                    *mOutputDims.d[2]*mOutputDims.d[3]*batchSize;

        size_t size_map = mChannelHeight * mChannelWidth
                                    * mNbAnchors * mNbClass
                                    * batchSize;

        //float outputDataCPU[size_output_cpy] = {0.0};

        //if (!outputDataCPU)
        //    throw std::runtime_error(
        //        "enqueue(): could not allocate memory");

        //float* input_parts(NULL);
        //float* input_templates(NULL);

        //input_parts = new float[mMaxParts*2];
        //input_templates = new float[mMaxTemplates*3];

        const unsigned int inputBatchOffset = mChannelWidth*mChannelHeight*(mNbAnchors*mNbClass * 6);
        unsigned int nbTotalPart = 0;
        unsigned int nbTotalTemplate = 0;

        /***TO IMPROVE!!!!!**/
        const double xRatio = std::ceil(mFeatureMapWidth / mChannelWidth);
        const double yRatio = std::ceil(mFeatureMapHeight / mChannelHeight);
        const float xOutputRatio = mStimuliWidth / (float) mFeatureMapWidth;
        const float yOutputRatio = mStimuliHeight / (float) mFeatureMapHeight;

        for(unsigned int c = 0; c < mNbClass; ++c) 
        {
            nbTotalPart += mPartsPerClass_HOST[c];
            nbTotalTemplate += mTemplatesPerClass_HOST[c];
        }

        checkCudaErrors( cudaMemset(mPixelMapSorted,
                                    -1,
                                    mChannelHeight * mChannelWidth
                                    * mNbAnchors * mNbClass
                                    * batchSize * sizeof(int)) );
        checkCudaErrors( cudaMemset(mPixelMap,
                                    -1,
                                    mChannelHeight * mChannelWidth
                                    * mNbAnchors * mNbClass
                                    * batchSize * sizeof(int)) );


        dim3 blocksPerGrid = {  (unsigned int) mBlockX,
                                (unsigned int) mBlockY,
                                (unsigned int) batchSize,
                            };

        dim3 threadsPerBlock = {(unsigned int) mThreadX,
                                (unsigned int) mThreadY,
                                (unsigned int) mThreadZ
                                };

        cudaSReduceIndex(  mChannelWidth*mChannelHeight*mNbAnchors,
                        inputBatchOffset,
                        mChannelWidth*mChannelHeight*mNbAnchors*mNbClass,
                        mChannelWidth,
                        mChannelHeight,
                        mNbAnchors,
                        reinterpret_cast<const float *>(mScoreThreshold),
                        reinterpret_cast<const float *>(inputs[0]),
                        reinterpret_cast<int *>(mPixelMap),
                        reinterpret_cast<float *>(mScores),
                        blocksPerGrid,
                        threadsPerBlock);

        std::vector<std::vector <unsigned int> > count(batchSize,
                                                    std::vector<unsigned int>(mNbClass));

        for(unsigned int batchPos = 0; batchPos < batchSize; ++batchPos)
        {
            for(unsigned int cls = 0; cls < mNbClass; ++cls)
            {
                const int pixelOffset = cls*mChannelWidth*mChannelHeight*mNbAnchors 
                                            +  mChannelWidth*mChannelHeight*mNbAnchors*mNbClass*batchPos;
                std::cout << "pixelOffset: " << pixelOffset << std::endl;
                const int nbMapDet = copy_if_int(  reinterpret_cast<int *>(mPixelMap) + pixelOffset,
                                                   reinterpret_cast<int *>(mPixelMapSorted) + pixelOffset,
                                                   mChannelWidth*mChannelHeight*mNbAnchors);
                std::cout << "nbMapDet: " << nbMapDet << std::endl;

                const int nbScoreDet = copy_if_float( reinterpret_cast<float *>(mScores) + pixelOffset,
                                                    reinterpret_cast<float *>(mScoresFiltered) + pixelOffset,
                                                    mChannelWidth*mChannelHeight*mNbAnchors);
                std::cout << "nbScoreDet: " << nbScoreDet << std::endl;

                if (nbScoreDet != nbMapDet)
                    throw std::runtime_error(
                        "Dont find the same number of valid boxes");

                count[batchPos][cls] = nbMapDet;

            }
        }

        std::vector< std::vector< std::vector<BBox_T >>> ROIs(  mNbClass, 
                                                                std::vector< std::vector <BBox_T>>(batchSize));

        std::vector< std::vector< std::vector<BBox_T >>> ROIsAnchors(   mNbClass, 
                                                                        std::vector< std::vector <BBox_T>>(batchSize));

        for(unsigned int cls = 0; cls < mNbClass; ++cls)
        {
            const int offset = cls*mNbAnchors*mChannelWidth*mChannelHeight;

            for(unsigned int batchPos = 0; batchPos < batchSize; ++batchPos)
            {
                const int batchOffset = batchPos*inputBatchOffset;

                unsigned int totalIdxPerClass = 0;

                if(count[batchPos][cls] > 0)
                {
                    const int offsetBase = mNbClass*mNbAnchors*mChannelWidth*mChannelHeight;

                    const int offsetCpy = cls*mNbAnchors*mChannelWidth*mChannelHeight
                                            + batchPos*mNbClass*mNbAnchors*mChannelWidth*mChannelHeight;

                    unsigned int nbElementNMS =  count[batchPos][cls];

                    thrust_sort_keys_int(   reinterpret_cast<float *>(mScoresFiltered) + offsetCpy,
                                            reinterpret_cast<int *>(mPixelMapSorted) + offsetCpy,
                                            nbElementNMS,
                                            0);


                    thrust_gather_int(reinterpret_cast<int*>(mPixelMapSorted) + offset + batchPos*mNbClass*mNbAnchors*mChannelWidth*mChannelHeight,
                                reinterpret_cast<const float *>(inputs[0]) + offsetBase + offset + batchOffset,
                                reinterpret_cast<float *>(mMxGPUIndex),
                                nbElementNMS,
                                0,
                                0);
                    thrust_gather_int(reinterpret_cast<int*>(mPixelMapSorted) + offset + batchPos*mNbClass*mNbAnchors*mChannelWidth*mChannelHeight,
                                reinterpret_cast<const float *>(inputs[0]) + 2*offsetBase + offset + batchOffset,
                                reinterpret_cast<float *>(mMyGPUIndex),
                                nbElementNMS,
                                0,
                                0);

                    thrust_gather_int(reinterpret_cast<int*>(mPixelMapSorted) + offset + batchPos*mNbClass*mNbAnchors*mChannelWidth*mChannelHeight,
                                reinterpret_cast<const float *>(inputs[0]) + 3*offsetBase + offset + batchOffset,
                                reinterpret_cast<float *>(mMwGPUIndex),
                                nbElementNMS,
                                0,
                                0);

                    thrust_gather_int(reinterpret_cast<int*>(mPixelMapSorted) + offset + batchPos*mNbClass*mNbAnchors*mChannelWidth*mChannelHeight,
                                reinterpret_cast<const float *>(inputs[0]) + 4*offsetBase + offset + batchOffset,
                                reinterpret_cast<float *>(mMhGPUIndex),
                                nbElementNMS,
                                0,
                                0);

                    int* pixelMap(NULL);
                    float* scoreMap(NULL);

                    pixelMap = new int[nbElementNMS];
                    scoreMap = new float[nbElementNMS];

                    CHECK_CUDA_STATUS(cudaMemcpy(pixelMap,
                            reinterpret_cast<int*>(mPixelMapSorted) + offsetCpy,
                            nbElementNMS*sizeof(int),
                            cudaMemcpyDeviceToHost));

                    CHECK_CUDA_STATUS(cudaMemcpy(scoreMap,
                            reinterpret_cast<float*>(mScoresFiltered) + offsetCpy,
                            nbElementNMS*sizeof(float),
                            cudaMemcpyDeviceToHost));

                    CHECK_CUDA_STATUS(cudaMemcpy(mMxCPUIndex,
                                                reinterpret_cast<float*>(mMxGPUIndex),
                                                nbElementNMS*sizeof(float),
                                                cudaMemcpyDeviceToHost));
                    CHECK_CUDA_STATUS(cudaMemcpy(mMyCPUIndex,
                                                reinterpret_cast<float*>(mMyGPUIndex),
                                                nbElementNMS*sizeof(float),
                                                cudaMemcpyDeviceToHost));
                    CHECK_CUDA_STATUS(cudaMemcpy(mMwCPUIndex,
                                                reinterpret_cast<float*>(mMwGPUIndex),
                                                nbElementNMS*sizeof(float),
                                                cudaMemcpyDeviceToHost));
                    CHECK_CUDA_STATUS(cudaMemcpy(mMhCPUIndex,
                                                reinterpret_cast<float*>(mMhGPUIndex),
                                                nbElementNMS*sizeof(float),
                                                cudaMemcpyDeviceToHost));


                    for(unsigned int idx = 0; idx < nbElementNMS; ++idx)
                    {
                        ROIs[cls][batchPos].push_back(BBox_T(  mMxCPUIndex[idx],
                                                                mMyCPUIndex[idx],
                                                                mMwCPUIndex[idx],
                                                                mMhCPUIndex[idx],
                                                                scoreMap[idx]));
                        ROIsAnchors[cls][batchPos].push_back(BBox_T(   pixelMap[idx]%mChannelWidth,
                                                                        (pixelMap[idx]/mChannelWidth)%mChannelHeight,
                                                                        (pixelMap[idx]/(mChannelWidth*mChannelHeight))%mNbAnchors,
                                                                        0.0,
                                                                        0.0));
                    }
                    delete[] pixelMap;
                    delete[] scoreMap;

                    // Non-Maximum Suppression (NMS)
                    /*for (unsigned int i = 0; i < ROIs.size() - 1 && i < 50; ++i)
                    {
                        const float x0 = ROIs[i].x;
                        const float y0 = ROIs[i].y;
                        const float w0 = ROIs[i].w;
                        const float h0 = ROIs[i].h;

                        for (unsigned int j = i + 1; j < ROIs.size(); ) {

                            const float x = ROIs[j].x;
                            const float y = ROIs[j].y;
                            const float w = ROIs[j].w;
                            const float h = ROIs[j].h;

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
                                    ROIs.erase(ROIs.begin() + j);
                                    keepIdx[batchPos][cls].erase(keepIdx[batchPos][cls].begin() + j);

                                    continue;
                                }
                            }
                            ++j;
                        }
                    }*/

                    std::vector<BBox_T> final_rois;
                    std::vector<BBox_T> final_anchors;

                    BBox_T next_candidate;
                    BBox_T next_anchors;
                    std::reverse(ROIs[cls][batchPos].begin(),ROIs[cls][batchPos].end());
                    std::reverse(ROIsAnchors[cls][batchPos].begin(),ROIsAnchors[cls][batchPos].end());

                    while (final_rois.size() < mNbProposals && !ROIs[cls][batchPos].empty()) {
                        next_candidate = ROIs[cls][batchPos].back();
                        ROIs[cls][batchPos].pop_back();
                        next_anchors = ROIsAnchors[cls][batchPos].back();
                        ROIsAnchors[cls][batchPos].pop_back();
                        // Overlapping boxes are likely to have similar scores,
                        // therefore we iterate through the previously selected boxes backwards
                        // in order to see if `next_candidate` should be suppressed.
                        bool should_select = true;
                        const float x0 = next_candidate.x;
                        const float y0 = next_candidate.y;
                        const float w0 = next_candidate.w;
                        const float h0 = next_candidate.h;

                        for (int j = static_cast<int>(final_rois.size()) - 1; j >= 0; --j) {

                            const float x = final_rois[j].x;
                            const float y = final_rois[j].y;
                            const float w = final_rois[j].w;
                            const float h = final_rois[j].h;
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
                                    should_select = false;
                                    break;

                                }
                            }

                        }

                        if (should_select) {
                            final_rois.push_back(next_candidate);
                            final_anchors.push_back(next_anchors);
                        }
                    }
                    ROIs[cls][batchPos].resize(final_rois.size());
                    ROIsAnchors[cls][batchPos].resize(final_anchors.size());

                    for(unsigned int f = 0; f < final_rois.size(); ++ f )
                    {
                        ROIs[cls][batchPos][f] = final_rois[f];
                        ROIsAnchors[cls][batchPos][f] = final_anchors[f];
                    }
                    
                }
                
            }
        }

        

        unsigned int* valid_rois(NULL);
        valid_rois = new unsigned int[batchSize];

        for(unsigned int cls = 0; cls < mNbClass; ++cls)
        {
            int mThreadX = 32;
            int mThreadY = 1;
            int mThreadZ = 1;

            int mBlockX = std::ceil(mNbProposals/(float) mThreadX);
            int mBlockY = std::max(mPartsPerClass_HOST[cls], mTemplatesPerClass_HOST[cls]) > 0 ? 
                            std::max(mPartsPerClass_HOST[cls], mTemplatesPerClass_HOST[cls]) : 1 ;
            int mBlockZ = batchSize;

            dim3 blocks = {  (unsigned int) mBlockX, (unsigned int) mBlockY, (unsigned int) batchSize};

            dim3 threads = {(unsigned int) mThreadX, (unsigned int) mThreadY, (unsigned int) mThreadZ };
            
            for(int i = 0; i < ROIs[cls].size(); ++i)
            {
                valid_rois[i] = ROIs[cls][i].size();
            }
            unsigned int keypointClsOffset = 0;
            unsigned int templateClsOffset = 0;

            for(unsigned int c = 0; c < cls; ++c) 
            {
                keypointClsOffset += mPartsPerClass_HOST[c] * 2;
                templateClsOffset += mTemplatesPerClass_HOST[c] * 3;
            }

            if(mIsCoordinatesAnchors) {
                keypointClsOffset *= mNbAnchors;
                templateClsOffset *= mNbAnchors;
            }

            for(unsigned int b = 0; b < batchSize; ++b)
            {
                const unsigned int offset = b*5*mNbProposals ;

                if(valid_rois[b] > 0)
                {
                    CHECK_CUDA_STATUS(cudaMemcpy(   mROIsBBOxFinal + offset,
                                                    ROIs[cls][b].data(),
                                                    valid_rois[b]*5*sizeof(float),
                                                    cudaMemcpyHostToDevice));
                                                    
                    CHECK_CUDA_STATUS(cudaMemcpy(   mROIsMapAnchorsFinal + offset,
                                                    ROIsAnchors[cls][b].data(),
                                                    valid_rois[b]*5*sizeof(float),
                                                    cudaMemcpyHostToDevice));

                }
            }

            CHECK_CUDA_STATUS(cudaMemcpy(   mROIsIndexFinal,
                                            valid_rois,
                                            batchSize*sizeof(unsigned int),
                                            cudaMemcpyHostToDevice));

            cudaS_SSD_output_gathering( batchSize,
                                        mNbClass,
                                        mNbAnchors,
                                        mChannelWidth,
                                        mChannelHeight,
                                        mNbProposals,
                                        mROIsIndexFinal,
                                        cls,
                                        nbTotalPart,
                                        nbTotalTemplate,
                                        mMaxParts,
                                        mMaxTemplates,
                                        keypointClsOffset,
                                        templateClsOffset,
                                        mPartsPerClass_HOST[cls],
                                        mTemplatesPerClass_HOST[cls],
                                        mIsCoordinatesAnchors,
                                        mIsPixelFormatXY,
                                        xRatio,
                                        yRatio,
                                        xOutputRatio,
                                        yOutputRatio,
                                        mROIsBBOxFinal,
                                        mROIsMapAnchorsFinal,
                                        mAnchors,
                                        reinterpret_cast<const float *>(inputs[2]),
                                        reinterpret_cast<const float *>(inputs[1]),
                                        reinterpret_cast<float*>(outputs[0]),
                                        blocks,
                                        threads);
            

        }

        //checkCudaErrors( cudaFree(gpu_anchors));
        //checkCudaErrors( cudaFree(gpu_rois_bbox));
        //checkCudaErrors( cudaFree(gpu_rois_anchors));
        //checkCudaErrors( cudaFree(gpu_valid_rois));
        //delete[] cpu_anchors;
        delete[] valid_rois;

        /* CHECK_CUDA_STATUS(cudaMemcpy(outputs[0],
         outputDataCPU,
         size_output_cpy*sizeof(float),
        cudaMemcpyHostToDevice));*/

	   // delete[] pixelMapCPU;
        //delete[] outputDataCPU;
        //delete[] input_parts;
        //delete[] input_templates;
        return 0;
    }

/*
	ObjDetGPUPlugin(unsigned int batchSize,
                    unsigned int nbOutputs,
                    unsigned int outputHeight,
                    unsigned int outputWidth,
                    unsigned int channelHeight,
                    unsigned int channelWidth,
                    unsigned int stimuliWidth,
                    unsigned int stimuliHeight,
                    unsigned int featureMapWidth,
                    unsigned int featureMapHeight,
                    unsigned int nbProposals,
                    unsigned int nbCls,
                    unsigned int nbAnchors,
                    bool isCoordinatesAnchors,
                    bool isPixelFormatXY,
                    double nmsIoU,
                    const float* scoreThreshold,
                    unsigned int maxParts,
                    unsigned int maxTemplates,
                    const unsigned int* numPartsPerClass,
                    const unsigned int* numTemplatesPerClass,
                    const float* anchor)

*/
	virtual size_t getSerializationSize() const 
#if NV_TENSORRT_MAJOR > 7
    noexcept
#endif
    override	{
        size_t proposalParamI = 15*sizeof(int) + (2 + 2*mNbClass)*sizeof(unsigned int); //
        size_t proposalParamD = (1)*sizeof(double) + mNbClass*sizeof(float); //RatioX and RatioY
        size_t PixelMapSize = mChannelHeight * mChannelWidth * mNbAnchors * mNbClass * mOutputDims.d[0]*sizeof(int);
        size_t M_Index = mChannelHeight * mChannelWidth * mNbAnchors * mNbClass * mOutputDims.d[0]*sizeof(float);
        size_t anchorsSize = 4*mNbAnchors*mNbClass*sizeof(float) ; // mNbAnchors and (x0 y0 x1 y1) * mNbAnchors + mScoreCls

        size_t finalIdxSize = (mNbProposals*5*mOutputDims.d[0])*2*sizeof(float) + mOutputDims.d[0]*sizeof(unsigned int);
        size_t formatSize = (2)*sizeof(bool);
        size_t outputDimSize = sizeof(nvinfer1::Dims);
        size_t scoreThreshold_HOSTSize = mNbClass*sizeof(float);
        size_t anchor_HOSTSize = 4*mNbAnchors*mNbClass*sizeof(float) ;
        size_t SerializationSize = proposalParamI + proposalParamD + 3*PixelMapSize 
                                    + 10*M_Index + anchorsSize + finalIdxSize + formatSize 
                                    + outputDimSize + scoreThreshold_HOSTSize + anchor_HOSTSize;

        return SerializationSize;
	}

	virtual void serialize(void* buffer) const 
#if NV_TENSORRT_MAJOR > 7
    noexcept
#endif
    override	{
        char* d = reinterpret_cast<char*>(buffer);
        char* a = d;
        *reinterpret_cast<nvinfer1::Dims*>(d) = mOutputDims;
        d += sizeof(nvinfer1::Dims);

        *reinterpret_cast<int*>(d) = mChannelHeight;
        d += sizeof(int);
        *reinterpret_cast<int*>(d) = mChannelWidth;
        d += sizeof(int);

        *reinterpret_cast<unsigned int*>(d) = mStimuliWidth;
        d += sizeof(unsigned int);
        *reinterpret_cast<unsigned int*>(d) = mStimuliHeight;
        d += sizeof(unsigned int);

        *reinterpret_cast<unsigned int*>(d) = mFeatureMapWidth;
        d += sizeof(unsigned int);
        *reinterpret_cast<unsigned int*>(d) = mFeatureMapHeight;
        d += sizeof(unsigned int);

        *reinterpret_cast<int*>(d) = mNbAnchors;
        d += sizeof(int);
        *reinterpret_cast<int*>(d) = mNbClass;
        d += sizeof(int);
        *reinterpret_cast<int*>(d) = mNbProposals;
        d += sizeof(int);

        *reinterpret_cast<unsigned int*>(d) = mMaxParts;
        d += sizeof(unsigned int);
        *reinterpret_cast<unsigned int*>(d) = mMaxTemplates;
        d += sizeof(unsigned int);

        *reinterpret_cast<bool*>(d) = mIsCoordinatesAnchors;
        d += sizeof(bool);
        *reinterpret_cast<bool*>(d) = mIsPixelFormatXY;
        d += sizeof(bool);

        *reinterpret_cast<double*>(d) = mNMS_IoU;
        d += sizeof(double);

        *reinterpret_cast<int*>(d) = mThreadX;
        d += sizeof(int);
        *reinterpret_cast<int*>(d) = mThreadY;
        d += sizeof(int);
        *reinterpret_cast<int*>(d) = mThreadZ;
        d += sizeof(int);
        *reinterpret_cast<int*>(d) = mBlockX;
        d += sizeof(int);
        *reinterpret_cast<int*>(d) = mBlockY;
        d += sizeof(int);
        *reinterpret_cast<int*>(d) = mBlockZ;
        d += sizeof(int);


        d += serializeFromDevice<int>(d, mPixelMap, mChannelHeight * mChannelWidth * mNbAnchors * mNbClass * mOutputDims.d[0]);
        d += serializeFromDevice<int>(d, mPixelMapSorted, mChannelHeight * mChannelWidth * mNbAnchors * mNbClass * mOutputDims.d[0]);
        d += serializeFromDevice<int>(d, mScoresIndex, mChannelHeight * mChannelWidth * mNbAnchors * mNbClass * mOutputDims.d[0]);
        d += serializeFromDevice<float>(d, mScores, mChannelHeight * mChannelWidth * mNbAnchors * mNbClass * mOutputDims.d[0]);
        d += serializeFromDevice<float>(d, mScoresFiltered, mChannelHeight * mChannelWidth * mNbAnchors * mNbClass * mOutputDims.d[0]);

        d += serializeFromDevice<float>(d, mMxGPUIndex, mChannelHeight * mChannelWidth * mNbAnchors * mNbClass * mOutputDims.d[0]);
        d += serializeFromDevice<float>(d, mMyGPUIndex, mChannelHeight * mChannelWidth * mNbAnchors * mNbClass * mOutputDims.d[0]);
        d += serializeFromDevice<float>(d, mMwGPUIndex, mChannelHeight * mChannelWidth * mNbAnchors * mNbClass * mOutputDims.d[0]);
        d += serializeFromDevice<float>(d, mMhGPUIndex, mChannelHeight * mChannelWidth * mNbAnchors * mNbClass * mOutputDims.d[0]);
        for(unsigned int k = 0; k < mChannelHeight * mChannelWidth * mNbAnchors * mNbClass * mOutputDims.d[0]; ++k) {
            *reinterpret_cast<float*>(d) = mMxCPUIndex[k];
            d += sizeof(float);
        }
        for(unsigned int k = 0; k < mChannelHeight * mChannelWidth * mNbAnchors * mNbClass * mOutputDims.d[0]; ++k){
            *reinterpret_cast<float*>(d) = mMyCPUIndex[k];
            d += sizeof(float);
        }
        for(unsigned int k = 0; k < mChannelHeight * mChannelWidth * mNbAnchors * mNbClass * mOutputDims.d[0]; ++k) {
            *reinterpret_cast<float*>(d) = mMwCPUIndex[k];
            d += sizeof(float);
        }
        for(unsigned int k = 0; k < mChannelHeight * mChannelWidth * mNbAnchors * mNbClass * mOutputDims.d[0]; ++k) {
            *reinterpret_cast<float*>(d) = mMhCPUIndex[k];
            d += sizeof(float);
        }
        
        d += serializeFromDevice<float>(d, mScoreThreshold, mNbClass);
        for(unsigned int k = 0; k <  mNbClass; ++k) {
            *reinterpret_cast<float*>(d) = mScoreThreshold_HOST[k];
            d += sizeof(float);
        }

        for(unsigned int k = 0; k < mNbClass; ++k) {
            *reinterpret_cast<unsigned int*>(d) = mPartsPerClass_HOST[k];
            d += sizeof(unsigned int);
        }

        for(unsigned int k = 0; k < mNbClass; ++k) {
            *reinterpret_cast<unsigned int*>(d) = mTemplatesPerClass_HOST[k];
            d += sizeof(unsigned int);
        }
        /*for(unsigned int i = 0; i < mNbAnchors; ++i)
        {
            write<float>(d, mAnchors[i].x0);
            write<float>(d, mAnchors[i].y0);
            write<float>(d, mAnchors[i].x1);
            write<float>(d, mAnchors[i].y1);
        }*/
        d += serializeFromDevice<float>(d, mAnchors, 4 * mNbAnchors * mNbClass);
        for(unsigned int k = 0; k < 4 * mNbAnchors * mNbClass; ++k) {
            *reinterpret_cast<float*>(d) = mAnchors_HOST[k];
            d += sizeof(float);
        }

        d += serializeFromDevice<float>(d, mROIsBBOxFinal, mNbProposals*5*mOutputDims.d[0]);
        d += serializeFromDevice<float>(d, mROIsMapAnchorsFinal, mNbProposals*5*mOutputDims.d[0]);
        d += serializeFromDevice<unsigned int>(d, mROIsIndexFinal, mOutputDims.d[0]);

        assert(d == a + getSerializationSize());
	}

    bool supportsFormat(nvinfer1::DataType type, nvinfer1::PluginFormat format) const 
#if NV_TENSORRT_MAJOR > 7
    noexcept
#endif
    override
    {
#if NV_TENSORRT_MAJOR > 5
        return (type == nvinfer1::DataType::kFLOAT && format == nvinfer1::PluginFormat::kLINEAR );
#else
        return (type == nvinfer1::DataType::kFLOAT && format == nvinfer1::PluginFormat::kNCHW );
#endif
    }
    
    const char* getPluginType() const 
#if NV_TENSORRT_MAJOR > 7
    noexcept
#endif
    override {
        const char* ObjDet_GPU_PLUGIN_NAME{"ObjDetGPUPlugin"};
        return ObjDet_GPU_PLUGIN_NAME;
     }
    const char* getPluginVersion() const 
#if NV_TENSORRT_MAJOR > 7
    noexcept
#endif
    override { 
        const char* ObjDet_GPU_PLUGIN_VERSION{"1"};
        return ObjDet_GPU_PLUGIN_VERSION;
    }
    void destroy() 
#if NV_TENSORRT_MAJOR > 7
    noexcept
#endif
    override {     delete this; }
    IPluginV2Ext* clone() const 
#if NV_TENSORRT_MAJOR > 7
    noexcept
#endif
    override {

        IPluginV2Ext* plugin = new ObjDetGPUPlugin( mOutputDims.d[0],
                                                    mOutputDims.d[1],
                                                    mOutputDims.d[2],
                                                    mOutputDims.d[3],
                                                    mChannelHeight,
                                                    mChannelWidth,
                                                    mStimuliWidth,
                                                    mStimuliHeight,
                                                    mFeatureMapWidth,
                                                    mFeatureMapHeight,
                                                    mNbProposals,
                                                    mNbClass,
                                                    mNbAnchors,
                                                    mIsCoordinatesAnchors,
                                                    mIsPixelFormatXY,
                                                    mNMS_IoU,
                                                    mScoreThreshold_HOST.data(),
                                                    mMaxParts,
                                                    mMaxTemplates,
                                                    mPartsPerClass_HOST.data(),
                                                    mTemplatesPerClass_HOST.data(),
                                                    mAnchors_HOST.data());

        plugin->setPluginNamespace("");
        return plugin;
     }


    void setPluginNamespace(const char* pluginNamespace) 
#if NV_TENSORRT_MAJOR > 7
    noexcept
#endif
    override {
        std::string mNamespace = pluginNamespace;
     }
    const char* getPluginNamespace() const 
#if NV_TENSORRT_MAJOR > 7
    noexcept
#endif
    override { 
        return mNamespace.c_str();
    }

    nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const 
#if NV_TENSORRT_MAJOR > 7
    noexcept
#endif
    override {
        return nvinfer1::DataType::kFLOAT;
     }

    bool isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const 
#if NV_TENSORRT_MAJOR > 7
    noexcept
#endif
    override {    
        return false;
     }

    bool canBroadcastInputAcrossBatch(int inputIndex) const
#if NV_TENSORRT_MAJOR > 7
    noexcept
#endif
    override
 {
        return false;
     }

    void attachToContext(
        cudnnContext* cudnnContext, cublasContext* cublasContext, nvinfer1::IGpuAllocator* gpuAllocator) 
#if NV_TENSORRT_MAJOR > 7
    noexcept
#endif
    override { }

    void configurePlugin(const nvinfer1::Dims* inputDims, int nbInputs, const nvinfer1::Dims* outputDims, int nbOutputs,
        const nvinfer1::DataType* inputTypes, const nvinfer1::DataType* outputTypes, const bool* inputIsBroadcast,
        const bool* outputIsBroadcast, nvinfer1::PluginFormat floatFormat, int maxBatchSize) 
#if NV_TENSORRT_MAJOR > 7
    noexcept
#endif
    override { }

    void detachFromContext() 
#if NV_TENSORRT_MAJOR > 7
    noexcept
#endif
    override { }


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
    size_t serializeFromDevice(char*& hostBuffer, const T* deviceWeights, const size_t dataSize) const
    {
        checkCudaErrors(cudaMemcpy(hostBuffer, deviceWeights, dataSize*sizeof(T), cudaMemcpyDeviceToHost));
        return dataSize*sizeof(T);
    }
    void gpuThreadAllocation()
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, 0);

        const unsigned int outputMaxSizePerCls = mNbAnchors * mChannelWidth * mChannelHeight;
        const unsigned int nbBlocks = std::ceil(outputMaxSizePerCls/(float) 32.0);


        const unsigned int maxSize
            = (unsigned int)deviceProp.maxThreadsPerBlock;
        const unsigned int prefMultiple = (unsigned int)deviceProp.warpSize;

        mThreadX = 32;
        mThreadY = 1;
        mThreadZ = 1;

        mBlockX = nbBlocks;
        mBlockY = mNbClass;
        mBlockZ = (int) mOutputDims.d[0];
        std::cout << "ObjectDet: "
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
    unsigned int mStimuliWidth;
    unsigned int mStimuliHeight;
    unsigned int mFeatureMapWidth;
    unsigned int mFeatureMapHeight;
    unsigned int mChannelHeight;
    unsigned int mChannelWidth;
    unsigned int mNbProposals;
    unsigned int mNbAnchors;
    unsigned int mNbClass;
    bool mIsCoordinatesAnchors;
    bool mIsPixelFormatXY;
    double mNMS_IoU;
    float* mScoreThreshold;
    std::vector<float> mScoreThreshold_HOST;
    unsigned int mMaxParts;
    unsigned int mMaxTemplates;
    unsigned int* mPartsPerClass;
    std::vector<unsigned int> mPartsPerClass_HOST;
    unsigned int* mTemplatesPerClass;
    std::vector<unsigned int> mTemplatesPerClass_HOST;

    float* mAnchors;
    std::vector<float> mAnchors_HOST;

    float* mROIsBBOxFinal;
    float* mROIsMapAnchorsFinal;
    unsigned int* mROIsIndexFinal;

    int* mPixelMapSorted;
    int* mPixelMap;

    int* mScoresIndex;
    float* mScores;
    float* mScoresFiltered;

    float* mMxGPUIndex;
    float* mMyGPUIndex;    
    float* mMwGPUIndex;
    float* mMhGPUIndex;

    float* mMxCPUIndex;
    float* mMyCPUIndex;    
    float* mMwCPUIndex;
    float* mMhCPUIndex;

    int mThreadX;
    int mThreadY;
    int mThreadZ;
    int mBlockX;
    int mBlockY;
    int mBlockZ;
    std::string mNamespace;
};




class ObjDet_GPUPluginCreator : public nvinfer1::IPluginCreator
{
public:
    ObjDet_GPUPluginCreator() {

        mPluginAttributes.emplace_back(nvinfer1::PluginField("batchSize", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
        mPluginAttributes.emplace_back(nvinfer1::PluginField("nbOutputs", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
        mPluginAttributes.emplace_back(nvinfer1::PluginField("outputHeight", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
        mPluginAttributes.emplace_back(nvinfer1::PluginField("outputWidth", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
        mPluginAttributes.emplace_back(nvinfer1::PluginField("channelHeight", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
        mPluginAttributes.emplace_back(nvinfer1::PluginField("channelWidth", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
        mPluginAttributes.emplace_back(nvinfer1::PluginField("stimuliWidth", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
        mPluginAttributes.emplace_back(nvinfer1::PluginField("stimuliHeight", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
        mPluginAttributes.emplace_back(nvinfer1::PluginField("featureMapWidth", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
        mPluginAttributes.emplace_back(nvinfer1::PluginField("featureMapHeight", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
        mPluginAttributes.emplace_back(nvinfer1::PluginField("nbProposals", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
        mPluginAttributes.emplace_back(nvinfer1::PluginField("nbCls", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
        mPluginAttributes.emplace_back(nvinfer1::PluginField("nbAnchors", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
        mPluginAttributes.emplace_back(nvinfer1::PluginField("isCoordinatesAnchors", nullptr, nvinfer1::PluginFieldType::kCHAR, 1));
        mPluginAttributes.emplace_back(nvinfer1::PluginField("isPixelFormatXY", nullptr, nvinfer1::PluginFieldType::kCHAR, 1));
        mPluginAttributes.emplace_back(nvinfer1::PluginField("nmsIoU", nullptr, nvinfer1::PluginFieldType::kFLOAT64, 1));
        mPluginAttributes.emplace_back(nvinfer1::PluginField("scoreThreshold", nullptr, nvinfer1::PluginFieldType::kFLOAT32, nbCls));
        mPluginAttributes.emplace_back(nvinfer1::PluginField("maxParts", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
        mPluginAttributes.emplace_back(nvinfer1::PluginField("maxTemplates", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
        mPluginAttributes.emplace_back(nvinfer1::PluginField("numPartsPerClass", nullptr, nvinfer1::PluginFieldType::kINT32, nbCls));
        mPluginAttributes.emplace_back(nvinfer1::PluginField("numTemplatesPerClass", nullptr, nvinfer1::PluginFieldType::kINT32, nbCls));
        mPluginAttributes.emplace_back(nvinfer1::PluginField("anchor", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 4*nbCls*nbAnchors));

        mFC.nbFields = mPluginAttributes.size();
        mFC.fields = mPluginAttributes.data();
    }

    ~ObjDet_GPUPluginCreator() 
#if NV_TENSORRT_MAJOR > 7
    noexcept
#endif
    override  {

    }

    const char* getPluginName() const 
#if NV_TENSORRT_MAJOR > 7
    noexcept
#endif
    override {
        const char* ObjDet_GPU_PLUGIN_NAME{"ObjDetGPUPlugin"};
        return ObjDet_GPU_PLUGIN_NAME;
    }

    const char* getPluginVersion() const 
#if NV_TENSORRT_MAJOR > 7
    noexcept
#endif
    override {
        std::cout << "Create plugin start..." << std::endl;
        const char* ObjDet_GPU_PLUGIN_VERSION{"1"};
        return ObjDet_GPU_PLUGIN_VERSION;
    }

    const nvinfer1::PluginFieldCollection* getFieldNames() 
#if NV_TENSORRT_MAJOR > 7
    noexcept
#endif
    override {
        return &mFC;
    }

    nvinfer1::IPluginV2Ext* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) 
#if NV_TENSORRT_MAJOR > 7
    noexcept
#endif
    override {
        const nvinfer1::PluginField* fields = fc->fields;
        int nbFields = fc->nbFields;

        for (int i = 0; i < nbFields; ++i)
        {
            const char* attrName = fields[i].name;
            if (!strcmp(attrName, "batchSize"))
            {
                //ASSERT(fields[i].type == nvinfer1::PluginFieldType::kINT32);
                batchSize = *(static_cast<const int*>(fields[i].data));
            }
            if (!strcmp(attrName, "nbOutputs"))
            {
                //ASSERT(fields[i].type == nvinfer1::PluginFieldType::kINT32);
                nbOutputs = *(static_cast<const int*>(fields[i].data));
            }
            if (!strcmp(attrName, "outputHeight"))
            {
                //ASSERT(fields[i].type == nvinfer1::PluginFieldType::kINT32);
                outputHeight = *(static_cast<const int*>(fields[i].data));
            }
            if (!strcmp(attrName, "outputWidth"))
            {
                //ASSERT(fields[i].type == nvinfer1::PluginFieldType::kINT32);
                outputWidth = *(static_cast<const int*>(fields[i].data));
            }

            if (!strcmp(attrName, "channelHeight"))
            {
                //ASSERT(fields[i].type == nvinfer1::PluginFieldType::kINT32);
                channelHeight = *(static_cast<const int*>(fields[i].data));
            }
            if (!strcmp(attrName, "channelWidth"))
            {
                //ASSERT(fields[i].type == nvinfer1::PluginFieldType::kINT32);
                channelWidth = *(static_cast<const int*>(fields[i].data));
            }
            if (!strcmp(attrName, "stimuliWidth"))
            {
                //ASSERT(fields[i].type == nvinfer1::PluginFieldType::kINT32);
                stimuliWidth = *(static_cast<const int*>(fields[i].data));
            }
            if (!strcmp(attrName, "stimuliHeight"))
            {
                //ASSERT(fields[i].type == nvinfer1::PluginFieldType::kINT32);
                stimuliHeight = *(static_cast<const int*>(fields[i].data));
            }

            if (!strcmp(attrName, "featureMapWidth"))
            {
                //ASSERT(fields[i].type == nvinfer1::PluginFieldType::kINT32);
                featureMapWidth = *(static_cast<const int*>(fields[i].data));
            }
            if (!strcmp(attrName, "featureMapHeight"))
            {
                //ASSERT(fields[i].type == nvinfer1::PluginFieldType::kINT32);
                featureMapHeight = *(static_cast<const int*>(fields[i].data));
            }
            if (!strcmp(attrName, "nbProposals"))
            {
                //ASSERT(fields[i].type == nvinfer1::PluginFieldType::kINT32);
                nbProposals = *(static_cast<const int*>(fields[i].data));
            }
            if (!strcmp(attrName, "nbCls"))
            {
                //ASSERT(fields[i].type == nvinfer1::PluginFieldType::kINT32);
                nbCls = *(static_cast<const int*>(fields[i].data));
            }
            if (!strcmp(attrName, "nbAnchors"))
            {
                //ASSERT(fields[i].type == nvinfer1::PluginFieldType::kINT32);
                nbAnchors = *(static_cast<const int*>(fields[i].data));
            }
            if (!strcmp(attrName, "isCoordinatesAnchors"))
            {
                //ASSERT(fields[i].type == nvinfer1::PluginFieldType::kCHAR);
                isCoordinatesAnchors = *(static_cast<const char*>(fields[i].data));
            }
            if (!strcmp(attrName, "isPixelFormatXY"))
            {
                //ASSERT(fields[i].type == nvinfer1::PluginFieldType::kCHAR);
                isPixelFormatXY = *(static_cast<const char*>(fields[i].data));
            }
            if (!strcmp(attrName, "nmsIoU"))
            {
                //ASSERT(fields[i].type == nvinfer1::PluginFieldType::kFLOAT64);
                nmsIoU = *(static_cast<const double*>(fields[i].data));
            }

            if (!strcmp(attrName, "scoreThreshold"))
            {
                //ASSERT(fields[i].type == nvinfer1::PluginFieldType::kFLOAT32);

                const float* threshold = static_cast<const float*>(fields[i].data);
                for (int j = 0; j < nbCls; ++j)
                {
                    scoreThreshold.push_back(*threshold);
                    threshold++;
                }
            }
            if (!strcmp(attrName, "maxParts"))
            {
                //ASSERT(fields[i].type == nvinfer1::PluginFieldType::kINT32);
                maxParts = *(static_cast<const int*>(fields[i].data));
            }
            if (!strcmp(attrName, "maxTemplates"))
            {
                //ASSERT(fields[i].type == nvinfer1::PluginFieldType::kINT32);
                maxTemplates = *(static_cast<const int*>(fields[i].data));
            }
            if (!strcmp(attrName, "numPartsPerClass"))
            {
                //ASSERT(fields[i].type == nvinfer1::PluginFieldType::kINT32);

                const int* p = static_cast<const int*>(fields[i].data);
                for (int j = 0; j < nbCls; ++j)
                {
                    numPartsPerClass.push_back(*p);
                    p++;
                }
            }
            if (!strcmp(attrName, "numTemplatesPerClass"))
            {
                //ASSERT(fields[i].type == nvinfer1::PluginFieldType::kINT32);

                const int* t = static_cast<const int*>(fields[i].data);
                for (int j = 0; j < nbCls; ++j)
                {
                    numTemplatesPerClass.push_back(*t);
                    t++;
                }
            }

            if (!strcmp(attrName, "anchor"))
            {
                //ASSERT(fields[i].type == nvinfer1::PluginFieldType::kFLOAT32);

                const float* anc = static_cast<const float*>(fields[i].data);
                for (int j = 0; j < 4*nbAnchors*nbCls; ++j)
                {
                    anchor.push_back(*anc);
                    anc++;
                }
            }
        }

        // This object will be deleted when the network is destroyed, which will
        // call RPROIPlugin::terminate()
        ObjDetGPUPlugin* plugin = new ObjDetGPUPlugin(    batchSize,
                                                            nbOutputs,
                                                            outputHeight,
                                                            outputWidth,
                                                            channelHeight,
                                                            channelWidth,
                                                            stimuliWidth,
                                                            stimuliHeight,
                                                            featureMapWidth,
                                                            featureMapHeight,
                                                            nbProposals,
                                                            nbCls,
                                                            nbAnchors,
                                                            isCoordinatesAnchors,
                                                            isPixelFormatXY,
                                                            nmsIoU,
                                                            scoreThreshold.data(),
                                                            maxParts,
                                                            maxTemplates,
                                                            numPartsPerClass.data(),
                                                            numTemplatesPerClass.data(),
                                                            anchor.data());

        //plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;


    }

    nvinfer1::IPluginV2Ext* deserializePlugin(const char* name, const void* serialData, size_t serialLength) 
#if NV_TENSORRT_MAJOR > 7
    noexcept
#endif
    override {
        ObjDetGPUPlugin* plugin = new ObjDetGPUPlugin(serialData, serialLength);
        return plugin;
    }
    void setPluginNamespace(const char* pluginNamespace) 
#if NV_TENSORRT_MAJOR > 7
    noexcept
#endif
    override    {
        mNamespace = pluginNamespace;
    }
    const char* getPluginNamespace() const 
#if NV_TENSORRT_MAJOR > 7
    noexcept
#endif
    override{
        return mNamespace.c_str();
    }


private:
    nvinfer1::PluginFieldCollection mFC;

    unsigned int batchSize;
    unsigned int nbOutputs;
    unsigned int outputHeight;
    unsigned int outputWidth;
    unsigned int channelHeight;
    unsigned int channelWidth;
    unsigned int stimuliWidth;
    unsigned int stimuliHeight;
    unsigned int featureMapWidth;
    unsigned int featureMapHeight;
    unsigned int nbProposals;
    unsigned int nbCls;
    unsigned int nbAnchors;
    bool isCoordinatesAnchors;
    bool isPixelFormatXY;
    double nmsIoU;
    std::vector<float> scoreThreshold;
    unsigned int maxParts;
    unsigned int maxTemplates;
    std::vector<unsigned int> numPartsPerClass;
    std::vector<unsigned int> numTemplatesPerClass;
    std::vector<float> anchor;
    std::string mNamespace;
    std::vector<nvinfer1::PluginField> mPluginAttributes;
};

REGISTER_TENSORRT_PLUGIN(ObjDet_GPUPluginCreator);

#endif
