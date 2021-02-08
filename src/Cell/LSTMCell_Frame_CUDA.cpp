/*
    (C) Copyright 2018 CEA LIST. All Rights Reserved.
    Contributor(s): Thibault ALLENET (thibault.allenet@cea.fr)
                    Olivier BICHLER (olivier.bichler@cea.fr)
                    David BRIAND (david.briand@cea.fr)
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
#ifdef CUDA
#include <cudnn.h>
#if CUDNN_VERSION >= 5000

#include "GradientCheck.hpp"
#include "Cell/LSTMCell_Frame_CUDA.hpp"
#include "DeepNet.hpp"
#include "third_party/half.hpp"

template <>
N2D2::Registrar<N2D2::LSTMCell>
N2D2::LSTMCell_Frame_CUDA<half_float::half>::mRegistrar("Frame_CUDA",
    N2D2::LSTMCell_Frame_CUDA<half_float::half>::create,
    N2D2::Registrar<N2D2::LSTMCell>::Type<half_float::half>());

template <>
N2D2::Registrar<N2D2::LSTMCell>
N2D2::LSTMCell_Frame_CUDA<float>::mRegistrar("Frame_CUDA",
    N2D2::LSTMCell_Frame_CUDA<float>::create,
    N2D2::Registrar<N2D2::LSTMCell>::Type<float>());

template <>
N2D2::Registrar<N2D2::LSTMCell>
N2D2::LSTMCell_Frame_CUDA<double>::mRegistrar("Frame_CUDA",
    N2D2::LSTMCell_Frame_CUDA<double>::create,
    N2D2::Registrar<N2D2::LSTMCell>::Type<double>());

template <class T>
N2D2::LSTMCell_Frame_CUDA<T>::LSTMCell_Frame_CUDA(const DeepNet& deepNet, 
    const std::string& name,
	unsigned int seqLength,
    unsigned int batchSize,
    unsigned int inputDim,
    unsigned int numberLayers,
    unsigned int hiddenSize,
    unsigned int algo,
    unsigned int nbOutputs,
    unsigned int bidirectional,
	unsigned int inputMode,
    float dropout,
	bool singleBackpropFeeding)
    : Cell(deepNet, name, nbOutputs),
      LSTMCell(deepNet, name,
				seqLength,
				batchSize,
				inputDim,
				numberLayers,
				hiddenSize,
				algo,
				nbOutputs,
				bidirectional,
				inputMode,
                dropout,
				singleBackpropFeeding),
      Cell_Frame_CUDA<T>(deepNet, name, nbOutputs),
      mWeights(std::make_shared<CudaTensor<T> >()),
	  mhx(std::make_shared<CudaTensor<T> >()),
	  mDiffhy(std::make_shared<CudaTensor<T> >()),
	  mcx(std::make_shared<CudaTensor<T> >()),
	  mDiffcy(std::make_shared<CudaTensor<T> >()),
	  mContinousBatch(false)

{ //ctr
    mWeightsSolver = std::make_shared<SGDSolver_Frame_CUDA<T> >();

	CHECK_CUDNN_STATUS(cudnnCreateDropoutDescriptor(&mDropoutDesc));

    CHECK_CUDNN_STATUS(cudnnCreateRNNDescriptor(&mLSTMDesc));

	CHECK_CUDNN_STATUS(cudnnCreateFilterDescriptor(&wDesc));
	CHECK_CUDNN_STATUS(cudnnCreateFilterDescriptor(&dwDesc));
}

template <class T>
void N2D2::LSTMCell_Frame_CUDA<T>::initialize(){
	//********************************************************************************
	//------------------------------------ALLOCATION----------------------------------
	//********************************************************************************
	//---------------------------------------------------------------
	//Dropout allocation & initialisation (needed for LSTM descriptor)
	//---------------------------------------------------------------

	seed = 133ull;

	CHECK_CUDNN_STATUS(cudnnDropoutGetStatesSize(CudaContext::cudnnHandle(), &mStatesize));

	CHECK_CUDA_STATUS(cudaMalloc(&mStates, mStatesize));

	CHECK_CUDNN_STATUS(cudnnSetDropoutDescriptor(mDropoutDesc,
                            	CudaContext::cudnnHandle(),
                            	mDropout,
                        		mStates,
                            	mStatesize,
                            	seed));

	//----------------------------
	//Create LSTM desc & initialize
	//----------------------------

	if (mInputMode == 0){
		mInput_Mode = CUDNN_SKIP_INPUT;
	} else if (mInputMode == 1){
		mInput_Mode = CUDNN_LINEAR_INPUT;
	} else {
		throw std::runtime_error("LSTMCell_Frame_CUDA InputMode invalid, LSTM name : " + mName + " should be 0 to skip or 1 for Linear");
	}

#if CUDNN_VERSION >= 6000
	if (mAlgo == 1){
		mCudnnAlgo = CUDNN_RNN_ALGO_PERSIST_STATIC;
	} else if (mAlgo == 0){
		mCudnnAlgo = CUDNN_RNN_ALGO_STANDARD;
	} else {
		throw std::runtime_error("LSTMCell_Frame_CUDA Algo invalid, LSTM name : " + mName + " should be 0 for Standard or 1 for STATIC");
	}


	CHECK_CUDNN_STATUS(cudnnSetRNNDescriptor_v6(CudaContext::cudnnHandle(),
							mLSTMDesc,
							mHiddenSize,
							mNumberLayers,
							mDropoutDesc,
							mInput_Mode,
							(mBidirectional ? CUDNN_BIDIRECTIONAL : CUDNN_UNIDIRECTIONAL),
							CUDNN_LSTM,
							mCudnnAlgo,
							CudaContext::data_type<T>::value));
#else
	CHECK_CUDNN_STATUS(cudnnSetRNNDescriptor(mLSTMDesc,
							mHiddenSize,
							mNumberLayers,
							mDropoutDesc,
							mInput_Mode,
							(mBidirectional ? CUDNN_BIDIRECTIONAL : CUDNN_UNIDIRECTIONAL),
							CUDNN_LSTM,
							CudaContext::data_type<T>::value));
#endif

	//----------------------------------------------------------------
	//Descriptors for Input and output + gradients for cudnn functions
	//----------------------------------------------------------------

	//----------------xDesc et dxDesc --------------
	xDesc.reset(new RNNInOutType<T>(
			mSeqLength,
			{(int)mBatchSize, (int)mInputDim, 1,1},
			{(int)mInputDim, 1, 1,1}));

	dxDesc.reset(new RNNInOutType<T>(
			mSeqLength,
			{(int)mBatchSize, (int)mInputDim, 1,1},
			{(int)mInputDim, 1, 1,1}));

	//---------------yDesc et dyDesc-----------------
	yDesc.reset(new RNNInOutType<T>(
			mSeqLength,
			{(int)mBatchSize,(int)(mHiddenSize * biDirScale), 1,1},
			{(int)(biDirScale * mHiddenSize), 1, 1,1}));

	dyDesc.reset(new RNNInOutType<T>(
			mSeqLength,
			{(int)mBatchSize,(int)(mHiddenSize * biDirScale), 1,1},
			{(int)(biDirScale * mHiddenSize), 1, 1,1}));

	//-----------------------------------------------
	// Create mWorkspace and Training reserved memory
	//-----------------------------------------------

	CHECK_CUDNN_STATUS(cudnnGetRNNWorkspaceSize(CudaContext::cudnnHandle(), mLSTMDesc, mSeqLength, xDesc->getdescs(), &mWorkSize));
	CHECK_CUDA_STATUS(cudaMalloc((void**)&mWorkspace, mWorkSize));

	CHECK_CUDNN_STATUS(cudnnGetRNNTrainingReserveSize(CudaContext::cudnnHandle(), mLSTMDesc, mSeqLength, xDesc->getdescs(), &mReserveSize));
	CHECK_CUDA_STATUS(cudaMalloc((void**)&mReserveSpace, mReserveSize));


	//---------------------------------------------------------------------------------------------------------------------------------------------------------
	// Allocate structures for last inference Hidden&Cell states (mhy/mcy) and previous time step ("time -1") hidden&Cell states differential (mDiffhx&mDiffcx)
	//---------------------------------------------------------------------------------------------------------------------------------------------------------

	if (mhy.empty()){
		mhy.resize({1,mHiddenSize,mBatchSize,mNumberLayers*biDirScale},T(0.));
	}else {
		if (mhy.dimX() != 1 || mhy.dimY() != mHiddenSize || mhy.dimZ() != mBatchSize || mhy.dimB() != mNumberLayers*biDirScale){
			throw std::runtime_error("Cell " + mName + ", wrong size for hy");
		}
	}

	if (mDiffhx.empty()){
		mDiffhx.resize({1,mHiddenSize,mBatchSize,mNumberLayers*biDirScale},T(0.));
	}else {
		if (mDiffhx.dimX() != 1 || mDiffhx.dimY() != mHiddenSize || mDiffhx.dimZ() != mBatchSize || mDiffhx.dimB() != mNumberLayers*biDirScale){
			throw std::runtime_error("Cell " + mName + ", wrong size for dhx");
		}
	}

	if (mcy.empty()){
		mcy.resize({1,mHiddenSize,mBatchSize,mNumberLayers*biDirScale},T(0.));
	}else {
		if (mcy.dimX() != 1 || mcy.dimY() != mHiddenSize || mcy.dimZ() != mBatchSize || mcy.dimB() != mNumberLayers*biDirScale){
			throw std::runtime_error("Cell " + mName + ", wrong size for cy");
		}
	}

	if (mDiffcx.empty()){
		mDiffcx.resize({1,mHiddenSize,mBatchSize,mNumberLayers*biDirScale},T(0.));
	}else {
		if (mDiffcx.dimX() != 1 || mDiffcx.dimY() != mHiddenSize || mDiffcx.dimZ() != mBatchSize || mDiffcx.dimB() != mNumberLayers*biDirScale){
			throw std::runtime_error("Cell " + mName + ", wrong size for dcx");
		}
	}

	//********************************************************************************
	//------------------------------------INITIALISATION------------------------------
	//********************************************************************************

	//------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	// Allocate and initialize structures for last inference Hidden&Cell states differential (mDiffhy&mDiffcy) and previous time step ("time -1") hidden&Cell states (mhx&mcx)
	//------------------------------------------------------------------------------------------------------------------------------------------------------------------------

	if (mhx->empty()) {
		mhx->resize({1, mHiddenSize, mBatchSize, mNumberLayers*biDirScale});
		mhxFiller->apply((*mhx));
		mhx->synchronizeHToD();
	}else {
		if (mhx->dimX() != 1 || mhx->dimY() != mHiddenSize || mhx->dimZ() != mBatchSize || mhx->dimB() != mNumberLayers*biDirScale){
			throw std::runtime_error("Cell " + mName + ", wrong size for hx");
		}
	}

	if (mDiffhy->empty()) {
		mDiffhy->resize({1, mHiddenSize, mBatchSize, mNumberLayers*biDirScale},T(0.));
	}else {
		if (mDiffhy->dimX() != 1 || mDiffhy->dimY() != mHiddenSize || mDiffhy->dimZ() != mBatchSize || mDiffhy->dimB() != mNumberLayers*biDirScale){
			throw std::runtime_error("Cell " + mName + ", wrong size for hy");
		}
	}

	if (mcx->empty()) {
		mcx->resize({1, mHiddenSize, mBatchSize, mNumberLayers*biDirScale});
		mcxFiller->apply((*mcx));
		mcx->synchronizeHToD();
	}else {
		if (mcx->dimX() != 1 || mcx->dimY() != mHiddenSize || mcx->dimZ() != mBatchSize || mcx->dimB() != mNumberLayers*biDirScale){
			throw std::runtime_error("Cell " + mName + ", wrong size for cx");
		}
	}


	if (mDiffcy->empty()) {
		mDiffcy->resize({1, mHiddenSize, mBatchSize, mNumberLayers*biDirScale},T(0.));
	}else {
		if (mDiffcy->dimX() != 1 || mDiffcy->dimY() != mHiddenSize || mDiffcy->dimZ() != mBatchSize || mDiffcy->dimB() != mNumberLayers*biDirScale){
			throw std::runtime_error("Cell " + mName + ", wrong size for cy");
		}
	}


	if (mOutputsLocal.empty()) {
		mOutputsLocal.resize({1, mHiddenSize*biDirScale, mBatchSize, mSeqLength},T(0.));
	}else {
		if (mOutputsLocal.dimX() != 1 || mOutputsLocal.dimY() != mHiddenSize*biDirScale || mOutputsLocal.dimZ() != mBatchSize || mOutputsLocal.dimB() !=  mSeqLength){
			throw std::runtime_error("Cell " + mName + ", wrong size for mOutputsLocal");
		}
	}

	if (mDiffInputsLocal.empty()) {
		mDiffInputsLocal.resize({1, mHiddenSize*biDirScale, mBatchSize, mSeqLength},T(0.));
	}else {
		if (mDiffInputsLocal.dimX() != 1 || mDiffInputsLocal.dimY() != mHiddenSize*biDirScale || mDiffInputsLocal.dimZ() != mBatchSize || mDiffInputsLocal.dimB() !=  mSeqLength){
			throw std::runtime_error("Cell " + mName + ", wrong size for mDiffInputsLocal");
		}
	}

	if(mDiffOutputs.empty()){
		mDiffOutputs.push_back(new CudaTensor<T>({1, mInputDim, mBatchSize, mSeqLength}));
	}else {
		if (mDiffOutputs.back().dimX() != 1 || mDiffOutputs.back().dimY() !=  mInputDim || mDiffOutputs.back().dimZ() != mBatchSize || mDiffOutputs.back().dimB() != mSeqLength){
			throw std::runtime_error("Cell " + mName + ", wrong size for mDiffOutputs");
		}
	}


	//--------------------------------
	// Allocate and initialize weights
	//--------------------------------
	CHECK_CUDNN_STATUS(cudnnGetRNNParamsSize(
				CudaContext::cudnnHandle(),
				mLSTMDesc,
				xDesc->getdescs()[0],
				&mWeightsSize,
				CudaContext::data_type<T>::value));

	if (mWeights->empty()){

		//fprintf(stdout,"mWeightsSize : %zu \n", mWeightsSize/sizeof(CudaContext::data_type<T>::value));
		const std::array<int, 3> dims{static_cast<int>(mWeightsSize/sizeof(CudaContext::data_type<T>::value)),1,1};

		CHECK_CUDNN_STATUS(cudnnSetFilterNdDescriptor(wDesc,CudaContext::data_type<T>::value, CUDNN_TENSOR_NCHW, 3, dims.data()));
		CHECK_CUDNN_STATUS(cudnnSetFilterNdDescriptor(dwDesc,CudaContext::data_type<T>::value, CUDNN_TENSOR_NCHW, 3, dims.data()));
		mWeights->resize({1, 1, 1,static_cast<unsigned int>(mWeightsSize/sizeof(CudaContext::data_type<T>::value))},T(0.));
		//mDiffWeights.resize({1, 1, 1,static_cast<unsigned int>(mWeightsSize/sizeof(CudaContext::data_type<T>::value))},T(0.));


		//-----------------------------Fill all Weights and Bias from previous layer :: for each layer ---------------------------------------//

		CudaInterface<T> fillInputWeights_1stLayer;
		CudaInterface<T> fillInputWeights;
		CudaInterface<T> fillRecurrentWeights;
		CudaInterface<T> fillBias;
		T *linLayerMat;
		T *linLayerBias;

		for (unsigned int layer = 0; layer < mNumberLayers * biDirScale; layer++) {
			for (unsigned int linLayerID = 0; linLayerID < 8; linLayerID++) {


				cudnnFilterDescriptor_t linLayerMatDesc;
				CHECK_CUDNN_STATUS(cudnnCreateFilterDescriptor(&linLayerMatDesc));

				CHECK_CUDNN_STATUS(cudnnGetRNNLinLayerMatrixParams( CudaContext::cudnnHandle(),
																	mLSTMDesc,
																	layer,
																	xDesc->getdescs()[0],
																	wDesc,
																	mWeights->getDevicePtr(),
																	linLayerID,
																	linLayerMatDesc,
																	(void**)&linLayerMat));
				cudnnDataType_t dataType;
				cudnnTensorFormat_t format;
				int nbDims;
				int filterDimA[3];
				CHECK_CUDNN_STATUS(cudnnGetFilterNdDescriptor(linLayerMatDesc,
															3,
															&dataType,
															&format,
															&nbDims,
															filterDimA));

				if (linLayerID== 0){

					Random::mtSeed(4);
					if (layer==0 || (layer==1 && mBidirectional==1)){

						fillInputWeights_1stLayer.push_back(new CudaTensor<T>({1,1,(unsigned int)(filterDimA[2]*filterDimA[1]*filterDimA[0]),1}));
						fillInputWeights_1stLayer.back().setDevicePtr(linLayerMat);
						mWeightsPreviousLayerInputGateFiller_1stLayer->apply(fillInputWeights_1stLayer.back());
						fillInputWeights_1stLayer.back().synchronizeHToD();
					} else {

						fillInputWeights.push_back(new CudaTensor<T>({1,1,(unsigned int)(filterDimA[2]*filterDimA[1]*filterDimA[0]),1}));
						fillInputWeights.back().setDevicePtr(linLayerMat);
						mWeightsPreviousLayerInputGateFiller->apply(fillInputWeights.back());
						fillInputWeights.back().synchronizeHToD();
					}


				}else if (linLayerID== 1){

					Random::mtSeed(5);
					if (layer==0 || (layer==1 && mBidirectional==1)){
						fillInputWeights_1stLayer.push_back(new CudaTensor<T>({1,1,(unsigned int)(filterDimA[2]*filterDimA[1]*filterDimA[0]),1}));
						fillInputWeights_1stLayer.back().setDevicePtr(linLayerMat);
						mWeightsPreviousLayerForgetGateFiller_1stLayer->apply(fillInputWeights_1stLayer.back());
						fillInputWeights_1stLayer.back().synchronizeHToD();
					} else {
						fillInputWeights.push_back(new CudaTensor<T>({1,1,(unsigned int)(filterDimA[2]*filterDimA[1]*filterDimA[0]),1}));
						fillInputWeights.back().setDevicePtr(linLayerMat);
						mWeightsPreviousLayerForgetGateFiller->apply(fillInputWeights.back());
						fillInputWeights.back().synchronizeHToD();
					}


				}else if (linLayerID== 2){

					Random::mtSeed(6);
					if (layer==0 || (layer==1 && mBidirectional==1)){
						fillInputWeights_1stLayer.push_back(new CudaTensor<T>({1,1,(unsigned int)(filterDimA[2]*filterDimA[1]*filterDimA[0]),1}));
						fillInputWeights_1stLayer.back().setDevicePtr(linLayerMat);
						mWeightsPreviousLayerCellGateFiller_1stLayer->apply(fillInputWeights_1stLayer.back());
						fillInputWeights_1stLayer.back().synchronizeHToD();
					} else {
						fillInputWeights.push_back(new CudaTensor<T>({1,1,(unsigned int)(filterDimA[2]*filterDimA[1]*filterDimA[0]),1}));
						fillInputWeights.back().setDevicePtr(linLayerMat);
						mWeightsPreviousLayerCellGateFiller->apply(fillInputWeights.back());
						fillInputWeights.back().synchronizeHToD();
					}


				}else if (linLayerID== 3){

					Random::mtSeed(7);
					if (layer==0 || (layer==1 && mBidirectional==1)){
						fillInputWeights_1stLayer.push_back(new CudaTensor<T>({1,1,(unsigned int)(filterDimA[2]*filterDimA[1]*filterDimA[0]),1}));
						fillInputWeights_1stLayer.back().setDevicePtr(linLayerMat);
						mWeightsPreviousLayerOutputGateFiller_1stLayer->apply(fillInputWeights_1stLayer.back());
						fillInputWeights_1stLayer.back().synchronizeHToD();
					} else {
						fillInputWeights.push_back(new CudaTensor<T>({1,1,(unsigned int)(filterDimA[2]*filterDimA[1]*filterDimA[0]),1}));
						fillInputWeights.back().setDevicePtr(linLayerMat);
						mWeightsPreviousLayerOutputGateFiller->apply(fillInputWeights.back());
						fillInputWeights.back().synchronizeHToD();
					}


				}else if (linLayerID== 4){
					Random::mtSeed(8);
					fillRecurrentWeights.push_back(new CudaTensor<T>({1,1,(unsigned int)(filterDimA[2]*filterDimA[1]*filterDimA[0]),1}));
					fillRecurrentWeights.back().setDevicePtr(linLayerMat);
					mWeightsRecurrentInputGateFiller->apply(fillRecurrentWeights.back());
					fillRecurrentWeights.back().synchronizeHToD();

				}else if (linLayerID== 5){
					Random::mtSeed(9);
					fillRecurrentWeights.push_back(new CudaTensor<T>({1,1,(unsigned int)(filterDimA[2]*filterDimA[1]*filterDimA[0]),1}));
					fillRecurrentWeights.back().setDevicePtr(linLayerMat);
					mWeightsRecurrentForgetGateFiller->apply(fillRecurrentWeights.back());
					fillRecurrentWeights.back().synchronizeHToD();

				}else if (linLayerID== 6){
					Random::mtSeed(10);
					fillRecurrentWeights.push_back(new CudaTensor<T>({1,1,(unsigned int)(filterDimA[2]*filterDimA[1]*filterDimA[0]),1}));
					fillRecurrentWeights.back().setDevicePtr(linLayerMat);
					mWeightsRecurrentCellGateFiller->apply(fillRecurrentWeights.back());
					fillRecurrentWeights.back().synchronizeHToD();

				}else if (linLayerID== 7){
					Random::mtSeed(11);
					fillRecurrentWeights.push_back(new CudaTensor<T>({1,1,(unsigned int)(filterDimA[2]*filterDimA[1]*filterDimA[0]),1}));
					fillRecurrentWeights.back().setDevicePtr(linLayerMat);
					mWeightsRecurrentOutputGateFiller->apply(fillRecurrentWeights.back());
					fillRecurrentWeights.back().synchronizeHToD();
				}

				CHECK_CUDNN_STATUS(cudnnDestroyFilterDescriptor(linLayerMatDesc));


				cudnnFilterDescriptor_t linLayerBiasDesc;
				CHECK_CUDNN_STATUS(cudnnCreateFilterDescriptor(&linLayerBiasDesc));


				CHECK_CUDNN_STATUS(cudnnGetRNNLinLayerBiasParams( CudaContext::cudnnHandle(),
																	mLSTMDesc,
																	layer,
																	xDesc->getdescs()[0],
																	wDesc,
																	mWeights->getDevicePtr(),
																	linLayerID,
																	linLayerBiasDesc,
																	(void**)&linLayerBias));

				CHECK_CUDNN_STATUS(cudnnGetFilterNdDescriptor(linLayerBiasDesc,
															3,
															&dataType,
															&format,
															&nbDims,
															filterDimA));

				if (linLayerID== 0){
					fillBias.push_back(new CudaTensor<T>({1,1,(unsigned int)(filterDimA[2]*filterDimA[1]*filterDimA[0]),1}));
					fillBias.back().setDevicePtr(linLayerBias);
					mBiasPreviousLayerInputGateFiller->apply(fillBias.back());
					fillBias.back().synchronizeHToD();

				}else if (linLayerID== 1){

					fillBias.push_back(new CudaTensor<T>({1,1,(unsigned int)(filterDimA[2]*filterDimA[1]*filterDimA[0]),1}));
					fillBias.back().setDevicePtr(linLayerBias);
					mBiasPreviousLayerForgetGateFiller->apply(fillBias.back());
					fillBias.back().synchronizeHToD();

				}else if (linLayerID== 2){

					fillBias.push_back(new CudaTensor<T>({1,1,(unsigned int)(filterDimA[2]*filterDimA[1]*filterDimA[0]),1}));
					fillBias.back().setDevicePtr(linLayerBias);
					mBiasPreviousLayerCellGateFiller->apply(fillBias.back());
					fillBias.back().synchronizeHToD();

				}else if (linLayerID== 3){

					fillBias.push_back(new CudaTensor<T>({1,1,(unsigned int)(filterDimA[2]*filterDimA[1]*filterDimA[0]),1}));
					fillBias.back().setDevicePtr(linLayerBias);
					mBiasPreviousLayerOutputGateFiller->apply(fillBias.back());
					fillBias.back().synchronizeHToD();

				}else if (linLayerID== 4){

					fillBias.push_back(new CudaTensor<T>({1,1,(unsigned int)(filterDimA[2]*filterDimA[1]*filterDimA[0]),1}));
					fillBias.back().setDevicePtr(linLayerBias);
					mBiasRecurrentInputGateFiller->apply(fillBias.back());
					fillBias.back().synchronizeHToD();

				}else if (linLayerID== 5){

					fillBias.push_back(new CudaTensor<T>({1,1,(unsigned int)(filterDimA[2]*filterDimA[1]*filterDimA[0]),1}));
					fillBias.back().setDevicePtr(linLayerBias);
					mBiasRecurrentForgetGateFiller->apply(fillBias.back());
					fillBias.back().synchronizeHToD();

				}else if (linLayerID== 6){

					fillBias.push_back(new CudaTensor<T>({1,1,(unsigned int)(filterDimA[2]*filterDimA[1]*filterDimA[0]),1}));
					fillBias.back().setDevicePtr(linLayerBias);
					mBiasRecurrentCellGateFiller->apply(fillBias.back());
					fillBias.back().synchronizeHToD();

				}else if (linLayerID== 7){

					fillBias.push_back(new CudaTensor<T>({1,1,(unsigned int)(filterDimA[2]*filterDimA[1]*filterDimA[0]),1}));
					fillBias.back().setDevicePtr(linLayerBias);
					mBiasRecurrentOutputGateFiller->apply(fillBias.back());
					fillBias.back().synchronizeHToD();

				}
				CHECK_CUDNN_STATUS(cudnnDestroyFilterDescriptor(linLayerBiasDesc));
			}
		}

	}else {
		if (mWeights->dimX() != 1 || mWeights->dimY() != 1 || mWeights->dimZ() != 1 || mWeights->dimB() != (mWeightsSize/sizeof(CudaContext::data_type<T>::value))){
			throw std::runtime_error("Cell " + mName + ", wrong size for Weights");
		}
	}
}

template <class T>
void N2D2::LSTMCell_Frame_CUDA<T>::propagate(bool inference){
	mInputs.synchronizeHBasedToD();
	mhx->synchronizeHBasedToD();
	mcx->synchronizeHBasedToD();

	std::shared_ptr<CudaDeviceTensor<T> > input0 = cuda_device_tensor_cast<T>(mInputs[0]);

	if (inference){

		CHECK_CUDNN_STATUS(cudnnRNNForwardInference(CudaContext::cudnnHandle(),
												mLSTMDesc,
												mSeqLength,
												xDesc->getdescs(),
												input0->getDevicePtr(),
												mhx->getCudnnTensorDesc(),
												mhx->getDevicePtr(),
												mcx->getCudnnTensorDesc(),
												mcx->getDevicePtr(),
												wDesc,
												mWeights->getDevicePtr(),
												yDesc->getdescs(),
												mOutputsLocal.getDevicePtr(),
												mhy.getCudnnTensorDesc(),
												mhy.getDevicePtr(),
												mcy.getCudnnTensorDesc(),
												mcy.getDevicePtr(),
												mWorkspace,
												mWorkSize));
	}else {


		CHECK_CUDNN_STATUS(cudnnRNNForwardTraining(CudaContext::cudnnHandle(),
												mLSTMDesc,
												mSeqLength,
												xDesc->getdescs(),
												input0->getDevicePtr(),
												mhx->getCudnnTensorDesc(),
												mhx->getDevicePtr(),
												mcx->getCudnnTensorDesc(),
												mcx->getDevicePtr(),
												wDesc,
												mWeights->getDevicePtr(),
												yDesc->getdescs(),
												mOutputsLocal.getDevicePtr(),
												mhy.getCudnnTensorDesc(),
												mhy.getDevicePtr(),
												mcy.getCudnnTensorDesc(),
												mcy.getDevicePtr(),
												mWorkspace,
												mWorkSize,
												mReserveSpace,
												mReserveSize));

	}

	if (mSingleBackpropFeeding){
		mOutputsLocal.synchronizeDToH();

		for (unsigned int z=0; z<mBatchSize;++z){
			for (unsigned int y=0; y<mHiddenSize*biDirScale;++y){
				mOutputs(0,0,y,z)=mOutputsLocal(0,y,z,(mSeqLength-1));
			}
		}

		mOutputs.synchronizeHToD();
	}else {

		mOutputsLocal.synchronizeDToH();

		for (unsigned int s=0 ; s<mSeqLength; ++s){
			for (unsigned int z=0; z<mBatchSize;++z){
				for (unsigned int y=0; y<mHiddenSize*biDirScale;++y){
					mOutputs(0,y,z,s)=mOutputsLocal(0,y,z,s);
				}
			}
		}
		mOutputs.synchronizeHToD();

		//mOutputs.setDevicePtr(mOutputsLocal.getDevicePtr());
	}
}
template <class T>
void N2D2::LSTMCell_Frame_CUDA<T>::backPropagate(){

	if (mSingleBackpropFeeding){


		mDiffInputs.synchronizeDToH();

		for (unsigned int b=0; b<mDiffInputsLocal.dimB();++b){
			for (unsigned int z=0; z<mDiffInputsLocal.dimZ();++z){
				for (unsigned int y=0; y<mDiffInputsLocal.dimY();++y){
					for (unsigned int x=0; x<mDiffInputsLocal.dimX();++x){
						mDiffInputsLocal(x,y,z,b) = 0.;
					}
				}
			}
		}

		for (unsigned int z=0; z<mBatchSize;++z){
			for (unsigned int y=0; y<mHiddenSize*biDirScale;++y){
				mDiffInputsLocal(0,y,z,(mSeqLength-1))=mDiffInputs(0,0,y,z);
				mDiffInputs(0,0,y,z)=T(0.);
			}
		}
		mDiffInputs.synchronizeHToD();

		mDiffInputsLocal.synchronizeHToD();

	}else {
		mDiffInputs.synchronizeDToH();
		for (unsigned int s=0 ; s<mSeqLength; ++s){
			for (unsigned int z=0; z<mBatchSize;++z){
				for (unsigned int y=0; y<mHiddenSize*biDirScale;++y){
					mDiffInputsLocal(0,y,z,s)=mDiffInputs(0,y,z,s);
				}
			}
		}
		mDiffInputsLocal.synchronizeHToD();

		//mDiffInputsLocal.setDevicePtr(mDiffInputs.getDevicePtr());
	}


	std::shared_ptr<CudaDeviceTensor<T> > input0 = cuda_device_tensor_cast_nocopy<T>(mInputs[0]);
    std::shared_ptr<CudaDeviceTensor<T> > diffOutput0
        = (mDiffOutputs[0].isValid())
            ? cuda_device_tensor_cast<T>(mDiffOutputs[0])
            : cuda_device_tensor_cast_nocopy<T>(mDiffOutputs[0]);

	mDiffWeights.resize({1, 1, 1,static_cast<unsigned int>(mWeightsSize/sizeof(CudaContext::data_type<T>::value))},T(0.));

		CHECK_CUDNN_STATUS(cudnnRNNBackwardData(CudaContext::cudnnHandle(),
												mLSTMDesc,
												mSeqLength,
												yDesc->getdescs(),
												mOutputsLocal.getDevicePtr(),
												dyDesc->getdescs(),
												mDiffInputsLocal.getDevicePtr(),
												mDiffhy->getCudnnTensorDesc(),
												mDiffhy->getDevicePtr(),
												mDiffcy->getCudnnTensorDesc(),
												mDiffcy->getDevicePtr(),
												wDesc,
												mWeights->getDevicePtr(),
												mhx->getCudnnTensorDesc(),
												mhx->getDevicePtr(),
												mcx->getCudnnTensorDesc(),
												mcx->getDevicePtr(),
												dxDesc->getdescs(),
												diffOutput0->getDevicePtr(),
												mDiffhx.getCudnnTensorDesc(),
												mDiffhx.getDevicePtr(),
												mDiffcx.getCudnnTensorDesc(),
												mDiffcx.getDevicePtr(),
												mWorkspace,
												mWorkSize,
												mReserveSpace,
												mReserveSize));


		CHECK_CUDNN_STATUS(cudnnRNNBackwardWeights(CudaContext::cudnnHandle(),
												mLSTMDesc,
												mSeqLength,
												xDesc->getdescs(),
												input0->getDevicePtr(),
												mhx->getCudnnTensorDesc(),
												mhx->getDevicePtr(),
												yDesc->getdescs(),
												mOutputsLocal.getDevicePtr(),
												mWorkspace,
												mWorkSize,
												dwDesc,
												mDiffWeights.getDevicePtr(),
												mReserveSpace,
												mReserveSize));

	if(!mDiffOutputs.empty())
		mDiffOutputs.synchronizeDToHBased();
	mDiffhx.synchronizeDToHBased();
	mDiffcx.synchronizeDToHBased();

}

template <class T>
void N2D2::LSTMCell_Frame_CUDA<T>::update(){


	mWeightsSolver->update(*mWeights, mDiffWeights, mBatchSize);

	if (mContinousBatch){
		mhy.synchronizeDToH();
		mcy.synchronizeDToH();
		for (unsigned int m = 0; m < mBatchSize; ++m) {
			for (unsigned int j = 0; j < mNumberLayers * (mBidirectional? 2 : 1); ++j) {
				for (unsigned int i = 0; i < mHiddenSize; ++i) {
					(*mhx)(0,i,m,j) = mhy(0,i,m,j);
					(*mcx)(0,i,m,j) = mcy(0,i,m,j);
				}
			}
		}
		mhx->synchronizeHToD();
		mcx->synchronizeHToD();
	}
    Cell_Frame_CUDA<T>::update();
}

template <class T>
void N2D2::LSTMCell_Frame_CUDA<T>::checksum(std::vector<double> &values, bool display ) {


	Tensor<T> diffoutput0 = tensor_cast_nocopy<T>(mDiffOutputs[0]);

		mOutputs.synchronizeDToH();
		if (mSingleBackpropFeeding){
			double checksumi= 0.;
			for (unsigned int m = 0; m < mBatchSize; ++m) {
				if (display)
					fprintf(stdout,"batch : %u\n",m);
				double localSumi = 0;
				for (unsigned int i = 0; i < mHiddenSize * biDirScale; ++i) {
					localSumi += mOutputs(0,i,m,0);
					if (display)
						std::cout << " mOutputs_Frame_Cuda(0,"<< i << ", " << m << ", 0)= " <<  mOutputs(0,i,m,0)<< std::endl;
				}
			if (display)
				fprintf(stdout,"\n");
			checksumi += localSumi;
			}
		fprintf(stdout,"-------------------------------------------------------------------------------------------------------\n");
		fprintf(stdout,"checksum Y : %E      \n", checksumi);
		fprintf(stdout,"-------------------------------------------------------------------------------------------------------\n");
		}else {
			double checksumi= 0.;
			for (unsigned int m = 0; m < mBatchSize; ++m) {
				if (display)
					fprintf(stdout,"batch : %u\n",m);
				double localSumi = 0;
				for (unsigned int j = 0; j < mSeqLength; ++j) {
					for (unsigned int i = 0; i < mHiddenSize * biDirScale; ++i) {
						localSumi += mOutputs(0,i,m,j);
						if (display)
							std::cout << " mOutputs_Frame_Cuda(0,"<< i <<", " << m <<", "<< j <<")= " <<   mOutputs(0,i,m,j)<< std::endl;
					}
				}
			if (display)
				fprintf(stdout,"\n");
			checksumi += localSumi;
			}
		fprintf(stdout,"-------------------------------------------------------------------------------------------------------\n");
		fprintf(stdout,"checksum Y : %E      \n", checksumi);
		fprintf(stdout,"-------------------------------------------------------------------------------------------------------\n");
		}

		mOutputsLocal.synchronizeDToH();
		mhy.synchronizeDToH();
		mcy.synchronizeDToH();

		double checksumi = 0.f;
		double checksumh = 0.f;
		double checksumc = 0.f;
		for (unsigned int m = 0; m < mBatchSize; ++m) {
			if (display)
				fprintf(stdout,"batch : %u\n",m);
			double localSumi = 0;
			double localSumh = 0;
			double localSumc = 0;
			for (unsigned int j = 0; j < mSeqLength; ++j) {
				for (unsigned int i = 0; i < mHiddenSize * biDirScale; ++i) {
					localSumi += mOutputsLocal(0,i,m,j);
					if (display)
						std::cout << " mOutputsLocal_Frame_Cuda(0,"<< i <<", " << m <<", "<< j <<")= " <<  mOutputsLocal(0,i,m,j) << std::endl;
				}
			}
			if (display)
				fprintf(stdout,"\n");
			for (unsigned int j = 0; j < mNumberLayers * biDirScale; ++j) {
				for (unsigned int i = 0; i < mHiddenSize; ++i) {
					localSumh += mhy(0,i,m,j);
					if (display)
						std::cout << " mhy_Frame_Cuda(0,"<< i <<", " << m <<", "<< j <<")= " <<  mhy(0,i,m,j) << std::endl;
					localSumc += mcy(0,i,m,j);
					if (display)
						std::cout << " mcy_Frame_Cuda(0,"<< i <<", " << m <<", "<< j <<")= "<<  mcy(0,i,m,j) << std::endl;
				}
			}
			if (display)
				fprintf(stdout,"\n");
			checksumi += localSumi;
			checksumh += localSumh;
			checksumc += localSumc;
		}
		if (display){
			fprintf(stdout,"-------------------------------------------------------------------------------------------------------\n");
			fprintf(stdout,"checksum Y : %E      Checksum hy : %E          Checksum cy : %E \n", checksumi,checksumh,checksumc);
			fprintf(stdout,"-------------------------------------------------------------------------------------------------------\n");
		}
		values.push_back((double)checksumi);
		values.push_back((double)checksumh);
		values.push_back((double)checksumc);




		mDiffOutputs.synchronizeDToH();
		mDiffhx.synchronizeDToH();
		mDiffcx.synchronizeDToH();
		Tensor<T> checkdiffoutput = tensor_cast_nocopy<T>(mDiffOutputs[0]);


		float checksumdi = 0.f;
		float checksumdh = 0.f;
		float checksumdc = 0.f;
		for (unsigned int m = 0; m < mBatchSize; ++m) {
			if (display)
				fprintf(stdout,"batch : %u\n",m);
			double localSumdi = 0.;
			double localSumdh = 0.;
			double localSumdc = 0.;
			for (unsigned int j = 0; j < mSeqLength; ++j) {
				for (unsigned int i = 0; i < mInputDim; ++i) {
					localSumdi += checkdiffoutput(0,i,m,j);
					if (display)
						std::cout << "  mDiffOutputs_Frame_Cuda[0](0,"<< i <<", " << m <<", "<< j <<")= " <<  checkdiffoutput(0,i,m,j) << std::endl;
				}
			}
			if (display)
				fprintf(stdout,"\n");
			for (unsigned int j = 0; j < mNumberLayers * biDirScale; ++j) {
				for (unsigned int i = 0; i < mHiddenSize; ++i) {
					localSumdh += mDiffhx(0,i,m,j);
					if (display)
						std::cout << " mDiffhx_Frame_Cuda(0,"<< i <<", " << m <<", "<< j <<")= " <<  mDiffhx(0,i,m,j) << std::endl;
					localSumdc += mDiffcx(0,i,m,j);
					if (display)
						std::cout << " mDiffcx_Frame_Cuda(0,"<< i <<", " << m <<", "<< j <<")=  " <<  mDiffcx(0,i,m,j) << std::endl;
				}
			}
			if (display)
				fprintf(stdout,"\n");
			checksumdi += localSumdi;
			checksumdh += localSumdh;
			checksumdc += localSumdc;

		}
		if (display){
			fprintf(stdout,"-------------------------------------------------------------------------------------------------------\n");
			fprintf(stdout,"checksum dY : %E      Checksum dhy : %E          Checksum dcy : %E \n", checksumdi,checksumdh,checksumdc);
			fprintf(stdout,"-------------------------------------------------------------------------------------------------------\n");
		}
		values.push_back((double)checksumdi);
		values.push_back((double)checksumdh);
		values.push_back((double)checksumdc);





		mDiffWeights.synchronizeDToH();

		double checksumdw = 0.;


		for (std::size_t i = 0; i < mWeightsSize/sizeof(float); ++i) {
			checksumdw += mDiffWeights(0,0,0,i);
			if (display)
				std::cout << " mDiffWeights_Frame_Cuda(0,0,0,"<< i <<")= "<<  mDiffWeights(0,0,0,i) << std::endl;
		}
		values.push_back((double)checksumdw);



}

template <class T>
void N2D2::LSTMCell_Frame_CUDA<T>::compareChecksum(double value, double refValue, double maxError,std::string varName){
	double error;
	error = abs(value-refValue);
			if (error>maxError) {
							std::cout << "Checksum error for \"" << varName <<  " in \""<< mName << " d'hyperparamètres \n \""
									<< "<seqLength> <numLayers> <inputSize> <hiddenSize> <batchSize> <dropout> <bidirectional>\n \""
									<< " @ (" << mSeqLength << ", " << mNumberLayers << ", " << mInputDim
									<< ", " << mHiddenSize << ", " << mBatchSize  << ", " << mDropout << ", " << mBidirectional <<  ")\n"
									<< "  Computed = " << value
									<< "\n"
										"  Reference = " << refValue
									<< "\n"
										"  abs(Error) = " << error
									<< std::endl;
							throw std::runtime_error("Checksum failed!");
			} else {
				std::cout << "Checksum passed for \"" << varName <<  " in \""<< mName << " d'hyperparamètres \n \""
				<< "<seqLength> <numLayers> <inputSize> <hiddenSize> <batchSize> <modeCudnn> <dropout> <bidirectional>\n \""
				<< " @ (" << mSeqLength << ", " << mNumberLayers << ", " << mInputDim
				<< ", " << mHiddenSize << ", " << mBatchSize << ", " <<  mDropout << ", " << mBidirectional <<  ")\n"
				<< "  with abs error = " << error
				<< std::endl;
			}
}

template <class T>
void N2D2::LSTMCell_Frame_CUDA<T>::checkGradient(double epsilon, double maxError)
{
    GradientCheck<T> gc(epsilon, maxError);
	Random::mtSeed(21);
	for (unsigned int index = 0; index < mhx->size(); ++index){
        (*mhx)(index) = Random::randUniform(-1.0, 1.0);
	}

	for (unsigned int index = 0; index < mcx->size(); ++index){
        (*mcx)(index) = Random::randUniform(-1.0, 1.0);
	}


    gc.initialize(mInputs,
                  mOutputs,
                  mDiffInputs,
                  std::bind(&LSTMCell_Frame_CUDA::propagate, this, false),
                  std::bind(&LSTMCell_Frame_CUDA::backPropagate, this));


    if (!mDiffOutputs.empty()) {
        for (unsigned int k = 0; k < mInputs.size(); ++k) {
            std::stringstream name;
            name << mName + "Frame_CUDA_mDiffOutputs[" << k << "]";

            gc.check(name.str(), mInputs[k], mDiffOutputs[k]);
        }
    } else {
        std::cout << Utils::cwarning << "Empty diff. outputs for cell " << mName
                  << ", could not check the gradient!" << Utils::cdef
                  << std::endl;
    }
	/*if (!mDiffhx.empty()) {
		std::stringstream name;
		name << mName + "_mDiffhx";
		gc.check(name.str(), (*mhx), mDiffhx);
	}else {
        std::cout << Utils::cwarning << "Empty diff.hx for cell " << mName
                  << ", could not check the gradient!" << Utils::cdef
                  << std::endl;
	}
	if (!mDiffcx.empty()) {
		std::stringstream name;
		name << mName + "_mDiffcx";
		gc.check(name.str(), (*mcx), mDiffcx);
	}else {
        std::cout << Utils::cwarning << "Empty diff.cx for cell " << mName
                  << ", could not check the gradient!" << Utils::cdef
                  << std::endl;
	}*/

	/*
	GradientCheck gcdw(5.0e-4,1.0e-2);
	gcdw.initialize(mInputs,
                  mOutputs,
                  mDiffInputs,
                  std::bind(&LSTMCell_Frame_CUDA::propagate, this, false),
                  std::bind(&LSTMCell_Frame_CUDA::backPropagate, this));
 	if (!mDiffWeights.empty()) {
		std::stringstream name;
		name << mName + "_mDiffWeights";
		gcdw.check(name.str(), (*mWeights), mDiffWeights);
	}else {
        std::cout << Utils::cwarning << "Empty diff. Weights for cell " << mName
                  << ", could not check the gradient!" << Utils::cdef
                  << std::endl;
    }*/
}



/* Surcharge de la fonction addInput pour le cas LSTM afin de mapper les éventuels hx et dhx sur sa cellule parente (hy et dhy) en plus de l'appel classique qui mappe
**/
template <class T>
void N2D2::LSTMCell_Frame_CUDA<T>::addInput(N2D2::Cell* cell, const Tensor<bool>& mapping){

	N2D2::Cell_Frame_CUDA<T>::addInput(cell,mapping);

	LSTMCell_Frame_CUDA<T>* cellLSTM = dynamic_cast<LSTMCell_Frame_CUDA<T>*>(cell);


	if (cellLSTM != NULL) {


		if (cellLSTM->getHiddenSize() == mHiddenSize && cellLSTM->getBatchSize() == mBatchSize && cellLSTM->getNumberLayers() == mNumberLayers && cellLSTM->getBidirectional() == mBidirectional){

			std::shared_ptr<CudaTensor<T> > cudamhx = std::dynamic_pointer_cast<CudaTensor<T> >(cellLSTM->getmhx());

    		if (!cudamhx) {
        		throw std::runtime_error("LSTMCell_Frame_CUDA<T>::setaddInput(): mhx"
                                 " must be a CudaTensor");
    		}

    		mhx = cudamhx;

			std::shared_ptr<CudaTensor<T> > cudamDiffhy = std::dynamic_pointer_cast<CudaTensor<T> >(cellLSTM->getmDiffhy());

    		if (!cudamDiffhy) {
        		throw std::runtime_error("LSTMCell_Frame_CUDA<T>::setaddInput(): mDiffhy"
                                 " must be a CudaTensor");
    		}

    		mDiffhy = cudamDiffhy;

			std::shared_ptr<CudaTensor<T> > cudamcx = std::dynamic_pointer_cast<CudaTensor<T> >(cellLSTM->getmcx());

    		if (!cudamcx) {
        		throw std::runtime_error("LSTMCell_Frame_CUDA<T>::setaddInput(): mcx"
                                 " must be a CudaTensor");
    		}

    		mcx = cudamcx;

			std::shared_ptr<CudaTensor<T> > cudamDiffcy = std::dynamic_pointer_cast<CudaTensor<T> >(cellLSTM->getmDiffcy());

    		if (!cudamDiffcy) {
        		throw std::runtime_error("LSTMCell_Frame_CUDA<T>::setaddInput(): mDiffcy"
                                 " must be a CudaTensor");
    		}

    		mDiffcy = cudamDiffcy;
		}
	}
}

template <class T>
void N2D2::LSTMCell_Frame_CUDA<T>::addInput( 	StimuliProvider& sp,
											unsigned int x0,
                            				unsigned int y0,
											unsigned int width,
											unsigned int height,
											const Tensor<bool>& mapping){

	N2D2::Cell_Frame_CUDA<T>::addInput(sp, x0, y0, width, height, mapping);

	if(mSingleBackpropFeeding){
		mOutputs.resize({1,1,mHiddenSize*biDirScale,mBatchSize});
		mDiffInputs.resize({1,1,mHiddenSize*biDirScale,mBatchSize});
	}
}

template <class T>
void N2D2::LSTMCell_Frame_CUDA<T>::setWeights(
    const std::shared_ptr<Tensor<T> >& weights)
{
    std::shared_ptr<CudaTensor<T> > cudaWeights
        = std::dynamic_pointer_cast<CudaTensor<T> >(weights);

    if (!cudaWeights) {
        throw std::runtime_error("LSTMCell_Frame_CUDA::setWeights(): weights"
                                 " must be a CudaTensor");
    }

    mWeights = cudaWeights;
}

template <class T>
void N2D2::LSTMCell_Frame_CUDA<T>::exportFreeParameters(const std::string
                                                          & fileName) const
{
	std::cout << "---------------------------------------------> Export Weights  in " << fileName << "<------------------------------------" << std::endl;
	synchronizeToH(false);
    LSTMCell::exportFreeParameters(fileName);
	keepInSync(true);

}
/*void N2D2::LSTMCell_Frame_CUDA::exportFreeParameters(const std::string
                                                          & fileName) const
{
	synchronizeToH(false);
	std::cout << "---------------------------------------------> Export Weights  in " << fileName << "<------------------------------------" << std::endl;
	long long int nbWeights;
    nbWeights = 4*(mBidirectional? 2 : 1)* (  mInputDim*mHiddenSize+mHiddenSize*mHiddenSize+2*mHiddenSize +
                                           (mNumberLayers-1)*( mHiddenSize*(mHiddenSize*(mBidirectional? 2 : 1))
                                                               +mHiddenSize*mHiddenSize+2*mHiddenSize ) );

	//std::cout << " ---------------------->le nombre de poids +biais nbWeights = " << nbWeights << std::endl;

    const std::string fileBase = Utils::fileBaseName(fileName);
    std::string fileExt = Utils::fileExtension(fileName);

    if (!fileExt.empty())
        fileExt = "." + fileExt;

    const std::string weightsFile = fileBase + "_Weights" + fileExt;

    std::ofstream weights(weightsFile.c_str());

    if (!weights.good())
        throw std::runtime_error("Could not create parameter file: "
                                 + weightsFile);


    for (unsigned int output = 0; output < (mWeightsSize/sizeof(CudaContext::data_type)); ++output) {
                weights << getWeight(output) << "\n";
    }

	keepInSync(true);
}*/

template <class T>
void N2D2::LSTMCell_Frame_CUDA<T>::importFreeParameters(const std::string
                                                          & fileName,
                                                          bool ignoreNotExists)
{
	keepInSync(false);
    LSTMCell::importFreeParameters(fileName, ignoreNotExists);
	synchronizeToD(true);
}

/*void N2D2::LSTMCell_Frame_CUDA::importFreeParameters(const std::string
                                                          & fileName,
                                                          bool ignoreNotExists)
{
	keepInSync(false);
    std::cout <<"-----------------------------> import Weights from -->" << fileName << std::endl;
	const std::string fileBase = Utils::fileBaseName(fileName);
    std::string fileExt = Utils::fileExtension(fileName);

    if (!fileExt.empty())
        fileExt = "." + fileExt;

    const bool singleFile = (std::ifstream(fileName.c_str()).good());
    const std::string weightsFile = (singleFile) ? fileName
        : fileBase + "_Weights" + fileExt;


    std::ifstream weights(weightsFile.c_str());

    if (!weights.good()) {
        if (ignoreNotExists) {
            std::cout << Utils::cnotice
                      << "Notice: Could not open synaptic file: " << weightsFile
                      << Utils::cdef << std::endl;
            return;
        } else
            throw std::runtime_error("Could not open synaptic file: "
                                     + weightsFile);
    }

    double w;
    long long int nbWeights;
    nbWeights = 4* (mBidirectional? 2 : 1)* (  mInputDim*mHiddenSize+mHiddenSize*mHiddenSize+2*mHiddenSize +
                                            (mNumberLayers-1)*( mHiddenSize*(mHiddenSize*(mBidirectional? 2 : 1))
                                                                +mHiddenSize*mHiddenSize+2*mHiddenSize ) );

    for (unsigned int output = 0; output < (mWeightsSize/sizeof(CudaContext::data_type)); ++output) {
        if (!(weights >> w))
        {
            if(!ignoreNotExists)
            throw std::runtime_error(
                    "Error while reading scale in parameter file: "
                    + weightsFile);
        } else {
            setWeight(output, w);
        }
    }
	std::cout <<"-----------------------------> Weights imported !!!!" << std::endl;

	synchronizeToD(true);
}*/

template <class T>
void N2D2::LSTMCell_Frame_CUDA<T>::logFreeParametersDistrib(const std::string
                                                         & fileName) const
{
	synchronizeToH(false);
    LSTMCell::logFreeParametersDistrib(fileName);
	keepInSync(true);
}

/*void N2D2::LSTMCell_Frame_CUDA::logFreeParametersDistrib(const std::string
                                                         & fileName) const
{

	synchronizeToH(false);

    // Append all weights
    std::vector<double> weights;
	long long int nbWeights;
    nbWeights = 4* (mBidirectional? 2 : 1)* (  mInputDim*mHiddenSize+mHiddenSize*mHiddenSize+2*mHiddenSize +
                                            (mNumberLayers-1)*( mHiddenSize*(mHiddenSize*(mBidirectional? 2 : 1))
                                                                +mHiddenSize*mHiddenSize+2*mHiddenSize ) );
    weights.reserve((mWeightsSize/sizeof(CudaContext::data_type)));

    for (unsigned int output = 0; output < (mWeightsSize/sizeof(CudaContext::data_type)); ++output) {
        weights.push_back(getWeight(output));
    }

    std::sort(weights.begin(), weights.end());

    // Write data file
    std::ofstream data(fileName.c_str());

    if (!data.good())
        throw std::runtime_error("Could not save weights distrib file.");

    std::copy(weights.begin(),
              weights.end(),
              std::ostream_iterator<double>(data, "\n"));
    data.close();

    const std::pair<double, double> meanStdDev = Utils::meanStdDev(weights);

    std::ostringstream label;
    label << "\"Average: " << meanStdDev.first << "\\n";
    label << "Std. dev.: " << meanStdDev.second << "\"";
    label << " at graph 0.7, graph 0.8 front";

    // Plot results
    Gnuplot gnuplot;
    gnuplot.set("grid front").set("key off");
    gnuplot << "binwidth=0.0078";   // < 1/128
    gnuplot << "bin(x,width)=width*floor(x/width+0.5)";
    gnuplot.set("boxwidth", "binwidth");
    gnuplot.set("style data boxes").set("style fill solid noborder");
    gnuplot.set("xtics", "0.2");
    gnuplot.set("mxtics", "2");
    gnuplot.set("grid", "mxtics");
    gnuplot.set("label", label.str());
    gnuplot.set("yrange", "[0:]");

    gnuplot.set("style rect fc lt -1 fs solid 0.15 noborder behind");
    gnuplot.set("obj rect from graph 0, graph 0 to -1, graph 1");
    gnuplot.set("obj rect from 1, graph 0 to graph 1, graph 1");

    const double minVal = (weights.front() < -1.0) ? weights.front() : -1.0;
    const double maxVal = (weights.back() > 1.0) ? weights.back() : 1.0;
    gnuplot.setXrange(minVal - 0.05, maxVal + 0.05);

    gnuplot.saveToFile(fileName);
    gnuplot.plot(fileName,
                 "using (bin($1,binwidth)):(1.0) smooth freq with boxes");

	keepInSync(true);
}*/

template <class T>
void N2D2::LSTMCell_Frame_CUDA<T>::setdytest(const std::shared_ptr<Filler>& filler){

	Random::mtSeed(50);
	setFillerTest(filler);
	mDiffInputs.resize({1,mHiddenSize * biDirScale, mBatchSize, mSeqLength});
	mFillerTest->apply(mDiffInputs);
	mDiffInputs.synchronizeHToD();

}

template <class T>
void N2D2::LSTMCell_Frame_CUDA<T>::setdhytest(const std::shared_ptr<Filler>& filler){

	setDiffhyFillerTest(filler);
	mDiffhy->resize({1, mHiddenSize, mBatchSize, mNumberLayers*biDirScale});
	mDiffhyFillerTest->apply((*mDiffhy));
	mDiffhy->synchronizeHToD();

}

template <class T>
void N2D2::LSTMCell_Frame_CUDA<T>::setdcytest(const std::shared_ptr<Filler>& filler){

	setDiffcyFillerTest(filler);
	mDiffcy->resize({1, mHiddenSize, mBatchSize, mNumberLayers*biDirScale});
	mDiffcyFillerTest->apply((*mDiffcy));
	mDiffcy->synchronizeHToD();

}

template <class T>
void N2D2::LSTMCell_Frame_CUDA<T>::synchronizeToH(bool keepInSync_) const
{
    mWeights->synchronizeDToH();
    keepInSync(keepInSync_);
}

template <class T>
void N2D2::LSTMCell_Frame_CUDA<T>::synchronizeToD(bool keepInSync_)
{
    mWeights->synchronizeHToD();
    keepInSync(keepInSync_);
}

template <class T>
N2D2::LSTMCell_Frame_CUDA<T>::~LSTMCell_Frame_CUDA(){

	/*if (mReserveSize > 0){
		cudaFree(mReserveSpace);
	}

    if (mWorkSize > 0){
		cudaFree(mWorkspace);
	}

	if (mStatesize > 0){
		cudaFree(mStates);
	}*/

	cudnnCreateFilterDescriptor(&wDesc);
	cudnnCreateFilterDescriptor(&dwDesc);

	cudnnDestroyDropoutDescriptor(mDropoutDesc);

	cudnnDestroyRNNDescriptor(mLSTMDesc);

}


namespace N2D2 {
    template class LSTMCell_Frame_CUDA<half_float::half>;
    template class LSTMCell_Frame_CUDA<float>;
    template class LSTMCell_Frame_CUDA<double>;
}
#endif
#endif
