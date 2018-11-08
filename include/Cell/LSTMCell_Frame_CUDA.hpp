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

#ifndef N2D2_LSTMCELL_FRAME_CUDA_H
#define N2D2_LSTMCELL_FRAME_CUDA_H

#include "Cell_Frame_CUDA.hpp"
#include "LSTMCell.hpp"
#include "CudaContext.hpp"
#include "CudaUtils.hpp"
#include "Solver/SGDSolver_Frame_CUDA.hpp"
#include "containers/CudaTensor.hpp"
#include "controler/RNNInOutType.hpp"
#include "Filler/ConstantFiller.hpp"



namespace N2D2 {
template <class T>
class LSTMCell_Frame_CUDA : public virtual LSTMCell, public Cell_Frame_CUDA<T> {
public:
using Cell_Frame_CUDA<T>::mInputs;
    using Cell_Frame_CUDA<T>::mOutputs;
    using Cell_Frame_CUDA<T>::mDiffInputs;
    using Cell_Frame_CUDA<T>::mDiffOutputs;
    using Cell_Frame_CUDA<T>::addInput;

	LSTMCell_Frame_CUDA(const std::string& name,
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
						bool singleBackpropFeeding);
    static std::shared_ptr<LSTMCell>
    create(Network& /*net*/,
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
    {
        return std::make_shared<LSTMCell_Frame_CUDA>(name,
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
													singleBackpropFeeding);
	}

	virtual ~LSTMCell_Frame_CUDA();

	virtual void initialize();

	virtual void propagate(bool inference);

	virtual void backPropagate();

	virtual void update();

	virtual void checksum(std::vector<double> &value, bool display = false);

	virtual void compareChecksum(double value, double refValue, double maxError, std::string varName);

	virtual void addInput(	Cell* cell,
							const Tensor<bool>& mapping = Tensor<bool>());

	virtual void addInput(  StimuliProvider& sp,
							unsigned int x0,
                            unsigned int y0,
                            unsigned int width,
                            unsigned int height,
                            const Tensor<bool>& mapping);

    void checkGradient(double /*epsilon*/ = 1.0e-4,
                       double /*maxError*/ = 1.0e-6);

	inline std::shared_ptr<CudaTensor<T> > getmhx(){
		return  mhx;
	};

	inline std::shared_ptr<CudaTensor<T> > getmDiffhy(){
		return mDiffhy;
	};

	inline std::shared_ptr<CudaTensor<T> > getmcx(){
		return  mcx;
	};

	inline std::shared_ptr<CudaTensor<T> > getmDiffcy(){
		return mDiffcy;
	};

	void setWeights(const std::shared_ptr<Tensor<T> >& weights);

	inline std::shared_ptr<Tensor<T> >  getWeights()
    {
        return mWeights;
    };

	/*void setWeight(unsigned int posWeight,
				T value)
	{
		(*mWeights)(0,0,0,posWeight) = value;

		if (!mSynchronized)
			mWeights->synchronizeHToD(0, 0, 0, posWeight, 1);
	};

	T getWeight(unsigned int posWeight) const
	{
		if (!mSynchronized){
			mWeights->synchronizeDToH(0, 0, 0, posWeight, 1);
		}
		return (*mWeights)(0, 0, 0, posWeight);
	};*/

	void exportFreeParameters(const std::string& fileName) const;
    void importFreeParameters(const std::string& fileName,
                              bool ignoreNotExists = false);
	/*void logFreeParameters(const std::string& fileName,
                            unsigned int output) const;*/
	void logFreeParametersDistrib(const std::string& fileName) const;

	inline void setBoolContinousBatch(bool val)
    {
        mContinousBatch = val;
    };




	void getWeightPLIG_1stLayer(unsigned int  inputidx,unsigned int  hiddenidx,unsigned int  bidir, BaseTensor& value) const {
		int pos;
		pos = getStartPosition(bidir,0,true);//	layer 0 ; InputGate = 0 ; weight => 1


		if (inputidx<mInputDim){
			pos += inputidx*mHiddenSize;
		}else {
			throw std::runtime_error("LSTMCell_FrameCuda::getWeightPLIG_1stLayer(): inputidx invalid");
		}

		if (hiddenidx<mHiddenSize){
			pos += hiddenidx;
		}else {
			throw std::runtime_error("LSTMCell_FrameCuda::getWeightPLIG_1stLayer(): hiddenidx invalid");
		}

		if (!mSynchronized)
			mWeights->synchronizeDToH(0, 0, 0, pos, 1);

		value.resize({1});
    	value = Tensor<T>({1}, (*mWeights)(0,0,0,pos));
	};
	void getWeightPLFG_1stLayer(unsigned int  inputidx,unsigned int  hiddenidx,unsigned int  bidir, BaseTensor& value) const {
		int pos;
		pos = getStartPosition(bidir,1,true);//	layer 0 ; ForgetGate = 1 ; weight => 1

		if (inputidx<mInputDim){
			pos += inputidx*mHiddenSize;
		}else {
			throw std::runtime_error("LSTMCell_FrameCuda::getWeightPLFG_1stLayer(): inputidx invalid");
		}

		if (hiddenidx<mHiddenSize){
			pos += hiddenidx;
		}else {
			throw std::runtime_error("LSTMCell_FrameCuda::getWeightPLFG_1stLayer(): hiddenidx invalid");
		}

		if (!mSynchronized)
			mWeights->synchronizeDToH(0, 0, 0, pos, 1);

		value.resize({1});
		value = Tensor<T>({1}, (*mWeights)(0,0,0,pos));
	};
	void getWeightPLCG_1stLayer(unsigned int  inputidx,unsigned int  hiddenidx,unsigned int  bidir, BaseTensor& value) const {
		int pos;
		pos = getStartPosition(bidir,2,true);//	layer 0 ; CellGate = 2 ; weight => 1

		if (inputidx<mInputDim){
			pos += inputidx*mHiddenSize;
		}else {
			throw std::runtime_error("LSTMCell_FrameCuda::getWeightPLCG_1stLayer(): inputidx invalid");
		}

		if (hiddenidx<mHiddenSize){
			pos += hiddenidx;
		}else {
			throw std::runtime_error("LSTMCell_FrameCuda::getWeightPLCG_1stLayer(): hiddenidx invalid");
		}

		if (!mSynchronized)
			mWeights->synchronizeDToH(0, 0, 0, pos, 1);

		value.resize({1});
		value = Tensor<T>({1}, (*mWeights)(0,0,0,pos));
	};
	void getWeightPLOG_1stLayer(unsigned int  inputidx,unsigned int  hiddenidx,unsigned int  bidir, BaseTensor& value) const {
		int pos;
		pos = getStartPosition(bidir,3,true);//	layer 0 ; OutputGate = 3 ; weight => 1

		if (inputidx<mInputDim){
			pos += inputidx*mHiddenSize;
		}else {
			throw std::runtime_error("LSTMCell_FrameCuda::getWeightPLOG_1stLayer(): inputidx invalid");
		}

		if (hiddenidx<mHiddenSize){
			pos += hiddenidx;
		}else {
			throw std::runtime_error("LSTMCell_FrameCuda::getWeightPLOG_1stLayer(): hiddenidx invalid");
		}

		if (!mSynchronized)
			mWeights->synchronizeDToH(0, 0, 0, pos, 1);

		value.resize({1});
		value = Tensor<T>({1}, (*mWeights)(0,0,0,pos));
	};




	void getWeightPLIG(unsigned int  channelhiddenidx,unsigned int  outputhiddenidx,unsigned int  nlbidir, BaseTensor& value) const {
		int pos;
		pos = getStartPosition(nlbidir+biDirScale,0,true);//	layer = nlbidir ; InputGate = 0 ; weight => 1

		if (channelhiddenidx<mHiddenSize*biDirScale){
			pos += channelhiddenidx*mHiddenSize;
		}else {
			throw std::runtime_error("LSTMCell_FrameCuda::getWeightPLIG(): channelhiddenidx invalid");
		}

		if (outputhiddenidx<mHiddenSize){
			pos += outputhiddenidx;
		}else {
			throw std::runtime_error("LSTMCell_FrameCuda::getWeightPLIG(): outputhiddenidx invalid");
		}

		if (!mSynchronized)
			mWeights->synchronizeDToH(0, 0, 0, pos, 1);

		value.resize({1});
		value = Tensor<T>({1}, (*mWeights)(0,0,0,pos));
	};
	void getWeightPLFG(unsigned int  channelhiddenidx,unsigned int  outputhiddenidx,unsigned int  nlbidir, BaseTensor& value) const {
		int pos;
		pos = getStartPosition(nlbidir+biDirScale,1,true);//	layer = nlbidir ; ForgetGate = 1 ; weight => 1

		if (channelhiddenidx<mHiddenSize*biDirScale){
			pos += channelhiddenidx*mHiddenSize;
		}else {
			throw std::runtime_error("LSTMCell_FrameCuda::getWeightPLFG(): channelhiddenidx invalid");
		}

		if (outputhiddenidx<mHiddenSize){
			pos += outputhiddenidx;
		}else {
			throw std::runtime_error("LSTMCell_FrameCuda::getWeightPLFG(): outputhiddenidx invalid");
		}

		if (!mSynchronized)
			mWeights->synchronizeDToH(0, 0, 0, pos, 1);

		value.resize({1});
		value = Tensor<T>({1}, (*mWeights)(0,0,0,pos));
	};
	void getWeightPLCG(unsigned int  channelhiddenidx,unsigned int  outputhiddenidx,unsigned int  nlbidir, BaseTensor& value) const {
		int pos;
		pos = getStartPosition(nlbidir+biDirScale,2,true);//	layer = nlbidir ; CellGate = 2 ; weight => 1

		if (channelhiddenidx<mHiddenSize*biDirScale){
			pos += channelhiddenidx*mHiddenSize;
		}else {
			throw std::runtime_error("LSTMCell_FrameCuda::getWeightPLCG(): channelhiddenidx invalid");
		}

		if (outputhiddenidx<mHiddenSize){
			pos += outputhiddenidx;
		}else {
			throw std::runtime_error("LSTMCell_FrameCuda::getWeightPLCG(): outputhiddenidx invalid");
		}

		if (!mSynchronized)
			mWeights->synchronizeDToH(0, 0, 0, pos, 1);

		value.resize({1});
		value = Tensor<T>({1}, (*mWeights)(0,0,0,pos));
	};
	void getWeightPLOG(unsigned int  channelhiddenidx,unsigned int  outputhiddenidx,unsigned int  nlbidir, BaseTensor& value) const {
		int pos;
		pos = getStartPosition(nlbidir+biDirScale,3,true);//	layer = nlbidir ; OutputGate = 3 ; weight => 1

		if (channelhiddenidx<mHiddenSize*biDirScale){
			pos += channelhiddenidx*mHiddenSize;
		}else {
			throw std::runtime_error("LSTMCell_FrameCuda::setWeightPLOG(): channelhiddenidx invalid");
		}

		if (outputhiddenidx<mHiddenSize){
			pos += outputhiddenidx;
		}else {
			throw std::runtime_error("LSTMCell_FrameCuda::setWeightPLOG(): outputhiddenidx invalid");
		}

		if (!mSynchronized)
			mWeights->synchronizeDToH(0, 0, 0, pos, 1);

		value.resize({1});
		value = Tensor<T>({1}, (*mWeights)(0,0,0,pos));
	};






	void getWeightRIG(unsigned int  channelhiddenidx,unsigned int  outputhiddenidx,unsigned int  nlbidir, BaseTensor& value) const {
		int pos;
		pos = getStartPosition(nlbidir,4,true);//	layer = nlbidir ; InputGate = 4 ; weight => true

		if (channelhiddenidx<mHiddenSize){
			pos += channelhiddenidx*mHiddenSize;
		}else {
			throw std::runtime_error("LSTMCell_FrameCuda::getWeightRIG(): channelhiddenidx invalid");
		}

		if (outputhiddenidx<mHiddenSize){
			pos += outputhiddenidx;
		}else {
			throw std::runtime_error("LSTMCell_FrameCuda::getWeightRIG(): outputhiddenidx invalid");
		}

		if (!mSynchronized)
			mWeights->synchronizeDToH(0, 0, 0, pos, 1);

		value.resize({1});
		value = Tensor<T>({1}, (*mWeights)(0,0,0,pos));
	};
	void getWeightRFG(unsigned int  channelhiddenidx,unsigned int  outputhiddenidx,unsigned int  nlbidir, BaseTensor& value) const {
		int pos;
		pos = getStartPosition(nlbidir,5,true);//	layer = nlbidir ; ForgetGate = 5 ; weight => true

		if (channelhiddenidx<mHiddenSize){
			pos += channelhiddenidx*mHiddenSize;
		}else {
			throw std::runtime_error("LSTMCell_FrameCuda::getWeightRFG(): channelhiddenidx invalid");
		}

		if (outputhiddenidx<mHiddenSize){
			pos += outputhiddenidx;
		}else {
			throw std::runtime_error("LSTMCell_FrameCuda::getWeightRFG(): outputhiddenidx invalid");
		}

		if (!mSynchronized)
			mWeights->synchronizeDToH(0, 0, 0, pos, 1);

		value.resize({1});
		value = Tensor<T>({1}, (*mWeights)(0,0,0,pos));
	};
	void getWeightRCG(unsigned int  channelhiddenidx,unsigned int  outputhiddenidx,unsigned int  nlbidir, BaseTensor& value) const {
		int pos;
		pos = getStartPosition(nlbidir,6,true);//	layer = nlbidir ; CellGate = 6 ; weight => true

		if (channelhiddenidx<mHiddenSize){
			pos += channelhiddenidx*mHiddenSize;
		}else {
			throw std::runtime_error("LSTMCell_FrameCuda::getWeightRCG(): channelhiddenidx invalid");
		}

		if (outputhiddenidx<mHiddenSize){
			pos += outputhiddenidx;
		}else {
			throw std::runtime_error("LSTMCell_FrameCuda::getWeightRCG(): outputhiddenidx invalid");
		}

		if (!mSynchronized)
			mWeights->synchronizeDToH(0, 0, 0, pos, 1);

		value.resize({1});
		value = Tensor<T>({1}, (*mWeights)(0,0,0,pos));
	};
	void getWeightROG(unsigned int  channelhiddenidx,unsigned int  outputhiddenidx,unsigned int  nlbidir, BaseTensor& value) const {
		int pos;
		pos = getStartPosition(nlbidir,7,true);//	layer = nlbidir ; OutputGate = 7 ; weight => true

		if (channelhiddenidx<mHiddenSize){
			pos += channelhiddenidx*mHiddenSize;
		}else {
			throw std::runtime_error("LSTMCell_FrameCuda::getWeightROG(): channelhiddenidx invalid");
		}

		if (outputhiddenidx<mHiddenSize){
			pos += outputhiddenidx;
		}else {
			throw std::runtime_error("LSTMCell_FrameCuda::getWeightROG(): outputhiddenidx invalid");
		}

		if (!mSynchronized)
			mWeights->synchronizeHToD(0, 0, 0, pos, 1);

		value.resize({1});
		value = Tensor<T>({1}, (*mWeights)(0,0,0,pos));
	};






	void getBiasPLIG(unsigned int  hiddenidx, unsigned int  nlbidir, BaseTensor& value) const {
		int pos;
		pos = getStartPosition(nlbidir,0,false);//	layer = nlbidir ; InputGate = 0 ; bias => false

		if (hiddenidx<mHiddenSize){
			pos += hiddenidx;
		}else {
			throw std::runtime_error("LSTMCell_FrameCuda::getBiasPLIG(): hiddenidx invalid");
		}

		if (!mSynchronized)
			mWeights->synchronizeDToH(0, 0, 0, pos, 1);

		value.resize({1});
		value = Tensor<T>({1}, (*mWeights)(0,0,0,pos));


	};
	void getBiasPLFG(unsigned int  hiddenidx, unsigned int  nlbidir, BaseTensor& value) const {
		int pos;
		pos = getStartPosition(nlbidir,1,false);//	layer = nlbidir ; ForgetGate = 1 ; bias => false

		if (hiddenidx<mHiddenSize){
			pos += hiddenidx;
		}else {
			throw std::runtime_error("LSTMCell_FrameCuda::getBiasPLFG(): hiddenidx invalid");
		}

		if (!mSynchronized)
			mWeights->synchronizeDToH(0, 0, 0, pos, 1);

		value.resize({1});
		value = Tensor<T>({1}, (*mWeights)(0,0,0,pos));
	};
	void getBiasPLCG(unsigned int  hiddenidx, unsigned int  nlbidir, BaseTensor& value) const {
		int pos;
		pos = getStartPosition(nlbidir,2,false);//	layer = nlbidir ; CellGate = 2 ; bias => false

		if (hiddenidx<mHiddenSize){
			pos += hiddenidx;
		}else {
			throw std::runtime_error("LSTMCell_FrameCuda::getBiasPLCG(): hiddenidx invalid");
		}

		if (!mSynchronized)
			mWeights->synchronizeDToH(0, 0, 0, pos, 1);

		value.resize({1});
		value = Tensor<T>({1}, (*mWeights)(0,0,0,pos));
	};
	void getBiasPLOG(unsigned int  hiddenidx, unsigned int  nlbidir, BaseTensor& value) const {
		int pos;
		pos = getStartPosition(nlbidir,3,false);//	layer = nlbidir ; OutputGate = 3 ; bias => false

		if (hiddenidx<mHiddenSize){
			pos += hiddenidx;
		}else {
			throw std::runtime_error("LSTMCell_FrameCuda::getBiasPLOG(): hiddenidx invalid");
		}

		if (!mSynchronized)
			mWeights->synchronizeDToH(0, 0, 0, pos, 1);

		value.resize({1});
		value = Tensor<T>({1}, (*mWeights)(0,0,0,pos));
	};





	void getBiasRIG(unsigned int  hiddenidx, unsigned int  nlbidir, BaseTensor& value) const {
		int pos;
		pos = getStartPosition(nlbidir,4,false);//	layer = nlbidir ; InputGate = 4 ; bias => false

		if (hiddenidx<mHiddenSize){
			pos += hiddenidx;
		}else {
			throw std::runtime_error("LSTMCell_FrameCuda::getBiasRIG(): hiddenidx invalid");
		}

		if (!mSynchronized)
			mWeights->synchronizeDToH(0, 0, 0, pos, 1);

		value.resize({1});
		value = Tensor<T>({1}, (*mWeights)(0,0,0,pos));
	};
	void getBiasRFG(unsigned int  hiddenidx, unsigned int  nlbidir, BaseTensor& value) const {
		int pos;
		pos = getStartPosition(nlbidir,5,false);//	layer = nlbidir ; ForgetGate = 5 ; bias => false

		if (hiddenidx<mHiddenSize){
			pos += hiddenidx;
		}else {
			throw std::runtime_error("LSTMCell_FrameCuda::getBiasRFG(): hiddenidx invalid");
		}

		if (!mSynchronized)
			mWeights->synchronizeDToH(0, 0, 0, pos, 1);

		value.resize({1});
		value = Tensor<T>({1}, (*mWeights)(0,0,0,pos));
	};
	void getBiasRCG(unsigned int  hiddenidx, unsigned int  nlbidir, BaseTensor& value) const {
		int pos;
		pos = getStartPosition(nlbidir,6,false);//	layer = nlbidir ; CellGate = 6 ; bias => false

		if (hiddenidx<mHiddenSize){
			pos += hiddenidx;
		}else {
			throw std::runtime_error("LSTMCell_FrameCuda::getBiasRCG(): hiddenidx invalid");
		}

		if (!mSynchronized)
			mWeights->synchronizeDToH(0, 0, 0, pos, 1);

		value.resize({1});
		value = Tensor<T>({1}, (*mWeights)(0,0,0,pos));
	};
	void getBiasROG(unsigned int  hiddenidx, unsigned int  nlbidir, BaseTensor& value) const {
		int pos;
		pos = getStartPosition(nlbidir,7,false);//	layer = nlbidir ; OutputGate = 7 ; bias => false

		if (hiddenidx<mHiddenSize){
			pos += hiddenidx;
		}else {
			throw std::runtime_error("LSTMCell_FrameCuda::getBiasROG(): hiddenidx invalid");
		}

		if (!mSynchronized)
			mWeights->synchronizeDToH(0, 0, 0, pos, 1);

		value.resize({1});
		value = Tensor<T>({1}, (*mWeights)(0,0,0,pos));
	};




	virtual void setdytest(const std::shared_ptr<Filler>& filler);
	virtual void setdhytest(const std::shared_ptr<Filler>& filler);
	virtual void setdcytest(const std::shared_ptr<Filler>& filler);

protected :

	inline int getStartPosition(unsigned int layer, unsigned int gate, bool weight) const {
		int layer0PLSize,layer0RSize,layer0Size,layerxPLSize,layerxRSize,layerxSize,allLayerSize;
		layer0PLSize= 4*mInputDim*mHiddenSize;
		layer0RSize= 4*mHiddenSize*mHiddenSize;
		layer0Size= layer0PLSize+layer0RSize;
		layerxPLSize= 4*mHiddenSize*biDirScale*mHiddenSize;
		layerxRSize= 4*mHiddenSize*mHiddenSize;
		layerxSize= layerxPLSize + layerxRSize;
		allLayerSize =(layer0Size + (mNumberLayers-1)*layerxSize);
		if (mBidirectional==0){
			if (weight){
				if (layer==0){
					if (gate<4){
						return (gate*mInputDim*mHiddenSize);
					}else if (gate>=4 && gate<8){
						return (layer0PLSize+(gate-4)*mHiddenSize*mHiddenSize);
					}else {
						throw std::runtime_error("LSTMCell_FrameCuda::getStartPosition(): gate invalid");
					}
				}
				if (gate<4){
					return (layer0Size+(layer-1)*layerxSize+gate*mHiddenSize*mHiddenSize);
				}else if (gate>=4 && gate<8){
					return (layer0Size+(layer-1)*layerxSize+layerxPLSize+(gate-4)*mHiddenSize*mHiddenSize);
				}else {
					throw std::runtime_error("LSTMCell_FrameCuda::getStartPosition(): gate invalid");
				}

			}else{//bias
				return (allLayerSize+layer*8*mHiddenSize+gate*mHiddenSize);
			}
		}else if (mBidirectional==1){
			if (weight){
				if (layer==0 || layer==1){
					if (gate<4){
						return (layer*layer0Size+gate*mInputDim*mHiddenSize);
					}else if (gate>=4 && gate<8){
						return (layer*layer0Size+layer0PLSize+(gate-4)*mHiddenSize*mHiddenSize);
					}else {
						throw std::runtime_error("LSTMCell_FrameCuda::getStartPosition(): gate invalid");
					}
				}
				if (gate<4){
					return (2*layer0Size+(layer-2)*layerxSize+gate*mHiddenSize*biDirScale*mHiddenSize);
				}else if (gate>=4 && gate<8){
					return (2*layer0Size+(layer-2)*layerxSize+layerxPLSize+(gate-4)*mHiddenSize*mHiddenSize);
				}else {
					throw std::runtime_error("LSTMCell_FrameCuda::getStartPosition(): gate invalid");
				}

			}else{ //bias
				return (2*allLayerSize+layer*8*mHiddenSize+gate*mHiddenSize);
			}
		}else {
			throw std::runtime_error("LSTMCell_FrameCuda::getStartPosition(): mBidirectional invalid");
		}
	};

	//---------------Setters for Previous layers weights for layer=0-----------------//

	inline void setWeightPLIG_1stLayer(unsigned int  inputidx,unsigned int  hiddenidx,unsigned int  bidir, BaseTensor& value){

		int pos;
		pos = getStartPosition(bidir,0,true);//	layer 0 ou 1 ; InputGate = 0 ; weight => 1

		if (inputidx<mInputDim){
			pos += inputidx*mHiddenSize;
		}else {
			throw std::runtime_error("LSTMCell_FrameCuda::setWeightPLIG_1stLayer(): inputidx invalid");
		}

		if (hiddenidx<mHiddenSize){
			pos += hiddenidx;
		}else {
			throw std::runtime_error("LSTMCell_FrameCuda::setWeightPLIG_1stLayer(): hiddenidx invalid");
		}

		(*mWeights)(0,0,0,pos) = tensor_cast<T>(value)(0);

		if (!mSynchronized)
			mWeights->synchronizeHToD(0, 0, 0, pos, 1);
	};
	inline void setWeightPLFG_1stLayer(unsigned int  inputidx,unsigned int  hiddenidx,unsigned int  bidir, BaseTensor& value){
		int pos;
		pos = getStartPosition(bidir,1,true);//	layer 0 ; ForgetGate = 1 ; weight => 1

		if (inputidx<mInputDim){
			pos += inputidx*mHiddenSize;
		}else {
			throw std::runtime_error("LSTMCell_FrameCuda::setWeightPLFG_1stLayer(): inputidx invalid");
		}

		if (hiddenidx<mHiddenSize){
			pos += hiddenidx;
		}else {
			throw std::runtime_error("LSTMCell_FrameCuda::setWeightPLFG_1stLayer(): hiddenidx invalid");
		}

		(*mWeights)(0,0,0,pos) = tensor_cast<T>(value)(0);

		if (!mSynchronized)
			mWeights->synchronizeHToD(0, 0, 0, pos, 1);
	};

	inline void setWeightPLCG_1stLayer(unsigned int  inputidx,unsigned int  hiddenidx,unsigned int  bidir, BaseTensor& value){
		int pos;
		pos = getStartPosition(bidir,2,true);//	layer 0 ; CellGate = 2 ; weight => 1

		if (inputidx<mInputDim){
			pos += inputidx*mHiddenSize;
		}else {
			throw std::runtime_error("LSTMCell_FrameCuda::setWeightPLCG_1stLayer(): inputidx invalid");
		}

		if (hiddenidx<mHiddenSize){
			pos += hiddenidx;
		}else {
			throw std::runtime_error("LSTMCell_FrameCuda::setWeightPLCG_1stLayer(): hiddenidx invalid");
		}

		(*mWeights)(0,0,0,pos) = tensor_cast<T>(value)(0);

		if (!mSynchronized)
			mWeights->synchronizeHToD(0, 0, 0, pos, 1);
	};

	inline void setWeightPLOG_1stLayer(unsigned int  inputidx,unsigned int  hiddenidx,unsigned int  bidir, BaseTensor& value){
		int pos;
		pos = getStartPosition(bidir,3,true);//	layer 0 ; OutputGate = 3 ; weight => 1

		if (inputidx<mInputDim){
			pos += inputidx*mHiddenSize;
		}else {
			throw std::runtime_error("LSTMCell_FrameCuda::setWeightPLOG_1stLayer(): inputidx invalid");
		}

		if (hiddenidx<mHiddenSize){
			pos += hiddenidx;
		}else {
			throw std::runtime_error("LSTMCell_FrameCuda::setWeightPLOG_1stLayer(): hiddenidx invalid");
		}

		(*mWeights)(0,0,0,pos) = tensor_cast<T>(value)(0);

		if (!mSynchronized)
			mWeights->synchronizeHToD(0, 0, 0, pos, 1);
	};

	//---------------Setters for Previous layers weights for layer>0-----------------//

	inline void setWeightPLIG(unsigned int  channelhiddenidx,unsigned int  outputhiddenidx,unsigned int  nlbidir, BaseTensor& value){
		int pos;
		pos = getStartPosition(nlbidir+biDirScale,0,true);//	layer = nlbidir ; InputGate = 0 ; weight => 1

		if (channelhiddenidx<mHiddenSize*biDirScale){
			pos += channelhiddenidx*mHiddenSize;
		}else {
			throw std::runtime_error("LSTMCell_FrameCuda::setWeightPLIG(): channelhiddenidx invalid");
		}

		if (outputhiddenidx<mHiddenSize){
			pos += outputhiddenidx;
		}else {
			throw std::runtime_error("LSTMCell_FrameCuda::setWeightPLIG(): outputhiddenidx invalid");
		}

		(*mWeights)(0,0,0,pos) = tensor_cast<T>(value)(0);

		if (!mSynchronized)
			mWeights->synchronizeHToD(0, 0, 0, pos, 1);
	};
	inline void setWeightPLFG(unsigned int  channelhiddenidx,unsigned int  outputhiddenidx,unsigned int  nlbidir, BaseTensor& value){
		int pos;
		pos = getStartPosition(nlbidir+biDirScale,1,true);//	layer = nlbidir ; ForgetGate = 1 ; weight => 1

		if (channelhiddenidx<mHiddenSize*biDirScale){
			pos += channelhiddenidx*mHiddenSize;
		}else {
			throw std::runtime_error("LSTMCell_FrameCuda::setWeightPLFG(): channelhiddenidx invalid");
		}

		if (outputhiddenidx<mHiddenSize){
			pos += outputhiddenidx;
		}else {
			throw std::runtime_error("LSTMCell_FrameCuda::setWeightPLFG(): outputhiddenidx invalid");
		}

		(*mWeights)(0,0,0,pos) = tensor_cast<T>(value)(0);

		if (!mSynchronized)
			mWeights->synchronizeHToD(0, 0, 0, pos, 1);
	};
	inline void setWeightPLCG(unsigned int  channelhiddenidx,unsigned int  outputhiddenidx,unsigned int  nlbidir, BaseTensor& value){
		int pos;
		pos = getStartPosition(nlbidir+biDirScale,2,true);//	layer = nlbidir ; CellGate = 2 ; weight => 1

		if (channelhiddenidx<mHiddenSize*biDirScale){
			pos += channelhiddenidx*mHiddenSize;
		}else {
			throw std::runtime_error("LSTMCell_FrameCuda::setWeightPLCG(): channelhiddenidx invalid");
		}

		if (outputhiddenidx<mHiddenSize){
			pos += outputhiddenidx;
		}else {
			throw std::runtime_error("LSTMCell_FrameCuda::setWeightPLCG(): outputhiddenidx invalid");
		}

		(*mWeights)(0,0,0,pos) = tensor_cast<T>(value)(0);

		if (!mSynchronized)
			mWeights->synchronizeHToD(0, 0, 0, pos, 1);
	};
	inline void setWeightPLOG(unsigned int  channelhiddenidx,unsigned int  outputhiddenidx,unsigned int  nlbidir, BaseTensor& value){
		int pos;
		pos = getStartPosition(nlbidir+biDirScale,3,true);//	layer = nlbidir ; OutputGate = 3 ; weight => 1

		if (channelhiddenidx<mHiddenSize*biDirScale){
			pos += channelhiddenidx*mHiddenSize;
		}else {
			throw std::runtime_error("LSTMCell_FrameCuda::setWeightPLOG(): channelhiddenidx invalid");
		}

		if (outputhiddenidx<mHiddenSize){
			pos += outputhiddenidx;
		}else {
			throw std::runtime_error("LSTMCell_FrameCuda::setWeightPLOG(): outputhiddenidx invalid");
		}

		(*mWeights)(0,0,0,pos) = tensor_cast<T>(value)(0);

		if (!mSynchronized)
			mWeights->synchronizeHToD(0, 0, 0, pos, 1);
	};

	//---------------Setters for Reccurent weights -----------------//


	inline void setWeightRIG(unsigned int  channelhiddenidx,unsigned int  outputhiddenidx,unsigned int  nlbidir, BaseTensor& value){
		int pos;
		pos = getStartPosition(nlbidir,4,true);//	layer = nlbidir ; InputGate = 4 ; weight => true

		if (channelhiddenidx<mHiddenSize){
			pos += channelhiddenidx*mHiddenSize;
		}else {
			throw std::runtime_error("LSTMCell_FrameCuda::setWeightRIG(): channelhiddenidx invalid");
		}

		if (outputhiddenidx<mHiddenSize){
			pos += outputhiddenidx;
		}else {
			throw std::runtime_error("LSTMCell_FrameCuda::setWeightRIG(): outputhiddenidx invalid");
		}

		(*mWeights)(0,0,0,pos) = tensor_cast<T>(value)(0);

		if (!mSynchronized)
			mWeights->synchronizeHToD(0, 0, 0, pos, 1);
	};
	inline void setWeightRFG(unsigned int  channelhiddenidx,unsigned int  outputhiddenidx,unsigned int  nlbidir, BaseTensor& value){
		int pos;
		pos = getStartPosition(nlbidir,5,true);//	layer = nlbidir ; ForgetGate = 5 ; weight => true

		if (channelhiddenidx<mHiddenSize){
			pos += channelhiddenidx*mHiddenSize;
		}else {
			throw std::runtime_error("LSTMCell_FrameCuda::setWeightRFG(): channelhiddenidx invalid");
		}

		if (outputhiddenidx<mHiddenSize){
			pos += outputhiddenidx;
		}else {
			throw std::runtime_error("LSTMCell_FrameCuda::setWeightRFG(): outputhiddenidx invalid");
		}

		(*mWeights)(0,0,0,pos) = tensor_cast<T>(value)(0);

		if (!mSynchronized)
			mWeights->synchronizeHToD(0, 0, 0, pos, 1);
	};
	inline void setWeightRCG(unsigned int  channelhiddenidx,unsigned int  outputhiddenidx,unsigned int  nlbidir, BaseTensor& value){
		int pos;
		pos = getStartPosition(nlbidir,6,true);//	layer = nlbidir ; CellGate = 6 ; weight => true

		if (channelhiddenidx<mHiddenSize){
			pos += channelhiddenidx*mHiddenSize;
		}else {
			throw std::runtime_error("LSTMCell_FrameCuda::setWeightRCG(): channelhiddenidx invalid");
		}

		if (outputhiddenidx<mHiddenSize){
			pos += outputhiddenidx;
		}else {
			throw std::runtime_error("LSTMCell_FrameCuda::setWeightRCG(): outputhiddenidx invalid");
		}

		(*mWeights)(0,0,0,pos) = tensor_cast<T>(value)(0);

		if (!mSynchronized)
			mWeights->synchronizeHToD(0, 0, 0, pos, 1);
	};
	inline void setWeightROG(unsigned int  channelhiddenidx,unsigned int  outputhiddenidx,unsigned int  nlbidir, BaseTensor& value){
		int pos;
		pos = getStartPosition(nlbidir,7,true);//	layer = nlbidir ; OutputGate = 7 ; weight => true

		if (channelhiddenidx<mHiddenSize){
			pos += channelhiddenidx*mHiddenSize;
		}else {
			throw std::runtime_error("LSTMCell_FrameCuda::setWeightROG(): channelhiddenidx invalid");
		}

		if (outputhiddenidx<mHiddenSize){
			pos += outputhiddenidx;
		}else {
			throw std::runtime_error("LSTMCell_FrameCuda::setWeightROG(): outputhiddenidx invalid");
		}

		(*mWeights)(0,0,0,pos) = tensor_cast<T>(value)(0);

		if (!mSynchronized)
			mWeights->synchronizeHToD(0, 0, 0, pos, 1);
	};

	inline void setBiasPLIG(unsigned int  hiddenidx, unsigned int  nlbidir, BaseTensor& value){
		int pos;
		pos = getStartPosition(nlbidir,0,false);//	layer = nlbidir ; InputGate = 0 ; bias => false

		if (hiddenidx<mHiddenSize){
			pos += hiddenidx;
		}else {
			throw std::runtime_error("LSTMCell_FrameCuda::setBiasPLIG(): hiddenidx invalid");
		}

		(*mWeights)(0,0,0,pos) = tensor_cast<T>(value)(0);

		if (!mSynchronized)
			mWeights->synchronizeHToD(0, 0, 0, pos, 1);
	};
	inline void setBiasPLFG(unsigned int  hiddenidx, unsigned int  nlbidir, BaseTensor& value){
		int pos;
		pos = getStartPosition(nlbidir,1,false);//	layer = nlbidir ; ForgetGate = 1 ; bias => false

		if (hiddenidx<mHiddenSize){
			pos += hiddenidx;
		}else {
			throw std::runtime_error("LSTMCell_FrameCuda::setBiasPLFG(): hiddenidx invalid");
		}

		(*mWeights)(0,0,0,pos) = tensor_cast<T>(value)(0);

		if (!mSynchronized)
			mWeights->synchronizeHToD(0, 0, 0, pos, 1);
	};
	inline void setBiasPLCG(unsigned int  hiddenidx, unsigned int  nlbidir, BaseTensor& value){
		int pos;
		pos = getStartPosition(nlbidir,2,false);//	layer = nlbidir ; CellGate = 2 ; bias => false

		if (hiddenidx<mHiddenSize){
			pos += hiddenidx;
		}else {
			throw std::runtime_error("LSTMCell_FrameCuda::setBiasPLCG(): hiddenidx invalid");
		}

		(*mWeights)(0,0,0,pos) = tensor_cast<T>(value)(0);

		if (!mSynchronized)
			mWeights->synchronizeHToD(0, 0, 0, pos, 1);
	};
	inline void setBiasPLOG(unsigned int  hiddenidx, unsigned int  nlbidir, BaseTensor& value){
		int pos;
		pos = getStartPosition(nlbidir,3,false);//	layer = nlbidir ; OutputGate = 3 ; bias => false

		if (hiddenidx<mHiddenSize){
			pos += hiddenidx;
		}else {
			throw std::runtime_error("LSTMCell_FrameCuda::setBiasPLOG(): hiddenidx invalid");
		}

		(*mWeights)(0,0,0,pos) = tensor_cast<T>(value)(0);

		if (!mSynchronized)
			mWeights->synchronizeHToD(0, 0, 0, pos, 1);
	};

	inline void setBiasRIG(unsigned int  hiddenidx ,unsigned int  nlbidir, BaseTensor& value){
		int pos;
		pos = getStartPosition(nlbidir,4,false);//	layer = nlbidir ; InputGate = 4 ; bias => false

		if (hiddenidx<mHiddenSize){
			pos += hiddenidx;
		}else {
			throw std::runtime_error("LSTMCell_FrameCuda::setBiasRIG(): hiddenidx invalid");
		}

		(*mWeights)(0,0,0,pos) = tensor_cast<T>(value)(0);

		if (!mSynchronized)
			mWeights->synchronizeHToD(0, 0, 0, pos, 1);
	};
	inline void setBiasRFG(unsigned int  hiddenidx ,unsigned int  nlbidir, BaseTensor& value){
		int pos;
		pos = getStartPosition(nlbidir,5,false);//	layer = nlbidir ; ForgetGate = 5 ; bias => false

		if (hiddenidx<mHiddenSize){
			pos += hiddenidx;
		}else {
			throw std::runtime_error("LSTMCell_FrameCuda::setBiasRFG(): hiddenidx invalid");
		}

		(*mWeights)(0,0,0,pos) = tensor_cast<T>(value)(0);

		if (!mSynchronized)
			mWeights->synchronizeHToD(0, 0, 0, pos, 1);
	};
	inline void setBiasRCG(unsigned int  hiddenidx ,unsigned int  nlbidir, BaseTensor& value){
		int pos;
		pos = getStartPosition(nlbidir,6,false);//	layer = nlbidir ; CellGate = 6 ; bias => false

		if (hiddenidx<mHiddenSize){
			pos += hiddenidx;
		}else {
			throw std::runtime_error("LSTMCell_FrameCuda::setBiasRCG(): hiddenidx invalid");
		}

		(*mWeights)(0,0,0,pos) = tensor_cast<T>(value)(0);

		if (!mSynchronized)
			mWeights->synchronizeHToD(0, 0, 0, pos, 1);
	};
	inline void setBiasROG(unsigned int  hiddenidx ,unsigned int  nlbidir, BaseTensor& value){
		int pos;
		pos = getStartPosition(nlbidir,7,false);//	layer = nlbidir ; OutputGate = 7 ; bias => false

		if (hiddenidx<mHiddenSize){
			pos += hiddenidx;
		}else {
			throw std::runtime_error("LSTMCell_FrameCuda::setBiasROG(): hiddenidx invalid");
		}

		(*mWeights)(0,0,0,pos) = tensor_cast<T>(value)(0);

		if (!mSynchronized)
			mWeights->synchronizeHToD(0, 0, 0, pos, 1);
	};


	CudaTensor<T> mOutputsLocal;
	CudaTensor<T> mDiffInputsLocal;


	std::shared_ptr<CudaTensor<T> > mWeights;
	CudaTensor<T> mDiffWeights;

	std::shared_ptr<CudaTensor<T> > mhx;
	CudaTensor<T> mDiffhx;

	CudaTensor<T> mhy;
	std::shared_ptr<CudaTensor<T> > mDiffhy;

	std::shared_ptr<CudaTensor<T> > mcx;
	CudaTensor<T> mDiffcx;

	CudaTensor<T> mcy;
	std::shared_ptr<CudaTensor<T> > mDiffcy;

	 /*
		Structure for inference:
		 mcx && mhx -------->|    |--------> mcx && mhx (passé à la couche suivante uniquement si les dimmensions correspondent : hidden_size/numlayers*bidirectionel)
                             |    |
                             |    |
 mDiffhy && mDiffcy <--------|    |<-------- mDiffhy & mDiffcy
		(passé à la couche parente
		uniquement si les dimmensions
		correspondent : hidden_size et
		numlayers*bidirectionel)
	*/

	mutable bool mSynchronized;
	mutable bool mContinousBatch;

private :

	static Registrar<LSTMCell> mRegistrar;

	std::unique_ptr<RNNInOutType<T> > xDesc;
	std::unique_ptr<RNNInOutType<T> > yDesc;
	std::unique_ptr<RNNInOutType<T> > dxDesc;
	std::unique_ptr<RNNInOutType<T>> dyDesc;

	// ---------------
	// Declare Dropout
	// ---------------


	unsigned long long seed;
	void *mStates;
	size_t mStatesize;

	cudnnDropoutDescriptor_t mDropoutDesc;
#if CUDNN_VERSION >= 5000

	// -----------------------
	// Declare options for LSTM
	// -----------------------

	cudnnRNNInputMode_t mInput_Mode;
	cudnnRNNMode_t mMode;
	cudnnRNNDescriptor_t mLSTMDesc;
#endif

#if CUDNN_VERSION >= 6000
	cudnnRNNAlgo_t mCudnnAlgo;
#endif

	// ---------------
	// Declare Weights
	// ---------------

	size_t mWeightsSize;

	cudnnFilterDescriptor_t wDesc, dwDesc;

	// ------------------------------------
	// Declare Workspace and Training Space
	// ------------------------------------


	void *mWorkspace;
	void *mReserveSpace;

	size_t mWorkSize;
	size_t mReserveSize;

	unsigned int biDirScale = (mBidirectional ? 2 : 1);



};
}


#endif // N2D2_LSTMCELL_FRAME_CUDA_H
