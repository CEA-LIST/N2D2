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

#ifndef N2D2_LSTMCELL_H
#define N2D2_LSTMCELL_H

#include <cassert>
#include <memory>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "Cell.hpp"
#include "Filler/NormalFiller.hpp"
#include "Solver/Solver.hpp"
#include "utils/Registrar.hpp"
#include "controler/Interface.hpp"

#ifdef WIN32
// For static library
/*#pragma comment(                                                               \
    linker,                                                                    \
    "/include:?mRegistrar@LSTMCell_Frame@N2D2@@0U?$Registrar@VLSTMCell@N2D2@@@2@A")*/
#ifdef CUDA
#pragma comment(                                                               \
    linker,                                                                    \
    "/include:?mRegistrar@LSTMCell_Frame_CUDA@N2D2@@0U?$Registrar@VLSTMCell@N2D2@@@2@A")
#endif
#endif

namespace N2D2 {
class LSTMCell : public virtual Cell {
public:
	typedef std::function<std::shared_ptr<LSTMCell>(
		Network&,
		const std::string&,
		unsigned int,
		unsigned int,
		unsigned int,
		unsigned int,
		unsigned int,
		unsigned int,
		unsigned int,
        unsigned int,
		unsigned int,
		float,
		bool)> RegistryCreate_T;

	static RegistryMap_T& registry()
	{
		static RegistryMap_T rMap;
		return rMap;
	}
	static const char* Type;

	LSTMCell(const std::string& name,
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


/*---------------------------Getters&Setters--------------------------*/

	const char* getType() const
	{
		return Type;
	};
    void getStats(Stats& stats) const;

	std::shared_ptr<Solver> getWeightsSolver()
    {
        return mWeightsSolver;
    };


	unsigned int getSeqLength() const
	{
		return mSeqLength;
	};
	unsigned int getBatchSize() const
	{
		return mBatchSize;
	};
	unsigned int getInputDim() const
	{
		return mInputDim;
	};
	unsigned int getNumberLayers() const
	{
		return mNumberLayers;
	};
	unsigned int getHiddenSize() const
	{
		return mHiddenSize;
	};
	unsigned int getAlgo() const
	{
		return mAlgo;
	};
	unsigned int getBidirectional() const
	{
		return mBidirectional;
	};
	unsigned int getInputMode() const
	{
		return mInputMode;
	};
	float getDropout() const
	{
		return mDropout;
	};
	bool getSingleBackpropFeeding() const
	{
		return mSingleBackpropFeeding;
	};





	virtual void getWeightPLIG_1stLayer(unsigned int  inputidx,unsigned int  hiddenidx,unsigned int  bidir, BaseTensor& value) const = 0;
	virtual void getWeightPLFG_1stLayer(unsigned int  inputidx,unsigned int  hiddenidx,unsigned int  bidir, BaseTensor& value) const = 0;
	virtual void getWeightPLCG_1stLayer(unsigned int  inputidx,unsigned int  hiddenidx,unsigned int  bidir, BaseTensor& value) const = 0;
	virtual void getWeightPLOG_1stLayer(unsigned int  inputidx,unsigned int  hiddenidx,unsigned int  bidir, BaseTensor& value) const = 0;

	virtual void getWeightPLIG(unsigned int  channelhiddenidx,unsigned int  outputhiddenidx,unsigned int  nlbidir, BaseTensor& value) const = 0;
	virtual void getWeightPLFG(unsigned int  channelhiddenidx,unsigned int  outputhiddenidx,unsigned int  nlbidir, BaseTensor& value) const = 0;
	virtual void getWeightPLCG(unsigned int  channelhiddenidx,unsigned int  outputhiddenidx,unsigned int  nlbidir, BaseTensor& value) const = 0;
	virtual void getWeightPLOG(unsigned int  channelhiddenidx,unsigned int  outputhiddenidx,unsigned int  nlbidir, BaseTensor& value) const = 0;

	virtual void getWeightRIG(unsigned int  channelhiddenidx,unsigned int  outputhiddenidx,unsigned int  nlbidir, BaseTensor& value) const = 0;
	virtual void getWeightRFG(unsigned int  channelhiddenidx,unsigned int  outputhiddenidx,unsigned int  nlbidir, BaseTensor& value) const = 0;
	virtual void getWeightRCG(unsigned int  channelhiddenidx,unsigned int  outputhiddenidx,unsigned int  nlbidir, BaseTensor& value) const = 0;
	virtual void getWeightROG(unsigned int  channelhiddenidx,unsigned int  outputhiddenidx,unsigned int  nlbidir, BaseTensor& value) const = 0;

	virtual void getBiasPLIG(unsigned int  hiddenidx, unsigned int  nlbidir, BaseTensor& value) const = 0;
	virtual void getBiasPLFG(unsigned int  hiddenidx, unsigned int  nlbidir, BaseTensor& value) const = 0;
	virtual void getBiasPLCG(unsigned int  hiddenidx, unsigned int  nlbidir, BaseTensor& value) const = 0;
	virtual void getBiasPLOG(unsigned int  hiddenidx, unsigned int  nlbidir, BaseTensor& value) const = 0;

	virtual void getBiasRIG(unsigned int  hiddenidx, unsigned int  nlbidir, BaseTensor& value) const = 0;
	virtual void getBiasRFG(unsigned int  hiddenidx, unsigned int  nlbidir, BaseTensor& value) const = 0;
	virtual void getBiasRCG(unsigned int  hiddenidx, unsigned int  nlbidir, BaseTensor& value) const = 0;
	virtual void getBiasROG(unsigned int  hiddenidx, unsigned int  nlbidir, BaseTensor& value) const = 0;


	void setWeightsPreviousLayerAllGateFiller_1stLayer(const std::shared_ptr<Filler>& filler)
	{
		mWeightsPreviousLayerInputGateFiller_1stLayer = filler;
		mWeightsPreviousLayerForgetGateFiller_1stLayer = filler;
		mWeightsPreviousLayerCellGateFiller_1stLayer = filler;
		mWeightsPreviousLayerOutputGateFiller_1stLayer = filler;
	};

	void setWeightsPreviousLayerAllGateFiller(const std::shared_ptr<Filler>& filler)
	{
		mWeightsPreviousLayerInputGateFiller = filler;
		mWeightsPreviousLayerForgetGateFiller = filler;
		mWeightsPreviousLayerCellGateFiller = filler;
		mWeightsPreviousLayerOutputGateFiller = filler;
	};

	void setWeightsRecurrentAllGateFiller(const std::shared_ptr<Filler>& filler)
	{
		mWeightsRecurrentInputGateFiller = filler;
		mWeightsRecurrentForgetGateFiller = filler;
		mWeightsRecurrentCellGateFiller = filler;
		mWeightsRecurrentOutputGateFiller = filler;
	};

	void setBiasAllGateFiller(const std::shared_ptr<Filler>& filler)
	{
		mBiasPreviousLayerInputGateFiller = filler;
		mBiasPreviousLayerForgetGateFiller = filler;
		mBiasPreviousLayerCellGateFiller = filler;
		mBiasPreviousLayerOutputGateFiller = filler;
		mBiasRecurrentInputGateFiller = filler;
		mBiasRecurrentForgetGateFiller = filler;
		mBiasRecurrentCellGateFiller = filler;
		mBiasRecurrentOutputGateFiller = filler;
	};

	void setBiasRecurrentAllGateFiller(const std::shared_ptr<Filler>& filler)
	{
		mBiasRecurrentInputGateFiller = filler;
		mBiasRecurrentForgetGateFiller = filler;
		mBiasRecurrentCellGateFiller = filler;
		mBiasRecurrentOutputGateFiller = filler;
	};
	void setBiasPreviousLayerAllGateFiller(const std::shared_ptr<Filler>& filler)
	{
		mBiasPreviousLayerInputGateFiller = filler;
		mBiasPreviousLayerForgetGateFiller = filler;
		mBiasPreviousLayerCellGateFiller = filler;
		mBiasPreviousLayerOutputGateFiller = filler;
	};



	void setWeightsPreviousLayerInputGateFiller_1stLayer(const std::shared_ptr<Filler>& filler)
	{
		mWeightsPreviousLayerInputGateFiller_1stLayer = filler;
	};
	void setWeightsPreviousLayerInputGateFiller(const std::shared_ptr<Filler>& filler)
	{
		mWeightsPreviousLayerInputGateFiller = filler;
	};
	void setWeightsPreviousLayerForgetGateFiller_1stLayer(const std::shared_ptr<Filler>& filler)
	{
		mWeightsPreviousLayerForgetGateFiller_1stLayer = filler;
	};
	void setWeightsPreviousLayerForgetGateFiller(const std::shared_ptr<Filler>& filler)
	{
		mWeightsPreviousLayerForgetGateFiller = filler;
	};
	void setWeightsPreviousLayerCellGateFiller_1stLayer(const std::shared_ptr<Filler>& filler)
	{
		mWeightsPreviousLayerCellGateFiller_1stLayer = filler;
	};
	void setWeightsPreviousLayerCellGateFiller(const std::shared_ptr<Filler>& filler)
	{
		mWeightsPreviousLayerCellGateFiller = filler;
	};
	void setWeightsPreviousLayerOutputGateFiller_1stLayer(const std::shared_ptr<Filler>& filler)
	{
		mWeightsPreviousLayerOutputGateFiller_1stLayer = filler;
	};
	void setWeightsPreviousLayerOutputGateFiller(const std::shared_ptr<Filler>& filler)
	{
		mWeightsPreviousLayerOutputGateFiller = filler;
	};


	void setWeightsRecurrentInputGateFiller(const std::shared_ptr<Filler>& filler)
	{
		mWeightsRecurrentInputGateFiller = filler;
	};
	void setWeightsRecurrentForgetGateFiller(const std::shared_ptr<Filler>& filler)
	{
		mWeightsRecurrentForgetGateFiller = filler;
	};
	void setWeightsRecurrentCellGateFiller(const std::shared_ptr<Filler>& filler)
	{
		mWeightsRecurrentCellGateFiller = filler;
	};
	void setWeightsRecurrentOutputGateFiller(const std::shared_ptr<Filler>& filler)
	{
		mWeightsRecurrentOutputGateFiller = filler;
	};




	void setBiasPreviousLayerInputGateFiller(const std::shared_ptr<Filler>& filler)
	{
		mBiasPreviousLayerInputGateFiller = filler;
	};
	void setBiasPreviousLayerForgetGateFiller(const std::shared_ptr<Filler>& filler)
	{
		mBiasPreviousLayerForgetGateFiller = filler;
	};
	void setBiasPreviousLayerCellGateFiller(const std::shared_ptr<Filler>& filler)
	{
		mBiasPreviousLayerCellGateFiller = filler;
	};
	void setBiasPreviousLayerOutputGateFiller(const std::shared_ptr<Filler>& filler)
	{
		mBiasPreviousLayerOutputGateFiller = filler;
	};


	void setBiasRecurrentInputGateFiller(const std::shared_ptr<Filler>& filler)
	{
		mBiasRecurrentInputGateFiller = filler;
	};
	void setBiasRecurrentForgetGateFiller(const std::shared_ptr<Filler>& filler)
	{
		mBiasRecurrentForgetGateFiller = filler;
	};
	void setBiasRecurrentCellGateFiller(const std::shared_ptr<Filler>& filler)
	{
		mBiasRecurrentCellGateFiller = filler;
	};
	void setBiasRecurrentOutputGateFiller(const std::shared_ptr<Filler>& filler)
	{
		mBiasRecurrentOutputGateFiller = filler;
	};




	void setHxFiller(const std::shared_ptr<Filler>& filler)
	{
		mhxFiller = filler;
	};

	void setCxFiller(const std::shared_ptr<Filler>& filler)
	{
		mcxFiller = filler;
	};

	void setWeightsSolver(const std::shared_ptr<Solver>& solver)
    {
        mWeightsSolver = solver;
    };

	virtual void exportFreeParameters(const std::string& /*fileName*/) const;
    virtual void importFreeParameters(const std::string& /*fileName*/,bool /*ignoreNotExists = false*/);
	/*virtual void logFreeParameters(const std::string& fileName,
                                   unsigned int output) const;*/
	virtual void logFreeParametersDistrib(const std::string& /*fileName*/) const;



	void setFillerTest(const std::shared_ptr<Filler>& filler)
	{
		mFillerTest = filler;
	};

	void setDiffhyFillerTest(const std::shared_ptr<Filler>& filler)
	{
		mDiffhyFillerTest = filler;
	};

	void setDiffcyFillerTest(const std::shared_ptr<Filler>& filler)
	{
		mDiffcyFillerTest = filler;
	};

	virtual ~LSTMCell() {};



protected:

	virtual void setWeightPLIG_1stLayer(unsigned int  inputidx,unsigned int  hiddenidx,unsigned int  bidir, BaseTensor& value)  = 0;
	virtual void setWeightPLFG_1stLayer(unsigned int  inputidx,unsigned int  hiddenidx,unsigned int  bidir, BaseTensor& value)  = 0;
	virtual void setWeightPLCG_1stLayer(unsigned int  inputidx,unsigned int  hiddenidx,unsigned int  bidir, BaseTensor& value)  = 0;
	virtual void setWeightPLOG_1stLayer(unsigned int  inputidx,unsigned int  hiddenidx,unsigned int  bidir, BaseTensor& value)  = 0;

	virtual void setWeightPLIG(unsigned int  channelhiddenidx,unsigned int  outputhiddenidx,unsigned int  nlbidir, BaseTensor& value)  = 0;
	virtual void setWeightPLFG(unsigned int  channelhiddenidx,unsigned int  outputhiddenidx,unsigned int  nlbidir, BaseTensor& value)  = 0;
	virtual void setWeightPLCG(unsigned int  channelhiddenidx,unsigned int  outputhiddenidx,unsigned int  nlbidir, BaseTensor& value)  = 0;
	virtual void setWeightPLOG(unsigned int  channelhiddenidx,unsigned int  outputhiddenidx,unsigned int  nlbidir, BaseTensor& value)  = 0;

	virtual void setWeightRIG(unsigned int  channelhiddenidx,unsigned int  outputhiddenidx,unsigned int  nlbidir, BaseTensor& value)  = 0;
	virtual void setWeightRFG(unsigned int  channelhiddenidx,unsigned int  outputhiddenidx,unsigned int  nlbidir, BaseTensor& value)  = 0;
	virtual void setWeightRCG(unsigned int  channelhiddenidx,unsigned int  outputhiddenidx,unsigned int  nlbidir, BaseTensor& value)  = 0;
	virtual void setWeightROG(unsigned int  channelhiddenidx,unsigned int  outputhiddenidx,unsigned int  nlbidir, BaseTensor& value)  = 0;

	virtual void setBiasPLIG(unsigned int  hiddenidx, unsigned int  nlbidir, BaseTensor& value)  = 0;
	virtual void setBiasPLFG(unsigned int  hiddenidx, unsigned int  nlbidir, BaseTensor& value)  = 0;
	virtual void setBiasPLCG(unsigned int  hiddenidx, unsigned int  nlbidir, BaseTensor& value)  = 0;
	virtual void setBiasPLOG(unsigned int  hiddenidx, unsigned int  nlbidir, BaseTensor& value)  = 0;

	virtual void setBiasRIG(unsigned int  hiddenidx, unsigned int  nlbidir, BaseTensor& value)  = 0;
	virtual void setBiasRFG(unsigned int  hiddenidx, unsigned int  nlbidir, BaseTensor& value)  = 0;
	virtual void setBiasRCG(unsigned int  hiddenidx, unsigned int  nlbidir, BaseTensor& value)  = 0;
	virtual void setBiasROG(unsigned int  hiddenidx, unsigned int  nlbidir, BaseTensor& value)  = 0;


	virtual void setOutputsDims();

	virtual void setdytest(){};

	std::shared_ptr<Filler> mWeightsPreviousLayerInputGateFiller_1stLayer;
	std::shared_ptr<Filler> mWeightsPreviousLayerInputGateFiller;
	std::shared_ptr<Filler> mWeightsPreviousLayerForgetGateFiller_1stLayer;
	std::shared_ptr<Filler> mWeightsPreviousLayerForgetGateFiller;
	std::shared_ptr<Filler> mWeightsPreviousLayerCellGateFiller_1stLayer;
	std::shared_ptr<Filler> mWeightsPreviousLayerCellGateFiller;
	std::shared_ptr<Filler> mWeightsPreviousLayerOutputGateFiller_1stLayer;
	std::shared_ptr<Filler> mWeightsPreviousLayerOutputGateFiller;

	std::shared_ptr<Filler> mWeightsRecurrentInputGateFiller;
	std::shared_ptr<Filler> mWeightsRecurrentForgetGateFiller;
	std::shared_ptr<Filler> mWeightsRecurrentCellGateFiller;
	std::shared_ptr<Filler> mWeightsRecurrentOutputGateFiller;


	std::shared_ptr<Filler> mBiasPreviousLayerInputGateFiller;
	std::shared_ptr<Filler> mBiasPreviousLayerForgetGateFiller;
	std::shared_ptr<Filler> mBiasPreviousLayerCellGateFiller;
	std::shared_ptr<Filler> mBiasPreviousLayerOutputGateFiller;

	std::shared_ptr<Filler> mBiasRecurrentInputGateFiller;
	std::shared_ptr<Filler> mBiasRecurrentForgetGateFiller;
	std::shared_ptr<Filler> mBiasRecurrentCellGateFiller;
	std::shared_ptr<Filler> mBiasRecurrentOutputGateFiller;

	//------ Filler to initialize dy for Testing the LSTMclass ; Called by setdyTest() -----//
	std::shared_ptr<Filler> mFillerTest;
	//--------------------------------------------------------------------------------------//


	std::shared_ptr<Solver> mWeightsSolver;

	std::shared_ptr<Filler> mDiffhyFillerTest;
	std::shared_ptr<Filler> mhxFiller;

	std::shared_ptr<Filler> mDiffcyFillerTest;
	std::shared_ptr<Filler> mcxFiller;

	const unsigned int mSeqLength;
	const unsigned int mBatchSize;
	const unsigned int mInputDim;
	const unsigned int mNumberLayers;
	const unsigned int mHiddenSize;
	const unsigned int mAlgo;
	unsigned int mBidirectional;
	const unsigned int mInputMode;
	const Float_T mDropout;
	const bool mSingleBackpropFeeding;

};
}


#endif // N2D2_LSTMCELL_H
