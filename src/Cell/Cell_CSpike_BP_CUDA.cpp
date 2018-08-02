/*
    (C) Copyright 2016 CEA LIST. All Rights Reserved.
    Contributor(s): Johannes THIELE (johannes.thiele@cea.fr)
                    Olivier BICHLER (olivier.bichler@cea.fr)

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

#include "Cell/Cell_CSpike_BP_CUDA.hpp"

N2D2::Cell_CSpike_BP_CUDA::Cell_CSpike_BP_CUDA(const std::string& name,
                                                unsigned int nbOutputs)
    : Cell(name, nbOutputs),
      Cell_CSpike_CUDA(name, nbOutputs),
      mExampleReset(this, "ExampleReset", true),
      mSubConnectIdx(this, "ConnectSubStimulus", 0)
{
    // ctor
}


void N2D2::Cell_CSpike_BP_CUDA::addInput(StimuliProvider& sp,
                                      unsigned int x0,
                                      unsigned int y0,
                                      unsigned int width,
                                      unsigned int height,
                                      const Matrix<bool>& mapping)
{

    CEnvironment* cEnv = dynamic_cast<CEnvironment*>(&sp);

    if (cEnv == NULL)
        throw std::runtime_error(
            "Cell_CSpike_CUDA::addInput(): CSpike models require CEnvironment");

    if (!mapping.empty()) {
        throw std::runtime_error("Cell_CSpike_BP_CUDA::addInput(): "
                            " mapping feature is not available. "
                            " Maybe you want to use the sub stimuli feature?");
    }


    unsigned int subMax = (cEnv->getTickData()).size();
    unsigned int subMin = 0;

    if (mSubConnectIdx > 0){
        subMin = mSubConnectIdx - 1;
        subMax = mSubConnectIdx;
    }

    bool firstInput = true;

    for (unsigned int k=subMin; k<subMax; ++k){

        Tensor<char>& subTickData = cEnv->getTickData(k);

        if (width == 0)
            width = subTickData.dimX() - x0;
        if (height == 0)
            height = subTickData.dimY() - y0;

        if (x0 > 0 || y0 > 0 || width < subTickData.dimX() || height < subTickData.dimY()) {
            throw std::runtime_error("Cell_CSpike_BP_CUDA::addInput(): adding a "
                                     "cropped environment channel map as input is "
                                     "not "
                                     "supported");
        }

        // Define input-output sizes
        /// Warning: mNbChannels, mChannelsWidth and mChannelSize have a different meaning
        /// than normal. Use Inputs dimensions instead in main code

        if(firstInput){
            setInputsDims({width, height, subTickData.dimZ()});
            firstInput = false;
        }
        else {
            setInputsDims({width, height, getNbChannels() + subTickData.dimZ()});
        }
        setOutputsDims();

        mInputs.push_back(&subTickData);
        mInputs.back().setValid();

    }

    if (mOutputs.empty()) {
        mOutputs.resize(
            {getOutputsWidth(), getOutputsHeight(), getNbOutputs(), sp.getBatchSize()});
        mOutputsActivity.resize(
            {getOutputsWidth(), getOutputsHeight(), getNbOutputs(), sp.getBatchSize()}, 0);
    }
}

void N2D2::Cell_CSpike_BP_CUDA::addInput(Cell* cell, const Matrix<bool>& mapping)
{

    Cell_CSpike_CUDA* cellCSpike = dynamic_cast<Cell_CSpike_CUDA*>(cell);

    if (cellCSpike == NULL)
        throw std::runtime_error(
            "Cell_CSpike_BP_CUDA::addInput(): Cast failed");

    // Define input-output sizes
    /// Warning: mNbChannels, mChannelsWidth and mChannelSize have a different meaning
    /// than normal. Use Inputs dimensions instead in main code
    setInputsDims({cellCSpike->getOutputsWidth(),
                   cellCSpike->getOutputsHeight(),
                   cellCSpike->getNbOutputs()});
    setOutputsDims();
    mSubConnectIdx = 1;

    if (!mapping.empty()) {
        throw std::runtime_error("Cell_CSpike_BP_CUDA::addInput(): "
                                 " mapping feature is not available. "
                                 " Maybe you want to use the sub stimuli feature?");
    }

    mInputs.push_back(&cellCSpike->getOutputs());

    if (mOutputs.empty()) {
        mOutputs.resize(
            {getOutputsWidth(), getOutputsHeight(), getNbOutputs(), mInputs.dimB()});
        mOutputsActivity.resize(
            {getOutputsWidth(), getOutputsHeight(), getNbOutputs(), mInputs.dimB()}, 0);
    }

    mLowerCell = dynamic_cast<Cell_CSpike_BP_CUDA*>(cell);
    if(!mLowerCell) {
         throw std::runtime_error(
            "Cell_CSpike_BP_CUDA::addInput(): mLowerCell cast failed");
    }

}

void N2D2::Cell_CSpike_BP_CUDA::addInput(Cell* cell,
                                unsigned int x0,
                                unsigned int y0,
                                unsigned int width,
                                unsigned int height)
{
    Cell_CSpike_CUDA* cellCSpike = dynamic_cast<Cell_CSpike_CUDA*>(cell);

    if (cellCSpike == NULL)
        throw std::runtime_error(
            "Cell_CSpike_BP_CUDA::addInput(): Cast Failed");

    if (width == 0)
        width = cellCSpike->getOutputsWidth() - x0;
    if (height == 0)
        height = cellCSpike->getOutputsHeight() - y0;

    if (x0 > 0 || y0 > 0 || width < cellCSpike->getOutputsWidth()
        || height < cellCSpike->getOutputsHeight())
        throw std::runtime_error("Cell_CSpike_BP_CUDA::addInput(): adding a "
                                 "cropped output map as input is not "
                                 "supported");

    Cell_CSpike_BP_CUDA::addInput(cellCSpike);
}




#endif
