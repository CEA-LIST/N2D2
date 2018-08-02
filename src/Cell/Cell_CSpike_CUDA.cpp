/*
    (C) Copyright 2016 CEA LIST. All Rights Reserved.
    Contributor(s): Olivier BICHLER (olivier.bichler@cea.fr)
                    Johannes THIELE (johannes.thiele@cea.fr)

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

#include "Cell/Cell_CSpike_CUDA.hpp"

N2D2::Cell_CSpike_CUDA::Cell_CSpike_CUDA(const std::string& name,
                                         unsigned int nbOutputs)
    : Cell(name, nbOutputs)
{
    // ctor
}

void N2D2::Cell_CSpike_CUDA::addInput(StimuliProvider& /*sp*/,
                                      unsigned int /*channel*/,
                                      unsigned int /*x0*/,
                                      unsigned int /*y0*/,
                                      unsigned int /*width*/,
                                      unsigned int /*height*/,
                                      const std::vector<bool>& /*mapping*/)
{
    throw std::runtime_error("Cell_CSpike_CUDA::addInput(): adding a single "
                             "environment channel as input is not supported");
}

/*
void N2D2::Cell_CSpike_CUDA::addInput(StimuliProvider& sp,
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

    unsigned int subMax = (cEnv->getTickData()).size();
    unsigned int subMin = 0;

    if (mSubConnectIdx > 0){
        subMin = mSubConnectIdx - 1;
        subMax = mSubConnectIdx;
    }

    //std::cout << "Add subStim " << k << " to cell" << std::endl;

    for (unsigned int k=subMin; k<subMax; ++k){

        std::cout << "Add subStim " << k << " to cell" << std::endl;

        Tensor4d<char>& subTickData = cEnv->getTickData(k);

        if (width == 0)
            width = subTickData.dimX() - x0;
        if (height == 0)
            height = subTickData.dimY() - y0;

        if (x0 > 0 || y0 > 0 || width < subTickData.dimX() || height < subTickData.dimY()) {
            throw std::runtime_error("Cell_CSpike_CUDA::addInput(): adding a "
                                     "cropped environment channel map as input is "
                                     "not "
                                     "supported");
        }

        // Define input-output sizes
        setInputsSize(width, height);
        mNbChannels += subTickData.dimZ();
        setOutputsSize();

        // Define input-output connections
        if (!mapping.empty() && mapping.rows() != subTickData.dimZ()) {
            throw std::runtime_error("Cell_CSpike_CUDA::addInput(): number of "
                                     "mapping rows must be equal to the number of "
                                     "input "
                                     "channels");
        }

        mMaps.resize(mNbOutputs, mNbChannels);
        const unsigned int channelOffset = mNbChannels - subTickData.dimZ();

        for (unsigned int output = 0; output < mNbOutputs; ++output) {
            for (unsigned int channel = 0; channel < subTickData.dimZ();
                 ++channel) {
                mMaps(output, channelOffset + channel)
                    = (!mapping.empty()) ? mapping(channel, output) : true;
            }
        }

        mInputs.push_back(&subTickData);
        mInputs.back().setValid();

    }

    if (mOutputs.empty()) {
        mOutputs.resize(
            mOutputsWidth, mOutputsHeight, mNbOutputs, sp.getBatchSize());
        mOutputsActivity.resize(
            mOutputsWidth, mOutputsHeight, mNbOutputs, sp.getBatchSize(), 0);
    }
}*/

void N2D2::Cell_CSpike_CUDA::addInput(StimuliProvider& sp,
                                 unsigned int x0,
                                 unsigned int y0,
                                 unsigned int width,
                                 unsigned int height,
                                 const Matrix<bool>& mapping)
{
    if (width == 0)
        width = sp.getSizeX() - x0;
    if (height == 0)
        height = sp.getSizeY() - y0;

    if (x0 > 0 || y0 > 0 || width < sp.getSizeX() || height < sp.getSizeY()) {
        throw std::runtime_error("Cell_CSpike_CUDA::addInput(): adding a "
                                 "cropped environment channel map as input is "
                                 "not "
                                 "supported");
    }

    // Define input-output sizes
    setInputsDims({width, height, sp.getNbChannels()});
    setOutputsDims();

    // Define input-output connections
    if (!mapping.empty() && mapping.rows() != sp.getNbChannels()) {
        throw std::runtime_error("Cell_CSpike_CUDA::addInput(): number of "
                                 "mapping rows must be equal to the number of "
                                 "input "
                                 "channels");
    }

    mMaps.resize({getNbOutputs(), getNbChannels()});
    const unsigned int channelOffset = getNbChannels() - sp.getNbChannels();

    for (unsigned int output = 0; output < getNbOutputs(); ++output) {
        for (unsigned int channel = 0; channel < sp.getNbChannels();
             ++channel) {
            mMaps(output, channelOffset + channel)
                = (!mapping.empty()) ? mapping(channel, output) : true;
        }
    }

    CEnvironment* cEnv = dynamic_cast<CEnvironment*>(&sp);

    if (cEnv == NULL)
        throw std::runtime_error(
            "Cell_CSpike_CUDA::addInput(): CSpike models require CEnvironment");

    unsigned int inputTickDataSize = (cEnv->getTickData()).size();

    for (unsigned int k=0; k<inputTickDataSize; ++k){
        mInputs.push_back(&(cEnv->getTickData(k)));
        mInputs.back().setValid();
    }

    if (mOutputs.empty()) {
        std::vector<size_t> outputsDims = mOutputsDims;
        outputsDims.push_back(sp.getBatchSize());

        mOutputs.resize(outputsDims);
        mOutputsActivity.resize(outputsDims, 0);
    }
}

void N2D2::Cell_CSpike_CUDA::addInput(Cell* cell, const Matrix<bool>& mapping)
{
    Cell_CSpike_CUDA* cellCSpike = dynamic_cast<Cell_CSpike_CUDA*>(cell);

    if (cellCSpike == NULL)
        throw std::runtime_error(
            "Cell_CSpike_CUDA::addInput(): cannot mix Spike and Frame models");

    // Define input-output sizes
    setInputsDims(cellCSpike->getOutputsDims());
    setOutputsDims();

    // Define input-output connections
    const unsigned int cellNbOutputs = cellCSpike->getNbOutputs();

    if (!mapping.empty() && mapping.rows() != cellNbOutputs) {
        throw std::runtime_error("Cell_CSpike_CUDA::addInput(): number of "
                                 "mapping rows must be equal to the number of "
                                 "input "
                                 "channels");
    }

    mMaps.resize({getNbOutputs(), getNbChannels()});
    const unsigned int channelOffset = getNbChannels() - cellNbOutputs;

    for (unsigned int output = 0; output < getNbOutputs(); ++output) {
        for (unsigned int channel = 0; channel < cellNbOutputs;
             ++channel) {
            mMaps(output, channelOffset + channel)
                = (!mapping.empty()) ? mapping(channel, output) : true;
        }
    }

    mInputs.push_back(&cellCSpike->mOutputs);

    if (mOutputs.empty()) {
        std::vector<size_t> outputsDims = mOutputsDims;
        outputsDims.push_back(mInputs.dimB());

        mOutputs.resize(outputsDims);
        mOutputsActivity.resize(outputsDims, 0);
    }
}

void N2D2::Cell_CSpike_CUDA::addInput(Cell* cell,
                                      unsigned int x0,
                                      unsigned int y0,
                                      unsigned int width,
                                      unsigned int height)
{
    Cell_CSpike_CUDA* cellCSpike = dynamic_cast<Cell_CSpike_CUDA*>(cell);

    if (cellCSpike == NULL)
        throw std::runtime_error(
            "Cell_CSpike_CUDA::addInput(): cannot mix Spike and Frame models");

    if (width == 0)
        width = cellCSpike->mOutputsDims[0] - x0;
    if (height == 0)
        height = cellCSpike->mOutputsDims[1] - y0;

    if (x0 > 0 || y0 > 0 || width < cellCSpike->mOutputsDims[0]
        || height < cellCSpike->mOutputsDims[1])
        throw std::runtime_error("Cell_CSpike_CUDA::addInput(): adding a "
                                 "cropped output map as input is not "
                                 "supported");

    Cell_CSpike_CUDA::addInput(cellCSpike);
}

bool N2D2::Cell_CSpike_CUDA::tick(Time_T /*timestamp*/)
{
    accumulate<Float_T>(&mOutputsActivity, &mOutputs);
    //TODO: Check if this sync is necessary (required by CMonitor?)
    //mOutputs.synchronizeDToH();
    return false;
}

void N2D2::Cell_CSpike_CUDA::reset(Time_T /*timestamp*/)
{
    mOutputsActivity.assign(mOutputsActivity.dims(), 0);
}

N2D2::Tensor<N2D2::Float_T>& N2D2::Cell_CSpike_CUDA::getOutputsActivity()
{
    mOutputsActivity.synchronizeDToH();
    return mOutputsActivity;
}

//TODO: Check where this is used
N2D2::Tensor<char>& N2D2::Cell_CSpike_CUDA::getOutputs()
{
    mOutputs.synchronizeDToH();
    return mOutputs;
}

namespace N2D2 {
template <>
void Cell_CSpike_CUDA::accumulate
    <float>(CudaTensor<float>* outputsActivity, CudaTensor<char>* outputs)
{
    cudaSaccumulate(outputsActivity->getDevicePtr(),
                    outputs->getDevicePtr(),
                    outputs->size());
}

template <>
void Cell_CSpike_CUDA::accumulate
    <double>(CudaTensor<double>* outputsActivity, CudaTensor<char>* outputs)
{
    cudaDaccumulate(outputsActivity->getDevicePtr(),
                    outputs->getDevicePtr(),
                    outputs->size());
}
}

#endif
