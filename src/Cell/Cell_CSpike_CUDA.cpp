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
                                      const Tensor<bool>& /*mapping*/)
{
    throw std::runtime_error("Cell_CSpike_CUDA::addInput(): adding a single "
                             "environment channel as input is not supported");
}


void N2D2::Cell_CSpike_CUDA::addInput(StimuliProvider& sp,
                                 unsigned int x0,
                                 unsigned int y0,
                                 unsigned int width,
                                 unsigned int height,
                                 const Tensor<bool>& mapping)
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
    if (!mapping.empty() && mapping.dimY() != sp.getNbChannels()) {
        throw std::runtime_error("Cell_CSpike_CUDA::addInput(): number of "
                                 "mapping rows must be equal to the number of "
                                 "input "
                                 "channels");
    }

    mMapping.append((!mapping.empty())
        ? mapping
        : Tensor<bool>({getNbOutputs(), sp.getNbChannels()}, true));

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

void N2D2::Cell_CSpike_CUDA::addInput(Cell* cell, const Tensor<bool>& mapping)
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

    if (!mapping.empty() && mapping.dimY() != cellNbOutputs) {
        throw std::runtime_error("Cell_CSpike_CUDA::addInput(): number of "
                                 "mapping rows must be equal to the number of "
                                 "input "
                                 "channels");
    }

    mMapping.append((!mapping.empty())
        ? mapping
        : Tensor<bool>({getNbOutputs(), cellNbOutputs}, true));

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
    accumulate<int>(&mOutputsActivity, &mOutputs);
    return false;
}

void N2D2::Cell_CSpike_CUDA::reset(Time_T /*timestamp*/)
{
    mOutputsActivity.assign(mOutputsActivity.dims(), 0);
}

N2D2::Tensor<int>& N2D2::Cell_CSpike_CUDA::getOutputsActivity()
{
    mOutputsActivity.synchronizeDToH();
    return mOutputsActivity;
}

N2D2::Tensor<int>& N2D2::Cell_CSpike_CUDA::getOutputs()
{
    mOutputs.synchronizeDToH();
    return mOutputs;
}

namespace N2D2 {
template <>
void Cell_CSpike_CUDA::accumulate
    <int>(CudaTensor<int>* outputsActivity, CudaTensor<int>* outputs)
{
    cudaIaccumulate(outputsActivity->getDevicePtr(),
                    outputs->getDevicePtr(),
                    outputs->size());
}

template <>
void Cell_CSpike_CUDA::accumulate
    <float>(CudaTensor<float>* outputsActivity, CudaTensor<int>* outputs)
{
    cudaSaccumulate(outputsActivity->getDevicePtr(),
                    outputs->getDevicePtr(),
                    outputs->size());
}

template <>
void Cell_CSpike_CUDA::accumulate
    <double>(CudaTensor<double>* outputsActivity, CudaTensor<int>* outputs)
{
    cudaDaccumulate(outputsActivity->getDevicePtr(),
                    outputs->getDevicePtr(),
                    outputs->size());
}
}

#endif
