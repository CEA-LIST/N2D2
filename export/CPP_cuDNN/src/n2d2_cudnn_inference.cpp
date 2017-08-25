/*
    (C) Copyright 2017 CEA LIST. All Rights Reserved.
    Contributor(s): Olivier BICHLER (olivier.bichler@cea.fr)
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

#include "n2d2_cudnn_inference.hpp"

n2d2_cudnn_inference::n2d2_cudnn_inference( unsigned int batchSize,
                                                  unsigned int devID,
                                                  bool profiling)
    : mBatchSize(batchSize),
      mDeviceID(devID)
{
    // ctor
    mProfiling = false;

    if (mBatchSize == 0)
        throw std::runtime_error("n2d2_cudnn_inference() constructor: "
                                 "batchSize must be at least 1");

    mNbTargets = getOutputNbTargets();
    std::cout << "The network provides "
                <<  mNbTargets
                << " output targets" << std::endl;

    if(profiling)
    {
        std::cout << "A layer wise profiling is set, it can decrease real time performances" << std::endl;
        mProfiling = true;
        setProfiling();
    }

    network_cudnn_init(mBatchSize, mDeviceID);

}

void n2d2_cudnn_inference::initialize(unsigned int batchSize,
                                         unsigned int devID,
                                         bool profiling)
{
    mProfiling = false;

    if (batchSize == 0)
        throw std::runtime_error("n2d2_cudnn_inference::initialize(): "
                                 "batchSize must be at least 1");
    mBatchSize = batchSize;

    mDeviceID = devID;

    network_cudnn_init(mBatchSize, mDeviceID);

    if(profiling)
    {
        std::cout << "A layer wise profiling is set, it can decrease real time performances" << std::endl;
        mProfiling = true;
        setProfiling();
    }

}

void n2d2_cudnn_inference::getProfiling(unsigned int nbIter)
{
    if(mProfiling)
        reportProfiling(nbIter);
}


void n2d2_cudnn_inference::setBatchSize(unsigned int batchSize)
{
    if (batchSize == 0)
        throw std::runtime_error("n2d2_cudnn_inference::setBatchSize(): "
                                 "batchSize must be at least 1");
    if(batchSize > mBatchSize)
        initialize(batchSize, mDeviceID, mProfiling);
    else
        mBatchSize = batchSize;

}

void n2d2_cudnn_inference::setDeviceID(unsigned int devID)
{
    if(devID != mDeviceID)
        initialize(mBatchSize, devID, mProfiling);
}

void n2d2_cudnn_inference::execute(DATA_T* input_data)
{
    network_cudnn_syncExe(input_data, mBatchSize);
}

void n2d2_cudnn_inference
    ::estimated(uint32_t* output_data, unsigned int target)
{
    if(target > mNbTargets)
        throw std::runtime_error("n2d2_cudnn_inference::estimated(): "
                                 "invalid target !");

    network_cudnn_output(output_data, mBatchSize, target);
}

unsigned int n2d2_cudnn_inference::inputDimX()
{
    return getInputDimX();
}

unsigned int n2d2_cudnn_inference::inputDimY()
{
    return getInputDimY();
}

unsigned int n2d2_cudnn_inference::inputDimZ()
{
    return getInputDimZ();
}

std::vector<unsigned int> n2d2_cudnn_inference::outputDimX()
{
    std::vector<unsigned int> dim;
    unsigned int nbTarget = getOutputNbTargets();

    for(unsigned int it = 0; it < nbTarget; ++it)
        dim.push_back(getOutputDimX(it));

    return dim;
}

std::vector<unsigned int> n2d2_cudnn_inference::outputDimY()
{
    std::vector<unsigned int> dim;
    unsigned int nbTarget = getOutputNbTargets();

    for(unsigned int it = 0; it < nbTarget; ++it)
        dim.push_back(getOutputDimY(it));

    return dim;
}

std::vector<unsigned int> n2d2_cudnn_inference::outputDimZ()
{
    std::vector<unsigned int> dim;
    unsigned int nbTarget = getOutputNbTargets();

    for(unsigned int it = 0; it < nbTarget; ++it)
        dim.push_back(getOutputDimZ(it));

    return dim;
}

std::vector<unsigned int> n2d2_cudnn_inference::outputTarget()
{
    std::vector<unsigned int> dim;
    unsigned int nbTarget = getOutputNbTargets();

    for(unsigned int it = 0; it < nbTarget; ++it)
        dim.push_back(getOutputTarget(it));

    return dim;
}



