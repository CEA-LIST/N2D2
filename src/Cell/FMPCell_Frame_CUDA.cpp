/*
    (C) Copyright 2014 CEA LIST. All Rights Reserved.
    Contributor(s): Olivier BICHLER (olivier.bichler@cea.fr)

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

#include "Cell/FMPCell_Frame_CUDA.hpp"

N2D2::Registrar<N2D2::FMPCell>
N2D2::FMPCell_Frame_CUDA::mRegistrar("Frame_CUDA",
                                     N2D2::FMPCell_Frame_CUDA::create);

N2D2::FMPCell_Frame_CUDA::FMPCell_Frame_CUDA(const std::string& name,
                                             double scalingRatio,
                                             unsigned int nbOutputs,
                                             const std::shared_ptr
                                             <Activation<Float_T> >& activation)
    : Cell(name, nbOutputs),
      FMPCell(name, scalingRatio, nbOutputs),
      Cell_Frame_CUDA(name, nbOutputs, activation)
{
    // ctor
}

void N2D2::FMPCell_Frame_CUDA::initialize()
{
    FMPCell::initialize();

    for (unsigned int k = 0, size = mInputs.size(); k < size; ++k) {
        if (mInputs[k].size() == 0)
            throw std::runtime_error("Zero-sized input for FMPCell " + mName);
    }

    if (!isUnitMap())
        throw std::domain_error(
            "FMPCell_Frame_CUDA::initialize(): only unit maps are supported.");
}

void N2D2::FMPCell_Frame_CUDA::propagate(bool /*inference*/)
{
    mInputs.synchronizeHBasedToD();

    generateRegions(mGridX, mInputs[0].dimX(), mOutputs.dimX());
    generateRegions(mGridY, mInputs[0].dimY(), mOutputs.dimY());
    mGridX.synchronizeHToD();
    mGridY.synchronizeHToD();

    cudaSFMPPropagate(mInputs[0].getDevicePtr(),
                      mGridX.getDevicePtr(),
                      mGridY.getDevicePtr(),
                      mOutputs.getDevicePtr(),
                      mNbChannels,
                      mInputs[0].dimX(),
                      mInputs[0].dimY(),
                      mNbOutputs,
                      mOutputs.dimX(),
                      mOutputs.dimY(),
                      mInputs.dimB(),
                      mOverlapping);
}

void N2D2::FMPCell_Frame_CUDA::backPropagate()
{
    throw std::runtime_error(
        "FMPCell_Frame_CUDA::backPropagate(): not implemented.");
}

void N2D2::FMPCell_Frame_CUDA::update()
{
}

void N2D2::FMPCell_Frame_CUDA::generateRegions(CudaTensor4d<unsigned int>& grid,
                                               unsigned int sizeIn,
                                               unsigned int sizeOut)
{
    grid.resize(1, 1, sizeOut, 1);

    if (mPseudoRandom) {
        // Compute the true scaling ratio
        // This is important to obtain the correct range
        const double scalingRatio = sizeIn / (double)sizeOut;
        const double u = Random::randUniform(0.0, 1.0, Random::OpenInterval);

        for (unsigned int i = 0; i < sizeOut; ++i)
            grid(i) = (unsigned int)std::ceil(scalingRatio * (i + u));
    } else {
        const unsigned int nb2 = sizeIn - sizeOut;
        // const unsigned int nb1 = 2*sizeOut - sizeIn;
        // assert(nb1 + nb2 == sizeOut);

        std::fill(grid.begin(), grid.begin() + nb2, 2);
        std::fill(grid.begin() + nb2, grid.end(), 1);

        // Random shuffle
        for (int i = grid.size() - 1; i > 0; --i)
            std::swap(grid(i), grid(Random::randUniform(0, i)));

        for (unsigned int i = 1; i < grid.size(); ++i)
            grid(i) += grid(i - 1);
    }
}

#endif
