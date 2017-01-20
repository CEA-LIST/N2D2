/*
    (C) Copyright 2016 CEA LIST. All Rights Reserved.
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

#include "CEnvironment.hpp"

N2D2::CEnvironment::CEnvironment(Database& database,
                                 unsigned int sizeX,
                                 unsigned int sizeY,
                                 unsigned int nbChannels,
                                 unsigned int batchSize,
                                 bool compositeStimuli)
    : StimuliProvider(
          database, sizeX, sizeY, nbChannels, batchSize, compositeStimuli),
      SpikeGenerator(),
      mInitialized(false),
      mTickData(sizeX, sizeY, nbChannels, batchSize, 0)
{
    // ctor
}

void N2D2::CEnvironment::addChannel(const CompositeTransformation
                                    & transformation)
{
    if (!mChannelsTransformations.empty())
        mTickData.resize(mTickData.dimX(),
                         mTickData.dimY(),
                         mTickData.dimZ() + 1,
                         mTickData.dimB());
    else
        mTickData.resize(
            mTickData.dimX(), mTickData.dimY(), 1, mTickData.dimB());

    StimuliProvider::addChannel(transformation);
}

void N2D2::CEnvironment::tick(Time_T timestamp, Time_T start, Time_T stop)
{
    SpikeGenerator::checkParameters();

    if (!mInitialized) {
        mNextEvent.assign(mData.dimX(),
                          mData.dimY(),
                          mData.dimZ(),
                          mData.dimB(),
                          std::make_pair(start, 0));

        for (unsigned int idx = 0, size = mData.size(); idx < size; ++idx)
            SpikeGenerator::nextEvent(mNextEvent(idx), mData(idx), start, stop);

        mInitialized = true;
    }

    for (unsigned int idx = 0, size = mData.size(); idx < size; ++idx) {
        if (mNextEvent(idx).second != 0 && mNextEvent(idx).first <= timestamp) {
            mTickData(idx) = mNextEvent(idx).second;
            std::pair<Time_T, char> event;

            // std::cout << "cenv(" << idx << "): spike (scheduled @ " <<
            // mNextEvent(idx).first << ") @ " << timestamp << std::endl;

            for (unsigned int k = 0; mNextEvent(idx).second != 0
                                     && mNextEvent(idx).first <= timestamp;
                 ++k) {
                if (k > 0) {
                    std::cout << Utils::cwarning << "cenv(" << idx
                              << "): lost spike (scheduled @ "
                              << mNextEvent(idx).first << ", previous @ "
                              << event.first << ") @ " << timestamp
                              << Utils::cdef << std::endl;
                }

                event = mNextEvent(idx);
                SpikeGenerator::nextEvent(
                    mNextEvent(idx), mData(idx), start, stop);
            }
        } else
            mTickData(idx) = 0;
    }
}

void N2D2::CEnvironment::reset(Time_T /*timestamp*/)
{
    mInitialized = false;
}

N2D2::CEnvironment::~CEnvironment()
{
    // dtor
}
