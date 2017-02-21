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

#include "Cell/PoolCell_Spike.hpp"

N2D2::Registrar<N2D2::PoolCell>
N2D2::PoolCell_Spike::mRegistrar("Spike", N2D2::PoolCell_Spike::create);

N2D2::PoolCell_Spike::PoolCell_Spike(Network& net,
                                     const std::string& name,
                                     unsigned int poolWidth,
                                     unsigned int poolHeight,
                                     unsigned int nbOutputs,
                                     unsigned int strideX,
                                     unsigned int strideY,
                                     unsigned int paddingX,
                                     unsigned int paddingY,
                                     Pooling pooling)
    : Cell(name, nbOutputs),
      PoolCell(name,
               poolWidth,
               poolHeight,
               nbOutputs,
               strideX,
               strideY,
               paddingX,
               paddingY,
               pooling),
      Cell_Spike(net, name, nbOutputs),
      mPoolNbChannels(nbOutputs, 0)
// IMPORTANT: Do not change the value of the parameters here! Use setParameter()
// or loadParameters().
{
    // ctor
}

void N2D2::PoolCell_Spike::initialize()
{
    mInputsActivity.resize(mChannelsWidth, mChannelsHeight, mNbChannels, 1);

    mInputMax.resize(mOutputsWidth, mOutputsHeight, mNbOutputs, 1, -1);
    mPoolActivity.resize(mOutputsWidth,
                         mOutputsHeight,
                         mNbOutputs,
                         1,
                         (mPooling == Max) ? std::numeric_limits<int>::min()
                                           : 0);
    mOutputsActivity.resize(mOutputsWidth, mOutputsHeight, mNbOutputs, 1, 0);

    for (unsigned int output = 0; output < mNbOutputs; ++output) {
        for (unsigned int channel = 0; channel < getNbChannels(); ++channel)
            mPoolNbChannels[output] += isConnection(channel, output);
    }
}

void N2D2::PoolCell_Spike::propagateSpike(NodeIn* origin,
                                          Time_T timestamp,
                                          EventType_T type)
{
    const Area& area = origin->getArea();
    const unsigned int channel = origin->getChannel();

    if (type)
        --mInputsActivity(area.x, area.y, channel, 0);
    else
        ++mInputsActivity(area.x, area.y, channel, 0);

    const unsigned int oxStride
        = mStrideX * (unsigned int)((mChannelsWidth - mPoolWidth + mStrideX)
                                    / (double)mStrideX);
    const unsigned int oyStride
        = mStrideY * (unsigned int)((mChannelsHeight - mPoolHeight + mStrideY)
                                    / (double)mStrideY);
    const unsigned int sxMax = std::min<unsigned int>(mPoolWidth, area.x + 1);
    const unsigned int syMax = std::min<unsigned int>(mPoolHeight, area.y + 1);

    for (unsigned int sy = area.y % mStrideY, sx0 = area.x % mStrideX;
         sy < syMax;
         sy += mStrideY) {
        if (area.y >= oyStride + sy)
            continue;

        for (unsigned int sx = sx0; sx < sxMax; sx += mStrideX) {
            // Border conditions
            if (area.x >= oxStride + sx)
                continue;

            // Output node coordinates
            const unsigned int ox = (area.x - sx) / mStrideX;
            const unsigned int oy = (area.y - sy) / mStrideY;

            for (unsigned int output = 0; output < mNbOutputs; ++output) {
                if (!isConnection(channel, output))
                    continue;

                incomingSpike(origin, timestamp, maps(output, ox, oy, type));
            }
        }
    }
}

void N2D2::PoolCell_Spike::incomingSpike(NodeIn* origin,
                                         Time_T timestamp,
                                         EventType_T type)
{
    // Input node coordinates
    const Area& area = origin->getArea();

    // Output node coordinates
    unsigned int output, ox, oy;
    bool negative;
    std::tie(output, ox, oy, negative) = unmaps(type);

    switch (mPooling) {
    case Max: {
        /*
                    // Ideal behavior
                    int maxActivity = std::numeric_limits<int>::min();

                    const unsigned int sxMax = std::min(mChannelsWidth -
           ox*mStrideX, mPoolWidth);
                    const unsigned int syMax = std::min(mChannelsHeight -
           oy*mStrideY, mPoolHeight);

                    for (unsigned int channel = 0; channel < mNbChannels;
           ++channel) {
                        if (!isConnection(channel, output))
                            continue;

                        for (unsigned int sy = 0; sy < syMax; ++sy) {
                            for (unsigned int sx = 0; sx < sxMax; ++sx) {
                                const unsigned int ix = ox*mStrideX + sx;
                                const unsigned int iy = oy*mStrideY + sy;

                                maxActivity = std::max(maxActivity,
           mInputsActivity[channel](iy,ix));
                            }
                        }
                    }

                    if (maxActivity != mPoolActivity(ox, oy, output, 0)) {
                        const bool negSpike = (maxActivity < mPoolActivity(ox,
           oy, output, 0));
                        mOutputs(ox, oy, output, 0)->incomingSpike(NULL,
           timestamp + 1*TimeFs, negSpike);

                        mPoolActivity(ox, oy, output, 0) = maxActivity;
                    }
        */
        // Approximated behavior
        const unsigned int inputSize = mChannelsHeight * mChannelsWidth;
        const unsigned int inputIdx = (area.x + mChannelsWidth * area.y)
                                      + origin->getChannel() * inputSize;

        if (mInputsActivity(area.x, area.y, origin->getChannel(), 0)
            > mPoolActivity(ox, oy, output, 0)
            || ((mInputMax(ox, oy, output, 0) == -1
                 || mInputMax(ox, oy, output, 0) == (int)inputIdx)
                && mInputsActivity(area.x, area.y, origin->getChannel(), 0)
                   < mPoolActivity(ox, oy, output, 0))) {
            const bool negSpike
                = (mInputsActivity(area.x, area.y, origin->getChannel(), 0)
                   < mPoolActivity(ox, oy, output, 0));
            mOutputs(ox, oy, output, 0)
                ->incomingSpike(NULL, timestamp + 1 * TimeFs, negSpike);

            mPoolActivity(ox, oy, output, 0)
                = mInputsActivity(area.x, area.y, origin->getChannel(), 0);
            mInputMax(ox, oy, output, 0) = inputIdx;
        }
    } break;

    case Average: {
        const unsigned int poolSize = mPoolHeight * mPoolWidth
                                      * mPoolNbChannels[output];

        if (negative)
            --mPoolActivity(ox, oy, output, 0);
        else
            ++mPoolActivity(ox, oy, output, 0);

        if ((int)Utils::round(
                std::fabs(mPoolActivity(ox, oy, output, 0) / (double)poolSize))
            > std::abs(mOutputsActivity(ox, oy, output, 0))) {
            const bool negSpike = (mPoolActivity(ox, oy, output, 0) < 0);
            mOutputs(ox, oy, output, 0)
                ->incomingSpike(NULL, timestamp + 1 * TimeFs, negSpike);

            if (negSpike)
                --mOutputsActivity(ox, oy, output, 0);
            else
                ++mOutputsActivity(ox, oy, output, 0);
        }
    } break;

    default:
        throw std::runtime_error("PoolCell_Spike::incomingSpike(): pooling "
                                 "layer should be of type \"Max\" or "
                                 "\"Average\".");
    }
}

void N2D2::PoolCell_Spike::notify(Time_T /*timestamp*/, NotifyType notify)
{
    if (notify == Reset) {
        mInputsActivity.assign(
            mChannelsWidth, mChannelsHeight, mNbChannels, 1, 0);

        mInputMax.assign(mOutputsWidth, mOutputsHeight, mNbOutputs, 1, -1);
        mPoolActivity.assign(mOutputsWidth,
                             mOutputsHeight,
                             mNbOutputs,
                             1,
                             (mPooling == Max) ? std::numeric_limits<int>::min()
                                               : 0);
        mOutputsActivity.assign(
            mOutputsWidth, mOutputsHeight, mNbOutputs, 1, 0);
    } else if (notify == Load)
        load(mNet.getLoadSavePath());
    else if (notify == Save)
        save(mNet.getLoadSavePath());
}
