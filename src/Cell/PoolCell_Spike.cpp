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
N2D2::PoolCell_Spike::mRegistrar("Spike",
    N2D2::PoolCell_Spike::create,
    N2D2::Registrar<N2D2::PoolCell>::Type<Float_T>());

N2D2::PoolCell_Spike::PoolCell_Spike(Network& net,
    const std::string& name,
    const std::vector<unsigned int>& poolDims,
    unsigned int nbOutputs,
    const std::vector<unsigned int>& strideDims,
    const std::vector<unsigned int>& paddingDims,
    Pooling pooling)
    : Cell(name, nbOutputs),
      PoolCell(name,
               poolDims,
               nbOutputs,
               strideDims,
               paddingDims,
               pooling),
      Cell_Spike(net, name, nbOutputs),
      mPoolNbChannels(nbOutputs, 0)
// IMPORTANT: Do not change the value of the parameters here! Use setParameter()
// or loadParameters().
{
    // ctor
    if (poolDims.size() != 2) {
        throw std::domain_error("PoolCell_Spike: only 2D pooling is"
                                " supported");
    }

    if (strideDims.size() != poolDims.size()) {
        throw std::domain_error("PoolCell_Spike: the number of dimensions"
                                " of stride must match the number of"
                                " dimensions of the pooling.");
    }

    if (paddingDims.size() != poolDims.size()) {
        throw std::domain_error("PoolCell_Spike: the number of dimensions"
                                " of padding must match the number of"
                                " dimensions of the pooling.");
    }
}

void N2D2::PoolCell_Spike::initialize()
{
    std::vector<size_t> inputsDims = mInputsDims;
    inputsDims.push_back(1);

    mInputsActivity.resize(inputsDims);

    std::vector<size_t> outputsDims = mOutputsDims;
    outputsDims.push_back(1);

    mInputMax.resize(outputsDims, -1);
    mPoolActivity.resize(outputsDims,
                         (mPooling == Max) ? std::numeric_limits<int>::min()
                                           : 0);
    mOutputsActivity.resize(outputsDims, 0);

    for (unsigned int output = 0; output < getNbOutputs(); ++output) {
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
        = mStrideDims[0] * (unsigned int)((mInputsDims[0] - mPoolDims[0] + mStrideDims[0])
                                    / (double)mStrideDims[0]);
    const unsigned int oyStride
        = mStrideDims[1] * (unsigned int)((mInputsDims[1] - mPoolDims[1] + mStrideDims[1])
                                    / (double)mStrideDims[1]);
    const unsigned int sxMax = std::min<unsigned int>(mPoolDims[0], area.x + 1);
    const unsigned int syMax = std::min<unsigned int>(mPoolDims[1], area.y + 1);

    for (unsigned int sy = area.y % mStrideDims[1], sx0 = area.x % mStrideDims[0];
         sy < syMax;
         sy += mStrideDims[1]) {
        if (area.y >= oyStride + sy)
            continue;

        for (unsigned int sx = sx0; sx < sxMax; sx += mStrideDims[0]) {
            // Border conditions
            if (area.x >= oxStride + sx)
                continue;

            // Output node coordinates
            const unsigned int ox = (area.x - sx) / mStrideDims[0];
            const unsigned int oy = (area.y - sy) / mStrideDims[1];

            for (unsigned int output = 0; output < getNbOutputs(); ++output) {
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

                    const unsigned int sxMax = std::min(mInputsDims[0] -
           ox*mStrideDims[0], mPoolDims[0]);
                    const unsigned int syMax = std::min(mInputsDims[1] -
           oy*mStrideDims[1], mPoolDims[1]);

                    for (unsigned int channel = 0; channel < getNbChannels();
           ++channel) {
                        if (!isConnection(channel, output))
                            continue;

                        for (unsigned int sy = 0; sy < syMax; ++sy) {
                            for (unsigned int sx = 0; sx < sxMax; ++sx) {
                                const unsigned int ix = ox*mStrideDims[0] + sx;
                                const unsigned int iy = oy*mStrideDims[1] + sy;

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
        const unsigned int inputSize = mInputsDims[1] * mInputsDims[0];
        const unsigned int inputIdx = (area.x + mInputsDims[0] * area.y)
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
        const unsigned int poolSize = mPoolDims[1] * mPoolDims[0]
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
        mInputsActivity.assign(mInputsActivity.dims(), 0);

        mInputMax.assign(mInputMax.dims(), -1);
        mPoolActivity.assign(mPoolActivity.dims(),
                             (mPooling == Max) ? std::numeric_limits<int>::min()
                                               : 0);
        mOutputsActivity.assign(mOutputsActivity.dims(), 0);
    } else if (notify == Load)
        load(mNet.getLoadSavePath());
    else if (notify == Save)
        save(mNet.getLoadSavePath());
}
