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

#ifndef N2D2_CENVIRONMENT_H
#define N2D2_CENVIRONMENT_H

#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "Network.hpp"
#include "SpikeGenerator.hpp"
#include "StimuliProvider.hpp"
#include "utils/Parameterizable.hpp"
#include "utils/Random.hpp"
#include "utils/Utils.hpp"

namespace N2D2 {
class CEnvironment : public StimuliProvider, public SpikeGenerator {
public:
    CEnvironment(Database& database,
                 unsigned int sizeX,
                 unsigned int sizeY = 1,
                 unsigned int nbChannels = 1,
                 unsigned int batchSize = 1,
                 bool compositeStimuli = false);
    virtual void addChannel(const CompositeTransformation& transformation);
    void tick(Time_T timestamp, Time_T start, Time_T stop);
    void reset(Time_T timestamp);
    Tensor4d<char>& getTickData()
    {
        return mTickData;
    };
    const Tensor4d<char>& getTickData() const
    {
        return mTickData;
    };
    virtual ~CEnvironment();

protected:
    /// For each scale, tensor (x, y, channel, batch)
    bool mInitialized;
    Tensor4d<char> mTickData;
    Tensor4d<std::pair<Time_T, char> > mNextEvent;
};
}

#endif // N2D2_CENVIRONMENT_H
