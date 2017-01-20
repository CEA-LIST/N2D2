/*
    (C) Copyright 2010 CEA LIST. All Rights Reserved.
    Contributor(s): Olivier BICHLER (olivier.bichler@cea.fr)
                    Damien QUERLIOZ (damien.querlioz@cea.fr)

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

#ifndef N2D2_ENVIRONMENT_H
#define N2D2_ENVIRONMENT_H

#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "Network.hpp"
#include "NodeEnv.hpp"
#include "Sound.hpp"
#include "SpikeGenerator.hpp"
#include "StimuliProvider.hpp"
#include "Transformation/FilterTransformation.hpp"
#include "utils/Parameterizable.hpp"
#include "utils/Random.hpp"
#include "utils/Utils.hpp"

namespace N2D2 {
/**
 * This class is used to generate all kind of stimuli for the SNN.
 * It contains nodes of type NodeEnv, which constitute the input nodes of the
 * network.
*/
class Environment : public StimuliProvider, public SpikeGenerator {
public:
    Environment(Network& network,
                Database& database,
                unsigned int sizeX,
                unsigned int sizeY = 1,
                unsigned int nbChannels = 1,
                unsigned int batchSize = 1,
                bool compositeStimuli = false);
    virtual void addChannel(const CompositeTransformation& transformation);
    void propagate(Time_T start, Time_T end);
    // DEPRECATED: unsafe legacy function
    const std::shared_ptr<FilterTransformation>
    getFilter(unsigned int channel) const;
    // DEPRECATED
    void testFrame(unsigned int channel, Time_T start, Time_T end);

    // TODO/DEPRECATED
    // void readRandom(Time_T start, Time_T end);
    // void exportStimuli(const std::string& dirName, StimulusSet set = Learn)
    // const;
    // cv::Mat reconstructFrame(unsigned int index, StimulusSet set = Learn)
    // const;
    // cv::Mat reconstructMeanFrame(const std::string& className, bool normalize
    // = false) const;
    // void reconstructMeanFrames(const std::string& dirName) const;
    // double baselineTestCompare(const std::string& fileName) const;
    // void reconstructFilters(const std::string& dirName) const;

    // Getters
    Network& getNetwork()
    {
        return mNetwork;
    };
    inline NodeEnv* getNode(unsigned int channel,
                            unsigned int x,
                            unsigned int y = 0,
                            unsigned int batchPos = 0) const;
    inline NodeEnv* getNodeByIndex(unsigned int channel,
                                   unsigned int node,
                                   unsigned int batchPos = 0) const;
    inline const std::vector<NodeEnv*>
    getNodes(unsigned int channel, unsigned int batchPos = 0) const;
    inline const std::vector<NodeEnv*> getNodes() const;
    inline unsigned int getNbNodes() const;
    inline unsigned int getNbNodes(unsigned int channel,
                                   unsigned int batchPos = 0) const;
    virtual ~Environment();

protected:
    Network& mNetwork;
    /// For each scale, tensor (x, y, channel, batch)
    Tensor4d<NodeEnv*> mNodes;
};

// DEPRECATED: legacy special empty database
extern Database EmptyDatabase;
}

N2D2::NodeEnv* N2D2::Environment::getNode(unsigned int channel,
                                          unsigned int x,
                                          unsigned int y,
                                          unsigned int batchPos) const
{
    return mNodes(x, y, channel, batchPos);
}

N2D2::NodeEnv* N2D2::Environment::getNodeByIndex(unsigned int channel,
                                                 unsigned int node,
                                                 unsigned int batchPos) const
{
    return mNodes[batchPos][channel](node);
}

const std::vector<N2D2::NodeEnv*>
N2D2::Environment::getNodes(unsigned int channel, unsigned int batchPos) const
{
    const Tensor2d<NodeEnv*> nodes = mNodes[batchPos][channel];
    return std::vector<NodeEnv*>(nodes.begin(), nodes.end());
}

const std::vector<N2D2::NodeEnv*> N2D2::Environment::getNodes() const
{
    return mNodes.data();
}

unsigned int N2D2::Environment::getNbNodes() const
{
    return mNodes.size();
}

unsigned int N2D2::Environment::getNbNodes(unsigned int channel,
                                           unsigned int batchPos) const
{
    return mNodes[batchPos][channel].size();
}

#endif // N2D2_ENVIRONMENT_H
