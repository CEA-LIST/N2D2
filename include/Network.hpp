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

#ifndef N2D2_NETWORK_H
#define N2D2_NETWORK_H

#include <chrono>
#include <fstream>
#include <functional>
#include <queue>
#include <set>
#include <stack>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#ifndef WIN32
#include <fenv.h>
#endif

#if !defined(WIN32) && !defined(__APPLE__)
#include <csignal>
#include <cstring>
#include <execinfo.h>
#include <unistd.h>
#endif

#include "utils/Random.hpp"
#include "utils/Utils.hpp"

namespace N2D2 {
class SpikeEvent;
class Xcell;
class Node;
class NodeNeuron;
class Network;

typedef unsigned long long int Time_T; // Should be at least 64 bits (1 s =
// 1,000,000,000,000,000 fs)
extern const Time_T TimeFs;
extern const Time_T TimePs;
extern const Time_T TimeNs;
extern const Time_T TimeUs;
extern const Time_T TimeMs;
extern const Time_T TimeS;

typedef unsigned int NodeId_T;
typedef unsigned int XcellId_T;

typedef unsigned long long int EventType_T;
typedef std::vector<std::pair<Time_T, EventType_T> > NodeEvents_T;

#if !defined(WIN32) && !defined(__APPLE__)
void exceptionHandler(int sig);
#endif

class NetworkObserver {
public:
    enum NotifyType {
        Initialize,
        Finalize,
        Load,
        Save,
        Reset
    };

    NetworkObserver(Network& net);
    virtual void notify(Time_T timestamp, NotifyType notify) = 0;
    virtual ~NetworkObserver();

protected:
    /// Reference to the Network attached to this object.
    Network& mNet;
};

/**
 * This class is the heart of the simulator. It maintains a priority queue of
 *the events scheduled by the nodes of the
 * network.
 *
 * Each event is a N2D2::SpikeEvent object that is defined by its timestamp, its
 *origin node and its destination node.
 * The events are ordered by their timestamp in the priority queue, such that at
 *any point in the simulation, the next event to
 * be handled is always the one with the lowest timestamp (the one on top of the
 *priority queue). @n
 * A node is any object type derived from N2D2::Node. It can be a neuron, or a
 *environment node (N2D2::NodeEnv), which is an
 * input of the network and generate the input events. To handle high-level
 *stimuli, such as image, sound or AER recording,
 * environment nodes can be created with the N2D2::Environment class. @n
 * A node always contains a list of child nodes (branches) to which it can emit
 *spike events and at least two methods,
 * N2D2::Node::incomingSpike() and N2D2::Node::emitSpike(). When it receives an
 *event from a parent node,
 * N2D2::Node::incomingSpike() is called. This method never directly generates
 *events towards its child nodes. Instead,
 * it generates an internal event (with no destination node), so that when it is
 *processed, the N2D2::Node::emitSpike() method
 * is called. This method handles all the internal events created by the node
 *itself, either in the N2D2::Node::incomingSpike()
 * or any other method and create events to its child nodes.
*/
class Network {
public:
    /// Constructor.
    /// @param seed Seed for the random generator, used in any N2D2 function. If
    /// left to 0, a seed based on the system clock
    /// is produced. If the seed is set to a positive value, it is garanteed
    /// that the simulation will always produce the
    /// same results.
    Network(unsigned int seed = 0);
    /// Process all the events in the network until no further event remains in
    /// the priority queue.
    /// @param stop If not 0, stop the simulation to the specified timestamp.
    /// Usefull for debug purpose, or to stop network
    /// simulations containing oscillations.
    bool run(Time_T stop = 0, bool clearActivity = true);
    void stop(Time_T stop = 0, bool discard = false)
    {
        mStop = stop;
        mDiscard = discard;
    };
    void reset(Time_T timestamp = 0);
    /// Save the entire network state in a given location (binary format, not
    /// portable).
    void save(const std::string& dirName);
    /// Load the entire network state from a given location (binary format, not
    /// portable).
    void load(const std::string& dirName);
    const std::unordered_map<NodeId_T, NodeEvents_T>& getSpikeRecording()
    {
        return mSpikeRecording;
    };
    const NodeEvents_T& getSpikeRecording(NodeId_T nodeId)
    {
        return mSpikeRecording[nodeId];
    };
    /// Returns first processed event time after calling Network::run()
    Time_T getFirstEvent() const
    {
        return mFirstEvent;
    };
    /// Returns last processed event time after calling Network::run()
    Time_T getLastEvent() const
    {
        return mLastEvent;
    };
    const std::string& getLoadSavePath() const
    {
        return mLoadSavePath;
    };
    /// Destructor.
    virtual ~Network();

    static unsigned int readSeed(const std::string& fileName);

    // Internal functions, not to be called directly
    void addObserver(NetworkObserver* obs);
    void removeObserver(NetworkObserver* obs);
    /// Create a new event and add it to the priority queue.
    SpikeEvent* newEvent(Node* origin,
                         Node* destination,
                         Time_T timestamp,
                         EventType_T type = 0);
    inline void
    recordSpike(NodeId_T nodeId, Time_T timestamp = 0, EventType_T type = 0);

private:
    // Internal variables
    std::set<NetworkObserver*> mObservers;
    std::string mLoadSavePath;
    /// The priority queue containing the events to be processed by the
    /// simulator.
    std::priority_queue
        <SpikeEvent*, std::vector<SpikeEvent*>, Utils::PtrLess<SpikeEvent*> >
    mEvents;
    std::unordered_map<NodeId_T, NodeEvents_T> mSpikeRecording;
    bool mInitialized;
    Time_T mFirstEvent;
    Time_T mLastEvent;
    Time_T mStop;
    bool mDiscard;
    std::stack<SpikeEvent*> mEventsPool;
    const std::chrono::high_resolution_clock::time_point mStartTime;
};
}

void
N2D2::Network::recordSpike(NodeId_T nodeId, Time_T timestamp, EventType_T type)
{
    mSpikeRecording[nodeId].push_back(std::make_pair(timestamp, type));
}

#endif // N2D2_NETWORK_H
