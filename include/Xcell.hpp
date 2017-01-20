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

#ifndef N2D2_XCELL_H
#define N2D2_XCELL_H

#include <algorithm>
#include <fstream>
#include <functional>
#include <numeric>
#include <queue>
#include <string>
#include <vector>

#include "HeteroEnvironment.hpp"
#include "Network.hpp"
#include "NodeNeuron_Behavioral.hpp"
#include "NodeNeuron_PCM.hpp"
#include "NodeNeuron_RRAM.hpp"
#include "NodeNeuron_Reflective.hpp"
#include "NodeSync.hpp"

namespace N2D2 {
/**
 * This class represents a crossbar cell (Xcell). It can take other cells as
 * inputs (for multilayer networks), or it can be directly
 * connected to the environment. It is the basic building block of a network in
 * the currently envisionned architecture based on
 * memristive devices.
 * However, it is possible of course to create a full custom network without the
 * use of Xcells.
*/
class Xcell : public NetworkObserver, public Parameterizable {
public:
    enum InhibitionTopology {
        NoInhibition,
        Symmetric,
        Hierarchical
    };

    /**
     * Constructor. Attach the Xcell to a Network.
     *
     * @param net Network attached to this Xcell.
    */
    Xcell(Network& net);

    /**
     * Add lateral inhibition from another Xcell.
     * The inhibition is not symmetric: a neuron in @p cell will inhibit the
     *neurons of this Xcell, but the neurons in this
     * Xcell cannot inhibit neurons in @p cell. For symmetric inhibition between
     *cell1 and cell2, one should call this function
     * two times:
     * cell1.addLateralInhibition(cell2)        // cell2 inhibits cell1
     * cell2.addLateralInhibition(cell1)        // cell1 inhibits cell2
     *
     * @param cell      Xcell that can inhibit this cell.
    */
    void addLateralInhibition(Xcell& cell);

    /**
     * Populate the Xcell with @p nbNeurons neurons.
     * This template function also requires to provide the neuron class name to
     *be used for this Xcell. The Xcell object itself
     * is not a template and is unaware of the type of neuron it contains. It is
     *thus a generic container.
     *
     * @param nbNeurons         Number of neurons in this Xcell.
     * @param inhibition        Inhibition topology.
    */
    template <class T>
    void populate(unsigned int nbNeurons,
                  InhibitionTopology inhibition = Symmetric);

    void addInput(Node* node);

    /**
     * Add the output of an other Xcell as input (for multilayer network).
     * Each neuron of the input Xcell is connected to each neuron of the Xcell
     *(fully connected topology).
     *
     * @param cell Xcell to connect to.
    */
    void addInput(Xcell& cell);

    /**
     * Connect the Xcell to a 1-D environment (first layer).
     * The connection starts at offset @p x0 and ends at @p x0 + @p length
     *
     * @param env The Environment
     * @param filter Filter in the @p map to connect to
     * @param x0 Offset to start the connection (in number of NodeEnv)
     * @param length Size of the connection, starting from offset @p x0 (in
     *number of NodeEnv)
    */
    void addInput(Environment& env,
                  unsigned int filter,
                  unsigned int x0,
                  unsigned int length);

    /**
     * Connect the Xcell to a 2-D environment (first layer).
    */
    void addInput(Environment& env,
                  unsigned int filter,
                  unsigned int x0,
                  unsigned int y0,
                  unsigned int width,
                  unsigned int height);
    void addInput(Environment& env,
                  unsigned int x0,
                  unsigned int y0,
                  unsigned int width,
                  unsigned int height);
    void addMultiscaleInput(HeteroEnvironment& env,
                            unsigned int x0,
                            unsigned int y0,
                            unsigned int width,
                            unsigned int height);
    void addBridge(Environment& env,
                   unsigned int filter,
                   unsigned int x0,
                   unsigned int length);
    void addBridge(Environment& env,
                   unsigned int filter,
                   unsigned int x0,
                   unsigned int y0,
                   unsigned int width,
                   unsigned int height);
    void addBridge(Environment& env,
                   unsigned int x0,
                   unsigned int y0,
                   unsigned int width,
                   unsigned int height);
    void addMultiscaleBridge(HeteroEnvironment& env,
                             unsigned int x0,
                             unsigned int y0,
                             unsigned int width,
                             unsigned int height);
    inline void incomingSpike(NodeSync* node);
    void notify(Time_T timestamp, NotifyType notify);
    void readActivity(const std::string& fileName);
    void save(const std::string& dirName, bool cellStateOnly = false) const;
    void load(const std::string& dirName, bool cellStateOnly = false);
    void logState(const std::string& dirName,
                  bool append = false,
                  bool plot = true) const;
    void logStats(std::ofstream& dataFile) const;
    void logStats(const std::string& fileName) const;
    void clearStats();
    cv::Mat reconstructPattern(unsigned int neuron,
                               bool normalize = false,
                               bool multiLayer = true) const;
    void reconstructPatterns(const std::string& dirName,
                             bool normalize = false,
                             bool multiLayer = true) const;

    /**
     * If activityRecording is true, the spike activity of neurons from this
     * Xcell will be recorded. To retrieve the
     * results, use the method Network::getSpikeRecording()
    */
    void setActivityRecording(bool activityRecording);
    std::string getNeuronsParameter(const std::string& name) const;
    template <class T> T getNeuronsParameter(const std::string& name) const;
    template <class T>
    void setNeuronsParameter(const std::string& name, T value);
    template <class T>
    void
    setNeuronsParameter(const std::string& name, T mean, Percent relStdDev);
    template <class T>
    void setNeuronsParameter(const std::string& name, T mean, double stdDev);
    template <class T>
    void setNeuronsParameterSpread(const std::string& name,
                                   Percent relStdDev = Percent(0));
    template <class T>
    void setNeuronsParameterSpread(const std::string& name, double stdDev);
    void setNeuronsParameters(const std::map<std::string, std::string>& params,
                              bool ignoreUnknown = false);
    void loadNeuronsParameters(const std::string& fileName,
                               bool ignoreNotExists = false,
                               bool ignoreUnknown = false);
    void saveNeuronsParameters(const std::string& fileName) const;
    void copyNeuronsParameters(const Xcell& from);
    void copyNeuronsParameters(const NodeNeuron& from);
    const std::vector<NodeNeuron*>& getNeurons() const
    {
        return mNeurons;
    };
    virtual ~Xcell();

private:
    // A Xcell has an unique ID and is therefore non-copyable.
    Xcell(const Xcell&); // non construction-copyable
    const Xcell& operator=(const Xcell&); // non-copyable

    // Parameters
    Parameter<Time_T> mSyncClock;

    // Internal variables
    const XcellId_T mId;
    std::vector<NodeSync*> mSyncs;
    std::queue<NodeSync*> mSyncFifo;
    std::vector<NodeNeuron*> mNeurons;

    static unsigned int mIdCnt;
};
}

void N2D2::Xcell::incomingSpike(NodeSync* origin)
{
    mSyncFifo.push(origin);
}

template <class T>
void N2D2::Xcell::populate(unsigned int nbNeurons,
                           InhibitionTopology inhibition)
{
    mNeurons.reserve(nbNeurons);

    for (unsigned int i = 0; i < nbNeurons; ++i)
        mNeurons.push_back(new T(mNet));

    if (inhibition != NoInhibition) {
        // Add lateral inhibition branches
        for (unsigned int i = 0; i < nbNeurons; ++i) {
            for (unsigned int j = 0; j < nbNeurons; ++j) {
                if ((inhibition == Symmetric && i != j)
                    || (inhibition == Hierarchical && i > j))
                    mNeurons[i]->addLateralBranch(mNeurons[j]);
            }
        }
    }
}

template <class T>
T N2D2::Xcell::getNeuronsParameter(const std::string& name) const
{
    const T value = mNeurons.front()->getParameter<T>(name);

    if (std::find_if(mNeurons.begin(),
                     mNeurons.end(),
                     std::bind(std::not_equal_to<T>(),
                               value,
                               std::bind(&NodeNeuron::getParameter<T>,
                                         std::placeholders::_1,
                                         name))) != mNeurons.end()) {
        throw std::runtime_error(
            "Different values within the cell for neurons parameter: " + name);
    }

    return value;
}

template <class T>
void N2D2::Xcell::setNeuronsParameter(const std::string& name, T value)
{
    std::for_each(mNeurons.begin(),
                  mNeurons.end(),
                  std::bind(static_cast
                            <void (NodeNeuron::*)(const std::string&, T)>(
                                &NodeNeuron::setParameter<T>),
                            std::placeholders::_1,
                            name,
                            value));
}

template <class T>
void N2D2::Xcell::setNeuronsParameter(const std::string& name,
                                      T mean,
                                      Percent relStdDev)
{
    std::for_each(
        mNeurons.begin(),
        mNeurons.end(),
        std::bind(static_cast
                  <void (NodeNeuron::*)(const std::string&, T, Percent)>(
                      &NodeNeuron::setParameter<T>),
                  std::placeholders::_1,
                  name,
                  mean,
                  relStdDev));
}

template <class T>
void
N2D2::Xcell::setNeuronsParameter(const std::string& name, T mean, double stdDev)
{
    std::for_each(
        mNeurons.begin(),
        mNeurons.end(),
        std::bind(static_cast
                  <void (NodeNeuron::*)(const std::string&, T, double)>(
                      &NodeNeuron::setParameter<T>),
                  std::placeholders::_1,
                  name,
                  mean,
                  stdDev));
}

template <class T>
void N2D2::Xcell::setNeuronsParameterSpread(const std::string& name,
                                            Percent relStdDev)
{
    std::for_each(mNeurons.begin(),
                  mNeurons.end(),
                  std::bind(static_cast
                            <void (NodeNeuron::*)(const std::string&, Percent)>(
                                &NodeNeuron::setParameterSpread<T>),
                            std::placeholders::_1,
                            name,
                            relStdDev));
}

template <class T>
void N2D2::Xcell::setNeuronsParameterSpread(const std::string& name,
                                            double stdDev)
{
    std::for_each(mNeurons.begin(),
                  mNeurons.end(),
                  std::bind(static_cast
                            <void (NodeNeuron::*)(const std::string&, double)>(
                                &NodeNeuron::setParameterSpread<T>),
                            std::placeholders::_1,
                            name,
                            stdDev));
}

#endif // N2D2_XCELL_H
