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

#include "NodeNeuron_Reflective.hpp"

N2D2::NodeNeuron_Reflective::NodeNeuron_Reflective(Network& net)
    : NodeNeuron(net),
      // IMPORTANT: Do not change the value of the parameters here! Use
      // setParameter() or loadParameters().
      mIncomingDelay(this, "IncomingDelay", 1 * TimePs, 100 * TimeFs),
      mWeightsMin(this, "WeightsMin", 1, 0.1),
      mWeightsMax(this, "WeightsMax", 100, 10.0),
      mWeightsInit(this, "WeightsInit", 80, 8.0),
      mWeightIncrement(this, "WeightIncrement", 50, 5.0),
      mWeightIncrementDamping(this, "WeightIncrementDamping", -3.0),
      mWeightDecrement(this, "WeightDecrement", 50, 5.0),
      mWeightDecrementDamping(this, "WeightDecrementDamping", -3.0),
      mThreshold(this, "Threshold", 10000.0),
      mEmitDelay(this, "EmitDelay", 100 * TimePs, 1 * TimePs),
      mBackwardThreshold(this, "BackwardThreshold", 10000.0),
      mBackwardStdpLtp(this, "BackwardStdpLtp", 1000 * TimePs),
      mBackwardLeak(this, "BackwardLeak", 0 * TimeS),
      mBackwardRefractory(this, "BackwardRefractory", 0 * TimeS),
      mBackwardInhibitRefractory(this, "BackwardInhibitRefractory", 0 * TimeS),
      mBackwardSubActivity(this, "BackwardSubActivity", 0.0),
      mBackwardEchoMeanPeriod(this, "BackwardEchoMeanPeriod", 0 * TimeS),
      mBackwardEchoMeanDuration(this, "BackwardEchoMeanDuration", 0 * TimeS),
      mBackwardSubEventStdp(this, "BackwardSubEventStdp", false),
      mBackwardEchoEventStdp(this, "BackwardEchoEventStdp", false),
      mBackwardPropagation(this, "BackwardPropagation", None),
      mStdpLearning(this, "StdpLearning", Forward),
      // private parameters
      mStdpLtp(this, "StdpLtp", 1000 * TimePs),
      mLeak(this, "Leak", 0 * TimePs),
      mRefractory(this, "Refractory", 0 * TimeS),
      mInhibitRefractory(this, "InhibitRefractory", 0 * TimeS),
      mSubActivity(this, "SubActivity", 0.0),
      mEchoMeanPeriod(this, "EchoMeanPeriod", 0 * TimeS),
      mEchoMeanDuration(this, "EchoMeanDuration", 0 * TimeS),
      mSubEventStdp(this, "SubEventStdp", false),
      mEchoEventStdp(this, "EchoEventStdp", false),
      mForwardPropagation(this, "ForwardPropagation", Forward),
      mShareIntegration(this, "ShareIntegration", false),
      // Internal variables
      mIntegration(0.0),
      mLastSpikeTime(0),
      mEvent(NULL),
      mRefractoryEnd(0),
      mEchoEnd(0),
      mBackwardRefractoryEnd(0),
      mBackwardIntegration(0.0),
      mBackwardLastSpikeTime(0),
      mBackwardEvent(NULL),
      mBackwardEchoEnd(0)
{
    // ctor
}

N2D2::Synapse* N2D2::NodeNeuron_Reflective::newSynapse() const
{
    return new Synapse_Behavioral(mIncomingDelay.spreadNormal(0),
                                  mWeightsMin.spreadNormal(0),
                                  mWeightsMax.spreadNormal(0),
                                  mWeightIncrement.spreadNormal(),
                                  mWeightIncrementDamping.spreadNormal(),
                                  mWeightDecrement.spreadNormal(),
                                  mWeightDecrementDamping.spreadNormal(),
                                  mWeightsInit.spreadNormal(0));
}

void N2D2::NodeNeuron_Reflective::propagateSpike(Node* origin,
                                                 Time_T timestamp,
                                                 EventType_T type)
{
    Synapse_Behavioral* synapse = NULL;

    if (type == ForwardEvent)
        synapse = static_cast<Synapse_Behavioral*>(mLinks[origin]);
    else if (type == BackwardEvent) {
        NodeNeuron_Reflective* reflective = static_cast
            <NodeNeuron_Reflective*>(origin);
        synapse = static_cast<Synapse_Behavioral*>(reflective->mLinks[this]);
    } else
        throw std::runtime_error(
            "Unexpected incoming event type for reflective node!");

    const Time_T delay = synapse->delay;

    if (delay > 0)
        mNet.newEvent(origin, this, timestamp + delay, type);
    else
        incomingSpike(origin, timestamp, type);
}

void N2D2::NodeNeuron_Reflective::incomingSpike(Node* origin,
                                                Time_T timestamp,
                                                EventType_T type)
{
    const bool forward = (type == ForwardEvent);

    Time_T& lastSpikeTime = (forward) ? mLastSpikeTime : mBackwardLastSpikeTime;
    Parameter<Time_T>& leak = (forward || mShareIntegration) ? mLeak
                                                             : mBackwardLeak;
    double& integration = (forward || mShareIntegration) ? mIntegration
                                                         : mBackwardIntegration;
    Time_T& refractoryEnd = (forward) ? mRefractoryEnd : mBackwardRefractoryEnd;
    ParameterWithSpread<double>& threshold = (forward) ? mThreshold
                                                       : mBackwardThreshold;
    SpikeEvent*& event = (forward) ? mEvent : mBackwardEvent;
    Parameter<double>& subActivity = (forward) ? mSubActivity
                                               : mBackwardSubActivity;

    Synapse_Behavioral* synapse = NULL;

    if (type == ForwardEvent)
        synapse = static_cast<Synapse_Behavioral*>(mLinks[origin]);
    else if (type == BackwardEvent) {
        NodeNeuron_Reflective* reflective = static_cast
            <NodeNeuron_Reflective*>(origin);
        synapse = static_cast<Synapse_Behavioral*>(reflective->mLinks[this]);
    } else
        throw std::runtime_error(
            "Unexpected incoming event type for reflective node!");

    const Time_T dt = timestamp - lastSpikeTime;

    // Integrates
    if (leak > 0) {
        const double expVal = -((double)dt) / ((double)leak);

        if (expVal > std::log(1e-20))
            integration *= std::exp(expVal);
        else {
            integration = 0.0;
            std::cout << "Notice: integration leaked to 0 (no activity during "
                      << dt / ((double)TimeS) << " s = " << (-expVal)
                      << " * leak)." << std::endl;
        }
    }

    if (timestamp >= refractoryEnd)
        integration += synapse->weight;

    lastSpikeTime = timestamp;

    if (mStateLog.is_open()) {
        mStateLog << timestamp / ((double)TimeS) << " " << integration << " "
                  << (int)(forward || mShareIntegration) << "\n";
    }

    bool fire = (integration >= threshold);

    if (!fire && subActivity > 0) {
        fire = (std::pow(integration / threshold, subActivity)
                > Random::randUniform());
        type = (type == ForwardEvent) ? ForwardSubEvent : BackwardSubEvent;
    }

    if (timestamp >= refractoryEnd && fire) {
        if (event != NULL
            && ((event->getType() == ForwardEchoEvent && type == ForwardEvent)
                || (event->getType() == BackwardEchoEvent
                    && type == BackwardEvent))) {
            event->discard();
            event = NULL;
        }

        if (event == NULL) {
            // Fires!
            if (mEmitDelay > 0)
                event = mNet.newEvent(this, NULL, timestamp + mEmitDelay, type);
            else
                emitSpike(timestamp, type);
        }
    }
}

void N2D2::NodeNeuron_Reflective::emitSpike(Time_T timestamp, EventType_T type)
{
    const bool forward = (type == ForwardEvent || type == ForwardSubEvent
                          || type == ForwardEchoEvent);

    double& integration = (forward || mShareIntegration) ? mIntegration
                                                         : mBackwardIntegration;
    Time_T& refractoryEnd = (forward) ? mRefractoryEnd : mBackwardRefractoryEnd;
    Parameter<Time_T>& refractory = (forward) ? mRefractory
                                              : mBackwardRefractory;
    SpikeEvent*& event = (forward) ? mEvent : mBackwardEvent;
    Parameter<Time_T>& stdpLtp = (forward) ? mStdpLtp : mBackwardStdpLtp;
    Parameter<Time_T>& echoMeanPeriod = (forward) ? mEchoMeanPeriod
                                                  : mBackwardEchoMeanPeriod;
    Parameter<Time_T>& echoMeanDuration = (forward) ? mEchoMeanDuration
                                                    : mBackwardEchoMeanDuration;
    Time_T& echoEnd = (forward) ? mEchoEnd : mBackwardEchoEnd;

    if ((type != ForwardSubEvent || mSubEventStdp)
        && (type != BackwardSubEvent || mBackwardSubEventStdp)
        && (type != ForwardEchoEvent || mEchoEventStdp)
        && (type != BackwardEchoEvent || mBackwardEchoEventStdp)) {
        integration = 0.0; // Reset interation
        refractoryEnd = timestamp + refractory;
    }

    if ((mStdpLearning == Forward || mStdpLearning == Both)
        && (type == ForwardEvent || (type == ForwardSubEvent && mSubEventStdp)
            || (type == ForwardEchoEvent && mEchoEventStdp))) {
        for (std::unordered_map<Node*, Synapse*>::const_iterator it
             = mLinks.begin(),
             itEnd = mLinks.end();
             it != itEnd;
             ++it) {
            Synapse_Behavioral* synapse = static_cast
                <Synapse_Behavioral*>((*it).second);

            if ((*it).first->getLastActivationTime() > 0
                && (*it).first->getLastActivationTime() + stdpLtp >= timestamp)
                increaseWeight(synapse);
            else
                decreaseWeight(synapse);
        }
    } else if ((mStdpLearning == Backward || mStdpLearning == Both)
               && (type == BackwardEvent
                   || (type == BackwardSubEvent && mBackwardSubEventStdp)
                   || (type == BackwardEchoEvent && mBackwardEchoEventStdp))) {
        for (std::vector<Node*>::const_iterator it = mBranches.begin(),
                                                itEnd = mBranches.end();
             it != itEnd;
             ++it) {
            NodeNeuron_Reflective* reflective = static_cast
                <NodeNeuron_Reflective*>((*it));
            Synapse_Behavioral* synapse = static_cast
                <Synapse_Behavioral*>(reflective->mLinks[this]);

            if (reflective->getLastActivationTime() > 0
                && reflective->getLastActivationTime() + stdpLtp >= timestamp)
                increaseWeight(synapse);
            else
                decreaseWeight(synapse);
        }
    }

    // Lateral inhibition
    if ((type != ForwardSubEvent || mSubEventStdp)
        && (type != BackwardSubEvent || mBackwardSubEventStdp)
        && (type != ForwardEchoEvent || mEchoEventStdp)
        && (type != BackwardEchoEvent || mBackwardEchoEventStdp)) {
        for (std::vector<NodeNeuron*>::const_iterator it
             = mLateralBranches.begin(),
             itEnd = mLateralBranches.end();
             it != itEnd;
             ++it) {
            NodeNeuron_Reflective* reflective = static_cast
                <NodeNeuron_Reflective*>((*it));
            reflective->lateralInhibition(timestamp, type);
        }
    }

    if ((forward
         && (mForwardPropagation == Forward || mForwardPropagation == Both))
        || (!forward && (mBackwardPropagation == Forward || mBackwardPropagation
                                                            == Both))) {
        Node::emitSpike(timestamp, ForwardEvent);
    }

    if ((forward
         && (mForwardPropagation == Backward || mForwardPropagation == Both))
        || (!forward && (mBackwardPropagation == Backward
                         || mBackwardPropagation == Both))) {
        if (mActivityRecording)
            mNet.recordSpike(mId, timestamp, BackwardEvent);

        mLastActivationTime = timestamp;

        // If this is in fact a NodeNeuron_ReflectiveBridge, mLinks is empty, so
        // no event is actually created but the activation of
        // the neuron is still reported.
        for (std::unordered_map<Node*, Synapse*>::const_iterator it
             = mLinks.begin(),
             itEnd = mLinks.end();
             it != itEnd;
             ++it)
            (*it).first->propagateSpike(this, timestamp, BackwardEvent);
    }

    event = NULL;

    if (echoMeanPeriod > 0 && echoMeanDuration > 0) {
        const EventType_T eventType = (forward) ? ForwardEchoEvent
                                                : BackwardEchoEvent;

        if (type != eventType || timestamp + echoMeanPeriod < echoEnd)
            event = mNet.newEvent(
                this, NULL, timestamp + echoMeanPeriod + mEmitDelay, eventType);

        if (type != eventType && echoEnd < timestamp)
            echoEnd = timestamp + echoMeanDuration;
    }
}

void N2D2::NodeNeuron_Reflective::lateralInhibition(Time_T timestamp,
                                                    EventType_T type)
{
    const bool forward = (type == ForwardEvent || type == ForwardSubEvent
                          || type == ForwardEchoEvent);

    Parameter<Time_T>& inhibitRefractory
        = (forward) ? mInhibitRefractory : mBackwardInhibitRefractory;
    Time_T& refractoryEnd = (forward) ? mRefractoryEnd : mBackwardRefractoryEnd;
    SpikeEvent*& event = (forward) ? mEvent : mBackwardEvent;

    bool discardEvent = false;

    if (inhibitRefractory > 0) {
        if (refractoryEnd < timestamp + inhibitRefractory)
            refractoryEnd = timestamp + inhibitRefractory;

        discardEvent = true;
    }

    if (discardEvent && event != NULL) {
        event->discard();
        event = NULL;
    }
}

void N2D2::NodeNeuron_Reflective::reset(Time_T timestamp)
{
    mIntegration = 0.0;
    mLastSpikeTime = timestamp;
    mRefractoryEnd = 0;
    mEchoEnd = 0;

    mBackwardRefractoryEnd = 0;
    mBackwardIntegration = 0.0;
    mBackwardLastSpikeTime = 0;
    mBackwardEchoEnd = 0;
}

void N2D2::NodeNeuron_Reflective::initialize()
{
    if (mThreshold <= 0.0)
        throw std::domain_error("mThreshold is <= 0.0");

    if (!mInitializedState) {
        mEmitDelay.spreadNormal(0);
        mThreshold.spreadNormal(0);
        mBackwardThreshold.spreadNormal(0);
        mInitializedState = true;
    }

    if (mStateLog.is_open()) {
        mStateLog << 0.0 << " " << mIntegration << " " << (int)true << "\n";

        if (!mShareIntegration) {
            mStateLog << mBackwardLastSpikeTime / ((double)TimeS) << " "
                      << mBackwardIntegration << " " << (int)false << "\n";
        }
    }
}

void N2D2::NodeNeuron_Reflective::increaseWeight(Synapse_Behavioral
                                                 * synapse) const
{
    if (synapse->weight < synapse->weightMax) {
        const double dw
            = synapse->weightIncrement
              * std::exp(synapse->weightIncrementDamping
                         * (synapse->weight - synapse->weightMin)
                         / (synapse->weightMax - synapse->weightMin));

        if (synapse->weight < synapse->weightMax - dw)
            synapse->weight += dw;
        else
            synapse->weight = synapse->weightMax;
    }
}

void N2D2::NodeNeuron_Reflective::decreaseWeight(Synapse_Behavioral
                                                 * synapse) const
{
    if (synapse->weight > synapse->weightMin) {
        const double dw
            = synapse->weightDecrement
              * std::exp(synapse->weightDecrementDamping
                         * (synapse->weightMax - synapse->weight)
                         / (synapse->weightMax - synapse->weightMin));

        if (synapse->weight > synapse->weightMin + dw)
            synapse->weight -= dw;
        else
            synapse->weight = synapse->weightMin;
    }
}

void N2D2::NodeNeuron_Reflective::saveInternal(std::ofstream& dataFile) const
{
    // The Parameters instances should not be saved here! (June 30,2011, Damien)
    dataFile.write(reinterpret_cast<const char*>(&mIntegration),
                   sizeof(mIntegration));
    dataFile.write(reinterpret_cast<const char*>(&mLastSpikeTime),
                   sizeof(mLastSpikeTime));
    dataFile.write(reinterpret_cast<const char*>(&mRefractoryEnd),
                   sizeof(mRefractoryEnd));
    dataFile.write(reinterpret_cast<const char*>(&mEchoEnd), sizeof(mEchoEnd));
    dataFile.write(reinterpret_cast<const char*>(&mBackwardRefractoryEnd),
                   sizeof(mBackwardRefractoryEnd));
    dataFile.write(reinterpret_cast<const char*>(&mBackwardIntegration),
                   sizeof(mBackwardIntegration));
    dataFile.write(reinterpret_cast<const char*>(&mBackwardLastSpikeTime),
                   sizeof(mBackwardLastSpikeTime));
    dataFile.write(reinterpret_cast<const char*>(&mBackwardEchoEnd),
                   sizeof(mBackwardEchoEnd));

    if (!dataFile.good())
        throw std::runtime_error(
            "NodeNeuron_Reflective::saveInternal(): error writing data");
}

void N2D2::NodeNeuron_Reflective::loadInternal(std::ifstream& dataFile)
{
    dataFile.read(reinterpret_cast<char*>(&mIntegration), sizeof(mIntegration));
    dataFile.read(reinterpret_cast<char*>(&mLastSpikeTime),
                  sizeof(mLastSpikeTime));
    dataFile.read(reinterpret_cast<char*>(&mRefractoryEnd),
                  sizeof(mRefractoryEnd));
    dataFile.read(reinterpret_cast<char*>(&mEchoEnd), sizeof(mEchoEnd));
    dataFile.read(reinterpret_cast<char*>(&mBackwardRefractoryEnd),
                  sizeof(mBackwardRefractoryEnd));
    dataFile.read(reinterpret_cast<char*>(&mBackwardIntegration),
                  sizeof(mBackwardIntegration));
    dataFile.read(reinterpret_cast<char*>(&mBackwardLastSpikeTime),
                  sizeof(mBackwardLastSpikeTime));
    dataFile.read(reinterpret_cast<char*>(&mBackwardEchoEnd),
                  sizeof(mBackwardEchoEnd));

    if (!dataFile.good())
        throw std::runtime_error(
            "NodeNeuron_Reflective::loadInternal(): error reading data");
}

void N2D2::NodeNeuron_Reflective::logStatePlot()
{
    std::ostringstream plotCmd;
    plotCmd << "using 1:($3==1 ? $2 : 1/0) title \"forward\" with steps, "
            << "\"\" using 1:($3==1 ? $2 : 1/0) notitle with points lt 4 pt 4, "
            << ((double)mThreshold)
            << " title \"forward threshold\" with lines, "
            << "\"\" using 1:($3==0 ? $2 : 1/0) title \"backward\" with steps, "
            << "\"\" using 1:($3==0 ? $2 : 1/0) notitle with points lt 4 pt 4, "
            << ((double)mBackwardThreshold)
            << " title \"backward threshold\" with lines";

    Gnuplot gnuplot;
    gnuplot.set("grid").set("key off");
    gnuplot.set("pointsize", 0.2);
    gnuplot.setXlabel("Time (s)");
    gnuplot.saveToFile(mStateLogFile);
    gnuplot.plot(mStateLogFile, plotCmd.str());
}

N2D2::NodeNeuron_ReflectiveBridge::NodeNeuron_ReflectiveBridge(Network& net)
    : NodeNeuron_Reflective(net), mLink(NULL)
{
    // ctor
    mStdpLearning = None;
}

void N2D2::NodeNeuron_ReflectiveBridge::addLink(Node* origin)
{
    if (mLink != NULL)
        throw std::logic_error(
            "A NodeNeuron_ReflectiveBridge object can only have one link.");

    mLink = origin;
    origin->addBranch(this);

    mScale = origin->getScale();
    mOrientation = origin->getOrientation();
    mArea = origin->getArea();
    mLayer = origin->getLayer() + 1;
}

void N2D2::NodeNeuron_ReflectiveBridge::propagateSpike(Node* origin,
                                                       Time_T timestamp,
                                                       EventType_T type)
{
    incomingSpike(origin, timestamp, type);
}

void N2D2::NodeNeuron_ReflectiveBridge::incomingSpike(Node* origin,
                                                      Time_T timestamp,
                                                      EventType_T type)
{
    if (type == 0)
        // Event coming from the non-reflective parent node, forward it to the
        // reflective node
        Node::emitSpike(timestamp, ForwardEvent);
    else
        // Reflective event
        NodeNeuron_Reflective::incomingSpike(origin, timestamp, type);
}

void N2D2::NodeNeuron_ReflectiveBridge::initialize()
{
    if (mStdpLearning == Forward || mStdpLearning == Both)
        throw std::runtime_error("Cannot perform forward STDP learning on a "
                                 "reflective bridge node!");

    NodeNeuron_Reflective::initialize();
}
