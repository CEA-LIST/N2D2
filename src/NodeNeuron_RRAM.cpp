/*
    (C) Copyright 2012 CEA LIST. All Rights Reserved.
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

#include "NodeNeuron_RRAM.hpp"

N2D2::NodeNeuron_RRAM::NodeNeuron_RRAM(Network& net)
    : NodeNeuron(net),
      // IMPORTANT: Do not change the value of the parameters here! Use
      // setParameter() or loadParameters().
      mIncomingDelay(this, "IncomingDelay", 1 * TimePs, 100 * TimeFs),
      mWeightsMinMean(this, "WeightsMinMean", 1, 0.1),
      mWeightsMaxMean(this, "WeightsMaxMean", 100, 10.0),
      mWeightsInitMean(this, "WeightsInitMean", 80),
      mWeightsMinVar(this, "WeightsMinVar", 0.0),
      mWeightsMaxVar(this, "WeightsMaxVar", 0.0),
      mWeightsMinVarSlope(this, "WeightsMinVarSlope", 0.0),
      mWeightsMinVarOrigin(this, "WeightsMinVarOrigin", 0.0),
      mWeightsMaxVarSlope(this, "WeightsMaxVarSlope", 0.0),
      mWeightsMaxVarOrigin(this, "WeightsMaxVarOrigin", 0.0),
      mWeightsSetProba(this, "WeightsSetProba", 1.0),
      mWeightsResetProba(this, "WeightsResetProba", 1.0),
      mThreshold(this, "Threshold", 10000.0),
      mEmitDelay(this, "EmitDelay", 100 * TimePs, 1 * TimePs),
      mFireOnlyOnce(this, "FireOnlyOnce", false),
      mStdpLtp(this, "StdpLtp", 1000 * TimePs),
      mLeak(this, "Leak", 0 * TimeS),
      mRefractory(this, "Refractory", 0 * TimeS),
      mInhibitRefractory(this, "InhibitRefractory", 0 * TimeS),
      mEnableStdp(this, "EnableStdp", true),
      mInhibitIntegration(this, "InhibitIntegration", 0.0),
      mSynapticRedundancy(this, "SynapticRedundancy", 1U),
      mLtpProba(this, "LtpProba", 0.2),
      mLtdProba(this, "LtdProba", 0.1),
      // Internal variables
      mSynapseType(CBRAM),
      mIntegration(0.0),
      mAllowFire(true),
      mLastSpikeTime(0),
      mEvent(NULL),
      mRefractoryEnd(0)
{
    // ctor
}

N2D2::Synapse* N2D2::NodeNeuron_RRAM::newSynapse() const
{
    if (mSynapseType == CBRAM) {
        return new Synapse_RRAM(false,
                                mSynapticRedundancy,
                                mIncomingDelay.spreadNormal(0),
                                mWeightsMinMean,
                                mWeightsMaxMean,
                                mWeightsMinVar,
                                mWeightsMaxVar,
                                mWeightsSetProba,
                                mWeightsResetProba,
                                mWeightsInitMean);
    } else {
        return new Synapse_RRAM(false,
                                mSynapticRedundancy,
                                mIncomingDelay.spreadNormal(0),
                                mWeightsMinMean,
                                mWeightsMaxMean,
                                mWeightsMinVarSlope,
                                mWeightsMinVarOrigin,
                                mWeightsMaxVarSlope,
                                mWeightsMaxVarOrigin,
                                mWeightsSetProba,
                                mWeightsResetProba,
                                mWeightsInitMean);
    }
}

void N2D2::NodeNeuron_RRAM::propagateSpike(Node* origin,
                                           Time_T timestamp,
                                           EventType_T type)
{
    const Time_T delay = static_cast<Synapse_RRAM*>(mLinks[origin])->delay;

    if (delay > 0)
        mNet.newEvent(origin, this, timestamp + delay, type);
    else
        incomingSpike(origin, timestamp, type);
}

void N2D2::NodeNeuron_RRAM::incomingSpike(Node* origin,
                                          Time_T timestamp,
                                          EventType_T /*type*/)
{
    Synapse_RRAM* synapse = static_cast<Synapse_RRAM*>(mLinks[origin]);

    // Stats
    ++synapse->statsReadEvents;

    for (unsigned int dev = 0; dev < mSynapticRedundancy; ++dev)
        synapse->stats[dev].statsReadEnergy += synapse->devices[dev].weight;

    const Time_T dt = timestamp - mLastSpikeTime;

    // Integrates
    if (mLeak > 0)
        mIntegration = mIntegration * std::exp(-((double)dt) / ((double)mLeak));

    // Mettre à jour l'intégration pour tenir compte de la fuite pendant la
    // période réfractaire, ou le faire uniquement à la fin
    // de la période réfractaire est strictement équivalent, comme le montre
    // l'égalité suivante :
    // I1 = I0*exp(-dt1/Tau)
    // I2 = I1*exp(-dt2/Tau) = I0*exp(-(dt1+dt2)/Tau)
    // Cependant, il est préférable de mettre à jour l'intégration à chaque
    // fois, plutôt que dans le "if" qui suit, dans le cas où
    // le seuil serait variable par exemple (si jamais il décroit avec le temps
    // et que l'on ne recalcule pas l'intégration durant
    // la période réfractaire, on ne tient pas compte de la fuite pendant ce
    // temps là, ce qui pourrait provoquer un déclenchement
    // erroné du neurone).
    if (timestamp >= mRefractoryEnd)
        mIntegration += synapse->getWeight() / mSynapticRedundancy;

    mLastSpikeTime = timestamp;

    if (mStateLog.is_open())
        mStateLog << timestamp / ((double)TimeS) << " " << mIntegration << "\n";

    if (timestamp >= mRefractoryEnd && mIntegration >= mThreshold && mAllowFire
        && mEvent == NULL) {
        // Fires!
        if (mEmitDelay > 0)
            mEvent
                = mNet.newEvent(this, NULL, timestamp + mEmitDelay, FireEvent);
        else
            emitSpike(timestamp, FireEvent);
    }
}

void N2D2::NodeNeuron_RRAM::emitSpike(Time_T timestamp, EventType_T type)
{
    if (type == 0) {
        // Default type means pass-through event (used to emulate the activity
        // of the neuron)
        Node::emitSpike(timestamp);
        return;
    }

    mIntegration = 0.0; // Reset interation
    mRefractoryEnd = timestamp + mRefractory;

    if (mEnableStdp) {
        for (std::unordered_map<Node*, Synapse*>::const_iterator it
             = mLinks.begin(),
             itEnd = mLinks.end();
             it != itEnd;
             ++it) {
            Synapse_RRAM* synapse = static_cast<Synapse_RRAM*>((*it).second);

            if ((*it).first->getLastActivationTime() > 0
                && (*it).first->getLastActivationTime() + mStdpLtp >= timestamp)
                increaseWeight(synapse);
            else
                decreaseWeight(synapse);
        }
    }

    if (mFireOnlyOnce)
        mAllowFire = false;

    // Lateral inhibition
    std::for_each(mLateralBranches.begin(),
                  mLateralBranches.end(),
                  std::bind(&NodeNeuron::lateralInhibition,
                            std::placeholders::_1,
                            timestamp, 0));

    Node::emitSpike(timestamp, type);
    mEvent = NULL;
}

void N2D2::NodeNeuron_RRAM::lateralInhibition(Time_T timestamp,
                                              EventType_T /*type*/)
{
    bool discardEvent = false;

    if (mInhibitIntegration > 0.0) {
        if (mIntegration > mInhibitIntegration)
            mIntegration -= mInhibitIntegration;
        else
            mIntegration = 0.0;

        discardEvent = true;
    }

    // Attention : si le neurone est déjà en période réfractaire (suite à
    // l'émission d'un spike), le fait de recevoir une inhibition
    // latérale d'un autre neurone ne doit pas diminuer cette période !
    if (mInhibitRefractory > 0) {
        if (mRefractoryEnd < timestamp + mInhibitRefractory)
            mRefractoryEnd = timestamp + mInhibitRefractory;

        discardEvent = true;
    }

    if (discardEvent && mEvent != NULL) {
        mEvent->discard();
        mEvent = NULL;
    }
}

void N2D2::NodeNeuron_RRAM::reset(Time_T timestamp)
{
    mIntegration = 0.0;
    mAllowFire = true;
    mLastSpikeTime = timestamp;
    mRefractoryEnd = 0;
}

void N2D2::NodeNeuron_RRAM::logWeights(const std::string& fileName) const
{
    std::ofstream data(fileName.c_str());

    if (!data.good())
        throw std::runtime_error("Could not create weights log file: "
                                 + fileName);

    for (std::unordered_map<Node*, Synapse*>::const_iterator it
         = mLinks.begin(),
         itEnd = mLinks.end();
         it != itEnd;
         ++it) {
        Synapse_RRAM* synapse = static_cast<Synapse_RRAM*>((*it).second);

        const Area& area = (*it).first->getArea();

        // MAP X Y WEIGHT TIMING
        data << (*it).first->getScale() << " " << area.x << " " << area.y << " "
             << (*it).second->getRelativeWeight() << " "
             << (*it).first->getLastActivationTime();

        for (unsigned int dev = 0; dev < mSynapticRedundancy; ++dev)
            data << " " << synapse->devices[dev].weight;

        data << "\n";
    }
}

void N2D2::NodeNeuron_RRAM::initialize()
{
    if (mThreshold <= 0.0)
        throw std::domain_error("mThreshold is <= 0.0");

    if (!mInitializedState) {
        mEmitDelay.spreadNormal(0);
        mThreshold.spreadNormal(0);
        mStdpLtp.spreadNormal(0);
        mLeak.spreadNormal(0);
        mRefractory.spreadNormal(0);
        mInhibitRefractory.spreadNormal(0);
        mInitializedState = true;
    }

    if (mStateLog.is_open())
        mStateLog << 0.0 << " " << mIntegration << "\n";
}

void N2D2::NodeNeuron_RRAM::increaseWeight(Synapse_RRAM* synapse) const
{
    for (unsigned int dev = 0, size = synapse->devices.size(); dev < size;
         ++dev) {
        Synapse_RRAM::Device& device = synapse->devices[dev];

        // Extrinsic switching probability
        if (Random::randUniform() <= mLtpProba) {
            // Intrinsic switching probability
            synapse->setPulse(device);
            ++synapse->stats[dev].statsSetEvents;
        }
    }
}

void N2D2::NodeNeuron_RRAM::decreaseWeight(Synapse_RRAM* synapse) const
{
    for (unsigned int dev = 0, size = synapse->devices.size(); dev < size;
         ++dev) {
        Synapse_RRAM::Device& device = synapse->devices[dev];

        // Extrinsic switching probability
        if (Random::randUniform() <= mLtdProba) {
            // Intrinsic switching probability
            synapse->resetPulse(device);
            ++synapse->stats[dev].statsResetEvents;
        }
    }
}

void N2D2::NodeNeuron_RRAM::logSynapticBehavior(const std::string& fileName,
                                                unsigned int nbSynapses,
                                                bool plot)
{
    std::ofstream data(fileName.c_str());

    if (!data.good())
        throw std::runtime_error("Could not create synaptic behavior log file: "
                                 + fileName);

    const double ltpProba = mLtpProba;
    const double ltdProba = mLtdProba;
    mLtpProba = 1.0;
    mLtdProba = 1.0;

    for (unsigned int n = 0; n < nbSynapses; ++n) {
        std::unique_ptr
            <Synapse_RRAM> synapse(static_cast<Synapse_RRAM*>(newSynapse()));

        for (unsigned int i = 0; i < 100; ++i) {
            decreaseWeight(synapse.get());
            const Weight_T w = synapse->getWeight();
            increaseWeight(synapse.get());
            data << (i + 1) << " " << w << " " << synapse->getWeight() << "\n";
        }
    }

    mLtpProba = ltpProba;
    mLtdProba = ltdProba;

    data.close();

    if (plot) {
        Gnuplot gnuplot;
        gnuplot.set("grid").set("key off");
        gnuplot.setXlabel("RESET/SET pulse number");
        gnuplot.setYlabel("Conductance (mS)");
        gnuplot.set("logscale y");
        gnuplot.saveToFile(fileName);
        gnuplot.plot(fileName,
                     "using 1:($2*1e3), \"\" using 1:($3*1e3) lt 1 lc 3");

        gnuplot.unset("logscale");
        gnuplot.saveToFile(fileName, "-linear");
        gnuplot.plot(fileName,
                     "using 1:($2*1e3), \"\" using 1:($3*1e3) lt 1 lc 3");
    }
}

void N2D2::NodeNeuron_RRAM::saveInternal(std::ofstream& dataFile) const
{
    // The Parameters instances should not be saved here! (June 30,2011, Damien)
    dataFile.write(reinterpret_cast<const char*>(&mIntegration),
                   sizeof(mIntegration));
    dataFile.write(reinterpret_cast<const char*>(&mAllowFire),
                   sizeof(mAllowFire));
    dataFile.write(reinterpret_cast<const char*>(&mLastSpikeTime),
                   sizeof(mLastSpikeTime));
    dataFile.write(reinterpret_cast<const char*>(&mRefractoryEnd),
                   sizeof(mRefractoryEnd));

    if (!dataFile.good())
        throw std::runtime_error(
            "NodeNeuron_RRAM::saveInternal(): error writing data");
}

void N2D2::NodeNeuron_RRAM::loadInternal(std::ifstream& dataFile)
{
    dataFile.read(reinterpret_cast<char*>(&mIntegration), sizeof(mIntegration));
    dataFile.read(reinterpret_cast<char*>(&mAllowFire), sizeof(mAllowFire));
    dataFile.read(reinterpret_cast<char*>(&mLastSpikeTime),
                  sizeof(mLastSpikeTime));
    dataFile.read(reinterpret_cast<char*>(&mRefractoryEnd),
                  sizeof(mRefractoryEnd));

    if (!dataFile.good())
        throw std::runtime_error(
            "NodeNeuron_RRAM::loadInternal(): error reading data");
}

void N2D2::NodeNeuron_RRAM::logStatePlot()
{
    std::ostringstream plotCmd;
    // plotCmd << "with steps, \"\" with points lt 4 pt 4, " << ((double)
    // mThreshold) << " with lines";
    // plotCmd << "using 1:2 with steps, \"\" using 1:3 with lines";
    plotCmd << "with steps, \"\" with points lt 4 pt 4";

    Gnuplot gnuplot;
    gnuplot.set("grid").set("key off");
    gnuplot.set("pointsize", 0.2);
    gnuplot.setXlabel("Time (s)");
    gnuplot.saveToFile(mStateLogFile);
    gnuplot.plot(mStateLogFile, plotCmd.str());
}
