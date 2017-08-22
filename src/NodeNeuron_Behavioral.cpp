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

#include "NodeNeuron_Behavioral.hpp"

N2D2::NodeNeuron_Behavioral::NodeNeuron_Behavioral(Network& net)
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
      mFireOnlyOnce(this, "FireOnlyOnce", false),
      mStdpLtp(this, "StdpLtp", 1000 * TimePs),
      mLeak(this, "Leak", 0 * TimeS),
      mRefractory(this, "Refractory", 0 * TimeS),
      mInhibitRefractory(this, "InhibitRefractory", 0 * TimeS),
      mInhibitIntegration(this, "InhibitIntegration", 0.0),
      mInhibitStdp(this, "InhibitStdp", 0U),
      mEnableStdp(this, "EnableStdp", true),
      mOrderStdp(this, "OrderStdp", 0U),
      mLinearLeak(this, "LinearLeak", false),
      mWeightBias(this, "WeightBias", 0.0),
      mStdpLtd(this, "StdpLtd", 0 * TimeS),
      mBiologicalStdp(this, "BiologicalStdp", false),
      // Internal variables
      mIntegration(0.0),
      mAllowFire(true),
      mAllowStdp(true),
      mInhibition(0),
      mLastSpikeTime(0),
      mEvent(NULL),
      mRefractoryEnd(0),
      mLastStdp(0)
{
    // ctor
}

N2D2::Synapse* N2D2::NodeNeuron_Behavioral::newSynapse() const
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

void N2D2::NodeNeuron_Behavioral::propagateSpike(Node* origin,
                                                 Time_T timestamp,
                                                 EventType_T type)
{
    const Time_T delay = static_cast
                         <Synapse_Behavioral*>(mLinks[origin])->delay;

    if (delay > 0)
        mNet.newEvent(origin, this, timestamp + delay, type);
    else
        incomingSpike(origin, timestamp, type);
}

void N2D2::NodeNeuron_Behavioral::incomingSpike(Node* origin,
                                                Time_T timestamp,
                                                EventType_T /*type*/)
{
    Synapse_Behavioral* synapse = static_cast
        <Synapse_Behavioral*>(mLinks[origin]);
    ++synapse->statsReadEvents;

    // LTP
    if (mEnableStdp && mOrderStdp > 0) {
        std::deque<Synapse_Behavioral*>::iterator it
            = std::find(mLtpFifo.begin(), mLtpFifo.end(), synapse);

        if (it == mLtpFifo.end()) {
            mLtpFifo.push_back(synapse);

            if (mLtpFifo.size() > mOrderStdp)
                mLtpFifo.pop_front();
        } else if (it != mLtpFifo.begin() + mLtpFifo.size() - 1) {
            mLtpFifo.erase(it);
            mLtpFifo.push_back(synapse);
        }
    }

    const Time_T dt = timestamp - mLastSpikeTime;

    // Integrates
    if (mLeak > 0) {
        const double leakVal = ((double)dt) / ((double)mLeak);

        if (mLinearLeak) {
            if (mIntegration > leakVal)
                mIntegration -= leakVal;
            else
                mIntegration = 0.0;
        } else {
            if (leakVal < std::log(1e20))
                mIntegration *= std::exp(-leakVal);
            else {
                mIntegration = 0.0;
                // std::cout << "Notice: integration leaked to 0 (no activity
                // during " << dt/((double) TimeS) << " s = "
                //    << leakVal << " * mLeak)." << std::endl;
            }
        }
    }

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
        mIntegration += synapse->weight;

    mLastSpikeTime = timestamp;

    if (mStateLog.is_open())
        mStateLog << timestamp / ((double)TimeS) << " " << mIntegration
                  << std::endl;

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

void N2D2::NodeNeuron_Behavioral::emitSpike(Time_T timestamp, EventType_T type)
{
    if (type == 0) {
        // Default type means pass-through event (used to emulate the activity
        // of the neuron)
        Node::emitSpike(timestamp);
        return;
    }

    mIntegration = 0.0; // Reset interation
    mRefractoryEnd = timestamp + mRefractory;

    if (mStateLog.is_open())
        mStateLog << timestamp / ((double)TimeS) << " " << mIntegration
                  << std::endl;

    if (mEnableStdp && mAllowStdp) {
        unsigned int ltp = 0;

        if (mOrderStdp > 0) {
            for (std::deque<Synapse_Behavioral*>::const_iterator it
                 = mLtpFifo.begin(),
                 itEnd = mLtpFifo.end();
                 it != itEnd;
                 ++it) {
                increaseWeight((*it), (*it)->weightIncrement);
                ++ltp;
            }

            for (std::unordered_map<Node*, Synapse*>::const_iterator it
                 = mLinks.begin(),
                 itEnd = mLinks.end();
                 it != itEnd;
                 ++it) {
                Synapse_Behavioral* synapse = static_cast
                    <Synapse_Behavioral*>((*it).second);

                if (std::find(mLtpFifo.begin(), mLtpFifo.end(), synapse)
                    == mLtpFifo.end())
                    decreaseWeight(synapse, synapse->weightDecrement);
            }
        } else {
            for (std::unordered_map<Node*, Synapse*>::const_iterator it
                 = mLinks.begin(),
                 itEnd = mLinks.end();
                 it != itEnd;
                 ++it) {
                Synapse_Behavioral* synapse = static_cast
                    <Synapse_Behavioral*>((*it).second);

                if (stdp(synapse,
                         (*it).first->getLastActivationTime(),
                         timestamp))
                    ++ltp;
            }
        }
        /*
                // DEBUG
                std::cout << "LTP (%) = " << 100.0*((double) ltp)/mLinks.size()
           << std::endl;
        */
        mLastStdp = timestamp;
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

void N2D2::NodeNeuron_Behavioral::lateralInhibition(Time_T timestamp,
                                                    EventType_T /*type*/)
{
    bool discardEvent = false;

    ++mInhibition;

    if (mInhibitStdp > 0 && mInhibition >= mInhibitStdp)
        mAllowStdp = false;

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

void N2D2::NodeNeuron_Behavioral::reset(Time_T timestamp)
{
    mIntegration = 0.0;
    mAllowFire = true;
    mAllowStdp = true;
    mInhibition = 0;
    mLastSpikeTime = timestamp;
    mRefractoryEnd = 0;

    if (mEnableStdp) {
        mLastStdp = 0;
        mLtpFifo.clear();
    }

    if (mStateLog.is_open())
        mStateLog << timestamp / ((double)TimeS) << " " << mIntegration
                  << std::endl;
}

void N2D2::NodeNeuron_Behavioral::initialize()
{
    if (mThreshold <= 0.0)
        throw std::domain_error("mThreshold is <= 0.0");

    if (mInhibitIntegration < 0.0)
        throw std::domain_error("mInhibitIntegration is < 0.0");

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
        mStateLog << 0.0 << " " << mIntegration << std::endl;
}

bool N2D2::NodeNeuron_Behavioral::stdp(Synapse_Behavioral* synapse,
                                       Time_T preTime,
                                       Time_T postTime) const
{
    double dw;

    if (mBiologicalStdp) {
        dw = mWeightBias;

        if (preTime > mLastStdp) {
            if (postTime - preTime < 20 * mStdpLtp)
                dw += (synapse->weightIncrement - mWeightBias)
                      * std::exp(-(double)(postTime - preTime) / mStdpLtp);

            if (mStdpLtd > 0 && preTime - mLastStdp < 20 * mStdpLtd)
                dw -= synapse->weightDecrement
                      * std::exp(-(double)(preTime - mLastStdp) / mStdpLtd);
        }
    } else {
        if (preTime > 0 && preTime + mStdpLtp >= postTime)
            dw = synapse->weightIncrement;
        else
            dw = -synapse->weightDecrement;
    }

    if (dw >= 0.0)
        increaseWeight(synapse, dw);
    else
        decreaseWeight(synapse, -dw);

    return (dw >= 0.0);
}

void N2D2::NodeNeuron_Behavioral::increaseWeight(Synapse_Behavioral* synapse,
                                                 double weightIncrement) const
{
    if (synapse->weight < synapse->weightMax) {
        double dw = weightIncrement
                    * std::exp(synapse->weightIncrementDamping
                               * (synapse->weight - synapse->weightMin)
                               / (synapse->weightMax - synapse->weightMin));

        if (synapse->weight < synapse->weightMax - dw)
            synapse->weight += dw;
        else
            synapse->weight = synapse->weightMax;
    }

    ++synapse->statsIncEvents;
}

void N2D2::NodeNeuron_Behavioral::decreaseWeight(Synapse_Behavioral* synapse,
                                                 double weightDecrement) const
{
    if (synapse->weight > synapse->weightMin) {
        double dw = weightDecrement
                    * std::exp(synapse->weightDecrementDamping
                               * (synapse->weightMax - synapse->weight)
                               / (synapse->weightMax - synapse->weightMin));

        if (synapse->weight > synapse->weightMin + dw)
            synapse->weight -= dw;
        else
            synapse->weight = synapse->weightMin;
    }

    ++synapse->statsDecEvents;
}

void N2D2::NodeNeuron_Behavioral::logSynapticBehavior(const std::string
                                                      & fileName,
                                                      unsigned int nbSynapses,
                                                      bool plot) const
{
    std::ofstream data(fileName.c_str());

    if (!data.good())
        throw std::runtime_error("Could not create synaptic behavior log file: "
                                 + fileName);

    std::vector<double> ltpProba;
    ltpProba.push_back(1.0);
    ltpProba.push_back(0.0);
    ltpProba.push_back(0.8);
    ltpProba.push_back(0.2);
    ltpProba.push_back(0.6);
    ltpProba.push_back(0.4);

    for (unsigned int n = 0; n < nbSynapses; ++n) {
        std::unique_ptr<Synapse_Behavioral> synapse(
            static_cast<Synapse_Behavioral*>(newSynapse()));
        unsigned int offset = 0;

        for (std::vector<double>::const_iterator it = ltpProba.begin(),
                                                 itEnd = ltpProba.end();
             it != itEnd;
             ++it) {
            for (unsigned int stdpEvent = 0; stdpEvent < 200; ++stdpEvent) {
                if (Random::randUniform() <= (*it))
                    increaseWeight(synapse.get(), synapse->weightIncrement);
                else
                    decreaseWeight(synapse.get(), synapse->weightDecrement);

                data << (offset + stdpEvent) << " "
                     << (*it) * synapse->weightMax << " " << synapse->weight
                     << " " << (*it) << " "
                     << synapse->weight / synapse->weightMax << std::endl;
            }

            offset += 200;
        }

        data << std::endl;
    }

    data.close();

    if (plot) {
        Gnuplot gnuplot;
        gnuplot.set("grid").set("key off");
        gnuplot.setXlabel("LTP/LTD");
        gnuplot.setYlabel("Weight");
        gnuplot.saveToFile(fileName);
        gnuplot.plot(fileName,
                     "using 1:3 with line,"
                     "\"\" using 1:2 with line");
        gnuplot.setYlabel("Weight (normalized)");
        gnuplot.saveToFile(fileName, "-normalized");
        gnuplot.plot(fileName,
                     "using 1:5 with line,"
                     "\"\" using 1:4 with line");
    }
}

void N2D2::NodeNeuron_Behavioral::logStdpBehavior(const std::string& fileName,
                                                  unsigned int nbPoints,
                                                  bool plot)
{
    std::ofstream data(fileName.c_str());

    if (!data.good())
        throw std::runtime_error("Could not create STDP behavior log file: "
                                 + fileName);

    std::unique_ptr<Synapse_Behavioral> synapse(
        static_cast<Synapse_Behavioral*>(newSynapse()));
    const Time_T timeInterval = std::max<Time_T>(mStdpLtp, mStdpLtd);

    for (unsigned int n = 0; n < nbPoints; ++n) {
        const double dt
            = Random::randUniform(-5.0 * timeInterval / (double)TimeS,
                                  5.0 * timeInterval / (double)TimeS);
        synapse->weight = synapse->weightMin
                          + (synapse->weightMax - synapse->weightMin)
                            * Random::randUniform();
        const Weight_T weight = synapse->getRelativeWeight();
        Time_T preTime, postTime;

        if (dt >= 0.0) {
            // PRE after POST
            mLastStdp = 0 * TimeS;
            preTime = mLastStdp + (Time_T)(dt * TimeS);
            postTime = 10 * timeInterval;
        } else {
            // POST after PRE
            mLastStdp = 0 * TimeS;
            preTime = 10 * timeInterval;
            postTime = preTime + (Time_T)((-dt) * TimeS);
        }

        stdp(synapse.get(), preTime, postTime);

        data << dt << " " << 100.0 * (synapse->getRelativeWeight() - weight)
             << " " << 100.0 * (synapse->getRelativeWeight() - weight) / weight
             << std::endl;
    }

    data.close();

    if (plot) {
        Gnuplot gnuplot;
        gnuplot.set("grid").set("key off");
        gnuplot.setXlabel("DT = tpre - tpost (s)");
        gnuplot.setYlabel("Relative weight change DW (%)");
        gnuplot.setYrange(-100, 100);
        gnuplot.saveToFile(fileName);

        std::ostringstream arrow;
        arrow << "arrow from " << -(mStdpLtp / (double)TimeS) << ",-100 to "
              << -(mStdpLtp / (double)TimeS) << ",100 as 1";
        gnuplot.set("style arrow 1 nohead lt 2");
        gnuplot.set(arrow.str());

        if (mStdpLtd > 0) {
            arrow.str(std::string());
            arrow << "arrow from " << mStdpLtd / (double)TimeS << ",-100 to "
                  << mStdpLtd / (double)TimeS << ",100 as 2";
            gnuplot.set("style arrow 2 nohead lt 3");
            gnuplot.set(arrow.str());
        }

        gnuplot.plot(fileName, "using 1:3");
    }
}

void N2D2::NodeNeuron_Behavioral::saveInternal(std::ofstream& dataFile) const
{
    // The Parameters instances should not be saved here! (June 30,2011, Damien)
    dataFile.write(reinterpret_cast<const char*>(&mIntegration),
                   sizeof(mIntegration));
    dataFile.write(reinterpret_cast<const char*>(&mAllowFire),
                   sizeof(mAllowFire));
    dataFile.write(reinterpret_cast<const char*>(&mAllowStdp),
                   sizeof(mAllowStdp));
    dataFile.write(reinterpret_cast<const char*>(&mInhibition),
                   sizeof(mInhibition));
    dataFile.write(reinterpret_cast<const char*>(&mLastSpikeTime),
                   sizeof(mLastSpikeTime));
    dataFile.write(reinterpret_cast<const char*>(&mRefractoryEnd),
                   sizeof(mRefractoryEnd));
    dataFile.write(reinterpret_cast<const char*>(&mLastStdp),
                   sizeof(mLastStdp));

    if (!dataFile.good())
        throw std::runtime_error(
            "NodeNeuron_Behavioral::saveInternal(): error writing data");
}

void N2D2::NodeNeuron_Behavioral::loadInternal(std::ifstream& dataFile)
{
    dataFile.read(reinterpret_cast<char*>(&mIntegration), sizeof(mIntegration));
    dataFile.read(reinterpret_cast<char*>(&mAllowFire), sizeof(mAllowFire));
    dataFile.read(reinterpret_cast<char*>(&mAllowStdp), sizeof(mAllowStdp));
    dataFile.read(reinterpret_cast<char*>(&mInhibition), sizeof(mInhibition));
    dataFile.read(reinterpret_cast<char*>(&mLastSpikeTime),
                  sizeof(mLastSpikeTime));
    dataFile.read(reinterpret_cast<char*>(&mRefractoryEnd),
                  sizeof(mRefractoryEnd));
    dataFile.read(reinterpret_cast<char*>(&mLastStdp), sizeof(mLastStdp));

    if (!dataFile.good())
        throw std::runtime_error(
            "NodeNeuron_Behavioral::loadInternal(): error reading data");
}

void N2D2::NodeNeuron_Behavioral::logStatePlot()
{
    std::ostringstream plotCmd;
    plotCmd << "with steps, \"\" with points lt 4 pt 4, "
            << ((double)mThreshold) << " with lines";

    Gnuplot gnuplot;
    gnuplot.set("grid").set("key off");
    gnuplot.set("pointsize", 0.2);
    gnuplot.setXlabel("Time (s)");
    gnuplot.saveToFile(mStateLogFile);
    gnuplot.plot(mStateLogFile, plotCmd.str());
}
