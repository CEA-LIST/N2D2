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

#include "NodeNeuron_PCM.hpp"

N2D2::NodeNeuron_PCM::NodeNeuron_PCM(Network& net)
    : NodeNeuron(net),
      // IMPORTANT: Do not change the value of the parameters here! Use
      // setParameter() or loadParameters().
      mIncomingDelay(this, "IncomingDelay", 1 * TimePs, 100 * TimeFs),
      mWeightsMin(this, "WeightsMin", 1, 0.1),
      mWeightsMax(this, "WeightsMax", 100, 10.0),
      mWeightsInit(this, "WeightsInit", 80, 8.0),
      mWeightIncrement(this, "WeightIncrement", 50, 5.0),
      mWeightIncrementVar(this, "WeightIncrementVar", 0.0),
      mWeightIncrementDamping(this, "WeightIncrementDamping", -3.0),
      mWeightStochastic(this, "WeightStochastic", false),
      mThreshold(this, "Threshold", 10000.0),
      mEmitDelay(this, "EmitDelay", 100 * TimePs, 1 * TimePs),
      mFireOnlyOnce(this, "FireOnlyOnce", false),
      mStdpLtp(this, "StdpLtp", 1000 * TimePs),
      mLeak(this, "Leak", 0 * TimeS),
      mRefractory(this, "Refractory", 0 * TimeS),
      mInhibitRefractory(this, "InhibitRefractory", 0 * TimeS),
      mEnableStdp(this, "EnableStdp", true),
      mRelativeLtpStrength(this, "RelativeLtpStrength", 2.0),
      mWeightsUpdateLimit(this, "WeightsUpdateLimit", 20U),
      mWeightsRefreshMethod(this, "WeightsRefreshMethod", Truncate),
      // Internal variables
      mIntegration(0.0),
      mAllowFire(true),
      mLastSpikeTime(0),
      mEvent(NULL),
      mRefractoryEnd(0),
      mWeightUpdate(0)
{
    // ctor
}

N2D2::Synapse* N2D2::NodeNeuron_PCM::newSynapse() const
{
    const double weightThres = (mRelativeLtpStrength - 1.0)
                               * mWeightsMin.mean();
    const double weightMinAbs = mRelativeLtpStrength * mWeightsMin.mean()
                                - mWeightsMax.mean();
    const double weightMaxAbs = mRelativeLtpStrength * mWeightsMax.mean()
                                - mWeightsMin.mean();
    const double relWeight = (mWeightsInit.mean() - mWeightsMin.mean())
                             / (mWeightsMax.mean() - mWeightsMin.mean());
    const double weightAbs = weightMinAbs + relWeight
                                            * (weightMaxAbs - weightMinAbs);

    const double weightInitMean
        = (weightAbs >= weightThres)
              ? (weightAbs + mWeightsMin.mean()) / mRelativeLtpStrength
              : -(mRelativeLtpStrength * mWeightsMin.mean() - weightAbs);

    const Spread<Weight_T> weightsInit(weightInitMean, mWeightsInit.stdDev());

    return new Synapse_PCM(true,
                           1,
                           mIncomingDelay.spreadNormal(0),
                           mWeightsMin,
                           mWeightsMax,
                           mWeightIncrement,
                           mWeightIncrementVar,
                           mWeightIncrementDamping,
                           &mExperimentalLtpModel,
                           weightsInit.spreadNormal(0.0, 1.0));
}

void N2D2::NodeNeuron_PCM::propagateSpike(Node* origin,
                                          Time_T timestamp,
                                          EventType_T type)
{
    const Time_T delay = static_cast<Synapse_PCM*>(mLinks[origin])->delay;

    if (delay > 0)
        mNet.newEvent(origin, this, timestamp + delay, type);
    else
        incomingSpike(origin, timestamp, type);
}

void N2D2::NodeNeuron_PCM::incomingSpike(Node* origin,
                                         Time_T timestamp,
                                         EventType_T /*type*/)
{
    Synapse_PCM* synapse = static_cast<Synapse_PCM*>(mLinks[origin]);

    // Stats
    ++synapse->statsReadEvents;
    synapse->stats[LTP].statsReadEnergy += synapse->devices[LTP].weight;
    synapse->stats[LTD].statsReadEnergy += synapse->devices[LTD].weight;

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
    if (timestamp >= mRefractoryEnd) {
        mIntegration += (synapse->getWeight(mRelativeLtpStrength)
                         + mWeightsMax.mean() + mWeightsMin.mean())
                        / (1.0 + mRelativeLtpStrength);
    }

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

void N2D2::NodeNeuron_PCM::emitSpike(Time_T timestamp, EventType_T type)
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
        ++mWeightUpdate;

        for (std::unordered_map<Node*, Synapse*>::const_iterator it
             = mLinks.begin(),
             itEnd = mLinks.end();
             it != itEnd;
             ++it) {
            Synapse_PCM* synapse = static_cast<Synapse_PCM*>((*it).second);

            if ((*it).first->getLastActivationTime() > 0
                && (*it).first->getLastActivationTime() + mStdpLtp >= timestamp)
                increaseWeight(synapse, LTP);
            else
                increaseWeight(synapse, LTD);
        }

        if (mWeightUpdate == mWeightsUpdateLimit) {
            for (std::unordered_map<Node*, Synapse*>::iterator it
                 = mLinks.begin(),
                 itEnd = mLinks.end();
                 it != itEnd;
                 ++it) {
                Synapse_PCM* synapse = static_cast<Synapse_PCM*>((*it).second);

                if (mWeightsRefreshMethod == Nearest) {
                    // Optimal algorithm, but may require a lot of read/write
                    // cycles during the reset
                    double wRef = synapse->getWeight(mRelativeLtpStrength);

                    decreaseWeight(synapse, LTP);
                    decreaseWeight(synapse, LTD);
                    double wEq = synapse->getWeight(mRelativeLtpStrength);

                    if (wRef >= 0.0) {
                        // Only give it a limited number of try to reach the
                        // target equivalent weight, to avoid an infinite loop
                        // if
                        // for example the weight increment coefficient is 0 (=
                        // faulty device).
                        for (unsigned int i = 0;
                             wEq < wRef && i < mWeightsUpdateLimit;
                             ++i) {
                            increaseWeight(synapse, LTP);
                            wEq = synapse->getWeight(mRelativeLtpStrength);
                        }
                    } else {
                        // Same comment as above.
                        for (unsigned int i = 0;
                             wEq > wRef && i < mWeightsUpdateLimit;
                             ++i) {
                            increaseWeight(synapse, LTD);
                            wEq = synapse->getWeight(mRelativeLtpStrength);
                        }
                    }
                } else if (mWeightsRefreshMethod == Truncate) {
                    // Algorithm with no additional read/write cycle
                    if (mRelativeLtpStrength * synapse->devices[LTP].weight
                        > 2 * synapse->devices[LTD].weight)
                        decreaseWeight(synapse, LTD);
                    else if (2 * mRelativeLtpStrength
                             * synapse->devices[LTP].weight
                             < synapse->devices[LTD].weight)
                        decreaseWeight(synapse, LTP);
                    else {
                        decreaseWeight(synapse, LTP);
                        decreaseWeight(synapse, LTD);
                    }
                }
            }

            mWeightUpdate = 0;
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

void N2D2::NodeNeuron_PCM::lateralInhibition(Time_T timestamp,
                                             EventType_T /*type*/)
{
    bool discardEvent = false;

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

void N2D2::NodeNeuron_PCM::reset(Time_T timestamp)
{
    mIntegration = 0.0;
    mAllowFire = true;
    mLastSpikeTime = timestamp;
    mRefractoryEnd = 0;
}

void N2D2::NodeNeuron_PCM::initialize()
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

void N2D2::NodeNeuron_PCM::increaseWeight(Synapse_PCM* synapse,
                                          DeviceId dev) const
{
    Synapse_PCM::Device& device = synapse->devices[dev];

    synapse->setPulse(device);
    ++synapse->stats[dev].statsSetEvents;
}

void N2D2::NodeNeuron_PCM::decreaseWeight(Synapse_PCM* synapse,
                                          DeviceId dev) const
{
    Synapse_PCM::Device& device = synapse->devices[dev];

    if (mWeightStochastic) {
        device.weightMin = mWeightsMin.spreadNormal(0);
        device.weightMax = mWeightsMax.spreadNormal(0);
        device.weightIncrement = mWeightIncrement.spreadNormal(0);
        device.weightIncrementDamping = mWeightIncrementDamping.spreadNormal();
    }

    synapse->resetPulse(device);
    ++synapse->stats[dev].statsResetEvents;
}

void N2D2::NodeNeuron_PCM::logSynapticBehavior(const std::string& fileName,
                                               unsigned int nbSynapses,
                                               bool plot) const
{
    std::ofstream data(fileName.c_str());

    if (!data.good())
        throw std::runtime_error("Could not create synaptic behavior log file: "
                                 + fileName);

    for (unsigned int n = 0; n < nbSynapses; ++n) {
        std::unique_ptr
            <Synapse_PCM> synapse(static_cast<Synapse_PCM*>(newSynapse()));

        decreaseWeight(synapse.get(), LTP);
        decreaseWeight(synapse.get(), LTD);
        data << 0 << " " << synapse->devices[LTP].weight << " "
             << synapse->devices[LTD].weight << "\n";

        for (unsigned int i = 0; i < mWeightsUpdateLimit; ++i) {
            increaseWeight(synapse.get(), LTP);
            increaseWeight(synapse.get(), LTD);
            data << (i + 1) << " " << synapse->devices[LTP].weight << " "
                 << synapse->devices[LTD].weight << "\n";
        }

        decreaseWeight(synapse.get(), LTP);
        decreaseWeight(synapse.get(), LTD);
        data << (mWeightsUpdateLimit + 1) << " " << synapse->devices[LTP].weight
             << " " << synapse->devices[LTD].weight << "\n\n";
    }

    data.close();

    if (plot) {
        Gnuplot gnuplot;
        gnuplot.set("grid").set("key off");
        gnuplot.setXlabel("Pulse number");
        gnuplot.setYlabel("Conductance (S)");
        gnuplot.saveToFile(fileName);
        gnuplot.plot(fileName,
                     "using 1:2 with line,"
                     "\"\" using 1:3 with line");
    }
}

void N2D2::NodeNeuron_PCM::experimentalLtpModel(const std::string& fileName)
{
    std::ifstream data(fileName.c_str());

    if (!data.good())
        throw std::runtime_error("Could not load experimental LTP data file: "
                                 + fileName);

    std::copy(std::istream_iterator<double>(data),
              std::istream_iterator<double>(),
              std::back_inserter(mExperimentalLtpModel));

    // Normalize
    double wmin = mExperimentalLtpModel[0];
    double wmax = mExperimentalLtpModel[0];

    for (std::vector<double>::const_iterator it = mExperimentalLtpModel.begin(),
                                             itEnd
                                             = mExperimentalLtpModel.end();
         it != itEnd;
         ++it) {
        wmin = std::min(wmin, (*it));
        wmax = std::max(wmax, (*it));
    }

    std::transform(
        mExperimentalLtpModel.begin(),
        mExperimentalLtpModel.end(),
        mExperimentalLtpModel.begin(),
        std::bind(std::divides<double>(),
                  std::bind
                  <double>(std::minus<double>(), std::placeholders::_1, wmin),
                  wmax - wmin));
}

void N2D2::NodeNeuron_PCM::saveInternal(std::ofstream& dataFile) const
{
    // The Parameters instances should not be saved here! (June 30,2011, Damien)
    dataFile.write(reinterpret_cast<const char*>(&mEmitDelay),
                   sizeof(mEmitDelay));

    if (!dataFile.good())
        throw std::runtime_error(
            "NodeNeuron_PCM::saveInternal(): error writing data");
}

void N2D2::NodeNeuron_PCM::loadInternal(std::ifstream& dataFile)
{
    dataFile.read(reinterpret_cast<char*>(&mEmitDelay), sizeof(mEmitDelay));

    if (!dataFile.good())
        throw std::runtime_error(
            "NodeNeuron_PCM::loadInternal(): error reading data");
}

void N2D2::NodeNeuron_PCM::logStatePlot()
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
