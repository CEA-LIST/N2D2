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

#include "Synapse_PCM.hpp"

N2D2::Synapse_PCM::ProgramMethod N2D2::Synapse_PCM::mProgramMethod = Ideal;
unsigned int N2D2::Synapse_PCM::mSetLimit = 1000;

N2D2::Synapse_PCM::Synapse_PCM(bool bipolar_,
                               unsigned int redundancy_,
                               Time_T delay_,
                               const Spread<Weight_T>& weightMinMean_,
                               const Spread<Weight_T>& weightMaxMean_,
                               const Spread<Weight_T>& weightIncrement_,
                               double weightIncrementVar_,
                               const Spread<double>& weightIncrementDamping_,
                               const std::vector<double>* weightIncrementModel_,
                               Weight_T weightInit_)
    : bipolar(bipolar_),
      weightIncrementModel(weightIncrementModel_),
      delay(delay_),
      weightMinGlobalMean(weightMinMean_.mean()),
      weightMaxGlobalMean(weightMaxMean_.mean())
{
    // ctor
    const unsigned int devSize = (bipolar) ? 2 * redundancy_ : redundancy_;
    devices.resize(devSize);

    for (unsigned int devIdx = 0; devIdx < devSize; ++devIdx) {
        Device& dev = devices[devIdx];

        dev.weightMin = weightMinMean_.spreadNormal(0);
        dev.weightMax = weightMaxMean_.spreadNormal(0);
        dev.weightIncrement = weightIncrement_.spreadNormal(0);
        dev.weightIncrementDamping = weightIncrementDamping_.spreadNormal();
        dev.weightIncrementVar = weightIncrementVar_;

        if (dev.weightMax < dev.weightMin) {
            std::cout << "Notice: wmax is lower than wmin, set wmin = wmax = "
                         "(wmin + wmax)/2" << std::endl;
            dev.weightMax = dev.weightMin = (dev.weightMax + dev.weightMin)
                                            / 2.0;
        }

        dev.pulseIndex = 0;
    }

    setRelativeWeight(
        (bipolar) ? Utils::clamp(weightInit_ / weightMaxGlobalMean, -1.0, 1.0)
                  : Utils::clamp((weightInit_ - weightMinGlobalMean)
                                 / (weightMaxGlobalMean - weightMinGlobalMean),
                                 0.0,
                                 1.0));

    stats.resize(devSize);
    clearStats();
}

void N2D2::Synapse_PCM::setRelativeWeight(double relWeight)
{
    if (((bipolar && relWeight < -1.0) || (!bipolar && relWeight < 0.0))
        || relWeight > 1.0)
        throw std::domain_error("Relative weight is out of range (must be in "
                                "[0,1] or [-1,1] range)");

    const unsigned int devSize = devices.size();
    const unsigned int devHalfSize = devSize / 2;

    const double targetWeightLow
        = weightMinGlobalMean + std::fabs(relWeight)
                                * (weightMaxGlobalMean - weightMinGlobalMean);
    const double targetWeightHigh
        = weightMinGlobalMean + (1.0 - std::fabs(relWeight))
                                * (weightMaxGlobalMean - weightMinGlobalMean);
    const double globalTargetWeightLow = ((bipolar) ? devHalfSize : devSize)
                                         * targetWeightLow;
    const double globalTargetWeightHigh = ((bipolar) ? devHalfSize : devSize)
                                          * targetWeightHigh;

    double globalWeight = 0.0;

    for (unsigned int devIdx = 0; devIdx < devSize; ++devIdx) {
        Device& dev = devices[devIdx];

        if (bipolar && ((devIdx < devHalfSize && relWeight < 0)
                        || (devIdx >= devHalfSize && relWeight >= 0))) {
            if (mProgramMethod == Ideal) {
                dev.weight = dev.weightMin;
                dev.pulseIndex = 0;
            } else if (mProgramMethod == SetResetHigh) {
                resetPulse(dev);

                for (unsigned int set = 0;
                     (dev.weight < targetWeightHigh) && (set < mSetLimit);
                     ++set)
                    setPulse(dev);
            } else if (mProgramMethod == SetResetGlobalHigh) {
                resetPulse(dev);

                for (unsigned int set = 0;
                     (globalWeight + dev.weight < globalTargetWeightHigh)
                     && (set < mSetLimit);
                     ++set)
                    setPulse(dev);

                globalWeight += dev.weight;
            } else
                resetPulse(dev);
        } else {
            if (mProgramMethod == Ideal) {
                dev.weight = Utils::clamp(
                    targetWeightLow, dev.weightMin, dev.weightMax);

                if (weightIncrementModel != NULL
                    && !(*weightIncrementModel).empty()) {
                    if (dev.weightMax != dev.weightMin) {
                        const double w = (targetWeightLow - dev.weightMin)
                                         / (dev.weightMax - dev.weightMin);
                        const std::vector<double>::const_iterator it
                            = std::lower_bound((*weightIncrementModel).begin(),
                                               (*weightIncrementModel).end(),
                                               w);
                        dev.pulseIndex = (it - (*weightIncrementModel).begin());
                    }
                }
            } else if (mProgramMethod == SetResetLow) {
                resetPulse(dev);

                for (unsigned int set = 0;
                     (dev.weight < targetWeightLow) && (set < mSetLimit);
                     ++set)
                    setPulse(dev);
            } else if (mProgramMethod == SetResetGlobalLow) {
                resetPulse(dev);

                for (unsigned int set = 0;
                     (globalWeight + dev.weight < globalTargetWeightLow)
                     && (set < mSetLimit);
                     ++set)
                    setPulse(dev);

                globalWeight += dev.weight;
            } else {
                resetPulse(dev);

                for (unsigned int set = 0; set < mSetLimit; ++set)
                    setPulse(dev);
            }
        }
    }
}

N2D2::Weight_T N2D2::Synapse_PCM::getWeight(double bipolarPosGain) const
{
    Weight_T weight = 0.0;

    if (bipolar) {
        const unsigned int devSize = devices.size();
        const unsigned int devHalfSize = devSize / 2;

        for (unsigned int devIdx = 0; devIdx < devHalfSize; ++devIdx)
            weight += bipolarPosGain * devices[devIdx].weight;

        for (unsigned int devIdx = devHalfSize; devIdx < devSize; ++devIdx)
            weight -= devices[devIdx].weight;
    } else {
        for (std::vector<Device>::const_iterator it = devices.begin(),
                                                 itEnd = devices.end();
             it != itEnd;
             ++it)
            weight += (*it).weight;
    }

    return weight;
}

void N2D2::Synapse_PCM::setPulse(Device& dev)
{
    ++dev.pulseIndex;

    if (weightIncrementModel != NULL && !(*weightIncrementModel).empty()) {
        const double weightNormalized
            = (dev.pulseIndex < (*weightIncrementModel).size())
                  ? (*weightIncrementModel)[dev.pulseIndex]
                  : (*weightIncrementModel).back();

        const double weight = dev.weightMin + (dev.weightMax - dev.weightMin)
                                              * weightNormalized;

        dev.weight = Utils::clamp(
            (dev.weightIncrementVar > 0.0)
                ? Random::randNormal(weight, weight * dev.weightIncrementVar)
                : weight,
            dev.weightMin,
            dev.weightMax);
    } else {
        // The following check is required to avoid division by 0 when the
        // dispersion is ~ > 20%
        if (dev.weightMax > dev.weightMin) {
            double dw = dev.weightIncrement
                        * std::exp(dev.weightIncrementDamping
                                   * (dev.weight - dev.weightMin)
                                   / (dev.weightMax - dev.weightMin));

            if (dev.weightIncrementVar > 0.0)
                dw = Random::randNormal(dw, dw * dev.weightIncrementVar);

            dev.weight
                = Utils::clamp(dev.weight + dw, dev.weightMin, dev.weightMax);
        }
    }
}

void N2D2::Synapse_PCM::resetPulse(Device& dev)
{
    dev.weight = dev.weightMin;
    dev.pulseIndex = 0;
}

void N2D2::Synapse_PCM::saveInternal(std::ofstream& dataFile) const
{
    dataFile.write(reinterpret_cast<const char*>(&delay), sizeof(delay));
    dataFile.write(reinterpret_cast<const char*>(&weightMinGlobalMean),
                   sizeof(weightMinGlobalMean));
    dataFile.write(reinterpret_cast<const char*>(&weightMaxGlobalMean),
                   sizeof(weightMaxGlobalMean));
    dataFile.write(reinterpret_cast<const char*>(&devices[0]),
                   devices.size() * sizeof(Device));

    if (!dataFile.good())
        throw std::runtime_error(
            "Synapse_PCM::saveInternal(): error writing data");
}

void N2D2::Synapse_PCM::loadInternal(std::ifstream& dataFile)
{
    dataFile.read(reinterpret_cast<char*>(&delay), sizeof(delay));
    dataFile.read(reinterpret_cast<char*>(&weightMinGlobalMean),
                  sizeof(weightMinGlobalMean));
    dataFile.read(reinterpret_cast<char*>(&weightMaxGlobalMean),
                  sizeof(weightMaxGlobalMean));
    dataFile.read(reinterpret_cast<char*>(&devices[0]),
                  devices.size() * sizeof(Device));

    if (!dataFile.good())
        throw std::runtime_error(
            "Synapse_PCM::loadInternal(): error reading data");
}

N2D2::Synapse_PCM::Stats* N2D2::Synapse_PCM::newStats() const
{
    return new Stats_PCM();
}

void N2D2::Synapse_PCM::getStats(Stats* statsObj) const
{
    Stats_PCM* myStats = static_cast<Stats_PCM*>(statsObj);
    ++myStats->nbSynapses;
    myStats->nbDevices += devices.size();
    myStats->readEvents += statsReadEvents * devices.size();
    myStats->maxReadEvents = std::max(myStats->maxReadEvents, statsReadEvents);

    for (unsigned int devIdx = 0, size = stats.size(); devIdx < size;
         ++devIdx) {
        myStats->readEnergy += stats[devIdx].statsReadEnergy;
        myStats->maxReadEnergy
            = std::max(myStats->maxReadEnergy, stats[devIdx].statsReadEnergy);
        myStats->setEvents += stats[devIdx].statsSetEvents;
        myStats->maxSetEvents
            = std::max(myStats->maxSetEvents, stats[devIdx].statsSetEvents);
        myStats->resetEvents += stats[devIdx].statsResetEvents;
        myStats->maxResetEvents
            = std::max(myStats->maxResetEvents, stats[devIdx].statsResetEvents);
    }
}

void N2D2::Synapse_PCM::logStats(std::ofstream& dataFile, Stats* statsObj) const
{
    Stats_PCM* myStats = static_cast<Stats_PCM*>(statsObj);

    dataFile << "Synapses: " << myStats->nbSynapses
             << "\n"
                "Devices: " << myStats->nbDevices << "\n"
                                                     "Devices per synapse: "
             << myStats->nbDevices / (double)myStats->nbSynapses
             << "\n"
                "Read events: " << myStats->readEvents
             << "\n"
                "Read events per synapse (average): "
             << myStats->readEvents / (double)myStats->nbSynapses
             << "\n"
                "Read events per device (average): "
             << myStats->readEvents / (double)myStats->nbDevices
             << "\n"
                "Max. read events per device: " << myStats->maxReadEvents
             << "\n"
                "Read energy: " << myStats->readEnergy
             << " J/(s.V^2)\n"
                "Read energy per synapse (average): "
             << myStats->readEnergy / (double)myStats->nbSynapses
             << " J/(s.V^2)\n"
                "Read energy per device (average): "
             << myStats->readEnergy / (double)myStats->nbDevices
             << " J/(s.V^2)\n"
                "Max. read energy per device: " << myStats->maxReadEnergy
             << "\n"
                "Set events: " << myStats->setEvents
             << "\n"
                "Set events per synapse (average): "
             << myStats->setEvents / (double)myStats->nbSynapses
             << "\n"
                "Set events per device (average): "
             << myStats->setEvents / (double)myStats->nbDevices
             << "\n"
                "Max. set events per device: " << myStats->maxSetEvents
             << "\n"
                "Reset events: " << myStats->resetEvents
             << "\n"
                "Reset events per synapse (average): "
             << myStats->resetEvents / (double)myStats->nbSynapses
             << "\n"
                "Reset events per device (average): "
             << myStats->resetEvents / (double)myStats->nbDevices
             << "\n"
                "Max. reset events per device: " << myStats->maxResetEvents
             << "\n";
}

void N2D2::Synapse_PCM::logStats(std::ofstream& dataFile,
                                 const std::string& suffix) const
{
    for (unsigned int devIdx = 0, size = stats.size(); devIdx < size;
         ++devIdx) {
        dataFile << "I " << devIdx << " " << statsReadEvents << " " << suffix
                 << "\n"
                    "IE " << devIdx << " " << stats[devIdx].statsReadEnergy
                 << " " << suffix << "\n"
                                     "S " << devIdx << " "
                 << stats[devIdx].statsSetEvents << " " << suffix << "\n"
                                                                     "R "
                 << devIdx << " " << stats[devIdx].statsResetEvents << " "
                 << suffix << "\n";
    }
}

void N2D2::Synapse_PCM::clearStats()
{
    statsReadEvents = 0;

    for (std::vector<DeviceStats>::iterator it = stats.begin(),
                                            itEnd = stats.end();
         it != itEnd;
         ++it) {
        (*it).statsReadEnergy = 0.0;
        (*it).statsSetEvents = 0;
        (*it).statsResetEvents = 0;
    }
}
