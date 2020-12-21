/*
    (C) Copyright 2013 CEA LIST. All Rights Reserved.
    Contributor(s): Olivier BICHLER (olivier.bichler@cea.fr)
                    Daniele GARBIN (daniele.garbin@cea.fr)

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

#include "Xnet/Synapse_RRAM.hpp"

N2D2::Synapse_RRAM::ProgramMethod N2D2::Synapse_RRAM::mProgramMethod = Ideal;

N2D2::Synapse_RRAM::Synapse_RRAM(bool bipolar_,
                                 unsigned int redundancy_,
                                 Time_T delay_,
                                 const Spread<Weight_T>& weightMinMean_,
                                 const Spread<Weight_T>& weightMaxMean_,
                                 const Spread<double>& weightMinVar_,
                                 const Spread<double>& weightMaxVar_,
                                 const Spread<double>& weightSetProba_,
                                 const Spread<double>& weightResetProba_,
                                 Weight_T weightInit_)
    : bipolar(bipolar_),
      delay(delay_),
      weightMinGlobalMean(weightMinMean_.mean()),
      weightMaxGlobalMean(weightMaxMean_.mean())
{
    // ctor
    const unsigned int devSize = (bipolar) ? 2 * redundancy_ : redundancy_;
    devices.resize(devSize);

    for (unsigned int devIdx = 0; devIdx < devSize; ++devIdx) {
        Device& dev = devices[devIdx];

        dev.weightMinMean = weightMinMean_.spreadLogNormal(true);
        dev.weightMaxMean = weightMaxMean_.spreadLogNormal(true);
        dev.weightMinVar = weightMinVar_.spreadNormal(0.0);
        dev.weightMaxVar = weightMaxVar_.spreadNormal(0.0);
        dev.weightSetProba = weightSetProba_.spreadNormal(0.0, 1.0);
        dev.weightResetProba = weightResetProba_.spreadNormal(0.0, 1.0);
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

N2D2::Synapse_RRAM::Synapse_RRAM(bool bipolar_,
                                 unsigned int redundancy_,
                                 Time_T delay_,
                                 const Spread<Weight_T>& weightMinMean_,
                                 const Spread<Weight_T>& weightMaxMean_,
                                 double weightMinVarSlope_,
                                 double weightMinVarOrigin_,
                                 double weightMaxVarSlope_,
                                 double weightMaxVarOrigin_,
                                 const Spread<double>& weightSetProba_,
                                 const Spread<double>& weightResetProba_,
                                 Weight_T weightInit_)
    : bipolar(bipolar_),
      delay(delay_),
      weightMinGlobalMean(weightMinMean_.mean()),
      weightMaxGlobalMean(weightMaxMean_.mean())
{
    // ctor
    const unsigned int devSize = (bipolar) ? 2 * redundancy_ : redundancy_;
    devices.resize(devSize);

    for (unsigned int devIdx = 0; devIdx < devSize; ++devIdx) {
        Device& dev = devices[devIdx];

        dev.weightMinMean = weightMinMean_.spreadLogNormal(true);
        dev.weightMaxMean = weightMaxMean_.spreadLogNormal(true);
        dev.weightMinVar
            = std::exp(weightMinVarOrigin_ + weightMinVarSlope_
                                             * std::log(dev.weightMinMean));
        dev.weightMaxVar
            = std::exp(weightMaxVarOrigin_ + weightMaxVarSlope_
                                             * std::log(dev.weightMaxMean));
        dev.weightSetProba = weightSetProba_.spreadNormal(0.0, 1.0);
        dev.weightResetProba = weightResetProba_.spreadNormal(0.0, 1.0);
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

void N2D2::Synapse_RRAM::setRelativeWeight(double relWeight)
{
    if (((bipolar && relWeight < -1.0) || (!bipolar && relWeight < 0.0))
        || relWeight > 1.0)
    {
        std::ostringstream msgStr;
        msgStr << "Relative weight (" << relWeight << ") is out of range"
            " (must be in [0,1] or [-1,1] range)";

        throw std::domain_error(msgStr.str());
    }

    const unsigned int devSize = devices.size();
    const unsigned int devHalfSize = devSize / 2;
    const unsigned int level = (unsigned int)Utils::round(
        ((bipolar) ? devHalfSize : devSize) * std::fabs(relWeight));

    for (unsigned int devIdx = 0; devIdx < devSize; ++devIdx) {
        Device& dev = devices[devIdx];

        if (bipolar && ((devIdx < devHalfSize && relWeight < 0)
                        || (devIdx >= devHalfSize && relWeight >= 0))) {
            dev.stateOn
                = (mProgramMethod == Ideal || mProgramMethod
                                              == IdealProbabilistic)
                      ? false
                      : !(dev.weightResetProba == 1.0
                          || Random::randUniform() <= dev.weightResetProba);
        } else {
            const bool switchOn = (!bipolar || devIdx < devHalfSize)
                                      ? (devIdx < level)
                                      : ((devIdx - devHalfSize) < level);

            dev.stateOn
                = (mProgramMethod == Ideal)
                      ? switchOn
                      : (mProgramMethod == IdealProbabilistic)
                            ? (Random::randUniform() <= std::fabs(relWeight))
                            : (mProgramMethod == SetReset)
                                  ? switchOn && (dev.weightSetProba == 1.0
                                                 || Random::randUniform()
                                                    <= dev.weightSetProba)
                                  : (Random::randUniform()
                                     <= std::fabs(relWeight))
                                    && (dev.weightSetProba == 1.0
                                        || Random::randUniform()
                                           <= dev.weightSetProba);
        }

        dev.weight = (dev.stateOn)
                         ? Random::randLogNormal(std::log(dev.weightMaxMean),
                                                 dev.weightMaxVar)
                         : Random::randLogNormal(std::log(dev.weightMinMean),
                                                 dev.weightMinVar);
    }
}

N2D2::Weight_T N2D2::Synapse_RRAM::getWeight(double bipolarPosGain) const
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

int N2D2::Synapse_RRAM::getDigitalWeight(double threshold) const
{
    int weight = 0;

    if (bipolar) {
        const unsigned int devSize = devices.size();
        const unsigned int devHalfSize = devSize / 2;

        for (unsigned int devIdx = 0; devIdx < devHalfSize; ++devIdx) {
            if (devices[devIdx].weight > threshold)
                ++weight;
        }

        for (unsigned int devIdx = devHalfSize; devIdx < devSize; ++devIdx) {
            if (devices[devIdx].weight > threshold)
                --weight;
        }
    } else {
        for (std::vector<Device>::const_iterator it = devices.begin(),
                                                 itEnd = devices.end();
             it != itEnd;
             ++it) {
            if ((*it).weight > threshold)
                ++weight;
        }
    }

    return weight;
}

void N2D2::Synapse_RRAM::setPulse(Device& dev)
{
    if (!dev.stateOn)
        dev.stateOn = (dev.weightSetProba == 1.0 || Random::randUniform()
                                                    <= dev.weightSetProba);

    dev.weight = (dev.stateOn)
                     ? Random::randLogNormal(std::log(dev.weightMaxMean),
                                             dev.weightMaxVar)
                     : Random::randLogNormal(std::log(dev.weightMinMean),
                                             dev.weightMinVar);
}

void N2D2::Synapse_RRAM::resetPulse(Device& dev)
{
    if (dev.stateOn)
        dev.stateOn = !(dev.weightResetProba == 1.0 || Random::randUniform()
                                                       <= dev.weightResetProba);

    dev.weight = (dev.stateOn)
                     ? Random::randLogNormal(std::log(dev.weightMaxMean),
                                             dev.weightMaxVar)
                     : Random::randLogNormal(std::log(dev.weightMinMean),
                                             dev.weightMinVar);
}

void N2D2::Synapse_RRAM::saveInternal(std::ofstream& dataFile) const
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
            "Synapse_RRAM::saveInternal(): error writing data");
}

void N2D2::Synapse_RRAM::loadInternal(std::ifstream& dataFile)
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
            "Synapse_RRAM::loadInternal(): error reading data");
}

N2D2::Synapse_RRAM::Stats* N2D2::Synapse_RRAM::newStats() const
{
    return new Stats_RRAM();
}

void N2D2::Synapse_RRAM::getStats(Stats* statsObj) const
{
    Stats_RRAM* myStats = static_cast<Stats_RRAM*>(statsObj);
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

void N2D2::Synapse_RRAM::logStats(std::ofstream& dataFile,
                                  Stats* statsObj) const
{
    Stats_RRAM* myStats = static_cast<Stats_RRAM*>(statsObj);

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

void N2D2::Synapse_RRAM::logStats(std::ofstream& dataFile,
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

void N2D2::Synapse_RRAM::clearStats()
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
