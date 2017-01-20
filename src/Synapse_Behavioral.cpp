/*
    (C) Copyright 2013 CEA LIST. All Rights Reserved.
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

#include "Synapse_Behavioral.hpp"

N2D2::Synapse_Behavioral::Synapse_Behavioral(Time_T delay_,
                                             Weight_T weightMin_,
                                             Weight_T weightMax_,
                                             Weight_T weightIncrement_,
                                             double weightIncrementDamping_,
                                             Weight_T weightDecrement_,
                                             double weightDecrementDamping_,
                                             Weight_T weightInit_)
    : delay(delay_),
      weightMin(weightMin_),
      weightMax(weightMax_),
      weightIncrement(weightIncrement_),
      weightIncrementDamping(weightIncrementDamping_),
      weightDecrement(weightDecrement_),
      weightDecrementDamping(weightDecrementDamping_)
{
    // ctor
    if (weightMax < weightMin) {
        std::cout << "Notice: wmax is lower than wmin, set wmin = wmax = (wmin "
                     "+ wmax)/2" << std::endl;
        weightMax = weightMin = (weightMax + weightMin) / 2.0;
    }

    setRelativeWeight(
        Utils::clamp((weightMin != weightMax)
                         ? ((weightInit_ - weightMin) / (weightMax - weightMin))
                         : 0.0,
                     0.0,
                     1.0));

    clearStats();
}

void N2D2::Synapse_Behavioral::setRelativeWeight(double relWeight)
{
    if (relWeight < 0.0 || relWeight > 1.0)
        throw std::domain_error(
            "Relative weight is out of range (must be >= 0.0 and <= 1.0)");

    weight = weightMin + relWeight * (weightMax - weightMin);
}

void N2D2::Synapse_Behavioral::saveInternal(std::ofstream& dataFile) const
{
    dataFile.write(reinterpret_cast<const char*>(&delay), sizeof(delay));
    dataFile.write(reinterpret_cast<const char*>(&weightMin),
                   sizeof(weightMin));
    dataFile.write(reinterpret_cast<const char*>(&weightMax),
                   sizeof(weightMax));
    dataFile.write(reinterpret_cast<const char*>(&weightIncrement),
                   sizeof(weightIncrement));
    dataFile.write(reinterpret_cast<const char*>(&weightIncrementDamping),
                   sizeof(weightIncrementDamping));
    dataFile.write(reinterpret_cast<const char*>(&weightDecrement),
                   sizeof(weightDecrement));
    dataFile.write(reinterpret_cast<const char*>(&weightDecrementDamping),
                   sizeof(weightDecrementDamping));
    dataFile.write(reinterpret_cast<const char*>(&weight), sizeof(weight));

    if (!dataFile.good())
        throw std::runtime_error(
            "Synapse_Behavioral::saveInternal(): error writing data");
}

void N2D2::Synapse_Behavioral::loadInternal(std::ifstream& dataFile)
{
    dataFile.read(reinterpret_cast<char*>(&delay), sizeof(delay));
    dataFile.read(reinterpret_cast<char*>(&weightMin), sizeof(weightMin));
    dataFile.read(reinterpret_cast<char*>(&weightMax), sizeof(weightMax));
    dataFile.read(reinterpret_cast<char*>(&weightIncrement),
                  sizeof(weightIncrement));
    dataFile.read(reinterpret_cast<char*>(&weightIncrementDamping),
                  sizeof(weightIncrementDamping));
    dataFile.read(reinterpret_cast<char*>(&weightDecrement),
                  sizeof(weightDecrement));
    dataFile.read(reinterpret_cast<char*>(&weightDecrementDamping),
                  sizeof(weightDecrementDamping));
    dataFile.read(reinterpret_cast<char*>(&weight), sizeof(weight));

    if (!dataFile.good())
        throw std::runtime_error(
            "Synapse_Behavioral::loadInternal(): error reading data");
}

N2D2::Synapse_Behavioral::Stats* N2D2::Synapse_Behavioral::newStats() const
{
    return new Stats_Behavioral();
}

void N2D2::Synapse_Behavioral::getStats(Stats* statsObj) const
{
    Stats_Behavioral* myStats = static_cast<Stats_Behavioral*>(statsObj);
    ++myStats->nbSynapses;
    myStats->readEvents += statsReadEvents;
    myStats->maxReadEvents = std::max(myStats->maxReadEvents, statsReadEvents);
    myStats->incEvents += statsIncEvents;
    myStats->maxIncEvents = std::max(myStats->maxIncEvents, statsIncEvents);
    myStats->decEvents += statsDecEvents;
    myStats->maxDecEvents = std::max(myStats->maxDecEvents, statsDecEvents);
}

void N2D2::Synapse_Behavioral::logStats(std::ofstream& dataFile,
                                        Stats* statsObj) const
{
    Stats_Behavioral* myStats = static_cast<Stats_Behavioral*>(statsObj);

    dataFile << "Synapses: " << myStats->nbSynapses
             << "\n"
                "Read events: " << myStats->readEvents
             << "\n"
                "Read events per synapse (average): "
             << myStats->readEvents / (double)myStats->nbSynapses
             << "\n"
                "Max. read events per synapse: " << myStats->maxReadEvents
             << "\n"
                "Inc. events: " << myStats->incEvents
             << "\n"
                "Inc. events per synapse (average): "
             << myStats->incEvents / (double)myStats->nbSynapses
             << "\n"
                "Max. inc. events per synapse: " << myStats->maxIncEvents
             << "\n"
                "Dec. events: " << myStats->decEvents
             << "\n"
                "Dec. events per synapse (average): "
             << myStats->decEvents / (double)myStats->nbSynapses
             << "\n"
                "Max. dec. events per synapse: " << myStats->maxDecEvents
             << "\n";
}

void N2D2::Synapse_Behavioral::logStats(std::ofstream& dataFile,
                                        const std::string& suffix) const
{
    dataFile << "R " << statsReadEvents << " " << suffix << "\n"
                                                            "I "
             << statsIncEvents << " " << suffix << "\n"
                                                   "D " << statsDecEvents << " "
             << suffix << "\n";
}

void N2D2::Synapse_Behavioral::clearStats()
{
    statsReadEvents = 0;
    statsIncEvents = 0;
    statsDecEvents = 0;
}
