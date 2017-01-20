/*
    (C) Copyright 2013 CEA LIST. All Rights Reserved.
    Contributor(s): Olivier BICHLER (olivier.bichler@cea.fr)

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

#include "Synapse_Normalized.hpp"

N2D2::Synapse_Normalized::Synapse_Normalized(Time_T delay_,
                                             double weightIncrement_,
                                             double weightIncrementDamping_,
                                             double weightDecrement_,
                                             double weightDecrementDamping_,
                                             double weightInit_)
    : delay(delay_),
      weightIncrement(weightIncrement_),
      weightIncrementDamping(weightIncrementDamping_),
      weightDecrement(weightDecrement_),
      weightDecrementDamping(weightDecrementDamping_)
{
    // ctor
    setRelativeWeight(weightInit_);

    clearStats();
}

void N2D2::Synapse_Normalized::setRelativeWeight(double relWeight)
{
    if (relWeight < 0.0 || relWeight > 1.0)
        throw std::domain_error(
            "Relative weight is out of range (must be >= 0.0 and <= 1.0)");

    weight = relWeight;
}

void N2D2::Synapse_Normalized::saveInternal(std::ofstream& dataFile) const
{
    dataFile.write(reinterpret_cast<const char*>(&delay), sizeof(delay));
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
            "Synapse_Normalized::saveInternal(): error writing data");
}

void N2D2::Synapse_Normalized::loadInternal(std::ifstream& dataFile)
{
    dataFile.read(reinterpret_cast<char*>(&delay), sizeof(delay));
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
            "Synapse_Normalized::loadInternal(): error reading data");
}

N2D2::Synapse_Normalized::Stats* N2D2::Synapse_Normalized::newStats() const
{
    return new Stats_Normalized();
}

void N2D2::Synapse_Normalized::getStats(Stats* statsObj) const
{
    Stats_Normalized* myStats = static_cast<Stats_Normalized*>(statsObj);
    ++myStats->nbSynapses;
    myStats->readEvents += statsReadEvents;
    myStats->maxReadEvents = std::max(myStats->maxReadEvents, statsReadEvents);
    myStats->incEvents += statsIncEvents;
    myStats->maxIncEvents = std::max(myStats->maxIncEvents, statsIncEvents);
    myStats->decEvents += statsDecEvents;
    myStats->maxDecEvents = std::max(myStats->maxDecEvents, statsDecEvents);
}

void N2D2::Synapse_Normalized::logStats(std::ofstream& dataFile,
                                        Stats* statsObj) const
{
    Stats_Normalized* myStats = static_cast<Stats_Normalized*>(statsObj);

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

void N2D2::Synapse_Normalized::logStats(std::ofstream& dataFile,
                                        const std::string& suffix) const
{
    dataFile << "R " << statsReadEvents << " " << suffix << "\n"
                                                            "I "
             << statsIncEvents << " " << suffix << "\n"
                                                   "D " << statsDecEvents << " "
             << suffix << "\n";
}

void N2D2::Synapse_Normalized::clearStats()
{
    statsReadEvents = 0;
    statsIncEvents = 0;
    statsDecEvents = 0;
}
