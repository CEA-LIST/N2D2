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

#include "Synapse_Static.hpp"

bool N2D2::Synapse_Static::mCheckWeightRange = true;

N2D2::Synapse_Static::Synapse_Static(bool bipolar_,
                                     Time_T delay_,
                                     double weightInit_)
    : bipolar(bipolar_), delay(delay_)
{
    // ctor
    setRelativeWeight(weightInit_);

    clearStats();
}

void N2D2::Synapse_Static::setRelativeWeight(double relWeight)
{
    if (mCheckWeightRange
        && (((bipolar && relWeight < -1.0) || (!bipolar && relWeight < 0.0))
            || relWeight > 1.0))
        throw std::domain_error("Relative weight is out of range (must be in "
                                "[0,1] or [-1,1] range)");

    weight = relWeight;
}

void N2D2::Synapse_Static::saveInternal(std::ofstream& dataFile) const
{
    dataFile.write(reinterpret_cast<const char*>(&delay), sizeof(delay));
    dataFile.write(reinterpret_cast<const char*>(&weight), sizeof(weight));

    if (!dataFile.good())
        throw std::runtime_error(
            "Synapse_Static::saveInternal(): error writing data");
}

void N2D2::Synapse_Static::loadInternal(std::ifstream& dataFile)
{
    dataFile.read(reinterpret_cast<char*>(&delay), sizeof(delay));
    dataFile.read(reinterpret_cast<char*>(&weight), sizeof(weight));

    if (!dataFile.good())
        throw std::runtime_error(
            "Synapse_Static::loadInternal(): error reading data");
}

N2D2::Synapse_Static::Stats* N2D2::Synapse_Static::newStats() const
{
    return new Stats_Static();
}

void N2D2::Synapse_Static::getStats(Stats* statsObj) const
{
    Stats_Static* myStats = static_cast<Stats_Static*>(statsObj);
    ++myStats->nbSynapses;
    myStats->readEvents += statsReadEvents;
    myStats->maxReadEvents = std::max(myStats->maxReadEvents, statsReadEvents);
}

void N2D2::Synapse_Static::logStats(std::ofstream& dataFile,
                                    Stats* statsObj) const
{
    Stats_Static* myStats = static_cast<Stats_Static*>(statsObj);

    dataFile << "Synapses: " << myStats->nbSynapses
             << "\n"
                "Read events: " << myStats->readEvents
             << "\n"
                "Read events per synapse (average): "
             << myStats->readEvents / (double)myStats->nbSynapses
             << "\n"
                "Max. read events per synapse: " << myStats->maxReadEvents
             << "\n";
}

void N2D2::Synapse_Static::logStats(std::ofstream& dataFile,
                                    const std::string& suffix) const
{
    dataFile << "R " << statsReadEvents << " " << suffix << "\n";
}

void N2D2::Synapse_Static::clearStats()
{
    statsReadEvents = 0;
}
