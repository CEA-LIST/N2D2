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

#ifndef N2D2_SYNAPSE_H
#define N2D2_SYNAPSE_H

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace N2D2 {
/**
 * Abstract synapse base class, which provides the minimum interface required
 * for a synapse. Any synapse model must inherit from
 * this class.
*/
struct Synapse {
    struct Stats {
        Stats() : nbSynapses(0), readEvents(0), maxReadEvents(0) {};

        unsigned int nbSynapses;
        unsigned long long int readEvents;
        unsigned long long int maxReadEvents;
    };

    /// Constructor
    Synapse() {};

    /**
     * Get the relative weight of the synapse (comprised between 0 and 1, or -1
     * and 1 if the synapse is bipolar and
     * allowBipolarRange is true).
     * @return Relative synaptic weight (in the [0,1] interval, or [-1,1] for
     * bipolar synapses)
    */
    virtual double getRelativeWeight(bool allowBipolarRange = false) const = 0;

    /**
     * Set the relative weight of the synapse (comprised between 0 and 1).
     * By default, throw an exception when not implemented, as some synapse
     * model do not allow to easily set a relative weight.
    */
    virtual void setRelativeWeight(double /*relWeight*/)
    {
        throw std::runtime_error(
            "Not possible to set the relative weight for this synapse model");
    };

    virtual void saveInternal(std::ofstream& dataFile) const = 0;
    virtual void loadInternal(std::ifstream& dataFile) = 0;

    virtual Stats* newStats() const = 0;
    virtual void getStats(Stats* statsObj) const = 0;
    virtual void logStats(std::ofstream& dataFile, Stats* statsObj) const = 0;
    /**
     * Appends the synaptic stats (dependent on the synaptic model: number of
     *read or write events, energy consumed, etc)
     * to a file stream.
     *
     * @param   dataFile        File stream to append the stats to
     * @param   suffix          String to append at the end of each stat line
     *(which is used to add the NodeNeuron ID,
     *                          Layer ID...)
    */
    virtual void logStats(std::ofstream& dataFile,
                          const std::string& suffix) const = 0;
    /// Clear synaptic stats
    virtual void clearStats() = 0;
    /// Destructor
    virtual ~Synapse() {};
};
}

#endif // N2D2_SYNAPSE_H
