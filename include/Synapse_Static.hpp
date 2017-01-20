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

#ifndef N2D2_SYNAPSE_STATIC_H
#define N2D2_SYNAPSE_STATIC_H

#include <algorithm>
#include <fstream>

#include "NodeNeuron.hpp"

namespace N2D2 {
struct Synapse_Static : public Synapse {
    struct Stats_Static : public Stats {
        Stats_Static() : Stats() {};
    };

    /**
     * Constructor
     * @param delay         Synaptic delay \f$w_{delay}\f$ (0 = no synaptic
     * delay). Upon calling NodeNeuron::newSynapse(),
     * \f$w_{delay} = \mathcal{N}(\overline{w_{delay}},
     * {\sigma_{w_{delay}}}^2)\f$ where \f$\overline{w_{delay}}\f$ and
     * \f$\sigma_{w_{delay}}\f$ are neural parameters
    */
    Synapse_Static(bool bipolar, Time_T delay, double weightInit);
    inline virtual double getRelativeWeight(bool allowBipolarRange
                                            = false) const;
    virtual void setRelativeWeight(double relWeight);
    virtual void saveInternal(std::ofstream& dataFile) const;
    virtual void loadInternal(std::ifstream& dataFile);
    virtual Stats* newStats() const;
    virtual void getStats(Stats* statsObj) const;
    virtual void logStats(std::ofstream& dataFile, Stats* statsObj) const;
    /**
     * Log synaptic stats in file stream dataFile.
     *
     * @param dataFile       File stream to save the stats into.
     * @param suffix         Suffix to append to each line (number of reads,
     *increments and decrements)
    */
    virtual void logStats(std::ofstream& dataFile,
                          const std::string& suffix) const;
    /// Clear synaptic stats. Reset statsReadEvents, statsIncEvents and
    /// statsDecEvents to 0.
    virtual void clearStats();
    static void setCheckWeightRange(bool checkWeightRange)
    {
        mCheckWeightRange = checkWeightRange;
    };
    /// Destructor
    virtual ~Synapse_Static() {};

    /// Is the synapse bipolar?
    const bool bipolar;
    /// Synaptic delay
    Time_T delay;
    /**
     * Weight \f$w\f$. Upon calling NodeNeuron::newSynapse(),
     * \f$w = \mathcal{N}(\overline{w_{init}}, {\sigma_{w_{init}}}^2) \in
     * [w_{min}, w_{max}]\f$ where
     * \f$\overline{w_{init}}\f$ and \f$\sigma_{w_{init}}\f$ are neural
     * parameters
    */
    double weight;

    // Stats
    /// Number of read events on the synapse since last clearStats() call.
    unsigned long long int statsReadEvents;

private:
    static bool mCheckWeightRange;
};
}

double N2D2::Synapse_Static::getRelativeWeight(bool allowBipolarRange) const
{
    return (bipolar && !allowBipolarRange) ? (weight + 1.0) / 2.0 : weight;
}

#endif // N2D2_SYNAPSE_STATIC_H
