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

#ifndef N2D2_SYNAPSE_NORMALIZED_H
#define N2D2_SYNAPSE_NORMALIZED_H

#include <algorithm>
#include <fstream>

#include "NodeNeuron.hpp"

namespace N2D2 {
/**
 * Synapse model implementing the behavioral model described in:
 * O Bichler, D Querlioz, SJ Thorpe, JP Bourgoin, C Gamrat, "Unsupervised
 *features extraction from asynchronous silicon retina
 * through Spike-Timing-Dependent Plasticity", Neural Networks (IJCNN), The 2011
 *International Joint Conference on, 859-866 @n
 *
 * This synapse can be used to implement the following behavior in the neuron:
 *@n
 * Synaptic increase:
 * \f$\Delta{}w_{+} = \alpha{}_{+}.\exp{}\left(-\beta{}_{+}.w\right)\f$ @n
 * Synaptic decrease:
 * \f$\Delta{}w_{-} = \alpha{}_{-}.\exp{}\left(-\beta{}_{-}.w\right)\f$ @n
*/
struct Synapse_Normalized : public Synapse {
    struct Stats_Normalized : public Stats {
        Stats_Normalized()
            : Stats(),
              incEvents(0),
              maxIncEvents(0),
              decEvents(0),
              maxDecEvents(0) {};

        unsigned long long int incEvents;
        unsigned long long int maxIncEvents;
        unsigned long long int decEvents;
        unsigned long long int maxDecEvents;
    };

    /**
     * Constructor
     * @param delay         Synaptic delay \f$w_{delay}\f$ (0 = no synaptic
     * delay). Upon calling NodeNeuron::newSynapse(),
     * \f$w_{delay} = \mathcal{N}(\overline{w_{delay}},
     * {\sigma_{w_{delay}}}^2)\f$ where \f$\overline{w_{delay}}\f$ and
     * \f$\sigma_{w_{delay}}\f$ are neural parameters
    */
    Synapse_Normalized(Time_T delay,
                       double weightIncrement,
                       double weightIncrementDamping,
                       double weightDecrement,
                       double weightDecrementDamping,
                       double weightInit);
    virtual double getRelativeWeight(bool /*allowBipolarRange*/ = false) const
    {
        return weight;
    };
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
    /// Destructor
    virtual ~Synapse_Normalized() {};

    /// Synaptic delay
    Time_T delay;
    /**
     * Weight increment \f$\alpha{}_{+}\f$. Upon calling
     * NodeNeuron::newSynapse(),
     * \f$\alpha{}_{+} = \mathcal{N}(\overline{\alpha{}_{+}},
     * {\sigma_{\alpha{}_{+}}}^2)\f$ where
     * \f$\overline{\alpha{}_{+}}\f$ and \f$\sigma_{\alpha{}_{+}}\f$ are neural
     * parameters
    */
    double weightIncrement;
    /**
     * Weight increment damping factor \f$\beta{}_{+}\f$. Upon calling
     * NodeNeuron::newSynapse(),
     * \f$\beta{}_{+} = \mathcal{N}(\overline{\beta{}_{+}},
     * {\sigma_{\beta{}_{+}}}^2)\f$ where
     * \f$\overline{\beta{}_{+}}\f$ and \f$\sigma_{\beta{}_{+}}\f$ are neural
     * parameters
    */
    double weightIncrementDamping;
    /**
     * Weight decrement \f$\alpha{}_{-}\f$. Upon calling
     * NodeNeuron::newSynapse(),
     * \f$\alpha{}_{-} = \mathcal{N}(\overline{\alpha{}_{-}},
     * {\sigma_{\alpha{}_{-}}}^2)\f$ where
     * \f$\overline{\alpha{}_{-}}\f$ and \f$\sigma_{\alpha{}_{-}}\f$ are neural
     * parameters
    */
    double weightDecrement;
    /**
     * Weight decrement damping factor \f$\beta{}_{-}\f$. Upon calling
     * NodeNeuron::newSynapse(),
     * \f$\beta{}_{-} = \mathcal{N}(\overline{\beta{}_{-}},
     * {\sigma_{\beta{}_{-}}}^2)\f$ where
     * \f$\overline{\beta{}_{-}}\f$ and \f$\sigma_{\beta{}_{-}}\f$ are neural
     * parameters
    */
    double weightDecrementDamping;
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
    /// Number of weight increment events on the synapse since last clearStats()
    /// call.
    unsigned long long int statsIncEvents;
    /// Number of weight decrement events on the synapse since last clearStats()
    /// call.
    unsigned long long int statsDecEvents;
};
}

#endif // N2D2_SYNAPSE_NORMALIZED_H
