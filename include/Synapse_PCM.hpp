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

#ifndef N2D2_SYNAPSE_PCM_H
#define N2D2_SYNAPSE_PCM_H

#include <algorithm>
#include <fstream>

#include "NodeNeuron.hpp"

namespace N2D2 {
typedef double Weight_T;

/**
 * Synapse model used by the NodeNeuron_PCM class (constituted of 2-PCM
 * devices).
*/
struct Synapse_PCM : public Synapse {
    struct Stats_PCM : public Stats {
        Stats_PCM()
            : Stats(),
              nbDevices(0),
              readEnergy(0.0),
              maxReadEnergy(0.0),
              setEvents(0),
              maxSetEvents(0),
              resetEvents(0),
              maxResetEvents(0) {};

        unsigned int nbDevices;
        double readEnergy;
        double maxReadEnergy;
        unsigned long long int setEvents;
        unsigned long long int maxSetEvents;
        unsigned long long int resetEvents;
        unsigned long long int maxResetEvents;
    };

    /**
     * SetResetLow: coarse discretization steps near 0, but reduced sensibility
     * to w_{min}/w_{max} variability
     * SetResetHigh: finer discretization steps near 0, but extremely sensitive
     * to w_{max} variability
    */
    enum ProgramMethod {
        Ideal,
        SetResetLow,
        SetResetGlobalLow,
        SetResetHigh,
        SetResetGlobalHigh
    };

    /// PCM device model parameters
    struct Device {
        /**
         * Minimum conductance \f$w_{min}\f$. Upon calling
         * NodeNeuron::newSynapse(),
         * \f$w_{min} = \mathcal{N}(\overline{w_{min}}, {\sigma_{w_{min}}}^2)\f$
         * where \f$\overline{w_{min}}\f$ and
         * \f$\sigma_{w_{min}}\f$ are neural parameters
        */
        Weight_T weightMin;
        /**
         * Maximum conductance \f$w_{max}\f$. Upon calling
         * NodeNeuron::newSynapse(),
         * \f$w_{max} = \mathcal{N}(\overline{w_{max}}, {\sigma_{w_{max}}}^2)\f$
         * where \f$\overline{w_{max}}\f$ and
         * \f$\sigma_{w_{max}}\f$ are neural parameters
        */
        Weight_T weightMax;
        /**
         * Conductance increment \f$\alpha{}_{+}\f$. Upon calling
         * NodeNeuron::newSynapse(),
         * \f$\alpha{}_{+} = \mathcal{N}(\overline{\alpha{}_{+}},
         * {\sigma_{\alpha{}_{+}}}^2)\f$ where
         * \f$\overline{\alpha{}_{+}}\f$ and \f$\sigma_{\alpha{}_{+}}\f$ are
         * neural parameters
        */
        Weight_T weightIncrement;
        /**
         * Conductance increment damping factor \f$\beta{}_{+}\f$. Upon calling
         * NodeNeuron::newSynapse(),
         * \f$\beta{}_{+} = \mathcal{N}(\overline{\beta{}_{+}},
         * {\sigma_{\beta{}_{+}}}^2)\f$ where
         * \f$\overline{\beta{}_{+}}\f$ and \f$\sigma_{\beta{}_{+}}\f$ are
         * neural parameters
        */
        double weightIncrementDamping;
        /// Increment variability
        double weightIncrementVar;
        /**
         * Conductance \f$w\f$. Upon calling NodeNeuron::newSynapse(),
         * \f$w = \mathcal{N}(\overline{w_{init}}, {\sigma_{w_{init}}}^2) \in
         * [w_{min}, w_{max}]\f$ where
         * \f$\overline{w_{init}}\f$ and \f$\sigma_{w_{init}}\f$ are neural
         * parameters
        */
        Weight_T weight;
        /// Potentiating pulse number in the experimental LTP sequence loaded
        /// with NodeNeuron_PCM::experimentalLtpModel()
        unsigned int pulseIndex;
    };

    /// PCM device stats
    struct DeviceStats {
        /// Image of the read energy (accumulation of the PCM conductance at
        /// each read event) since last clearStats() call.
        double statsReadEnergy;
        /// Number of PCM device SET events since last clearStats() call.
        unsigned long long int statsSetEvents;
        /// Number of PCM device RESET events since last clearStats() call.
        unsigned long long int statsResetEvents;
    };

    Synapse_PCM(bool bipolar_,
                unsigned int redundancy_,
                Time_T delay_,
                const Spread<Weight_T>& weightMinMean_,
                const Spread<Weight_T>& weightMaxMean_,
                const Spread<Weight_T>& weightIncrement_,
                double weightIncrementVar_,
                const Spread<double>& weightIncrementDamping_,
                const std::vector<double>* weightIncrementModel_,
                Weight_T weightInit_);
    inline virtual double getRelativeWeight(bool allowBipolarRange
                                            = false) const;
    virtual void setRelativeWeight(double relWeight);
    virtual Weight_T getWeight(double bipolarPosGain = 1.0) const;
    virtual void setPulse(Device& dev);
    virtual void resetPulse(Device& dev);
    virtual void saveInternal(std::ofstream& dataFile) const;
    virtual void loadInternal(std::ifstream& dataFile);
    virtual Stats* newStats() const;
    virtual void getStats(Stats* statsObj) const;
    virtual void logStats(std::ofstream& dataFile, Stats* statsObj) const;
    virtual void logStats(std::ofstream& dataFile,
                          const std::string& suffix) const;
    virtual void clearStats();
    static void setProgramMethod(ProgramMethod method)
    {
        mProgramMethod = method;
    };
    static void setSetLimit(unsigned int setLimit)
    {
        mSetLimit = setLimit;
    };
    virtual ~Synapse_PCM() {};

    /// Is the synapse bipolar?
    const bool bipolar;
    /// Experimental increment model
    const std::vector<double>* weightIncrementModel;
    /// Synaptic delay
    Time_T delay;
    Weight_T weightMinGlobalMean;
    Weight_T weightMaxGlobalMean;
    std::vector<Device> devices;

    // Stats
    /// Number of read events on the synapse since last clearStats() call
    /// (shared between the PCM devices).
    unsigned long long int statsReadEvents;
    std::vector<DeviceStats> stats;

private:
    static ProgramMethod mProgramMethod;
    static unsigned int mSetLimit;
};
}

namespace {
template <>
const char* const EnumStrings<N2D2::Synapse_PCM::ProgramMethod>::data[]
    = {"Ideal",        "SetResetLow",       "SetResetGlobalLow",
       "SetResetHigh", "SetResetGlobalHigh"};
}

double N2D2::Synapse_PCM::getRelativeWeight(bool allowBipolarRange) const
{
    const unsigned int devSize = devices.size();

    if (bipolar) {
        const unsigned int devHalfSize = devSize / 2;
        const double weightAbs = devHalfSize
                                 * (weightMaxGlobalMean - weightMinGlobalMean);

        return (allowBipolarRange)
                   ? Utils::clamp(getWeight() / weightAbs, -1.0, 1.0)
                   : Utils::clamp(
                         (getWeight() / weightAbs + 1.0) / 2.0, 0.0, 1.0);
    } else
        return Utils::clamp((getWeight() / devSize - weightMinGlobalMean)
                            / (weightMaxGlobalMean - weightMinGlobalMean),
                            0.0,
                            1.0);
}

#endif // N2D2_SYNAPSE_PCM_H
