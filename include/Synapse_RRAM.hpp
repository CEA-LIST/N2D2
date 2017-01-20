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

#ifndef N2D2_SYNAPSE_RRAM_H
#define N2D2_SYNAPSE_RRAM_H

#include <algorithm>
#include <fstream>

#include "NodeNeuron.hpp"

namespace N2D2 {
typedef double Weight_T;

/**
 * Synapse model used by the NodeNeuron_RRAM class (constituted of a variable
 * number of RRAM devices).
*/
struct Synapse_RRAM : public Synapse {
    struct Stats_RRAM : public Stats {
        Stats_RRAM()
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

    enum ProgramMethod {
        Ideal,
        IdealProbabilistic,
        SetReset,
        SetResetProbabilistic
    };

    /// RRAM device model parameters
    struct Device {
        /// Device mean (for intrinsic variability) minimum conductance
        /// \f$w_{min_{mean}}\f$
        Weight_T weightMinMean;
        /// Device mean (for intrinsic variability) maximum conductance
        /// \f$w_{max_{mean}}\f$
        Weight_T weightMaxMean;
        /// Device minimum conductance intrinsic variability \f$w_{min_{var}}\f$
        double weightMinVar;
        /// Device maximum conductance intrinsic variability \f$w_{max_{var}}\f$
        double weightMaxVar;
        /// Intrinsic SET switching probability \f$P_{SET}\f$
        double weightSetProba;
        /// Intrinsic RESET switching probability \f$P_{RESET}\f$
        double weightResetProba;
        /// Is the device in the ON state?
        bool stateOn;
        /// Device current conductance
        Weight_T weight;
    };

    /// RRAM device stats
    struct DeviceStats {
        /// Image of the read energy (accumulation of the RRAM conductance at
        /// each read event) since last clearStats() call.
        double statsReadEnergy;
        /// Number of RRAM device SET events since last clearStats() call.
        unsigned long long int statsSetEvents;
        /// Number of RRAM device RESET events since last clearStats() call.
        unsigned long long int statsResetEvents;
    };

    Synapse_RRAM(bool bipolar_,
                 unsigned int redundancy_,
                 Time_T delay_,
                 const Spread<Weight_T>& weightMinMean_,
                 const Spread<Weight_T>& weightMaxMean_,
                 const Spread<double>& weightMinVar_,
                 const Spread<double>& weightMaxVar_,
                 const Spread<double>& weightSetProba_,
                 const Spread<double>& weightResetProba_,
                 Weight_T weightInit_);
    Synapse_RRAM(bool bipolar_,
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
                 Weight_T weightInit_);
    inline virtual double getRelativeWeight(bool allowBipolarRange
                                            = false) const;
    virtual void setRelativeWeight(double relWeight);
    virtual Weight_T getWeight(double bipolarPosGain = 1.0) const;
    virtual int getDigitalWeight(double threshold) const;
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
    virtual ~Synapse_RRAM() {};

    /// Is the synapse bipolar?
    const bool bipolar;
    /// Synaptic delay
    Time_T delay;
    Weight_T weightMinGlobalMean;
    Weight_T weightMaxGlobalMean;
    std::vector<Device> devices;

    // Stats
    /// Number of read events on the synapse since last clearStats() call
    /// (shared between the RRAM devices).
    unsigned long long int statsReadEvents;
    std::vector<DeviceStats> stats;

private:
    static ProgramMethod mProgramMethod;
};
}

namespace {
template <>
const char* const EnumStrings<N2D2::Synapse_RRAM::ProgramMethod>::data[]
    = {"Ideal", "IdealProbabilistic", "SetReset", "SetResetProbabilistic"};
}

double N2D2::Synapse_RRAM::getRelativeWeight(bool allowBipolarRange) const
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

#endif // N2D2_SYNAPSE_RRAM_H
