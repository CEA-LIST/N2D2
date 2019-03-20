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

#ifndef N2D2_SPIKEGENERATOR_H
#define N2D2_SPIKEGENERATOR_H

#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "Network.hpp"
#include "utils/Parameterizable.hpp"
#include "utils/Utils.hpp"

namespace N2D2 {
class SpikeGenerator : virtual public Parameterizable {
public:
    enum StimulusType {
        SingleBurst,
        Periodic,
        JitteredPeriodic,
        Poissonian,
        Linear
    };

    void setNbQuantizationLevels(unsigned int nb)
    {
        mNbQuantizationLevels = nb;
    }

    SpikeGenerator();
    virtual ~SpikeGenerator();

protected:
    void checkParameters() const;
    void nextEvent(std::pair<Time_T, int>& event,
                   double value,
                   Time_T start,
                   Time_T end) const;
    unsigned int quantize(double value) const;

    // Parameters
    Parameter<StimulusType> mStimulusType;
    /// The pixels in the pre-processed stimuli with a value above this limit
    /// never generate spiking events
    Parameter<double> mDiscardedLateStimuli;
    /// Mean minimum period \f$\overline{T_{min}}\f$, used for periodic temporal
    /// codings, corresponding to pixels in the
    /// pre-processed stimuli with a
    /// value of 0 (which are supposed to be the most significant pixels)
    Parameter<Time_T> mPeriodMeanMin;
    /// Mean maximum period \f$\overline{T_{max}}\f$, used for periodic temporal
    /// codings, corresponding to pixels in the
    /// pre-processed stimuli with a
    /// value of 1 (which are supposed to be the least significant pixels). This
    /// maximum period may be never reached if @p
    /// mDiscardedLateStimuli is lower than 1
    Parameter<Time_T> mPeriodMeanMax;
    /// Relative standard deviation, used for periodic temporal codings, applied
    /// to the spiking period of a pixel
    Parameter<double> mPeriodRelStdDev;
    /// Absolute minimum period, or spiking interval, used for periodic temporal
    /// codings, for any pixel
    Parameter<Time_T> mPeriodMin;

    Parameter<double> mMaxFrequency;

    unsigned int mNbQuantizationLevels;
};
}

namespace {
template <>
const char* const EnumStrings<N2D2::SpikeGenerator::StimulusType>::data[]
    = {"SingleBurst", "Periodic", "JitteredPeriodic", "Poissonian", "Linear"};
}

#endif // N2D2_SPIKEGENERATOR_H
