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

#include "Xnet/SpikeGenerator.hpp"

N2D2::SpikeGenerator::SpikeGenerator()
    : // IMPORTANT: Do not change the value of the parameters here! Use
      // setParameter() or loadParameters().
      mStimulusType(this, "StimulusType", SingleBurst),
      mDiscardedLateStimuli(this, "DiscardedLateStimuli", 1.0),
      mPeriodMeanMin(this, "PeriodMeanMin", 50 * TimeMs),
      mPeriodMeanMax(this, "PeriodMeanMax", 12 * TimeS),
      mPeriodRelStdDev(this, "PeriodRelStdDev", 0.1),
      mPeriodMin(this, "PeriodMin", 11 * TimeMs),
      mMaxFrequency(this, "MaxFrequency", 1.0 / TimeNs),
      mNbQuantizationLevels(2)
{
    // ctor
}

void N2D2::SpikeGenerator::checkParameters() const
{
    // Parameter checks
    if (mPeriodMeanMin >= mPeriodMeanMax)
        throw std::runtime_error(
            "Environment: PeriodMeanMin must be lower than PeriodMeanMax");

    if (mPeriodMin > mPeriodMeanMin)
        throw std::runtime_error("Environment: PeriodMin must be lower than or "
                                 "equal to PeriodMeanMin");

    if (mDiscardedLateStimuli < 0.0 || mDiscardedLateStimuli > 1.0)
        throw std::domain_error("Environment: DiscardedLateStimuli is out of "
                                "range (must be >= 0.0 and <= 1.0)");
}

void N2D2::SpikeGenerator::nextEvent(std::pair<Time_T, int>& event,
                                     double value,
                                     Time_T start,
                                     Time_T end) const
{
    const double delay = 1.0 - std::fabs(value);
    const bool negSpike = (value < 0);

    const unsigned int quantizationLevel = quantize(value);
    const double freq =
        ((double)quantizationLevel)/(mNbQuantizationLevels-1) * mMaxFrequency;

    if (delay > mDiscardedLateStimuli)
        return;

    if (mStimulusType == SingleBurst) {
        if (event.second == 0) {
            const Time_T t = (Time_T)(start + delay * (end - start));
            event = std::make_pair(t, (negSpike) ? -1 : 1);
        } else
            event = std::make_pair(0, 0);
    } else {
        const double freqMeanMax = 1.0 / mPeriodMeanMin;
        const double freqMeanMin = 1.0 / mPeriodMeanMax;
        // value = 0 => most significant => maximal frequency (or minimal
        // period)
        const Time_T periodMean = (Time_T)(
            1.0 / (freqMeanMax + (freqMeanMin - freqMeanMax) * delay));

        Time_T t = event.first;
        Time_T dt = 0;

        if (mStimulusType == Poissonian) {
            // The time between arrivals in a Poisson process is exponentially
            // distributed
            dt = (Time_T)Random::randExponential(periodMean);
        }
        else if (mStimulusType == Linear) {
            if (freq >= 1.0/(end-start)){
                dt = (Time_T) std::llround(1.0/freq);
                /*if (t == 0) {
                    dt = (Time_T) std::llround(0.5/freq);
                }*/
            }
            else
                dt = end + 1;
        }
        else {
            dt = (Time_T)Random::randNormal(periodMean,
                                            periodMean * mPeriodRelStdDev);

            if (mStimulusType == JitteredPeriodic && (event.second == 0))
                dt *= Random::randUniform();
        }

        if (t > start && dt < mPeriodMin)
            dt = mPeriodMin;

        t += dt;

        if (t <= end)
            event = std::make_pair(t, (negSpike) ? -1 : 1);
        else
            event = std::make_pair(0, 0);
    }
}

unsigned int N2D2::SpikeGenerator::quantize(double value) const
{
    unsigned int level = std::round(std::fabs(value) * (mNbQuantizationLevels-1));
    return level;
}

N2D2::SpikeGenerator::~SpikeGenerator()
{
    // dtor
}
