/*
    (C) Copyright 2011 CEA LIST. All Rights Reserved.
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

#ifndef N2D2_SOUND_H
#define N2D2_SOUND_H

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "Network.hpp"
#include "utils/DSP.hpp"
#include "utils/Gnuplot.hpp"
#include "utils/Random.hpp"
#include "utils/Utils.hpp"
#include "utils/WindowFunction.hpp"

namespace N2D2 {
/**
 * Signal processing (and more specifically sound processing) facilities for
 * N2D2. Supports the WAV file format for read and
 * write.
*/
class Sound {
public:
    enum FilterType {
        Butterworth,
        Chebyshev
    };
    enum FilterFunction {
        LowPass,
        HighPass,
        BandPass,
        BandStop
    };
    struct AgusBlockType {
        enum Value {
            N,
            RN,
            RefRN
        };
    };
    typedef long double Real_T; // Yes, long double is needed for stability!
    typedef std::pair<std::vector<Real_T>, std::vector<Real_T> > Filter_T;
    typedef std::complex<Real_T> Complex_T;

    Sound(unsigned int samplingFrequency = 44100,
          unsigned short bitPerSample = 16);
    Sound(const std::vector<double>& data,
          unsigned int samplingFrequency = 44100);
    Sound(const Sound& sound, double start = 0.0, double end = 0.0);
    template <class InputIterator>
    Sound(InputIterator first,
          InputIterator last,
          unsigned int samplingFrequency = 0);
    std::vector<double>& operator()(unsigned int channel = 0);
    unsigned int getNbChannels() const
    {
        return mData.size();
    }
    void
    load(const std::string& fileName, double start = 0.0, double end = 0.0);
    void loadSignal(const std::string& fileName,
                    unsigned int samplingFrequency = 0);
    void normalize(unsigned int channel, double value = 0.0);

    /**
     * Reverses the signal (the first sample becomes the last and inversely).
     *
     * @param channel   Audio channel
    */
    void reverse(unsigned int channel = 0);

    /**
     * Increase the sampling rate by an integer factor.
     * Increases the sampling rate of channel @p channel by inserting @p p-1
     *zeros between samples. If N is the initial number
     * of samples, the number of samples after upsampling is p*N.
     *
     * @param channel   Audio channel
     * @param p         Upsampling factor
    */
    void upsample(unsigned int channel = 0, unsigned int p = 1);

    /**
     * Decrease the sampling rate by an integer factor.
     * Decreases the sampling rate of channel @p channel by keeping every @p
     *q-th sample starting with the first sample. If N is
     * the initial number of samples, the number of samples after downsampling
     *is ceil(N/q).
     *
     * @param channel   Audio channel
     * @param q         Downsampling factor
    */
    void downsample(unsigned int channel = 0, unsigned int q = 1);

    /**
     * Change the sampling rate by a rational factor. Upsample, apply a
     *specified FIR filter, and downsample a signal.
     *
     * @param channel   Audio channel
     * @param p         Upsampling factor
     * @param q         Downsampling factor
     * @param n         The length of the FIR filter resample uses is
     *proportional to @p n; larger values of @p n provide better
     *                  accuracy at the expense of more computation time.
     * @param beta      @p beta is the design parameter for the Kaiser window
     *that resample employs in designing the lowpass
     *                  filter.
    */
    void resample(unsigned int channel = 0,
                  unsigned int p = 1,
                  unsigned int q = 1,
                  unsigned int n = 10,
                  double beta = 5.0);
    Filter_T newFirFilter(FilterFunction func,
                          unsigned int n,
                          const WindowFunction<Real_T>& wFunction,
                          double cornerFreq1,
                          double cornerFreq2 = 0.0) const;
    void halfWaveRectify(unsigned int channel = 0, double compression = 1.0);
    void fullWaveRectify(unsigned int channel = 0, double compression = 1.0);
    void save(const std::string& fileName,
              bool normalize = false,
              double normalizeValue = 0.0) const;
    void saveSignal(const std::string& fileName,
                    unsigned int channel = 0,
                    bool plot = true) const;
    void genNoise(unsigned int channel, Time_T duration);
    std::vector<unsigned int> genNoisePattern(unsigned int channel,
                                              Time_T duration,
                                              unsigned int nbChunk,
                                              unsigned int nbPattern);

    /**
     * Generate a sound using the same protocol as experiment #1 of the Agus
     *paper.
     *
     * See T. R. Agus, S. J. Thorpe and D. Pressnitzer, "Rapid formation of
     *auditory memories: Insights from noise",
     * Neuron (2010), 66: 610-618.
     *
     * @param channel           Audio channel
     * @param trialDuration     Duration of each trial part of the block
     * @param nbRepetition      Number of repetitions in the repetitive trials
     * @param nbN               Total number of N trials
     * @param nbRn              Total number of RN trials
     * @param nbRefRn           Total number of RefRN trials
     * @param refRn             Vector containing the RefRN trial (if empty, a
     *random RefRN trial is generated)
     *
     * @return                  Vector containing the description of the
     *sequence
    */
    std::vector<unsigned int>
    genAgusExperimentalBlock(unsigned int channel,
                             Time_T trialDuration = 1 * TimeS,
                             unsigned int nbRepetition = 2,
                             unsigned int nbN = 100,
                             unsigned int nbRn = 50,
                             unsigned int nbRefRn = 50,
                             const std::vector<double>& refRn = std::vector
                             <double>());
    void genAgusExperimentalBlock(unsigned int channel,
                                  Time_T trialDuration,
                                  unsigned int nbRepetition,
                                  const std::vector<unsigned int>& block,
                                  const std::vector<double>& refRn = std::vector
                                  <double>());
    void distortAgusExperimentalBlock(unsigned int channel,
                                      Time_T trialDuration,
                                      unsigned int nbRepetition,
                                      const std::vector<unsigned int>& block,
                                      double distortRatio);
    void loadTrialAgusExperimentalBlock(const std::string& fileName,
                                        double normalizeValue,
                                        AgusBlockType::Value type,
                                        unsigned int nbRepetition,
                                        unsigned int position = 0);
    Filter_T newFilter(FilterType filter,
                       FilterFunction func,
                       unsigned int order,
                       double cornerFreq1,
                       double cornerFreq2 = 0.0,
                       double chebyshevRipple = 0.0) const;

    /**
     * Create a new gammatone filter (in fact, a set of 4 second order filters,
     *to be applied successively)
     * Adapted from the Matlab function MakeERBFilters(), by Malcolm Slaney at
     *Interval (June 11, 1998)
     *
     * @param centerFreq        Center frequency of the filter
     * @param bandwidth         Bandwidth of the filter
     * @return                  Tuple of 4 second order filters
    */
    std::tuple<Filter_T, Filter_T, Filter_T, Filter_T>
    newGammatoneFilter(double centerFreq, double bandwidth) const;
    void saveFilterResponse(const Filter_T& filter,
                            const std::string& fileName,
                            unsigned int nbSteps = 1000,
                            bool append = false,
                            bool plot = true) const;
    void saveWindowFunction(const WindowFunction<double>& wFunction,
                            const std::string& fileName,
                            unsigned int n = 100,
                            bool append = false,
                            bool plot = true) const;
    void applyFilter(const Filter_T& filter,
                     unsigned int channel = 0,
                     bool appendTrailing = false);
    std::vector<std::vector<double> >
    spectrogram(unsigned int channel = 0,
                unsigned int nFft = 0,
                const WindowFunction<double>& wFunction = Hann<double>(),
                unsigned int wSize = 0,
                int nOverlap = 0,
                bool logScale = true) const;
    void spectrogram(const std::string& fileName,
                     unsigned int channel = 0,
                     unsigned int nFft = 0,
                     const WindowFunction<double>& wFunction = Hann<double>(),
                     unsigned int wSize = 0,
                     int nOverlap = 0,
                     bool logScale = true,
                     bool plot = true) const;

    /**
     * Returns the sampling frequency, in hertz.
     *
     * @return Sampling frequency
    */
    unsigned int getSamplingFrequency() const
    {
        return mSamplingFrequency;
    };

    /**
     * Returns the duration of the sound, in seconds.
     *
     * @param channel           Channel number (has no effect if all the
     *channels have the same duration)
     * @return Duration of the sound for the given channel
    */
    double getDuration(unsigned int channel = 0) const;

    /**
     * Clear the audio data for all channels.
    */
    void clear();

    static void saveDescriptor(const std::vector<unsigned int>& desc,
                               const std::string& fileName,
                               bool plot = true);
    static std::vector<unsigned int> loadDescriptor(const std::string
                                                    & fileName);
    static void saveActivityFromXlabel(const std::string& xlabelFileName,
                                       const std::string& activityFileName,
                                       Time_T offset = 0);

private:
    template <typename T> static T realPart(const std::complex<T>& z);
    template <typename T>
    static std::complex<T> bilinearTransform(const std::complex<T>& z);
    template <typename T>
    static std::vector<std::complex<T> >
    expandPolynomial(const std::vector<std::complex<T> >& pz);
    template <typename T1, typename T2>
    static std::complex<T1> evaluate(const std::vector<T2>& num,
                                     const std::vector<T2>& denom,
                                     const std::complex<T1>& z);
    template <typename T1, typename T2>
    static std::complex<T1> evaluatePolynomial(const std::vector<T2>& poly,
                                               const std::complex<T1>& z);

    unsigned int mSamplingFrequency;
    unsigned short mBitPerSample;
    std::vector<std::vector<double> > mData;
};
}

namespace {
template <>
const char* const EnumStrings<N2D2::Sound::FilterType>::data[]
    = {"Butterworth", "Chebyshev"};
template <>
const char* const EnumStrings<N2D2::Sound::FilterFunction>::data[]
    = {"LowPass", "HighPass", "BandPass", "BandStop"};
}

template <class InputIterator>
N2D2::Sound::Sound(InputIterator first,
                   InputIterator last,
                   unsigned int samplingFrequency)
    : mData(1, std::vector<double>(first, last))
{
    if (samplingFrequency > 0)
        mSamplingFrequency = samplingFrequency;
}

template <typename T> T N2D2::Sound::realPart(const std::complex<T>& z)
{
    return z.real();
}

template <typename T>
std::complex<T> N2D2::Sound::bilinearTransform(const std::complex<T>& z)
{
    return ((T)2.0 + z) / ((T)2.0 - z);
}

template <typename T>
std::vector<std::complex<T> >
N2D2::Sound::expandPolynomial(const std::vector<std::complex<T> >& pz)
{
    std::vector<std::complex<T> > coeffs(pz.size() + 1, 0.0);
    coeffs[0] = 1.0;

    for (unsigned int i = 0, size = pz.size(); i < size; ++i) {
        for (unsigned int j = pz.size(); j >= 1; --j)
            coeffs[j] = (-pz[i] * coeffs[j]) + coeffs[j - 1];

        coeffs[0] *= -pz[i];
    }

    for (unsigned int i = 0, size = pz.size() + 1; i < size; ++i) {
        if (std::fabs(coeffs[i].imag()) > 1e-10)
            throw std::runtime_error("Poles/zeros are not complex conjugates");
    }

    return coeffs;
}

template <typename T1, typename T2>
std::complex<T1> N2D2::Sound::evaluate(const std::vector<T2>& num,
                                       const std::vector<T2>& denom,
                                       const std::complex<T1>& z)
{
    return evaluatePolynomial(num, z) / evaluatePolynomial(denom, z);
}

template <typename T1, typename T2>
std::complex<T1> N2D2::Sound::evaluatePolynomial(const std::vector<T2>& poly,
                                                 const std::complex<T1>& z)
{
    std::complex<T1> val(0.0);

    for (int i = poly.size() - 1; i >= 0; --i)
        val = (val * z) + poly[i];

    return val;
}

#endif // N2D2_SOUND_H
