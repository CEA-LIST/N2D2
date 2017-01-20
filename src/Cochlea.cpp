/*
    (C) Copyright 2010 CEA LIST. All Rights Reserved.
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

#include "Cochlea.hpp"

N2D2::Cochlea::Cochlea(unsigned int nbChannels)
    : mNbChannels(nbChannels), mNormalize(this, "Normalize", true)
{
    // ctor
}

unsigned int N2D2::Cochlea::load(const std::string& fileName,
                                 unsigned int order,
                                 double lowFreq,
                                 double upFreq,
                                 double threshold,
                                 Time_T leak,
                                 Time_T refractory,
                                 FilterSpace filterSpace,
                                 double earQ,
                                 double minBw,
                                 double start,
                                 double end)
{
    Sound audio;
    audio.load(fileName, start, end);

    if (mNormalize)
        audio.normalize(0, 1.0 / Utils::rms(audio(0)));

    const Time_T dt = (Time_T)(TimeS / audio.getSamplingFrequency());
    const double expLeak
        = (leak > 0.0) ? std::exp(-((double)dt) / ((double)leak)) : 1.0;
    const Sound::Filter_T filterLowPass
        = audio.newFilter(Sound::Butterworth, Sound::LowPass, 1, 65);

    Aer::AerData_T events;
    std::vector<unsigned int> nbEvents;
    nbEvents.reserve(mNbChannels);

#pragma omp parallel for ordered schedule(dynamic)
    for (int i = 0; i < (int)mNbChannels; ++i) {
        const double centerFreq
            = (filterSpace == LinearSpace)
                  ? lowFreq + (upFreq - lowFreq) * i / (mNbChannels - 1)
                  : -earQ * minBw
                    + (lowFreq + earQ * minBw)
                      * std::exp((std::log(upFreq + earQ * minBw)
                                  - std::log(lowFreq + earQ * minBw)) * i
                                 / (double)mNbChannels);

        const double freqBand = (earQ > 0.0) ? centerFreq / earQ + minBw
                                             : minBw;

        Sound filteredAudio(audio);

        if (order > 0) {
            // Use only order 1 & order 2 filters to ensure stability at all
            // frequency ranges!
            const Sound::Filter_T filter
                = filteredAudio.newFilter(Sound::Butterworth,
                                          Sound::BandPass,
                                          2,
                                          centerFreq - freqBand / 2.0,
                                          centerFreq + freqBand / 2.0);

            for (unsigned int f = 0; f < order / 2; ++f)
                filteredAudio.applyFilter(filter);

            if (order % 2 == 1) {
                const Sound::Filter_T filter1
                    = filteredAudio.newFilter(Sound::Butterworth,
                                              Sound::BandPass,
                                              1,
                                              centerFreq - freqBand / 2.0,
                                              centerFreq + freqBand / 2.0);
                filteredAudio.applyFilter(filter1);
            }
        } else {
            // Use Gammatone filters
            Sound::Filter_T filter1, filter2, filter3, filter4;
            std::tie(filter1, filter2, filter3, filter4)
                = filteredAudio.newGammatoneFilter(centerFreq, freqBand);

            filteredAudio.applyFilter(filter1);
            filteredAudio.applyFilter(filter2);
            filteredAudio.applyFilter(filter3);
            filteredAudio.applyFilter(filter4);
        }

        /// Half-wave rectification and then low-pass filter, suggested by
        /// Daniel Pressnitzer
        filteredAudio.halfWaveRectify();
        /// !!! No signal compression (usually x^(1.0/3.0), as seen in Brian
        /// Hears), see SPECIAL MODEL !!!
        filteredAudio.applyFilter(filterLowPass);
        /*
                // DEBUG
                std::ostringstream audioFile;
                audioFile << "signal_" << i << ".dat";
                //filteredAudio.save(audioFile.str());
                filteredAudio.saveSignal(audioFile.str());

                audioFile.str(std::string());
                audioFile << "filter_" << i << ".dat";
                filteredAudio.saveFilterResponse(filter, audioFile.str());

                audioFile.str(std::string());
                audioFile << "spectrogram_" << i << ".dat";
                filteredAudio.spectrogram(audioFile.str(), 0, 0, Hann<double>(),
           0, 0, true);
        */
        Time_T refractoryEnd = 0;
        double integration = 0.0;
        Aer::AerData_T filterEvents;
        /*
                // DEBUG
                std::ostringstream audioGraph;
                audioGraph << "test_" << i;
                std::ofstream graph(std::string(audioGraph.str() +
           ".dat").c_str());
        */
        Time_T timestamp = 0;

        /// !!! SPECIAL MODEL: relative bandwidth dependent threshold instead of
        /// signal compression !!!
        const double freqThres
            = (earQ > 0.0)
                  ? threshold
                    * std::pow(freqBand / (lowFreq / earQ + minBw), 1.0 / 3.0)
                  : threshold;

        for (std::vector<double>::const_iterator it = filteredAudio(0).begin(),
                                                 itEnd = filteredAudio(0).end();
             it != itEnd;
             ++it) {
            // DEBUG
            // graph << timestamp/(double) TimeS << " " << (*it) << " " <<
            // integration << " " << std::endl;

            if (freqThres > 0.0) {
                if (timestamp >= refractoryEnd)
                    integration = integration * expLeak
                                  + (*it)
                                    * (1.0 / audio.getSamplingFrequency());

                if (integration >= freqThres) {
                    integration = 0.0;
                    refractoryEnd = timestamp + refractory;

                    filterEvents.push_back(
                        std::make_pair(timestamp, AerEvent::unmaps(0, 0, i)));
                }
            } else {
                if (-(*it) * freqThres > Random::randUniform())
                    filterEvents.push_back(
                        std::make_pair(timestamp, AerEvent::unmaps(0, 0, i)));
            }

            timestamp += dt;
        }
/*
        // DEBUG
        graph.close();

        Gnuplot gnuplot;
        gnuplot.set("grid").set("key off");
        gnuplot.setXrange(0, 1.0);
        gnuplot.saveToFile(audioGraph.str());
        gnuplot.plot(audioGraph.str() + ".dat", "using 1:2 with lines,"
            "\"\" using 1:3 with lines");
*/
#pragma omp ordered
        {
            events.insert(
                events.end(), filterEvents.begin(), filterEvents.end());
            nbEvents.push_back(filterEvents.size());
            std::cout << "[loadCochlea] input #" << i << ": "
                      << filterEvents.size() << " events" << std::endl;
        }
    }

    std::sort(events.begin(),
              events.end(),
              Utils::PairFirstPred<Time_T, unsigned int>());

    std::string shortName = Utils::fileBaseName(fileName);

    if (threshold <= 0.0)
        shortName += "-stoch";

    Aer::save(shortName + ".dat", events);

    if (!events.empty()) {
        Gnuplot gnuplot;
        gnuplot.set("key off");
        gnuplot.set("bars", 0);
        gnuplot.set("pointsize", 0.01);
        gnuplot.setXlabel("Time (s)");
        gnuplot.setYlabel("Channel");
        gnuplot.setYrange(0.0, (double)mNbChannels);
        gnuplot.saveToFile(shortName);
        gnuplot.plot("-", "using 1:2:($2+0.8):($2) with yerrorbars");

        for (Aer::AerData_T::const_iterator it = events.begin(),
                                            itEnd = events.end();
             it != itEnd;
             ++it) {
            std::stringstream cmd;
            cmd << std::setprecision(std::numeric_limits<double>::digits10 + 1)
                << ((*it).first / (double)TimeS) << " " << (*it).second;
            gnuplot << cmd.str();
        }

        gnuplot << "e";

        const double duration = audio.getDuration();

        gnuplot.saveToFile(shortName + "-events");
        gnuplot.set("style data boxes").set("style fill solid border -1");
        gnuplot.set("boxwidth", 0.9);
        gnuplot.setXrange(0 - 0.5, mNbChannels + 0.5);
        gnuplot.set("yrange [*:*]");
        gnuplot.set("yrange [0:]");
        gnuplot.setXlabel("Channel");
        gnuplot.setYlabel("Events / second");

        std::ostringstream plotCmd;
        plotCmd << "using 1:2 with boxes, "
                << events.size() / (double)mNbChannels / duration
                << " with lines";
        gnuplot.plot("-", plotCmd.str());

        for (int i = 0; i < (int)mNbChannels; ++i) {
            std::stringstream cmd;
            cmd << i << " " << nbEvents[i] / duration;
            gnuplot << cmd.str();
        }

        gnuplot << "e";
    }

    std::cout << "[loadCochlea] *** " << events.size()
              << " events generated for " << mNbChannels << " inputs"
              << std::endl;
    return events.size();
}
