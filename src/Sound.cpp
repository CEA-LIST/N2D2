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

#include "Sound.hpp"

N2D2::Sound::Sound(unsigned int samplingFrequency, unsigned short bitPerSample)
    : mSamplingFrequency(samplingFrequency), mBitPerSample(bitPerSample)
{
    // ctor
    mData.resize(1);
}

N2D2::Sound::Sound(const std::vector<double>& data,
                   unsigned int samplingFrequency)
{
    mData.push_back(data);
    mSamplingFrequency = samplingFrequency;
}

N2D2::Sound::Sound(const Sound& sound, double start, double end)
    : mSamplingFrequency(sound.mSamplingFrequency),
      mBitPerSample(sound.mBitPerSample)
{
    mData.resize(sound.mData.size());

    const unsigned int startSample = (unsigned int)(start * mSamplingFrequency);
    const unsigned int endSample = (unsigned int)(end * mSamplingFrequency);

    // For each channel
    for (unsigned int channel = 0, nbChannels = mData.size();
         channel < nbChannels;
         ++channel) {
        if (startSample > sound.mData[channel].size())
            throw std::out_of_range(
                "Start extraction time higher than the sound duration");

        if (endSample > sound.mData[channel].size())
            throw std::out_of_range(
                "End extraction time higher than the sound duration");

        mData[channel].assign(sound.mData[channel].begin() + startSample,
                              (endSample > 0)
                                  ? sound.mData[channel].begin() + endSample
                                  : sound.mData[channel].end());
    }
}

std::vector<double>& N2D2::Sound::operator()(unsigned int channel)
{
    return mData.at(channel);
}

void N2D2::Sound::load(const std::string& fileName, double start, double end)
{
    std::ifstream data(fileName.c_str(), std::fstream::binary);

    if (!data.good())
        throw std::runtime_error("Could not open sound file: " + fileName);

    std::string chunkId(4, 0);
    unsigned int chunkSize;

    // RIFF chunk
    data.read(reinterpret_cast<char*>(&chunkId[0]), 4);

    if (!data.good())
        throw std::runtime_error("Unreadable sound file: " + fileName);

    if (chunkId != "RIFF")
        throw std::runtime_error("Unknown sound file format (\""
                                 + Utils::escapeBinary(chunkId)
                                 + "\") in file: " + fileName);

    data.read(reinterpret_cast<char*>(&chunkSize), 4);

    if (!data.good())
        throw std::runtime_error("Unreadable sound file: " + fileName);

    // Check the size of the file
    const std::streampos pos = data.tellg(); // Get current position
    data.seekg(0, data.end); // Get end-of-file position

    if ((unsigned int)(data.tellg() - pos) != chunkSize)
        throw std::runtime_error(
            "Invalid file size (the file may be corrupted): " + fileName);

    data.seekg(pos); // Get back to previous position

    data.read(reinterpret_cast<char*>(&chunkId[0]), 4);

    if (chunkId != "WAVE") {
        throw std::runtime_error(
            "Unknown RIFF type (\"" + Utils::escapeBinary(chunkId)
            + "\"), only WAVE is supported, in file: " + fileName);
    }

    while (data.read(reinterpret_cast<char*>(&chunkId[0]), 4)) {
        data.read(reinterpret_cast<char*>(&chunkSize), 4);

        if (chunkId == "fmt ") {
            unsigned short format;
            unsigned short channels;
            unsigned int bytePerSecond;
            unsigned short bytePerBlock;

            data.read(reinterpret_cast<char*>(&format), 2);

            if (format != 1)
                throw std::runtime_error("Unknown sound file format in file: "
                                         + fileName);

            data.read(reinterpret_cast<char*>(&channels), 2);
            mData.resize(channels);

            data.read(reinterpret_cast<char*>(&mSamplingFrequency), 4);
            data.read(reinterpret_cast<char*>(&bytePerSecond), 4);
            data.read(reinterpret_cast<char*>(&bytePerBlock), 2);
            data.read(reinterpret_cast<char*>(&mBitPerSample), 2);

            if (bytePerSecond != mSamplingFrequency * bytePerBlock)
                throw std::runtime_error("Invalid sound file (bytePerSecond != "
                                         "mSamplingFrequency*bytePerBlock): "
                                         + fileName);

            if (bytePerBlock != mData.size() * mBitPerSample / 8)
                throw std::runtime_error("Invalid sound file (bytePerBlock != "
                                         "mData.size()*mBitPerSample/8): "
                                         + fileName);
        } else if (chunkId == "data") {
            // For each channel
            for (std::vector<std::vector<double> >::iterator it = mData.begin(),
                                                             itEnd
                                                             = mData.end();
                 it != itEnd;
                 ++it) {
                // Make sure it's empty (if the object was already used)
                (*it).clear();
                // Reserve the memory for the samples
                (*it).reserve(chunkSize / mData.size() / (mBitPerSample / 8));
            }

            const unsigned int nbSamples = chunkSize / mData.size()
                                           / (mBitPerSample / 8);
            const unsigned int startSample
                = (unsigned int)(start * mSamplingFrequency);
            const unsigned int endSample
                = (end > 0.0) ? (unsigned int)(end * mSamplingFrequency)
                              : nbSamples;

            if (startSample > nbSamples)
                throw std::out_of_range("Start extraction time higher than the "
                                        "sound duration for file: " + fileName);

            if (endSample > nbSamples)
                throw std::out_of_range("End extraction time higher than the "
                                        "sound duration for file: " + fileName);

            int byte;

            // Discard samples before start time
            for (unsigned int s = 0; s < startSample; ++s) {
                for (unsigned int i = 0, size = mData.size(); i < size; ++i)
                    data.read(reinterpret_cast<char*>(&byte),
                              mBitPerSample / 8);
            }

            // For each sample
            for (unsigned int s = startSample; s < endSample; ++s) {
                // For each channel
                for (std::vector<std::vector<double> >::iterator it
                     = mData.begin(),
                     itEnd = mData.end();
                     it != itEnd;
                     ++it) {
                    data.read(reinterpret_cast<char*>(&byte),
                              mBitPerSample / 8);

                    if (mBitPerSample == 8) {
                        if (byte & 0x80)
                            byte |= ~0xFF;
                        else
                            byte &= 0xFF;
                    } else if (mBitPerSample == 16) {
                        if (byte & 0x8000)
                            byte |= ~0xFFFF;
                        else
                            byte &= 0x0000FFFF;
                    } else if (mBitPerSample == 24) {
                        if (byte & 0x800000)
                            byte |= ~0xFFFFFF;
                        else
                            byte &= 0xFFFFFF;
                    }

                    // Append the sample to the channel
                    (*it).push_back(byte);
                }
            }

            // Discard samples after end time
            for (unsigned int s = endSample; s < nbSamples; ++s) {
                for (unsigned int i = 0, size = mData.size(); i < size; ++i)
                    data.read(reinterpret_cast<char*>(&byte),
                              mBitPerSample / 8);
            }
        } else {
            std::cout << "Notice: Unsupported WAV file chunk (\""
                      << Utils::escapeBinary(chunkId)
                      << "\") in file: " << fileName << std::endl;
            data.ignore(chunkSize);
        }
    }
}

void N2D2::Sound::loadSignal(const std::string& fileName,
                             unsigned int samplingFrequency)
{
    std::ifstream data(fileName.c_str());

    if (!data.good())
        throw std::runtime_error("Could not load signal data file: "
                                 + fileName);

    std::copy(std::istream_iterator<double>(data),
              std::istream_iterator<double>(),
              std::back_inserter(mData[0]));

    if (samplingFrequency > 0)
        mSamplingFrequency = samplingFrequency;
}

void N2D2::Sound::normalize(unsigned int channel, double value)
{
    if (value == 0.0) {
        for (std::vector<double>::const_iterator it = mData[channel].begin(),
                                                 itEnd = mData[channel].end();
             it != itEnd;
             ++it)
            value = std::max(std::fabs(*it), value);

        value = 1.0 / value;
    }

    std::transform(
        mData[channel].begin(),
        mData[channel].end(),
        mData[channel].begin(),
        std::bind(std::multiplies<double>(), std::placeholders::_1, value));
}

void N2D2::Sound::reverse(unsigned int channel)
{
    std::reverse(mData[channel].begin(), mData[channel].end());
}

void N2D2::Sound::upsample(unsigned int channel, unsigned int p)
{
    if (p < 1)
        throw std::runtime_error("Upsampling factor must be >= 1.");

    const unsigned int size = mData[channel].size();
    std::vector<double> data(size * p, 0.0);

    for (unsigned int i = 0; i < size; ++i)
        data[i * p] = mData[channel][i];

    mData[channel] = data;
    mSamplingFrequency *= p;
}

void N2D2::Sound::downsample(unsigned int channel, unsigned int q)
{
    if (q < 1)
        throw std::runtime_error("Downsampling factor must be >= 1.");

    const unsigned int size
        = (unsigned int)std::ceil(mData[channel].size() / (double)q);
    std::vector<double> data(size, 0.0);

    for (unsigned int i = 0; i < size; ++i)
        data[i] = mData[channel][i * q];

    mData[channel] = data;
    mSamplingFrequency /= q;
}

void N2D2::Sound::resample(unsigned int channel,
                           unsigned int p,
                           unsigned int q,
                           unsigned int n,
                           double beta)
{
    const unsigned int size = mData[channel].size();
    const unsigned int resampledSize
        = (unsigned int)std::ceil(size * p / (double)q);

    // (1) Upsampling by p (zero insertion). p defaults to 1 if not specified.
    upsample(channel, p);

    // (2) Anti-aliasing (lowpass) FIR filtering
    const unsigned int filterSize = 2 * n * std::max(p, q) + 1;
    Filter_T filter = newFirFilter(LowPass,
                                   filterSize,
                                   Kaiser<Real_T>(beta),
                                   mSamplingFrequency / (2.0 * std::max(p, q)));

    // DEBUG
    // saveFilterResponse(filter, "filter.dat");
    // saveWindowFunction(Kaiser<double>(beta), "window.dat");

    std::transform(
        filter.first.begin(),
        filter.first.end(),
        filter.first.begin(),
        std::bind(std::multiplies<Real_T>(), p, std::placeholders::_1));

    // Need to delay output so that downsampling by q hits center tap of filter.
    unsigned int filterHalfSize = (filterSize - 1) / 2;
    const unsigned int zeroPad = q - (filterHalfSize % q);
    filter.first.insert(filter.first.begin(), zeroPad, 0.0);
    filterHalfSize += zeroPad;

    // Number of samples removed from beginning of output sequence to compensate
    // for delay of linear phase filter:
    const unsigned int delay = filterHalfSize / q;

    // Need to zero-pad so output length is exactly resampledSize.
    unsigned int zeroPadEnd = 0;

    while (std::ceil(((size - 1) * p + filter.first.size() + zeroPadEnd)
                     / (double)q) < delay + resampledSize)
        zeroPadEnd += 1;

    filter.first.insert(filter.first.end(), zeroPadEnd, 0.0);

    // DEBUG
    // saveFilterResponse(filter, "filter_padded.dat");

    // Apply filter
    applyFilter(filter, channel, true);

    // (3) Downsampling by Q (throwing away samples). q defaults to 1 if not
    // specified.
    downsample(channel, q);

    // Get rid of trailing and leading data so input and output signals line up
    // temporally:
    mData[channel]
        .erase(mData[channel].begin(), mData[channel].begin() + delay - 1);
    mData[channel]
        .erase(mData[channel].begin() + resampledSize, mData[channel].end());
}

N2D2::Sound::Filter_T N2D2::Sound::newFirFilter(FilterFunction func,
                                                unsigned int n,
                                                const WindowFunction
                                                <Real_T>& wFunction,
                                                double cornerFreq1,
                                                double cornerFreq2) const
{
    if (cornerFreq1 > mSamplingFrequency / 2.0 || cornerFreq2
                                                  > mSamplingFrequency / 2.0)
        throw std::runtime_error(
            "Corner frequency is higher than the Nyquist frequency!");

    const Real_T w1 = 2.0 * M_PI * (cornerFreq1 / mSamplingFrequency);
    // const Real_T w2 = 2.0 * M_PI * (cornerFreq2 / mSamplingFrequency);

    std::vector<Real_T> h;
    h.reserve(n);

    if (func == LowPass) {
        for (unsigned int k = 0; k < n; ++k) {
            Real_T x = k - std::floor(n / 2.0) + ((n + 1) % 2) / 2.0;
            h.push_back((x == 0) ? w1 / M_PI : std::sin(w1 * x) / (M_PI * x));
        }
    } else
        throw std::runtime_error(
            "Sound::newFirFilter(): filter function not implemented.");

    const std::vector<Real_T> w = wFunction(n);
    std::transform(
        h.begin(), h.end(), w.begin(), h.begin(), std::multiplies<Real_T>());

    return Filter_T(h, std::vector<Real_T>(1, 1.0));
}

void N2D2::Sound::halfWaveRectify(unsigned int channel, double compression)
{
    if (compression != 1.0) {
        std::transform(
            mData[channel].begin(),
            mData[channel].end(),
            mData[channel].begin(),
            std::bind(
                static_cast<double (*)(double, double)>(&std::pow),
                std::bind(Utils::max<double>(), 0.0, std::placeholders::_1),
                compression));
    } else {
        std::transform(
            mData[channel].begin(),
            mData[channel].end(),
            mData[channel].begin(),
            std::bind(Utils::max<double>(), 0.0, std::placeholders::_1));
    }
}

void N2D2::Sound::fullWaveRectify(unsigned int channel, double compression)
{
    if (compression != 1.0) {
        std::transform(
            mData[channel].begin(),
            mData[channel].end(),
            mData[channel].begin(),
            std::bind(static_cast<double (*)(double, double)>(&std::pow),
                      std::bind(static_cast<double (*)(double)>(&std::fabs),
                                std::placeholders::_1),
                      compression));
    } else {
        std::transform(mData[channel].begin(),
                       mData[channel].end(),
                       mData[channel].begin(),
                       std::bind(static_cast<double (*)(double)>(&std::fabs),
                                 std::placeholders::_1));
    }
}

void N2D2::Sound::save(const std::string& fileName,
                       bool normalize,
                       double normalizeValue) const
{
    std::ofstream data(fileName.c_str(), std::fstream::binary);

    if (!data.good())
        throw std::runtime_error("Sound::save(): could not create sound file: "
                                 + fileName);

    unsigned int blockSize;

    // RIFF chunk
    data.write("RIFF", 4);

    blockSize = 44 - 8 + (mBitPerSample / 8) * mData.size() * mData[0].size();
    data.write(reinterpret_cast<const char*>(&blockSize), 4);
    data.write("WAVE", 4);

    // FORMAT chunk
    data.write("fmt ", 4);

    blockSize = 16;
    const unsigned short format = 1;
    const unsigned short channels = mData.size();
    const unsigned short bytePerBlock = mData.size() * mBitPerSample / 8;
    const unsigned int bytePerSecond = mSamplingFrequency * bytePerBlock;

    data.write(reinterpret_cast<const char*>(&blockSize), 4);
    data.write(reinterpret_cast<const char*>(&format), 2);
    data.write(reinterpret_cast<const char*>(&channels), 2);
    data.write(reinterpret_cast<const char*>(&mSamplingFrequency), 4);
    data.write(reinterpret_cast<const char*>(&bytePerSecond), 4);
    data.write(reinterpret_cast<const char*>(&bytePerBlock), 2);
    data.write(reinterpret_cast<const char*>(&mBitPerSample), 2);

    // DATA chunk
    data.write("data", 4);

    blockSize = (mBitPerSample / 8) * mData.size() * mData[0].size();
    data.write(reinterpret_cast<const char*>(&blockSize), 4);

    double maxValue = 0.0;

    if (normalize) {
        if (normalizeValue > 0.0)
            maxValue = normalizeValue;
        else {
            for (unsigned int s = 0, size = mData[0].size(); s < size; ++s) {
                for (std::vector<std::vector<double> >::const_iterator it
                     = mData.begin(),
                     itEnd = mData.end();
                     it != itEnd;
                     ++it)
                    maxValue = std::max(std::fabs(mData[0][s]), maxValue);
            }
        }
    }

    const double boundary = (1 << (mBitPerSample - 1));

    for (unsigned int s = 0, size = mData[0].size(); s < size; ++s) {
        for (std::vector<std::vector<double> >::const_iterator it
             = mData.begin(),
             itEnd = mData.end();
             it != itEnd;
             ++it) {
            double value = (*it)[s];

            if (maxValue > 0.0)
                value *= (boundary - 1) / maxValue;

            if (value < -boundary || value > (boundary - 1))
                throw std::runtime_error(
                    "Sound::save(): output overflow! Try to normalize first.");

            int byte = (int)value;
            data.write(reinterpret_cast<const char*>(&byte), mBitPerSample / 8);
        }
    }

    if (!data.good())
        throw std::runtime_error("Sound::save(): error writing sound file: "
                                 + fileName);
}

void N2D2::Sound::saveSignal(const std::string& fileName,
                             unsigned int channel,
                             bool plot) const
{
    std::ofstream data(fileName.c_str());

    if (!data.good())
        throw std::runtime_error("Could not create desc file: " + fileName);

    std::copy(mData[channel].begin(),
              mData[channel].end(),
              std::ostream_iterator<double>(data, "\n"));
    data.close();

    if (plot && !mData[channel].empty()) {
        Gnuplot gnuplot;
        gnuplot.set("grid").set("key off");
        gnuplot.setXlabel("Time (s)");
        gnuplot.saveToFile(fileName);

        std::ostringstream plotCmd;
        plotCmd << "using ($0/" << mSamplingFrequency << "):1 with lines";
        gnuplot.plot(fileName, plotCmd.str());
    }
}

void N2D2::Sound::genNoise(unsigned int channel, Time_T duration)
{
    const unsigned int nbSample
        = (unsigned int)((duration / (double)TimeS) * mSamplingFrequency);

    mData.at(channel).reserve(nbSample);

    for (unsigned int s = 0; s < nbSample; ++s)
        mData[channel].push_back(Random::randNormal(0.0, 1.0, -6.0, 6.0));
}

std::vector<unsigned int> N2D2::Sound::genNoisePattern(unsigned int channel,
                                                       Time_T duration,
                                                       unsigned int nbChunk,
                                                       unsigned int nbPattern)
{
    const unsigned int samplesPerChunk
        = (unsigned int)((duration / (double)TimeS) / nbChunk
                         * mSamplingFrequency);
    std::map<unsigned int, unsigned int> patterns;

    std::vector<unsigned int> block;
    block.reserve(nbChunk);

    mData.at(channel).reserve(samplesPerChunk * nbChunk);

    for (unsigned int i = 0; i < nbChunk; ++i) {
        const unsigned int pattern = Random::randUniform(0, nbPattern - 1);

        if (patterns.find(pattern) != patterns.end()) {
            mData[channel].insert(
                mData[channel].end(),
                mData[channel].begin() + patterns[i] * samplesPerChunk,
                mData[channel].begin() + (patterns[i] + 1) * samplesPerChunk);
        } else {
            for (unsigned int s = 0; s < samplesPerChunk; ++s)
                mData[channel]
                    .push_back(Random::randNormal(0.0, 1.0, -6.0, 6.0));

            patterns.insert(std::make_pair(pattern, i));
        }

        block.push_back(pattern);
    }

    return block;
}

std::vector<unsigned int>
N2D2::Sound::genAgusExperimentalBlock(unsigned int channel,
                                      Time_T trialDuration,
                                      unsigned int nbRepetition,
                                      unsigned int nbN,
                                      unsigned int nbRn,
                                      unsigned int nbRefRn,
                                      const std::vector<double>& refRn)
{
    const unsigned int nbTrial = nbN + nbRn + nbRefRn;

    // Populate a list of trials
    std::vector<unsigned int> block(nbTrial, AgusBlockType::N);

    // Pre-calculate the RefRN distribution to avoid consecutives RefRN trials
    std::vector<int> trials;
    trials.reserve(nbTrial);

    for (unsigned int i = 0; i < nbTrial; ++i)
        trials.push_back(i);

    unsigned int nbRemaining = nbRefRn;

    while (nbRemaining > 0 && !trials.empty()) {
        const unsigned int i = Random::randUniform(0, trials.size() - 1);
        const int index = trials[i];

        block[index] = AgusBlockType::RefRN;

        if (i < (trials.size() - 1) && trials[i + 1] == index + 1)
            trials.erase(trials.begin() + i + 1);

        trials.erase(trials.begin() + i);

        if (i > 0 && trials[i - 1] == index - 1)
            trials.erase(trials.begin() + i - 1);

        --nbRemaining;
    }

    if (nbRemaining > 0) {
        std::cout << "Notice: could not construct Agus experimental block "
                     "without consecutives RefRN trials" << std::endl;

        for (unsigned int i = 0; i < nbTrial; ++i) {
            if (block[i] != AgusBlockType::RefRN)
                trials.push_back(i);
        }

        while (nbRemaining > 0 && !trials.empty()) {
            const unsigned int i = Random::randUniform(0, trials.size() - 1);
            const int index = trials[i];

            block[index] = AgusBlockType::RefRN;
            trials.erase(trials.begin() + i);

            --nbRemaining;
        }

        if (nbRemaining > 0)
            throw std::logic_error(
                "Could not construct Agus experimental block");
    }

    // Then add RN locations
    trials.clear();

    for (unsigned int i = 0; i < nbTrial; ++i) {
        if (block[i] == AgusBlockType::N)
            trials.push_back(i);
    }

    nbRemaining = nbRn;

    while (nbRemaining > 0 && !trials.empty()) {
        const unsigned int i = Random::randUniform(0, trials.size() - 1);

        block[trials[i]] = AgusBlockType::RN;
        trials.erase(trials.begin() + i);
        --nbRemaining;
    }

    genAgusExperimentalBlock(
        channel, trialDuration, nbRepetition, block, refRn);

    return block;
}

void N2D2::Sound::genAgusExperimentalBlock(unsigned int channel,
                                           Time_T trialDuration,
                                           unsigned int nbRepetition,
                                           const std::vector
                                           <unsigned int>& block,
                                           const std::vector<double>& refRn)
{
    const unsigned int nbSamplePerTrial
        = (unsigned int)((trialDuration / (double)TimeS) * mSamplingFrequency);
    const unsigned int nbSamplePerRepetition = nbSamplePerTrial / nbRepetition;

    // Generate the RefRN trial
    std::vector<double> refRnGen;

    if (refRn.empty()) {
        refRnGen.reserve(nbSamplePerRepetition);

        for (unsigned int s = 0; s < nbSamplePerRepetition; ++s)
            refRnGen.push_back(Random::randNormal(0.0, 1.0, -6.0, 6.0));
    } else if (refRn.size() != nbSamplePerRepetition)
        throw std::runtime_error("Wrong size for the RefRN trial");

    const std::vector<double>& refRnTrial = (refRn.empty()) ? refRnGen : refRn;

    // Generate the block of trials
    mData.at(channel).reserve(nbSamplePerTrial * block.size());

    for (std::vector<unsigned int>::const_iterator it = block.begin(),
                                                   itEnd = block.end();
         it != itEnd;
         ++it) {
        if ((*it) == AgusBlockType::RefRN) {
            // RefRN trial
            for (unsigned int r = 0; r < nbRepetition; ++r)
                mData[channel].insert(
                    mData[channel].end(), refRnTrial.begin(), refRnTrial.end());
        } else if ((*it) == AgusBlockType::RN) {
            // RN trial
            for (unsigned int s = 0; s < nbSamplePerRepetition; ++s)
                mData[channel]
                    .push_back(Random::randNormal(0.0, 1.0, -6.0, 6.0));

            for (unsigned int r = 0; r < (nbRepetition - 1); ++r)
                mData[channel]
                    .insert(mData[channel].end(),
                            mData[channel].end() - nbSamplePerRepetition,
                            mData[channel].end());
        } else {
            // N trial
            for (unsigned int s = 0; s < nbSamplePerTrial; ++s)
                mData[channel]
                    .push_back(Random::randNormal(0.0, 1.0, -6.0, 6.0));
        }
    }
}

void N2D2::Sound::distortAgusExperimentalBlock(unsigned int channel,
                                               Time_T trialDuration,
                                               unsigned int nbRepetition,
                                               const std::vector
                                               <unsigned int>& block,
                                               double distortRatio)
{
    const unsigned int nbSamplePerTrial
        = (unsigned int)((trialDuration / (double)TimeS) * mSamplingFrequency);
    const unsigned int nbSamplePerRepetition = nbSamplePerTrial / nbRepetition;

    // See tshiftrepeatednoise.m from Trevor Agus
    unsigned int fillerSize = (unsigned int)std::ceil(
        nbSamplePerRepetition * (1.0 / distortRatio - 1.0));

    while (std::ceil((fillerSize + nbSamplePerRepetition) * distortRatio)
           > nbSamplePerRepetition) {
        if (fillerSize == 0)
            throw std::runtime_error("Negative filler size!");

        --fillerSize;
    }

    for (std::vector<unsigned int>::const_iterator it = block.begin(),
                                                   itBegin = block.begin(),
                                                   itEnd = block.end();
         it != itEnd;
         ++it) {
        if ((*it) != AgusBlockType::RefRN)
            continue;

        // RefRN trial
        for (unsigned int r = 0; r < nbRepetition; ++r) {
            std::vector<double>::iterator itTrialBegin
                = mData[channel].begin() + (it - itBegin) * nbSamplePerTrial
                  + r * nbSamplePerRepetition;

            Sound dataChunck(itTrialBegin,
                             itTrialBegin + nbSamplePerRepetition,
                             mSamplingFrequency);

            // See tshiftrepeatednoise.m from Trevor Agus
            std::vector<double> filler;
            filler.reserve(fillerSize);

            for (unsigned int s = 0; s < fillerSize; ++s)
                filler.push_back(Random::randNormal(0.0, 1.0, -6.0, 6.0));

            dataChunck().insert(
                dataChunck().end(), filler.begin(), filler.end());

            // See tshiftnoise.m from Trevor Agus
            const int denom
                = (int)Utils::round(1.0 / Utils::gcd(distortRatio, 1.0));
            dataChunck.resample(
                0, (int)Utils::round(distortRatio * denom), denom);

            if (dataChunck().size() != nbSamplePerRepetition)
                throw std::runtime_error("Size of distorted trial does not "
                                         "match original trial size");

            // Renormalization that seems to work perfectly to preserve the
            // activity rate (waiting to be proven mathematically!)
            std::transform(dataChunck().begin(),
                           dataChunck().end(),
                           dataChunck().begin(),
                           std::bind(std::multiplies<double>(),
                                     std::sqrt(1.0 / distortRatio),
                                     std::placeholders::_1));

            std::copy(dataChunck().begin(), dataChunck().end(), itTrialBegin);
        }
    }
}

void N2D2::Sound::loadTrialAgusExperimentalBlock(const std::string& fileName,
                                                 double normalizeValue,
                                                 AgusBlockType::Value type,
                                                 unsigned int nbRepetition,
                                                 unsigned int position)
{
    const std::string baseName = Utils::fileBaseName(fileName);
    const std::vector<unsigned int> desc
        = loadDescriptor(baseName + ".desc.dat");
    const unsigned int nbTrial = desc.size();

    unsigned int absPosition = 0;
    unsigned int relPosition = 0;

    for (; absPosition < nbTrial; ++absPosition) {
        if (desc[absPosition] != (unsigned int)type)
            continue;

        if (relPosition == position)
            break;

        ++relPosition;
    }

    load(fileName);

    const unsigned int nbSamples = mData[0].size();

    if (nbSamples % nbTrial != 0)
        throw std::runtime_error(
            "The number of samples is not a multiple of the number of trials!");

    const unsigned int nbSamplePerTrial = nbSamples / nbTrial;
    const unsigned int nbSamplePerRepetition = nbSamplePerTrial / nbRepetition;
    const std::vector<double>::iterator offset
        = mData[0].begin() + absPosition * nbSamplePerTrial;

    mData[0].assign(offset, offset + nbSamplePerRepetition);
    normalize(0, normalizeValue / ((1 << (mBitPerSample - 1)) - 1));
}

N2D2::Sound::Filter_T N2D2::Sound::newFilter(FilterType filter,
                                             FilterFunction func,
                                             unsigned int order,
                                             double cornerFreq1,
                                             double cornerFreq2,
                                             double chebyshevRipple) const
{
    if (cornerFreq1 > mSamplingFrequency / 2.0 || cornerFreq2
                                                  > mSamplingFrequency / 2.0)
        throw std::runtime_error(
            "Corner frequency is higher than the Nyquist frequency!");

    std::vector<Complex_T> sPlanePoles;
    std::vector<Complex_T> sPlaneZeros;

    // Compute S-plane poles for prototype LP filter
    if (filter == Butterworth || filter == Chebyshev) {
        for (unsigned int i = 0; i < 2 * order; ++i) {
            const Complex_T z(std::polar(
                1.0,
                (order & 1) ? (i * M_PI) / order : ((i + 0.5) * M_PI) / order));

            // Choose the poles
            if (z.real() < 0.0)
                sPlanePoles.push_back(z);
        }

        if (filter == Chebyshev) {
            if (chebyshevRipple >= 0.0)
                throw std::runtime_error("Chebyshev ripple must be < 0.0");

            const double rip = std::pow(10.0, -chebyshevRipple / 10.0);
            const double eps = std::sqrt(rip - 1.0);
            const double y = asinh(1.0 / eps) / (double)order;

            if (y <= 0.0)
                throw std::runtime_error("Chebyshev y must be > 0.0");

            for (unsigned int i = 0, size = sPlanePoles.size(); i < size; ++i)
                sPlanePoles[i]
                    = Complex_T(sPlanePoles[i].real() * std::sinh(y),
                                sPlanePoles[i].imag() * std::cosh(y));
        }
    }

    // Transform prototype into appropriate filter type
    const Real_T rawAlpha1 = cornerFreq1 / mSamplingFrequency;
    const Real_T rawAlpha2 = cornerFreq2 / mSamplingFrequency;
    const Real_T warpedAlpha1 = std::tan(M_PI * rawAlpha1) / M_PI;
    const Real_T warpedAlpha2 = std::tan(M_PI * rawAlpha2) / M_PI;

    const Real_T w1 = 2.0 * M_PI * warpedAlpha1;
    const Real_T w2 = 2.0 * M_PI * warpedAlpha2;

    if (func == LowPass) {
        std::transform(
            sPlanePoles.begin(),
            sPlanePoles.end(),
            sPlanePoles.begin(),
            std::bind(std::multiplies<Complex_T>(), w1, std::placeholders::_1));
    } else if (func == HighPass) {
        std::transform(
            sPlanePoles.begin(),
            sPlanePoles.end(),
            sPlanePoles.begin(),
            std::bind(std::divides<Complex_T>(), w1, std::placeholders::_1));
        sPlaneZeros.resize(sPlanePoles.size(), 0.0);
    } else if (func == BandPass) {
        const Real_T w0 = std::sqrt(w1 * w2);
        const Real_T bw = w2 - w1;

        sPlaneZeros.resize(sPlanePoles.size(), 0.0);

        for (unsigned int i = 0, size = sPlanePoles.size(); i < size; ++i) {
            const Complex_T hba((Real_T)0.5 * (bw * sPlanePoles[i]));
            const Complex_T temp(sqrt((Real_T)1.0 - (w0 / hba) * (w0 / hba)));
            sPlanePoles[i] = hba * ((Real_T)1.0 + temp);
            sPlanePoles.push_back(hba * ((Real_T)1.0 - temp));
        }
    } else if (func == BandStop) {
        const Real_T w0 = std::sqrt(w1 * w2);
        const Real_T bw = w2 - w1;

        const Complex_T z(0.0, w0);
        sPlaneZeros.resize(sPlanePoles.size(), z);
        sPlaneZeros.resize(2 * sPlanePoles.size(), -z);

        for (unsigned int i = 0, size = sPlanePoles.size(); i < size; ++i) {
            const Complex_T hba((Real_T)0.5 * (bw / sPlanePoles[i]));
            const Complex_T temp(sqrt((Real_T)1.0 - (w0 / hba) * (w0 / hba)));
            sPlanePoles[i] = hba * ((Real_T)1.0 + temp);
            sPlanePoles.push_back(hba * ((Real_T)1.0 - temp));
        }
    }
    /*
        // DEBUG
        for (unsigned int i = 0, size = sPlaneZeros.size(); i < size; ++i)
            std::cout << "z[" << i << "] = " << sPlaneZeros[i] << std::endl;

        for (unsigned int i = 0, size = sPlanePoles.size(); i < size; ++i)
            std::cout << "p[" << i << "] = " << sPlanePoles[i] << std::endl;
    */
    // Given S-plane poles & zeros, compute Z-plane poles & zeros, by bilinear
    // transform
    std::vector<Complex_T> zPlanePoles;
    std::vector<Complex_T> zPlaneZeros;

    std::transform(
        sPlanePoles.begin(),
        sPlanePoles.end(),
        std::back_inserter(zPlanePoles),
        std::bind(&bilinearTransform<Real_T>, std::placeholders::_1));

    std::transform(
        sPlaneZeros.begin(),
        sPlaneZeros.end(),
        std::back_inserter(zPlaneZeros),
        std::bind(&bilinearTransform<Real_T>, std::placeholders::_1));

    zPlaneZeros.resize(zPlanePoles.size(), -1.0);

    // Given Z-plane poles & zeros, compute top & bot polynomials in Z, and then
    // recurrence relation
    const std::vector<Complex_T> num(expandPolynomial(zPlaneZeros));
    const std::vector<Complex_T> denom(expandPolynomial(zPlanePoles));

    const Real_T theta = M_PI * (rawAlpha1 + rawAlpha2);
    const Complex_T fcGain
        = evaluate(num, denom, std::polar((Real_T)1.0, theta));

    std::vector<Real_T> realNum;
    std::vector<Real_T> realDenom;
    const Real_T norm = denom.back().real();

    std::transform(
        num.begin(),
        num.end(),
        std::back_inserter(realNum),
        std::bind(std::divides<Real_T>(),
                  std::bind(&realPart<Real_T>, std::placeholders::_1),
                  norm));

    std::transform(
        denom.begin(),
        denom.end(),
        std::back_inserter(realDenom),
        std::bind(std::divides<Real_T>(),
                  std::bind(&realPart<Real_T>, std::placeholders::_1),
                  norm));

    realDenom.back() *= abs(fcGain);
    /*
        // DEBUG
        for (unsigned int i = 0, size = realNum.size(); i < size; ++i)
            std::cout << "num[" << i << "] = " << realNum[i] << std::endl;

        for (unsigned int i = 0, size = realDenom.size(); i < size; ++i)
            std::cout << "denom[" << i << "] = " << realDenom[i] << std::endl;

        std::cout << "gain at dc = " << abs(evaluate<Real_T>(num, denom, 1.0))
       << std::endl;
        std::cout << "gain at center = " << abs(fcGain) << std::endl;
        std::cout << "gain at hf = " << abs(evaluate<Real_T>(num, denom, -1.0))
       << std::endl;
    */
    return Filter_T(realNum, realDenom);
}

std::tuple<N2D2::Sound::Filter_T,
           N2D2::Sound::Filter_T,
           N2D2::Sound::Filter_T,
           N2D2::Sound::Filter_T>
N2D2::Sound::newGammatoneFilter(double centerFreq, double bandwidth) const
{
    const Real_T T = 1.0 / mSamplingFrequency;
    const Real_T wc = 2.0 * M_PI * centerFreq * T;
    const Real_T bw = 1.019 * 2.0 * M_PI * bandwidth;

    const Real_T cosWc = std::cos(wc);
    const Real_T sinWc = std::sin(wc);
    const Real_T sqrtP = std::sqrt(3.0 + std::pow(2.0, 3.0 / 2.0));
    const Real_T sqrtN = std::sqrt(3.0 - std::pow(2.0, 3.0 / 2.0));
    const Real_T expBw = std::exp(bw * T);
    const Complex_T expWc = std::polar<Real_T>(1.0, 2.0 * wc);
    const Complex_T expBwWc = std::exp(Complex_T(-bw * T, wc));

    const Real_T gain
        = std::abs(T * (-expWc + expBwWc * (cosWc - sqrtN * sinWc)) * T
                   * (-expWc + expBwWc * (cosWc + sqrtN * sinWc)) * T
                   * (-expWc + expBwWc * (cosWc - sqrtP * sinWc)) * T
                   * (-expWc + expBwWc * (cosWc + sqrtP * sinWc))
                   / std::pow(-1.0 / std::exp(2.0 * bw * T) - expWc
                              + (Complex_T(1.0, 0.0) + expWc) / expBw,
                              (Real_T)4.0));

    std::vector<Real_T> denom;
    denom.push_back(std::exp(-2.0 * bw * T));
    denom.push_back(-2.0 * cosWc / expBw);
    denom.push_back(1.0);

    std::vector<Real_T> num1;
    num1.push_back(0.0);
    num1.push_back((-T * (cosWc + sqrtP * sinWc) / expBw) / gain);
    num1.push_back(T / gain);

    std::vector<Real_T> num2;
    num2.push_back(0.0);
    num2.push_back(-T * (cosWc - sqrtP * sinWc) / expBw);
    num2.push_back(T);

    std::vector<Real_T> num3;
    num3.push_back(0.0);
    num3.push_back(-T * (cosWc + sqrtN * sinWc) / expBw);
    num3.push_back(T);

    std::vector<Real_T> num4;
    num4.push_back(0.0);
    num4.push_back(-T * (cosWc - sqrtN * sinWc) / expBw);
    num4.push_back(T);

    return std::make_tuple(Filter_T(num1, denom),
                           Filter_T(num2, denom),
                           Filter_T(num3, denom),
                           Filter_T(num4, denom));
}

void N2D2::Sound::applyFilter(const Filter_T& filter,
                              unsigned int channel,
                              bool appendTrailing)
{
    if (filter.second.size() > 1) {
        // IIR Filter
        std::deque<double> in(filter.first.size(), 0.0);
        std::deque<double> out(filter.second.size() - 1, 0.0);
        const Real_T gain = filter.second.back();

        for (unsigned int s = 0, sSize = mData.at(channel).size(); s < sSize;
             ++s) {
            in.pop_front();
            in.push_back((double)(mData[channel][s] / gain));

            mData[channel][s] = std::inner_product(
                in.begin(),
                in.end(),
                filter.first.begin(),
                -std::inner_product(
                     out.begin(), out.end(), filter.second.begin(), 0.0));

            out.pop_front();
            out.push_back(mData[channel][s]);
        }
    } else {
        // FIR Filter
        std::deque<double> in(filter.first.size(), 0.0);
        const Real_T gain = filter.second.back();

        for (unsigned int s = 0, sSize = mData.at(channel).size(); s < sSize;
             ++s) {
            in.pop_front();
            in.push_back((double)(mData[channel][s] / gain));

            mData[channel][s] = std::inner_product(
                in.begin(), in.end(), filter.first.begin(), 0.0);
        }

        if (appendTrailing) {
            mData[channel]
                .reserve(mData.at(channel).size() + filter.first.size());

            for (unsigned int s = 0, sSize = filter.first.size(); s < sSize;
                 ++s) {
                in.pop_front();
                in.push_back(0.0);

                mData[channel].push_back(std::inner_product(
                    in.begin(), in.end(), filter.first.begin(), 0.0));
            }
        }
    }
}

void N2D2::Sound::saveFilterResponse(const Filter_T& filter,
                                     const std::string& fileName,
                                     unsigned int nbSteps,
                                     bool append,
                                     bool plot) const
{
    std::ofstream data;

    if (append)
        data.open(fileName.c_str(), std::ofstream::app);
    else
        data.open(fileName.c_str());

    if (!data.good())
        throw std::runtime_error("Could not create data file: " + fileName);

    std::vector<Real_T> realDenom(filter.second);
    const Real_T gain = realDenom.back();
    realDenom.back() = 1.0;

    for (unsigned int i = 0; i < nbSteps; ++i) {
        const Real_T alpha = 0.5 * (i + 1) / (nbSteps + 1);
        const Complex_T resp
            = evaluate(filter.first,
                       realDenom,
                       std::polar((Real_T)1.0, 2.0 * M_PI * alpha));

        data << alpha* mSamplingFrequency << " "
             << 20.0 * std::log10(std::abs(resp) / gain) << " "
             << Utils::radToDeg(std::arg(resp)) << "\n";
    }

    data << "\n";
    data.close();

    if (plot) {
        Gnuplot gnuplot;
        gnuplot.set("grid").set("key off");
        gnuplot.setXlabel("Frequency (Hz)");
        gnuplot.setYlabel("Magnitude (dB)");
        gnuplot.unset("colorbox");
        gnuplot.saveToFile(fileName);
        gnuplot.plot(fileName, "using 1:2:0 with line palette");

        gnuplot.saveToFile(fileName, "-log");
        gnuplot.set("logscale x");
        gnuplot.plot(fileName, "using 1:2:0 with line palette");
    }
}

void N2D2::Sound::saveWindowFunction(const WindowFunction<double>& wFunction,
                                     const std::string& fileName,
                                     unsigned int n,
                                     bool append,
                                     bool plot) const
{
    std::ofstream data;

    if (append)
        data.open(fileName.c_str(), std::ofstream::app);
    else
        data.open(fileName.c_str());

    if (!data.good())
        throw std::runtime_error("Could not create data file: " + fileName);

    const std::vector<double> w = wFunction(n);

    for (unsigned int i = 0; i < n; ++i)
        data << i << " " << w[i] << "\n";

    data << "\n";
    data.close();

    if (plot) {
        Gnuplot gnuplot;
        gnuplot.set("grid");
        gnuplot.saveToFile(fileName);
        gnuplot.plot(fileName, "using 1:2 with line");
    }
}

std::vector<std::vector<double> > N2D2::Sound::spectrogram(unsigned int channel,
                                                           unsigned int nFft,
                                                           const WindowFunction
                                                           <double>& wFunction,
                                                           unsigned int wSize,
                                                           int nOverlap,
                                                           bool logScale) const
{
    return DSP::spectrogram(
        mData.at(channel), nFft, wFunction, wSize, nOverlap, logScale);
}

void N2D2::Sound::spectrogram(const std::string& fileName,
                              unsigned int channel,
                              unsigned int nFft,
                              const WindowFunction<double>& wFunction,
                              unsigned int wSize,
                              int nOverlap,
                              bool logScale,
                              bool plot) const
{
    // Default values
    if (nFft == 0)
        nFft = std::min((unsigned int)256,
                        (unsigned int)mData.at(channel).size());

    // Ensure that nFft is a power of 2, because the size of y has to match the
    // vector size returned by fft()
    if ((nFft & (nFft - 1)) != 0)
        nFft = 1 << ((int)std::ceil(std::log((double)nFft) / std::log(2.0)));

    if (wSize == 0) {
        wSize = nFft;
        nOverlap = wSize / 2;
    }

    const unsigned int nHop = (nOverlap < 0) ? -nOverlap : wSize - nOverlap;

    // Write file
    std::ofstream data(fileName.c_str());

    if (!data.good())
        throw std::runtime_error("Could not save data file: " + fileName);

    std::vector<std::vector<double> > y = DSP::spectrogram(
        mData.at(channel), nFft, wFunction, wSize, nOverlap, logScale);

    for (unsigned int f = 0, fSize = y.size(); f < fSize; ++f) {
        std::copy(
            y[f].begin(), y[f].end(), std::ostream_iterator<double>(data, " "));
        data << "\n";
    }

    data.close();

    if (plot) {
        // Plot result
        Gnuplot gnuplot;
        gnuplot.set("palette defined (0 0.0 0.0 0.5," // Matlab JET colormap
                    "1 0.0 0.0 1.0,"
                    "2 0.0 0.5 1.0,"
                    "3 0.0 1.0 1.0,"
                    "4 0.5 1.0 0.5,"
                    "5 1.0 1.0 0.0,"
                    "6 1.0 0.5 0.0,"
                    "7 1.0 0.0 0.0,"
                    "8 0.5 0.0 0.0)");
        gnuplot.set("pm3d map").set("key off");
        gnuplot << "if (!exists(\"xoffset\")) xoffset=0";
        gnuplot << "if (!exists(\"xsubset\")) xsubset=1";

        std::stringstream cmdStr;
        cmdStr << "set xrange [xoffset:xoffset+"
               << mData.at(channel).size() / (double)mSamplingFrequency << "]";
        gnuplot << cmdStr.str();

        gnuplot.setYrange(0, mSamplingFrequency / 2.0);
        gnuplot.setXlabel("Time (s)");
        gnuplot.setYlabel("Frequency (Hz)");
        gnuplot.saveToFile(fileName);

        std::ostringstream plotCmd;
        plotCmd << "matrix using (xoffset+$1*"
                << (nHop / (double)mSamplingFrequency) << "):($2*"
                << (((double)mSamplingFrequency) / 2.0 / (y.size() - 1.0))
                << "):3 every xsubset";

        gnuplot.splot(fileName, plotCmd.str());
    }
}

double N2D2::Sound::getDuration(unsigned int channel) const
{
    return mData[channel].size() / (double)mSamplingFrequency;
}

void N2D2::Sound::clear()
{
    for (std::vector<std::vector<double> >::iterator it = mData.begin(),
                                                     itEnd = mData.end();
         it != itEnd;
         ++it)
        (*it).clear();
}

void N2D2::Sound::saveDescriptor(const std::vector<unsigned int>& desc,
                                 const std::string& fileName,
                                 bool plot)
{
    std::ofstream data(fileName.c_str());

    if (!data.good())
        throw std::runtime_error("Could not create desc file: " + fileName);

    std::copy(desc.begin(),
              desc.end(),
              std::ostream_iterator<unsigned int>(data, "\n"));
    data.close();

    if (plot && !desc.empty()) {
        Gnuplot gnuplot;
        gnuplot.set("grid").set("key off");
        gnuplot.set("bars", 0);
        gnuplot.set("pointsize", 0.01);
        gnuplot.set("ytics", 1);
        gnuplot.saveToFile(fileName);
        gnuplot.plot(fileName, "using 0:1:($1+0.8):($1) with yerrorbars");
    }
}

std::vector<unsigned int> N2D2::Sound::loadDescriptor(const std::string
                                                      & fileName)
{
    std::ifstream data(fileName.c_str());

    if (!data.good())
        throw std::runtime_error("Could not load desc file: " + fileName);

    std::vector<unsigned int> desc;
    std::copy(std::istream_iterator<unsigned int>(data),
              std::istream_iterator<unsigned int>(),
              std::back_inserter(desc));
    return desc;
}

void N2D2::Sound::saveActivityFromXlabel(const std::string& xlabelFileName,
                                         const std::string& activityFileName,
                                         Time_T offset)
{
    std::ofstream activity;

    if (offset > 0)
        activity.open(activityFileName.c_str(), std::ofstream::app);
    else
        activity.open(activityFileName.c_str());

    if (!activity.good())
        throw std::runtime_error("Could not create activity file: "
                                 + activityFileName);

    // Use the full double precision to keep accuracy even on small scales
    activity.precision(std::numeric_limits<double>::digits10 + 1);

    std::ifstream xlabel(xlabelFileName.c_str());

    if (!xlabel.good())
        throw std::runtime_error("Could not open xlabel file: "
                                 + xlabelFileName);

    std::string line;
    std::string separator(";");

    do {
        getline(xlabel, line);

        if (line.compare(0, 9, "separator") == 0) {
            std::stringstream separatorStr(line.substr(9));
            separatorStr >> separator;
        }
    } while (line[0] != '#');

    do {
        double time;
        unsigned int color;
        std::string label;

        xlabel >> time;
        xlabel >> color;
        getline(xlabel, label);

        if (xlabel.good()) {
            std::stringstream trimmer;
            trimmer << label.substr(0, label.find_first_of(separator));
            label.clear();
            trimmer >> label;

            activity << label << " " << (offset / ((double)TimeS) + time)
                     << "\n";
        }
    } while (xlabel.good());
}
