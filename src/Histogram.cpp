/*
    (C) Copyright 2013 CEA LIST. All Rights Reserved.
    Contributor(s): Olivier BICHLER (olivier.bichler@cea.fr)
                    Johannes THIELE (johannes.thiele@cea.fr)

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

#include "Histogram.hpp"
#include "utils/Utils.hpp"
#include "utils/Gnuplot.hpp"

N2D2::Histogram::Histogram(double minVal_,
                           double maxVal_,
                           unsigned int nbBins_)
    : mMinVal(minVal_),
      mMaxVal(maxVal_),
      mNbBins(nbBins_),
      mValues(nbBins_, 0),
      mNbValues(0),
      mMaxBin(0)
{
    // ctor
}

void N2D2::Histogram::operator()(double value, unsigned long long int count)
{
    const unsigned int binIdx = getBinIdx(value);

    if (binIdx > mMaxBin)
        mMaxBin = binIdx;

    mValues[binIdx] += count;
    mNbValues += count;
}

unsigned int N2D2::Histogram::enlarge(double value)
{
    if (value > mMaxVal) {
        const double binWidth = getBinWidth();

        mNbBins += (unsigned int)std::ceil((value - mMaxVal) / binWidth);
        mValues.resize(mNbBins, 0);
        mMaxVal = mMinVal + mNbBins * binWidth;

        assert(mMaxVal >= value);
    }

    return mNbBins;
}

unsigned int N2D2::Histogram::truncate(double value) {
    if (value < mMaxVal) {
        const unsigned int newMaxBin = getBinIdx(value);
        mMaxVal = mMinVal + (newMaxBin + 1) * getBinWidth();

        for (unsigned int bin = newMaxBin + 1; bin <= mMaxBin; ++bin)
            mValues[newMaxBin] += mValues[bin];

        mMaxBin = newMaxBin;
        mNbBins = (newMaxBin + 1);
        mValues.resize(mNbBins);
    }

    return mNbBins;
}

unsigned int N2D2::Histogram::getBinIdx(double value) const
{
    const double clampedValue = Utils::clamp(value, mMinVal, mMaxVal);
    unsigned int binIdx = (mMaxVal != mMinVal)
        ? (unsigned int)(mNbBins * (clampedValue - mMinVal) / (mMaxVal - mMinVal))
        : 0;

    if (binIdx == mNbBins)
        --binIdx;

    return  binIdx;
}

void N2D2::Histogram::log(const std::string& fileName,
                                   const std::map<std::string, double>&
                                        thresholds) const
{
    std::ofstream histData(fileName.c_str());

    if (!histData.good()) {
        throw std::runtime_error("Could not open histogram file: "
                                 + fileName);
    }

    for (unsigned int bin = 0; bin <= mMaxBin; ++bin) {
        histData << bin << " " << getBinValue(bin) << " " << mValues[bin]
            << " " << (mValues[bin] / (double)mNbValues) << "\n";
    }

    histData.close();

    Gnuplot gnuplot;
    gnuplot.setXlabel("Output value");
    gnuplot.setYlabel("Normalized number of counts");
    gnuplot.set("grid");
    gnuplot.set("key off");
    gnuplot.set("xrange [0:]");
    gnuplot.set("logscale y");

    unsigned int i = 0;

    for (std::map<std::string, double>::const_iterator it = thresholds.begin(),
        itEnd = thresholds.end(); it != itEnd; ++it)
    {
        std::stringstream cmdStr;
        cmdStr << "arrow from " << (*it).second << ", "
            "graph 0 to " << (*it).second << ", graph 1 nohead "
            "lt " << (3 + i) << " lw 2";
        gnuplot.set(cmdStr.str());

        cmdStr.str(std::string());
        cmdStr << "label " << (i + 1) << " \" " << (*it).first << " = "
            << (*it).second << "\"";
        gnuplot.set(cmdStr.str());

        cmdStr.str(std::string());
        cmdStr << "label " << (i + 1) << " at " << (*it).second
            << ", graph " << (0.85 - i * 0.05) << " tc lt " << (3 + i);
        gnuplot.set(cmdStr.str());

        ++i;
    }

    gnuplot.saveToFile(fileName);
    gnuplot.plot(fileName, "using 2:4 with points");
}

N2D2::Histogram N2D2::Histogram::quantize(
    double newMaxVal,
    unsigned int newNbBins) const
{
    Histogram hist(mMinVal, newMaxVal, newNbBins);

    for (unsigned int bin = 0; bin <= mMaxBin; ++bin)
        hist(getBinValue(bin), mValues[bin]);

    assert(mNbValues == hist.mNbValues);
    return hist;
}

double N2D2::Histogram::calibrateKL(unsigned int nbLevels,
                                             double maxError,
                                             unsigned int maxIters) const
{
    double lowerVal = mMinVal;
    double upperVal = getBinValue(mMaxBin);
    unsigned int iter = 0;

    while ((upperVal - lowerVal) > maxError && iter < maxIters) {
        const double middleVal = (upperVal + lowerVal) / 2.0;
        const double middleLowerVal = (middleVal + lowerVal) / 2.0;
        const double middleUpperVal = (upperVal + middleVal) / 2.0;
        const double lowerDiv = KLDivergence(*this,
                                            quantize(middleLowerVal, nbLevels));
        const double upperDiv = KLDivergence(*this,
                                            quantize(middleUpperVal, nbLevels));

        if (upperDiv < lowerDiv)
            lowerVal = middleLowerVal;
        else
            upperVal = middleUpperVal;

        // DEBUG
        //std::cout << "div = " << ((upperDiv < lowerDiv) ? upperDiv : lowerDiv)
        //    << "   [" << lowerVal << ", " << upperVal << "]" << std::endl;
        ++iter;
    }

    return (upperVal + lowerVal) / 2.0;
}

double N2D2::Histogram::KLDivergence(const Histogram& ref,
                                              const Histogram& quant)
{
    assert(ref.mNbValues == quant.mNbValues);
    assert(ref.mMaxBin >= quant.mMaxBin);
    assert(ref.mMaxVal >= quant.mMaxVal);

    double qNorm = 0.0;

    for (unsigned int bin = 0; bin <= ref.mMaxBin; ++bin) {
        const unsigned int quantIdx = quant.getBinIdx(ref.getBinValue(bin));
        qNorm += quant.mValues[quantIdx];
    }

    double divergence = 0.0;

    for (unsigned int bin = 0; bin <= ref.mMaxBin; ++bin) {
        const unsigned int quantIdx = quant.getBinIdx(ref.getBinValue(bin));

        // Sum of p and q over bin [0,ref.mMaxBin] must be 1
        const double p = (ref.mValues[bin] / (double)ref.mNbValues);
        const double q = (quant.mValues[quantIdx] / qNorm);

        // q = 0 => p = 0
        // p = 0 => contribution = 0
        if (p != 0) {
            assert(q != 0);  // q = 0 => p = 0
            divergence += p * std::log(p / q);
        }
    }

    return divergence;
}

void N2D2::Histogram::save(std::ostream& state) const {
    state.write(reinterpret_cast<const char*>(&mMinVal), sizeof(mMinVal));
    state.write(reinterpret_cast<const char*>(&mMaxVal), sizeof(mMaxVal));
    state.write(reinterpret_cast<const char*>(&mNbBins), sizeof(mNbBins));
    const size_t valuesSize = mValues.size();
    state.write(reinterpret_cast<const char*>(&valuesSize), sizeof(valuesSize));
    state.write(reinterpret_cast<const char*>(&mValues[0]),
                valuesSize * sizeof(mValues[0]));
    state.write(reinterpret_cast<const char*>(&mNbValues), sizeof(mNbValues));
    state.write(reinterpret_cast<const char*>(&mMaxBin), sizeof(mMaxBin));
}

void N2D2::Histogram::load(std::istream& state) {
    state.read(reinterpret_cast<char*>(&mMinVal), sizeof(mMinVal));
    state.read(reinterpret_cast<char*>(&mMaxVal), sizeof(mMaxVal));
    state.read(reinterpret_cast<char*>(&mNbBins), sizeof(mNbBins));
    size_t valuesSize;
    state.read(reinterpret_cast<char*>(&valuesSize), sizeof(valuesSize));
    mValues.resize(valuesSize);
    state.read(reinterpret_cast<char*>(&mValues[0]),
                valuesSize * sizeof(mValues[0]));
    state.read(reinterpret_cast<char*>(&mNbValues), sizeof(mNbValues));
    state.read(reinterpret_cast<char*>(&mMaxBin), sizeof(mMaxBin));
}

void N2D2::Histogram::saveOutputsHistogram(const std::string& fileName,
                               const std::map
                               <std::string, Histogram>& outputsHistogram)
{
    std::ofstream state(fileName.c_str(), std::fstream::binary);

    if (!state.good())
        throw std::runtime_error("Could not create state file: " + fileName);

    const size_t mapSize = outputsHistogram.size();
    state.write(reinterpret_cast<const char*>(&mapSize), sizeof(mapSize));

    for (std::map<std::string, Histogram>::const_iterator it
         = outputsHistogram.begin(), itEnd = outputsHistogram.end();
         it != itEnd; ++it)
    {
        const size_t nameSize = (*it).first.size();
        const std::string& nameStr = (*it).first;
        state.write(reinterpret_cast<const char*>(&nameSize), sizeof(nameSize));
        state.write(reinterpret_cast<const char*>(&nameStr[0]),
                    nameSize * sizeof(nameStr[0]));

        (*it).second.save(state);
    }

    if (!state.good())
        throw std::runtime_error("Error writing state file: " + fileName);
}

void N2D2::Histogram::loadOutputsHistogram(const std::string& fileName,
                            std::map<std::string, Histogram>& outputsHistogram)
{
    std::ifstream state(fileName.c_str(), std::fstream::binary);

    if (!state.good())
        throw std::runtime_error("Could not open state file: " + fileName);

    size_t mapSize;
    state.read(reinterpret_cast<char*>(&mapSize), sizeof(mapSize));

    for (size_t n = 0; n < mapSize; ++n) {
        size_t nameSize;
        state.read(reinterpret_cast<char*>(&nameSize), sizeof(nameSize));
        std::string nameStr(nameSize, ' ');
        state.read(reinterpret_cast<char*>(&nameStr[0]),
                    nameSize * sizeof(nameStr[0]));

        std::map<std::string, Histogram>::iterator it;
        std::tie(it, std::ignore)
            = outputsHistogram.insert(std::make_pair(nameStr, Histogram()));

        (*it).second.load(state);
    }

    if (state.eof())
        throw std::runtime_error(
            "End-of-file reached prematurely in state file: "
            + fileName);
    else if (!state.good())
        throw std::runtime_error("Error while reading state file: "
                                 + fileName);
    else if (state.get() != std::fstream::traits_type::eof())
        throw std::runtime_error(
            "State file size larger than expected: " + fileName);
}

void N2D2::Histogram::logOutputsHistogram(const std::string& dirName,
                               const std::map
                               <std::string, Histogram>& outputsHistogram,
                               unsigned int nbLevels)
{
    Utils::createDirectories(dirName);

    for (std::map<std::string, Histogram>::const_iterator it
         = outputsHistogram.begin(), itEnd = outputsHistogram.end();
         it != itEnd; ++it)
    {
        std::map<std::string, double> thresholds;

        Histogram hist = (*it).second;

        // First pass
        double threshold = hist.calibrateKL(nbLevels);
        thresholds["KL 1-pass"] = threshold;

        // Second pass on truncated hist
        hist.truncate(threshold);
        threshold = hist.calibrateKL(nbLevels);
        thresholds["KL 2-passes"] = threshold;

        // Third pass
        hist.truncate(threshold);
        threshold = hist.calibrateKL(nbLevels);
        thresholds["KL 3-passes"] = threshold;

        // Fourth pass
        hist.truncate(threshold);
        threshold = hist.calibrateKL(nbLevels);
        thresholds["KL 4-passes"] = threshold;

        // Fifth pass
        hist.truncate(threshold);
        threshold = hist.calibrateKL(nbLevels);
        thresholds["KL 5-passes"] = threshold;

        (*it).second.log(dirName + "/" + (*it).first + ".dat",
                                  thresholds);
        (*it).second.quantize(threshold).log(dirName + "/"
                                            + (*it).first + "_quant.dat");

        //std::cout << (*it).first << ": " << threshold << std::endl;
    }
}
